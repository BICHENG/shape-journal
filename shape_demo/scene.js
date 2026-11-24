import * as THREE from 'three';
import {STLExporter} from 'three/addons/exporters/STLExporter.js';
import {connectIntake} from './intake.js';
import {createSource} from './source.js';
import {blendCameras, createFigure, decode} from './figure.js';
import {createContours} from './contours.js';
import {createTapes} from './tapes.js';
import {advance, playbackEffect, stanceMotion} from './playback.js';
import {openGroup, peopleLink} from './group.js';
import {buildStage, boxCorners, frameVisibility, groupFade, overviewYaw, isoElevation, projectedFrame, station} from './stage.js';

const settings = {
  abdomen: {label: 'Waist', yaw: .82},
  legs: {label: 'Legs', yaw: .22},
  arms: {label: 'Upper Arms', yaw: .24},
};

const get = id => document.getElementById(id);
const shell = get('shell');
const viewport = get('scene'), rail = get('time');
const front = get('front'), side = get('side'), compare = get('reference');
const framing = get('framing'), play = get('play'), playLabel = get('play-label');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const recordYaw = .52, recordElevation = .14;

function easeTo(value, target, amount, tolerance) {
  const next = THREE.MathUtils.lerp(value, target, amount);
  return amount > 0 && Math.abs(next - target) <= tolerance ? target : next;
}

function initialize(app, data) {
  const base = decode(data.positions, Float32Array);
  const indices = decode(data.triangles, Uint16Array);
  let stature = 0;
  for (let i = 1; i < base.length; i += 3) stature = Math.max(stature, base[i] / 1700);
  const archive = data.mode === 'archive';
  const supportsReplay = 'requestVideoFrameCallback' in HTMLVideoElement.prototype && 'DecompressionStream' in globalThis;
  const maxTick = data.records.length > 1 ? Math.max(1000, data.records.length - 1) : 0;
  const days = data.records.map(record => Date.parse(record.date));
  const dated = days.every(Number.isFinite) && days.at(-1) > days[0];
  let lastTick = -1;
  const records = data.records.map((record, i) => {
    const progress = dated ? (days[i] - days[0]) / (days.at(-1) - days[0]) : i / Math.max(1, days.length - 1);
    const tick = Math.min(maxTick - (days.length - i - 1), Math.max(lastTick + 1, Math.round(maxTick * progress)));
    lastTick = tick;
    return {...record, tick, delta: record.delta ? decode(record.delta, Float32Array) : null};
  });
  let selected = records.find(record => record.key === app?.record) ?? records.at(-1);
  let holding = false, cameraHeld = false;
  let browsingSource = false, returning = false, savedView;
  const lens = {value: 0, target: 0, velocity: 0};
  const pose = stanceMotion(1, reducedMotion);
  let time = 0, started = null, resumeAt = 0, playingSource = false, contact = null;
  let frame = 0, lastFrame = 0, width = 1, height = 1;
  let yaw = recordYaw, yawTarget = recordYaw, elevation = recordElevation, elevationTarget = recordElevation;
  let zoom = 1, zoomTarget = 1, size = 0, recordMix = 1;
  let before, after, fraction, shapeDirty, activeLayer = null;
  get('mode').textContent = data.subject ?? (archive ? 'My Log' : 'Sample');
  peopleLink(app, 'Library');
  get('time-label').textContent = archive ? 'Visits' : 'Sample';
  rail.max = String(Math.max(1, maxTick));
  rail.disabled = play.disabled = records.length === 1;

  const renderer = new THREE.WebGLRenderer({alpha: true, antialias: true, powerPreference: 'high-performance'});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  viewport.append(renderer.domElement);
  const scene = new THREE.Scene();
  const orbit = new THREE.OrthographicCamera(-1, 1, 1, -1, .05, 40), matchCamera = blendCameras();
  const gaze = new THREE.Vector3(), target = new THREE.Vector3(), direction = new THREE.Vector3();
  const horizontal = new THREE.Vector3(), vertical = new THREE.Vector3();
  const geometry = new THREE.BufferGeometry();
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  const position = new THREE.BufferAttribute(base.slice(), 3).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', position);
  geometry.computeVertexNormals();
  const wipeX = {value: -1};
  const figure = createFigure(geometry, data, station, wipeX);
  const {body, model, film} = figure;
  const contourLegend = get('contour-legend');
  const tape = data.tape && createTapes(scene, renderer, station, data, base, indices);
  let drawContours;
  const wholeBody = {id: 'whole', label: 'Full Body', yaw: recordYaw, shape: geometry, reference: geometry.clone()};
  const parts = [wholeBody, ...data.views.map(regionView)];
  let selectedPart = wholeBody;
  const contextWire = new THREE.MeshBasicMaterial({
    color: '#a3bac4', wireframe: true, transparent: true, opacity: .08, depthWrite: false,
  });
  contextWire.onBeforeCompile = shader => {
    const varying = 'varying vec2 screenPosition;\n';
    shader.vertexShader = varying + shader.vertexShader.replace(
      '#include <project_vertex>', '#include <project_vertex>\nscreenPosition = gl_Position.xy / gl_Position.w;');
    shader.fragmentShader = varying + shader.fragmentShader.replace('#include <color_fragment>', `
      #include <color_fragment>
      float border = max(abs(screenPosition.x), abs(screenPosition.y));
      diffuseColor.a *= 1. - smoothstep(.78, .99, border);
    `);
  };
  const armContext = new THREE.Mesh(geometry, contextWire);
  armContext.name = 'arm-context';
  armContext.position.copy(station);
  armContext.scale.setScalar(.001);
  armContext.frustumCulled = false;
  armContext.renderOrder = -.2;
  armContext.visible = false;
  scene.add(model, armContext);
  const rig = data.rig;
  const targetJoints = rig && Float32Array.from(rig.joints);
  let source, capture, evidence, effect, fadeGuides, fadeFinder;
  let world = buildStage(null, null, selectLayer);

  const screenFrame = get('screen-frame');
  const capturedView = get('capture-pose'), breakdownView = get('decompose-view'), standard = get('shape-view');
  const sourcePlay = get('source-play'), sourceTime = get('source-time'), sourceClock = get('source-clock');
  const sourceStatus = get('source-status'), takeSelect = get('take-select'), takeStatus = get('take-status');
  const description = get('description'), partsNav = get('parts');
  const download = get('download'), pipelineSteps = get('pipeline-steps');
  const viewCorners = Array.from({length: 8}, () => new THREE.Vector3());
  const screenExtent = {}, viewCorrection = new THREE.Matrix4();
  let measuredPosition, measuredVersion;
  let layerButtons = [], prominence = {};
  let boundTake, takeRequest, frameToRemember;
  let original = false, compared = false, comparisonSplit;
  body.userData.pick = () => selectLayer('fit');

  function releaseGroup(group) {
    scene.remove(group);
    group.traverse(node => {
      node.geometry?.dispose();
      node.material?.dispose();
    });
  }

  function bindSource(payload) {
    compared = false;
    comparisonSplit = undefined;
    const next = createSource(payload, film, station, redraw, sourceTimeAt, finishSource);
    const stage = buildStage(next, payload, selectLayer);
    if (capture) {
      capture.dispose();
      releaseGroup(capture.group);
      releaseGroup(world.group);
    }
    source = payload;
    capture = next;
    effect = playbackEffect(next, reducedMotion);
    world = stage;
    evidence = capture.evidence;
    capture.finder.userData.pick = lookThrough;
    fadeGuides = groupFade(capture.guides);
    fadeFinder = groupFade(capture.finder);
    scene.add(world.group, capture.group);
    world.layout(width / height);
    capture.fit(width, height);
    frameToRemember = null;

    const pointCount = evidence ? evidence.count.toLocaleString('en-US') + ' points' : '';
    const layers = evidence ? [
      ['source', 'Frame', `${source.width} × ${source.height}`],
      ['cloud', 'Point Cloud', pointCount],
      ['fit', 'Fitted Shape', (base.length / 3).toLocaleString('en-US') + ' vertices'],
    ] : [];
    prominence = Object.fromEntries(layers.map(([key]) => [key, 1]));
    layerButtons = layers.map(([key, label, value]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.layer = key;
      button.textContent = label;
      button.title = value;
      button.addEventListener('click', () => selectLayer(key));
      return button;
    });
    pipelineSteps.replaceChildren(...layerButtons);
    get('source-label').textContent = source.count < source.total ? 'Clip' : 'Video';
    sourceTime.min = String(source.first);
    sourceTime.max = String(source.first + source.count - 1);
    sourceTimeAt(source.frame - source.first);
  }
  function selectLayer(key) {
    if (!browsingSource || !evidence) return;
    if (!original && activeLayer === key) {
      toSpace();
      return;
    }
    original = false;
    pose.target = 0;
    lens.target = 0;
    aimLayer(key);
  }

  function aimLayer(key) {
    activeLayer = key;
    if (lens.target === 0) {
      const shot = world.shots[key];
      cameraHeld = false;
      yawTarget = shot.yaw;
      elevationTarget = shot.lift;
      zoomTarget = 1;
    }
    reflectControls();
  }

  const stops = get('stops'), dateReadout = get('timeline-date');
  let markedTick = -1;
  const fractions = records.map(record => record.tick / Math.max(1, maxTick));
  const markers = records.map((record, i) => {
    const point = fractions[i];
    const left = i === 0 ? 0 : (point + fractions[i - 1]) / 2;
    const right = i === records.length - 1 ? 1 : (point + fractions[i + 1]) / 2;
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.record = record.key;
    button.title = record.title ?? record.label;
    button.setAttribute('aria-label', button.title);
    button.style.left = left * 100 + '%';
    button.style.width = (right - left) * 100 + '%';
    button.style.setProperty('--point', (point - left) / (right - left) * 100 + '%');
    const caption = document.createElement(record.date ? 'time' : 'span');
    if (record.date) caption.dateTime = record.date;
    caption.className = 'date';
    caption.textContent = record.label;
    button.append(caption);
    button.addEventListener('click', () => {
      reviewRecord();
      selectRecord(record);
    });
    stops.append(button);
    return {button, caption};
  });

  function labelTicks() {
    const width = stops.clientWidth;
    let last = 0;
    markers.forEach(({caption}, i) => {
      const x = fractions[i] * width;
      const visible = i === 0 || i === markers.length - 1 || (x - last >= 51 && width - x >= 51);
      caption.hidden = !visible;
      if (visible) last = x;
    });
  }
  const partButtons = parts.map(part => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = part.label;
    button.dataset.part = part.id;
    button.addEventListener('click', () => {
      reviewRecord();
      selectedPart = part;
      inspect(part.yaw, recordElevation);
    });
    get('parts').append(button);
    return button;
  });

  function regionView({id, triangles, coverage}) {
    const index = new THREE.BufferAttribute(decode(triangles, Uint16Array), 1);
    const opacity = new THREE.BufferAttribute(decode(coverage, Uint8Array), 1, true);
    const vertices = new Set(index.array);
    const box = new THREE.Box3(), point = new THREE.Vector3();
    // Include every recorded shape, so camera scale stays fixed during comparison.
    for (const {delta} of records) {
      for (const vertex of vertices) {
        const i = vertex * 3;
        point.set(base[i] + (delta?.[i] ?? 0), base[i + 1] + (delta?.[i + 1] ?? 0), base[i + 2] + (delta?.[i + 2] ?? 0));
        box.expandByPoint(point);
      }
    }
    if (id === 'arms') box.max.y = Math.max(box.max.y, stature * 1700);
    box.min.multiplyScalar(.001).add(station);
    box.max.multiplyScalar(.001).add(station);
    const bounds = Array.from({length: 8}, () => new THREE.Vector3());
    boxCorners(box, bounds);
    const withFaces = (full, faces) => {
      const region = new THREE.BufferGeometry().setIndex(faces);
      for (const [name, attribute] of Object.entries(full.attributes)) region.setAttribute(name, attribute);
      return region;
    };
    const shape = withFaces(geometry, index).setAttribute('coverage', opacity);
    const reference = withFaces(wholeBody.reference, index).setAttribute('coverage', opacity);
    const part = {id, ...settings[id], bounds, shape, reference};
    if (id === 'arms') {
      // The focused faces keep their order in the full mesh.
      const remainder = new Uint16Array(indices.length - index.count);
      let arm = 0, rest = 0;
      for (let at = 0; at < indices.length; at += 3) {
        if (indices[at] === index.array[arm] && indices[at + 1] === index.array[arm + 1] && indices[at + 2] === index.array[arm + 2]) {
          arm += 3;
        } else {
          remainder.set(indices.subarray(at, at + 3), rest);
          rest += 3;
        }
      }
      const remainingFaces = new THREE.BufferAttribute(remainder, 1);
      part.context = {
        shape: withFaces(geometry, remainingFaces),
        reference: withFaces(wholeBody.reference, remainingFaces),
      };
    }
    return part;
  }

  function redraw() {
    if (!frame) frame = requestAnimationFrame(draw);
  }

  function updateShape() {
    if (!shapeDirty) return;
    shapeDirty = false;
    const vertices = position.array, a = before.delta, b = after.delta;
    for (let i = 0; i < base.length; i++) {
      const start = a?.[i] ?? 0, end = b?.[i] ?? 0;
      vertices[i] = base[i] + Math.fround(start + (end - start) * fraction);
    }
    position.needsUpdate = true;
    geometry.computeVertexNormals();
    if (targetJoints) {
      const first = before.joints ?? rig.joints, last = after.joints ?? rig.joints;
      for (let i = 0; i < targetJoints.length; i++) targetJoints[i] = THREE.MathUtils.lerp(first[i], last[i], fraction);
    }
  }

  function labelRecord() {
    const exact = before === after;
    const available = archive ? selected.takes?.some(Boolean) : !!capture;
    description.textContent = archive && exact ? after.count + (after.count === 1 ? ' clip' : ' clips') : '';
    description.hidden = !description.textContent;
    capturedView.disabled = breakdownView.disabled = !!takeRequest || !available || (!exact && !browsingSource);
    download.disabled = !exact || holding;
  }

  function updateTime(value) {
    const tick = Math.round(THREE.MathUtils.clamp(value, 0, 1) * maxTick);
    time = tick / Math.max(1, maxTick);
    const index = records.findIndex(record => record.tick >= tick);
    after = records[index];
    before = tick === after.tick ? after : records[index - 1];
    fraction = (tick - before.tick) / Math.max(1, after.tick - before.tick);
    rail.value = String(tick);
    rail.style.setProperty('--progress', time * 100 + '%');
    const nearest = fraction < .5 ? before : after;
    const caption = nearest.title ?? nearest.label;
    if (dateReadout.textContent !== caption) {
      dateReadout.textContent = caption;
      dateReadout.dateTime = nearest.date ?? '';
      dateReadout.style.setProperty('--date-half', dateReadout.offsetWidth / 2 + 'px');
      rail.setAttribute('aria-valuetext', caption);
    }
    dateReadout.style.setProperty('--date-position', time * 100 + '%');
    labelRecord();
    const currentTick = before === after ? index : -1;
    if (currentTick !== markedTick) {
      markers[markedTick]?.button.removeAttribute('aria-current');
      markedTick = currentTick;
      markers[markedTick]?.button.setAttribute('aria-current', 'step');
    }
    shapeDirty = true;
    redraw();
  }

  function selectRecord(record, persist = true) {
    if (record !== selected) cancelTakeLoad();
    selected = record;
    updateTime(record.tick / Math.max(1, maxTick));
    reflectControls();
    if (persist && app?.view === 'archive') {
      const url = new URL(location.href);
      url.searchParams.set('visit', record.key);
      url.searchParams.delete('take');
      url.searchParams.delete('frame');
      if (url.href !== location.href) history.replaceState(history.state, '', url);
    }
  }

  function snapRecord() {
    selectRecord(fraction < .5 ? before : after);
  }

  function signalSource(text) {
    sourceStatus.textContent = text;
    sourceStatus.hidden = !text;
  }

  function playbackSource(value, from) {
    const wasPlaying = playingSource;
    playingSource = value;
    if (value && !document.hidden) capture.play(from);
    else capture.pause();
    sourcePlay.dataset.running = String(value);
    sourcePlay.setAttribute('aria-label', value ? 'Pause video' : 'Play video');
    signalSource(value && capture.frame === null ? 'Loading video' : '');
    if (wasPlaying && !value) rememberFrame();
  }

  function sourceTimeAt(index) {
    const clock = frame => {
      const tenths = Math.round(frame * 10 / source.fps);
      return Math.floor(tenths / 600) + ':' + (tenths % 600 / 10).toFixed(1).padStart(4, '0');
    };
    const original = source.first + index;
    sourceTime.value = String(original);
    sourceTime.style.setProperty('--progress', index * 100 / Math.max(1, source.count - 1) + '%');
    sourceTime.setAttribute('aria-valuetext', 'Frame ' + original + ' · ' + clock(original));
    sourceClock.title = 'Frame ' + original;
    sourceClock.value = clock(index) + ' / ' + clock(source.count);
    signalSource('');
    if (frameToRemember === original) {
      frameToRemember = null;
      rememberFrame();
    }
  }

  function finishSource(failed = false) {
    frameToRemember = null;
    playbackSource(false);
    if (failed) {
      sourceTimeAt((capture.frame ?? source.frame) - source.first);
      signalSource('Video unavailable');
    } else {
      compared = true;
      effect.clear();
      reflectControls();
    }
  }

  function seekFrame(frameNumber, remember = false) {
    const index = frameNumber - source.first;
    if (index < 0 || index >= source.count) return;
    playbackSource(false);
    if (original && pose.target === 1) lookThrough();
    frameToRemember = remember ? frameNumber : null;
    sourceTime.style.setProperty('--progress', index * 100 / Math.max(1, source.count - 1) + '%');
    signalSource(capture.frame === null ? 'Loading frame' : 'Seeking');
    capture.seek(index);
    reflectControls();
  }

  function rememberFrame() {
    if (!browsingSource || !boundTake) return;
    const url = new URL(location.href);
    url.searchParams.set('visit', selected.key);
    url.searchParams.set('take', boundTake);
    url.searchParams.set('frame', String(capture.frame ?? source.frame));
    if (url.href !== location.href) history.replaceState(history.state, '', url);
  }

  function showTakeStatus(message) {
    takeStatus.textContent = message;
    takeStatus.hidden = !message;
  }

  function cancelTakeLoad() {
    takeRequest?.abort();
    takeRequest = null;
    showTakeStatus('');
  }

  async function openTake(key, original) {
    if (!key) return false;
    cancelTakeLoad();
    if (key !== boundTake) {
      const request = new AbortController();
      takeRequest = request;
      capturedView.disabled = breakdownView.disabled = true;
      showTakeStatus('Loading clip');
      try {
        const response = await fetch('/api/captures/' + key, {signal: request.signal});
        if (!response.ok) throw new Error('Source metadata unavailable');
        const payload = await response.json();
        if (takeRequest !== request) return false;
        if (capture) playbackSource(false);
        bindSource(payload);
        boundTake = key;
        takeSelect.replaceChildren(...selected.takes.map((take, index) => {
          const option = new Option('Clip ' + (index + 1), take ?? '');
          option.disabled = !take;
          return option;
        }));
        cameraHeld = false;
        if (activeLayer && evidence) aimLayer(activeLayer);
        showTakeStatus('');
      } catch {
        if (takeRequest === request) showTakeStatus('Could not open this clip. Try again.');
        return false;
      } finally {
        if (takeRequest === request) {
          takeRequest = null;
          takeSelect.value = boundTake ?? '';
          reflectControls();
        }
      }
    }
    useCapture();
    if (supportsReplay && source.motion && Number.isInteger(original)
      && original >= source.first && original < source.first + source.count) {
      compared = true;
      lookThrough();
      seekFrame(original, true);
    } else rememberFrame();
    return true;
  }

  function reflectControls() {
    const view = !browsingSource ? 'shape' : original ? 'video' : 'breakdown';
    const isShape = view === 'shape';
    shell.dataset.view = view;
    partsNav.hidden = !isShape;
    pipelineSteps.hidden = view !== 'breakdown' || !evidence;
    get('observations').hidden = !isShape;
    const player = browsingSource && supportsReplay && Boolean(source?.motion);
    const takePicker = browsingSource && selected.takes?.length > 1;
    get('source-transport').hidden = !player && !takePicker;
    get('source-player').hidden = !player;
    get('take-control').hidden = !takePicker;
    layerButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.layer === activeLayer)));
    capturedView.setAttribute('aria-pressed', String(view === 'video'));
    breakdownView.setAttribute('aria-pressed', String(view === 'breakdown'));
    standard.setAttribute('aria-pressed', String(isShape));
    partButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.part === selectedPart.id)));
    front.setAttribute('aria-pressed', String(Math.abs(yawTarget) < .01));
    side.setAttribute('aria-pressed', String(Math.abs(yawTarget - Math.PI / 2) < .01));
    framing.setAttribute('aria-pressed', String(Math.abs(yawTarget - selectedPart.yaw) < .01));
    labelRecord();
    redraw();
  }

  function toSpace() {
    original = false;
    if (browsingSource) pose.target = 0;
    activeLayer = null;
    lens.target = 0;
    cameraHeld = false;
    yawTarget = browsingSource ? overviewYaw : selectedPart.yaw;
    elevationTarget = browsingSource ? isoElevation : recordElevation;
    zoomTarget = 1;
    reflectControls();
  }

  function holdCamera() {
    if (returning) {
      pose.jump(1);
      lens.value = lens.target = 0;
      returning = original = false;
    }
    cameraHeld = true;
    lens.target = lens.value;
    lens.velocity = 0;
    yawTarget = yaw;
    elevationTarget = elevation;
    zoomTarget = zoom;
    reflectControls();
  }

  function useCapture() {
    if (started !== null) {
      stopPlaying();
      snapRecord();
    }
    holdStart(false);
    if (browsingSource) return;
    if (!returning) savedView = {yaw, elevation, zoom};
    returning = false;
    browsingSource = true;
    pose.target = 0;
    toSpace();
  }

  function lookThrough() {
    useCapture();
    if (!original && playingSource) effect.start();
    original = true;
    compared ||= !supportsReplay || !source.motion;
    activeLayer = 'fit';
    cameraHeld = false;
    pose.target = 0;
    lens.target = 1;
    if (!playingSource) effect.clear();
    reflectControls();
  }

  function standSource() {
    if (!browsingSource || !original) return;
    pose.target = 1;
    lens.target = 0;
    cameraHeld = false;
    yawTarget = recordYaw;
    elevationTarget = recordElevation;
    zoomTarget = 1;
    reflectControls();
  }

  function startSource(from) {
    compared = false;
    if (original) {
      effect.start(pose.value > .14 && body.visible ? body : null);
      pose.target = 0;
      lens.target = 1;
      cameraHeld = false;
    }
    playbackSource(true, from);
    reflectControls();
  }

  function playMeasureTail() {
    lookThrough();
    if (!supportsReplay || !source.motion) return;
    playbackSource(false);
    startSource((source.measure_start ?? source.frame) - source.first);
  }

  function reviewRecord() {
    const loading = !!takeRequest;
    if (loading) cancelTakeLoad();
    if (started !== null) {
      stopPlaying();
      snapRecord();
    }
    if (!browsingSource) {
      if (loading) reflectControls();
      return;
    }
    returning = !reducedMotion && (pose.value < 1 || lens.value > 0);
    browsingSource = false;
    if (!returning) original = false;
    pose.target = 1;
    playbackSource(false);
    capture.freeze();
    frameToRemember = null;
    sourceTimeAt((capture.frame ?? source.frame) - source.first);
    lens.target = 0;
    activeLayer = null;
    cameraHeld = false;
    ({yaw: yawTarget, elevation: elevationTarget, zoom: zoomTarget} = savedView);
    selectRecord(selected);
  }

  function inspect(yawAngle, lift) {
    cameraHeld = false;
    lens.target = 0;
    yawTarget = yawAngle;
    elevationTarget = lift;
    zoomTarget = 1;
    reflectControls();
  }

  function resize() {
    width = Math.max(1, viewport.clientWidth);
    height = Math.max(1, viewport.clientHeight);
    renderer.setSize(width, height, false);
    world.layout(width / height);
    capture?.fit(width, height);
    labelTicks();
    dateReadout.style.setProperty('--date-half', dateReadout.offsetWidth / 2 + 'px');
    redraw();
  }

  let activeCamera = orbit;
  function bodyBounds() {
    const mesh = body.geometry;
    const position = mesh.attributes.position;
    if (measuredPosition !== position || measuredVersion !== position.version) {
      mesh.computeBoundingBox();
      measuredPosition = position;
      measuredVersion = position.version;
    }
    body.updateWorldMatrix(true, false);
    boxCorners(mesh.boundingBox, viewCorners);
    for (const corner of viewCorners) corner.applyMatrix4(body.matrixWorld);
    return viewCorners;
  }

  function containBody(camera, corners, strength = 1) {
    const {left, right, bottom, top} = projectedFrame(corners, camera, screenExtent);
    const edge = .94;
    const scale = Math.min(1, 2 * edge / (right - left), 2 * edge / (top - bottom));
    const x = THREE.MathUtils.clamp(0, -edge - left * scale, edge - right * scale);
    const y = THREE.MathUtils.clamp(0, -edge - bottom * scale, edge - top * scale);
    if (scale === 1 && x === 0 && y === 0) return;
    const correctedScale = THREE.MathUtils.lerp(1, scale, strength);
    viewCorrection.makeScale(correctedScale, correctedScale, 1);
    viewCorrection.elements[12] = x * strength;
    viewCorrection.elements[13] = y * strength;
    camera.projectionMatrix.premultiply(viewCorrection);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }

  function alignPlate(ready) {
    screenFrame.hidden = !ready;
    wipeX.value = -1;
    if (!ready) return;
    const {left, right, top, bottom} = projectedFrame(capture.bounds, activeCamera, screenExtent);
    const tracked = evidence?.camera;
    const center = tracked ? (tracked[9] + tracked[11] / 2) / source.width : .5;
    comparisonSplit ??= THREE.MathUtils.clamp(center, .12, .88);
    const x = (left + (right - left) * comparisonSplit + 1) * width / 2;
    screenFrame.style.left = x + 'px';
    screenFrame.style.top = (1 - top) * height / 2 + 'px';
    screenFrame.style.height = (top - bottom) * height / 2 + 'px';
    wipeX.value = x * renderer.getPixelRatio();
  }

  function fitBounds(points, center, aspect, margin) {
    let loX = Infinity, hiX = -Infinity, loY = Infinity, hiY = -Infinity, loZ = Infinity, hiZ = -Infinity;
    for (const point of points) {
      const x = point.dot(horizontal), y = point.dot(vertical), z = point.dot(direction);
      loX = Math.min(loX, x); hiX = Math.max(hiX, x);
      loY = Math.min(loY, y); hiY = Math.max(hiY, y);
      loZ = Math.min(loZ, z); hiZ = Math.max(hiZ, z);
    }
    center.copy(horizontal).multiplyScalar((loX + hiX) / 2)
      .addScaledVector(vertical, (loY + hiY) / 2).addScaledVector(direction, (loZ + hiZ) / 2);
    return Math.max(hiY - loY, (hiX - loX) / aspect) * margin;
  }

  function draw(now) {
    frame = 0;
    const dt = lastFrame ? Math.min(.034, (now - lastFrame) / 1000) : 1 / 60;
    lastFrame = now;
    const follow = reducedMotion ? 1 : 1 - Math.exp(-dt / .096);
    const optical = original && (browsingSource || returning);
    pose.update(now);
    if (!cameraHeld) {
      if (optical && (pose.value > 0 || pose.target === 1)) {
        lens.value = 1 - pose.value;
        lens.velocity = 0;
      } else advance(lens, dt, 15, reducedMotion);
    }
    if (returning && lens.value === 0 && pose.value === 1) {
      returning = original = false;
    }
    const videoOnly = browsingSource && original && pose.target === 0;
    const showComparison = compared && videoOnly && pose.value === 0 && lens.value === 1 && capture.picture.visible;
    const overlay = effect?.update(dt, optical && capture.advancing, optical ? pose.value : 1, renderer.getPixelRatio()) ?? 0;
    const cameraFollow = cameraHeld ? 0 : follow;
    const rotationFollow = contact?.moved ? 1 : cameraFollow;
    yaw = easeTo(yaw, yawTarget, rotationFollow, .001);
    elevation = easeTo(elevation, elevationTarget, rotationFollow, .001);
    zoom = easeTo(zoom, zoomTarget, cameraFollow, .002);
    const recordView = !browsingSource || (original && pose.target === 1);
    const recordTarget = Number(recordView);
    recordMix = THREE.MathUtils.lerp(recordMix, recordTarget, cameraFollow);
    if (Math.abs(recordMix - recordTarget) < .002) recordMix = recordTarget;
    if (started !== null) {
      started ||= now;
      updateTime(resumeAt + (now - started) / 4200);
      if (time === 1) {
        stopPlaying();
        selectRecord(records.at(-1));
      }
    }
    updateShape();

    const c = Math.cos(elevation), aspect = width / height;
    direction.set(Math.sin(yaw) * c, Math.sin(elevation), Math.cos(yaw) * c);
    horizontal.set(Math.cos(yaw), 0, -Math.sin(yaw));
    vertical.crossVectors(direction, horizontal);
    const part = !browsingSource && !returning ? selectedPart : wholeBody;
    const shot = !recordView && world.shots?.[activeLayer];
    let extent;
    if (recordView) {
      if (part.bounds) extent = fitBounds(part.bounds, target, aspect, 1.2);
      else {
        target.copy(station);
        target.y += .85 * stature;
        extent = stature * Math.max(1.94, 1.06 / aspect);
      }
    } else if (shot) {
      extent = fitBounds(shot.bounds, target, aspect, 1.15);
    } else {
      extent = fitBounds(world.bounds, target, aspect, 1.12);
    }
    if (!size) gaze.copy(target);
    gaze.lerp(target, cameraFollow);
    if (cameraFollow && gaze.distanceToSquared(target) <= .002 ** 2) gaze.copy(target);
    const mediaView = evidence && !recordView && (!shot || activeLayer === 'source');
    const targetSize = extent / (mediaView ? Math.min(1, zoom) : zoom);
    size = size ? easeTo(size, targetSize, cameraFollow, .002) : targetSize;
    orbit.position.copy(gaze).addScaledVector(direction, 14);
    orbit.lookAt(gaze);
    orbit.left = -size * aspect / 2;
    orbit.right = -orbit.left;
    orbit.top = size / 2;
    orbit.bottom = -orbit.top;
    orbit.updateProjectionMatrix();
    orbit.updateMatrixWorld();

    const travel = lens.value;
    const fitTravel = THREE.MathUtils.smoothstep(travel, .32, .99);
    const approach = activeLayer === 'fit' ? fitTravel : THREE.MathUtils.smoothstep(travel, .14, .96);
    activeCamera = capture ? matchCamera(capture, orbit, gaze, travel, approach) : orbit;
    const context = optical && pose.target === 1 ? 0 : (1 - THREE.MathUtils.smoothstep(recordMix, .1, .84))
      * (videoOnly ? 1 - fitTravel : 1);
    const bodyPose = optical ? pose.value : pose.value * (1 - THREE.MathUtils.smoothstep(travel, .35, .86));
    figure.place(holding ? part.reference : part.shape, capture, bodyPose, holding ? rig?.joints : targetJoints);
    armContext.visible = Boolean(part.context && bodyPose === 1 && travel === 0);
    if (armContext.visible) armContext.geometry = holding ? part.context.reference : part.context.shape;
    model.position.addScaledVector(world.fitShift, context * (1 - bodyPose) * (1 - fitTravel));
    if (recordView && travel < 1) {
      const framing = returning ? 1 - THREE.MathUtils.smoothstep(travel, .61, .98) : 1;
      containBody(activeCamera, part.bounds ?? bodyBounds(), framing);
    } else if (videoOnly && travel < 1) {
      const photoFits = frameVisibility(capture.bounds, activeCamera);
      containBody(activeCamera, bodyBounds(), 1 - THREE.MathUtils.smoothstep(photoFits, .5, .98));
    }
    let changingLayer = false;
    for (const key of Object.keys(prominence)) {
      const goal = shot && activeLayer !== key ? 0 : 1;
      let value = THREE.MathUtils.lerp(prominence[key], goal, cameraFollow);
      if (Math.abs(value - goal) < .003) value = goal;
      prominence[key] = value;
      changingLayer ||= value !== goal;
    }
    const assembled = THREE.MathUtils.smoothstep(travel, .17, .88);
    const visible = key => THREE.MathUtils.lerp(prominence[key] ?? 1, 1, assembled);
    if (capture) {
      const filmVisibility = visible('source') * frameVisibility(capture.bounds, activeCamera);
      const guideOpacity = filmVisibility * (evidence ? (activeLayer === 'source' ? .55 : 0) : 1);
      fadeGuides(guideOpacity * context * (1 - THREE.MathUtils.smoothstep(travel, .38, .86)));
      fadeFinder(guideOpacity * context * (1 - THREE.MathUtils.smoothstep(travel, .12, .44)));
      const rear = activeCamera.position.z < capture.group.position.z + capture.picture.position.z;
      capture.picture.renderOrder = rear ? 5 : -1;
      const opacity = rear ? .22 : THREE.MathUtils.lerp(.81, .53, bodyPose);
      const dissolve = optical ? 1 - THREE.MathUtils.smoothstep(pose.value, .07, .75) : 1;
      capture.picture.material.opacity = filmVisibility
        * THREE.MathUtils.lerp(context * opacity, 1, travel) * dissolve;
      renderer.domElement.dataset.sourceFrame = String(capture.frame ?? source.frame);
    }
    world.update({travel, context: context * (evidence ? 1 - bodyPose : 1), active: activeLayer,
      frame: capture?.frame ?? source?.frame, camera: activeCamera, visible, pixelRatio: renderer.getPixelRatio(), width});
    let glass = evidence
      ? THREE.MathUtils.lerp(1, THREE.MathUtils.lerp(visible('fit'), 1, travel), context * (1 - bodyPose)) : 1;
    if (optical && !showComparison) {
      const sourceReveal = THREE.MathUtils.smoothstep(capture.picture.material.opacity, .10, .88);
      const transfer = videoOnly && !playingSource && travel < 1 ? 1 - sourceReveal : 0;
      glass *= Math.max(transfer, THREE.MathUtils.lerp(overlay, 1, pose.value));
    } else if (!recordView && travel < 1 && glass > .003) glass *= frameVisibility(bodyBounds(), activeCamera);
    const onFilm = THREE.MathUtils.smoothstep(travel, .36, .91);
    figure.paint(glass, THREE.MathUtils.lerp(.073, .19, onFilm) * glass);
    const hideExport = browsingSource || pose.value !== 1;
    if (download.hidden !== hideExport) download.hidden = hideExport;
    alignPlate(showComparison);
    const inspect = part !== wholeBody && !holding && travel === 0 && pose.value === 1;
    tape?.show(inspect && before === after ? part.id : null, records[0], after);
    renderer.render(scene, activeCamera);
    const outline = inspect && time > 0;
    contourLegend.hidden = !outline;
    if (outline) {
      drawContours ??= createContours(renderer, station);
      drawContours(activeCamera, part.reference, part.shape);
    }

    const movingCamera = !cameraHeld && (travel !== lens.target || Math.abs(yaw - yawTarget) > .001
      || Math.abs(elevation - elevationTarget) > .001 || Math.abs(zoom - zoomTarget) > .002
      || Math.abs(size - targetSize) > .002 || gaze.distanceTo(target) > .002 || recordMix !== recordTarget);
    if ((optical && playingSource && effect?.animating) || movingCamera || (!cameraHeld && changingLayer) || pose.value !== pose.target) redraw();
  }

  function stopPlaying(paused = false) {
    started = null;
    resumeAt = paused ? time : 0;
    play.dataset.running = 'false';
    playLabel.textContent = paused ? 'Resume' : 'Play';
    play.setAttribute('aria-label', paused ? 'Resume timeline' : 'Play timeline');
  }

  async function showCapture(asVideo) {
    if (archive && !browsingSource && !returning) {
      if (started !== null) {
        stopPlaying();
        snapRecord();
      }
      const key = selected.takes.includes(boundTake) ? boundTake : selected.takes.find(Boolean);
      if (key !== boundTake && !await openTake(key)) return;
    }
    if (asVideo) lookThrough();
    else if (browsingSource) toSpace();
    else useCapture();
  }
  capturedView.addEventListener('click', () => showCapture(true));
  breakdownView.addEventListener('click', () => showCapture(false));
  takeSelect.addEventListener('change', () => openTake(takeSelect.value));
  standard.addEventListener('click', reviewRecord);
  standard.addEventListener('touchend', event => {
    if (event.touches.length || event.changedTouches.length !== 1) return;
    const {clientX, clientY} = event.changedTouches[0];
    if (!standard.contains(document.elementFromPoint(clientX, clientY))) return;
    // Chromium can suppress the click after a sequence of captured touch drags.
    event.preventDefault();
    reviewRecord();
  }, {passive: false});
  sourcePlay.addEventListener('click', () => {
    const pause = playingSource;
    if (pause) {
      playbackSource(false);
      standSource();
    } else {
      const replay = original && capture.frame === source.first + source.count - 1;
      startSource(replay ? (source.measure_start ?? source.frame) - source.first : undefined);
    }
  });
  sourceTime.addEventListener('pointerdown', () => {
    playbackSource(false);
    reflectControls();
  });
  sourceTime.addEventListener('input', () => seekFrame(sourceTime.valueAsNumber));
  sourceTime.addEventListener('change', () => {
    const original = sourceTime.valueAsNumber;
    if (original === (capture.frame ?? source.frame)) rememberFrame();
    else frameToRemember = original;
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) capture?.pause();
    else if (playingSource) capture.play();
  });
  play.addEventListener('click', () => {
    if (started !== null) {
      snapRecord();
      return stopPlaying(time !== 1);
    }
    reviewRecord();
    started = 0;
    play.dataset.running = 'true';
    playLabel.textContent = 'Pause';
    play.setAttribute('aria-label', 'Pause timeline');
    redraw();
  });
  rail.addEventListener('pointerdown', reviewRecord);
  rail.addEventListener('input', () => {
    const value = Number(rail.value) / Math.max(1, maxTick);
    reviewRecord();
    updateTime(value);
  });
  rail.addEventListener('change', snapRecord);
  rail.addEventListener('keydown', event => {
    const step = {ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -1, PageUp: 1}[event.key];
    if (step === undefined && event.key !== 'Home' && event.key !== 'End') return;
    event.preventDefault();
    reviewRecord();
    const index = event.key === 'Home' ? 0 : event.key === 'End' ? records.length - 1 : records.indexOf(selected) + step;
    selectRecord(records[THREE.MathUtils.clamp(index, 0, records.length - 1)]);
  });
  front.addEventListener('click', () => inspect(0, .05));
  side.addEventListener('click', () => inspect(Math.PI / 2, .05));
  framing.addEventListener('click', () => inspect(selectedPart.yaw, recordElevation));

  function holdStart(value) {
    if (holding === value) return;
    holding = value;
    compare.textContent = value ? 'Release for Current' : 'Hold for Start';
    compare.setAttribute('aria-pressed', String(value));
    download.disabled = value || before !== after;
    redraw();
  }

  compare.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    reviewRecord();
    compare.setPointerCapture(event.pointerId);
    holdStart(true);
  });
  compare.addEventListener('lostpointercapture', () => holdStart(false));
  compare.addEventListener('keydown', event => {
    if (event.key !== ' ' && event.key !== 'Enter') return;
    event.preventDefault();
    reviewRecord();
    holdStart(true);
  });
  compare.addEventListener('keyup', event => {
    if (event.key === ' ' || event.key === 'Enter') holdStart(false);
  });
  compare.addEventListener('blur', () => holdStart(false));

  download.addEventListener('click', () => {
    if (started !== null) stopPlaying();
    snapRecord();
    updateShape();
    const bytes = new STLExporter().parse(new THREE.Mesh(geometry), {binary: true});
    const url = URL.createObjectURL(new Blob([bytes], {type: 'model/stl'}));
    const link = document.createElement('a');
    link.href = url;
    link.download = selected.stl;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  viewport.addEventListener('pointerdown', event => {
    if (event.button !== 0 || event.target.closest('button') || !screenFrame.hidden) return;
    holdCamera();
    contact = {id: event.pointerId, x: event.clientX, y: event.clientY, yaw, elevation, lens: lens.value,
      at: performance.now(), moved: false};
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener('pointermove', event => {
    if (contact?.id !== event.pointerId) return;
    const x = event.clientX - contact.x, y = event.clientY - contact.y;
    if (!contact.moved && Math.hypot(x, y) < 4) return;
    cameraHeld = false;
    contact.moved = true;
    lens.value = lens.target = contact.lens * Math.max(0, 1 - Math.hypot(x, y) / 175);
    yawTarget = contact.yaw - x * .008;
    elevationTarget = THREE.MathUtils.clamp(contact.elevation + y * .006, .03, 1.28);
    reflectControls();
  });
  viewport.addEventListener('pointerup', event => {
    if (!browsingSource || contact?.id !== event.pointerId || contact.moved || performance.now() - contact.at > 260
      || lens.value > .90) return;
    const bounds = viewport.getBoundingClientRect();
    world.pick(activeCamera, event.clientX - bounds.left, event.clientY - bounds.top, width, height,
      [capture?.finder, capture?.picture.material.opacity > .02 ? capture.picture : null,
        evidence && model.visible ? body : null]);
  });
  viewport.addEventListener('lostpointercapture', () => { contact = null; });
  viewport.addEventListener('wheel', event => {
    if (lens.value === 1) return;
    event.preventDefault();
    cameraHeld = false;
    zoomTarget = THREE.MathUtils.clamp(zoomTarget * Math.exp(-event.deltaY * .0017), .8, 2.3);
    redraw();
  }, {passive: false});
  viewport.addEventListener('keydown', event => {
    if (event.target !== viewport || !screenFrame.hidden) return;
    if (event.key === 'Home') {
      event.preventDefault();
      return original && browsingSource ? lookThrough() : toSpace();
    }
    const direction = {ArrowLeft: -1, ArrowRight: 1}[event.key];
    if (!direction) return;
    event.preventDefault();
    cameraHeld = false;
    lens.target = 0;
    yawTarget += direction * .18;
    reflectControls();
  });
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape' || document.querySelector('dialog[open]')) return;
    holdCamera();
    if (capture && playingSource) {
      playbackSource(false);
      standSource();
    }
    if (started !== null) stopPlaying(true);
    holdStart(false);
    reflectControls();
  });

  if (data.source) bindSource(data.source);
  compare.hidden = records.length === 1;
  selectRecord(selected, false);
  const params = new URLSearchParams(location.search);
  if (!archive && capture) {
    lens.value = 1;
    pose.jump(0);
    if (reducedMotion || params.has('add')) lookThrough();
    else playMeasureTail();
  }
  resize();
  new ResizeObserver(() => resize()).observe(viewport);
  const take = params.get('take');
  if (take && selected.takes?.includes(take)) openTake(take, Number.parseInt(params.get('frame'), 10));
}

const app = JSON.parse(get('app-data').textContent);
try {
  let data = JSON.parse(get('mesh-data').textContent), navigation = app;
  if (!app && data.people?.[0]?.records) {
    const user = new URLSearchParams(location.search).get('user');
    const person = data.people.find(item => item.id === user);
    navigation = {embedded: true, group: true};
    if (person) data = {...person, subject: person.name, mode: 'demo', triangles: data.triangles};
  }
  if (data.people) openGroup(navigation, data);
  else initialize(navigation, data);
} catch (error) {
  const message = get('scene-error');
  message.textContent = '3D could not start. Open in a browser with WebGL.';
  message.hidden = false;
  console.error(error);
}
connectIntake(app);
