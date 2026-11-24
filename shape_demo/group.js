import * as THREE from 'three';
import {createSource} from './source.js';
import {blendCameras, createFigure, decode} from './figure.js';
import {playbackEffect, stanceMotion, STANCE_SECONDS, TAIL_LEAD_SECONDS} from './playback.js';
import {frameVisibility, station} from './stage.js';

const get = id => document.getElementById(id);
const {clamp, lerp, smootherstep, smoothstep} = THREE.MathUtils;
const RETREAT_SECONDS = 1.08;
const HANDOFF_SECONDS = .42;

function subjectUrl(user, embedded) {
  const url = new URL(embedded ? location.href : '/', location.href);
  url.search = url.hash = '';
  if (user) url.searchParams.set('user', user);
  else url.searchParams.set('people', '1');
  return url.href;
}

export function peopleLink(app, label) {
  if (!app?.group) return;
  const link = get('people-link');
  link.href = subjectUrl(null, app.embedded);
  link.textContent = label;
  link.hidden = false;
}

function gridCell(index, [x, y, w, h], columns, count) {
  const width = w / columns, height = h / Math.ceil(count / columns);
  return [x + index % columns * width, y + Math.floor(index / columns) * height, width, height];
}

function moveFrame(from, to, amount, bendX, bendY) {
  const [x, y, w, h] = from, [tx, ty, tw, th] = to;
  const width = w * Math.pow(tw / w, amount), height = h * Math.pow(th / h, amount);
  const arc = Math.sin(Math.PI * amount);
  return [
    lerp(x + w / 2, tx + tw / 2, amount) + bendX * arc - width / 2,
    lerp(y + h / 2, ty + th / 2, amount) + bendY * arc - height / 2,
    width, height,
  ];
}

export function openGroup(app, {people, triangles}) {
  const picking = new URLSearchParams(location.search).get('people') === '1';
  if (!picking) peopleLink(app, 'View All');
  get('mode').hidden = true;
  get('shell').dataset.view = 'group';
  const group = get('group');
  group.hidden = false;
  const stage = get('group-people'), play = get('group-play'), rail = get('group-time');
  const status = get('group-status'), action = get('group-action');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const narrow = matchMedia('(max-width: 820px)');
  const renderer = new THREE.WebGLRenderer({alpha: true, antialias: true, powerPreference: 'high-performance'});
  const density = Math.min(devicePixelRatio, 1.5);
  renderer.setPixelRatio(density);
  renderer.setClearColor(0x000000, 0);
  renderer.autoClear = false;
  renderer.domElement.setAttribute('aria-hidden', 'true');
  stage.append(renderer.domElement);

  let nextEntrance = 0, loaded = false;
  let state = 'loading', elapsed = 0, began = 0, frame = 0, lastDraw = 0, width = 0, height = 0;
  const gaze = station.clone().add(new THREE.Vector3(0, .85, 0));
  const index = new THREE.BufferAttribute(decode(triangles, Uint16Array), 1);
  const match = blendCameras();
  const supported = 'requestVideoFrameCallback' in HTMLVideoElement.prototype && 'DecompressionStream' in globalThis;
  const actors = people.map(person => {
    const card = document.createElement(app ? 'a' : 'div'), name = document.createElement('span');
    card.className = 'group-person';
    card.dataset.person = person.id;
    if (app) card.href = subjectUrl(person.id, app.embedded);
    name.textContent = person.name;
    card.append(name);
    stage.append(card);

    const positions = decode(person.positions, Float32Array), neutral = new THREE.BufferGeometry();
    neutral.setIndex(index);
    neutral.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    neutral.computeVertexNormals();
    const figure = createFigure(neutral, person, station), scene = new THREE.Scene();
    scene.name = person.id;
    const orbit = new THREE.OrthographicCamera(-1, 1, 1, -1, .05, 40);
    orbit.position.copy(gaze).add(new THREE.Vector3(6.0, 1.45, 13));
    orbit.lookAt(gaze);
    orbit.updateMatrixWorld();
    const {fps, count, first} = person.source;
    const movie = count / fps;
    const turnAt = Math.max(0, movie - TAIL_LEAD_SECONDS);
    const retreatAt = turnAt + STANCE_SECONDS + .10;
    const actor = {card, orbit, scene, neutral, figure, fps, count, start: nextEntrance,
      movie, turnAt, retreatAt, pose: stanceMotion(0, reduced), ready: false, playing: false, ended: false};
    nextEntrance += retreatAt + RETREAT_SECONDS - HANDOFF_SECONDS;
    const capture = createSource(person.source, figure.film, station, redraw, at => {
      card.dataset.frame = String(first + at);
    }, failed => {
      actor.playing = false;
      actor.ended = true;
      if (failed) {
        actor.ready = false;
        showUnavailable();
      }
      redraw();
    }, 880);
    actor.capture = capture;
    actor.effect = playbackEffect(capture, reduced);
    capture.guides.visible = capture.finder.visible = false;
    scene.add(capture.group, figure.model);
    return actor;
  });

  const lineupAt = nextEntrance + HANDOFF_SECONDS;
  const duration = lineupAt + .85;

  function timeAt(now) {
    return Math.min(duration, elapsed + (state === 'playing' ? Math.max(0, now - began) / 1000 : 0));
  }

  function setState(next) {
    state = next;
    group.dataset.state = state;
    const label = {loading: 'Loading', playing: 'Pause', paused: 'Resume',
      ended: picking ? 'Play' : 'Replay', scrubbing: 'Play'}[state];
    play.dataset.running = String(state === 'playing');
    play.disabled = rail.disabled = !loaded;
    play.setAttribute('aria-label', label);
    action.textContent = label;
    redraw();
  }

  function showUnavailable() {
    const missing = actors.filter(actor => !actor.ready).length;
    status.textContent = missing ? 'Video unavailable for ' + missing + (missing === 1 ? ' person.' : ' people.') : '';
  }

  function pauseMedia(actor) {
    if (!actor.playing) return;
    actor.capture.pause();
    actor.playing = false;
    actor.resume = true;
  }

  function seekTo(next) {
    elapsed = next;
    for (const actor of actors) {
      pauseMedia(actor);
      actor.effect.clear();
      const local = next - actor.start;
      actor.ended = local >= actor.movie;
      actor.pose.jump(local >= actor.movie || !actor.ready ? 1 : 0);
      actor.resume = local >= 0 && local < actor.movie;
      if (local >= 0 && actor.ready) {
        const at = clamp(Math.round(local * actor.fps), 0, actor.count - 1);
        if (at !== actor.requested) actor.capture.seek(at);
        actor.requested = at;
      }
    }
  }

  function playShow() {
    if (elapsed >= duration) {
      elapsed = 0;
      for (const actor of actors) {
        actor.effect.clear();
        actor.pose.jump(0);
        actor.ended = actor.resume = false;
        actor.requested = undefined;
      }
    }
    began = performance.now();
    setState('playing');
  }

  function pauseShow() {
    elapsed = timeAt(performance.now());
    for (const actor of actors) pauseMedia(actor);
    setState('paused');
  }

  function resize() {
    const rect = stage.getBoundingClientRect();
    width = Math.max(1, rect.width);
    height = Math.max(1, rect.height);
    renderer.setSize(width, height, false);
    redraw();
  }

  function redraw() {
    frame ||= requestAnimationFrame(draw);
  }

  function draw(now) {
    frame = 0;
    const dt = lastDraw ? Math.min(.041, (now - lastDraw) / 1000) : 1 / 60;
    lastDraw = now;
    const time = timeAt(now);
    if (state === 'playing' && time === duration) {
      elapsed = duration;
      setState('ended');
    }
    const progress = time / duration;
    rail.value = String(Math.round(progress * 1000));
    rail.style.setProperty('--progress', progress * 100 + '%');
    const mobile = narrow.matches;
    const hero = mobile ? [0, 0, width, height * .58] : [0, 0, width * .59, height];
    const shelf = mobile ? [0, height * .665, width, height * .335] : [width * .62, 0, width * .38, height];
    const lineup = reduced ? Number(time >= lineupAt) : smootherstep(time, lineupAt, duration);
    let turning = false;
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.setScissorTest(true);
    for (const [i, actor] of actors.entries()) {
      const local = time - actor.start;
      actor.card.hidden = local < 0;
      if (local < 0) continue;
      const {capture, figure, orbit, neutral, scene, pose, effect} = actor;
      const wantsMovie = state === 'playing' && actor.ready && !actor.ended && (local < actor.movie || actor.playing);
      if (wantsMovie && !actor.playing) {
        effect.start(pose.value > .14 && figure.body.visible && actor.resume ? figure.body : null);
        actor.playing = true;
        capture.play(actor.resume ? undefined : Math.min(actor.count - 1, Math.floor(local * actor.fps)));
        actor.requested = undefined;
      } else if (!wantsMovie) pauseMedia(actor);
      pose.target = Number(state !== 'loading' && (!actor.ready || local >= actor.turnAt
        || (state !== 'playing' && state !== 'scrubbing')));
      const stand = pose.update(now);
      turning ||= stand !== pose.target;

      const arrive = reduced ? 1 : smootherstep(local, 0, 1.08);
      const dock = reduced ? Number(local >= actor.retreatAt)
        : smootherstep(local, actor.retreatAt, actor.retreatAt + RETREAT_SECONDS);
      const compact = gridCell(i, shelf, mobile ? 4 : 2, actors.length);
      const wide = gridCell(i, [0, 0, width, height], mobile ? 2 : 4, actors.length);
      const seat = compact.map((v, axis) => lerp(v, wide[axis], lineup));
      const firstCenter = !mobile && i === 0 ? (width - hero[2]) / 2 : 0;
      const front = [firstCenter, 0, hero[2], hero[3]];
      const distant = [firstCenter + hero[2] * .11, hero[3] * .18, hero[2] * .46, hero[3] * .46];
      const [x, y, w, h] = dock
        ? moveFrame(front, seat, dock, mobile ? -width * .065 : 0, mobile ? 0 : -height * .075)
        : moveFrame(distant, front, arrive, 0, -hero[3] * .04);
      const caption = lerp(lerp(mobile ? 31 : 38, mobile ? 55 : 64, arrive),
        lerp(mobile ? 19 : 23, mobile ? 26 : 32, lineup), dock);
      actor.card.style.transform = 'translate(' + x + 'px,' + y + 'px)';
      actor.card.style.width = w + 'px';
      actor.card.style.height = h + 'px';
      actor.card.style.setProperty('--caption', caption + 'px');
      const exitName = reduced ? Number(local >= actor.retreatAt)
        : smootherstep(local, actor.turnAt + .45, actor.retreatAt + .18);
      const name = reduced || state === 'loading' ? 1 : lerp(.18, 1, smoothstep(local, 0, .52));
      const titleSize = lerp(mobile ? 19 : 25, mobile ? 28 : 39, arrive);
      actor.card.style.setProperty('--name-size', lerp(titleSize, lerp(mobile ? 11 : 12, mobile ? 13 : 14, lineup), dock) + 'px');
      actor.card.style.setProperty('--name-alpha', Math.max(name * (1 - exitName), smoothstep(dock, .94, 1)));
      actor.card.style.setProperty('--name-depth', lerp(lerp(.92, 1, name) - .11 * exitName, 1, dock));
      actor.card.style.setProperty('--name-weight', dock < .5 ? '300' : '450');
      actor.card.style.zIndex = dock < 1 ? String(i + 1) : '0';
      const imageHeight = Math.max(1, h - caption);
      const aspect = Math.max(.01, w / imageHeight);
      capture.fit(Math.max(1, w), imageHeight);
      const span = Math.max(1.93, 1.10 / aspect);
      orbit.top = span / 2;
      orbit.bottom = -orbit.top;
      orbit.right = orbit.top * aspect;
      orbit.left = -orbit.right;
      orbit.updateProjectionMatrix();

      const travel = 1 - stand;
      const camera = match(capture, orbit, gaze, travel, travel);
      const glassAtSource = effect.update(dt, actor.playing && capture.advancing, stand, density * .79);
      capture.picture.material.opacity = (1 - smoothstep(stand, .06, .70)) * frameVisibility(capture.bounds, camera);
      figure.place(neutral, capture, stand);
      const glass = lerp(glassAtSource, 1, stand);
      figure.paint(glass, lerp(.10, .055, stand) * glass);
      renderer.setViewport(x, height - y - imageHeight, w, imageHeight);
      renderer.setScissor(x, height - y - imageHeight, w, imageHeight);
      renderer.clearDepth();
      renderer.render(scene, camera);
    }
    renderer.setScissorTest(false);
    if (state === 'playing' || turning) redraw();
  }

  play.addEventListener('click', () => state === 'playing' ? pauseShow() : playShow());
  rail.addEventListener('input', () => {
    seekTo(rail.valueAsNumber / 1000 * duration);
    setState('scrubbing');
  });
  rail.addEventListener('change', () => setState(elapsed === duration ? 'ended' : 'paused'));
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && state === 'playing') pauseShow();
  });
  new ResizeObserver(resize).observe(stage);
  resize();
  if (picking) {
    elapsed = duration;
    actors.forEach(actor => actor.pose.jump(1));
  }
  setState(picking ? 'ended' : 'loading');
  Promise.all(actors.map(async actor => {
    if (!supported) return;
    try {
      await actor.capture.ready();
      actor.ready = true;
    } catch {}
  })).then(() => {
    loaded = true;
    showUnavailable();
    if (picking || reduced || document.hidden || !actors.some(actor => actor.ready)) {
      elapsed = duration;
      setState('ended');
    } else playShow();
  });
}
