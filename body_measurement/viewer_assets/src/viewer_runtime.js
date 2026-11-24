import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh';
import { reconstructSurfaceHit } from './surface_measurement.js';
import {
  cleanDisplayText, createMeasurementRecords,
  currentPoseLabel, lowerBodyPoseLabel, positiveNumber,
  round3, round6, tapeReviewLabel,
} from './measurement_records.js';

function element(tag, className, ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  node.append(...children);
  return node;
}

const tapeName = tape => tape.record.name || '软尺';
const isCircumference = tape => tape.record.method === 'mesh_section';

export function start(options = {}) {
  THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
  THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
  THREE.Mesh.prototype.raycast = acceleratedRaycast;

  const payloadElement = document.getElementById(options.payloadElementId || 'viewer-payload');
  const payload = JSON.parse(payloadElement?.textContent || '{}');
  const inspect = new URLSearchParams(location.search).get('inspect');
  const evidenceView = inspect === 'evidence';
  const {
    initialState, poseAtHeight, startDraft, placeTape, tapeFromRecord, confirmTape, resizeState,
    isTapeSavable, totalCrotchAdmitted, poseForRecord,
    currentMeasurementRows, compareReference, isPublicRecord, measurementCsv, readWorkProgress, workProgress,
  } = createMeasurementRecords(payload);
  const referenceMeasurements = Array.isArray(payload.reference_measurements) ? payload.reference_measurements : [];
  const personalCalibration = payload.personal_calibration || null;
  const scaleDisplayLabel = cleanDisplayText(personalCalibration?.scale_display_label);
  const poses = Array.isArray(payload.poses) ? payload.poses : [];
  const meshInfo = payload.mesh || {};
  const sourceDisplay = payload.source_display || {};
  const hasPublicScope = Array.isArray(payload.public_item_ids);
  const legacyProgressKey = /^\/workbench\/[a-f0-9]{32}\/viewer\.html$/.test(location.pathname)
    ? "bmwork:" + location.pathname : '';
  const progressKey = payload.workbench_id && location.protocol !== 'file:'
    ? "bmwork:" + payload.workbench_id : legacyProgressKey;
  if (legacyProgressKey) document.getElementById('uploadHome').hidden = false;
  let state = initialState(restoreWorkProgress());
  let activeTape = null;
  let gesture = null;
  let lengthLabelPoint = null;
  const rowNodes = new Map();
  let publicRows = [];
  let navigationRows = [];
  let frameRequest = 0;
  const TAPE_COLORS = {length: 0x317463, circumference: 0x334adb};
  const container = document.getElementById('viewer3d');
  const scene = new THREE.Scene();
  const raycaster = new THREE.Raycaster();
  const camera = new THREE.PerspectiveCamera(38, 1, 1, 8000);
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.addEventListener('change', requestRender);
  controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
  controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x819099, 2.1));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.3);
  keyLight.position.set(240, 560, 320);
  scene.add(keyLight);
  const tapeGroup = new THREE.Group();
  scene.add(tapeGroup);

  let displayedPose = poseAtHeight(state.selectedPose, state.currentHeightCm);
  const bodyMesh = new THREE.Mesh(bodyGeometry(displayedPose), new THREE.MeshStandardMaterial({
    color: 0xbbb8b1, roughness: 0.65, metalness: 0.02, side: THREE.DoubleSide,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 3,
  }));
  scene.add(bodyMesh);
  camera.aspect = container.clientWidth / Math.max(container.clientHeight, 1);
  frameCamera();
  const measurementProjects = document.getElementById('measurementProjects');
  const downloads = [
    ['downloadModel', '-当前人台-mm.stl', () => new STLExporter().parse(bodyMesh), 'model/stl'],
    ['downloadTable', '-当前量体表.csv', () => measurementCsv(state), 'text/csv;charset=utf-8'],
  ];
  document.getElementById('sourceReview').hidden = !hasPublicScope;
  if (hasPublicScope) setText('measurementListTitle', '已获准量体项');
  const compact = window.matchMedia('(max-width: 700px)');
  if (compact.matches && measurementProjects) measurementProjects.open = false;
  renderSourceReference();
  renderContext();
  refreshWorkbench();
  const openingRecords = publicRows
    .map(row => row.record)
    .filter(record => record && poseForRecord(record) === state.selectedPose);
  const openingRecord = openingRecords.find(record => record.review_status === 'automatic') || openingRecords[0];
  if (!openingRecord || !selectSavedTape(openingRecord)) {
    showTape(null);
  }
  bindEvents();
  window.addEventListener('resize', resize);
  if (typeof ResizeObserver === 'function') {
    new ResizeObserver(() => resize()).observe(container);
  }
  resize();
  if (frameRequest) cancelAnimationFrame(frameRequest);
  drawFrame();

  if (evidenceView) document.querySelector('.stageHelp').textContent = '虚线表示被人台遮挡的尺段';
  if (inspect === '1' || evidenceView) {
    window.__bmProbes = {
      ready: true,
      exportState: () => workProgress(state),
      visualState,
      pickViewportSurface,
    };
  }
  function applyHeightInput() {
    const input = document.getElementById('heightInput');
    const next = positiveNumber(input.value);
    if (!next || next < 120 || next > 230) {
      setText('poseStatus', '请输入 120 至 230 cm 的身高。');
      return;
    }
    const changed = next !== state.currentHeightCm;
    if (changed && activeTape?.definition) {
      setText('poseStatus', '请先确认或取消当前落尺，再修改身高。');
      return;
    }
    let resized;
    try {
      resized = resizeState(state, next);
    } catch (error) {
      setText('poseStatus', '部分尺线在该身高下无法复核，已保留当前身高和记录。');
      console.warn(error);
      return;
    }
    displayState(resized);
    refreshWorkbench();
    if (changed) {
      const record = state.records.get(activeTape?.key);
      showTape(record ? tapeFromRecord(record, displayedPose) : null);
    }
    persistWorkProgress();
    setText('poseStatus', `已按 ${next.toFixed(1)} cm 更新人台和尺寸。`);
  }

  function bodyGeometry(pose) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pose.vertices.flat()), 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(meshInfo.faces.flat()), 1));
    geometry.computeVertexNormals();
    geometry.computeBoundsTree({indirect: true});
    geometry.computeBoundingBox();
    return geometry;
  }
  function displayState(next) {
    const changed = next.selectedPose !== state.selectedPose || next.currentHeightCm !== state.currentHeightCm;
    state = next;
    renderContext();
    if (!changed) return;
    const direction = camera.position.clone().sub(controls.target);
    displayedPose = poseAtHeight(state.selectedPose, state.currentHeightCm);
    bodyMesh.geometry.disposeBoundsTree();
    bodyMesh.geometry.dispose();
    bodyMesh.geometry = bodyGeometry(displayedPose);
    frameCamera(direction);
  }
  function frameCamera(direction = null) {
    const box = bodyMesh.geometry.boundingBox;
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    box.getCenter(center);
    box.getSize(size);
    const targetY = center.y - size.y * 0.04;
    controls.target.set(center.x, targetY, center.z);
    const offset = new THREE.Vector3(size.x * 1.35, size.y * 0.16, size.y * 1.42);
    const horizontalTangent = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect;
    const fit = 1.06 * Math.max(offset.length(), Math.max(size.x, size.z) * 0.57 / Math.max(horizontalTangent, 0.05));
    camera.position.copy(controls.target).addScaledVector((direction || offset).normalize(), fit);
    camera.near = 1;
    camera.far = Math.max(6000, size.y * 5);
    camera.updateProjectionMatrix();
    controls.update();
    requestRender();
  }
  function nearestPose(angle) {
    const candidates = poses.filter(pose => pose && pose.kind === 'apose');
    return candidates.reduce((best, pose) => (
      Math.abs(Number(pose.arm_down_degrees) - angle) < Math.abs(Number(best.arm_down_degrees) - angle) ? pose : best
    ), candidates[0] || state.selectedPose);
  }
  function bindEvents() {
    const canvas = renderer.domElement;
    for (const event of ['pointerdown', 'pointermove', 'pointerup', 'pointercancel', 'lostpointercapture']) {
      canvas.addEventListener(event, handleTapePointer, {capture: true});
    }
    canvas.addEventListener('contextmenu', event => event.preventDefault());
    document.getElementById('saveTapeButton').addEventListener('click', saveCurrentTape);
    document.getElementById('remeasureButton').addEventListener('click', () => {
      const tape = activeTape;
      if (tape?.slot) startMeasurement(tape.slot);
    });
    document.getElementById('cancelTapeButton').addEventListener('click', cancelCurrentDraft);
    document.querySelectorAll('[data-open-dialog]').forEach(button => {
      button.addEventListener('click', () => openWorkbenchDialog(button.dataset.openDialog));
    });
    document.getElementById('nextMeasurementButton').addEventListener('click', () => {
      const next = nextMeasurementChoice();
      if (next) {
        rowNodes.get(next.slot.key).scrollIntoView({block: 'nearest'});
        selectMeasurement(next);
      } else {
        openWorkbenchDialog('resultsDialog');
      }
    });
    const viewDirections = {front: [0, 0, 1], side: [1, 0, 0], back: [0, 0, -1]};
    document.querySelectorAll('[data-body-view]').forEach(button => {
      button.addEventListener('click', () => {
        const vector = viewDirections[button.dataset.bodyView];
        frameCamera(vector ? new THREE.Vector3(...vector) : null);
      });
    });
    document.getElementById('angleSlider').addEventListener('input', event => {
      if (activeTape?.definition) {
        renderContext();
        setText('poseStatus', '请先确认或取消当前落尺，再调整站姿。');
        return;
      }
      displayState({...state, selectedPose: nearestPose(Number(event.target.value))});
      showTape(null);
      persistWorkProgress();
    });
    document.getElementById('applyHeightButton').addEventListener('click', applyHeightInput);
    document.getElementById('heightInput').addEventListener('keydown', event => {
      if (event.key === 'Enter') applyHeightInput();
    });
    for (const [id, suffix, exportFile, mime] of downloads) {
      document.getElementById(id).addEventListener('click', event => {
        event.preventDefault();
        downloadText(currentFileStem() + suffix, exportFile(), mime);
      });
    }
    document.getElementById('saveWorkButton').addEventListener('click', saveWorkProgress);
    document.getElementById('openWorkButton').addEventListener('click', () => {
      document.getElementById('workFileInput').click();
    });
    document.getElementById('workFileInput').addEventListener('change', event => {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      file.text().then(text => {
        importStateFromText(text);
        persistWorkProgress();
        setText('workProgressStatus', `已打开工作进度：${file.name}`);
      }).catch(error => {
        setText('workProgressStatus', `工作进度读取失败：${error.message || error}`);
      });
      event.target.value = '';
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && activeTape?.definition && !document.querySelector('dialog[open]')) cancelCurrentDraft();
    });
  }
  function openWorkbenchDialog(id) {
    const dialog = document.getElementById(id);
    if (id === 'poseControls') {
      renderContext();
      setText('poseStatus', state.heightIsEstimate ? '当前为暂定身高。填入实测值后点「应用身高」。' : '');
    }
    if (id === 'resultsDialog') renderMeasurementResults(publicRows);
    dialog.showModal();
  }
  function saveWorkProgress() {
    const candidate = workProgress(state);
    try {
      readWorkProgress(candidate);
    } catch (error) {
      setText('workProgressStatus', `工作进度保存失败：${error.message || error}`);
      return;
    }
    const filename = currentFileStem() + '-量体工作进度.bmwork';
    downloadText(filename, JSON.stringify(candidate, null, 2), 'application/json');
    setText('workProgressStatus', '已保存工作进度：' + filename);
  }
  function handleTapePointer(event) {
    const {type, pointerId, clientX, clientY} = event;
    if (type === 'pointerdown') {
      if (!activeTape?.definition || event.button !== 0) return;
    } else {
      if (pointerId !== gesture?.pointerId) return;
      claimTapePointerEvent(event);
    }
    if (type === 'pointercancel' || type === 'lostpointercapture') {
      const {status} = gesture;
      clearTapeGesture();
      showTape(activeTape, status);
      return;
    }

    const preview = type !== 'pointerup';
    const hit = surfaceHit(pickAt(clientX, clientY)) || (preview ? null : gesture.latestHit);
    if (!hit) return;
    if (type === 'pointerdown') {
      claimTapePointerEvent(event);
      renderer.domElement.setPointerCapture?.(pointerId);
      controls.enabled = false;
      gesture = {pointerId, startHit: hit, clientX, clientY, status: document.getElementById('tapeStatus').textContent};
    }
    const drag = Math.hypot(clientX - gesture.clientX, clientY - gesture.clientY) > 4;
    const {startHit} = gesture;
    gesture.latestHit = hit;
    if (!preview) clearTapeGesture();
    if (!activeTape?.definition) return;
    const twoHits = isCircumference(activeTape) ? activeTape.slot.preset.required_hit_ids?.length > 1 : drag;
    const tape = placeTape(activeTape, displayedPose, twoHits ? [startHit, hit] : [hit], state);
    if (preview) {
      gesture.preview = tape;
      renderTapeVisual(tape);
      setText('tapeStatus', tape.pathError && isCircumference(tape) ? '未成圈' : placementStatus(tape));
    } else {
      showTape(tape, placementStatus(tape));
    }
  }
  function placementStatus(tape) {
    const {item_id, value_mm, anchors} = tape.record;
    const name = tapeName(tape);
    if (tape.pathError) return `${name}暂不可确认：${tape.pathError}`;
    if (isCircumference(tape)) return `${name} ${formatCm(value_mm)}`;
    if (item_id === 'total_crotch' && !totalCrotchAdmitted) {
      return '待位置签署：请确认前腰、裆底、后腰';
    }
    return Number.isFinite(value_mm) ? `${name} ${formatCm(value_mm)} · ${anchors.length} 点`
      : `${name}已取 ${anchors.length}/${tape.definition.anchor_ids.length} 点`;
  }
  function clearTapeGesture() {
    const pointerId = gesture?.pointerId;
    gesture = null;
    if (pointerId != null) {
      try {
        renderer.domElement.releasePointerCapture?.(pointerId);
      } catch (_error) {
        // Pointer capture may already be gone after cancel/lostpointercapture.
      }
    }
    controls.enabled = true;
  }
  function showTape(tape, status = '') {
    activeTape = tape;
    renderTapeVisual(tape);
    updateSelection();
    setText('tapeStatus', status);
    updateSaveTapeButton();
  }
  function cancelCurrentDraft() {
    clearTapeGesture();
    const previous = state.records.get(activeTape?.key);
    if (!previous || !selectSavedTape(previous)) showTape(null, '已取消落尺');
  }
  function claimTapePointerEvent(event) {
    event.preventDefault();
    event.stopImmediatePropagation();
  }
  function saveCurrentTape() {
    let record;
    try {
      record = confirmTape(state, activeTape, displayedPose);
    } catch (error) {
      setText('tapeStatus', error.message || String(error));
      return;
    }
    if (!record) return;
    clearTapeGesture();
    refreshMeasurements();
    showTape(tapeFromRecord(record, displayedPose), '已保存尺线。');
    persistWorkProgress();
  }
  function updateSaveTapeButton() {
    const tape = activeTape;
    const {side, value_mm, review_status} = tape?.record || {};
    const editing = Boolean(tape?.definition);
    const confirming = !editing && review_status === 'automatic';
    const button = document.getElementById('saveTapeButton');
    button.disabled = editing ? !isTapeSavable(tape) : !confirming;
    button.hidden = !editing && !confirming;
    document.getElementById('remeasureButton').hidden = editing || !tape?.slot;
    document.getElementById('cancelTapeButton').hidden = !editing;
    const next = document.getElementById('nextMeasurementButton');
    next.hidden = editing;
    const choice = nextMeasurementChoice();
    next.textContent = choice ? '检查' + choice.slot.label : '尺寸与导出';
    const title = tape ? tape.slot?.preset.label || tapeName(tape) : '检查尺线';
    setText('inspectionHeading', side && side !== 'center' ? `${title} · ${sideLabel(side)}` : title);
    const review = tape?.slot && !tape.slot.public;
    setText('inspectionMode', review ? '来源复查' : editing ? '正在落尺' : tape ? tapeReviewLabel(review_status) : '');
    setText('inspectionValue', Number.isFinite(value_mm) ? formatCm(value_mm) : '');
  }
  function nextMeasurementChoice() {
    return navigationRows[navigationRows.findIndex(row => row.slot === activeTape?.slot) + 1];
  }
  function pickAt(clientX, clientY) {
    const rect = renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1,
      -(((clientY - rect.top) / Math.max(rect.height, 1)) * 2 - 1)
    );
    raycaster.setFromCamera(pointer, camera);
    return raycaster.intersectObject(bodyMesh, false)[0] || null;
  }
  function pickViewportSurface(xFraction, yFraction) {
    const rect = renderer.domElement.getBoundingClientRect();
    const hit = pickAt(rect.left + rect.width * xFraction, rect.top + rect.height * yFraction);
    return hit ? {
      hit: true,
      faceIndex: hit.faceIndex,
      point: hit.point.toArray().map(round3),
      bodySegment: meshInfo.face_body_segment?.[hit.faceIndex] || null,
    } : {hit: false};
  }
  function resize() {
    const {clientWidth, clientHeight} = container;
    const width = Math.max(1, clientWidth);
    const height = Math.max(1, clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    requestRender();
  }
  function requestRender() {
    if (!frameRequest) frameRequest = requestAnimationFrame(drawFrame);
  }
  function drawFrame() {
    frameRequest = 0;
    controls.update();
    renderer.render(scene, camera);
    positionTapeHud();
  }
  function refreshMeasurements() {
    const root = document.getElementById('measurementList');
    const focusedKey = document.activeElement?.dataset?.slot;
    const {publicRows: rows, reviewRows} = currentMeasurementRows(state);
    publicRows = rows;
    rowNodes.clear();
    const offset = measurementProjects.scrollTop;
    const groups = [
      ['躯干与裆部', rows.filter(row => row.slot.side === 'center')],
      ['腿部', rows.filter(row => row.slot.side !== 'center')],
    ];
    navigationRows = groups.flatMap(([, items]) => items);
    root.replaceChildren(...groups.filter(([, items]) => items.length).map(([label, items]) =>
      element('section', 'measurementGroup', element('h3', '', label), ...items.map(measurementChoice))
    ));
    setText('measurementCoverage', rows.length ? `${rows.filter(row => row.record).length} / ${rows.length} 项有尺寸` : '');
    if (!rows.length) root.textContent = '当前没有已获准量体项';
    if (hasPublicScope) {
      document.getElementById('sourceMeasurementList').replaceChildren(
        ...reviewRows.map(measurementChoice),
      );
    }
    measurementProjects.scrollTop = offset;
    rowNodes.get(focusedKey)?.focus({preventScroll: true});
    updateSelection();
    if (document.getElementById('resultsDialog').open) renderMeasurementResults(rows);
  }
  function updateSelection() {
    for (const [key, choice] of rowNodes) {
      const active = activeTape?.slot?.key === key;
      choice.dataset.active = String(active);
      choice.setAttribute('aria-pressed', String(active));
    }
  }
  function measurementChoice(item) {
    const {slot, record, valueCm} = item;
    const {key, label, side, preset} = slot;
    const name = element('span', 'measurementName', element('strong', '', label));
    if (side !== 'center') name.append(element('small', '', sideLabel(side)));
    const size = element('span', 'measurementCurrent');
    if (record) size.append(element('b', '', formatValueCm(valueCm)));
    const status = element('span', 'measurementStatus', record ? tapeReviewLabel(record.review_status) : '待量');
    status.dataset.review = record?.review_status || '';
    size.append(status);
    const choice = element('button', 'measurementRow', name, size);
    choice.type = 'button';
    Object.assign(choice.dataset, {
      slot: key, metricId: preset.metric_id, side,
      [slot.public ? 'publicItem' : 'reviewItem']: preset.item_id,
    });
    if (record) choice.dataset.recordId = record.id;
    rowNodes.set(key, choice);
    choice.addEventListener('click', () => selectMeasurement(item));
    return choice;
  }
  function selectMeasurement(row) {
    if (activeTape?.definition && activeTape.slot === row.slot) return;
    if (row.record) selectSavedTape(row.record);
    else startMeasurement(row.slot);
    if (compact.matches) {
      measurementProjects.open = false;
      if (!row.slot.public) document.getElementById('sourceReview').open = false;
      container.scrollIntoView({block: 'start'});
    }
  }
  function renderMeasurementResults(rows) {
    const root = document.getElementById('measurementResults');
    root.replaceChildren();
    if (!rows.length) {
      root.textContent = '当前没有已获准量体项';
      return;
    }
    const compare = referenceMeasurements.length > 0;
    if (personalCalibration) {
      const {device_id: device, device_model: model} = payload.source_identity || {};
      const deviceRole = [device, model].includes(personalCalibration.fit_device_id) ? '拟合设备' : '留出设备';
      const title = scaleDisplayLabel || '个人校准 · 仅当前人物 · 拟合设备/视频 '
        + (personalCalibration.fit_device_id || personalCalibration.fit_video_id);
      const note = element('p', 'calibrationNote', `${title} · 当前：${deviceRole} ${model || device || sourceDisplay.group || '本次来源'}`);
      note.dataset.personalCalibration = 'true';
      root.append(note);
    }
    const columns = personalCalibration ? ['量体项', '自动原值', '个人校准值'] : ['量体项', '当前'];
    if (compare) columns.push('人工', '误差');
    root.style.setProperty('--value-columns', String(columns.length - 1));
    const header = element('div', 'reportRow measurementHeader', ...columns.map(label => element('span', '', label)));
    header.setAttribute('role', 'row');
    for (const cell of header.children) cell.setAttribute('role', 'columnheader');
    root.append(header, ...rows.map(row => measurementResult(row, compare)));
  }
  function measurementResult(item, compare) {
    const {slot: {label, side, preset}, record, valueCm, calibratedCm} = item;
    const name = label + (side !== 'center' ? ' · ' + sideLabel(side) : '');
    const row = element('div', 'reportRow',
      element('span', 'measurementName', element('strong', '', name)),
      element('div', 'measurementCell measurementCurrent',
        element('b', '', formatValueCm(valueCm)),
        element('span', '', record ? tapeReviewLabel(record.review_status) : '尚未测量'),
      ),
    );
    row.setAttribute('role', 'row');
    Object.assign(row.dataset, {metricId: preset.metric_id, side});
    if (personalCalibration) row.append(element('div', 'measurementCell measurementCalibrated',
      element('b', '', formatValueCm(calibratedCm)),
    ));
    if (compare) {
      const {reference, error_cm, tolerance_cm, status} = compareReference(item);
      const unknown = reference?.side === 'unknown';
      const source = reference ? (unknown ? '工作簿单值' : '工作簿人工值') : '缺人工值';
      const referenceCell = element('div', 'measurementCell measurementReference',
        element('strong', '', formatValueCm(reference?.value_cm)), element('small', '', source),
      );
      if (reference) {
        referenceCell.title = [reference.source_sheet, reference.source_cell].filter(Boolean).join('!') || '工作簿';
        if (unknown) referenceCell.title += ' · 未分侧，左右共用这一读数';
      }
      const error = Number.isFinite(error_cm) ? (error_cm >= 0 ? '+' : '') + formatValueCm(error_cm) : '--';
      const detail = status === 'passed' || status === 'failed'
        ? `允许 ±${tolerance_cm.toFixed(1)} · ${status === 'passed' ? '通过' : '超限'}` : '不判定';
      const result = element('div', 'measurementCell measurementResult',
        element('strong', '', error), element('small', '', detail),
      );
      result.dataset.status = status;
      row.append(referenceCell, result);
    }
    for (const cell of row.children) cell.setAttribute('role', 'cell');
    return row;
  }
  function selectSavedTape(item) {
    const pose = poseForRecord(item);
    if (!pose) {
      setText('tapeStatus', '缺少原始姿态 ' + lowerBodyPoseLabel(item.lower_body_pose_id || item.pose_id));
      return false;
    }
    clearTapeGesture();
    if (pose !== state.selectedPose) displayState({...state, selectedPose: pose});
    showTape(tapeFromRecord(item, displayedPose));
    return true;
  }
  function refreshWorkbench() {
    refreshMeasurements();
    for (const [id, suffix] of downloads) document.getElementById(id).download = currentFileStem() + suffix;
  }
  function renderContext() {
    setText('angleValue', Number(state.selectedPose.arm_down_degrees || 0).toFixed(0));
    const heightInput = document.getElementById('heightInput');
    if (heightInput && document.activeElement !== heightInput) heightInput.value = state.currentHeightCm ? Number(state.currentHeightCm).toFixed(1) : '';
    const slider = document.getElementById('angleSlider');
    slider.min = String(payload.apose_angle_range?.min ?? 0);
    slider.max = String(payload.apose_angle_range?.max ?? 90);
    slider.value = String(state.selectedPose.arm_down_degrees || 0);
    renderIdentity();
  }
  function renderSourceReference() {
    const {source_frame, pose_preview} = payload.source_reference || {};
    const tiles = [source_frame || {}, pose_preview].filter(Boolean).map(reference => {
      const label = cleanDisplayText(reference.label) || '原视频参考';
      const path = cleanDisplayText(reference?.image_path);
      const visual = reference?.available && path
        ? Object.assign(element('img'), {src: path, alt: label})
        : Object.assign(element('span', '', '暂无参考图'), {
          title: cleanDisplayText(reference?.reason) || '参考图未生成',
        });
      return element('div', 'sourceTile', visual, element('strong', '', label));
    });
    document.getElementById('sourceStrip').replaceChildren(...tiles);
  }
  function renderIdentity() {
    setText('subjectTitle', subjectName());
    setText('subjectHeightLabel', scaleDisplayLabel ? '等效高度' : state.heightIsEstimate ? '暂定身高' : '身高');
    setText('heightInputLabel', scaleDisplayLabel ? '等效高度 cm' : '实测身高（cm）');
    setText('applyHeightButton', scaleDisplayLabel ? '应用高度' : '应用身高');
    setText('subjectHeight', state.currentHeightCm ? `${Number(state.currentHeightCm).toFixed(1)} cm` : (sourceDisplay.height_label || '--'));
    setText('subjectPose', currentPoseLabel(state.selectedPose));
    const source = document.getElementById('subjectSource');
    source.textContent = cleanDisplayText(sourceDisplay.group);
    source.hidden = !source.textContent;
  }
  function startMeasurement(slot) {
    clearTapeGesture();
    const {pose, tape, status} = startDraft(slot, state);
    if (pose) displayState({...state, selectedPose: pose});
    showTape(tape ?? null, status);
  }
  function sideLabel(side) {
    return ({center: '中线', left: '左侧', right: '右侧'})[side] || side;
  }
  function renderTapeVisual(tape) {
    for (const part of tapeGroup.children) {
      part.geometry.dispose();
      part.material.dispose();
    }
    tapeGroup.clear();
    requestRender();
    lengthLabelPoint = null;
    if (!(tape?.points?.length >= 2)) return;
    const closed = isCircumference(tape);
    const color = closed ? TAPE_COLORS.circumference : TAPE_COLORS.length;
    const points = closed ? [...tape.points, tape.points[0]] : tape.points;
    if (!closed) lengthLabelPoint = pathMidpoint(points);
    const geometry = new LineGeometry().setPositions(points.flat());
    const line = new Line2(geometry, new LineMaterial({color, linewidth: 3, depthWrite: false}));
    line.renderOrder = 1;
    tapeGroup.add(line);
    const anchors = tape.anchorPoints.slice();
    if (evidenceView) {
      const hidden = new Line2(geometry.clone(), new LineMaterial({
        color, linewidth: 2.5, transparent: true, opacity: 0.72, depthWrite: false, depthFunc: THREE.GreaterDepth,
        dashed: true, dashSize: 13, gapSize: 8,
      }));
      hidden.computeLineDistances();
      hidden.renderOrder = 2;
      tapeGroup.add(hidden);
      if (closed) {
        const {required_hits, seed} = tape.record;
        const hits = required_hits?.length ? required_hits : [seed].filter(Boolean);
        const vertices = displayedPose.vertices || [];
        anchors.push(...hits.map(hit => reconstructSurfaceHit(vertices, meshInfo.faces || [], hit)).filter(Boolean));
      }
    }
    const handles = !anchors.length && !closed ? [points[0], points.at(-1)] : anchors;
    handles.forEach((point, index) => {
      const handle = new THREE.Mesh(
        new THREE.SphereGeometry(6, 12, 8),
        new THREE.MeshBasicMaterial({
          color: evidenceView || index > 0 ? color : 0xffffff,
          depthTest: !evidenceView,
        })
      );
      if (evidenceView) handle.renderOrder = 3;
      handle.position.set(...point);
      tapeGroup.add(handle);
    });
  }
  function positionTapeHud() {
    const tape = gesture?.preview ?? activeTape;
    const hud = document.getElementById('tapeHud');
    const value = tape?.record.value_mm;
    if (!(tape?.points?.length >= 2) || !Number.isFinite(value)) {
      hud.style.display = 'none';
      return;
    }
    const projected = isCircumference(tape) ? ringLabelPosition(tape.points) : lengthLabelPoint.clone().project(camera);
    const canvas = renderer.domElement.getBoundingClientRect();
    const host = (hud.offsetParent || container).getBoundingClientRect();
    hud.style.left = (canvas.left - host.left + (projected.x * 0.5 + 0.5) * canvas.width) + 'px';
    hud.style.top = (canvas.top - host.top + (0.5 - projected.y * 0.5) * canvas.height) + 'px';
    hud.textContent = tapeName(tape) + (tape.definition ? ' ' + formatCm(value) : '');
    hud.style.display = Math.abs(projected.z) <= 1 ? 'block' : 'none';
  }
  function ringLabelPosition(points) {
    let visible, fallback;
    for (const point of points) {
      const position = new THREE.Vector3(...point).project(camera);
      if (!fallback || position.x > fallback.x) fallback = position;
      if (Math.abs(position.z) <= 1 && (!visible || position.x > visible.x)) visible = position;
    }
    return visible || fallback;
  }
  function pathMidpoint(points) {
    const vertices = points.map(point => new THREE.Vector3(...point));
    const path = new THREE.CurvePath();
    for (let index = 1; index < vertices.length; index += 1) {
      path.add(new THREE.LineCurve3(vertices[index - 1], vertices[index]));
    }
    return path.getLength() > 0 ? path.getPoint(0.5) : vertices[0];
  }
  function surfaceHit(hit) {
    const face = meshInfo.faces[hit?.faceIndex];
    if (!face) return null;
    const [a, b, c] = face.map(index => new THREE.Vector3(...displayedPose.vertices[index]));
    const bary = THREE.Triangle.getBarycoord(hit.point, a, b, c, new THREE.Vector3());
    if (!bary) return null;
    const weights = bary.toArray().map(value => Math.max(0, value));
    const total = weights.reduce((sum, value) => sum + value, 0);
    return total > 0 ? {face_index: hit.faceIndex, weights: weights.map(value => value / total)} : null;
  }
  function formatCm(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number / 10).toFixed(1)} cm` : '--';
  }
  function formatValueCm(value) {
    return Number.isFinite(value) ? value.toFixed(1) + ' cm' : '--';
  }
  function downloadText(filename, text, type) {
    const blob = new Blob([text], {type});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }
  function currentFileStem() {
    const height = Number(state.currentHeightCm).toFixed(0);
    return safeFilePart(`${subjectName()}-${state.heightIsEstimate ? '暂定' : ''}${height}cm`);
  }
  function subjectName() {
    return cleanDisplayText(sourceDisplay.subject) || cleanDisplayText(sourceDisplay.file_stem) || '量体结果';
  }
  function safeFilePart(value) {
    const cleaned = String(value || '').replace(/[\\/:*?"<>|\s]+/g, '_').replace(/^[._]+|[._]+$/g, '');
    return cleaned || '量体结果';
  }
  function importStateFromText(text) {
    const raw = String(text || '').trim();
    if (!raw) throw new Error('工作进度为空');
    const candidate = readWorkProgress(JSON.parse(raw));
    clearTapeGesture();
    displayState(candidate);
    setText('poseStatus', '');
    refreshWorkbench();
    const lastRecord = [...state.records.values()].findLast(isPublicRecord);
    if (lastRecord && poseForRecord(lastRecord) === state.selectedPose) selectSavedTape(lastRecord);
    else showTape(null);
  }
  function persistWorkProgress() {
    if (!progressKey) return;
    try {
      localStorage.setItem(progressKey, JSON.stringify(workProgress(state)));
      setText('autosaveStatus', '');
    } catch {
      setText('autosaveStatus', '自动保存不可用，请在「尺寸与导出」手动保存进度。');
    }
  }
  function restoreWorkProgress() {
    if (!progressKey) return null;
    try {
      const saved = localStorage.getItem(progressKey);
      return saved ? readWorkProgress(JSON.parse(saved)) : null;
    } catch {
      setText('autosaveStatus', '上次自动保存的进度无法恢复，可在「尺寸与导出」打开进度文件。');
      return null;
    }
  }
  function visualState() {
    const tape = activeTape;
    return {
      defaultView: {
        camera: {
          position: camera.position.toArray(),
          target: controls.target.toArray(),
        },
      },
      controlsEnabled: controls.enabled,
      activeTapeGesture: Boolean(gesture),
      pointerFace: gesture?.startHit.face_index ?? null,
      currentTape: tape ? {
        id: tape.record.id || null,
        placing: Boolean(tape.definition),
        poseId: state.selectedPose.id,
        pathFaceCount: tape.record.path_faces?.length || 0,
        pathError: tape.pathError || null,
        anchors: (tape.record.anchors || []).map(({hit}, index) => ({hit, point: tape.anchorPoints[index]})),
        valueMm: tape.record.value_mm ?? null,
      } : null,
    };
  }
  function setText(id, value) {
    document.getElementById(id).textContent = String(value);
  }
}
