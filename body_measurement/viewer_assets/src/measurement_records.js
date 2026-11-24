import {
  automaticLowerBodyDraft,
  distance,
  meshPlaneSection,
  legVisualPath,
  projectedLength,
  recomputeSurfaceTapeRecord,
  reconstructSurfaceHit,
  resolveInseamIntent,
  resolveTotalCrotchIntent,
  surfacePathHits,
  totalCrotchPlaneSection,
  torsoCircumference,
} from './surface_measurement.js';

export const SURFACE_EPSILON = 1e-6;
const WORKBENCH_TAPE_SCHEMA_ID = 'lower-body-8-v1';
const RECORD_DEFINITION_FIELDS = ['item_id', 'metric_id', 'definition_id', 'version', 'method', 'pose_id'];

export function segmentedSurfacePath({
  vertices, faces, anchors, definition, side,
  body_up, joints = null, joint_names = null, face_body_segment = null,
  body_height_mm = null, paired_outseam_mm = null,
}) {
  if (!definition || definition.method !== 'surface_path') {
    throw new Error('surface_path requires one complete definition');
  }
  if (!['path_length', 'body_up_projection'].includes(definition.value_rule)) {
    throw new Error('surface_path value_rule is invalid');
  }
  if (!Array.isArray(anchors) || !Array.isArray(definition.anchor_ids)
      || anchors.length !== definition.anchor_ids.length
      || anchors.some((anchor, index) => anchor?.id !== definition.anchor_ids[index])) {
    throw new Error('surface_path anchor IDs/count/order mismatch');
  }
  if (!Array.isArray(body_up) || body_up.length !== 3 || body_up.some(value => !Number.isFinite(value))) {
    throw new Error('surface_path body_up is invalid');
  }
  if (Math.hypot(...body_up) <= SURFACE_EPSILON) throw new Error('surface_path body_up must be non-zero');
  if (definition.item_id === 'total_crotch') {
    if (anchors.length !== 3 || definition.value_rule !== 'path_length') {
      throw new Error('total_crotch requires front, crotch, and back anchors and path length');
    }
    const path = totalCrotchPlaneSection({
      vertices, faces, front_hit: anchors[0].hit, crotch_hit: anchors[1].hit,
      back_hit: anchors[2].hit, body_up, face_body_segment, body_segments: definition.body_segments,
    });
    return {...path, path_length_mm: path.value_mm};
  }
  const {item_id} = definition;
  if (item_id !== 'outseam' && item_id !== 'inseam') {
    throw new Error('surface_path requires a workbook length item');
  }
  if (anchors.length !== 2 || definition.value_rule !== 'body_up_projection') {
    throw new Error(`${item_id} requires two anchors and body_up projection`);
  }
  const path = legVisualPath({
    item_id, vertices, faces, face_body_segment, side, body_up, joints, joint_names, anchors,
    body_segments: definition.body_segments,
  });
  const endpoints = [0, path.path_faces.length - 1].map(index => reconstructSurfaceHit(vertices, faces,
    {face_index: path.path_faces[index], weights: path.path_weights[index]}));
  const value_mm = projectedLength(endpoints, body_up);
  if (item_id === 'outseam' && !(value_mm > 0 && value_mm < body_height_mm)) {
    throw new Error('outseam must satisfy 0 < outseam < body_height');
  }
  if (item_id === 'inseam'
      && !(value_mm > 0 && value_mm < paired_outseam_mm && paired_outseam_mm < body_height_mm)) {
    throw new Error('inseam/outseam must satisfy 0 < inseam < outseam < body_height');
  }
  return {...path, value_mm, path_length_mm: path.value_mm};
}

function normalizeHitKind(kind) {
  return kind === 'subject_confirmed' ? 'subject-confirmed' : kind;
}

function validSurfaceHit(hit) {
  return Boolean(hit && Number.isInteger(hit.face_index) && Array.isArray(hit.weights)
    && hit.weights.length === 3 && hit.weights.every(Number.isFinite));
}

function requiredHitLabel(definition, index, id, anchor = null) {
  return anchor?.label || definition?.anchors?.[index]?.label || id;
}

export function definitionRequiredHits(definition) {
  const ids = definition?.required_hit_ids;
  if (ids == null) return [];
  if (!Array.isArray(ids) || !ids.length
      || ids.some(id => typeof id !== 'string' || !id)
      || new Set(ids).size !== ids.length) {
    throw new Error('量法定义的必经落点列表无效');
  }
  if (!Array.isArray(definition.anchors)) {
    const labels = ids.map((id, index) => requiredHitLabel(definition, index, id));
    throw new Error(`量法定义缺少必经落点“${labels.join(' / ')}”的 canonical SurfaceHit`);
  }
  return ids.map((id, index) => {
    const matches = definition.anchors.filter(anchor => anchor && typeof anchor === 'object' && anchor.id === id);
    const anchor = matches.length === 1 ? matches[0] : null;
    const hit = anchor?.surface_hit || anchor?.hit;
    const label = requiredHitLabel(definition, index, id, anchor);
    if (normalizeHitKind(anchor?.kind || anchor?.hit_kind) !== 'canonical' || !validSurfaceHit(hit)) {
      throw new Error(`量法定义缺少必经落点“${label}”的 canonical SurfaceHit`);
    }
    return hit;
  });
}

export function meshSectionFromDefinition({definition, ...geometry}) {
  try {
    return meshPlaneSection({
      ...geometry,
      allowed_faces: definition?.allowed_faces ?? geometry.allowed_faces,
      required_hits: geometry.required_hits ?? definitionRequiredHits(definition),
      required_hit_tolerance: geometry.required_hit_tolerance
        ?? (Number(definition?.tolerance_cm || 0) * 10 || undefined),
    });
  } catch (error) {
    const message = error?.message || String(error);
    const match = /^section required hit (\d+) (.+)$/.exec(message);
    if (!match) throw error;
    const index = Number(match[1]);
    const id = definition.required_hit_ids?.[index] || `manual_required_hit_${index + 1}`;
    const anchor = definition.anchors?.find(item => item?.id === id);
    const label = requiredHitLabel(definition, index, id, anchor);
    if (match[2] === 'is not on the measurement plane') {
      throw new Error(`“${label}”未与尺带截面保持同一水平`);
    }
    if (match[2] === 'is outside the segment/side filter') {
      throw new Error(`“${label}”不在量法声明的身体区域`);
    }
    throw new Error(`尺带未经过“${label}”所在的同一圈`);
  }
}

function measurementReferenceComparison(metricId, side, currentValueCm, references, definition) {
  const matches = (Array.isArray(references) ? references : [])
    .filter(reference => reference?.metric_id === metricId);
  const exact = matches.filter(reference => reference?.side === side);
  const unknown = matches.filter(reference => reference?.side === 'unknown');
  const reference = exact.length === 1 ? exact[0] : (exact.length === 0 && unknown.length === 1 ? unknown[0] : null);
  const referenceValueCm = reference?.value_cm;
  const toleranceCm = definition?.tolerance_cm;
  if (typeof referenceValueCm !== 'number' || !Number.isFinite(referenceValueCm)) {
    return {status: 'missing_reference', reference: null, error_cm: null, tolerance_cm: toleranceCm ?? null};
  }
  if (typeof currentValueCm !== 'number' || !Number.isFinite(currentValueCm)
      || typeof toleranceCm !== 'number' || !Number.isFinite(toleranceCm)) {
    return {status: 'not_evaluated', reference, error_cm: null, tolerance_cm: toleranceCm ?? null};
  }
  const errorCm = currentValueCm - referenceValueCm;
  return {
    status: Math.abs(errorCm) <= toleranceCm ? 'passed' : 'failed',
    reference,
    error_cm: errorCm,
    tolerance_cm: toleranceCm,
  };
}

export function personalCalibratedValueCm(metricId, automaticValueCm, calibration, subjectId) {
  if (!calibration) return null;
  if (calibration.subject_id !== subjectId) throw new Error('personal_calibration 只能用于相同人物');
  if (calibration.method !== 'additive_offset_cm' || calibration.applicability !== 'same_subject_only') {
    throw new Error('personal_calibration 方法或适用范围无效');
  }
  const offsetCm = calibration.offsets_cm?.[metricId];
  if (!Number.isFinite(automaticValueCm) || !Number.isFinite(offsetCm)) return null;
  return automaticValueCm + offsetCm;
}


export function scaledPose(pose, scale) {
  const factor = positiveNumber(scale) || 1;
  if (Math.abs(factor - 1) < 1e-9) return pose;
  return {
    ...pose,
    height_cm: Number(pose.height_cm || 0) * factor,
    vertices: (pose.vertices || []).map(vertex => vertex.map(value => round6(Number(value) * factor))),
    joints: pose.joints ? {
      ...pose.joints,
      points: (pose.joints.points || []).map(point => point.map(value => round6(Number(value) * factor))),
    } : pose.joints,
  };
}

function hitSpec(id, entry, kind) {
  const hit = entry?.surface_hit || entry?.hit || (Number.isInteger(entry?.face_index) ? entry : null);
  return {id, kind: normalizeHitKind(kind || entry?.kind || entry?.hit_kind || (hit ? 'canonical' : null)), hit};
}

function definitionAnchorSpec(definition, index) {
  const id = definition.anchor_ids[index];
  return hitSpec(id, definition.anchors?.find(item => item?.id === id) || definition.anchors?.[index]);
}

function definitionSeedSpec(definition) {
  const seed = definition.seed;
  return hitSpec(seed?.id || definition.seed_id || null, seed, definition.seed_kind);
}

function resolveSideValue(value, side) {
  return typeof value === 'string' ? value.replaceAll('{side}', side) : value;
}

function resolveDefinitionSide(definition, side) {
  if (!definition) return null;
  return {
    ...definition,
    body_segments: definition.body_segments?.map(segment => resolveSideValue(segment, side)),
    anchor_ids: definition.anchor_ids?.map(id => resolveSideValue(id, side)),
    anchors: definition.anchors?.map(anchor => ({...anchor, id: resolveSideValue(anchor?.id, side)})),
    seed_id: resolveSideValue(definition.seed_id, side),
    seed: definition.seed ? {...definition.seed, id: resolveSideValue(definition.seed.id, side)} : definition.seed,
  };
}

function surfaceHitsEqual(left, right) {
  return validSurfaceHit(left) && validSurfaceHit(right)
    && left.face_index === right.face_index
    && left.weights.every((value, index) => Math.abs(value - right.weights[index]) <= SURFACE_EPSILON);
}

export function lowerBodyPoseLabel(id) {
  switch (id) {
    case 'natural_standing': return '自然站立';
    case 'legs_together': return '双腿并拢';
    case 'legs_shoulder_width': return '双脚与肩同宽';
    default: return id || '';
  }
}

export function currentPoseLabel(pose) {
  const armAngle = Number(pose.arm_down_degrees || 0).toFixed(0);
  const upper = pose.kind === 'tpose' ? 'T字站姿' : `A字站姿 ${armAngle}度`;
  return pose.lower_body_pose_id && pose.lower_body_pose_id !== 'natural_standing'
    ? `${upper} · ${lowerBodyPoseLabel(pose.lower_body_pose_id)}`
    : upper;
}

export function positiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

export function round3(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return number;
  return Math.sign(number) * Math.round((Math.abs(number) + 1e-9) * 1000) / 1000;
}

export function round6(value) {
  return Number(Number(value).toFixed(6));
}

export function cleanDisplayText(value) {
  return String(value || '').trim();
}

function csvCell(value) {
  const text = String(value ?? '');
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function csvNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '';
}

export function tapeReviewLabel(reviewStatus) {
  if (reviewStatus === 'automatic') return '自动初值';
  if (reviewStatus === 'confirmed') return '人工确认';
  return '已记录';
}

export function createMeasurementRecords(payload) {
  const catalog = payload.measurement_definitions || {};
  const definitions = (catalog.definitions || []).map(withStatus);
  const definitionsByItem = new Map(definitions.map(definition => [definition.item_id, definition]));
  const definitionVersions = new Map([...definitions, ...(catalog.historical_definitions || []).map(withStatus)]
    .map(definition => [JSON.stringify([definition.item_id, definition.version]), definition]));
  const workbook = definitions.filter(definition => definition.preset_schema_id === WORKBENCH_TAPE_SCHEMA_ID);
  const publicItems = Array.isArray(payload.public_item_ids) ? new Set(payload.public_item_ids) : null;
  const measurementPresets = Object.fromEntries([
    ...definitions.filter(item => item.section_placement),
    ...workbook.filter(item => !item.section_placement),
  ].map(item => [item.metric_id, item]));
  const slots = new Map(Object.values(measurementPresets).flatMap(preset => preset.sides.map(side => {
    const key = measurementSlot(preset.item_id, side);
    return [key, {key, preset, side, label: preset.label || preset.metric_id || '量体项',
      public: !publicItems || publicItems.has(preset.item_id)}];
  })));
  const totalCrotchAdmitted = isSignedCrotch(definitionsByItem.get('total_crotch'));
  const schema = payload.viewer_state_schema?.surface_tape_record;
  const poses = payload.poses || [];
  const meshInfo = payload.mesh || {};
  const jointSchemas = payload.joint_schemas || {};
  const landmarkSet = Object.values(payload.measurement_landmark_sets || {})
    .find(item => item?.mesh_id === meshInfo.mesh_id && item?.hits) || null;
  const referenceMeasurements = payload.reference_measurements || [];
  const personalCalibration = payload.personal_calibration || null;
  const scaleDisplayLabel = cleanDisplayText(personalCalibration?.scale_display_label);
  const subjectId = payload.source_identity?.subject_id || payload.source_display?.subject;
  if (personalCalibration) personalCalibratedValueCm('', 0, personalCalibration, subjectId);
  const {height_cm: builtHeightCm, ...sourceIdentity} = payload.source_identity || {};
  const initialPose = pickInitialPose();
  const referenceHeightCm = positiveNumber(builtHeightCm) || positiveNumber(initialPose.height_cm) || 0;
  const projections = new Map();
  let projectedHeight;

  return {
    initialState, poseAtHeight, startDraft, placeTape, tapeFromRecord, confirmTape, resizeState,
    isTapeSavable, totalCrotchAdmitted, poseForRecord,
    currentMeasurementRows, compareReference, isPublicRecord, measurementCsv, readWorkProgress, workProgress,
  };

  function initialState(restored = null) {
    const state = restored || {
      currentHeightCm: referenceHeightCm, heightIsEstimate: payload.height_is_estimate === true,
      selectedPose: initialPose, records: new Map(),
    };
    let pose = state.selectedPose;
    for (const slot of slots.values()) {
      if (slot.preset.auto_place === false || state.records.has(slot.key)) continue;
      const {pose: next, tape} = prepareDraft(slot, pose, state);
      if (!tape) continue;
      pose = next;
      confirmTape(state, tape, poseAtHeight(pose, state.currentHeightCm), 'automatic');
    }
    return state;
  }

  function prepareDraft(slot, selectedPose, state) {
    const {preset, side} = slot;
    const {pose, definition, missing} = lowerBodyAvailability(preset, side, selectedPose);
    if (missing.length) return {missing};
    try {
      validateDefinition(definition, side);
    } catch (error) {
      return {missing: [error.message || String(error)]};
    }
    return {pose, tape: automaticDraft(slot, definition, poseAtHeight(pose, state.currentHeightCm), state)};
  }

  function startDraft(slot, state) {
    const {preset} = slot;
    const {pose, tape, missing} = prepareDraft(slot, state.selectedPose, state);
    const prompts = (preset.anchors?.map(anchor => anchor.label) || [preset.seed?.label]).join(' / ');
    const status = missing
      ? preset.label + '暂不可确认：' + missing.join('；') + '。所需落点：' + prompts
      : isTapeSavable(tape) ? '检查自动落点，也可在人台上重新落尺。' : '请在 ' + prompts + ' 落尺。' + preset.measurement_rule;
    return {pose, tape, status};
  }

  function pickInitialPose() {
    return poses.find(pose => pose && pose.id === 'selected_apose')
      || poses.find(pose => pose && pose.kind === 'apose')
      || poses[0]
      || {id: '', vertex_key: '', arm_down_degrees: 0, vertices: []};
  }

  function poseGeometry(pose) {
    return {
      vertices: pose.vertices || [], faces: meshInfo.faces || [],
      face_body_segment: meshInfo.face_body_segment || [], body_up: pose.body_up,
      joints: pose.joints?.points, joint_names: jointSchemas[pose.joints?.schema_id]?.names,
      landmark_hits: landmarkSet?.hits || {},
    };
  }

  function measureSection(definition, pose, seed, requiredHits) {
    const geometry = poseGeometry(pose);
    const {vertices, faces, body_up} = geometry;
    if (definition.section_placement) {
      if (pose.arm_down_degrees !== definition.arm_down_degrees
          || pose.lower_body_pose_id !== definition.lower_body_pose_id) {
        throw new Error('请使用该量项的参考姿势');
      }
      const result = torsoCircumference({...geometry, definition, seed});
      seed = result.seed || seed;
      return {...result, seed, required_hits: requiredHits, plane_normal: body_up,
        plane_origin: reconstructSurfaceHit(vertices, faces, seed)};
    }
    const plane = {plane_origin: reconstructSurfaceHit(vertices, faces, seed), plane_normal: body_up};
    const result = meshSectionFromDefinition({vertices, faces, definition, seed, required_hits: requiredHits, ...plane});
    return {...result, seed, required_hits: requiredHits, ...plane};
  }

  function newDraft(slot, definition, pose, state) {
    const {preset, side, key} = slot;
    const previous = state.records.get(key);
    const record = {
      schema_version: 1,
      id: (previous && editableSlot(previous) && previous.id) || `auto-${preset.item_id}-${side}`,
      item_id: definition.item_id,
      metric_id: definition.metric_id,
      definition_id: definition.definition_id,
      version: definition.version,
      definition_status: definition.definition_status || '',
      subject_id: pose.subject_id,
      name: preset.label,
      side,
      method: definition.method,
      units: 'mm',
      mesh_id: pose.mesh_id,
      pose_id: pose.pose_id,
      path_faces: undefined, path_weights: undefined, value_mm: undefined,
    };
    if (definition.preset_schema_id) record.preset_schema_id = definition.preset_schema_id;
    record.lower_body_pose_id = definition.lower_body_pose_id;
    if (record.method === 'mesh_section') {
      const seedSpec = definitionSeedSpec(definition);
      const segments = definition.body_segments.filter(segment =>
        side === 'center' || ['torso', 'pelvis'].includes(segment) || segment.startsWith(`${side}_`));
      Object.assign(record, {
        seed: undefined, seed_id: seedSpec.id || undefined, seed_kind: seedSpec.kind,
        body_segment: segments.length ? segments : definition.body_segments,
        plane_origin: undefined, plane_normal: undefined,
      });
    } else {
      record.anchors = [];
    }
    return {record, definition, slot, key, points: [], anchorPoints: []};
  }

  function tapeWithSolution(tape, pose, solution) {
    const {record} = tape;
    record.path_faces = solution.path_faces;
    record.path_weights = solution.path_weights;
    record.value_mm = solution.value_mm;
    if (record.method === 'mesh_section') {
      record.seed = copyHit(solution.seed);
      record.plane_origin = solution.plane_origin.slice();
      record.plane_normal = solution.plane_normal.slice();
      if (solution.required_hits?.length) record.required_hits = solution.required_hits.map(copyHit);
      else delete record.required_hits;
    }
    return {
      ...tape, points: surfacePathPointsForPose(record, pose), pathError: null,
      pathLengthMm: solution.path_length_mm, snapped: Boolean(solution.resolved_crotch_hit),
    };
  }

  function confirmTape({currentHeightCm, records}, tape, pose, review_status = 'confirmed') {
    if (!tape) return null;
    const {key, definition} = tape;
    let record = records.get(key);
    if (definition) {
      if (!isTapeSavable(tape)) return null;
      const valueMm = tape.record.value_mm;
      record = {...tape.record, value_mm: round6(valueMm)};
      // A snapped crotch anchor changes the input to the path solver.
      if (tape.snapped) validateRecordGeometry(record, definition, pose, currentHeightCm * 10, records);
      else recomputeTapeValue(record, definition, pose, tape.pathLengthMm, valueMm);
    } else if (record?.review_status !== 'automatic') return null;
    const confirmed = {...record, review_status};
    records.delete(key);
    records.set(key, confirmed);
    return confirmed;
  }

  function isTapeSavable(tape) {
    return Boolean(Number.isFinite(tape?.record.value_mm) && !tape.pathError
      && (tape.record.item_id !== 'total_crotch' || totalCrotchAdmitted));
  }

  function isSignedCrotch(definition) {
    const ids = definition?.anchor_ids || [];
    return definition?.definition_status === 'admitted' && definition.signoff != null
      && ids.length === 3 && Array.isArray(definition.anchors) && definition.anchors.length === ids.length
      && ids.every(id => {
        const matches = definition.anchors.filter(anchor => anchor?.id === id);
        return matches.length === 1 && validSurfaceHit(matches[0].surface_hit);
      });
  }

  function poseForRecord(record) {
    return poses.find(candidate => (candidate?.pose_id === record?.pose_id || candidate?.id === record?.pose_id)
      && (!record?.lower_body_pose_id || candidate?.lower_body_pose_id === record.lower_body_pose_id));
  }

  function lowerBodyAvailability(preset, side, selectedPose) {
    const matches = pose => pose?.lower_body_pose_id === preset.lower_body_pose_id
      && (!preset.section_placement || pose.arm_down_degrees === preset.arm_down_degrees);
    const pose = (matches(selectedPose) ? selectedPose : null) || poses.find(matches) || null;
    const definition = interactiveDefinition(preset, side, pose);
    const missing = [];
    if (!definition) missing.push('缺少当前侧别的量法定义');
    if (!pose) missing.push(`缺少姿态 ${lowerBodyPoseLabel(preset.lower_body_pose_id)}`);
    if (!meshInfo.mesh_id) missing.push('缺少 mesh_id');
    if (!pose?.mesh_id || pose.mesh_id !== meshInfo.mesh_id) missing.push('PosePayload mesh_id 不一致');
    if (!pose?.pose_id) missing.push('PosePayload 缺少 pose_id');
    if (!Array.isArray(pose?.body_up) || pose.body_up.length !== 3
        || pose.body_up.some(value => !Number.isFinite(value)) || Math.hypot(...pose.body_up) <= SURFACE_EPSILON) missing.push('缺少 body_up');
    const jointNames = jointSchemas[pose?.joints?.schema_id]?.names;
    if (!Array.isArray(pose?.joints?.points) || !Array.isArray(jointNames)
        || pose.joints.points.length !== jointNames.length) missing.push('缺少关节数据');
    return {pose, definition, missing};
  }

  function interactiveDefinition(raw, side, pose) {
    if ((!raw?.section_placement && raw?.preset_schema_id !== WORKBENCH_TAPE_SCHEMA_ID)
        || !raw?.sides?.includes(side)) return null;
    const definition = resolveDefinitionSide(raw, side);
    const segments = new Set(definition.body_segments);
    const anchor_ids = definition.anchor_ids?.length
      ? definition.anchor_ids
      : (definition.required_hit_ids || []).map(id => resolveSideValue(id, side));
    const labels = new Map(definition.anchors?.map(anchor => [anchor.id, anchor.label]));
    const confirmation = {kind: 'subject-confirmed', subject_confirmation_rule: definition.measurement_rule};
    return {
      ...definition,
      pose_id: pose?.pose_id || definition.pose_id,
      allowed_faces: definition.method === 'mesh_section' && !definition.section_placement
        ? index => segments.has(meshInfo.face_body_segment?.[index]) : undefined,
      anchor_ids,
      anchors: anchor_ids.map(id => ({id, label: labels.get(id) ?? null, ...confirmation})),
      required_hit_ids: null,
      ...(definition.method === 'mesh_section' &&
        {seed: {id: definition.metric_id + '_manual_seed', ...confirmation}}),
    };
  }

  function withStatus(definition) {
    return {...definition, definition_status: definition.definition_status || catalog.definition_status};
  }

  function validateDefinition(definition, side) {
    if (!definition || typeof definition !== 'object') throw new Error('缺少 definition');
    for (const field of RECORD_DEFINITION_FIELDS) {
      if (typeof definition[field] !== 'string' || !definition[field]) throw new Error(`definition 缺少 ${field}`);
    }
    if (!Array.isArray(definition.sides) || !definition.sides.includes(side)) throw new Error('definition side 不匹配');
    if (!poses.some(pose => pose?.pose_id === definition.pose_id || pose?.id === definition.pose_id)) {
      throw new Error('definition pose_id 不存在');
    }
    if (!Array.isArray(definition.body_segments) || !definition.body_segments.length) {
      throw new Error('definition 缺少 body_segments');
    }

    const closed = definition.method === 'mesh_section';
    if (!closed) {
      const ids = definition.anchor_ids;
      if (!['surface_path', 'straight'].includes(definition.method)) throw new Error('definition method 无效');
      if (!Array.isArray(ids) || ids.length < 2) throw new Error('definition 缺少 anchor_ids');
      if (definition.method === 'straight' && ids.length !== 2) throw new Error('straight 必须恰好两个 anchor');
      if (!Array.isArray(definition.anchors) || definition.anchors.length !== ids.length) throw new Error('definition 缺少 anchors');
    }
    const specs = closed ? [definitionSeedSpec(definition)]
      : definition.anchor_ids.map((id, index) => definitionAnchorSpec(definition, index));
    specs.forEach(({id, kind, hit}, index) => {
      const role = closed ? 'seed' : `anchor ${id}`;
      if (!['canonical', 'subject-confirmed'].includes(kind)) throw new Error(`definition ${role} 身份不完整`);
      if (kind === 'canonical' && ((closed && !id) || !validSurfaceHit(hit))) {
        throw new Error(`definition ${role} 缺少 canonical SurfaceHit`);
      }
      const entry = closed ? definition.seed : definition.anchors[index];
      if (kind === 'subject-confirmed' && !entry?.subject_confirmation_rule) {
        throw new Error(`definition ${role} 缺少 subject confirmation rule`);
      }
    });
    if (closed) definitionRequiredHits(definition);
    if (definition.method === 'surface_path' && !['path_length', 'body_up_projection'].includes(definition.value_rule)) {
      throw new Error('definition value_rule 无效');
    }
    return specs;
  }

  function validateRecordDefinition(record, definition, pose) {
    const specs = validateDefinition(definition, record.side);
    for (const field of RECORD_DEFINITION_FIELDS) {
      if (record[field] !== definition[field]) throw new Error(`record ${field} 与 definition 不一致`);
    }
    const closed = record.method === 'mesh_section';
    const hits = closed ? [{id: record.seed_id, hit_kind: record.seed_kind, hit: record.seed}] : record.anchors;
    if (!Array.isArray(hits) || hits.length !== specs.length) throw new Error('record anchors 数量不一致');
    specs.forEach((spec, index) => {
      const {id, hit_kind, hit} = hits[index];
      if (closed && id && spec.id && id !== spec.id) throw new Error('record seed_id 与 definition 不一致');
      if (!closed && id !== spec.id) throw new Error('record anchor IDs/count/order 不一致');
      const kind = hit_kind || spec.kind;
      if (kind !== spec.kind) throw new Error('record hit 身份与 definition 不一致');
      if (kind === 'canonical' && !surfaceHitsEqual(hit, spec.hit)) throw new Error('canonical SurfaceHit 与 definition 不一致');
      if (kind === 'subject-confirmed') {
        if (!validSurfaceHit(hit)) throw new Error(`${kind} hit 无效`);
        if (record.subject_id !== pose.subject_id) throw new Error(`${kind} hit subject 身份不一致`);
      }
    });
  }

  function validateRecordGeometry(record, definition, pose, bodyHeightMm, records) {
    if (!(Number.isFinite(bodyHeightMm) && bodyHeightMm > 0)) throw new Error('body height must be positive');
    const rebuilt = rebuildRecordGeometry(record, definition, pose, bodyHeightMm, records);
    if (record.method === 'mesh_section'
        && (!vectorsEqual(record.plane_origin, rebuilt.plane_origin) || !vectorsEqual(record.plane_normal, pose.body_up))) {
      throw new Error('mesh_section plane 必须由 seed 与 pose.body_up 决定');
    }
    assertRebuiltPath(record, rebuilt);
    return recomputeTapeValue(record, definition, pose, rebuilt.path_length_mm, rebuilt.value_mm);
  }

  function recomputeTapeValue(record, definition, pose, pathLengthMm, valueMm) {
    const tolerance = schema.value_tolerance_mm ?? 0.001;
    const projection = record.method === 'surface_path' && definition.value_rule === 'body_up_projection'
      ? {axis: pose.body_up, pathLengthMm} : null;
    const measured = recomputeSurfaceTapeRecord(record, pose.vertices, meshInfo.faces, tolerance, projection);
    if (Math.abs(valueMm - record.value_mm) > tolerance) {
      throw new Error(`SurfaceTapeRecord definition value mismatch: ${valueMm} != ${record.value_mm}`);
    }
    return measured;
  }

  function rebuildRecordGeometry(record, definition, pose, bodyHeightMm, records) {
    if (definition.method === 'mesh_section') {
      return measureSection(definition, pose, record.seed, record.required_hits);
    }
    const geometry = poseGeometry(pose);
    if (definition.method === 'surface_path') {
      const outseam = records.get(measurementSlot('outseam', record.side));
      return segmentedSurfacePath({
        ...geometry, definition, anchors: record.anchors, side: record.side,
        body_height_mm: bodyHeightMm, paired_outseam_mm: outseam?.value_mm,
      });
    }
    const hits = record.anchors.map(anchor => anchor.hit);
    const [a, b] = hits.map(hit => reconstructSurfaceHit(geometry.vertices, geometry.faces, hit));
    return {path_faces: hits.map(hit => hit.face_index), path_weights: hits.map(hit => hit.weights.slice()),
      value_mm: distance(a, b)};
  }

  function assertRebuiltPath(record, rebuilt) {
    if (rebuilt.path_faces.length !== record.path_faces.length
        || rebuilt.path_faces.some((face, index) => face !== record.path_faces[index])
        || rebuilt.path_weights.some((weights, index) => weights.some((weight, axis) =>
          Math.abs(weight - record.path_weights[index][axis]) > SURFACE_EPSILON))) {
      throw new Error('SurfaceTapeRecord faces/weights 与 definition 重建结果不一致');
    }
  }

  function vectorsEqual(left, right) {
    return Array.isArray(left) && Array.isArray(right) && left.length === right.length
      && left.every((value, index) => Number.isFinite(value)
        && Number.isFinite(right[index]) && Math.abs(value - right[index]) <= SURFACE_EPSILON);
  }

  function definitionForRecord(record) {
    const definition = definitionVersions.get(JSON.stringify([record.item_id, record.version]));
    return editableSlot(record)
      ? interactiveDefinition(definition, record.side, poseForRecord(record))
      : resolveDefinitionSide(definition, record.side);
  }

  function editableSlot(record) {
    const slot = slots.get(measurementSlot(record.item_id, record.side));
    return slot && slot.preset.metric_id === record.metric_id
      && (record.preset_schema_id === WORKBENCH_TAPE_SCHEMA_ID || slot.preset.section_placement) ? slot : null;
  }

  function copyHit({face_index, weights}) {
    return {face_index, weights: weights.slice()};
  }

  function surfacePoint(hit, pose) {
    return reconstructSurfaceHit(pose.vertices, meshInfo.faces, hit).map(round6);
  }

  function draftAnchor(id, hit, label = id) {
    return {id, label, hit_kind: 'subject-confirmed', hit: copyHit(hit)};
  }

  function placeTape(tape, pose, hits, state) {
    const record = {...tape.record};
    const draft = {...tape, record};
    const {definition, slot} = draft;
    try {
      if (record.method === 'mesh_section') {
        let seed = hits.at(-1);
        const twoHits = slot.preset.required_hit_ids?.length > 1;
        const required = twoHits ? hits : record.required_hits;
        if (twoHits) {
          const heights = hits.map(hit => surfacePoint(hit, pose)
            .reduce((sum, value, axis) => sum + value * pose.body_up[axis], 0));
          const difference = Math.abs(heights[0] - heights[1]);
          const tolerance = Number(definition.tolerance_cm);
          if (Number.isFinite(tolerance) && difference > tolerance * 10) {
            throw new Error(`两侧落点高度差 ${(difference / 10).toFixed(1)} cm，请在 ±${tolerance} cm 内重新落尺`);
          }
          seed = heights[0] <= heights[1] ? hits[0] : hits[1];
        }
        return tapeWithSolution(draft, pose, measureSection(definition, pose, seed, required));
      }
      const expected = definition.anchor_ids.length;
      const anchors = record.anchors = [...record.anchors];
      const points = draft.anchorPoints = [...tape.anchorPoints];
      for (const hit of hits) {
        const point = surfacePoint(hit, pose);
        const last = points.at(-1);
        const replaceLast = last && distance(last, point) <= SURFACE_EPSILON;
        const index = Math.min(expected - 1, anchors.length - Number(Boolean(replaceLast)));
        anchors[index] = draftAnchor(definition.anchor_ids[index], hit, anchors[index]?.label);
        points[index] = point;
      }
      if (anchors.length === expected) {
        record.anchors = resolveLengthAnchors(record, pose);
        return solveLengthTape(draft, pose, state);
      }
      return {...draft, points, pathError: null};
    } catch (error) {
      Object.assign(record, {path_faces: undefined, path_weights: undefined, value_mm: undefined});
      delete record.required_hits;
      return {...draft, points: [], pathError: error.message || String(error)};
    }
  }

  function resolveLengthAnchors({anchors, item_id}, pose) {
    let resolved;
    if (item_id === 'inseam') {
      resolved = resolveInseamIntent({anchors, landmark_hits: poseGeometry(pose).landmark_hits});
      if (!resolved) throw new Error('内档腿长缺少会阴或同侧内踝 SurfaceHit');
    } else if (item_id === 'total_crotch') {
      resolved = resolveTotalCrotchIntent({
        ...poseGeometry(pose), anchors, definition: definitionsByItem.get(item_id),
        waist_definition: definitionsByItem.get('iliac_waist'),
      });
      if (!resolved) throw new Error('总裆长缺少髂嵴环或裆底 SurfaceHit');
    } else {
      return anchors;
    }
    return resolved;
  }

  function solveLengthTape(tape, pose, {currentHeightCm, records}) {
    const {record, definition} = tape;
    tape.anchorPoints = record.anchors.map(anchor => surfacePoint(anchor.hit, pose));
    const result = rebuildRecordGeometry(record, definition, pose, currentHeightCm * 10, records);
    if (result.resolved_crotch_hit) {
      record.anchors[1] = {...record.anchors[1], hit: copyHit(result.resolved_crotch_hit)};
      tape.anchorPoints[1] = surfacePoint(result.resolved_crotch_hit, pose);
    }
    return tapeWithSolution(tape, pose, result);
  }

  function tapeFromRecord(record, pose) {
    return {
      record,
      key: measurementSlot(record.item_id, record.side),
      slot: editableSlot(record),
      points: surfacePathPointsForPose(record, pose),
      anchorPoints: record.anchors?.map(anchor => surfacePoint(anchor.hit, pose)) || [],
    };
  }

  function surfacePathPointsForPose(path, pose) {
    return surfacePathHits(path).map(hit => surfacePoint(hit, pose));
  }

  function scaleAtHeight(heightCm) {
    return referenceHeightCm > 0 && heightCm > 0 ? heightCm / referenceHeightCm : 1;
  }

  function poseAtHeight(pose, heightCm) {
    if (projectedHeight !== heightCm) {
      projections.clear();
      projectedHeight = heightCm;
    }
    if (!projections.has(pose)) projections.set(pose, scaledPose(pose, scaleAtHeight(heightCm)));
    return projections.get(pose);
  }

  function resizeState(state, currentHeightCm) {
    if (currentHeightCm === state.currentHeightCm) return {...state, heightIsEstimate: false};
    const records = new Map(state.records);
    const ordered = [...records].sort(([, left], [, right]) => (left.item_id === 'inseam') - (right.item_id === 'inseam'));
    for (const [key, record] of ordered) {
      const pose = poseAtHeight(poseForRecord(record), currentHeightCm);
      const rebuilt = rebuildRecordGeometry(record, definitionForRecord(record), pose, currentHeightCm * 10, records);
      records.set(key, {
        ...record,
        path_faces: rebuilt.path_faces,
        path_weights: rebuilt.path_weights,
        value_mm: round6(rebuilt.value_mm),
        ...(record.method === 'mesh_section' ? {plane_origin: rebuilt.plane_origin.map(round6)} : {}),
      });
    }
    return {...state, currentHeightCm, heightIsEstimate: false, records};
  }

  function isPublicRecord(record) {
    if (!publicItems) return true;
    if (!publicItems.has(record.item_id)) return false;
    const active = definitionsByItem.get(record.item_id);
    return record.definition_id === active?.definition_id && record.version === active?.version;
  }

  function measurementCsv(state) {
    const calibrated = personalCalibration && !publicItems;
    const rows = [calibrated
      ? ['姿态', '量体项', '量体项ID', '侧别', '自动原值cm', '个人校准值cm', '加放量cm', '成衣参考cm', '结果来源', scaleDisplayLabel ? '等效高度cm' : '当前身高cm']
      : ['姿态', '量体项', '量体项ID', '侧别', '净体cm', '加放量cm', '当前cm', '成衣参考cm', '结果来源', '当前身高cm']];
    rows[0].push('尺线状态', '身高状态');
    for (const {slot: {label, preset: {item_id}, side}, record, valueCm, calibratedCm}
      of currentMeasurementRows(state).publicRows) {
      const raw = csvNumber(valueCm);
      const values = calibrated ? [raw, csvNumber(calibratedCm), '', ''] : [raw, '', raw, ''];
      let source = '尚未测量';
      if (record) source = !calibrated ? '软尺量体' : (Number.isFinite(calibratedCm) ? '个人校准' : '自动原值');
      rows.push([
        record ? currentPoseLabel(poseForRecord(record)) : '', label, item_id, side, ...values, source,
        csvNumber(state.currentHeightCm), record ? tapeReviewLabel(record.review_status) : '待量',
        state.heightIsEstimate ? '暂定' : scaleDisplayLabel ? '等效' : '实测',
      ]);
    }
    return `\ufeff${rows.map(row => row.map(csvCell).join(',')).join('\r\n')}\r\n`;
  }

  function currentMeasurementRows(state) {
    const publicRows = [], reviewRows = [];
    for (const slot of slots.values()) {
      let record = state.records.get(slot.key) || null;
      if (record && slot.public && !isPublicRecord(record)) record = null;
      const valueCm = record ? round3(record.value_mm / 10) : null;
      const calibratedCm = personalCalibratedValueCm(slot.preset.metric_id, valueCm, personalCalibration, subjectId);
      (slot.public ? publicRows : reviewRows).push({slot, record, valueCm, calibratedCm});
    }
    return {publicRows, reviewRows};
  }

  function compareReference({slot: {preset: {metric_id}, side}, valueCm, calibratedCm}) {
    return measurementReferenceComparison(metric_id, side, calibratedCm ?? valueCm,
      referenceMeasurements, definitions.findLast(definition => definition.metric_id === metric_id));
  }

  function measurementSlot(itemId, side) {
    return JSON.stringify([itemId, side]);
  }

  function selectedPoseRecord(pose) {
    return {
      pose_id: pose.id || '',
      vertex_key: pose.vertex_key || '',
      arm_down_degrees: Number(pose.arm_down_degrees || 0),
      visual_proxy: false,
      angle_source: payload.viewer_state_schema?.angle_sources?.exported_mesh || 'exported_result_package_mesh',
    };
  }

  function workProgress(state) {
    return {
      state_kind: payload.viewer_state_schema?.state_kind || 'body_measurement_viewer_state',
      contract_version: payload.contract_version || 1,
      ...(payload.workbench_id
        ? {workbench_id: payload.workbench_id, selected_pose_id: state.selectedPose.id}
        : {package_dir: '.', source_identity: {...sourceIdentity, mesh_id: meshInfo.mesh_id},
          selected_pose: selectedPoseRecord(state.selectedPose)}),
      current_height_cm: state.currentHeightCm,
      height_is_estimate: state.heightIsEstimate,
      measurements: [...state.records.values()]
    };
  }

  function readWorkProgress(candidate, recordChecks = null) {
    if (!candidate || candidate.state_kind !== payload.viewer_state_schema.state_kind) {
      throw new Error('工作进度格式无效');
    }
    const version = candidate.contract_version;
    let selectedPose;
    if (version === 2) {
      if (!payload.workbench_id || candidate.workbench_id !== payload.workbench_id) {
        throw new Error('这份进度属于另一份工作台');
      }
      selectedPose = poses.find(pose => pose.id === candidate.selected_pose_id);
      if (!selectedPose) throw new Error('工作进度的姿态不存在');
    } else if (version === 1) {
      validateLegacyIdentity(candidate.source_identity);
      const saved = candidate.selected_pose;
      selectedPose = poses.find(pose => pose.id === saved?.pose_id);
      if (!selectedPose || saved.visual_proxy !== false || saved.vertex_key !== selectedPose.vertex_key
          || !Number.isFinite(saved.arm_down_degrees)
          || Math.abs(saved.arm_down_degrees - selectedPose.arm_down_degrees) > SURFACE_EPSILON
          || ('angle_source' in saved && typeof saved.angle_source !== 'string')
          || saved.angle_source === 'visual_preview_interpolated_mesh') {
        throw new Error('旧工作进度的姿态与人台不一致');
      }
    } else {
      throw new Error('工作进度版本不受支持');
    }

    const currentHeightCm = candidate.current_height_cm;
    if (!Number.isFinite(currentHeightCm) || currentHeightCm <= 0
        || (payload.height_is_estimate && (currentHeightCm < 120 || currentHeightCm > 230))) {
      throw new Error('工作进度的身高无效');
    }
    if ((version === 2 || 'height_is_estimate' in candidate) && typeof candidate.height_is_estimate !== 'boolean') {
      throw new Error('工作进度的身高状态无效');
    }
    const heightIsEstimate = candidate.height_is_estimate ?? (payload.height_is_estimate && currentHeightCm === referenceHeightCm);
    if (!Array.isArray(candidate.measurements)) throw new Error('工作进度缺少测量列表');
    const records = new Map();
    for (const record of candidate.measurements) {
      if (!record || typeof record !== 'object' || Array.isArray(record)
          || schema.required_fields.some(field => !(field in record))) {
        throw new Error('尺线记录字段不完整');
      }
      if (schema.retired_fields?.some(field => Object.hasOwn(record, field))) {
        throw new Error('尺线含有已停用的字段');
      }
      const key = measurementSlot(record.item_id, record.side);
      if (records.has(key)) throw new Error('工作进度存在重复的量体项和侧别');
      if (version === 1) {
        const {height_cm, ...normalized} = record;
        records.set(key, normalized);
      } else {
        records.set(key, record);
      }
    }
    for (const record of records.values()) {
      const pose = poseForRecord(record);
      if (!pose || record.mesh_id !== pose.mesh_id) throw new Error('尺线不属于当前人台姿态');
      if (record.schema_version !== schema.schema_version || !schema.sides.includes(record.side)
          || !schema.methods.includes(record.method) || record.units !== schema.units
          || typeof record.id !== 'string' || !record.id.trim()
          || !Number.isFinite(record.value_mm) || record.value_mm < 0) {
        throw new Error('尺线记录格式无效');
      }
      if (record.item_id === 'total_crotch' && !totalCrotchAdmitted) {
        throw new Error('总裆长度需要已签署的量法');
      }
      const definition = definitionForRecord(record);
      validateRecordDefinition(record, definition, pose);
      const value_mm = validateRecordGeometry(
        record, definition, poseAtHeight(pose, currentHeightCm), currentHeightCm * 10, records,
      );
      if (recordChecks) {
        const {mesh_id, pose_id, side, method, units, metric_id, name, path_faces} = record;
        recordChecks.push({
          name: name || metric_id, mesh_id, pose_id, side, method, units, value_mm,
          path_point_count: path_faces.length, visual_proxy: Boolean(pose.visual_proxy),
        });
      }
    }
    return {currentHeightCm, heightIsEstimate, selectedPose, records};
  }

  function validateLegacyIdentity(identity) {
    if (!identity || typeof identity !== 'object' || Array.isArray(identity)) {
      throw new Error('旧工作进度缺少来源身份');
    }
    const withMesh = Object.hasOwn(identity, 'mesh_id');
    const expected = withMesh ? {...sourceIdentity, mesh_id: meshInfo.mesh_id} : payload.source_identity;
    for (const [key, value] of Object.entries(expected)) {
      const actual = identity[key];
      if (['manifest_path', 'mesh_path', 'preview_path'].includes(key) && typeof actual === 'string') {
        const portable = actual.replaceAll('\\', '/').replace(/^\.\//, '');
        if (portable === value || (withMesh && portable.endsWith('/' + value))) continue;
      } else if (actual === value) {
        continue;
      }
      throw new Error('这份旧进度与当前人台的来源不一致');
    }
  }

  function automaticDraft(slot, definition, pose, state) {
    const {preset, side} = slot;
    const tape = newDraft(slot, definition, pose, state);
    if (preset.auto_place === false) return tape;
    try {
      if (!landmarkSet || landmarkSet.subject_id !== pose.subject_id) {
        throw new Error('automatic draft landmark subject does not match pose');
      }
      if (definition.section_placement) return tapeWithSolution(tape, pose, measureSection(definition, pose));
      const draft = automaticLowerBodyDraft({...poseGeometry(pose), definition: preset, side});
      if (!draft) return tape;
      if (tape.record.method === 'mesh_section') {
        return tapeWithSolution(tape, pose, measureSection(definition, pose, draft.seed, draft.required_hits || []));
      }
      const anchors = draft.anchors.map((hit, index) => {
        const id = definition.anchor_ids[index];
        return draftAnchor(id, hit, preset.anchors[index]?.label || id);
      });
      return solveLengthTape({...tape, record: {...tape.record, anchors}}, pose, state);
    } catch (error) {
      return {...tape, pathError: error.message || String(error)};
    }
  }
}
