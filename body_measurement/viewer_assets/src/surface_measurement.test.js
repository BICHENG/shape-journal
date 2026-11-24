const assert = require('node:assert/strict');
const fs = require('node:fs');

async function main() {
  const {
    automaticLowerBodyDraft,
    circumferenceFits,
    torsoCircumference,
    meshPlaneSection,
    recomputeSurfaceTapeRecord,
    reconstructSurfaceHit,
    resolveInseamIntent,
  } = await import('./surface_measurement.js');
  const {createMeasurementRecords, definitionRequiredHits, meshSectionFromDefinition,
    personalCalibratedValueCm, scaledPose, segmentedSurfacePath} =
    await import('./measurement_records.js');
  const definitions = JSON.parse(fs.readFileSync(__dirname + '/../../specs/measurement_definitions.json', 'utf8')).definitions;
  const definitionFor = id => definitions.find(definition => definition.item_id === id);

  const unscaled = {height_cm: 158, vertices: [[64, -32, 16]], joints: {points: [[8, -16, 24]]}, body_up: [0, 1, 0]};
  assert.deepEqual(scaledPose(unscaled, 180 / 160), {
    height_cm: 177.75, vertices: [[72, -36, 18]], joints: {points: [[9, -18, 27]]}, body_up: [0, 1, 0],
  });
  assert.deepEqual(unscaled.joints.points, [[8, -16, 24]]);


function prismLoops(polygons) {
  const vertices = [];
  const faces = [];
  const ranges = [];
  polygons.forEach(polygon => {
    const startVertex = vertices.length;
    const startFace = faces.length;
    polygon.forEach(([x, y]) => vertices.push([x, y, -1]));
    polygon.forEach(([x, y]) => vertices.push([x, y, 1]));
    for (let i = 0; i < polygon.length; i += 1) {
      const next = (i + 1) % polygon.length;
      const low = startVertex + i;
      const lowNext = startVertex + next;
      const high = startVertex + polygon.length + i;
      const highNext = startVertex + polygon.length + next;
      faces.push([low, lowNext, highNext], [low, highNext, high]);
    }
    ranges.push({startFace, faceCount: polygon.length * 2});
  });
  return {vertices, faces, ranges};
}

function section(mesh, faceIndex = 0, extra = {}) {
  return meshPlaneSection({
    vertices: mesh.vertices,
    faces: mesh.faces,
    plane_origin: [0, 0, 0],
    plane_normal: [0, 0, 1],
    seed: {face_index: faceIndex, weights: [0.5, 0, 0.5]},
    ...extra,
  });
}

function circumferenceRecord(mesh, faceIndex = 0, extra = {}) {
  const result = section(mesh, faceIndex, extra);
  return {
    method: 'mesh_section',
    units: 'mm',
    side: extra.side || 'center',
    body_segment: extra.body_segment,
    path_faces: result.path_faces,
    path_weights: result.path_weights,
    value_mm: result.value_mm,
  };
}

function rangeFaces(range) {
  return Array.from({length: range.faceCount}, (_, index) => range.startFace + index);
}

const lowerBodyJoints = (leftX = -1, rightX = 1) => ({
  joint_names: ['pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
  joints: [
    [(leftX + rightX) / 2, 6, 0], [leftX, 6, 0], [rightX, 6, 0],
    [leftX, 3, 0], [rightX, 3, 0], [leftX, 0, 0], [rightX, 0, 0],
  ],
});

// Two equally high cylinders remain separate; the SurfaceHit chooses the intended side.
const octagon = (cx) => Array.from({length: 8}, (_, i) => {
  const angle = 2 * Math.PI * i / 8;
  return [cx + 0.7 * Math.cos(angle), 0.7 * Math.sin(angle)];
});
const legs = prismLoops([octagon(-2), octagon(2)]);
assert.deepEqual(reconstructSurfaceHit(
  [[0, 0, 0], [3, 0, 0], [0, 3, 0]], [[0, 1, 2]],
  {face_index: 0, weights: [0.33333334, 0.33333334, 0.33333334]},
), [1, 1, 0]);
const legSegments = legs.faces.map((_, face) => face < legs.ranges[1].startFace ? 'left_thigh' : 'right_shin');
const legSides = legs.faces.map((_, face) => face < legs.ranges[1].startFace ? 'left' : 'right');
const leftLegFaces = rangeFaces(legs.ranges[0]);
const rightLegFaces = rangeFaces(legs.ranges[1]);
const left = section(legs, legs.ranges[0].startFace, {
  allowed_faces: leftLegFaces, face_body_segment: legSegments, body_segment: 'left_thigh', face_side: legSides, side: 'left',
});
const right = section(legs, legs.ranges[1].startFace, {
  allowed_faces: rightLegFaces, face_body_segment: legSegments, body_segment: 'right_shin', face_side: legSides, side: 'right',
});

// Package hip-crest SurfaceHits are the waist seed, required ring points, and outseam origin.
const waistMesh = prismLoops([octagon(0)]);
const waistSegments = waistMesh.faces.map(() => 'pelvis');
const waistSection = section(waistMesh);
const waistHits = waistSection.path_faces.map((face_index, index) => ({
  face_index, weights: waistSection.path_weights[index],
}));
const waistPoints = waistHits.map(hit => reconstructSurfaceHit(waistMesh.vertices, waistMesh.faces, hit));
const leftHipCrest = waistHits[waistPoints.findIndex(point => point[0] === Math.max(...waistPoints.map(item => item[0])))];
const rightHipCrest = waistHits[waistPoints.findIndex(point => point[0] === Math.min(...waistPoints.map(item => item[0])))];
const hipLandmarks = {left_hip_crest: leftHipCrest, right_hip_crest: rightHipCrest};
const waistDraft = automaticLowerBodyDraft({
  definition: definitionFor('iliac_waist'), side: 'center',
  vertices: waistMesh.vertices, faces: waistMesh.faces, face_body_segment: waistSegments,
  joint_names: ['pelvis', 'left_hip', 'right_hip', 'spine1'],
  joints: [[0, 0, -0.5], [0.2, 0, -0.5], [-0.2, 0, -0.5], [0, 0, 0.5]],
  body_up: [0, 0, 1],
  landmark_hits: hipLandmarks,
});
assert.deepEqual(waistDraft, {seed: leftHipCrest, required_hits: [leftHipCrest, rightHipCrest]});
const waistRing = meshPlaneSection({
  vertices: waistMesh.vertices, faces: waistMesh.faces,
  plane_origin: reconstructSurfaceHit(waistMesh.vertices, waistMesh.faces, waistDraft.seed),
  plane_normal: [0, 0, 1], seed: waistDraft.seed, required_hits: waistDraft.required_hits,
});
assert.ok(waistDraft.required_hits.every(hit => waistRing.path_faces.includes(hit.face_index)));
const waistExtrema = waistDraft.required_hits.map(hit => reconstructSurfaceHit(waistMesh.vertices, waistMesh.faces, hit));
assert.ok(Math.abs(waistExtrema[0][2] - waistExtrema[1][2]) < 1e-9);
const outseamDraft = automaticLowerBodyDraft({
  definition: definitionFor('outseam'), side: 'left',
  vertices: waistMesh.vertices, faces: waistMesh.faces, face_body_segment: waistSegments,
  ...lowerBodyJoints(), body_up: [0, 0, 1],
  landmark_hits: {...hipLandmarks, left_lateral_malleolus: waistHits[1]},
});
assert.equal(outseamDraft.anchors[0], leftHipCrest);
const totalCrotchDraft = automaticLowerBodyDraft({
  definition: definitionFor('total_crotch'), waist_definition: definitionFor('iliac_waist'), side: 'center',
  vertices: waistMesh.vertices, faces: waistMesh.faces, face_body_segment: waistSegments,
  ...lowerBodyJoints(), body_up: [0, 0, 1],
  landmark_hits: {...hipLandmarks, crotch: {...waistHits[2], unambiguous: true}},
});
assert.equal(totalCrotchDraft.anchors[1].face_index, waistHits[2].face_index);
assert.ok([totalCrotchDraft.anchors[0], totalCrotchDraft.anchors[2]].every(hit =>
  Math.abs(reconstructSurfaceHit(waistMesh.vertices, waistMesh.faces, hit)[2] - waistExtrema[0][2]) < 1e-9));

const inseamIntent = [
  {id: 'crotch', hit_kind: 'subject-confirmed', hit: waistHits[0]},
  {id: 'right_medial_malleolus', hit_kind: 'subject-confirmed', hit: waistHits[1]},
];
const resolvedInseam = resolveInseamIntent({
  anchors: inseamIntent,
  landmark_hits: {crotch: waistHits[2], right_medial_malleolus: waistHits[3]},
});
assert.deepEqual(resolvedInseam.map(anchor => anchor.hit), [waistHits[2], waistHits[3]]);
assert.deepEqual(inseamIntent.map(anchor => anchor.hit), [waistHits[0], waistHits[1]]);
assert.equal(resolveInseamIntent({anchors: inseamIntent, landmark_hits: {}}), null);

assert.ok(left.path_faces.every(face => face < legs.ranges[1].startFace));
assert.ok(right.path_faces.every(face => face >= legs.ranges[1].startFace));
assert.ok(Math.abs(left.value_mm - right.value_mm) < 1e-12);
const requiredHitMesh = {...legs, vertices: legs.vertices.map(([x, y, z]) => [x, y, z * 100])};
const coplanarRequiredHits = [0, 8].map(face_index => ({face_index, weights: [0.25, 0.25, 0.5]}));
const requiredHitDefinition = {
  method: 'mesh_section', metric_id: 'iliac_waist_circumference',
  tolerance_cm: 0.5,
  required_hit_ids: ['left_hip_crest', 'right_hip_crest'],
  anchors: [
    {...definitionFor('iliac_waist').anchors[0], surface_hit: coplanarRequiredHits[0]},
    {...definitionFor('iliac_waist').anchors[1], surface_hit: coplanarRequiredHits[1]},
  ],
};
const runtimeSection = definition => meshSectionFromDefinition({
  definition,
  vertices: requiredHitMesh.vertices,
  faces: requiredHitMesh.faces,
  plane_origin: [0, 0, 0],
  plane_normal: [0, 0, 1],
  seed: {face_index: 0, weights: [0.5, 0, 0.5]},
});
assert.ok(runtimeSection(requiredHitDefinition).value_mm > 0);
assert.throws(
  () => runtimeSection({...requiredHitDefinition, allowed_faces: [8]}),
  /seed face is outside the segment\/side filter/,
);
const nearbyHitMesh = {
  vertices: [...requiredHitMesh.vertices, [-1.28, 0.1, 0], [-1.28, 0.1, 1], [10, 0.2, 0.5]],
  faces: [...requiredHitMesh.faces, [requiredHitMesh.vertices.length, requiredHitMesh.vertices.length + 1, requiredHitMesh.vertices.length + 2]],
};
assert.ok(meshPlaneSection({
  vertices: nearbyHitMesh.vertices, faces: nearbyHitMesh.faces,
  plane_origin: [0, 0, 0], plane_normal: [0, 0, 1], seed: {face_index: 0, weights: [0.5, 0, 0.5]},
  required_hits: [{face_index: nearbyHitMesh.faces.length - 1, weights: [1, 0, 0]}],
  required_hit_tolerance: 5,
}).value_mm > 0);
assert.throws(() => meshPlaneSection({
  vertices: nearbyHitMesh.vertices, faces: nearbyHitMesh.faces,
  plane_origin: [0, 0, 0], plane_normal: [0, 0, 1], seed: {face_index: 0, weights: [0.5, 0, 0.5]},
  required_hits: [{face_index: nearbyHitMesh.faces.length - 1, weights: [0, 0, 1]}],
  required_hit_tolerance: 5,
}), /not on the selected unique ring/);
const offPlaneDefinition = JSON.parse(JSON.stringify(requiredHitDefinition));
offPlaneDefinition.anchors[1].surface_hit.weights = [0, 0.25, 0.75];
assert.throws(() => runtimeSection(offPlaneDefinition), /右胯骨最高点.*同一水平/);
const otherRingDefinition = JSON.parse(JSON.stringify(requiredHitDefinition));
otherRingDefinition.anchors[1].surface_hit = {
  face_index: legs.ranges[1].startFace, weights: [0.25, 0.25, 0.5],
};
assert.throws(() => runtimeSection(otherRingDefinition), /尺带未经过.*右胯骨最高点.*同一圈/);
const missingRequiredDefinition = JSON.parse(JSON.stringify(requiredHitDefinition));
missingRequiredDefinition.anchors.pop();
assert.throws(() => definitionRequiredHits(missingRequiredDefinition), /right_hip_crest.*canonical SurfaceHit/);
assert.deepEqual(definitionRequiredHits({method: 'surface_path'}), []);
assert.ok(section(requiredHitMesh, 0, {required_hits: coplanarRequiredHits}).value_mm > 0);
assert.throws(() => section(requiredHitMesh, 0, {required_hits: [
  coplanarRequiredHits[0], {face_index: 8, weights: [0, 0.25, 0.75]},
]}), /required hit 1 is not on the measurement plane/); // 50 mm above the section.
assert.throws(() => section(requiredHitMesh, 0, {required_hits: [
  {face_index: legs.ranges[1].startFace, weights: [0.25, 0.25, 0.5]},
]}), /required hit 0 is not on the selected unique ring/);
assert.throws(() => section(legs, legs.ranges[1].startFace, {
  allowed_faces: leftLegFaces, face_body_segment: legSegments, body_segment: 'left_thigh', face_side: legSides, side: 'left',
}), /outside the segment\/side filter/);
assert.ok(section(legs, legs.ranges[1].startFace).value_mm > 0);
assert.throws(() => section(legs, legs.ranges[0].startFace, {
  allowed_faces: leftLegFaces, face_body_segment: legSegments, body_segment: 'left_thigh', side: 'left',
}), /face_side must match faces/);
assert.throws(() => section(legs, legs.ranges[0].startFace, {
  allowed_faces: leftLegFaces.slice(0, -2), face_body_segment: legSegments, body_segment: 'left_thigh', face_side: legSides, side: 'left',
}), /degree-2 loop/);
assert.ok(section(legs, legs.ranges[0].startFace).value_mm > 0);

// A torso and two larger arm rings share one plane; the torso seed wins without a global-size heuristic.
const torsoAndArms = prismLoops([octagon(0), octagon(-3).map(([x, y]) => [-3 + (x + 3) * 1.4, y * 1.4]),
  octagon(3).map(([x, y]) => [3 + (x - 3) * 1.4, y * 1.4])]);
const torsoFaceEnd = torsoAndArms.ranges[1].startFace;
const torsoSegments = torsoAndArms.faces.map((_, face) => face < torsoFaceEnd ? 'torso' : 'arm');
const torso = section(torsoAndArms);
assert.ok(torso.path_faces.every(face => face < torsoFaceEnd));
const filteredTorso = section(torsoAndArms, 0, {
  face_body_segment: torsoSegments, body_segment: 'torso',
});
assert.ok(filteredTorso.path_faces.every(face => torsoSegments[face] === 'torso'));
assert.throws(() => section(torsoAndArms, torsoAndArms.ranges[1].startFace, {
  allowed_faces: rangeFaces(torsoAndArms.ranges[0]), face_body_segment: torsoSegments, body_segment: 'torso',
}), /outside the segment\/side filter/);

// A silhouette crossing both arms is a valid geometric loop only until the signed torso filter is applied.
const chestOutline = [[-1, -2], [1, -2], [1, 0], [3, 0], [3, 1], [1, 1], [1, 2], [-1, 2], [-1, 1], [-3, 1], [-3, 0], [-1, 0]];
const chestMesh = prismLoops([chestOutline]);
const chestSegments = chestMesh.faces.map((_, face) => {
  const edge = Math.floor(face / 2);
  const next = (edge + 1) % chestOutline.length;
  return Math.abs((chestOutline[edge][0] + chestOutline[next][0]) / 2) > 1 ? 'upper_arm' : 'torso';
});
const chestTorsoFaces = chestSegments.flatMap((segment, face) => segment === 'torso' ? [face] : []);
const unsafeChest = circumferenceRecord(chestMesh);
assert.ok(unsafeChest.path_faces.some(face => chestSegments[face] === 'upper_arm'));
assert.throws(() => circumferenceRecord(chestMesh, 0, {
  allowed_faces: chestTorsoFaces, face_body_segment: chestSegments, body_segment: 'torso',
}), /degree-2 loop/);

// Likewise, one pelvis-and-legs silhouette must not become a hip record after the pelvis filter opens the loop.
const hipOutline = [[-2, 2], [2, 2], [2, -3], [0.6, -3], [0.6, 0], [-0.6, 0], [-0.6, -3], [-2, -3]];
const hipMesh = prismLoops([hipOutline]);
const hipSegments = hipMesh.faces.map((_, face) => {
  const edge = Math.floor(face / 2);
  const next = (edge + 1) % hipOutline.length;
  return hipOutline[edge][1] >= 0 && hipOutline[next][1] >= 0 ? 'pelvis' : 'thigh';
});
const pelvisFaces = hipSegments.flatMap((segment, face) => segment === 'pelvis' ? [face] : []);
const unsafeHip = circumferenceRecord(hipMesh);
assert.ok(unsafeHip.path_faces.some(face => hipSegments[face] === 'thigh'));
assert.throws(() => circumferenceRecord(hipMesh, 0, {
  allowed_faces: pelvisFaces, face_body_segment: hipSegments, body_segment: 'pelvis',
}), /degenerate loop|degree-2 loop/);

// The fullest hip loop crosses the pelvis-thigh label seams and remains pure to those three segments.
const lowerHipSegments = hipMesh.faces.map((_, face) => {
  const edge = Math.floor(face / 2);
  if (hipOutline[edge][1] >= 0 && hipOutline[(edge + 1) % hipOutline.length][1] >= 0) return 'pelvis';
  return (hipOutline[edge][0] + hipOutline[(edge + 1) % hipOutline.length][0]) / 2 < 0
    ? 'left_thigh' : 'right_thigh';
});
const lowerHipAllowed = ['pelvis', 'left_thigh', 'right_thigh'];
const lowerHipRing = section(hipMesh, 0, {
  face_body_segment: lowerHipSegments, body_segment: lowerHipAllowed,
});
assert.ok(lowerHipAllowed.every(segment => lowerHipRing.path_faces.some(face => lowerHipSegments[face] === segment)));
assert.ok(lowerHipRing.path_faces.every(face => lowerHipAllowed.includes(lowerHipSegments[face])));
const hipPlacement = {
  body_segments: lowerHipAllowed,
  circumference_fit: 'surface',
  section_placement: {
    bounds: [{hits: ['crotch']}, {joints: ['pelvis']}],
    scan_fraction: [0.12, 0.9],
    select: 'buttock_prominence',
  },
};
const automaticHip = torsoCircumference({
  definition: hipPlacement,
  vertices: hipMesh.vertices, faces: hipMesh.faces, face_body_segment: lowerHipSegments,
  joint_names: ['pelvis', 'left_hip', 'right_hip'],
  joints: [[0, 1, 1], [-1, 1, 1], [1, 1, 1]], body_up: [0, 0, 1],
  landmark_hits: {crotch: {face_index: 8, weights: [0.5, 0.5, 0]}},
});
assert.ok(lowerHipAllowed.every(segment => automaticHip.path_faces.some(face => lowerHipSegments[face] === segment)));

const chestPlacement = {
  definition: {
    body_segments: ['torso'], circumference_fit: 'surface',
    section_placement: {bounds: [{joints: ['pelvis']}, {joints: ['neck']}]},
  },
  vertices: chestMesh.vertices, faces: chestMesh.faces, face_body_segment: chestSegments,
  joint_names: ['pelvis', 'left_hip', 'right_hip', 'neck'],
  joints: [[0, 0, -0.5], [-0.5, 0, -0.5], [0.5, 0, -0.5], [0, 0, 0.5]],
  body_up: [0, 0, 1], seed: {face_index: 0, weights: [0.5, 0, 0.5]},
};
assert.throws(() => torsoCircumference(chestPlacement), /degree-2 loop/);
assert.throws(() => torsoCircumference({...chestPlacement, fit: 'basic'}), /degree-2 loop/);

const notch = [[-4, -3], [4, -3], [4, 4], [1, 4], [1, 2], [-1, 2], [-1, 4], [-4, 4]]
  .map(([x,y]) => [x,y,0]);
const fits = circumferenceFits(notch, [0, 0, 1]);
assert.equal(fits.surface_mm, 34);
assert.equal(fits.basic_mm, 30);
assert.equal(fits.indices.length, 4);

// A closed thigh ring that hugs the pelvis label seam is not eligible for the fullest-thigh search.
function pairedTaperedLegs() {
  const vertices = [];
  const faces = [];
  const segments = [];
  for (const [side, cx] of [['left', -2], ['right', 2]]) {
    const start = vertices.length;
    const levels = [[6, 0.9], [5, 0.85], [0, 0.55]];
    levels.forEach(([y, radius]) => octagon(cx).forEach(([x, z]) =>
      vertices.push([cx + (x - cx) * radius / 0.7, y, z * radius / 0.7])));
    for (let band = 0; band < levels.length - 1; band += 1) {
      for (let edge = 0; edge < 8; edge += 1) {
        const next = (edge + 1) % 8;
        const low = start + band * 8 + edge;
        const lowNext = start + band * 8 + next;
        const high = low + 8;
        const highNext = lowNext + 8;
        faces.push([low, high, highNext], [low, highNext, lowNext]);
        segments.push(`${side}_thigh`, `${side}_thigh`);
      }
    }
    if (side === 'left') {
      vertices.push([cx, 7, 0]);
      faces.push([start, start + 8, vertices.length - 1]);
      segments.push('pelvis');
    }
  }
  return {vertices, faces, segments};
}
const pairedThighs = pairedTaperedLegs();
const pairedThighGeometry = {
  vertices: pairedThighs.vertices, faces: pairedThighs.faces, face_body_segment: pairedThighs.segments,
  ...lowerBodyJoints(-2, 2), body_up: [0, 1, 0], landmark_hits: {},
};
const pairedLeftDraft = automaticLowerBodyDraft({
  definition: definitionFor('thigh'), side: 'left', ...pairedThighGeometry,
});
const pairedRightDraft = automaticLowerBodyDraft({
  definition: definitionFor('thigh'), side: 'right', ...pairedThighGeometry,
});
const pairedDraftHeights = [pairedLeftDraft, pairedRightDraft].map(draft =>
  reconstructSurfaceHit(pairedThighs.vertices, pairedThighs.faces, draft.seed)[1]);
assert.ok(pairedDraftHeights[0] < 5);
assert.ok(pairedDraftHeights[1] > 5);
for (const contaminant of ['torso', 'left_upper_arm', 'left_shin']) {
  const contaminated = [...lowerHipSegments];
  contaminated[6] = contaminated[7] = contaminant;
  assert.throws(() => section(hipMesh, 0, {
    face_body_segment: contaminated, body_segment: lowerHipAllowed,
  }), /degree-2 loop/);
}

// Branches, duplicate two-edge cycles, plane-edge tangency and projected crossings are all rejected.
const branchVertices = [[0, 0, 0], [-2, 0, -1], [-2, 0, 1], [0, 2, -1], [0, 2, 1], [2, 0, -1], [2, 0, 1], [0, -2, -1], [0, -2, 1]];
const branchFaces = [[0, 1, 2], [0, 3, 4], [0, 5, 6], [0, 7, 8]];
assert.throws(() => meshPlaneSection({
  vertices: branchVertices, faces: branchFaces, plane_origin: [0, 0, 0], plane_normal: [0, 0, 1],
  seed: {face_index: 0, weights: [1, 0, 0]},
}), /degree-2 loop/);
const duplicateFaceMesh = {vertices: [[-1, 0, -1], [1, 0, -1], [0, 1, 1]], faces: [[0, 1, 2], [0, 2, 1]]};
assert.throws(() => section(duplicateFaceMesh), /degenerate loop/);
assert.throws(() => meshPlaneSection({
  vertices: legs.vertices, faces: legs.faces, plane_origin: [0, 0, -1], plane_normal: [0, 0, 1],
  seed: {face_index: 0, weights: [1, 0, 0]}, allowed_faces: leftLegFaces,
}), /contains an edge/);
const pentagon = Array.from({length: 5}, (_, index) => {
  const angle = 2 * Math.PI * index / 5;
  return [Math.cos(angle), Math.sin(angle)];
});
const starMesh = prismLoops([[0, 2, 4, 1, 3].map(index => pentagon[index])]);
assert.throws(() => section(starMesh), /self-intersects/);

// A concave section follows shared mesh edges. Its length is the source polygon perimeter.
const concavePolygon = [[-2, -1], [2, -1], [2, 1], [0.5, 1], [0.5, 0], [-0.5, 0], [-0.5, 1], [-2, 1]];
const concaveMesh = prismLoops([concavePolygon]);
const concave = section(concaveMesh);
const perimeter = concavePolygon.reduce((sum, point, i) => {
  const next = concavePolygon[(i + 1) % concavePolygon.length];
  return sum + Math.hypot(point[0] - next[0], point[1] - next[1]);
}, 0);
assert.ok(Math.abs(concave.value_mm - perimeter) < 1e-12);

const stripVertices = [[0, 3, 0], [1, 2, 0], [2, 4, 0], [3, 1, 0], [0, 2, 1], [1.5, 4, 1], [3, 1, 1]];
const stripFaces = [[0, 1, 4], [1, 2, 5], [2, 3, 6]];
const stripAnchors = [
  {face_index: 0, weights: [1, 0, 0]},
  {face_index: 2, weights: [0, 1, 0]},
];
// Total crotch is the midsagittal section arc selected by the required crotch hit.
const crotchMesh = prismLoops([[[-2, 3], [-2, 1.1], [-1.8, 1.2], [-1, 0], [0, 0], [0.5, -3],
  [1, 0], [1.8, 1.2], [2, 1.1], [2, 3]]]);
const crotchVertices = crotchMesh.vertices;
const crotchFaces = crotchMesh.faces;
const crotchRing = section(crotchMesh);
const crotchRingHits = crotchRing.path_faces.map((face_index, index) =>
  ({face_index, weights: crotchRing.path_weights[index]}));
const crotchHitAt = (x, y) => crotchRingHits.find(hit => {
  const point = reconstructSurfaceHit(crotchVertices, crotchFaces, hit);
  return Math.abs(point[0] - x) < 1e-9 && Math.abs(point[1] - y) < 1e-9;
});
const crotchAnchors = [
  {id: 'front_waist', hit: crotchHitAt(-2, 3)},
  {id: 'crotch', hit: crotchHitAt(-1, 0)},
  {id: 'back_waist', hit: crotchHitAt(2, 3)},
];
const crotchDefinition = {
  definition_status: 'admitted', item_id: 'total_crotch', metric_id: 'total_crotch_length',
  definition_id: 'lower-body-8-v1', version: 'v1', method: 'surface_path', value_rule: 'path_length',
  pose_id: 'legs-together-001',
  body_segments: definitionFor('total_crotch').body_segments,
  sides: ['center'], anchor_ids: ['front_waist', 'crotch', 'back_waist'], crotch_anchor_id: 'crotch',
  segment_corridors: [
    {start_id: 'front_waist', end_id: 'crotch', side: 'center', direction: 'descending', allowed_faces: [0]},
    {start_id: 'crotch', end_id: 'back_waist', side: 'center', direction: 'ascending', allowed_faces: [0, 1]},
  ],
};
const solveCrotch = (definition = crotchDefinition, anchors = crotchAnchors) => segmentedSurfacePath({
  vertices: crotchVertices, faces: crotchFaces, anchors, definition,
  side: 'center',
  body_up: [0, 1, 0], ...lowerBodyJoints(), face_body_segment: crotchFaces.map(() => 'right_thigh'), body_height_mm: 10,
});
const crotchPath = solveCrotch();
assert.throws(() => solveCrotch({...crotchDefinition, body_segments: ['torso']}), /leaves the crotch body segments/);
assert.deepEqual(reconstructSurfaceHit(crotchVertices, crotchFaces, crotchPath.resolved_crotch_hit), [-1, 0, 0]);
assert.equal(crotchPath.value_mm, crotchPath.path_length_mm);
const crotchPoints = crotchPath.path_faces.map((face_index, index) => reconstructSurfaceHit(
  crotchVertices, crotchFaces, {face_index, weights: crotchPath.path_weights[index]},
));
const resolvedCrotchIndex = crotchPoints.findIndex(point => Math.abs(point[0] + 1) < 1e-9
  && Math.abs(point[1]) < 1e-9);
assert.ok(resolvedCrotchIndex > 0 && resolvedCrotchIndex < crotchPoints.length - 2);
assert.ok([[-2, 1.1], [-1.8, 1.2], [1.8, 1.2], [2, 1.1]].every(([x, y]) =>
  crotchPoints.some(point => Math.abs(point[0] - x) < 1e-9 && Math.abs(point[1] - y) < 1e-9)));
assert.ok([crotchPoints[resolvedCrotchIndex - 1], crotchPoints[resolvedCrotchIndex + 1]]
  .some(point => point[1] === crotchPoints[resolvedCrotchIndex][1]));
assert.equal(Math.min(...crotchPoints.map(point => point[1])), crotchPoints[resolvedCrotchIndex][1]);
assert.throws(() => solveCrotch({...crotchDefinition, value_rule: 'unknown'}), /value_rule/);
assert.throws(() => segmentedSurfacePath({
  vertices: crotchVertices, faces: crotchFaces, anchors: crotchAnchors, definition: crotchDefinition,
  side: 'center',
  body_up: [0, 0, 0], ...lowerBodyJoints(), face_body_segment: crotchFaces.map(() => 'right_thigh'), body_height_mm: 10,
}), /body_up must be non-zero/);
assert.deepEqual(reconstructSurfaceHit(crotchVertices, crotchFaces,
  {face_index: crotchPath.path_faces[0], weights: crotchPath.path_weights[0]}), [-2, 3, 0]);
assert.deepEqual(reconstructSurfaceHit(crotchVertices, crotchFaces,
  {face_index: crotchPath.path_faces.at(-1), weights: crotchPath.path_weights.at(-1)}), [2, 3, 0]);
assert.ok(crotchPoints.some(point => Math.abs(point[0] - 1) < 1e-9 && Math.abs(point[1]) < 1e-9));
assert.ok(crotchPoints.every(point => point[1] >= 0));
assert.ok(crotchPoints.every(point => Math.abs(point[0]) > 1e-9));
const rebuiltCrotch = solveCrotch(crotchDefinition, [
  crotchAnchors[0], {id: 'crotch', hit: crotchPath.resolved_crotch_hit}, crotchAnchors[2],
]);
assert.deepEqual(rebuiltCrotch.path_faces, crotchPath.path_faces);
assert.deepEqual(rebuiltCrotch.path_weights, crotchPath.path_weights);
assert.deepEqual(solveCrotch({...crotchDefinition, segment_corridors: []}).path_faces, crotchPath.path_faces);
assert.throws(() => solveCrotch(crotchDefinition, [crotchAnchors[1], crotchAnchors[0], crotchAnchors[2]]), /anchor IDs\/count\/order/);
assert.throws(() => segmentedSurfacePath({
  vertices: crotchVertices, faces: crotchFaces, anchors: [crotchAnchors[0], crotchAnchors[2]],
  definition: {...crotchDefinition, anchor_ids: ['front_waist', 'back_waist'],
    segment_corridors: [{start_id: 'front_waist', end_id: 'back_waist', side: 'center', direction: 'free', allowed_faces: [2]}]},
  side: 'center',
  body_up: [0, 1, 0], ...lowerBodyJoints(), face_body_segment: crotchFaces.map(() => 'pelvis'), body_height_mm: 10,
}), /requires front, crotch, and back/);

const legVertices = [[0, 6, 0], [0, 0, 0], [1, 0, 0]];
const legFaces = [[0, 1, 2]];
const legDefinition = (item_id, anchor_ids) => ({
  definition_status: 'admitted', item_id, metric_id: `${item_id}_length`, definition_id: 'lower-body-8-v1',
  version: 'v1', method: 'surface_path',
  value_rule: ['outseam', 'inseam'].includes(item_id) ? 'body_up_projection' : 'path_length',
  pose_id: 'leg-pose', sides: ['left'], anchor_ids,
  body_segments: definitionFor(item_id)?.body_segments.map(segment => segment.replace('{side}', 'left')),
  segment_corridors: [{start_id: anchor_ids[0], end_id: anchor_ids[1], side: 'left',
    direction: 'descending', allowed_faces: [0]}],
});
const outDefinition = legDefinition('outseam', ['hip', 'lateral_malleolus']);
const outseamVertices = [
  [0, 6, 0], [0.5, 3, 0], [1, 0, 0],
  [0, 6, 100], [0.5, 3, 100], [1, 0, 100], [1, 0, 101],
];
const outseamFaces = [[0, 1, 3], [1, 2, 4], [2, 5, 6]];
const outseamAnchors = [
  {id: 'hip', hit: {face_index: 0, weights: [1, 0, 0]}},
  {id: 'lateral_malleolus', hit: {face_index: 2, weights: [1, 0, 0]}},
];
const solveOutseam = (body_up = [0, 1, 0], body_height_mm = 10) => segmentedSurfacePath({
  vertices: outseamVertices, faces: outseamFaces, anchors: outseamAnchors,
  definition: outDefinition,
  side: 'left', body_up, ...lowerBodyJoints(1, 3),
  face_body_segment: ['left_thigh', 'left_shin', 'left_foot'], body_height_mm,
});
const outPath = solveOutseam();
assert.equal(outPath.value_mm, 6);
assert.ok(outPath.path_length_mm > outPath.value_mm);
assert.ok(outPath.path_faces.length > 2);
const outPoints = outPath.path_faces.map((face_index, index) => reconstructSurfaceHit(
  outseamVertices, outseamFaces, {face_index, weights: outPath.path_weights[index]},
));
assert.ok(outPoints.slice(1).every((point, index) => point[1] <= outPoints[index][1]));
assert.ok(outPoints.every(point => point[2] === 0));
assert.throws(() => segmentedSurfacePath({vertices: outseamVertices, faces: outseamFaces, anchors: outseamAnchors,
  definition: {...outDefinition, body_segments: ['torso', 'pelvis', 'left_thigh', 'left_shin', 'left_foot']},
  side: 'left', body_up: [0, 1, 0],
  ...lowerBodyJoints(1, 3), face_body_segment: ['right_thigh', 'right_shin', 'right_foot'], body_height_mm: 10,
}), /no same-side surface sample/);
assert.equal(solveOutseam([0, 2, 0]).value_mm, 6);
assert.throws(() => solveOutseam([0, 1, 0], 5), /outseam < body_height/);
const inAnchors = [
  {id: 'crotch', hit: {face_index: 0, weights: [2 / 3, 1 / 3, 0]}},
  {id: 'medial_malleolus', hit: {face_index: 2, weights: [0, 0, 1]}},
];
const inDefinition = legDefinition('inseam', ['crotch', 'medial_malleolus']);
const solveInseam = paired_outseam_mm => segmentedSurfacePath({
  vertices: outseamVertices, faces: outseamFaces, anchors: inAnchors,
  definition: inDefinition,
  side: 'left', body_up: [0, 1, 0], ...lowerBodyJoints(-1, 1),
  face_body_segment: ['left_thigh', 'left_shin', 'left_foot'], body_height_mm: 10, paired_outseam_mm,
});
assert.ok(solveInseam(outPath.value_mm).value_mm < outPath.value_mm);
assert.equal(solveInseam(outPath.value_mm).value_mm, 5);
assert.ok(solveInseam(outPath.value_mm).path_length_mm > solveInseam(outPath.value_mm).value_mm);
const inSegments = solveInseam(outPath.value_mm).path_faces.map(face =>
  ['left_thigh', 'left_shin', 'left_foot'][face]);
assert.deepEqual(inSegments.filter((segment, index) => index === 0 || segment !== inSegments[index - 1]),
  ['left_thigh', 'left_shin', 'left_foot']);
assert.deepEqual([solveInseam(outPath.value_mm).path_faces.at(-1),
  solveInseam(outPath.value_mm).path_weights.at(-1)], [2, [0, 0, 1]]);
assert.throws(() => solveInseam(3), /inseam < outseam/);
assert.throws(() => segmentedSurfacePath({vertices: stripVertices, faces: stripFaces,
  anchors: [{id: 'hip', hit: stripAnchors[0]}, {id: 'ankle', hit: stripAnchors[1]}],
  definition: {...legDefinition('outseam', ['hip', 'ankle']), segment_corridors: [
    {start_id: 'hip', end_id: 'ankle', side: 'left', direction: 'descending', allowed_faces: [0, 1, 2]}]},
  side: 'left', body_up: [0, 1, 0], ...lowerBodyJoints(4, 6), body_height_mm: 10}),
  /face_body_segment must match faces/);
assert.throws(() => segmentedSurfacePath({vertices: stripVertices, faces: stripFaces,
  anchors: [{id: 'hip', hit: stripAnchors[0]}, {id: 'ankle', hit: stripAnchors[1]}],
  definition: legDefinition('free_length', ['hip', 'ankle']),
  side: 'left',
  body_up: [0, 1, 0]}), /requires a workbook length item/);
assert.throws(() => segmentedSurfacePath({vertices: stripVertices, faces: stripFaces,
  anchors: [{id: 'hip', hit: stripAnchors[0]}, {id: 'toe', hit: stripAnchors[1]}],
  definition: {...legDefinition('outseam', ['hip', 'toe']), segment_corridors: [
    {start_id: 'hip', end_id: 'toe', side: 'left', direction: 'descending', allowed_faces: [0, 1]}]},
  side: 'left', body_up: [0, 1, 0],
  ...lowerBodyJoints(4, 6), body_height_mm: 10}), /face_body_segment must match faces/);

const surfaceRecord = {units: 'mm', method: 'surface_path', path_faces: outPath.path_faces,
  path_weights: outPath.path_weights, value_mm: outPath.path_length_mm};
const straightRecord = {units: 'mm', method: 'straight', path_faces: [0, 0],
  path_weights: [[1, 0, 0], [0, 1, 0]], value_mm: 6};
assert.equal(recomputeSurfaceTapeRecord(surfaceRecord, outseamVertices, outseamFaces), outPath.path_length_mm);
assert.equal(recomputeSurfaceTapeRecord(straightRecord, legVertices, legFaces), 6);
const projection = {axis: [0, 1, 0], pathLengthMm: outPath.path_length_mm};
const projectedRecord = {...surfaceRecord, value_mm: outPath.value_mm};
assert.equal(recomputeSurfaceTapeRecord(projectedRecord, outseamVertices, outseamFaces, 0.001, projection), outPath.value_mm);
assert.throws(() => recomputeSurfaceTapeRecord(surfaceRecord, outseamVertices, outseamFaces, 0.001, projection), /value_mm mismatch/);

const savedPose = {id: 'pose', pose_id: 'pose', mesh_id: 'mesh', height_cm: 20, vertices: legVertices};
const savedDefinition = {
  item_id: 'length', metric_id: 'length', definition_id: 'length:v1', version: 'v1',
  method: 'straight', pose_id: 'pose', sides: ['center'], body_segments: ['torso'],
  anchor_ids: ['start', 'end'],
  anchors: straightRecord.path_weights.map((weights, i) => ({
    id: i === 0 ? 'start' : 'end', surface_hit: {face_index: 0, weights},
  })),
};
const savedTape = {
  ...straightRecord, schema_version: 1, id: 'tape', mesh_id: 'mesh', pose_id: 'pose',
  item_id: 'length', metric_id: 'length', definition_id: 'length:v1', version: 'v1',
  side: 'center', anchors: savedDefinition.anchors.map(({id, surface_hit}) => ({id, hit: surface_hit})),
};
const savedProgress = {
  state_kind: 'body_measurement_viewer_state', contract_version: 2, workbench_id: 'workbench',
  selected_pose_id: 'pose', current_height_cm: 20, height_is_estimate: false, measurements: [savedTape],
};
const records = createMeasurementRecords({
  workbench_id: 'workbench', poses: [savedPose], mesh: {faces: legFaces},
  measurement_definitions: {definitions: [savedDefinition]},
  viewer_state_schema: {
    state_kind: savedProgress.state_kind,
    surface_tape_record: {
      schema_version: 1, sides: ['center'], methods: ['straight'], units: 'mm',
      required_fields: Object.keys(savedTape), retired_fields: ['points'],
    },
  },
});
assert.deepEqual(records.workProgress(records.readWorkProgress(savedProgress)).measurements, [savedTape]);
assert.throws(() => records.readWorkProgress({...savedProgress, measurements: [
  {...savedTape, points: []},
]}), /已停用/);
assert.throws(() => records.readWorkProgress({...savedProgress, measurements: [
  {...savedTape, value_mm: 6.01},
]}), /mismatch/);

const personalCalibration = {
  subject_id: 'subject-1', fit_video_id: 'iphone-video', method: 'additive_offset_cm',
  applicability: 'same_subject_only', offsets_cm: {hip_circumference: -0.7},
};
assert.equal(personalCalibratedValueCm('hip_circumference', 90.7, personalCalibration, 'subject-1'), 90);
assert.equal(personalCalibratedValueCm('knee_circumference', 40, personalCalibration, 'subject-1'), null);
assert.equal(personalCalibratedValueCm('hip_circumference', 90.7, null, 'subject-1'), null);
assert.throws(
  () => personalCalibratedValueCm('hip_circumference', 90.7, personalCalibration, 'subject-2'),
  /相同人物/,
);

// Bad face/weights fail before a record can be accepted; a closed result recomputes exactly.
assert.throws(() => reconstructSurfaceHit(legs.vertices, legs.faces, {face_index: 999, weights: [1, 0, 0]}), /out of range/);
assert.throws(() => reconstructSurfaceHit(legs.vertices, legs.faces, {face_index: 0, weights: [0.7, 0.7, -0.4]}), /barycentric/);
const float32Hit = {face_index: 0, weights: [0.0009221306536346674, 0, 0.9990778565406799]};
assert.deepEqual(reconstructSurfaceHit(legs.vertices, legs.faces, float32Hit),
  [0, 1, 2].map(axis => float32Hit.weights.reduce((sum, weight, index) =>
    sum + weight * legs.vertices[legs.faces[0][index]][axis], 0) / float32Hit.weights.reduce((sum, weight) => sum + weight, 0)));
assert.throws(() => reconstructSurfaceHit(legs.vertices, legs.faces, {face_index: 0, weights: [0.49, 0.49, 0]}), /sum to 1/);
const record = {
  units: 'mm', method: 'mesh_section', path_faces: left.path_faces,
  path_weights: left.path_weights, value_mm: left.value_mm,
};
assert.equal(recomputeSurfaceTapeRecord(record, legs.vertices, legs.faces), left.value_mm);
assert.throws(() => recomputeSurfaceTapeRecord({...record, value_mm: record.value_mm + 1}, legs.vertices, legs.faces), /mismatch/);
assert.throws(() => recomputeSurfaceTapeRecord({
  ...record, path_faces: record.path_faces.slice(0, 2), path_weights: record.path_weights.slice(0, 2),
}, legs.vertices, legs.faces), /path_faces\/path_weights/);

const presetEntries = definitions.filter(item => item.preset_schema_id === 'lower-body-8-v1');
assert.equal(presetEntries.filter(item => item.method === 'mesh_section').length, 5);
assert.equal(presetEntries.filter(item => item.method === 'surface_path').length, 3);
const lowerBodyPoseIds = new Set(['natural_standing', 'legs_together', 'legs_shoulder_width']);
const lowerBodySegmentIds = new Set(['torso', 'pelvis', 'left_thigh', 'left_shin', 'left_foot', 'right_thigh', 'right_shin', 'right_foot']);
assert.equal(presetEntries.length, 8);
presetEntries.forEach(item => {
  assert.ok(lowerBodyPoseIds.has(item.lower_body_pose_id));
  assert.ok(item.sides.every(side => item.body_segments.every(segment =>
    lowerBodySegmentIds.has(segment.replace('{side}', side)))));
});
  console.log('surface_measurement: ok');
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
