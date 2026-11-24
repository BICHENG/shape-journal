const EPSILON = 1e-9;

function compareRank(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (Math.abs(left[index] - right[index]) > EPSILON) return left[index] - right[index];
  }
  return 0;
}

function jointCoordinates(joints, names, operation) {
  if (!Array.isArray(joints) || !Array.isArray(names) || joints.length !== names.length) {
    throw new Error(`${operation} joints must match joint_names`);
  }
  const coordinates = new Map(names.map((name, index) => [name, vector3(joints[index], `joint ${name}`)]));
  return name => {
    const point = coordinates.get(name);
    if (!point) throw new Error(`${operation} is missing joint ${name}`);
    return point;
  };
}

function requireFaceSegments(faces, segments, operation) {
  if (!Array.isArray(segments) || segments.length !== faces.length) {
    throw new Error(`${operation} face_body_segment must match faces`);
  }
}

export function surfacePathHits({path_faces, path_weights}) {
  return path_faces.map((face_index, index) => ({face_index, weights: path_weights[index]}));
}

export function automaticLowerBodyDraft({
  definition, waist_definition, side = 'center', vertices, faces, face_body_segment, joints, joint_names,
  body_up, landmark_hits = {},
}) {
  const up = normalize(vector3(body_up, 'body_up'));
  requireFaceSegments(faces, face_body_segment, 'automatic draft');
  const joint = jointCoordinates(joints, joint_names, 'automatic draft');
  const pelvis = joint('pelvis');
  const leftHip = joint('left_hip');
  const rightHip = joint('right_hip');
  const bodyRight = normalize(subtract(rightHip, leftHip));
  const front = normalize(cross(up, bodyRight));
  const landmark = id => {
    const hit = landmark_hits[id];
    if (!hit) throw new Error(`automatic draft is missing landmark ${id}`);
    reconstructSurfaceHit(vertices, faces, hit);
    return hit;
  };
  const height = point => dot(point, up);
  const profile = (start, end, startFraction, endFraction, segments, oppositeStart = null, oppositeEnd = null, score = ring => [ring.value_mm]) => {
    let best = null;
    for (let index = 0; index < 16; index += 1) {
      const t = startFraction + (endFraction - startFraction) * index / 15;
      const axis = mix(start, end, t);
      const opposite = oppositeStart && oppositeEnd ? mix(oppositeStart, oppositeEnd, t) : null;
      const ring = automaticRing({
        vertices, faces, face_body_segment, segments, up, bodyRight, front,
        plane_height: height(axis), axis, exclude_axis: opposite,
      });
      if (!ring) continue;
      const rank = score(ring);
      if (rank.every(Number.isFinite) && (!best || compareRank(rank, best.rank) > 0)) best = {...ring, rank};
    }
    return best;
  };
  const hipCrestRing = () => {
    const points = waist_definition.required_hit_ids.map(id =>
      reconstructSurfaceHit(vertices, faces, landmark(id)));
    const segments = waist_definition.body_segments;
    const axis = mix(points[0], points[1], 0.5);
    const planeHeight = height(axis);
    const seed = nearestPlaneSeed(
      vertices, faces, face_body_segment, new Set(segments), up, planeHeight, axis,
    );
    if (!seed) throw new Error('hip crest plane does not cut pelvis or torso');
    return {seed, ...sectionRing({
      vertices, faces, face_body_segment, body_segment: segments,
      plane_origin: up.map(value => value * planeHeight), plane_normal: up, seed,
    })};
  };
  if (definition?.required_hit_ids) {
    const hits = definition.required_hit_ids.map(landmark);
    return {seed: hits[0], required_hits: hits};
  }
  const placement = definition?.automatic_placement;
  const segments = definition?.body_segments.map(segment => segment.replace('{side}', side));
  const opposite = side === 'left' ? 'right' : 'left';
  if (placement?.joint_span) {
    const [start, end] = placement.joint_span;
    const boundaryEdges = placement.exclude_segment_boundary
      ? segmentBoundaryEdges(faces, face_body_segment) : [];
    const ring = profile(joint(`${side}_${start}`), joint(`${side}_${end}`), ...placement.scan_fraction, segments,
      joint(`${opposite}_${start}`), joint(`${opposite}_${end}`),
      candidate => candidate.points.some(point => boundaryEdges.some(([a, b]) =>
        pointSegmentDistance(point, vertices[a], vertices[b]) <= 0.1 * distance(vertices[a], vertices[b])))
        ? [NaN] : [candidate.value_mm]);
    return ring ? {seed: ring.seed, required_hits: []} : null;
  }
  if (placement?.joint) {
    const point = reconstructSurfaceHit(vertices, faces, landmark(definition.seed.id.replace('{side}', side)));
    const planeHeight = height(point);
    const onPlane = point => point.map((value, index) => value + up[index] * (planeHeight - height(point)));
    const ring = automaticRing({
      vertices, faces, face_body_segment, segments, up, bodyRight, front,
      plane_height: planeHeight,
      axis: onPlane(joint(`${side}_${placement.joint}`)),
      exclude_axis: onPlane(joint(`${opposite}_${placement.joint}`)),
    });
    return ring ? {seed: ring.seed, required_hits: []} : null;
  }
  if (definition.item_id === 'total_crotch') {
    const crotch = definition.crotch_anchor_id;
    if (!landmark_hits[crotch]?.unambiguous) return null;
    const ring = hipCrestRing();
    const midline = midsagittalRingHits({
      vertices, faces, face_body_segment, ring, up, bodyRight, front, pelvis,
    });
    if (!midline) return null;
    return {
      anchors: [midline.front.hit, landmark(crotch), midline.back.hit],
    };
  }
  return definition?.method === 'surface_path'
    ? {anchors: definition.anchor_ids.map(id => landmark(id.replace('{side}', side)))} : null;
}

function segmentBoundaryEdges(faces, faceSegments) {
  const edgeSegments = new Map();
  faces.forEach((face, faceIndex) => face.forEach((vertex, index) => {
    const next = face[(index + 1) % face.length];
    const edge = vertex < next ? `${vertex}:${next}` : `${next}:${vertex}`;
    if (!edgeSegments.has(edge)) edgeSegments.set(edge, new Set());
    edgeSegments.get(edge).add(faceSegments[faceIndex]);
  }));
  return [...edgeSegments].flatMap(([edge, segments]) =>
    segments.size > 1 ? [edge.split(':').map(Number)] : []);
}

export function resolveInseamIntent({anchors, landmark_hits}) {
  if (!Array.isArray(anchors) || anchors.length !== 2) return null;
  const hits = anchors.map(anchor => landmark_hits?.[anchor.id]);
  if (hits.some(hit => !hit || !Number.isInteger(hit.face_index)
      || !Array.isArray(hit.weights) || hit.weights.length !== 3 || hit.weights.some(weight => !Number.isFinite(weight)))) {
    return null;
  }
  return hits.map((hit, index) => ({
    ...anchors[index],
    hit: {face_index: hit.face_index, weights: hit.weights.slice()},
  }));
}

export function resolveTotalCrotchIntent({anchors, ...geometry}) {
  if (!Array.isArray(anchors) || anchors.length !== 3) return null;
  const {definition, landmark_hits} = geometry;
  const crotch = definition.crotch_anchor_id;
  const intent = anchors.find(anchor => anchor.id === crotch)?.hit;
  if (!intent) return null;
  const draft = automaticLowerBodyDraft({
    ...geometry,
    side: 'center',
    landmark_hits: {...landmark_hits, [crotch]: {...intent, unambiguous: true}},
  });
  if (!draft?.anchors || draft.anchors.length !== 3) return null;
  return draft.anchors.map((hit, index) => ({
    ...anchors[index],
    hit: {face_index: hit.face_index, weights: hit.weights.slice()},
  }));
}

export function legVisualPath({
  item_id, vertices, faces, face_body_segment, side, body_up, joints, joint_names, anchors,
  body_segments, sample_count = 24,
}) {
  const up = normalize(vector3(body_up, 'body_up'));
  if (!['outseam', 'inseam'].includes(item_id)) throw new Error('leg visual path item must be outseam or inseam');
  if (!['left', 'right'].includes(side)) throw new Error('leg visual path side must be left or right');
  if (!Number.isInteger(sample_count) || sample_count < 2) throw new Error('leg visual path sample_count must be at least 2');
  requireFaceSegments(faces, face_body_segment, 'leg visual path');
  const joint = jointCoordinates(joints, joint_names, 'leg visual path');
  const leftHip = joint('left_hip');
  const rightHip = joint('right_hip');
  const hipDelta = subtract(rightHip, leftHip).map((value, index) =>
    value - up[index] * dot(subtract(rightHip, leftHip), up));
  const bodyRight = normalize(hipDelta);
  const front = normalize(cross(up, bodyRight));
  const sideSign = side === 'right' ? 1 : -1;
  const axis = ['hip', 'knee', 'ankle'].map(name => joint(`${side}_${name}`));
  const axisHeights = axis.map(point => dot(point, up));
  if (axisHeights.some((height, index) => index && height >= axisHeights[index - 1] - EPSILON)) {
    throw new Error('leg visual path joint heights must descend');
  }
  const axisAtHeight = height => {
    const pair = height >= axisHeights[1] ? [axis[0], axis[1]] : [axis[1], axis[2]];
    const upper = dot(pair[0], up);
    const lower = dot(pair[1], up);
    return mix(pair[0], pair[1], Math.max(0, Math.min(1, (upper - height) / (upper - lower))));
  };
  const points = anchors.map(anchor => reconstructSurfaceHit(vertices, faces, anchor.hit || anchor));
  const startHeight = dot(points[0], up);
  const endHeight = dot(points[1], up);
  if (startHeight <= endHeight + EPSILON) throw new Error('leg visual path anchors must descend');
  const coordinate = (point, direction) => dot(subtract(point, axisAtHeight(dot(point, up))), direction);
  const startLateral = sideSign * coordinate(points[0], bodyRight);
  const endLateral = sideSign * coordinate(points[1], bodyRight);
  const startFront = coordinate(points[0], front);
  const endFront = coordinate(points[1], front);
  const vertexHits = new Map();
  faces.forEach((face, faceIndex) => face.forEach((vertex, local) => {
    const segment = face_body_segment[faceIndex];
    if (!vertexHits.has(vertex)) vertexHits.set(vertex, new Map());
    if (!vertexHits.get(vertex).has(segment)) {
      vertexHits.get(vertex).set(segment, {
        face_index: faceIndex,
        weights: [0, 1, 2].map(index => Number(index === local)),
      });
    }
  }));
  const proximal = body_segments.filter(segment => segment === 'torso' || segment === 'pelvis'
    || (item_id === 'inseam' && segment.endsWith('_thigh')));
  const segmentAt = height => {
    if (height >= axisHeights[0]) return proximal;
    const part = height >= axisHeights[1] ? 'thigh' : height >= axisHeights[2] ? 'shin' : 'foot';
    const segment = `${side}_${part}`;
    return body_segments.includes(segment) ? [segment] : [];
  };
  const output = [anchors[0].hit || anchors[0]];
  let previousHeight = startHeight;
  for (let index = 1; index < sample_count; index += 1) {
    const t = index / sample_count;
    const height = startHeight * (1 - t) + endHeight * t;
    const target = axisAtHeight(height).map((value, axisIndex) => value
      + bodyRight[axisIndex] * sideSign * (startLateral * (1 - t) + endLateral * t)
      + front[axisIndex] * (startFront * (1 - t) + endFront * t));
    const segments = segmentAt(height);
    let best = null;
    vertexHits.forEach((hits, vertex) => {
      const hit = hits.get(segments.find(segment => hits.has(segment)));
      if (!hit) return;
      const vertexHeight = dot(vertices[vertex], up);
      if (vertexHeight > previousHeight + EPSILON || vertexHeight < endHeight - EPSILON) return;
      const rank = [distance(vertices[vertex], target), vertex];
      if (!best || compareRank(rank, best.rank) < 0) best = {hit, height: vertexHeight, rank};
    });
    if (!best) throw new Error('leg visual path has no same-side surface sample');
    appendDistinctHit(output, best.hit, vertices, faces);
    previousHeight = best.height;
  }
  appendDistinctHit(output, anchors[1].hit || anchors[1], vertices, faces);
  const pathPoints = output.map(hit => reconstructSurfaceHit(vertices, faces, hit));
  if (pathPoints.some((point, index) => index && dot(point, up) > dot(pathPoints[index - 1], up) + EPSILON)) {
    throw new Error('leg visual path must descend');
  }
  return pathResult(output, polylineLength(pathPoints, false));
}

function automaticRing({
  vertices, faces, face_body_segment, segments, up, bodyRight, front,
  plane_height, axis, exclude_axis = null,
}) {
  const seed = nearestPlaneSeed(vertices, faces, face_body_segment, new Set(segments), up, plane_height, axis);
  if (!seed) return null;
  try {
    const ring = sectionRing({
      vertices, faces, face_body_segment, body_segment: segments,
      plane_origin: up.map(value => value * plane_height), plane_normal: up, seed,
    });
    if (!ringContains(ring.points, axis, bodyRight, front)
        || (exclude_axis && ringContains(ring.points, exclude_axis, bodyRight, front))) return null;
    return {seed, ...ring};
  } catch (_error) {
    return null;
  }
}

export function limbAxisCircumference({
  definition, side, vertices, faces, face_body_segment, joints, joint_names,
}) {
  const joint = jointCoordinates(joints, joint_names, 'limb circumference');
  const placement = definition.automatic_placement;
  const [start, end] = placement.joint_span.map(name => joint(side + '_' + name));
  const axis = mix(start, end, placement.fraction);
  const up = normalize(subtract(end, start));
  const bodyRight = normalize(cross(up, Math.abs(up[0]) < .9 ? [1, 0, 0] : [0, 1, 0]));
  const front = cross(up, bodyRight);
  const segments = definition.body_segments.map(segment => segment.replace('{side}', side));
  const ring = automaticRing({
    vertices, faces, face_body_segment, segments, up, bodyRight, front,
    plane_height: dot(axis, up), axis,
  });
  return ring && pathResult(ring.hits, ring.value_mm);
}

function nearestPlaneSeed(vertices, faces, faceSegments, segments, normal, planeHeight, axis) {
  let best = null;
  faces.forEach((face, faceIndex) => {
    if (!segments.has(faceSegments[faceIndex])) return;
    planeFaceHits(vertices, face, faceIndex, normal, planeHeight).forEach(candidate => {
      const point = reconstructSurfaceHit(vertices, faces, candidate);
      const value = distance(point, axis);
      if (!best || value < best.value) best = {hit: candidate, value};
    });
  });
  return best?.hit || null;
}

function planeFaceHits(vertices, face, faceIndex, normal, planeHeight) {
  const distances = face.map(vertex => dot(vertices[vertex], normal) - planeHeight);
  return triangleCuts(vertices, face, faceIndex, distances, EPSILON).map(cut => cut.hit);
}

function ringContains(points, axis, right, front) {
  const polygon = points.map(point => [dot(point, right), dot(point, front)]);
  const target = [dot(axis, right), dot(axis, front)];
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const a = polygon[index];
    const b = polygon[previous];
    if ((a[1] > target[1]) !== (b[1] > target[1])
        && target[0] < (b[0] - a[0]) * (target[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
  }
  return inside;
}

function midsagittalRingHits({vertices, faces, face_body_segment, ring, up, bodyRight, front, pelvis}) {
  const planeHeight = dot(ring.points[0], up);
  let section;
  try {
    section = sectionRing({
      vertices, faces, plane_origin: up.map(value => value * planeHeight), plane_normal: up,
      seed: ring.seed, required_hit_tolerance: 1e-5,
    });
  } catch (_error) {
    return null;
  }
  const candidates = [];
  faces.forEach((face, faceIndex) => {
    if (!['torso', 'pelvis'].includes(face_body_segment[faceIndex])) return;
    const planeHits = planeFaceHits(vertices, face, faceIndex, up, planeHeight);
    if (planeHits.length !== 2) return;
    const points = planeHits.map(hit => reconstructSurfaceHit(vertices, faces, hit));
    const lateral = points.map(point => dot(subtract(point, pelvis), bodyRight));
    if (lateral[0] * lateral[1] > 0) return;
    const denominator = lateral[0] - lateral[1];
    if (Math.abs(denominator) <= EPSILON) return;
    const t = lateral[0] / denominator;
    const weights = [0, 1, 2].map(index => planeHits[0].weights[index] * (1 - t) + planeHits[1].weights[index] * t);
    const hit = {face_index: faceIndex, weights};
    try {
      const point = section.checkRequiredHit(hit, 0);
      candidates.push({hit, point, depth: dot(subtract(point, pelvis), front)});
    } catch (_error) {
      // Candidate belongs to another coplanar component.
    }
  });
  if (candidates.length < 2) return null;
  candidates.sort((a, b) => a.depth - b.depth);
  return {back: candidates[0], front: candidates[candidates.length - 1]};
}

export function reconstructSurfaceHit(vertices, faces, hit, epsilon = EPSILON) {
  if (!hit || !Number.isInteger(hit.face_index) || hit.face_index < 0 || hit.face_index >= faces.length) {
    throw new Error('SurfaceHit face_index is out of range');
  }
  const face = faces[hit.face_index];
  validateFace(face, vertices.length, hit.face_index);
  const weights = hit.weights;
  if (!Array.isArray(weights) || weights.length !== 3 || weights.some(weight =>
    !Number.isFinite(weight) || weight < -epsilon || weight > 1 + epsilon)) {
    throw new Error('SurfaceHit weights must be three finite barycentric values');
  }
  const weightSum = weights[0] + weights[1] + weights[2];
  if (Math.abs(weightSum - 1) > epsilon && Math.fround(weightSum) !== 1) {
    throw new Error('SurfaceHit weights must sum to 1');
  }
  return [0, 1, 2].map(axis =>
    (weights[0] * coordinate(vertices[face[0]], axis)
      + weights[1] * coordinate(vertices[face[1]], axis)
      + weights[2] * coordinate(vertices[face[2]], axis)) / weightSum);
}

export function meshPlaneSection(geometry, firstCut = false) {
  const {hits, value_mm} = sectionRing(geometry, firstCut);
  return pathResult(hits, value_mm);
}

function sectionRing({
  vertices, faces, plane_origin, plane_normal, seed, allowed_faces = null,
  face_body_segment = null, body_segment = null, face_side = null, side = null,
  required_hits = [], required_hit_tolerance = null, allow_coplanar_edges = false, epsilon = EPSILON,
}, firstCut = false) {
  const origin = vector3(plane_origin, 'plane_origin');
  const normalInput = vector3(plane_normal, 'plane_normal');
  const normalLength = Math.hypot(...normalInput);
  if (normalLength <= epsilon) throw new Error('plane_normal must be non-zero');
  const normal = normalInput.map(value => value / normalLength);
  const seedPoint = firstCut ? null : reconstructSurfaceHit(vertices, faces, seed, epsilon);
  if (seedPoint && Math.abs(dot(subtract(seedPoint, origin), normal)) > epsilon) {
    throw new Error('section seed is not on the measurement plane');
  }

  const allowed = constrainedFaceFilter({
    faces, allowed_faces, face_body_segment, body_segment, face_side, side,
  });
  if (!firstCut && !allowed(seed.face_index)) throw new Error('section seed face is outside the segment/side filter');
  if (!Array.isArray(required_hits)) throw new Error('section required_hits must be an array');
  const hitTolerance = required_hit_tolerance == null ? epsilon : Number(required_hit_tolerance);
  if (!Number.isFinite(hitTolerance) || hitTolerance < epsilon) {
    throw new Error('section required_hit_tolerance must be finite and at least epsilon');
  }

  const nodes = new Map();
  const segments = [];
  const cutFaces = new Set();
  const coplanarEdges = new Set();
  faces.forEach((face, faceIndex) => {
    if (!allowed(faceIndex)) return;
    validateFace(face, vertices.length, faceIndex);
    const distances = face.map(vertex => dot(subtract(vertices[vertex], origin), normal));
    const onPlane = distances.flatMap((value, local) => Math.abs(value) <= epsilon ? [local] : []);
    if (onPlane.length === 3) throw new Error(`measurement plane contains face ${faceIndex}`);
    if (onPlane.length === 2) {
      if (!allow_coplanar_edges) throw new Error(`measurement plane contains an edge of face ${faceIndex}`);
      const edge = edgeKey(...onPlane.map(local => face[local]));
      if (coplanarEdges.has(edge)) return;
      coplanarEdges.add(edge);
    }
    const pair = triangleCuts(vertices, face, faceIndex, distances, epsilon);
    if (pair.length !== 2 || (onPlane.length < 2 && distance(pair[0].point, pair[1].point) <= epsilon)) return;
    const ends = pair.map(cut => {
      if (!nodes.has(cut.key)) nodes.set(cut.key, {...cut, segments: []});
      return nodes.get(cut.key);
    });
    const segment = {face_index: faceIndex, ends};
    for (const node of ends) node.segments.push(segment);
    segments.push(segment);
    cutFaces.add(faceIndex);
  });
  if (!segments.length) throw new Error('measurement plane does not cut the allowed mesh');

  const contains = ({ends: [a, b]}, point) =>
    pointSegmentDistance(point, a.point, b.point, 0) <= hitTolerance;
  const seeds = firstCut ? [segments[0]] : segments.filter(segment => segment.face_index === seed.face_index
    || (allow_coplanar_edges && contains(segment, seedPoint)));
  if (!seeds.length) throw new Error('section seed does not select one unique component');
  const loop = walkClosedSection(seeds[0]);
  if (seeds.some(segment => !loop.has(segment))) {
    throw new Error('section seed does not select one unique component');
  }
  const byFace = new Map([...loop.keys()].map(segment => [segment.face_index, segment]));
  function checkRequiredHit(hit, index) {
    const point = reconstructSurfaceHit(vertices, faces, hit, epsilon);
    if (Math.abs(dot(subtract(point, origin), normal)) > hitTolerance) {
      throw new Error(`section required hit ${index} is not on the measurement plane`);
    }
    if (!allowed(hit.face_index)) throw new Error(`section required hit ${index} is outside the segment/side filter`);
    const sameFace = byFace.get(hit.face_index);
    if (sameFace) {
      if (!contains(sameFace, point)) {
        throw new Error(`section required hit ${index} does not lie on the selected unique ring`);
      }
    } else if (cutFaces.has(hit.face_index) || ![...loop.keys()].some(segment => contains(segment, point))) {
      throw new Error(`section required hit ${index} is not on the selected unique ring`);
    }
    return point;
  }
  required_hits.forEach(checkRequiredHit);
  const hits = orderedSectionHits(loop);
  const points = hits.map(hit => reconstructSurfaceHit(vertices, faces, hit, epsilon));
  assertNoProjectedSelfIntersection(points, normal, epsilon);
  return {hits, points, value_mm: polylineLength(points, true), checkRequiredHit};
}

export function circumferenceFits(points, normal) {
  const project = planeProjection(normalize(vector3(normal, 'section normal')));
  const projected = points.map((point, index) => ({index, uv: project(point)}));
  const ordered = projected.slice().sort((a, b) => a.uv[0] - b.uv[0] || a.uv[1] - b.uv[1] || a.index - b.index);
  const turn = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const half = sequence => {
    const chain = [];
    for (const point of sequence) {
      while (chain.length >= 2 && turn(chain.at(-2).uv, chain.at(-1).uv, point.uv) <= EPSILON) chain.pop();
      chain.push(point);
    }
    return chain.slice(0, -1);
  };
  const hull = [...half(ordered), ...half(ordered.reverse())];
  if (hull.length < 3) throw new Error('section perimeter is degenerate');
  const indices = hull.map(point => point.index);
  return {indices, basic_mm: polylineLength(indices.map(index => points[index]), true),
    surface_mm: polylineLength(points, true)};
}

export function torsoCircumference({
  definition, vertices, faces, face_body_segment, body_up, joints, joint_names, landmark_hits,
  seed: suppliedSeed = null, fit = definition.circumference_fit,
}) {
  const placement = definition.section_placement;
  const segments = definition.body_segments;
  const up = normalize(vector3(body_up, 'body_up'));
  const joint = name => vector3(joints[joint_names.indexOf(name)], name);
  const pelvis = joint('pelvis');
  const right = normalize(subtract(joint('right_hip'), joint('left_hip')));
  const front = normalize(cross(up, right));
  const mean = points => points.reduce((a, b) => a.map((value, index) => value + b[index]), [0, 0, 0])
    .map(value => value / points.length);
  const landmark = group => mean(group.joints ? group.joints.map(joint)
    : group.vertices ? group.vertices.map(index => vector3(vertices[index], 'landmark vertex'))
      : group.hits.map(name => reconstructSurfaceHit(vertices, faces, landmark_hits[name])));
  const [start, end] = placement.bounds.map(landmark);
  const [lo, hi] = [dot(start, up), dot(end, up)];
  if (hi <= lo) throw new Error('anatomical section bounds are inverted');
  const axisAt = height => mix(start, end, (height - lo) / (hi - lo));

  const closedSection = seed => {
    const origin = reconstructSurfaceHit(vertices, faces, seed);
    const height = dot(origin, up);
    if (height < lo - 1e-6 || height > hi + 1e-6) throw new Error('落尺超出当前量项的解剖范围');
    const ring = sectionRing({
      vertices, faces, face_body_segment, body_segment: segments,
      plane_origin: origin, plane_normal: body_up, seed,
    });
    if (!ringContains(ring.points, axisAt(height), right, front)) throw new Error('尺线未围住目标躯干');
    return {seed, height, ...ring};
  };

  const seedAt = height => nearestPlaneSeed(vertices, faces, face_body_segment, new Set(segments),
    up, height, axisAt(height));
  let section;
  if (suppliedSeed || placement.at) {
    const seed = suppliedSeed || seedAt(dot(landmark(placement.at), up));
    if (!seed) throw new Error('解剖测位缺少可用截面');
    section = closedSection(seed);
  } else {
    const [first, last] = placement.scan_fraction || [0, 1];
    let best = -Infinity;
    for (let step = 0; step < 16; step += 1) {
      const height = lo + (hi - lo) * (first + (last - first) * step / 15);
      const seed = seedAt(height);
      if (!seed) continue;
      try {
        const candidate = closedSection(seed);
        const score = placement.select === 'narrowest' ? -candidate.value_mm
          : Math.max(...candidate.points.map(point => -dot(subtract(point, pelvis), front)));
        if (score > best) {
          section = candidate;
          best = score;
        }
      } catch (_error) {
        // A level without a valid closed torso section cannot define the circumference.
      }
    }
    if (!section) throw new Error('解剖范围内缺少有效闭合围线');
  }

  if (fit === 'surface') return {...pathResult(section.hits, section.value_mm), seed: section.seed};
  if (fit !== 'basic') throw new Error('unsupported circumference fit');
  const {indices, basic_mm} = circumferenceFits(section.points, up);
  const outline = indices.map(index => section.points[index]);
  const project = planeProjection(up);
  const ring = outline.map(project);
  const segmentsAllowed = new Set(segments);
  const obstructed = faces.some((face, faceIndex) => {
    if (segmentsAllowed.has(face_body_segment[faceIndex])) return false;
    const cut = planeFaceHits(vertices, face, faceIndex, up, section.height)
      .map(hit => project(reconstructSurfaceHit(vertices, faces, hit)));
    return cut.some((point, index) => ring.every((a, i) => {
      const b = ring[(i + 1) % ring.length];
      return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= -EPSILON;
    }) || ring.some((a, i) => segmentsIntersect(
      a, ring[(i + 1) % ring.length], point, cut[(index + 1) % cut.length], EPSILON,
    )));
  });
  if (obstructed) throw new Error('外绕尺线碰到量体范围以外的部位');
  return {...pathResult(indices.map(index => section.hits[index]), basic_mm), seed: section.seed};
}

export function totalCrotchPlaneSection({
  vertices, faces, front_hit, crotch_hit, back_hit, body_up,
  face_body_segment, body_segments, epsilon = EPSILON,
}) {
  const up = normalize(vector3(body_up, 'body_up'));
  const front = reconstructSurfaceHit(vertices, faces, front_hit);
  const back = reconstructSurfaceHit(vertices, faces, back_hit);
  const waistDirection = subtract(back, front);
  const normal = normalize(cross(up, waistDirection));
  const {hits, points: ringPoints} = sectionRing({
    vertices, faces, plane_origin: front, plane_normal: normal, seed: front_hit,
    required_hits: [front_hit, back_hit], allow_coplanar_edges: true, epsilon,
  });
  const resolvedCrotchHit = nearestRingSurfaceHit(vertices, faces, ringPoints, crotch_hit, epsilon);
  requireFaceSegments(faces, face_body_segment, 'total_crotch');
  const allowedSegments = new Set(body_segments);
  const arc = ringArcThroughHit(
    vertices, faces, hits, ringPoints, front_hit, resolvedCrotchHit, back_hit, epsilon,
  );
  const crotchIndex = arc.indexOf(resolvedCrotchHit);
  const crotchHeight = dot(reconstructSurfaceHit(vertices, faces, resolvedCrotchHit), up);
  const pathHits = bridgeCrotchSuffix(
    vertices, faces, arc, crotchIndex, up, crotchHeight, epsilon,
  );
  if (pathHits.some(hit => !allowedSegments.has(face_body_segment[hit.face_index]))) {
    throw new Error('total_crotch midsagittal path leaves the crotch body segments');
  }
  const points = pathHits.map(hit => reconstructSurfaceHit(vertices, faces, hit));
  const heights = points.map(point => dot(point, up));
  const pathCrotchIndex = pathHits.indexOf(resolvedCrotchHit);
  if (Math.abs(heights[0] - heights.at(-1)) > epsilon
      || pathCrotchIndex < 1 || pathCrotchIndex >= heights.length - 1
      || heights[pathCrotchIndex] >= Math.min(heights[0], heights.at(-1)) - epsilon
      || heights.some(height => height < crotchHeight - epsilon)) {
    throw new Error('total_crotch must descend from a level waist through the crotch and ascend to the waist');
  }
  return {...pathResult(pathHits, polylineLength(points, false)), resolved_crotch_hit: resolvedCrotchHit};
}

function nearestRingSurfaceHit(vertices, faces, points, intentHit, epsilon) {
  const intent = reconstructSurfaceHit(vertices, faces, intentHit);
  let best = null;
  points.forEach((start, index) => {
    const end = points[(index + 1) % points.length];
    const t = segmentFraction(intent, start, end);
    const offset = subtract(intent, mix(start, end, t));
    const distance2 = dot(offset, offset);
    if (!best || distance2 < best.distance2) best = {index, t, distance2};
  });
  return ringSegmentHit(vertices, faces, points[best.index], points[(best.index + 1) % points.length], best.t, epsilon);
}

function barycentricOnFace(vertices, faces, faceIndex, point, epsilon) {
  const [a, b, c] = faces[faceIndex].map(vertex => vertices[vertex]);
  const v0 = subtract(b, a);
  const v1 = subtract(c, a);
  const v2 = subtract(point, a);
  const d00 = dot(v0, v0);
  const d01 = dot(v0, v1);
  const d11 = dot(v1, v1);
  const denominator = d00 * d11 - d01 * d01;
  if (Math.abs(denominator) <= Number.EPSILON * Math.max(1, d00 * d11)) return null;
  const d20 = dot(v2, v0);
  const d21 = dot(v2, v1);
  const weights = [
    1 - (d11 * d20 - d01 * d21 + d00 * d21 - d01 * d20) / denominator,
    (d11 * d20 - d01 * d21) / denominator,
    (d00 * d21 - d01 * d20) / denominator,
  ];
  if (weights.some(weight => weight < -epsilon || weight > 1 + epsilon)) return null;
  const normalized = weights.map(weight => Math.max(0, Math.min(1, weight)));
  const sum = normalized.reduce((total, weight) => total + weight, 0);
  const result = normalized.map(weight => weight / sum);
  const hit = {face_index: faceIndex, weights: result};
  return distance(reconstructSurfaceHit(vertices, faces, hit), point) <= epsilon ? result : null;
}

function ringArcThroughHit(vertices, faces, ringHits, points, frontHit, crotchHit, backHit, epsilon) {
  const required = [frontHit, crotchHit, backHit];
  const placements = required.map((hit, requiredIndex) => {
    const point = reconstructSurfaceHit(vertices, faces, hit);
    const candidates = points.flatMap((start, index) => {
      const end = points[(index + 1) % points.length];
      const t = segmentFraction(point, start, end);
      return distance(point, mix(start, end, t)) <= epsilon ? [{index, t, hit, requiredIndex}] : [];
    });
    if (!candidates.length) throw new Error(`total_crotch required hit ${requiredIndex} is not on the section ring`);
    candidates.sort((a, b) => (a.t >= 1 - epsilon) - (b.t >= 1 - epsilon));
    return candidates[0];
  });
  const augmented = [];
  const positions = Array(required.length).fill(-1);
  ringHits.forEach((hit, index) => {
    const atStart = placements.filter(item => item.index === index && item.t <= epsilon);
    if (atStart.length > 1) throw new Error('total_crotch required hits collapse to one section point');
    augmented.push(atStart[0]?.hit || hit);
    if (atStart.length) positions[atStart[0].requiredIndex] = augmented.length - 1;
    const inside = placements.filter(item => item.index === index && item.t > epsilon && item.t < 1 - epsilon)
      .sort((a, b) => a.t - b.t);
    inside.forEach(item => {
      if (augmented.length && distance(
        reconstructSurfaceHit(vertices, faces, augmented.at(-1)),
        reconstructSurfaceHit(vertices, faces, item.hit),
      ) <= epsilon) throw new Error('total_crotch required hits collapse to one section point');
      augmented.push(item.hit);
      positions[item.requiredIndex] = augmented.length - 1;
    });
  });
  if (positions.some(index => index < 0)) throw new Error('total_crotch required hit placement is ambiguous');
  const arc = (start, end, step) => {
    const output = [];
    for (let index = start; ; index = (index + step + augmented.length) % augmented.length) {
      output.push(index);
      if (index === end) return output;
    }
  };
  const candidates = [arc(positions[0], positions[2], 1), arc(positions[0], positions[2], -1)]
    .filter(indices => indices.includes(positions[1]));
  if (candidates.length !== 1) {
    throw new Error('total_crotch crotch hit does not select one unique U-shaped arc');
  }
  return candidates[0].map(index => augmented[index]);
}

function bridgeCrotchSuffix(vertices, faces, arc, crotchIndex, up, height, epsilon) {
  const points = arc.map(hit => reconstructSurfaceHit(vertices, faces, hit));
  if (points.slice(0, crotchIndex).some(point => dot(point, up) < height - epsilon)) {
    throw new Error('total_crotch front arc descends below the crotch hit');
  }
  const low = points.flatMap((point, index) =>
    index > crotchIndex && dot(point, up) < height - epsilon ? [index] : []);
  if (!low.length) throw new Error('total_crotch section has no rear crotch-height bridge');
  const index = Math.max(...low);
  const bridge = levelHit(vertices, faces, arc, points, index, index + 1, up, height, epsilon);
  return [...arc.slice(0, crotchIndex + 1), bridge.hit, ...arc.slice(bridge.keepIndex)];
}

function levelHit(vertices, faces, arc, points, startIndex, endIndex, up, height, epsilon) {
  const startHeight = dot(points[startIndex], up) - height;
  const endHeight = dot(points[endIndex], up) - height;
  if (Math.abs(startHeight) <= epsilon) return {hit: arc[startIndex], keepIndex: startIndex};
  if (Math.abs(endHeight) <= epsilon) return {hit: arc[endIndex], keepIndex: endIndex + 1};
  if ((startHeight < 0) === (endHeight < 0)) {
    throw new Error('total_crotch low section interval has no level boundary');
  }
  return {
    hit: ringSegmentHit(
      vertices, faces, points[startIndex], points[endIndex], startHeight / (startHeight - endHeight), epsilon,
    ),
    keepIndex: endIndex,
  };
}

function ringSegmentHit(vertices, faces, start, end, t, epsilon) {
  const point = mix(start, end, t);
  for (let faceIndex = 0; faceIndex < faces.length; faceIndex += 1) {
    const startWeights = barycentricOnFace(vertices, faces, faceIndex, start, epsilon);
    const endWeights = barycentricOnFace(vertices, faces, faceIndex, end, epsilon);
    if (!startWeights || !endWeights) continue;
    const weights = startWeights.map((weight, index) => weight * (1 - t) + endWeights[index] * t);
    const hit = {face_index: faceIndex, weights};
    if (distance(reconstructSurfaceHit(vertices, faces, hit), point) <= epsilon) return hit;
  }
  throw new Error('total_crotch selected section segment has no triangle representation');
}

export function projectedLength(points, axis) {
  return Math.abs(dot(subtract(points.at(-1), points[0]), axis) / Math.hypot(...axis));
}

export function recomputeSurfaceTapeRecord(record, vertices, faces, tolerance_mm = 0.001, projection = null) {
  if (!record || record.units !== 'mm') throw new Error('SurfaceTapeRecord units must be mm');
  if (!['surface_path', 'mesh_section', 'straight'].includes(record.method)) {
    throw new Error('SurfaceTapeRecord method is invalid');
  }
  const minimumPoints = record.method === 'mesh_section' ? 3 : 2;
  if (!Array.isArray(record.path_faces) || !Array.isArray(record.path_weights)
      || record.path_faces.length !== record.path_weights.length || record.path_faces.length < minimumPoints) {
    throw new Error('SurfaceTapeRecord path_faces/path_weights are invalid');
  }
  const points = surfacePathHits(record).map(hit => reconstructSurfaceHit(vertices, faces, hit));
  const length = polylineLength(points, record.method === 'mesh_section');
  if (projection && Math.abs(length - projection.pathLengthMm) > tolerance_mm) {
    throw new Error('SurfaceTapeRecord saved path length differs from its rebuilt path');
  }
  const valueMm = projection ? projectedLength(points, projection.axis) : length;
  if (!Number.isFinite(record.value_mm) || Math.abs(valueMm - record.value_mm) > tolerance_mm) {
    throw new Error(`SurfaceTapeRecord value_mm mismatch: ${valueMm} != ${record.value_mm}`);
  }
  return valueMm;
}

function edgeKey(a, b) {
  return `e:${Math.min(a, b)}:${Math.max(a, b)}`;
}

function triangleCuts(vertices, face, faceIndex, distances, epsilon) {
  const cuts = new Map();
  const vertexCut = local => {
    const key = `v:${face[local]}`;
    cuts.set(key, {key, point: [...vertices[face[local]]],
      hit: {face_index: faceIndex, weights: [0, 1, 2].map(index => Number(index === local))}});
  };
  for (let a = 0; a < 3; a += 1) {
    const b = (a + 1) % 3;
    const da = distances[a], db = distances[b];
    const aOn = Math.abs(da) <= epsilon, bOn = Math.abs(db) <= epsilon;
    if (aOn) vertexCut(a);
    if (bOn) vertexCut(b);
    if (aOn || bOn || (da < 0) === (db < 0)) continue;
    const t = da / (da - db);
    const weights = [0, 0, 0];
    weights[a] = 1 - t;
    weights[b] = t;
    const key = edgeKey(face[a], face[b]);
    cuts.set(key, {key, point: mix(vertices[face[a]], vertices[face[b]], t),
      hit: {face_index: faceIndex, weights}});
  }
  return [...cuts.values()];
}

function walkClosedSection(seed) {
  const loop = new Map();
  let segment = seed;
  let start = seed.ends[0];
  while (!loop.has(segment)) {
    loop.set(segment, start);
    const [a, b] = segment.ends;
    start = start === a ? b : a;
    const [first, second] = start.segments;
    if (start.segments.length !== 2) throw new Error(`section component is not a degree-2 loop at ${start.key}`);
    segment = first === segment ? second : first;
  }
  if (segment !== seed) throw new Error('section component is not a single cycle');
  if (loop.size < 3) throw new Error('section component is a degenerate loop');
  return loop;
}

function orderedSectionHits(loop) {
  // Use the first triangle in mesh order and its original direction for saved paths.
  const segments = [...loop.keys()];
  const first = segments.reduce((a, b) => a.face_index < b.face_index ? a : b);
  const starts = [...loop.values()];
  const offset = segments.indexOf(first);
  if (starts[offset] === first.ends[0]) return [...starts.slice(offset), ...starts.slice(0, offset)].map(node => node.hit);
  const next = (offset + 1) % starts.length;
  return [...starts.slice(0, next + 1).reverse(), ...starts.slice(next + 1).reverse()].map(node => node.hit);
}

function planeProjection(normal) {
  const u = normalize(cross(normal, Math.abs(normal[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
  const v = cross(normal, u);
  return point => [dot(point, u), dot(point, v)];
}

function assertNoProjectedSelfIntersection(points, normal, epsilon) {
  const projected = points.map(planeProjection(normal));
  const area2 = projected.reduce((sum, point, index) => {
    const next = projected[(index + 1) % projected.length];
    return sum + point[0] * next[1] - next[0] * point[1];
  }, 0);
  if (Math.abs(area2) <= epsilon) throw new Error('section loop is degenerate after plane projection');
  for (let a = 0; a < projected.length; a += 1) {
    const aNext = (a + 1) % projected.length;
    for (let b = a + 1; b < projected.length; b += 1) {
      const bNext = (b + 1) % projected.length;
      if (a === b || aNext === b || bNext === a) continue;
      if (segmentsIntersect(projected[a], projected[aNext], projected[b], projected[bNext], epsilon)) {
        throw new Error('section loop self-intersects after plane projection');
      }
    }
  }
}

function segmentsIntersect(a, b, c, d, epsilon) {
  const orient = (p, q, r) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  const values = [orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)];
  const onSegment = (p, q, r) => q[0] >= Math.min(p[0], r[0]) - epsilon
    && q[0] <= Math.max(p[0], r[0]) + epsilon
    && q[1] >= Math.min(p[1], r[1]) - epsilon
    && q[1] <= Math.max(p[1], r[1]) + epsilon;
  if (Math.abs(values[0]) <= epsilon && onSegment(a, c, b)) return true;
  if (Math.abs(values[1]) <= epsilon && onSegment(a, d, b)) return true;
  if (Math.abs(values[2]) <= epsilon && onSegment(c, a, d)) return true;
  if (Math.abs(values[3]) <= epsilon && onSegment(c, b, d)) return true;
  return ((values[0] > epsilon && values[1] < -epsilon) || (values[0] < -epsilon && values[1] > epsilon))
    && ((values[2] > epsilon && values[3] < -epsilon) || (values[2] < -epsilon && values[3] > epsilon));
}

function constrainedFaceFilter({faces, allowed_faces, face_body_segment, body_segment, face_side, side}) {
  const allowed = faceFilter(allowed_faces, faces.length);
  const segments = valueSet(body_segment);
  if (segments && (!Array.isArray(face_body_segment) || face_body_segment.length !== faces.length)) {
    throw new Error('face_body_segment must match faces when body_segment is selected');
  }
  if (side && (!Array.isArray(face_side) || face_side.length !== faces.length)) {
    throw new Error('face_side must match faces when side is selected');
  }
  return index => allowed(index)
    && (!segments || segments.has(face_body_segment[index]))
    && (!side || face_side[index] === side);
}

function faceFilter(value, count) {
  if (value == null) return () => true;
  if (typeof value === 'function') return index => Boolean(value(index));
  const values = value instanceof Set ? value : new Set(value);
  return index => Number.isInteger(index) && index >= 0 && index < count && values.has(index);
}

function valueSet(value) {
  if (value == null) return null;
  if (value instanceof Set) return value;
  return new Set(Array.isArray(value) ? value : [value]);
}

function pathResult(hits, value_mm) {
  return {
    path_faces: hits.map(hit => hit.face_index),
    path_weights: hits.map(hit => [...hit.weights]),
    value_mm,
  };
}

function appendDistinctHit(output, hit, vertices, faces) {
  if (!hit) throw new Error('allowed face graph contains an unrepresentable vertex');
  if (!output.length || distance(
    reconstructSurfaceHit(vertices, faces, output[output.length - 1]),
    reconstructSurfaceHit(vertices, faces, hit),
  ) > EPSILON) output.push(hit);
}

function validateFace(face, vertexCount, faceIndex) {
  if (!Array.isArray(face) || face.length !== 3 || face.some(index =>
    !Number.isInteger(index) || index < 0 || index >= vertexCount)) {
    throw new Error(`face ${faceIndex} is not a valid triangle`);
  }
}

function coordinate(point, axis) {
  if (!Array.isArray(point) || point.length < 3 || !Number.isFinite(point[axis])) {
    throw new Error('vertices must contain finite xyz coordinates');
  }
  return point[axis];
}

function vector3(value, name) {
  if (!Array.isArray(value) || value.length !== 3 || value.some(item => !Number.isFinite(item))) {
    throw new Error(`${name} must be three finite coordinates`);
  }
  return [...value];
}

export function polylineLength(points, closed) {
  let total = 0;
  for (let index = 1; index < points.length; index += 1) total += distance(points[index - 1], points[index]);
  if (closed) total += distance(points[points.length - 1], points[0]);
  return total;
}

export function distance(a, b) { return Math.hypot(...subtract(a, b)); }
function segmentFraction(point, start, end, minLengthSquared = -1) {
  const edge = subtract(end, start);
  const lengthSquared = dot(edge, edge);
  if (lengthSquared <= minLengthSquared) return 0;
  return Math.max(0, Math.min(1, dot(subtract(point, start), edge) / lengthSquared));
}
function pointSegmentDistance(point, start, end, minLengthSquared = EPSILON) {
  return distance(point, mix(start, end, segmentFraction(point, start, end, minLengthSquared)));
}
function subtract(a, b) { return [0, 1, 2].map(axis => coordinate(a, axis) - coordinate(b, axis)); }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function mix(a, b, t) { return [0, 1, 2].map(axis => coordinate(a, axis) * (1 - t) + coordinate(b, axis) * t); }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function normalize(value) { const length = Math.hypot(...value); return value.map(item => item / length); }
