import {Line2} from 'three/addons/lines/Line2.js';
import {LineGeometry} from 'three/addons/lines/LineGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';
import {decode} from './figure.js';
import {
  automaticLowerBodyDraft, limbAxisCircumference, meshPlaneSection, reconstructSurfaceHit,
  surfacePathHits, torsoCircumference,
} from '../body_measurement/viewer_assets/src/surface_measurement.js';

const regions = {
  abdomen: {item: 'natural_waist', name: 'Natural waist', side: 'center'},
  legs: {item: 'calf', name: 'Left calf', side: 'left'},
  arms: {item: 'upper_arm', name: 'Left upper arm', side: 'left'},
};

function triples(values, component) {
  return Array.from({length: values.length / 3}, (_, index) =>
    [0, 1, 2].map(axis => component(values[index * 3 + axis], index * 3 + axis, axis)));
}

export function createTapes(scene, renderer, station, data, base, indices) {
  const mesh = data.tape;
  const faces = triples(indices, value => value);
  const faceSegments = Array.from(decode(mesh.face_segments, Uint8Array), id => mesh.segment_ids[id]);
  const palette = getComputedStyle(renderer.domElement);
  const lines = ['--outline-start', '--outline-current'].map((name, index) => {
    const line = new Line2(new LineGeometry(), new LineMaterial({
      color: palette.getPropertyValue(name).trim(), linewidth: index === 0 ? 4 : 2,
      depthTest: false, depthWrite: false, transparent: true, opacity: .93,
    }));
    line.name = index === 0 ? 'tape-start' : 'tape-current';
    line.position.copy(station);
    line.scale.setScalar(.001);
    line.frustumCulled = false;
    line.renderOrder = 6 + index;
    line.visible = false;
    scene.add(line);
    return line;
  });
  const output = document.getElementById('measurement-readout');
  const name = document.getElementById('measurement-name');
  const values = ['measurement-start', 'measurement-current'].map(id => document.getElementById(id));
  const cache = new Map();
  let visibleKey = null;

  function measure(region, record) {
    const key = region.item + ':' + record.key;
    if (cache.has(key)) return cache.get(key);
    const definition = mesh.definitions[region.item];
    const vertices = triples(base, (value, i) => Math.fround(value + (record.delta?.[i] ?? 0)));
    const joints = triples(record.joints ?? data.rig.joints, (value, i, axis) => value + data.pose_root[axis]);
    const geometry = {vertices, faces, face_body_segment: faceSegments, joints, definition,
      body_up: mesh.up, joint_names: mesh.joint_names, landmark_hits: mesh.landmark_hits, side: region.side};
    let result = null;
    try {
      let ring;
      if (definition.section_placement) {
        const hits = definition.section_placement.bounds.flatMap(bound => bound.hits || []);
        if (hits.every(id => mesh.landmark_hits[id])) ring = torsoCircumference(geometry);
      }
      else if (definition.plane_normal === 'limb_axis') ring = limbAxisCircumference(geometry);
      else {
        const draft = automaticLowerBodyDraft(geometry);
        if (draft) ring = meshPlaneSection({vertices, faces, face_body_segment: faceSegments,
          body_segment: definition.body_segments.map(segment => segment.replace('{side}', region.side)),
          plane_origin: reconstructSurfaceHit(vertices, faces, draft.seed), plane_normal: mesh.up, seed: draft.seed});
      }
      if (ring) {
        const points = surfacePathHits(ring).map(hit => reconstructSurfaceHit(vertices, faces, hit));
        points.push(points[0]);
        result = {path: points.flat(), cm: (ring.value_mm / 10).toFixed(1)};
      }
    } catch (error) {
      console.warn('Tape could not close on this record', region.item, record.key, error.message);
    }
    cache.set(key, result);
    return result;
  }

  return {
    show(part, start, current) {
      const key = part && part + ':' + current.key;
      if (key === visibleKey) return;
      visibleKey = key;
      const region = regions[part];
      const tapes = region ? [measure(region, start), measure(region, current)] : [];
      const valid = tapes.length === lines.length && tapes.every(Boolean);
      output.hidden = !region;
      lines.forEach((line, i) => {
        line.visible = valid;
        if (valid) {
          line.geometry.dispose();
          line.geometry = new LineGeometry().setPositions(tapes[i].path);
        }
        values[i].textContent = valid ? tapes[i].cm : '—';
        const record = i ? current : start;
        values[i].title = record.title ?? record.label;
      });
      if (region) name.textContent = region.name;
    },
  };
}
