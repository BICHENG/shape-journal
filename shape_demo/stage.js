import * as THREE from 'three';

const atlasYaw = .44;
export const overviewYaw = atlasYaw;
export const isoElevation = .235;
export const station = new THREE.Vector3(1.15, 0, -.08);

const mix = THREE.MathUtils.lerp;
const ease = THREE.MathUtils.smoothstep;
const clipPoint = new THREE.Vector3();

export function frameVisibility(corners, camera) {
  let edge = 0;
  for (const corner of corners) {
    clipPoint.copy(corner).project(camera);
    if (Math.abs(clipPoint.z) > 1) return 0;
    edge = Math.max(edge, Math.abs(clipPoint.x), Math.abs(clipPoint.y));
  }
  return 1 - ease(edge, .97, 1);
}

export function projectedFrame(corners, camera, result) {
  result.left = result.bottom = Infinity;
  result.right = result.top = -Infinity;
  for (const corner of corners) {
    clipPoint.copy(corner).project(camera);
    result.left = Math.min(result.left, clipPoint.x);
    result.right = Math.max(result.right, clipPoint.x);
    result.bottom = Math.min(result.bottom, clipPoint.y);
    result.top = Math.max(result.top, clipPoint.y);
  }
  return result;
}

export function boxCorners(box, corners) {
  let i = 0;
  for (const x of [box.min.x, box.max.x])
    for (const y of [box.min.y, box.max.y])
      for (const z of [box.min.z, box.max.z]) corners[i++].set(x, y, z);
}

function subjectBounds(center, size) {
  const corners = [];
  for (const x of [-.52, .52]) for (const y of [-.5, .5]) for (const z of [-.38, .38]) {
    corners.push(center.clone().add(new THREE.Vector3(x, y * size, z)));
  }
  return corners;
}

export function groupFade(group) {
  const originals = new Map();
  group.traverse(object => {
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) if (material)
      originals.set(material, {opacity: material.opacity, transparent: material.transparent});
  });
  let previous = -1;
  return amount => {
    if (amount === previous) return;
    previous = amount;
    group.visible = amount > .002;
    for (const [material, original] of originals) {
      material.opacity = original.opacity * amount;
      material.transparent = original.transparent || amount < 1;
    }
  };
}

function pointCloud(count, joints) {
  const geometry = new THREE.BufferGeometry();
  const position = new THREE.BufferAttribute(new Float32Array(count * 3), 3).setUsage(THREE.DynamicDrawUsage);
  const strength = new THREE.BufferAttribute(new Float32Array(count), 1).setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', position);
  geometry.setAttribute('strength', strength);
  geometry.setAttribute('role', new THREE.BufferAttribute(Float32Array.from({length: count},
    (_, i) => i >= count - joints ? -1 : i * .61803398875 % 1), 1));
  const material = new THREE.ShaderMaterial({
    uniforms: {
      amount: {value: 1}, ink: {value: new THREE.Color('#c6e1e5')}, jointInk: {value: new THREE.Color('#f7a66f')},
      pixelRatio: {value: 1}, diameter: {value: 2}, retain: {value: 1},
    },
    transparent: true, depthWrite: false,
    vertexShader: `
      attribute float strength;
      attribute float role;
      uniform float pixelRatio, diameter, retain;
      varying float energy, joint, coverage, feather;
      void main() {
        energy = strength;
        joint = 1. - step(0., role);
        coverage = 1. - smoothstep(retain, retain + .07, role);
        gl_PointSize = pixelRatio * (diameter + joint * 1.35);
        feather = clamp(.48 / gl_PointSize, .06, .30);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.);
      }`,
    fragmentShader: `
      uniform float amount;
      uniform vec3 ink, jointInk;
      varying float energy, joint, coverage, feather;
      void main() {
        float roundInk = 1. - smoothstep(.44 - feather, .44 + feather, length(gl_PointCoord - .5));
        gl_FragColor = vec4(mix(ink, jointInk, joint), amount * coverage * roundInk * (.53 + .47 * energy));
        #include <colorspace_fragment>
      }`,
  });
  const cloud = new THREE.Points(geometry, material);
  cloud.frustumCulled = false;
  return cloud;
}

function pointEnergy(uncertainty) {
  return 1 / (1 + Math.max(0, uncertainty) / 380);
}

export function cloudEntrance(capture) {
  const {count, joints} = capture.evidence;
  const cloud = pointCloud(count, joints);
  const {position, strength} = cloud.geometry.attributes;
  const targets = new Float32Array(position.array.length);
  const starts = new Float32Array(position.array.length);
  cloud.name = 'measurement-cloud';
  cloud.renderOrder = 3;
  cloud.material.depthTest = false;
  const {smootherstep} = THREE.MathUtils;
  const {amount, diameter, pixelRatio} = cloud.material.uniforms;
  let previousPoints, previousProgress, seeded = false;
  const sample = new THREE.Vector3();
  return {
    cloud,
    seed(body) {
      seeded = !!body;
      previousProgress = previousPoints = undefined;
      if (!body) return;
      body.updateWorldMatrix(true, false);
      const vertices = body.geometry.attributes.position;
      for (let i = 0; i < count; i++) {
        sample.fromBufferAttribute(vertices, i * 557 % vertices.count)
          .applyMatrix4(body.matrixWorld).sub(capture.group.position).toArray(starts, i * 3);
      }
    },
    update(progress, opacity, density) {
      const {points} = capture.evidence;
      const moved = points !== previousPoints;
      if (moved) {
        previousPoints = points;
        const px = capture.root.x * .001, pz = capture.root.z * .001;
        for (let i = 0, j = 0, k = 0; i < count; i++, j += 3, k += 6) {
          const x = points[k + 2] * .001, y = points[k + 3] * .001, z = points[k + 4] * .001;
          const angle = (i * .61803398875 % 1) * Math.PI * 2;
          const spread = .045 + .085 * (i * .41421356237 % 1);
          targets[j] = x;
          targets[j + 1] = y;
          targets[j + 2] = z;
          if (!seeded) {
            starts[j] = x + (x - px) * .21 + Math.cos(angle) * spread;
            starts[j + 1] = y;
            starts[j + 2] = z + (z - pz) * .21 + Math.sin(angle) * spread;
          }
          strength.setX(i, pointEnergy(points[k + 5]));
        }
        strength.needsUpdate = true;
      }
      if (moved || progress !== previousProgress) {
        previousProgress = progress;
        if (progress === 1) position.array.set(targets);
        else for (let i = 0, j = 0; i < count; i++, j += 3) {
          const delay = (i * .61803398875 % 1) * .20;
          const travel = smootherstep(progress, delay, .85 + delay * .7);
          const rise = .056 * Math.sin(travel * Math.PI);
          const spread = seeded ? .21 * Math.sin(Math.PI * Math.min(1, progress * 1.45)) : 0;
          const angle = i * 2.399963;
          position.setXYZ(i, mix(starts[j], targets[j], travel) + Math.cos(angle) * spread,
            mix(starts[j + 1], targets[j + 1], travel) + rise,
            mix(starts[j + 2], targets[j + 2], travel) + Math.sin(angle) * spread);
        }
        position.needsUpdate = true;
      }
      amount.value = .96 * opacity;
      diameter.value = mix(3.2, 2.6, progress);
      pixelRatio.value = density;
    },
  };
}

export function buildStage(capture, source, selectLayer) {
  const group = new THREE.Group();
  group.name = 'inference-layers';
  const input = capture?.evidence;
  const fitShift = new THREE.Vector3();
  const bounds = capture ? [...capture.bounds] : [
    new THREE.Vector3(station.x - .64, -.06, station.z + .40),
    new THREE.Vector3(station.x + .64, 1.84, station.z - .37),
  ];
  const ray = new THREE.Raycaster();
  ray.params.Points.threshold = .055;
  const near = new THREE.Vector3(), far = new THREE.Vector3();
  const cloud = input && pointCloud(input.count, input.joints ?? 24);
  function pick(camera, x, y, widthOnScreen, heightOnScreen, objects) {
    const nx = x / widthOnScreen * 2 - 1, ny = 1 - y / heightOnScreen * 2;
    near.set(nx, ny, -1).unproject(camera);
    far.set(nx, ny, 1).unproject(camera);
    ray.set(near, far.sub(near).normalize());
    const targets = [...objects, cloud].filter(node => node?.visible);
    let node = ray.intersectObjects(targets, true)[0]?.object;
    while (node && !node.userData.pick) node = node.parent;
    node?.userData.pick?.();
  }
  if (!input) return {group, bounds, fitShift, pick, layout() {}, update() {}};

  const [fx, fy, cx, cy] = source.lens;
  const projection = capture.group.position.clone().add(capture.picture.position);
  const cloudCenter = new THREE.Vector3(), fitCenter = new THREE.Vector3(), cloudShift = new THREE.Vector3();
  const referenceRoot = capture.group.position.clone().addScaledVector(capture.root, .001);
  const planeX = new THREE.Vector3(Math.cos(atlasYaw), 0, -Math.sin(atlasYaw));
  const planeDepth = new THREE.Vector3(Math.sin(atlasYaw) * Math.cos(isoElevation), Math.sin(isoElevation), Math.cos(atlasYaw) * Math.cos(isoElevation));
  const planeY = new THREE.Vector3().crossVectors(planeDepth, planeX);
  const rawHalfWidth = capture.picture.geometry.parameters.width * planeX.x / 2;
  const belowRaw = Math.min(...capture.bounds.map(corner => corner.dot(planeY))) - projection.dot(planeY);
  const place = (point, x, y, depth) => point.copy(projection).addScaledVector(planeX, x)
    .addScaledVector(planeY, y).addScaledVector(planeDepth, depth);
  const shots = {
    source: {bounds: capture.bounds, yaw: .12, lift: .055},
    cloud: {yaw: .61, lift: .20},
    fit: {yaw: .74, lift: .19},
  };
  let portrait, layoutChanged = false;
  function layout(aspect) {
    const compact = aspect < 1.22;
    if (compact === portrait) return;
    portrait = compact;
    if (compact) {
      const row = belowRaw - 1.20;
      place(cloudCenter, -.73, row, .46);
      place(fitCenter, .73, row, .70);
    } else {
      const column = rawHalfWidth + .93;
      place(cloudCenter, column, -.10, .46);
      place(fitCenter, column + 1.43, -.10, .70);
    }
    shots.cloud.bounds = subjectBounds(cloudCenter, 1.93);
    shots.fit.bounds = subjectBounds(fitCenter, 1.93);
    cloudShift.copy(cloudCenter).sub(referenceRoot);
    fitShift.copy(fitCenter).sub(referenceRoot);
    bounds.splice(0, bounds.length, ...capture.bounds, ...shots.cloud.bounds, ...shots.fit.bounds);
    layoutChanged = true;
  }
  layout(2);
  cloud.name = 'nlf-3d';
  cloud.userData.pick = () => selectLayer('cloud');
  const boxPositions = new THREE.BufferAttribute(new Float32Array(12), 3);
  const boxGeometry = new THREE.BufferGeometry().setAttribute('position', boxPositions);
  const box = new THREE.LineLoop(boxGeometry, new THREE.LineBasicMaterial({
    color: '#f3af72', transparent: true, depthWrite: false,
  }));
  box.name = 'follow-box';
  box.frustumCulled = false;
  const cloudCorners = Array.from({length: 8}, () => new THREE.Vector3());
  const pointBounds = new THREE.Box3(), cloudFrame = new THREE.Box3();
  const cloudCoords = cloud.geometry.attributes.position, cloudStrength = cloud.geometry.attributes.strength;
  let lastFrame, lastTravel;

  function readEvidence() {
    const {points, count, camera} = input;
    const [x, y, w, h] = camera.slice(9, 13);
    const corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
    for (let index = 0; index < corners.length; index++) {
      const [u, v] = corners[index];
      boxPositions.setXYZ(index,
        capture.group.position.x + (u - cx) * source.plane_depth / fx,
        capture.group.position.y + (cy - v) * source.plane_depth / fy,
        projection.z + .010);
    }
    boxPositions.needsUpdate = true;
    for (let i = 0, k = 0; i < count; i++, k += 6) {
      cloudCoords.setXYZ(i, points[k + 2] * .001, points[k + 3] * .001, points[k + 4] * .001);
      cloudStrength.setX(i, pointEnergy(points[k + 5]));
    }
    cloudCoords.needsUpdate = cloudStrength.needsUpdate = true;
    pointBounds.setFromBufferAttribute(cloudCoords);
  }

  function update({travel, context, active, frame, camera, visible, pixelRatio, width}) {
    const fresh = frame !== lastFrame;
    if (fresh) {
      lastFrame = frame;
      readEvidence();
    }
    const cloudTravel = ease(travel, .24, .92);
    if (fresh || lastTravel !== travel || layoutChanged) {
      lastTravel = travel;
      layoutChanged = false;
      cloud.position.copy(capture.group.position).addScaledVector(cloudShift, 1 - cloudTravel);
      cloudFrame.copy(pointBounds).translate(cloud.position).expandByScalar(.026);
      boxCorners(cloudFrame, cloudCorners);
    }
    const uniforms = cloud.material.uniforms;
    const focus = active === 'cloud' ? 1 : cloudTravel;
    const aligned = active === 'cloud' ? 1 : active === 'source' ? 0 : .36;
    const unstacked = 1 - ease(travel, .93, 1);
    uniforms.amount.value = context * visible('cloud') * mix(.98, aligned, cloudTravel)
      * frameVisibility(cloudCorners, camera) * unstacked;
    uniforms.pixelRatio.value = pixelRatio;
    uniforms.diameter.value = mix(1.8, 2.75, focus);
    uniforms.retain.value = mix(THREE.MathUtils.clamp(width / 1420, .25, .72), 1, focus);
    cloud.visible = uniforms.amount.value > .003;
    box.material.opacity = context * visible('source') * frameVisibility(capture.bounds, camera)
      * mix(.92, active === 'source' ? 0 : .46, cloudTravel) * unstacked;
    box.visible = box.material.opacity > .003;
  }

  cloud.renderOrder = box.renderOrder = .6;
  capture.picture.userData.pick = () => selectLayer('source');
  group.add(cloud, box);
  return {group, bounds, fitShift, shots, pick, layout, update};
}
