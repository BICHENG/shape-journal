import * as THREE from 'three';
import {createPoseBridge} from './pose.js';

export function decode(value, Type) {
  return new Type(Uint8Array.from(atob(value), char => char.charCodeAt(0)).buffer);
}

function glassMaterial(wipeX) {
  return new THREE.ShaderMaterial({
    uniforms: {presence: {value: 1}, wipeX},
    transparent: true, depthWrite: false,
    vertexShader: `
      attribute float coverage;
      varying float surfaceAlpha;
      varying vec3 bodyNormal;
      varying vec2 screenUV;
      void main() {
        bodyNormal = normalMatrix * normal;
        surfaceAlpha = coverage;
        vec4 view = modelViewMatrix * vec4(position, 1.);
        gl_Position = projectionMatrix * view;
        screenUV = gl_Position.xy / gl_Position.w;
      }`,
    fragmentShader: `
      uniform float presence, wipeX;
      varying float surfaceAlpha;
      varying vec3 bodyNormal;
      varying vec2 screenUV;
      float softbox(vec2 uv) {
        return exp(-7. * abs(uv.x - .26)) * (.55 + .45 * cos(uv.y * 2.4));
      }
      void main() {
        if (gl_FragCoord.x < wipeX || surfaceAlpha <= .005) discard;
        vec3 n = normalize(bodyNormal);
        vec3 ray = refract(vec3(0., 0., -1.), n, .68);
        vec3 reflected = reflect(vec3(0., 0., -1.), n);
        vec2 uv = screenUV * .33;
        vec3 dispersion = vec3(softbox(uv + ray.xy * .75),
                                softbox(uv + ray.xy * .81),
                                softbox(uv + ray.xy * .87));
        float edge = pow(1. - clamp(n.z, 0., 1.), 2.35);
        float highlight = exp(-13. * abs(reflected.x + .28)) * smoothstep(-.36, .68, reflected.y);
        float fill = max(0., dot(n, normalize(vec3(.58, .40, .76))));
        vec3 glass = vec3(.055, .089, .107) + vec3(.19, .30, .35) * (fill * .48 + edge * .77);
        glass += vec3(.46, .60, .67) * dispersion * .30;
        glass += vec3(.79, .89, .92) * highlight * .44;
        float opacity = min(.93, .56 + edge * .24 + highlight * .25);
        gl_FragColor = vec4(glass, opacity * presence * surfaceAlpha);
        #include <colorspace_fragment>
      }`,
  });
}

export function createFigure(geometry, data, anchor, wipeX = {value: -1}) {
  const opaque = new Uint8Array(geometry.attributes.position.count).fill(255);
  geometry.setAttribute('coverage', new THREE.BufferAttribute(opaque, 1, true));
  const body = new THREE.Mesh(geometry, glassMaterial(wipeX));
  // Use the nearest surface before blending the glass to avoid translucent triangle seams.
  const depthMaterial = body.material.clone();
  depthMaterial.uniforms.wipeX = wipeX;
  depthMaterial.colorWrite = false;
  depthMaterial.depthWrite = true;
  const depth = new THREE.Mesh(geometry, depthMaterial);
  const wire = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color: '#a6bccc', wireframe: true, transparent: true, opacity: .073, depthWrite: false,
  }));
  wire.material.onBeforeCompile = shader => {
    shader.uniforms.wipeX = wipeX;
    shader.vertexShader = 'attribute float coverage;\nvarying float surfaceAlpha;\n' + shader.vertexShader.replace(
      '#include <begin_vertex>', '#include <begin_vertex>\nsurfaceAlpha = coverage;');
    shader.fragmentShader = 'uniform float wipeX;\nvarying float surfaceAlpha;\n' + shader.fragmentShader.replace(
      '#include <clipping_planes_fragment>',
      '#include <clipping_planes_fragment>\nif (gl_FragCoord.x < wipeX) discard;').replace(
      '#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.a *= surfaceAlpha;');
  };
  const passes = [depth, body, wire];
  for (const mesh of passes) {
    mesh.scale.setScalar(.001);
    mesh.frustumCulled = false;
  }
  body.name = 'body';
  depth.renderOrder = .9;
  body.renderOrder = 1;
  wire.renderOrder = 2;
  const model = new THREE.Group();
  model.add(...passes);

  const film = geometry.clone();
  const handoff = film.clone();
  const root = new THREE.Vector3(...data.pose_root);
  const standingRoot = root.clone().multiplyScalar(.001).add(anchor);
  const inverse = new THREE.Quaternion(), neutralRotation = new THREE.Quaternion(), sample = new THREE.Vector3();
  const rig = data.rig;
  const blendPose = rig && createPoseBridge({
    ...rig, influences: decode(rig.influences, Uint8Array), weights: decode(rig.weights, Float32Array),
  }, root);

  function place(record, capture, amount, joints = rig?.joints) {
    model.quaternion.identity();
    if (amount === 0) {
      model.position.copy(capture.group.position);
      for (const mesh of passes) mesh.geometry = film;
      return;
    }
    if (amount === 1) {
      model.position.copy(anchor);
      for (const mesh of passes) mesh.geometry = record;
      return;
    }
    const raw = film.attributes.position.array, neutral = record.attributes.position.array;
    const mixed = handoff.attributes.position;
    const posture = capture.posture;
    if (blendPose && posture) {
      blendPose(capture, posture, raw, neutral, joints, amount, mixed.array);
    } else {
      inverse.copy(capture.orientation).invert();
      for (let i = 0, index = 0; i < raw.length; i += 3, index++) {
        sample.fromArray(raw, i).sub(capture.root).applyQuaternion(inverse);
        mixed.setXYZ(index,
          THREE.MathUtils.lerp(sample.x, neutral[i] - root.x, amount),
          THREE.MathUtils.lerp(sample.y, neutral[i + 1] - root.y, amount),
          THREE.MathUtils.lerp(sample.z, neutral[i + 2] - root.z, amount));
      }
    }
    mixed.needsUpdate = true;
    handoff.computeVertexNormals();
    model.position.copy(capture.group.position).addScaledVector(capture.root, .001).lerp(standingRoot, amount);
    model.quaternion.slerpQuaternions(capture.orientation, neutralRotation, amount);
    for (const mesh of passes) mesh.geometry = handoff;
  }

  function paint(glass, edges) {
    body.material.uniforms.presence.value = glass;
    body.visible = glass > .003;
    wire.material.opacity = edges;
    model.visible = body.visible || edges > .003;
  }

  return {body, model, film, place, paint};
}

export function blendCameras() {
  const camera = new THREE.Camera();
  const eye = new THREE.Vector3(), rawEye = new THREE.Vector3();
  const center = new THREE.Vector3(), focus = new THREE.Vector3();
  return (capture, orbit, gaze, travel, approach) => {
    if (travel === 0) return orbit;
    const raw = capture.camera;
    if (travel === 1) return raw;
    center.copy(capture.group.position).addScaledVector(capture.root, .001);
    camera.quaternion.slerpQuaternions(orbit.quaternion, raw.quaternion, travel);
    eye.copy(gaze).applyMatrix4(orbit.matrixWorldInverse);
    rawEye.copy(center).applyMatrix4(raw.matrixWorldInverse);
    focus.copy(gaze).lerp(center, approach);
    camera.position.copy(eye.lerp(rawEye, travel)).applyQuaternion(camera.quaternion).negate().add(focus);
    const a = orbit.projectionMatrix.elements, b = raw.projectionMatrix.elements;
    const perspective = travel / Math.max(.6, -eye.z);
    const p = camera.projectionMatrix.elements;
    for (let i = 0; i < 16; i++) p[i] = a[i] * (1 - travel) + b[i] * perspective;
    // Match the scale at the tracked pelvis during the lens change.
    const scale = THREE.MathUtils.lerp(a[5], b[5] / Math.max(.6, -rawEye.z), travel) / p[5];
    p[0] *= scale;
    p[5] *= scale;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    camera.updateMatrixWorld();
    return camera;
  };
}
