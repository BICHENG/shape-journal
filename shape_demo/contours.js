import * as THREE from 'three';

export function createContours(renderer, anchor) {
  const mask = new THREE.WebGLRenderTarget(1, 1, {depthBuffer: false});
  const size = new THREE.Vector2();
  const silhouettes = new THREE.Scene();

  function addSilhouette(channel) {
    const mat = new THREE.ShaderMaterial({
      uniforms: {channel: {value: channel}},
      transparent: true, depthTest: false, depthWrite: false,
      blending: THREE.CustomBlending, blendEquation: THREE.MaxEquation,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor,
      vertexShader: /* glsl */ `
        attribute float coverage;
        varying float surfaceAlpha;
        void main() {
          surfaceAlpha = coverage;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 channel;
        varying float surfaceAlpha;
        void main() {
          gl_FragColor = vec4(channel * surfaceAlpha, 1.);
        }`,
    });
    const mesh = new THREE.Mesh(undefined, mat);
    mesh.position.copy(anchor);
    mesh.scale.setScalar(.001);
    mesh.frustumCulled = false;
    silhouettes.add(mesh);
    return mesh;
  }
  const reference = addSilhouette(new THREE.Vector3(1, 0, 0));
  const current = addSilhouette(new THREE.Vector3(0, 1, 0));

  const overlay = new THREE.Scene(), screen = new THREE.Camera();
  const palette = getComputedStyle(renderer.domElement);
  const color = name => new THREE.Color(palette.getPropertyValue(name).trim());
  const material = new THREE.ShaderMaterial({
    uniforms: {
      mask: {value: mask.texture}, size: {value: size},
      startColor: {value: color('--outline-start')}, currentColor: {value: color('--outline-current')},
    },
    transparent: true, depthTest: false, depthWrite: false,
    vertexShader: /* glsl */ `
      varying vec2 coords;
      void main() {
        coords = uv;
        gl_Position = vec4(position, 1.);
      }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D mask;
      uniform vec2 size;
      uniform vec3 startColor, currentColor;
      varying vec2 coords;
      void main() {
        vec2 stepX = vec2(1.4 / size.x, 0.);
        vec2 edges = abs(texture2D(mask, coords + stepX).rg - texture2D(mask, coords - stepX).rg);
        float baseline = smoothstep(.42, .86, edges.r) * .98;
        float selected = smoothstep(.42, .86, edges.g) * .98;
        float alpha = selected + baseline * (1. - selected);
        if (alpha < .01) discard;
        vec3 color = (startColor * baseline * (1. - selected) + currentColor * selected) / alpha;
        gl_FragColor = vec4(color, alpha);
        #include <colorspace_fragment>
      }`,
  });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  overlay.add(quad);

  return (camera, start, selected) => {
    renderer.getSize(size);
    if (mask.width !== size.x || mask.height !== size.y) mask.setSize(size.x, size.y);
    reference.geometry = start;
    current.geometry = selected;
    renderer.setRenderTarget(mask);
    renderer.render(silhouettes, camera);
    renderer.setRenderTarget(null);
    renderer.autoClear = false;
    renderer.render(overlay, screen);
    renderer.autoClear = true;
  };
}
