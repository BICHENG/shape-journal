import * as THREE from 'three';
import {decode} from './figure.js';

export function createSource(source, mesh, location, redraw, onFrame, onEnd, pixelLimit = Infinity) {
  const still = decode(source.positions, Float32Array);
  const lifecycle = new AbortController();
  const {signal} = lifecycle;
  const [fx, fy, cx, cy] = source.lens;
  const {width, height, plane_depth: depth} = source;
  const near = 0.04;
  const lens = new THREE.Matrix4().makePerspective(
    -cx * near / fx, (width - cx) * near / fx, cy * near / fy, -(height - cy) * near / fy, near, 40,
  );
  const letterbox = new THREE.Matrix4(), camera = new THREE.PerspectiveCamera();
  const group = new THREE.Group();
  group.position.copy(location).add(new THREE.Vector3(...source.origin));
  camera.position.copy(group.position);
  camera.updateMatrixWorld();

  const pixels = document.createElement('canvas');
  const resolution = Math.min(1, pixelLimit / Math.max(width, height));
  pixels.width = Math.max(1, Math.round(width * resolution));
  pixels.height = Math.max(1, Math.round(height * resolution));
  const context = pixels.getContext('2d', {alpha: false});
  if (!context) throw new Error('Could not read source frames');
  const texture = new THREE.CanvasTexture(pixels);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  const w = width * depth / fx, h = height * depth / fy;
  const picture = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({
    map: texture, transparent: true, depthWrite: false,
    side: THREE.DoubleSide, toneMapped: false,
  }));
  picture.name = 'source-plane';
  picture.position.set((width / 2 - cx) * depth / fx, (cy - height / 2) * depth / fy, -depth);
  picture.visible = false;
  picture.renderOrder = -1;
  const corners = [[-1, 1], [1, 1], [1, -1], [-1, -1]].map(([x, y]) =>
    new THREE.Vector3(x * w / 2, y * h / 2, 0).add(picture.position));
  const guides = new THREE.Group();
  const frameMaterial = new THREE.LineBasicMaterial({color: '#615e53', transparent: true, opacity: .55});
  const wire = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(corners), frameMaterial);
  const rays = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(corners.flatMap(point => [new THREE.Vector3(), point])),
    new THREE.LineBasicMaterial({color: '#6d7165', transparent: true, opacity: .29, depthWrite: false}),
  );
  const housing = new THREE.Mesh(new THREE.BoxGeometry(.22, .12, .12), new THREE.MeshBasicMaterial({color: '#4b514a'}));
  housing.position.z = .074;
  const barrel = new THREE.Mesh(new THREE.CylinderGeometry(.047, .064, .06, 16), new THREE.MeshBasicMaterial({color: '#777c6f'}));
  barrel.rotation.x = Math.PI / 2;
  barrel.position.z = .015;
  const finder = new THREE.Group();
  finder.name = 'capture-camera-marker';
  finder.add(housing, barrel);
  guides.add(wire, rays);
  group.add(picture, guides, finder);

  let frame = null, pending, video, played = false, sequence = 0, request = null;
  const evidence = source.pipeline && {...source.pipeline, points: decode(source.pipeline.points, Int16Array)};
  const postures = source.postures && decode(source.postures, Float32Array);
  const cameraStride = evidence ? 14 : 0;
  const pointStride = evidence ? evidence.count * 6 : 0;
  const postureStride = (evidence?.joints ?? 24) * 6;
  let seekTo = source.frame - source.first;
  const seekTime = index => (index + 0.25) / source.fps;
  const root = new THREE.Vector3(), axis = new THREE.Vector3(), orientation = new THREE.Quaternion();
  const cameraAxes = new THREE.Quaternion(1, 0, 0, 0);
  const rotation = new THREE.Quaternion();

  function place(vertices, offset, scale, rig, rigOffset = 0) {
    root.fromArray(rig, rigOffset).multiplyScalar(1000);
    axis.fromArray(rig, rigOffset + 3);
    const angle = axis.length();
    rotation.setFromAxisAngle(axis.normalize(), angle).premultiply(cameraAxes);
    const sign = orientation.dot(rotation) < 0 ? -1 : 1;
    orientation.set(sign * rotation.x, sign * rotation.y, sign * rotation.z, sign * rotation.w);
    const position = mesh.attributes.position;
    for (let i = 0; i < still.length; i++) position.array[i] = vertices[offset + i] * scale;
    position.needsUpdate = true;
    mesh.computeVertexNormals();
  }

  place(still, 0, 1000, source.pose_root);
  const image = new Image();
  image.addEventListener('load', () => {
    if (frame !== null) return;
    context.drawImage(image, 0, 0, pixels.width, pixels.height);
    texture.needsUpdate = picture.visible = true;
    redraw();
  }, {once: true, signal});
  image.src = source.image;

  function fit(widthOnStage, heightOnStage) {
    const scale = Math.min(widthOnStage / width, heightOnStage / height) * .96;
    letterbox.makeScale(width * scale / widthOnStage, height * scale / heightOnStage, 1);
    camera.projectionMatrix.multiplyMatrices(letterbox, lens);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }


  async function prepare() {
    const response = await fetch(source.motion, {signal});
    if (!response.ok) throw new Error('Aligned source unavailable');
    const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
    const packet = await new Response(stream).arrayBuffer();
    if (signal.aborted) throw signal.reason;
    const count = new DataView(packet).getUint32(0, true);
    const rootOffset = 4 + count * still.length * 2;
    const cameraOffset = rootOffset + count * 6 * 4;
    const pointOffset = cameraOffset + count * cameraStride * 4;
    const movieOffset = pointOffset + count * pointStride * 2;
    if (count !== source.count || packet.byteLength <= movieOffset) throw new Error('Invalid aligned source');
    const vertices = new Int16Array(packet, 4, count * still.length);
    for (let i = still.length; i < vertices.length; i++) vertices[i] += vertices[i - still.length];
    const cameras = new Float32Array(packet, cameraOffset, count * cameraStride);
    const points = new Int16Array(packet, pointOffset, count * pointStride);
    for (let i = pointStride; i < points.length; i++) points[i] += points[i - pointStride];
    const clip = {vertices, roots: new Float32Array(packet, rootOffset, count * 6), cameras, points, count};
    const media = document.createElement('video');
    media.muted = true;
    media.playsInline = true;
    media.preload = 'auto';
    media.addEventListener('seeked', () => {
      if (media === video && !played && seekTo !== null && !video.seeking
        && Math.floor(video.currentTime * source.fps + 0.001) === seekTo) present(clip, seekTo);
    }, {signal});
    media.addEventListener('ended', () => {
      if (media !== video || !media.ended || !played) return;
      if (frame !== source.first + count - 1) present(clip, count - 1);
      pause();
      onEnd();
    }, {signal});
    video = media;
    await new Promise((resolve, reject) => {
      media.addEventListener('loadeddata', resolve, {once: true, signal});
      media.addEventListener('error', () => reject(new Error('Source video could not be decoded')), {once: true, signal});
      signal.addEventListener('abort', () => reject(signal.reason), {once: true});
      media.src = URL.createObjectURL(new Blob([new Uint8Array(packet, movieOffset)], {type: 'video/mp4'}));
    });
    return clip;
  }

  function present({vertices, roots, cameras, points}, index) {
    context.drawImage(video, 0, 0, pixels.width, pixels.height);
    place(vertices, index * still.length, 1, roots, index * 6);
    if (evidence) {
      evidence.camera = cameras.subarray(index * cameraStride, (index + 1) * cameraStride);
      evidence.points = points.subarray(index * pointStride, (index + 1) * pointStride);
    }
    texture.needsUpdate = picture.visible = true;
    frame = source.first + index;
    if (seekTo === index) seekTo = null;
    onFrame(index);
    redraw();
  }

  function failed(token, error) {
    if (token !== sequence) return;
    pause();
    releaseVideo();
    pending = null;
    onEnd(true);
    console.warn('Aligned source unavailable', error);
  }

  function ready() {
    return pending ??= prepare();
  }

  async function seek(index) {
    pause();
    seekTo = index;
    const token = ++sequence;
    try {
      const clip = await ready();
      if (token !== sequence) return;
      if (!video.seeking && video.readyState >= 2 && Math.abs(video.currentTime - seekTime(index)) < 0.001) {
        present(clip, index);
      } else {
        video.currentTime = seekTime(index);
      }
    } catch (error) {
      failed(token, error);
    }
  }

  async function play(from) {
    if (played) return;
    if (Number.isInteger(from)) seekTo = from;
    played = true;
    const token = ++sequence;
    try {
      const clip = await ready();
      if (token !== sequence) return;
      if (seekTo !== null) {
        video.currentTime = seekTime(seekTo);
        seekTo = null;
      } else if (video.ended || frame === source.first + clip.count - 1) video.currentTime = 0;
      else {
        const displayed = (frame ?? source.frame) - source.first;
        if (Math.floor(video.currentTime * source.fps + .001) !== displayed) {
          video.currentTime = seekTime(displayed);
        }
      }
      await video.play();
      if (token !== sequence) return;
      function decoded(_, info) {
        if (token !== sequence) return;
        const index = Math.min(clip.count - 1, Math.max(0, Math.round(info.mediaTime * source.fps)));
        request = video.requestVideoFrameCallback(decoded);
        present(clip, index);
      }
      request = video.requestVideoFrameCallback(decoded);
    } catch (error) {
      failed(token, error);
    }
  }

  function pause() {
    if (!played) return;
    played = false;
    sequence++;
    if (!video) return;
    video.pause();
    if (request !== null) video.cancelVideoFrameCallback(request);
    request = null;
  }

  function freeze() {
    pause();
    seekTo = null;
    sequence++;
  }

  function releaseVideo() {
    if (!video) return;
    const url = video.src;
    video.removeAttribute('src');
    video.load();
    video = null;
    URL.revokeObjectURL(url);
  }

  function dispose() {
    pause();
    sequence++;
    lifecycle.abort();
    releaseVideo();
    texture.dispose();
  }

  return {
    group, picture, guides, finder, camera, root, orientation, evidence, fit, ready, seek, play, pause, freeze, dispose,
    bounds: corners.map(point => point.clone().add(group.position)),
    get frame() { return frame; },
    get advancing() { return played && request !== null && frame !== null; },
    get posture() {
      const start = ((frame ?? source.frame) - source.first) * postureStride;
      return postures?.subarray(start, start + postureStride);
    },
  };
}
