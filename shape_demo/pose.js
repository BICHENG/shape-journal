import * as THREE from 'three';

export function createPoseBridge({parents, rotations, influences, weights}, targetRoot) {
  const joints = () => parents.map(() => ({
    center: new THREE.Vector3(), offset: new THREE.Vector3(),
    rotation: new THREE.Quaternion(), world: new THREE.Quaternion(), inverse: new THREE.Matrix4(),
  }));
  const from = joints(), to = joints(), pose = joints();
  const fromSkin = parents.map(() => new THREE.Matrix4());
  const toSkin = parents.map(() => new THREE.Matrix4());
  const unit = new THREE.Vector3(1, 1, 1), axis = new THREE.Vector3();
  const fromCamera = new THREE.Matrix4(), toPelvis = new THREE.Matrix4()
    .makeTranslation(-targetRoot.x, -targetRoot.y, -targetRoot.z);
  const jointMatrix = new THREE.Matrix4(), inverseParent = new THREE.Quaternion();
  const rawVertex = new THREE.Vector3(), targetVertex = new THREE.Vector3();
  const sample = new THREE.Vector3(), vertex = new THREE.Vector3();

  function bind(joint, parent, angles, offset) {
    axis.fromArray(angles, offset);
    const angle = axis.length();
    joint.rotation.setFromAxisAngle(axis.normalize(), angle);
    joint.world.multiplyQuaternions(parent.world, joint.rotation);
    joint.offset.subVectors(joint.center, parent.center)
      .applyQuaternion(inverseParent.copy(parent.world).invert());
    joint.inverse.compose(joint.center, joint.world, unit).invert();
  }

  // Re-express the existing endpoints in each bone's coordinates, then rotate
  // the bones along the pose change. The saved meshes supply both endpoints.
  return function blend(capture, posture, raw, target, targetJoints, amount, output) {
    fromCamera.compose(capture.root, capture.orientation, unit).invert();
    for (let i = 1; i < parents.length; i++) {
      const parent = parents[i];
      from[i].center.fromArray(posture, i * 6 + 3).applyMatrix4(fromCamera);
      bind(from[i], from[parent], posture, i * 6);
      to[i].center.fromArray(targetJoints, i * 3);
      bind(to[i], to[parent], rotations, i * 3);

      const joint = pose[i], previous = pose[parent];
      joint.rotation.slerpQuaternions(from[i].rotation, to[i].rotation, amount);
      joint.world.multiplyQuaternions(previous.world, joint.rotation);
      joint.center.lerpVectors(from[i].offset, to[i].offset, amount)
        .applyQuaternion(previous.world).add(previous.center);
    }
    for (let i = 0; i < parents.length; i++) {
      const {center, world} = pose[i];
      jointMatrix.compose(center, world, unit);
      fromSkin[i].multiplyMatrices(jointMatrix, from[i].inverse).multiply(fromCamera);
      toSkin[i].multiplyMatrices(jointMatrix, to[i].inverse).multiply(toPelvis);
    }
    const remaining = 1 - amount;
    const weightsPerVertex = influences.length * 3 / raw.length;
    for (let i = 0, slot = 0; i < raw.length; i += 3) {
      rawVertex.fromArray(raw, i);
      targetVertex.fromArray(target, i);
      vertex.set(0, 0, 0);
      for (let k = 0; k < weightsPerVertex; k++, slot++) {
        const weight = weights[slot];
        if (!weight) continue;
        const bone = influences[slot];
        vertex.addScaledVector(sample.copy(rawVertex).applyMatrix4(fromSkin[bone]), weight * remaining);
        vertex.addScaledVector(sample.copy(targetVertex).applyMatrix4(toSkin[bone]), weight * amount);
      }
      vertex.toArray(output, i);
    }
  };
}
