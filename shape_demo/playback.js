import {MathUtils} from 'three';
import {cloudEntrance} from './stage.js';

const {smoothstep} = MathUtils;

export function advance(motion, dt, rate, instant = false) {
  if (motion.value === motion.target && motion.velocity === 0) return;
  if (instant) {
    motion.value = motion.target;
    motion.velocity = 0;
    return;
  }
  const error = motion.value - motion.target;
  const drift = motion.velocity + rate * error;
  const decay = Math.exp(-rate * dt);
  const value = motion.target + (error + drift * dt) * decay;
  motion.value = MathUtils.clamp(value, 0, 1);
  motion.velocity = motion.value === value ? (motion.velocity - rate * drift * dt) * decay : 0;
  if (Math.abs(motion.value - motion.target) + Math.abs(motion.velocity / rate) < .001) {
    motion.value = motion.target;
    motion.velocity = 0;
  }
}

export const TAIL_LEAD_SECONDS = .38;
export const STANCE_SECONDS = 1.75;

const CLOUD_END_SECONDS = 2.10;

export function stanceMotion(value, instant) {
  let from = value, target = value, began = 0, duration = 1;
  function update(now) {
    value = MathUtils.lerp(from, target, MathUtils.smootherstep(now - began, 0, duration));
    return value;
  }
  function jump(next) {
    from = target = value = next;
  }
  return {
    get value() { return value; },
    get target() { return target; },
    set target(next) {
      if (next === target) return;
      if (instant) {
        jump(next);
        return;
      }
      began = performance.now();
      from = value;
      target = next;
      duration = next === 1 ? STANCE_SECONDS * 1000 : 650;
    },
    update,
    jump,
  };
}

export function playbackEffect(capture, reducedMotion) {
  const dots = capture.evidence && cloudEntrance(capture);
  let age = 0;
  let started = false;
  if (dots) {
    dots.cloud.visible = false;
    capture.group.add(dots.cloud);
  }
  function start(body) {
    age = 0;
    started = true;
    dots?.seed(body);
  }
  function update(dt, playing, standing, density) {
    if (playing) age = Math.min(CLOUD_END_SECONDS, age + dt);
    const phase = reducedMotion || !dots ? CLOUD_END_SECONDS : age;
    const glass = started ? .66 * (.34 + .66 * smoothstep(phase, .07, .46)) : 0;
    if (dots) {
      const hold = 1 - .48 * smoothstep(phase, .34, .65);
      const fade = 1 - smoothstep(phase, 1.42, CLOUD_END_SECONDS);
      const opacity = started && !reducedMotion
        ? hold * fade * (playing ? 1 : 1 - smoothstep(standing, .04, .61)) : 0;
      dots.cloud.visible = opacity > .003;
      if (dots.cloud.visible) {
        dots.update(Math.min(1, phase / .56), opacity, density);
      }
    }
    return glass;
  }
  function clear() {
    started = false;
    if (dots) dots.cloud.visible = false;
  }
  return {start, clear, update, get animating() { return !!dots && !reducedMotion && started && age < CLOUD_END_SECONDS; }};
}
