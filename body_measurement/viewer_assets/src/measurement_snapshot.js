import { readFileSync } from 'node:fs';
import { createMeasurementRecords } from './measurement_records.js';
import { meshPlaneSection } from './surface_measurement.js';

try {
  const {landmark_section, payload, progress = null, validate_only} = JSON.parse(readFileSync(0, 'utf8'));
  let output;
  if (landmark_section) {
    output = meshPlaneSection(landmark_section, true);
  } else {
    const model = createMeasurementRecords(payload);
    const records = validate_only ? [] : null;
    const state = validate_only || progress !== null ? model.readWorkProgress(progress, records) : model.initialState();
    output = validate_only ? {passed: true, failures: [], checks: [
      {name: 'current_height_cm', value: state.currentHeightCm},
      {name: 'selected_pose', pose_id: state.selectedPose.id},
      ...records,
    ]} : {progress: model.workProgress(state), csv: model.measurementCsv(state)};
  }
  process.stdout.write(JSON.stringify(output));
} catch (error) {
  process.stderr.write(error.message);
  process.exitCode = 1;
}
