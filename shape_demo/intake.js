const stepLabels = {
  upload: 'Uploading',
  detection: 'Finding the person',
  pose: 'Reading pose',
  body: 'Building the 3D shape',
  alignment: 'Aligning the video',
  saving: 'Saving the visit',
};

export function connectIntake(app) {
  if (!app) return;
  const get = id => document.getElementById(id);
const clipCount = count => count + (count === 1 ? ' clip' : ' clips');
  const dialog = get('intake');
  const form = get('capture-form');
  const open = get('open-intake');
  const link = get('archive-link');
  const picker = get('pick-files');
  const input = get('video-files');
  const visits = get('visit-select');
  const day = get('capture-date');
  const device = get('capture-device');
  const height = get('capture-height');
  const send = get('send-capture');
  const progress = get('capture-progress');
  const status = get('capture-status');
  let files = [], busy = false, request = null, timer = null;

  open.hidden = link.hidden = false;
  if (app.view === 'archive') {
    link.textContent = 'Library';
    link.href = '/';
  }
  get('visit-field').hidden = !app.visits.length;
  get('height-field').hidden = !!app.visits.length;
  height.value = app.height_cm;
  const now = new Date();
  day.value = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  const lastDevice = app.visits.at(-1)?.device;
  const defaultDevice = '默认拍摄';
  device.value = lastDevice && lastDevice !== defaultDevice ? lastDevice : '';
  for (const visit of app.visits) {
    const camera = visit.device === defaultDevice ? 'Default camera' : visit.device;
    visits.add(new Option(visit.date + ' · ' + camera + ' · ' + clipCount(visit.count), visit.key));
  }

  function syncFields() {
    get('new-fields').hidden = !!visits.value;
    visits.disabled = picker.disabled = busy;
    day.disabled = device.disabled = busy || !!visits.value;
    height.disabled = busy || !!app.visits.length;
    send.disabled = busy || !files.length;
  }

  function message(text, error = false) {
    status.textContent = text;
    status.dataset.error = String(error);
  }

  function setBusy(value) {
    busy = value;
    progress.hidden = !busy;
    open.textContent = busy ? 'Processing' : 'Add Video';
    form.setAttribute('aria-busy', String(busy));
    syncFields();
  }

  function selectFiles(list) {
    files = Array.from(list);
    get('file-list').replaceChildren(...files.map(file => {
      const row = document.createElement('li');
      row.textContent = file.name;
      return row;
    }));
    message(files.length ? clipCount(files.length) + ' selected' : '');
    syncFields();
  }

  function openDialog() {
    dialog.showModal();
  }

  async function readProgress() {
    try {
      const response = await fetch('/api/status', {cache: 'no-store'});
      const state = await response.json();
      if (state.step === 'idle') {
        if (!request && busy) {
          setBusy(false);
          message('View the saved visit in My Log.');
        }
        return;
      }
      setBusy(true);
      progress.removeAttribute('value');
      const clip = state.total > 1 ? 'Clip ' + state.clip + ' of ' + state.total + ' · ' : '';
      message(clip + stepLabels[state.step]);
      timer = setTimeout(readProgress, 650);
    } catch {
      if (request) timer = setTimeout(readProgress, 1200);
    }
  }

  open.addEventListener('click', openDialog);
  get('close-intake').addEventListener('click', () => dialog.close());
  visits.addEventListener('change', syncFields);
  picker.addEventListener('click', () => input.click());
  input.addEventListener('change', () => selectFiles(input.files));
  picker.addEventListener('dragover', event => {
    event.preventDefault();
    picker.dataset.drop = String(!busy);
  });
  picker.addEventListener('dragleave', () => delete picker.dataset.drop);
  picker.addEventListener('drop', event => {
    event.preventDefault();
    delete picker.dataset.drop;
    if (!busy) selectFiles(event.dataTransfer.files);
  });

  form.addEventListener('submit', event => {
    event.preventDefault();
    if (busy || !files.length) return;
    const data = new FormData(form);
    for (const file of files) data.append('videos', file);
    const xhr = new XMLHttpRequest();
    request = xhr;
    xhr.open('POST', '/api/captures');
    xhr.responseType = 'json';
    setBusy(true);
    message('Uploading');
    xhr.upload.onprogress = event => {
      if (!event.lengthComputable) return;
      progress.max = event.total;
      progress.value = event.loaded;
    };
    xhr.upload.onload = () => {
      progress.removeAttribute('value');
      message('Preparing');
      clearTimeout(timer);
      timer = setTimeout(readProgress, 250);
    };
    function finish(error) {
      clearTimeout(timer);
      request = null;
      setBusy(false);
      message(error, true);
    }
    xhr.onerror = () => finish('Connection lost. Try again with the selected clips.');
    xhr.onload = () => {
      if (xhr.status !== 200 || !xhr.response?.visit) {
        finish(xhr.response?.detail ?? 'Could not save. Try again.');
        return;
      }
      const {visit, added, duplicates} = xhr.response;
      const receipt = added ? 'Saved ' + clipCount(added) : 'These clips are already in this visit';
      sessionStorage.setItem('shape-journal.receipt', receipt + (added && duplicates ? '. Duplicate clips skipped.' : ''));
      location.assign('/archive?visit=' + encodeURIComponent(visit));
    };
    xhr.send(data);
  });

  const receipt = sessionStorage.getItem('shape-journal.receipt');
  if (receipt) {
    sessionStorage.removeItem('shape-journal.receipt');
    const notice = get('receipt');
    notice.textContent = receipt;
    notice.hidden = false;
    setTimeout(() => { notice.hidden = true; }, 4200);
  }
  syncFields();
  readProgress();
  if (new URL(location.href).searchParams.has('add')) openDialog();
}
