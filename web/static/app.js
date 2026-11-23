'use strict';

const $ = (id) => document.getElementById(id);
const state = { file: null, original: null, restored: null, ready: false, busy: false, view: 'original', appliedStrength: null };
const tabs = [...document.querySelectorAll('[data-view]')];
const maxUploadBytes = 20 * 1024 * 1024;

function announce(message) {
  $('liveStatus').textContent = message;
}

function showError(message) {
  $('errorNotice').textContent = message;
  $('errorNotice').hidden = false;
  announce(message);
}

function clearError() {
  $('errorNotice').hidden = true;
  $('errorNotice').textContent = '';
}

function updateControls() {
  $('restoreButton').disabled = !state.file || !state.ready || state.busy;
  $('downloadButton').disabled = !state.restored || state.busy;
  ['sampleButton', 'dropZone', 'replaceImage', 'removeImage', 'imageUpload', 'restorationStrength'].forEach((id) => { $(id).disabled = state.busy; });
  tabs.forEach((tab) => {
    tab.disabled = state.busy || (tab.dataset.view === 'original' ? !state.original : !state.restored);
    tab.setAttribute('aria-pressed', String(tab.dataset.view === state.view));
  });
  $('restoreSpinner').hidden = !state.busy;
  $('restoreIcon').hidden = state.busy;
  const changed = state.restored && Number($('restorationStrength').value) / 100 !== state.appliedStrength;
  $('restoreLabel').textContent = state.busy ? 'Please wait…' : changed ? 'Update preview' : 'Restore image';
  $('restoreButton').setAttribute('aria-busy', String(state.busy));
  $('previewStage').setAttribute('aria-busy', String(state.busy));
  if (!state.busy) {
    $('processingHint').textContent = !state.ready ? 'Connecting to the restoration model…' : changed ? 'Select Update preview to apply this strength.' : state.restored ? 'Ready to compare or download.' : state.file ? 'Your original file will stay untouched.' : 'Add an image to get started.';
  }
}

function setBusy(busy, preview = false) {
  state.busy = busy;
  $('busyOverlay').hidden = !busy;
  $('busyTitle').textContent = preview ? 'Opening your image' : 'Restoring your image';
  $('busyDescription').textContent = preview ? 'Preparing your original preview.' : 'Creating a new preview. Your original stays untouched.';
  if (busy) {
    $('processingHint').textContent = preview ? 'Preparing your image…' : 'Restoring your image…';
  }
  updateControls();
}

async function requestImage(endpoint, file) {
  let response;
  try {
    response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file });
  } catch {
    throw new Error('Connection lost. Refresh the page and try again.');
  }
  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error('The image could not be processed. Please try again.');
  }
  if (!response.ok) throw new Error(result.error || 'The image could not be processed. Please try again.');
  return result;
}

function setSplit(value) {
  $('imageCanvas').style.setProperty('--split', `${value}%`);
  $('comparisonSlider').setAttribute('aria-valuetext', `${value}% original, ${100 - value}% restored`);
}

function renderView(view) {
  state.view = view;
  const compare = view === 'compare' && Boolean(state.restored);
  $('comparisonOverlay').hidden = !compare;
  $('comparisonSlider').hidden = !compare;
  $('splitLine').hidden = !compare;
  $('rightTag').hidden = !compare;
  if (compare) {
    $('baseImage').src = state.restored;
    $('baseImage').alt = 'Restored image';
    $('overlayImage').src = state.original;
    $('leftTag').textContent = 'Original';
    setSplit($('comparisonSlider').value);
  } else {
    $('baseImage').src = view === 'restored' ? state.restored : state.original;
    $('baseImage').alt = view === 'restored' ? 'Restored image' : 'Original uploaded image';
    $('leftTag').textContent = view === 'restored' ? 'Restored' : 'Original';
  }
  updateControls();
}

async function chooseFile(file) {
  if (state.busy || !file) return;
  clearError();
  if (!file.size) { showError('This file is empty. Please choose another image.'); return; }
  if (file.size > maxUploadBytes) { showError('This file is too large. Choose an image under 20 MB.'); return; }
  setBusy(true, true);
  try {
    const result = await requestImage('/api/preview', file);
    state.file = file;
    state.original = result.original;
    state.restored = null;
    state.appliedStrength = null;
    $('dropZone').hidden = true;
    $('fileCard').hidden = false;
    $('fileName').textContent = file.name;
    $('fileName').title = file.name;
    const size = file.size < 1024 * 1024 ? `${Math.max(1, Math.round(file.size / 1024))} KB` : `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
    $('fileMeta').textContent = `${result.width} × ${result.height} px · ${size}`;
    $('fileThumbnail').src = result.original;
    $('emptyState').hidden = true;
    $('imageCanvas').hidden = false;
    $('reviewNotice').hidden = true;
    $('previewSubtitle').textContent = 'Your original image, ready to restore';
    $('statusMark').classList.remove('complete');
    $('statusMark').firstElementChild.firstElementChild.setAttribute('href', '#i-image');
    $('statusTitle').textContent = 'Image ready';
    $('statusDetail').textContent = `${result.width} × ${result.height} pixels`;
    $('comparisonSlider').value = 50;
    renderView('original');
    announce('Image uploaded. Select Restore image to continue.');
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
    $('imageUpload').value = '';
  }
}

function resetImage() {
  if (state.busy) return;
  Object.assign(state, { file: null, original: null, restored: null, view: 'original', appliedStrength: null });
  $('fileCard').hidden = true;
  $('dropZone').hidden = false;
  $('emptyState').hidden = false;
  $('imageCanvas').hidden = true;
  $('reviewNotice').hidden = true;
  $('baseImage').removeAttribute('src');
  $('overlayImage').removeAttribute('src');
  $('fileThumbnail').removeAttribute('src');
  $('fileName').textContent = '';
  $('fileMeta').textContent = '';
  $('imageUpload').value = '';
  $('previewSubtitle').textContent = 'Your comparison will appear here';
  $('statusTitle').textContent = 'Ready when you are';
  $('statusDetail').textContent = 'Add an image to your workspace';
  $('statusMark').classList.remove('complete');
  $('statusMark').firstElementChild.firstElementChild.setAttribute('href', '#i-image');
  clearError();
  updateControls();
  announce('Image removed. Choose another image to continue.');
  $('dropZone').focus();
}

$('dropZone').addEventListener('click', () => $('imageUpload').click());
$('replaceImage').addEventListener('click', () => $('imageUpload').click());
$('imageUpload').addEventListener('change', (event) => chooseFile(event.target.files[0]));
$('removeImage').addEventListener('click', resetImage);
$('dropZone').addEventListener('dragover', (event) => {
  event.preventDefault();
  if (!state.busy) $('dropZone').classList.add('dragging');
});
$('dropZone').addEventListener('dragleave', () => $('dropZone').classList.remove('dragging'));
$('dropZone').addEventListener('drop', (event) => {
  event.preventDefault();
  $('dropZone').classList.remove('dragging');
  if (event.dataTransfer.files.length !== 1) {
    showError('Choose one image at a time.');
    return;
  }
  chooseFile(event.dataTransfer.files[0]);
});
// Avoid navigating away if a file is accidentally dropped outside the upload area.
window.addEventListener('dragover', (event) => event.preventDefault());
window.addEventListener('drop', (event) => event.preventDefault());

$('sampleButton').addEventListener('click', async () => {
  if (state.busy) return;
  clearError();
  setBusy(true, true);
  try {
    const response = await fetch('/sample.png');
    if (!response.ok) throw new Error('The sample could not be loaded. Try uploading your own image.');
    const sample = new File([await response.blob()], 'sample-xray.png', { type: 'image/png' });
    setBusy(false);
    await chooseFile(sample);
  } catch (error) {
    setBusy(false);
    showError(error.message || 'The sample could not be loaded. Please try again.');
  }
});

$('restoreButton').addEventListener('click', async () => {
  if (!state.file || !state.ready || state.busy) return;
  clearError();
  setBusy(true);
  announce('Restoring your image.');
  try {
    const strength = Number($('restorationStrength').value) / 100;
    const result = await requestImage(`/api/restore?strength=${strength}`, state.file);
    state.original = result.original;
    state.restored = result.restored;
    state.appliedStrength = result.strength;
    $('previewSubtitle').textContent = 'Drag the slider to inspect the difference';
    $('statusMark').classList.add('complete');
    $('statusMark').firstElementChild.firstElementChild.setAttribute('href', '#i-check');
    $('statusTitle').textContent = 'Restoration complete';
    const time = result.elapsed_ms >= 1000 ? `${(result.elapsed_ms / 1000).toFixed(1)} s` : `${Math.round(result.elapsed_ms)} ms`;
    $('statusDetail').textContent = `${result.width} × ${result.height} pixels · ${time} · ${Math.round(result.strength * 100)}% strength`;
    $('reviewNotice').hidden = !result.review_recommended;
    $('reviewNotice').textContent = result.detail_check?.weaker_edges ? 'This result has weaker edges. Lower the strength and inspect fine details against the original.' : 'Review recommended. Compare the restored image carefully with the original.';
    $('comparisonSlider').value = 50;
    renderView('compare');
    announce('Restoration complete. Compare the images or download your restored PNG.');
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
});

tabs.forEach((tab) => tab.addEventListener('click', () => renderView(tab.dataset.view)));
$('comparisonSlider').addEventListener('input', (event) => setSplit(event.target.value));
$('restorationStrength').addEventListener('input', () => {
  $('strengthValue').textContent = `${$('restorationStrength').value}%`;
  updateControls();
});
$('downloadButton').addEventListener('click', () => {
  if (!state.restored || state.busy) return;
  const name = state.file.name.replace(/\.[^/.]+$/, '').replace(/[^\w-]+/g, '_').slice(0, 80) || 'image';
  const link = document.createElement('a');
  link.href = state.restored;
  link.download = `${name}_restored.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  announce('Your restored image is downloading.');
});

$('helpButton').addEventListener('click', () => $('helpDialog').showModal());
['closeHelp', 'gotIt'].forEach((id) => $(id).addEventListener('click', () => $('helpDialog').close()));
$('helpDialog').addEventListener('click', (event) => {
  const bounds = $('helpDialog').getBoundingClientRect();
  if (event.target === $('helpDialog') && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) $('helpDialog').close();
});

async function connect() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error();
    const status = await response.json();
    state.ready = status.ready;
    $('modelResolution').textContent = `${status.processing_size} × ${status.processing_size}`;
    if (!state.ready) throw new Error();
    updateControls();
  } catch {
    $('processingHint').textContent = 'The restoration model is unavailable.';
    showError('Could not connect to the model. Refresh this page and try again.');
  }
}

setSplit(50);
updateControls();
connect();
