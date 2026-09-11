import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.min.js'
import RegionsPlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.esm.min.js'

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  fileId: null,
  ws: null,
  regionsPlugin: null,
  activeRegion: null,
  videoEl: null,
  results: null,
  selectedHitIdx: null,
  settings: { handedness: 'right', style: 'crossed', ambiguity_threshold: 0.65 },
}

// ── DOM shortcuts ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id)

// ── Settings panel ────────────────────────────────────────────────────────────
$('settings-toggle').addEventListener('click', () => {
  const panel = $('settings-panel')
  const btn = $('settings-toggle')
  panel.classList.toggle('hidden')
  btn.classList.toggle('active')
})

function setupToggleGroup(id, key) {
  $(`${id}-group`).addEventListener('click', e => {
    const btn = e.target.closest('.toggle-btn')
    if (!btn) return
    $(`${id}-group`).querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'))
    btn.classList.add('active')
    state.settings[key] = btn.dataset.value
  })
}
setupToggleGroup('handedness', 'handedness')
setupToggleGroup('style', 'style')

$('threshold-slider').addEventListener('input', e => {
  const v = parseInt(e.target.value)
  state.settings.ambiguity_threshold = v / 100
  $('threshold-val').textContent = `${v}%`
})

// ── Upload ────────────────────────────────────────────────────────────────────
const uploadZone = $('upload-zone')
const fileInput = $('file-input')

uploadZone.addEventListener('click', () => fileInput.click())
$('browse-btn').addEventListener('click', e => { e.stopPropagation(); fileInput.click() })
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleUpload(e.target.files[0]) })

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over') })
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'))
uploadZone.addEventListener('drop', e => {
  e.preventDefault()
  uploadZone.classList.remove('drag-over')
  if (e.dataTransfer.files[0]) handleUpload(e.dataTransfer.files[0])
})

async function handleUpload(file) {
  setUploading(true)

  const form = new FormData()
  form.append('file', file)

  let data
  try {
    const res = await fetch('/api/upload', { method: 'POST', body: form })
    if (!res.ok) throw new Error(await res.text())
    data = await res.json()
  } catch (err) {
    setUploading(false)
    showError('Upload failed: ' + err.message)
    return
  }

  state.fileId = data.file_id
  setUploading(false)
  showEditor(data)
}

function setUploading(loading) {
  const btn = $('browse-btn')
  btn.textContent = loading ? 'Uploading…' : 'Browse files'
  btn.disabled = loading
  uploadZone.style.opacity = loading ? '0.6' : '1'
}

// ── Editor ────────────────────────────────────────────────────────────────────
function showEditor(uploadData) {
  $('upload-zone').classList.add('hidden')
  $('results').classList.add('hidden')
  $('editor').classList.remove('hidden')

  $('waveform-filename').textContent = uploadData.filename

  if (uploadData.video_url) {
    const vp = $('video-panel')
    vp.classList.remove('hidden')
    $('video-filename').textContent = uploadData.filename
    const vid = $('video-player')
    vid.src = uploadData.video_url
    state.videoEl = vid
  } else {
    $('video-panel').classList.add('hidden')
    state.videoEl = null
  }

  initWaveSurfer(uploadData.audio_url)
}

function initWaveSurfer(audioUrl) {
  if (state.ws) { state.ws.destroy(); state.ws = null }

  const ws = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#3d4f6e',
    progressColor: '#7c6af7',
    cursorColor: '#cdd9e5',
    cursorWidth: 1,
    barWidth: 2,
    barRadius: 2,
    barGap: 1,
    height: 90,
    normalize: true,
    interact: true,
    backend: 'WebAudio',
  })

  const regions = ws.registerPlugin(RegionsPlugin.create())
  state.regionsPlugin = regions

  regions.enableDragSelection({ color: 'rgba(124,106,247,0.18)' })

  // Only allow one region at a time
  regions.on('region-created', region => {
    regions.getRegions().forEach(r => { if (r.id !== region.id) r.remove() })
    state.activeRegion = region
    onRegionChange(region)
  })
  regions.on('region-updated', region => {
    state.activeRegion = region
    onRegionChange(region)
  })

  ws.on('timeupdate', t => {
    $('waveform-time').textContent = fmtTime(t)
    if (state.videoEl && Math.abs(state.videoEl.currentTime - t) > 0.15) {
      state.videoEl.currentTime = t
    }
  })

  ws.on('play', () => {
    $('play-icon').classList.add('hidden')
    $('pause-icon').classList.remove('hidden')
    state.videoEl?.play()
  })
  ws.on('pause', () => {
    $('play-icon').classList.remove('hidden')
    $('pause-icon').classList.add('hidden')
    state.videoEl?.pause()
  })

  ws.load(audioUrl)
  state.ws = ws
}

$('play-btn').addEventListener('click', () => state.ws?.playPause())

function onRegionChange(region) {
  $('region-hint').classList.add('hidden')
  $('region-info').classList.remove('hidden')
  $('region-start').textContent = fmtTime(region.start)
  $('region-end').textContent = fmtTime(region.end)

  const analyzeBtn = $('analyze-btn')
  analyzeBtn.disabled = false
  $('analyze-label').textContent = `Analyze ${fmtTime(region.start)} → ${fmtTime(region.end)}`
}

$('clear-region-btn').addEventListener('click', () => {
  state.regionsPlugin?.getRegions().forEach(r => r.remove())
  state.activeRegion = null
  $('region-hint').classList.remove('hidden')
  $('region-info').classList.add('hidden')
  const btn = $('analyze-btn')
  btn.disabled = true
  $('analyze-label').textContent = 'Select a region to analyze'
})

// ── Analyze ───────────────────────────────────────────────────────────────────
$('analyze-btn').addEventListener('click', analyzeRegion)

async function analyzeRegion() {
  const region = state.activeRegion
  if (!region || !state.fileId) return

  setAnalyzing(true)

  let data
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_id: state.fileId,
        start: region.start,
        end: region.end,
        ...state.settings,
      }),
    })
    if (!res.ok) {
      const msg = await res.text()
      throw new Error(msg)
    }
    data = await res.json()
  } catch (err) {
    setAnalyzing(false)
    showError('Analysis failed: ' + err.message)
    return
  }

  setAnalyzing(false)
  state.results = data
  renderResults(data)
}

function setAnalyzing(loading) {
  const btn = $('analyze-btn')
  btn.disabled = loading
  $('analyze-label').textContent = loading ? 'Analyzing…' : `Analyze ${fmtTime(state.activeRegion?.start ?? 0)} → ${fmtTime(state.activeRegion?.end ?? 0)}`
  $('analyze-spinner').classList.toggle('hidden', !loading)
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  $('results').classList.remove('hidden')
  $('hit-detail').classList.add('hidden')
  state.selectedHitIdx = null

  // Meta bar
  const { meta, hits } = data
  $('results-meta').innerHTML = `
    <span class="meta-item"><strong>${meta.tempo_bpm}</strong> BPM</span>
    <span class="meta-item"><strong>${meta.meter}</strong></span>
    <span class="meta-item"><strong>${meta.in_region}</strong> hits</span>
    ${meta.flagged > 0
      ? `<span class="meta-item meta-flag"><strong>${meta.flagged}</strong> ambiguous</span>`
      : '<span class="meta-item" style="color:var(--L)">✓ no ambiguous hits</span>'}
  `

  renderHitStrip(hits)
  renderTopSequences(hits)
  renderPatterns(data.patterns)

  $('results').scrollIntoView({ behavior: 'smooth', block: 'nearest' })
}

// Hit strip
function renderHitStrip(hits) {
  const strip = $('hit-strip')
  strip.innerHTML = ''
  hits.forEach((hit, idx) => {
    const cls = hit.flagged ? 'limb-amb' : `limb-${hit.limb}`
    const label = hit.flagged ? '?' : hit.limb
    const chip = document.createElement('div')
    chip.className = `hit-chip ${cls}`
    chip.dataset.idx = idx
    chip.innerHTML = `
      <span class="chip-label">${label}</span>
      <span class="chip-time">${fmtTime(hit.time)}</span>
      <div class="chip-conf-bar"><div class="chip-conf-fill" style="width:${Math.round(hit.confidence * 100)}%"></div></div>
    `
    chip.addEventListener('click', () => selectHit(idx))
    strip.appendChild(chip)
  })
}

// Hit detail
function selectHit(idx) {
  const hits = state.results.hits
  if (!hits) return

  // Deselect previous
  document.querySelectorAll('.hit-chip.selected').forEach(el => el.classList.remove('selected'))

  if (state.selectedHitIdx === idx) {
    state.selectedHitIdx = null
    $('hit-detail').classList.add('hidden')
    return
  }

  state.selectedHitIdx = idx
  const chip = document.querySelector(`.hit-chip[data-idx="${idx}"]`)
  chip?.classList.add('selected')

  const hit = hits[idx]
  $('detail-time').textContent = fmtTime(hit.time)
  $('detail-source').textContent = hit.source

  const flagEl = $('detail-flag')
  if (hit.flagged) flagEl.classList.remove('hidden')
  else flagEl.classList.add('hidden')

  // Probability bars — all limbs present in limb_probs
  const bars = $('prob-bars')
  bars.innerHTML = ''
  const sorted = Object.entries(hit.limb_probs).sort((a, b) => b[1] - a[1])
  sorted.forEach(([limb, prob]) => {
    const pct = Math.round(prob * 100)
    bars.innerHTML += `
      <div class="prob-row">
        <span class="prob-label ${limb}">${limb}</span>
        <div class="prob-track"><div class="prob-fill ${limb}" style="width:${pct}%"></div></div>
        <span class="prob-pct">${pct}%</span>
      </div>`
  })

  $('hit-detail').classList.remove('hidden')

  // Seek WaveSurfer to this hit
  state.ws?.seekTo(hit.time / (state.ws.getDuration() || 1))
}

$('detail-close').addEventListener('click', () => {
  $('hit-detail').classList.add('hidden')
  document.querySelectorAll('.hit-chip.selected').forEach(el => el.classList.remove('selected'))
  state.selectedHitIdx = null
})

// Top sequences
function renderTopSequences(hits) {
  const seqs = computeTopSequences(hits)
  const list = $('sequences-list')
  list.innerHTML = ''

  if (seqs.length === 0) {
    list.innerHTML = '<span class="no-patterns">Not enough hand hits to compute sequences.</span>'
    return
  }

  seqs.forEach((seq, i) => {
    const pct = Math.round(seq.probability * 100)
    const tokens = seq.sticking
      .split('')
      .map(t => `<span class="seq-token ${t}">${t}</span>`)
      .join('')

    list.innerHTML += `
      <div class="seq-row">
        <span class="seq-rank">#${i + 1}</span>
        <div class="seq-tokens">${tokens}</div>
        <div class="seq-prob">
          <div class="seq-prob-track"><div class="seq-prob-fill" style="width:${pct}%"></div></div>
          <span class="seq-prob-label">${pct}%</span>
        </div>
      </div>`
  })
}

// Compute top-N sticking sequences from per-hit probabilities.
// Enumerate all combinations of flipping the most uncertain hand hits (up to 4),
// score each by joint log-probability, return top 3.
function computeTopSequences(hits, topN = 3) {
  const handHits = hits.filter(h => h.limb !== 'F')
  if (handHits.length < 2) return []

  // Identify up to 4 most uncertain hits (those most worth exploring alternatives for)
  const uncertain = handHits
    .map((h, i) => ({ i, conf: h.confidence }))
    .filter(x => x.conf < 0.82)
    .sort((a, b) => a.conf - b.conf)
    .slice(0, 4)
    .map(x => x.i)

  const nCombos = 1 << uncertain.length
  const sequences = []

  for (let mask = 0; mask < nCombos; mask++) {
    const seq = handHits.map((h, i) => {
      const flipBit = uncertain.indexOf(i)
      if (flipBit >= 0 && (mask & (1 << flipBit))) {
        return h.limb === 'L' ? 'R' : 'L'
      }
      return h.limb
    })

    let logProb = 0
    handHits.forEach((h, i) => {
      const p = h.limb_probs[seq[i]] ?? 0.01
      logProb += Math.log(Math.max(p, 0.001))
    })
    sequences.push({ seq: seq.join(''), logProb })
  }

  sequences.sort((a, b) => b.logProb - a.logProb)

  // Deduplicate (same sticking string can arise from different flip combos)
  const seen = new Set()
  const unique = sequences.filter(s => { if (seen.has(s.seq)) return false; seen.add(s.seq); return true })

  // Convert to relative probabilities via softmax over top candidates
  const top = unique.slice(0, topN)
  const maxLP = top[0].logProb
  const exps = top.map(s => Math.exp(s.logProb - maxLP))
  const total = exps.reduce((a, b) => a + b, 0)

  return top.map((s, i) => ({ sticking: s.seq, probability: exps[i] / total }))
}

// Patterns
function renderPatterns(patterns) {
  const list = $('patterns-list')
  list.innerHTML = ''

  if (!patterns || patterns.length === 0) {
    list.innerHTML = '<span class="no-patterns">No recurring patterns detected in this region.</span>'
    return
  }

  patterns.slice(0, 8).forEach(p => {
    const rudiment = p.rudiment_match ? `<span class="pattern-rudiment">${p.rudiment_match}</span>` : ''
    const positions = p.positions?.slice(0, 4).map(x => typeof x === 'number' ? `bar ${x.toFixed(1)}` : x).join(', ') ?? ''
    list.innerHTML += `
      <div class="pattern-row">
        <span class="pattern-sticking">${p.sticking}</span>
        <span class="pattern-count">×${p.occurrences}</span>
        ${rudiment}
        ${positions ? `<span class="pattern-positions">@ ${positions}</span>` : ''}
      </div>`
  })
}

// ── Re-upload ─────────────────────────────────────────────────────────────────
$('reupload-btn').addEventListener('click', () => {
  state.fileId = null
  state.ws?.destroy(); state.ws = null
  state.activeRegion = null
  state.results = null
  $('editor').classList.add('hidden')
  $('results').classList.add('hidden')
  $('upload-zone').classList.remove('hidden')
  fileInput.value = ''
})

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmtTime(secs) {
  if (secs == null || isNaN(secs)) return '0:00.0'
  const m = Math.floor(secs / 60)
  const s = (secs % 60).toFixed(1).padStart(4, '0')
  return `${m}:${s}`
}

function showError(msg) {
  // Simple inline error — replace with a toast if desired
  const el = document.createElement('div')
  el.style.cssText = 'position:fixed;bottom:24px;right:24px;background:#b91c1c;color:#fff;padding:12px 20px;border-radius:8px;font-size:13px;z-index:9999;max-width:380px;'
  el.textContent = msg
  document.body.appendChild(el)
  setTimeout(() => el.remove(), 6000)
}
