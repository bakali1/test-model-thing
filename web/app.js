const thread = document.getElementById('thread');
const log    = document.getElementById('log');
const q      = document.getElementById('q');
const stat   = document.getElementById('stat');
const send   = document.getElementById('send');
let messages = [];
let busy = false;
let abortFetch = null;
let CUR = {body: null, reply: '', learning: false, stopped: false};
const $ = id => document.getElementById(id);

// Both counters and the memory report are the SERVER's. The page mirrors
// them per byte so the model's behaviour is visible while it is happening,
// and re-syncs from the server whenever a step or a reply resets something
// it cannot work out on its own.
let MECH = {pending:0, chunk:0, steps:0, ids:[], maxTokens:400,
            threshold:0.35, layers:16, longBytes:4, long: undefined,
            snrTrunk: undefined, snrDecay: undefined};
let LAST = {latent:null, stops:null, heat:null, delta:null, conf:null,
            trace:null};

function note(text, kind){
  const n = $('note');
  n.textContent = text;
  n.className = 'note on' + (kind ? ' ' + kind : '');
  clearTimeout(note._t);
  note._t = setTimeout(() => { n.className = 'note'; }, 2600);
}
function flare(el){ el.classList.remove('fired');
                    void el.offsetWidth;
                    el.classList.add('fired'); }

function drawLearn(){
  const {pending, chunk} = MECH;
  // While a pass is running the meter belongs to the pass, not to the counter
  // toward the next step: those are different questions and the one being
  // asked right now is "is it moving", which a bar that fills toward a chunk
  // the reply has not reached yet cannot answer.
  if (MECH.training){
    const t = MECH.training;
    const pct = t.total ? 100 * t.i / t.total : 0;
    $('learn-n').textContent = pct.toFixed(0) + '% of the exchange'
      + (t.ce !== null && t.ce !== undefined ? '  ·  ce ' + t.ce.toFixed(3) : '')
      + (t.elapsed !== undefined ? '  ·  ' + t.elapsed + 's' : '');
    $('learn-f').style.width = pct.toFixed(1) + '%';
    return;
  }
  if (!chunk){ $('learn-n').textContent = 'off'; $('learn-f').style.width = '0';
               return; }
  const left = Math.max(0, chunk - pending);
  // the step is taken after the reply finishes, not mid-sentence, so a full
  // bar is true but would read as though nothing were going to happen
  $('learn-n').textContent = left
      ? left + ' B to next update'
      : (busy ? 'updates after this reply' : 'ready to update');
  $('learn-f').style.width = (100 * Math.min(pending, chunk) / chunk) + '%';
}

// Long memory: the count of dimensions past the half-life threshold, and how
// many of those are actually holding state. A dimension with a 32-byte
// half-life and zero state is not memory, and counting it as such is the
// mistake both plans warn about -- so the two are reported separately.
function drawMem(rep, churn){
  if (!rep) return;
  MECH.long = rep.n_long;
  $('mem-n').textContent = rep.n_long + ' dims > ' + rep.threshold + 'B'
                         + (rep.n_live < rep.n_long
                            ? '  (' + rep.n_live + ' live)' : '');
  // the bar is a proportion of a stated ceiling, not a raw count: 8 dims is
  // "every long-lived dimension this checkpoint has ever grown", so the full
  // bar means memory has saturated rather than that there are 8 of them
  $('mem-f').style.width = Math.min(100, 100 * rep.n_live / 8) + '%';
  // The exact-credit set is a DIFFERENT count from the half-life chips above
  // -- tail_credit keeps a dense per-dim accumulator only for the dims whose
  // sigma has crossed ring_k, which is a stricter test than 4 bytes. Both are
  // labelled so the two cannot be read as the same number twice.
  const ch = churn ? '  ·  exact-credit set ' + churn.n + ' dims'
                     + (churn.entered || churn.left
                        ? ' (' + (churn.entered ? '+' + churn.entered : '')
                          + (churn.entered && churn.left ? ', ' : '')
                          + (churn.left ? '\u2212' + churn.left : '')
                          + ' since last turn)' : '')
                   : '';
  $('m-mem').title = rep.n_live + ' of ' + rep.n_long + ' dims over '
                  + rep.threshold + ' bytes are holding state; the bar fills '
                  + 'at 8, which is every long-lived dim this checkpoint has '
                  + 'ever grown.' + ch;
  drawChips(rep);
}

// SNR is the standing blocker and it is now TWO bars, not one. An overall
// ~0.05 is consistent with a trunk that is barely learning AND with a healthy
// trunk beside a decay group that has never had a coherent signal -- and
// numpy_port's decay_init comment argues for the second. Both bars are scaled
// to 1.0, so 0.05 is visibly nothing.
function drawSnr(s){
  const set = (nid, fid, v) => {
    if (v === undefined || v === null){
      $(nid).textContent = 'no step yet'; $(fid).style.width = '0'; return;
    }
    $(nid).textContent = v;
    $(fid).style.width = Math.max(0, Math.min(100, 100 * v)) + '%';
  };
  if (!s){
    set('snr-n','snr-f',null); set('dec-n','dec-f',null);
    drawLayers(null); return;
  }
  set('snr-n','snr-f', s.trunk);
  set('dec-n','dec-f', s.decay);
  // Until the EMAs have seen a few dozen steps the two numbers are both
  // pinned near 0.1 by construction and their RATIO is noise that happens to
  // look like "they agree". Marking it is the difference between a warm-up
  // reading and a conclusion about which role the optimiser feeds.
  if (s.warm){
    $('snr-n').textContent = s.trunk + '  warming up (' + s.steps + ' steps)';
    $('dec-n').textContent = s.decay + '  not yet comparable';
  } else {
    const r = (s.decay > 0) ? s.decay / Math.max(s.trunk, 1e-9) : 0;
    // The ce that arrives with the optimizer step is the TRAINING ce on the
    // exchange just fed -- the model is being fitted to that exact text, so
    // a low number here is not evidence of quality, and in this project it
    // is not a result at all without a seed, a byte count and an SE. It is
    // shown because it moves with the update, but it is named so it cannot
    // be read as "how well the model writes".
    $('snr-n').textContent = s.trunk
      + (s.ce !== null && s.ce !== undefined ? '  train ce ' + s.ce : '');
    $('dec-n').textContent = s.decay
      + (r > 0 ? '  (' + (r < 0.01 ? '<1%' : Math.round(r * 100) + '%')
                         + ' of trunk)' : '');
  }
  const ceNote = (s.ce === null || s.ce === undefined) ? ''
    : '  ·  train ce ' + s.ce + (s.best_ce !== null && s.best_ce !== undefined
                                ? ' (best this session ' + s.best_ce + ')' : '')
    + ' is the loss on the exchange just fed, which'
    + ' the model is being fitted to as it goes; it is not held out and not'
    + ' a result (RESULTS.md needs a seed, a byte count and an SE, and treats'
    + ' anything inside 0.03 as noise)';
  MECH.snrTrunk = s.trunk; MECH.snrDecay = s.decay;
  // `all` is the whole-vector figure RESULTS.md quotes, so it is a hover
  // rather than a third bar -- three bars for two questions is noise. The
  // mean |g| rides along because SNR without scale is half a reading: a
  // group can be perfectly consistent on a gradient of 1e-9.
  $('m-snr').title = 'whole-vector SNR ' + s.all
                  + '  ·  ' + s.steps + ' optimiser steps'
                  + '  ·  mean |g| ' + s.trunk_g + ' trunk, ' + s.decay_g
                  + ' decay'
                  + '  ·  ||EMA(g)||^2 / EMA(||g||^2), telemetry.py' + ceNote;
  drawLayers(s);
}

function drawChips(rep){
  const seen = new Set(MECH.ids);
  const now = rep.dims.map(d => d.id);
  const fresh = MECH.ids.length ? new Set(now.filter(i => !seen.has(i))) : null;
  MECH.ids = now;
  const box = $('chips');
  box.innerHTML = '';
  for (const d of rep.dims){
    const c = document.createElement('span');
    c.className = 'chip' + (d.live ? ' live' : '')
                + (fresh && fresh.has(d.id) ? ' new' : '');
    c.textContent = d.hl + 'B';
    // colour alone cannot be the only signal, so the same fact is in the
    // text for anyone who cannot see the green and in the label for a
    // screen reader
    c.setAttribute('role', 'img');
    const held = d.live ? 'holding state, RMS ' + d.rms
                        : 'not holding anything';
    c.setAttribute('aria-label', d.id + ': half-life ' + d.hl
                              + ' bytes, ' + held);
    c.title = d.id + ' — half-life ' + d.hl + ' bytes, sigma ' + d.sigma
            + ', state ' + (d.live ? d.rms : '0 (not holding anything)');
    box.appendChild(c);
  }
  if (fresh && fresh.size) setTimeout(() => {
    box.querySelectorAll('.chip.new').forEach(c => c.classList.remove('new'));
  }, 1400);
}

// The half-life histogram. The chip row above shows the handful of dims past
// the threshold; this is all 8,192 of them, which is the number RESULTS.md
// keeps quoting and the page never showed. The point is usually that the top
// buckets are empty -- that reads as a wall of zeroes, so a bucket with no
// dims is dimmed rather than hidden, and the table says how many dims it is
// covering so "all empty" cannot be mistaken for "no data".
function drawHist(h){
  const box = $('hl');
  box.innerHTML = '';
  if (!h || !h.buckets || !h.total){
    $('hl-note').textContent = '';
    box.innerHTML = '<div class="kv"><span class="empty">no model loaded</span></div>';
    return;
  }
  const max = Math.max(1, ...h.buckets.map(b => b.n));
  for (const b of h.buckets){
    const row = document.createElement('div');
    row.className = 'hlrow' + (b.n ? (b.lo >= h.threshold ? ' long' : '')
                                   : ' dim');
    const lab = document.createElement('span');
    lab.className = 'lab'; lab.textContent = b.label + 'B';
    const bar = document.createElement('span');
    bar.className = 'hlbar';
    const fill = document.createElement('i');
    fill.style.width = (100 * b.n / max) + '%';
    bar.appendChild(fill);
    const n = document.createElement('span');
    n.className = 'n'; n.textContent = b.n;
    const live = document.createElement('span');
    live.className = 'live'; live.textContent = b.n ? b.live + ' live' : '';
    row.append(lab, bar, n, live);
    // the bar length is the count, and colour alone cannot carry the meaning
    row.setAttribute('role', 'img');
    row.setAttribute('aria-label',
      b.n + ' of ' + h.total + ' dimensions have a half-life of ' + b.label
      + ' bytes, ' + b.live + ' of them holding state');
    row.title = b.n + ' dims, ' + b.live + ' live'
              + (b.bytes !== null ? ', mean ' + b.bytes + 'B' : '');
    box.appendChild(row);
  }
  const long = h.buckets.filter(b => b.lo >= h.threshold)
                        .reduce((a, b) => a + b.n, 0);
  $('hl-note').textContent = h.total + ' dims \u00b7 '
    + long + ' past ' + h.threshold + 'B \u00b7 '
    + h.n_live_long + ' of those live';
}

// The reply's own shape. Every one of these is a thing cross-entropy is
// provably blind to: \"FS-N has a newline-collapse mode (16 consecutive
// newlines) that the other three never approach ... It is invisible to CE.\"
// That was measured offline by hand; here it is on every reply.
function drawDeg(d){
  const box = $('deg');
  box.innerHTML = '';
  if (!d){
    box.innerHTML = '<div class="empty">nothing generated yet</div>';
    $('deg-note').textContent = '';
    return;
  }
  const row = (k, v, flag) => {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd');
    dd.textContent = v;
    if (flag) dd.className = 'flag';
    box.append(dt, dd);
  };
  row('unique bytes', d.unique + ' of 256');
  row('longest byte run', d.max_run, d.max_run >= 8);
  row('longest newline run', d.max_newlines, d.max_newlines >= 8);
  row('repeated byte pairs', (100 * d.repeat).toFixed(1) + '%', d.repeat > 0.5);
  row('entropy', d.entropy.toFixed(2) + ' bits/byte');
  $('deg-note').textContent = d.degenerate ? 'looks degenerate' : 'no collapse mode';
}

// Accumulated wall clock, and the split inside a turn. Measured here at about
// 97% of a turn being the training pass, which is the single most actionable
// fact about this page -- and one turn's figure cannot tell you whether that
// holds.
function drawClock(d){
  const box = $('clk');
  box.innerHTML = '';
  const gen = d.gen_s_total, learn = d.learn_s_total;
  const tot = gen + learn;
  const row = (k, v, flag) => {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd');
    dd.textContent = v; if (flag) dd.className = 'flag';
    box.append(dt, dd);
  };
  if (!tot){
    box.innerHTML = '<div class="empty">no turn completed yet</div>';
    $('clk-note').textContent = '';
    return;
  }
  row('generating', gen.toFixed(1) + ' s');
  row('training on the exchange', learn.toFixed(1) + ' s');
  row('this turn', (d.gen_s + d.learn_s).toFixed(1) + ' s');
  row('training share', (100 * learn / tot).toFixed(0) + '%', learn / tot > 0.9);
  $('clk-note').textContent = tot.toFixed(0) + ' s of session';
}

// Per-layer SNR and |g|. The two group meters average 16 layers into one bar;
// this is what that average is hiding. The decay column is the queued
// experiment: SNR is direction consistency and |g| is scale, and a layer can
// have a perfectly respectable SNR on a gradient of 1e-9.
function drawLayers(snr){
  const box = $('lyr');
  box.innerHTML = '';
  const rows = snr && snr.layers;
  if (!rows || !rows.length){
    box.innerHTML = '<div class="kv"><span class="empty">no optimiser step yet</span></div>';
    $('lyr-note').textContent = '';
    return;
  }
  const hd = document.createElement('div');
  hd.className = 'lyrhd';
  for (const t of ['layer', 'snr', 'decay snr', '|g|'])
    hd.appendChild(Object.assign(document.createElement('span'), {textContent: t}));
  box.appendChild(hd);
  // two columns of eight so sixteen layers do not make the column twice as
  // tall as it needs to be
  const half = Math.ceil(rows.length / 2);
  const cols = [rows.slice(0, half), rows.slice(half)];
  const grid = document.createElement('div');
  grid.style.display = 'contents';
  for (let c = 0; c < 2; c++){
    for (const r of cols[c]){
      const row = document.createElement('div');
      row.className = 'lyrrow';
      const mk = (v, cls) => Object.assign(document.createElement('span'),
        {textContent: v, className: cls || ''});
      row.append(mk('L' + r.layer, 'lid'),
                 mk(r.snr.toFixed(3), r.snr < 0.02 ? 'dead' : ''),
                 mk(r.dsnr.toFixed(3), r.dsnr < 0.02 ? 'dead' : ''),
                 mk(r.gmean.toExponential(1)));
      row.title = 'layer ' + r.layer + ' — snr ' + r.snr + ' (|g| '
                + r.gmean + '); decay snr ' + r.dsnr + ' (|g| ' + r.dgmean + ')';
      grid.appendChild(row);
      row.setAttribute('role', 'img');
      row.setAttribute('aria-label',
        'layer ' + r.layer + ': signal-to-noise ' + r.snr + ' with mean gradient '
        + r.gmean + '; its decay signal-to-noise ' + r.dsnr + ' with mean gradient '
        + r.dgmean);
    }
  }
  box.appendChild(grid);
  const dead = rows.filter(r => r.dsnr < 0.02).length;
  $('lyr-note').textContent = snr.warm
    ? 'warming \u2014 only ' + snr.steps + ' step' + (snr.steps === 1 ? '' : 's')
      + ', of which nothing can be said yet'
    : dead + ' of ' + rows.length + ' layers with dead decay';
}

// The window label. Every per-byte series the server sends is capped at its
// history length, so on a long reply the panels are showing the TAIL and not
// the reply. All four panels cover the same window, so they all say so, from
// one helper -- a plot that quietly shows the last 200 bytes of a 1,800-byte
// reply is worse than no plot, because it looks like the whole thing.
//
// The condition is `from > 0`, which is the only thing that means "this is a
// suffix". An earlier version compared `from` against `to - n`, which is the
// same number by construction, so the label never appeared at all.
function winNote(c, n){
  if (!c || c.from === undefined || c.to === undefined) return '';
  if (!(c.from > 0)) return '';
  return 'bytes ' + c.from + '\u2013' + c.to + ' of ' + c.to + '  \u00b7  ';
}

// ------------------------------------------------ the confidence panel
// p is the probability the SAMPLE gave the byte it actually emitted, so this
// is a reading about this reply and not a loss: by the time cross-entropy
// reaches the trainer, every byte has been averaged into one number, and that
// average is precisely what hides the misspelling. The entropy trace rides
// along because a confident byte drawn from a flat distribution is a
// different failure from a confident byte drawn from a peaked one.
// The probability below which a byte counts as a guess. The server sends its
// own value with each turn and this is overwritten by it, so the rule drawn
// on the plot, the underline under a byte and the count in the byline are
// always the same number.
let UNSURE = 0.1;
function drawConf(c, mark){
  const cv = $('conf');
  const note = $('conf-note');
  const {g, w, h} = fit(cv);
  if (!c || !c.p || !c.p.length){ note.textContent = ''; return; }
  const p = c.p, n = p.length;
  const x = i => 6 + (i / Math.max(1, n - 1)) * (w - 12);
  const y = v => h - 6 - Math.max(0, Math.min(1, v)) * (h - 12);
  const mean = p.reduce((a, b) => a + b, 0) / n;
  // area first, edge on top: the fill is what makes a run of low confidence
  // legible at a glance, and the 1px edge is what keeps an isolated
  // confident byte from disappearing into the space above it
  g.fillStyle = CSSVAR('--accent'); g.globalAlpha = .16;
  g.beginPath(); g.moveTo(x(0), h - 6);
  for (let i = 0; i < n; i++) g.lineTo(x(i), y(p[i]));
  g.lineTo(x(n - 1), h - 6); g.closePath(); g.fill();
  g.globalAlpha = 1;
  g.strokeStyle = CSSVAR('--accent'); g.lineWidth = 1.2;
  g.beginPath();
  for (let i = 0; i < n; i++) i ? g.lineTo(x(i), y(p[i])) : g.moveTo(x(i), y(p[i]));
  g.stroke();
  if (c.h && c.h.length){
    g.strokeStyle = CSSVAR('--dim'); g.lineWidth = 1; g.setLineDash([3, 3]);
    g.beginPath();
    for (let i = 0; i < c.h.length; i++)
      i ? g.lineTo(x(i), y(c.h[i])) : g.moveTo(x(i), y(c.h[i]));
    g.stroke(); g.setLineDash([]);
  }
  // the mean, as a rule. "The model was sure on average" is true of a reply
  // that is confident everywhere and of one that is confident everywhere
  // except the eight bytes that carry the meaning, and those two are the
  // same sentence to anyone reading only the average.
  g.strokeStyle = CSSVAR('--dim'); g.lineWidth = 1; g.setLineDash([2, 4]);
  g.beginPath(); g.moveTo(0, y(mean)); g.lineTo(w, y(mean)); g.stroke();
  // and the UNSURE line, because a threshold nobody can see is a threshold
  // the reader has to take on trust
  g.strokeStyle = CSSVAR('--warn'); g.setLineDash([4, 3]);
  g.beginPath(); g.moveTo(0, y(UNSURE)); g.lineTo(w, y(UNSURE)); g.stroke();
  g.setLineDash([]);
  if (mark !== null && mark !== undefined && mark >= 0 && mark < n)
    cursor(g, x(mark), h, w);
  const low = p.filter(v => v < UNSURE).length;
  const temps = c.t || [];
  note.textContent = winNote(c, n) + n + ' bytes  \u00b7  mean p '
    + mean.toFixed(3) + '  \u00b7  ' + low + ' under ' + UNSURE
    + (temps.length ? '  \u00b7  temp ' + Math.min.apply(null, temps).toFixed(2)
                     + '\u2013' + Math.max.apply(null, temps).toFixed(2) : '');
}

// The loss across the exchange, one point per byte, in the order it was
// trained on. The question it answers is whether a single exchange moves the
// loss at all: the step only lands every `chunk` bytes, so a fall here is
// mostly the model fitting the bytes it is looking at, and the jumps are the
// step boundaries. Watching it arrive is also the only way to tell a pass
// that is working through a long exchange from one that has stopped.
function drawTrace(t){
  const box = $('lrow');
  if (!t || !t.trace || !t.trace.length){ box.hidden = true; return; }
  const cv = $('ltrace');
  const {g, w, h} = fit(cv);
  box.hidden = false;
  const v = t.trace, n = v.length;
  $('lchunk').textContent = t.chunk;
  const lo = Math.min(...v), hi = Math.max(...v);
  // a flat trace has zero range, and dividing by it would draw a line at
  // infinity; give it a floor so "no change" renders as a flat middle line
  const span = (hi - lo) > 1e-9 ? (hi - lo) : Math.max(hi * 0.1, 0.1);
  const y = q => h - 6 - (q - lo) / span * (h - 12);
  const x = i => 6 + (i / Math.max(1, n - 1)) * (w - 12);
  // the step boundaries, as rules. They are the only reason the trace has
  // vertical structure, and they are what ties the shape to the meter above.
  g.strokeStyle = CSSVAR('--line'); g.lineWidth = 1;
  const first = t.trace_from || 0;
  for (let i = t.chunk; i < n + first; i += t.chunk){
    g.beginPath();
    g.moveTo(x(i - first), 0); g.lineTo(x(i - first), h); g.stroke();
  }
  g.strokeStyle = CSSVAR('--accent'); g.lineWidth = 1.2;
  g.beginPath();
  for (let i = 0; i < n; i++) i ? g.lineTo(x(i), y(v[i])) : g.moveTo(x(i), y(v[i]));
  g.stroke();
  const fell = v[n - 1] - v[0];
  // the trace is capped at the same history length as the panels, so a long
  // exchange is a tail here too -- and a loss trace with no axis at all is
  // exactly the kind of reading that gets quoted as if it were the whole turn
  const from = t.trace_from || 0;
  const win = from > 0
            ? 'bytes ' + from + '\u2013' + (from + n) + ' of ' + (from + n)
              + '  \u00b7  ' : '';
  $('lnote').textContent = win + v[0].toFixed(3) + ' → ' + v[n - 1].toFixed(3)
    + '  (' + (fell <= 0 ? '−' : '+') + Math.abs(fell).toFixed(3) + ')'
    + '  ·  peak ' + hi.toFixed(3);
}

// ------------------------------------------------- pinned exchanges (A/B)
// Two turns, side by side, with the difference. This project's whole method is
// matched comparison -- RESULTS.md will not accept a number without its
// checkpoint, its ruler setting and its byte count -- and until now the page
// could show exactly one turn at a time, so the comparison had to happen in
// someone's head or in a terminal after the fact.
//
// TWO, not five. A five-column table of eleven-digit floats in a 400px column
// is not a comparison, it is a wall, and a third pinned turn would push the
// first one off the end. The oldest pin is dropped and the tray says so.
const PIN_KEY = 'tmt.pins';
let PINS = [];
try {
  const raw = localStorage.getItem(PIN_KEY);
  if (raw) PINS = JSON.parse(raw) || [];
} catch (e) { PINS = []; }

// Each row: the label, how to read the value, how to format it, and -- where
// there is one -- which direction is good. The direction is the reason this
// table is not simply "green for up".
//
// An earlier version coloured every increase green, which painted "repeated
// byte pairs 93.7% -> 95.5%" green: a repetition collapse getting worse,
// signed as though it were an improvement. A colour is a claim about which way
// is better, so it is only applied where the claim is defensible, and the
// rows with no defensible direction are left dim. `null` means the server
// never sent the number for that turn, and it prints as an em dash rather
// than as a zero -- the difference between "not measured" and "measured zero"
// is the whole discipline.
const PINROWS = [
  ['bytes written',        r => r.n,          'int',  null],
  ['non-printable',        r => r.nonprint,   'int',  'bad'],
  ['writing rate',         r => r.cps,        'dec',  null],
  ['mean p of those bytes', r => r.pmean,     'dec3', 'good'],
  ['bytes below ' + '0.1', r => r.unsure,    'int',  'bad'],
  ['repeated byte pairs',  r => r.repeat,     'pct',  'bad'],
  ['entropy',              r => r.entropy,    'dec',  null],
  ['longest newline run',  r => r.newlines,   'int',  'bad'],
  ['stop head',            r => r.stop,       'text', null],
  ['SNR trunk',            r => r.snrTrunk,   'dec3', 'good'],
  ['SNR decay',            r => r.snrDecay,   'dec3', 'good'],
  ['long-lived dims',      r => r.long,       'int',  'good'],
  ['  of those live',      r => r.live,       'int',  'good'],
  ['bytes fitted',         r => r.fitted,     'int',  null],
  ['optimiser steps',      r => r.steps,      'int',  null],
  ['training ce',          r => r.ce,         'dec3', null],
  ['training, % of turn',  r => r.share,      'pct',  null],
  ['Δθ/θ trunk',      r => r.dTrunk,    'pct',  null],
  ['Δθ/θ decay',      r => r.dDecay,    'pct',  null],
];
function fmt(v, kind){
  if (v === null || v === undefined) return '—';
  switch (kind){
    case 'int':   return String(v);
    case 'dec':   return Number(v).toFixed(1);
    case 'dec3':  return Number(v).toFixed(3);
    case 'pct':   return Number(v).toFixed(1) + '%';
    case 'text':  return String(v);
    default:      return String(v);
  }
}
function pinRecord(d, L){
  const deg = d.degeneracy || {}, conf = d.conf || {}, mem = d.memory || {};
  const p = conf.p || [];
  const sure = p.filter(v => v != null);
  const u = (conf.u !== undefined ? conf.u : UNSURE);
  const stops = d.stops || [];
  const rel = {};
  for (const row of (d.delta || [])) rel[row.role] = row.rel * 100;
  const turn = (d.gen_s || 0) + (d.learn_s || 0);
  return {
    prompt: (CUR.prompt || '').slice(0, 40),
    n: d.n, nonprint: d.nonprint, cps: d.cps,
    pmean: sure.length ? sure.reduce((a, b) => a + b, 0) / sure.length : null,
    unsure: sure.length ? sure.filter(v => v < u).length : null,
    repeat: deg.repeat === undefined ? null : deg.repeat * 100,
    entropy: deg.entropy, newlines: deg.max_newlines,
    stop: stops.length ? (Math.max.apply(null, stops) > (d.threshold || 9e9)
                          ? 'fired' : 'never') : null,
    snrTrunk: d.snr ? d.snr.trunk : null,
    snrDecay: d.snr ? d.snr.decay : null,
    long: mem.n_long, live: mem.n_live,
    fitted: L ? L.bytes : null, steps: L ? L.total : null,
    ce: L ? L.ce : null,
    share: turn > 0 ? 100 * (d.learn_s || 0) / turn : null,
    dTrunk: rel.trunk === undefined ? null : rel.trunk,
    dDecay: rel.decay === undefined ? null : rel.decay,
  };
}
function pinTurn(rec){
  // a turn already pinned is not pinned twice, and pinning the same exchange
  // again after re-asking is a different exchange, so it replaces
  PINS = PINS.filter(p => p.prompt !== rec.prompt || p.n !== rec.n);
  PINS.push(rec);
  while (PINS.length > 2) PINS.shift();
  try { localStorage.setItem(PIN_KEY, JSON.stringify(PINS)); } catch (e) {}
  drawPins();
}
function drawPins(){
  const box = $('pinrow'), t = $('pintab');
  if (!PINS.length){ box.hidden = true; t.innerHTML = ''; return; }
  box.hidden = false;
  const A = PINS[0], B = PINS[1];
  let html = '<tr><th>exchange</th>';
  for (const r of PINS)
    html += '<th title="' + (r.prompt || '').replace(/"/g, '') + '">'
          + (r.prompt ? r.prompt.slice(0, 22) : '—') + '</th>';
  if (B) html += '<th>Δ</th>';
  html += '<th></th></tr>';
  for (const [label, get, kind, dir] of PINROWS){
    const a = get(A), b = B ? get(B) : null;
    html += '<tr><td>' + label + '</td>';
    html += '<td>' + fmt(a, kind) + '</td>';
    if (B) html += '<td>' + fmt(b, kind) + '</td>';
    if (B){
      // The difference is only printed when both numbers exist. Printing a
      // difference against a missing value is how a comparison ends up
      // claiming a change that was never measured.
      let cell = '<td class="d">—</td>';
      if (kind !== 'text' && a != null && b != null){
        const diff = b - a;
        const eps = Math.abs(a) > 1e-12 ? Math.abs(a) * 1e-9 : 1e-12;
        // no colour unless the row says which way is better
        let cls = '';
        if (Math.abs(diff) > eps && dir){
          const better = diff > 0 ? dir === 'good' : dir === 'bad';
          cls = better ? ' up' : ' dn';
        }
        cell = '<td class="d' + cls + '">'
             + (diff > 0 ? '+' : '') + fmt(diff, kind) + '</td>';
      }
      html += cell;
    }
    html += '<td></td></tr>';
  }
  t.innerHTML = html;
  const clr = document.createElement('button');
  clr.className = 'clearpin';
  clr.type = 'button';
  clr.textContent = 'clear';
  clr.title = 'forget both pinned exchanges (this browser only)';
  clr.onclick = () => { PINS = [];
                        try { localStorage.removeItem(PIN_KEY); } catch (e) {}
                        drawPins(); };
  t.querySelector('tr').appendChild(clr);
  $('pinnote').textContent = PINS.length === 1
    ? 'pin another to compare'
    : 'later minus earlier';
}

// ------------------------------------------------- look: theme, size, layout
// Three settings that belong to the reader rather than to the page, all three
// remembered, all three applied before the first paint of the instruments --
// the canvases read their colours out of the CSS variables at draw time, so a
// theme applied late leaves every plot drawn in the previous palette.
const PREF = 'tmt.look';
let LOOK = {theme: null, size: 1, mech: null, folds: {}};
try { Object.assign(LOOK, JSON.parse(localStorage.getItem(PREF) || '{}')); }
catch (e) { LOOK = {theme: null, size: 1, mech: null, folds: {}}; }
function saveLook(){
  try { localStorage.setItem(PREF, JSON.stringify(LOOK)); } catch (e) {}
}
function applyTheme(){
  if (LOOK.theme) document.body.dataset.theme = LOOK.theme;
  else delete document.body.dataset.theme;
  $('theme').textContent = LOOK.theme === 'dark' ? 'Light'
                         : (LOOK.theme === 'light' ? 'Dark' : 'Dark');
  // the label names what the button WILL do, not what is on screen
  $('theme').title = 'switch to the '
    + (LOOK.theme === 'dark' ? 'light' : 'dark') + ' palette';
  // every plot has to be redrawn: it read its colours when it was drawn, and
  // a palette switch does not reach into a canvas
  requestAnimationFrame(redrawAll);
}
$('theme').onclick = () => {
  const sys = matchMedia('(prefers-color-scheme: dark)').matches
            ? 'dark' : 'light';
  // an unset theme means "follow the system", and the first click pins it to
  // the OTHER one, which is the one the reader is not already looking at
  LOOK.theme = (LOOK.theme || sys) === 'dark' ? 'light' : 'dark';
  saveLook(); applyTheme();
};
function applySize(){
  // on the ROOT, because every text size in the stylesheet is in rem
  if (LOOK.size > 1) document.documentElement.dataset.size = String(LOOK.size);
  else delete document.documentElement.dataset.size;
  $('bigger').textContent = 'A' + (LOOK.size > 1 ? '-' : '+');
  $('bigger').title = LOOK.size > 1
    ? 'smaller text (currently ' + LOOK.size + ' steps up)'
    : 'larger text — the instrument labels are 10-11px by default';
  // the canvases are sized in px and the panels reflow at a larger font, so
  // the plots have to be re-measured or they end up stretched
  requestAnimationFrame(redrawAll);
}
$('bigger').onclick = () => {
  LOOK.size = LOOK.size >= 3 ? 1 : LOOK.size + 1;
  saveLook(); applySize();
};

// The divider. Drag with the pointer, nudge with the arrow keys, and remember
// where it was left -- the panels are 44% by default, which is a guess, and
// the right width depends on whether you are reading a plot or a reply.
function applyMechWidth(px){
  const split = $('split');
  if (px === null){ delete split.style.getPropertyValue('--mechw');
                   return; }
  // clamped to a range where neither side can be squeezed out of usefulness
  const w = Math.max(260, Math.min(split.clientWidth - 300, px));
  split.style.setProperty('--mechw', w + 'px');
  $('mech').style.flex = '0 0 ' + w + 'px';
  $('mech').style.width = w + 'px';
  // the canvases are sized from clientWidth, so a drag that does not redraw
  // leaves every plot stretched to the width it had when the drag started
  clearTimeout(applyMechWidth._t);
  applyMechWidth._t = setTimeout(redrawAll, 90);
}
const grip = $('grip');
let drag = null;
grip.addEventListener('pointerdown', e => {
  drag = e.clientX;
  grip.classList.add('drag');
  grip.setPointerCapture(e.pointerId);
  e.preventDefault();
});
grip.addEventListener('pointermove', e => {
  if (drag === null) return;
  applyMechWidth(e.clientX - $('mech').getBoundingClientRect().left);
});
const endDrag = () => {
  if (drag === null) return;
  drag = null;
  grip.classList.remove('drag');
  LOOK.mech = $('mech').getBoundingClientRect().width;
  saveLook();
};
grip.addEventListener('pointerup', endDrag);
grip.addEventListener('pointercancel', endDrag);
grip.addEventListener('keydown', e => {
  const step = e.shiftKey ? 48 : 16;
  const cur = $('mech').getBoundingClientRect().width;
  if (e.key === 'ArrowLeft'){ e.preventDefault();
                               applyMechWidth(cur - step); }
  else if (e.key === 'ArrowRight'){ e.preventDefault();
                                    applyMechWidth(cur + step); }
  else if (e.key === 'Home'){ e.preventDefault(); applyMechWidth(260); }
  else if (e.key === 'Enter' || e.key === ' '){
    e.preventDefault();
    LOOK.mech = null; saveLook();
    $('mech').style.flex = ''; $('mech').style.width = '';
    requestAnimationFrame(redrawAll);
  }
});

// Per-panel collapse, keyed by the panel's own heading so a fold survives a
// reload and cannot be applied to the wrong panel when one is added.
function applyFolds(){
  for (const row of document.querySelectorAll('#instr .irow')){
    const key = row.dataset.fold || '';
    const on = !!LOOK.folds[key];
    row.classList.toggle('folded', on);
    const b = row.querySelector('.icol');
    if (b){
      b.textContent = on ? '+' : '–';
      b.setAttribute('aria-expanded', String(!on));
      b.title = on ? 'show this panel' : 'fold this panel away';
    }
  }
  requestAnimationFrame(redrawAll);
}
function wireFolds(){
  for (const row of document.querySelectorAll('#instr .irow')){
    const head = row.querySelector('.ihdr');
    if (!head) continue;
    const key = head.querySelector('span').textContent.trim().slice(0, 40);
    row.dataset.fold = key;
    const b = document.createElement('button');
    b.className = 'icol';
    b.type = 'button';
    head.appendChild(b);
    b.onclick = () => {
      LOOK.folds[key] = !LOOK.folds[row.dataset.fold];
      saveLook(); applyFolds();
    };
  }
  applyFolds();
}
applyTheme();
applySize();
wireFolds();
if (LOOK.mech) requestAnimationFrame(() => applyMechWidth(LOOK.mech));
// While the theme is unset the stylesheet follows the OS, but the canvases
// read their colours at draw time and know nothing about that -- so an OS
// switch at sunset leaves every plot drawn in the old palette until the next
// resize. This is the one case where the page can be wrong without anything
// being clicked.
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (!LOOK.theme) requestAnimationFrame(redrawAll);
});

// ------------------------------------------------ the byte cursor
// One index into the reply, drawn on every per-byte panel at once and read
// out in the strip. Before this the plots were pictures: seven panels of
// shapes with no way to ask what a specific byte was doing, which is the only
// question worth asking of a per-byte instrument.
//
// The cursor lives on the STRIP, not on the canvases. Four focusable canvases
// that all mean "move along the reply" is four tab stops for one control, and
// a screen-reader user gets the same single cursor either way -- so the strip
// takes focus and the arrows, and the panels follow.
let MARK = null;        // index within the PLOTTED window
let MARKLOCK = false;   // a click pins it there
const WINDOW = () => (LAST.conf && LAST.conf.from !== undefined)
                  ? LAST.conf : {from: 0, to: 0, u: UNSURE};
function markSpan(){ return WINDOW().to - WINDOW().from; }
function setMark(i, lock){
  const n = markSpan();
  if (i === null || !n){ MARK = null; MARKLOCK = false; }
  else {
    MARK = Math.max(0, Math.min(n - 1, i));
    if (lock !== undefined) MARKLOCK = lock;
  }
  paintMark();
}
function paintMark(){
  // the cursor has to be cleared on a new turn as well as moved within one:
  // an index that meant byte 300 of the last reply means something else here
  redrawMarked();
  drawReadout();
  for (const el of thread.querySelectorAll('.body span.sel'))
    el.classList.remove('sel');
  if (MARK === null) return;
  const abs = WINDOW().from + MARK;
  const body = thread.querySelector('.msg:last-child .body');
  const span = body && body.children[abs];
  if (span) span.classList.add('sel');
}
function redrawMarked(){
  if (LAST.conf) drawConf(LAST.conf, MARK);
  if (LAST.stops) drawStop(LAST.stops, MECH.threshold, MARK);
  if (LAST.heat) drawHeat(LAST.heat, MECH.layers, MARK);
  if (LAST.latent) drawLatent(LAST.latent, MARK);
}
function drawReadout(){
  const row = $('xrow');
  if (MARK === null || !LAST.conf){ row.hidden = true; return; }
  row.hidden = false;
  const c = LAST.conf || {}, w = WINDOW();
  const abs = w.from + MARK;
  const p = (c.p || [])[MARK];
  const h = (c.h || [])[MARK];
  const s = (LAST.stops || [])[MARK];
  const heat = LAST.heat || [];
  // the state of a byte is 16 numbers, one per layer, and the interesting
  // number is the biggest of them: "the recurrent state was busy HERE" is
  // what the panel is for, and its mean would hide exactly that
  const rms = heat[MARK] && heat[MARK].length
            ? Math.max.apply(null, heat[MARK]) : null;
  const pt = LAST.latent && LAST.latent.points
           ? LAST.latent.points[MARK] : null;
  $('x-i').textContent = abs + (w.from ? ' of ' + w.to : '');
  $('x-p').textContent = p == null ? '—' : p.toFixed(3);
  $('x-b').textContent = (p == null || p <= 0) ? '—'
                      : (-Math.log2(p)).toFixed(2);
  $('x-s').textContent = s == null ? '—' : s.toFixed(3);
  $('x-r').textContent = rms == null ? '—' : rms.toFixed(3);
  $('x-l').textContent = pt ? '(' + pt[0].toFixed(2) + ', '
                             + pt[1].toFixed(2) + ')' : '—';
  $('x-hint').textContent = (h != null
      ? 'entropy ' + h.toFixed(2) + ' bits  ·  ' : '')
    + '← → to move  ·  Esc to clear'
    + (MARKLOCK ? '  ·  pinned' : '');
}
// hover anywhere over the reply moves the cursor; a click pins it, because a
// cursor that vanishes the moment the pointer lands on a panel is useless
// when the pointer IS the thing you are reading against
thread.addEventListener('mouseover', e => {
  const s = e.target.closest ? e.target.closest('.body span[data-i]') : null;
  if (s && !MARKLOCK) setMark(+s.dataset.i - WINDOW().from);
});
thread.addEventListener('mouseleave', () => { if (!MARKLOCK) setMark(null); });
thread.addEventListener('click', e => {
  const s = e.target.closest ? e.target.closest('.body span[data-i]') : null;
  if (!s) return;
  if (MARKLOCK && MARK === +s.dataset.i - WINDOW().from){ setMark(null); return; }
  setMark(+s.dataset.i - WINDOW().from, true);
});
// the plots are the other way in: hover one, and the cursor is on that byte
for (const id of ['conf', 'stop', 'heat']){
  const cv = $(id);
  cv.addEventListener('mousemove', e => {
    if (MARKLOCK) return;
    const r = cv.getBoundingClientRect();
    // the draw code insets 6px either side and spans the rest, so the
    // pointer position has to be inverted the same way or the byte under the
    // cursor is off by up to a character
    const f = (e.clientX - r.left - 6) / Math.max(1, r.width - 12);
    setMark(Math.round(f * (markSpan() - 1)));
  });
  cv.addEventListener('mouseleave', () => { if (!MARKLOCK) setMark(null); });
}
$('xrow').addEventListener('keydown', e => {
  const n = markSpan();
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft'){
    e.preventDefault();
    const d = e.key === 'ArrowRight' ? 1 : -1;
    setMark(MARK === null ? 0 : MARK + d, MARKLOCK);
  } else if (e.key === 'Home'){ e.preventDefault(); setMark(0, MARKLOCK); }
  else if (e.key === 'End'){ e.preventDefault(); setMark(n - 1, MARKLOCK); }
  else if (e.key === 'Escape'){
    // Esc on a focused control belongs to that control. Without this it also
    // reaches the document handler and stops a running reply, so clearing
    // the cursor would silently cancel the turn.
    e.stopPropagation();
    setMark(null);
  }
});
// ------------------------------------------------ the instrument panels
// Three canvases and a table. All of them exist to show a thing the ruler
// cannot: what the latent does, what the stop head does, where the recurrent
// state actually is, and what the weights did. They are drawn on the done
// frame rather than per byte, so a long reply does not repaint 400 times.
const CSSVAR = n => getComputedStyle(document.documentElement)
                           .getPropertyValue(n).trim() || '#888';

function fit(cv){
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || cv.width, h = cv.clientHeight || cv.height;
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return {g, w, h};
}

// PCA of the trailing latents. The variance readout is not decoration: a
// scatter whose first two axes hold 1% of the variance is basically noise,
// and drawing it without saying so is how a projection becomes a lie.
// The cursor line, drawn identically on every per-byte panel so the eye can
// follow one byte across four plots without re-finding it in each. Solid and
// full-height, where the panels' own reference lines are dashed and partial: a
// dashed line is a threshold, a solid one is "you are here".
function cursor(g, px, h){
  g.save();
  g.strokeStyle = CSSVAR('--fg'); g.globalAlpha = .55; g.lineWidth = 1;
  g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke();
  g.restore();
}

function drawLatent(d, mark){
  const cv = $('lat');
  if (!d) return;
  const {g, w, h} = fit(cv);
  const pts = d.points || [];
  $('lat-var').textContent = pts.length
      ? pts.length + ' bytes · axes hold '
        + Math.round(100 * (d.var[0] + d.var[1])) + '% of variance'
      : 'too few bytes';
  if (pts.length < 2) return;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x1 - x0) > 1e-12 ? (w - 16) / (x1 - x0) : 0;
  const sy = (y1 - y0) > 1e-12 ? (h - 16) / (y1 - y0) : 0;
  const px = p => 8 + (p[0] - x0) * sx, py = p => h - 8 - (p[1] - y0) * sy;
  g.strokeStyle = CSSVAR('--accent'); g.lineWidth = 1.4; g.globalAlpha = .85;
  g.beginPath();
  pts.forEach((p, i) => i ? g.lineTo(px(p), py(p)) : g.moveTo(px(p), py(p)));
  g.stroke();
  // the newest byte, marked -- the rest of the line is the past
  g.globalAlpha = 1; g.fillStyle = CSSVAR('--accent');
  const last = pts[pts.length - 1];
  g.beginPath(); g.arc(px(last), py(last), 3.2, 0, 6.284); g.fill();
  // and the byte under the cursor, ringed. The latent has no byte axis -- it
  // is a projection, not a time series -- so here the cursor is a position in
  // the sequence rather than a line through the plot, and drawing a line
  // would imply an axis that does not exist.
  if (mark !== null && mark !== undefined && pts[mark]){
    const m = pts[mark];
    g.strokeStyle = CSSVAR('--fg'); g.lineWidth = 1.4;
    g.beginPath(); g.arc(px(m), py(m), 5, 0, 6.284); g.stroke();
  }
}

// the stop head, per byte, against the threshold. This is what turns "the
// 600k checkpoint hit the 2000-token cap" from an anecdote into something
// you can look at.
function drawStop(stops, thr, mark){
  const cv = $('stop');
  const {g, w, h} = fit(cv);
  if (!stops || !stops.length){ $('stop-note').textContent = ''; return; }
  // the same inset the cursor is inverted against, named once so the two
  // cannot drift apart and put the cursor a character away from the byte
  const at = i => 6 + (i / Math.max(1, stops.length - 1)) * (w - 12);
  const peak = Math.max(...stops);
  const hi = Math.max(thr * 1.35, peak * 1.1, 0.1);
  const y = v => h - 6 - (v / hi) * (h - 12);
  // the threshold as a rule, not a number you have to imagine
  g.strokeStyle = CSSVAR('--err'); g.lineWidth = 1; g.setLineDash([4, 3]);
  g.beginPath(); g.moveTo(0, y(thr)); g.lineTo(w, y(thr)); g.stroke();
  g.setLineDash([]);
  g.strokeStyle = CSSVAR('--accent'); g.lineWidth = 1.5;
  g.beginPath();
  stops.forEach((v, i) => {
    const px = at(i);
    i ? g.lineTo(px, y(v)) : g.moveTo(px, y(v));
  });
  g.stroke();
  if (mark !== null && mark !== undefined && mark >= 0 && mark < stops.length)
    cursor(g, at(mark), h);
  const fired = peak > thr;
  $('stop-note').textContent = 'peak ' + peak.toFixed(3) + ' vs cut ' + thr
                             + (fired ? ' · FIRED' : ' · never fired');
  $('stop-note').style.color = fired ? CSSVAR('--err') : '';
}

// per-layer state RMS across the reply. Every layer lights up here even
// though only ~11 dims anywhere remember past 4 bytes -- that CONTRAST is
// the finding: the state is busy everywhere, long memory is almost nowhere.
function drawHeat(heat, nlayers, mark){
  const cv = $('heat');
  const {g, w, h} = fit(cv);
  if (!heat || !heat.length || !nlayers) return;
  const n = heat.length;
  let hi = 0;
  for (const row of heat) for (const v of row) if (v > hi) hi = v;
  hi = hi || 1;
  const cw = w / n, ch = h / nlayers;
  for (let i = 0; i < n; i++)
    for (let L = 0; L < nlayers; L++){
      const v = heat[i][L] / hi;
      g.fillStyle = 'rgba(63,185,80,' + (0.06 + 0.94 * v).toFixed(3) + ')';
      g.fillRect(i * cw, h - (L + 1) * ch, Math.ceil(cw), Math.ceil(ch));
    }
  g.fillStyle = CSSVAR('--dim'); g.font = '10px ui-monospace,Menlo,monospace';
  g.textBaseline = 'middle';
  for (let L = 0; L < nlayers; L += Math.max(1, Math.round(nlayers / 8)))
    g.fillText('L' + L, 2, h - (L + 0.5) * ch);
  if (mark !== null && mark !== undefined && mark >= 0 && mark < n)
    cursor(g, (mark + 0.5) * cw, h);
  $('heat-note').textContent = n + ' bytes · ' + nlayers + ' layers · peak '
                             + hi.toFixed(3) + ' RMS';
}

// what the exchange did to theta. `rel` is the update-to-parameter ratio,
// so the decay row is the one to read: it is the role the queued factorial
// is about to spend quota on.
function drawDelta(rows, learn){
  const t = $('delta');
  const panel = t.closest('.ipanel');
  if (!rows || !rows.length){
    // an empty table under a heading reads as a bug; say WHY it is empty
    t.innerHTML = '';
    panel.style.display = learn ? '' : 'none';
    $('delta-note').textContent = learn ? 'no change recorded' : '';
    return;
  }
  panel.style.display = '';
  const head = '<tr><th>role</th><th class="n">params</th>'
             + '<th class="r">RMS &Delta;&theta;</th>'
             + '<th class="r">&Delta;/&theta;</th></tr>';
  const body = rows.map(r =>
    '<tr' + (r.role === 'decay' ? ' class="hl"' : '') + '>'
    + '<td>' + r.role + '</td>'
    + '<td class="n">' + r.n.toLocaleString() + '</td>'
    + '<td class="r">' + r.rms.toExponential(2) + '</td>'
    + '<td class="r">' + (r.rel * 100).toExponential(2) + '%</td></tr>').join('');
  t.innerHTML = head + body;
  $('delta-note').textContent = learn ? 'update-to-parameter ratio' : '';
}

let MODE = {learn: true};
function applyState(d){
  if (d.learn){ MECH.pending = d.learn.pending; MECH.chunk = d.learn.chunk;
                MECH.steps = d.learn.steps; MECH.learn = true; }
  else if (d.learn === null){ MECH.learn = false; }
  // the SERVER is the authority for the byte cap and the stop threshold.
  // The page used to hardcode max_new, which silently ignored --max-tokens:
  // asking for 70 and getting 400 back is the kind of quiet mismatch that
  // makes a number mean something other than what it says.
  if (d.max_tokens !== undefined) MECH.maxTokens = d.max_tokens;
  if (d.threshold !== undefined) MECH.threshold = d.threshold;
  if (d.meta && d.meta.layers) MECH.layers = d.meta.layers;
  if (d.meta && d.meta.long_bytes) MECH.longBytes = d.meta.long_bytes;
  if (d.snr !== undefined) drawSnr(d.snr);
  if (d.memory) drawMem(d.memory, d.churn || null);
  if (d.histogram) drawHist(d.histogram);
  if (d.degeneracy !== undefined) drawDeg(d.degeneracy);
  if (d.gen_s_total !== undefined) drawClock(d);
  drawLearn();
  if ($('mech').classList.contains('collapsed')) drawStrip();
}

connWrap(fetch('/api/state')).then(r => r.json()).then(d => {
  if (d.path) $('hdr').textContent = d.path;
  if (d.markers) $('hdr').textContent += '  ·  turn markers on';
  if (d.learn === null) MODE.learn = false; else if (d.learn) MODE.learn = true;
  if (d.meta && d.meta.t !== undefined){
    // the digest is ON SCREEN, not only in a tooltip: this tool's whole ethos
    // is that a number belongs to the checkpoint that produced it
    $('prov').textContent = 'sha256 ' + (d.meta.digest || '?').slice(0, 12)
                         + ' · t=' + d.meta.t
                         + ' · ' + (d.meta.params || 0).toLocaleString() + 'p';
    $('hdr').title = d.meta.path + '  ·  bytes=' + d.meta.bytes_seen
                  + '  dim=' + d.meta.dim + '  layers=' + d.meta.layers
                  + '  ring_k=' + d.meta.ring_k
                  + '  prime=' + d.meta.prime_bytes + 'B';
  }
  applyState(d);
}).catch(() => { setConn('bad', 'server unreachable'); });

// what the model is answering from, besides what you typed. It is in the
// state before the first message and is not part of any turn, so it is shown
// here rather than left to be taken on trust.
connWrap(fetch('/api/prime')).then(r => r.json()).then(p => {
  if (!p.chars) return;
  const d = document.createElement('details');
  d.className = 'prime';
  d.innerHTML = '<summary></summary><pre></pre>';
  d.querySelector('summary').textContent =
    'primed with ' + p.chars + ' bytes of held-out corpus, fed into the '
    + 'state once at startup before your first message';
  d.querySelector('pre').textContent = p.text;
  $('primebox').appendChild(d);
}).catch(() => {});   // the prime is a nicety; the state fetch above has
                      // already reported the connection

// The panels are tall and the conversation is the thing. Fold them away, keep
// the headline numbers in a one-line strip so the model's state is still
// readable at a glance, and remember the choice. Default: folded on a short
// viewport, because that is where they were eating the thread.
const MECH_KEY = 'tmt.mech';
function drawStrip(){
  // the strip mirrors the meters, so folding the panels away does not lose
  // the numbers. Called from applyState as well as from applyMech, because
  // the first render happens before /api/state has answered and would
  // otherwise sit there claiming the model was read-only.
  const parts = [];
  parts.push('training <b>' + (MECH.learn
      ? (MECH.pending + '/' + MECH.chunk + ' B to next step')
      : 'read-only') + '</b>');
  if (MECH.long !== undefined)
    parts.push('long memory <b>' + MECH.long + ' dims</b>');
  if (MECH.snrTrunk !== undefined)
    parts.push('SNR <b>' + MECH.snrTrunk + '</b> / <b>' + MECH.snrDecay + '</b>');
  $('mechbar').innerHTML = parts.join(' &middot; ');
}
function applyMech(collapsed){
  $('mech').classList.toggle('collapsed', collapsed);
  $('mechbar').hidden = !collapsed;
  $('toggle').textContent = collapsed ? 'Show panels' : 'Hide panels';
  $('toggle').setAttribute('aria-expanded', String(!collapsed));
  // the split is the default at any width that can hold two columns; folding
  // is the way to buy the whole window for the conversation, so the old
  // "fold on a short viewport" default only applies once the layout has
  // already fallen back to stacked
  if (collapsed) drawStrip();
  try { localStorage.setItem(MECH_KEY, collapsed ? '1' : '0'); } catch (e) {}
  // the canvases are sized off the panel being visible, so they have to be
  // redrawn after the fold or they come back at the wrong width
  if (!collapsed) requestAnimationFrame(redrawAll);
}
$('toggle').onclick = () =>
  applyMech(!$('mech').classList.contains('collapsed'));
function redrawAll(){
  if (LAST.latent) drawLatent(LAST.latent, MARK);
  if (LAST.stops) drawStop(LAST.stops, MECH.threshold, MARK);
  if (LAST.heat) drawHeat(LAST.heat, MECH.layers, MARK);
  if (LAST.delta) drawDelta(LAST.delta, MODE.learn);
  if (LAST.conf) drawConf(LAST.conf, MARK);
  if (LAST.trace) drawTrace(LAST.trace);
  drawReadout();
}
// The canvases are sized from clientWidth, so anything that changes the
// column width leaves them stretched or blank. That is now three things --
// folding, a window resize, and crossing the narrow-viewport breakpoint --
// so it is handled once, here, rather than at each call site.
let resizeTimer = null;
addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(redrawAll, 120);
});
// read the saved preference, else fold it only when the layout has already
// fallen back to stacked and the panels would eat the thread again
let saved = null;
try { saved = localStorage.getItem(MECH_KEY); } catch (e) {}
const stacked = () => matchMedia('(max-width:900px), (max-height:560px)').matches;
applyMech(saved === null ? stacked() : saved === '1');
if (stacked()) matchMedia('(max-width:900px), (max-height:560px)')
  .addEventListener('change', () => redrawAll());

// ------------------------------------------------------- connection state
// The server can go away mid-session -- a training run holding the card is the
// usual reason -- and until now the only sign of it was a status line that
// stopped updating, which reads as "the model is thinking". A pill in the
// header is the difference between a hang and an answer.
const CONN = $('conn');
function setConn(kind, text){
  CONN.className = 'conn' + (kind ? ' ' + kind : '');
  CONN.textContent = text || '';
}
// Every request goes through here, so the pill cannot claim to be connected
// while the last call actually failed.
function connWrap(promise){
  return promise.then(
    r => { setConn('ok', 'connected'); return r; },
    e => { setConn('bad', 'server unreachable'); throw e; });
}

// ------------------------------------------------ the empty state and seeds
// Three prompts that make a 4.5M-parameter model mid-training do something
// visible, each labelled with what it is there to show. They fill the box and
// stop: sending a prompt to a model that learns from what it is told is not
// something a first-time visitor should have done to them by clicking a
// suggestion.
const SEEDS = [
  ['The word is sepa', 'spelling, byte by byte'],
  ['He said, "', 'whether it closes the quote'],
  ['Once upon a time there was a small ', 'continuation'],
];
function empty(){
  thread.innerHTML = '<div class="empty">Ask it something.<br>'
    + 'It is 4.5M parameters and mid-training, and it will show.'
    + '<div class="seeds" id="seeds"></div></div>';
  const box = $('seeds');
  for (const [text, why] of SEEDS){
    const b = document.createElement('button');
    b.type = 'button';
    b.innerHTML = '<span></span><em></em>';
    b.querySelector('span').textContent = text;
    b.querySelector('em').textContent = why;
    b.title = 'fill the box with this — ' + why;
    b.onclick = () => { q.value = text; grow(); q.focus();
                        q.setSelectionRange(text.length, text.length); };
    box.appendChild(b);
  }
}
// Copy, with a fallback: navigator.clipboard needs a secure context, and this
// page is routinely opened on a LAN address over plain http, where it is
// simply undefined and the button would silently do nothing.
async function copyText(text){
  try {
    if (navigator.clipboard && window.isSecureContext){
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (e) { /* fall through to the old way */ }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    return ok;
  } catch (e) { return false; }
}
function add(role, text){
  if (thread.querySelector('.empty')) thread.innerHTML = '';
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = '<div class="who"></div><div class="body"></div>'
              + '<div class="acts"></div>';
  d.querySelector('.who').textContent = role === 'user' ? 'You' : 'TMT';
  d.querySelector('.body').textContent = text || '';
  const acts = d.querySelector('.acts');
  const copy = document.createElement('button');
  copy.type = 'button';
  copy.textContent = 'copy';
  copy.onclick = async () => {
    copy.textContent = (await copyText(d.querySelector('.body').textContent))
                     ? 'copied' : 'copy failed';
    setTimeout(() => { copy.textContent = 'copy'; }, 1600);
  };
  acts.appendChild(copy);
  thread.appendChild(d);
  return d.querySelector('.body');
}
// "Ask again" on a finished reply. Deliberately NOT called regenerate: the
// model has already been trained on that exchange, so the second answer is
// not a second sample from the same state and calling it one would make a
// difference in the output look like a difference in the sampling. The title
// says so, because that is the whole difference between an experiment and a
// reroll.
function offerAgain(msgEl, prompt, rec){
  const acts = msgEl.querySelector('.acts');
  if (!acts || acts.querySelector('.again')) return;
  const pin = document.createElement('button');
  pin.type = 'button';
  pin.className = 'pinbtn';
  pin.textContent = 'pin';
  pin.title = 'keep this exchange\\u2019s numbers, to compare against another';
  pin.onclick = () => pinTurn(rec);
  acts.appendChild(pin);
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'again';
  b.textContent = 'ask again';
  b.title = 'send the same prompt again — note the model has already been '
          + 'trained on the exchange above, so this is a different state, '
          + 'not a resample of the same one';
  b.onclick = () => {
    if (busy) return;
    // drop this exchange from the thread and the record, so the retry is a
    // fresh turn rather than a second reply hanging under the first
    const idx = Array.prototype.indexOf.call(thread.children, msgEl);
    if (idx > 0) thread.children[idx - 1].remove();
    msgEl.remove();
    if (messages.length >= 2) messages.splice(messages.length - 2, 2);
    q.value = prompt;
    grow();
    $('f').requestSubmit();
  };
  acts.appendChild(b);
}
// Only follow the text down while the reader is already at the bottom, so
// scrolling up to re-read something is not undone by the next byte.
function atBottom(){ return log.scrollHeight - log.scrollTop - log.clientHeight < 60; }
function follow(was){ if (was) log.scrollTop = log.scrollHeight; }
// Streaming paints on an animation frame and appends to ONE text node, rather
// than re-assigning textContent for the whole reply on every byte. At 400
// bytes that was 400 parses and 400 layouts of the entire thread, which is
// the reason a long reply could out-run its own painting. Appending a
// character to a text node is the one mutation that does not re-tokenize
// everything before it.
let painting = false;
function paint(){
  if (painting) return;
  painting = true;
  requestAnimationFrame(() => {
    painting = false;
    // the node is detached once the reply has been annotated or replaced, and
    // appending to a detached node is a silent no-op that looks like a stall
    if (!CUR.node || !CUR.body || CUR.node.parentNode !== CUR.body) return;
    if (CUR.tail){ CUR.node.appendData(CUR.tail); CUR.tail = ''; }
    follow(CUR.was);
  });
}
// The finished reply, one span per byte, each carrying the probability the
// draw gave it and the byte's own value. Built in a single pass when the
// reply is over: 400 spans cost nothing once, whereas making them while the
// text is still arriving is 400 reflows for a panel nobody can read yet.
// Bytes with no recorded probability (a suffix, or a frame from a server
// that predates this) fall through as plain text rather than being dropped.
function annotate(body, text, ps, bs){
  body.textContent = '';
  if (!text) return;
  const frag = document.createDocumentFragment();
  for (let i = 0; i < text.length; i++){
    const p = ps ? ps[i] : null, b = bs ? bs[i] : null;
    if (p == null && b == null){
      frag.appendChild(document.createTextNode(text[i])); continue;
    }
    const s = document.createElement('span');
    s.textContent = text[i];
    // the byte's own index, so hovering the reply can drive the cursor. It is
    // the index into the VISIBLE string, which includes the placeholder drawn
    // for a non-printable byte -- and the server indexes its series by bytes,
    // so the two agree, which is why the placeholder has to stay in place
    // rather than being dropped.
    s.dataset.i = i;
    let t = 'byte ' + i;
    if (b != null) t += ' (0x' + b.toString(16).toUpperCase()
                  .padStart(2, '0') + ')';
    if (p != null) t += '  \u00b7  p ' + p.toFixed(3)
                  + (p > 0 ? '  \u00b7  ' + (-Math.log2(p)).toFixed(1)
                            + ' bits to guess' : '');
    s.title = t;
    if (p != null && p < UNSURE) s.className = 'u';
    const printable = b != null && ((b >= 32 && b < 127) || b === 9
                                    || b === 10 || b === 13);
    if (b != null && !printable) s.className += ' np';
    frag.appendChild(s);
  }
  body.appendChild(frag);
}
// One place that knows how a finished reply is rendered, so the stop, error
// and done paths cannot each grow their own idea of it. Idempotent, because
// more than one of those paths can fire for a single turn: Esc is followed by
// the aborted fetch landing in the catch, and an error frame is followed by a
// done frame. Without the guard, the second one rebuilds the body from
// CUR.reply and silently drops the "[stopped]" or the error text -- which is
// the reply losing its own diagnosis to a second render of itself.
function finishBody(suffix){
  if (!CUR.body || CUR.finished) return;
  CUR.finished = true;
  annotate(CUR.body, CUR.reply, CUR.ps, CUR.bs);
  CUR.node = null;
  if (suffix) CUR.body.appendChild(document.createTextNode(suffix));
}
// Collapse to zero before measuring: 'auto' does not shrink a textarea
// inside a flex row, so scrollHeight reports the height it already has and
// the box only ever grows.
function grow(){ q.style.height = '0px';
                 q.style.height = Math.min(q.scrollHeight, 180) + 'px'; }
q.addEventListener('input', grow);
q.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault();
                                         document.getElementById('f').requestSubmit(); }
});
$('reset').onclick = () => { if (busy) return; messages = []; empty();
                             stat.textContent = '';
                             LAST = {latent:null, stops:null, heat:null,
                                    delta:null, conf:null, trace:null};
                             // the pins are the one thing that deliberately
                             // SURVIVES a reset: they are the comparisons you
                             // built, and the point of them is the pair
                             drawPins();
                             for (const id of ['lat','stop','heat','conf'])
                               $(id).getContext('2d').clearRect(0,0,$(id).width,$(id).height);
                             // the loss trace belongs to the exchange that
                             // made it, and a trace from before a reset is a
                             // reading about a conversation that no longer
                             // exists on screen
                             drawTrace(null);
                             setMark(null);
                             MECH.training = null;
                             $('m-learn').classList.remove('wide');
                             drawLearn();
                             $('lat-var').textContent = '';
                             $('conf-note').textContent = '';
                             $('stop-note').textContent = '';
                             $('heat-note').textContent = '';
                             drawDelta(null, MODE.learn);
                             // the per-turn panels have nothing to say after a
                             // reset, and stale readings are worse than none
                             drawDeg(null); drawClock({gen_s_total: 0,
                                                       learn_s_total: 0});
                             $('export').disabled = true;
                             $('export').textContent = 'Export';
                             q.focus(); };

// Export the session as JSON. Every number this page produces is one the
// project would otherwise throw away, and RESULTS.md's rule is that a number
// without its checkpoint, its ruler setting and its byte count is not a
// result -- so the export carries all three.
$('export').onclick = async () => {
  try {
    const r = await connWrap(fetch('/api/session'));
    const j = await r.json();
    const blob = new Blob([JSON.stringify(j, null, 2)],
                          {type: 'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'tmt-session-' + (j.meta.digest || 'nodigest').slice(0, 16)
                + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
    $('export').textContent = 'saved ' + j.turns.length;
    setTimeout(() => { $('export').textContent = 'Export'; }, 2500);
  } catch (e) { note('export failed: ' + e); }
};

// Esc stops the reply. The server is also told, because aborting the fetch
// alone would leave it generating to its cap while every later request
// queued behind the lock. The current reply is kept on CUR so this handler,
// which lives outside the submit closure, can see it.
//
// It CANNOT stop the learning pass. Once remember() has begun the weights are
// already moving and that pass runs to its end regardless -- measured: a
// cancel sent inside that window still took a step and advanced t. So the
// status says so instead of implying the turn was halted.
// Esc and the Stop button are the same action, so they are the same function.
// The button exists because Esc is a key and this page is going to be opened
// on a phone, where there is no Esc key and no way to stop a 400-byte reply
// short of reloading the tab and losing the conversation.
function stopTurn(){
  if (!busy) return;
  if (CUR.learning){
    stat.textContent = 'the reply is done; the exchange is still being '
                     + 'trained on and cannot be interrupted now'
                     + (MECH.chunk ? ' (' + MECH.pending + '/' + MECH.chunk
                                    + ' B to the next step)' : '');
    note('too late to stop — already training');
    return;
  }
  if (abortFetch) abortFetch.abort();
  fetch('/api/cancel', {method: 'POST'}).catch(() => {});
  CUR.stopped = true;
  if (CUR.body){
    CUR.body.classList.remove('caret');
    // annotated first, suffix second: the bytes the model DID emit are still
    // the interesting part, and they are annotated exactly as they would
    // have been had the reply run to its cap
    finishBody('\n\n[stopped]');
  }
  stat.textContent = 'stopped after ' + CUR.reply.length + ' bytes';
  note('stopped');
  busy = false;
  setBusy();
  q.focus();
}
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape' || !busy) return;
  e.preventDefault();
  stopTurn();
});
$('stopbtn').onclick = stopTurn;
// The composer has three states, not two: idle, generating (stoppable), and
// training (NOT stoppable, because the weights are already moving). A stop
// button that is merely disabled in the third state would be a button that
// does not explain itself, so it says why it cannot be pressed.
function setBusy(){
  send.disabled = busy;
  const b = $('stopbtn');
  b.hidden = !busy;
  b.disabled = !!(CUR && CUR.learning);
  b.textContent = (CUR && CUR.learning) ? 'Training…' : 'Stop';
  b.title = (CUR && CUR.learning)
          ? 'the reply is finished; the weights are already being updated and '
          + 'this pass cannot be interrupted'
          : 'stop generating (Esc)';
}

$('f').onsubmit = async (e) => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text || busy) return;
  busy = true;
  // CUR is replaced BEFORE setBusy(), so the button reflects the turn that is
  // starting rather than the tail of the one that just finished
  add('user', text);
  messages.push({role:'user', content:text});
  const body = add('bot', '');
  CUR = {body: body, reply: '', node: document.createTextNode(''), tail: '',
         ps: [], bs: [], was: true, learning: false, stopped: false,
         finished: false, prompt: text};
  setBusy();
  q.value = ''; grow();
  body.classList.add('caret');
  body.appendChild(CUR.node);
  log.scrollTop = log.scrollHeight;
  stat.textContent = 'thinking...';

  let reply = '';
  abortFetch = new AbortController();
  setConn('ok', 'connected');
  try {
    const r = await fetch('/api/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      signal: abortFetch.signal,
      body: JSON.stringify({messages, max_new: MECH.maxTokens,
                            threshold: MECH.threshold})
    });
    // A 500 comes back as an HTML error page, which is not SSE: every frame
    // check below would silently discard it and the turn would look like it
    // produced nothing. Check the status before trying to read a stream.
    if (!r.ok){
      throw new Error('server returned ' + r.status
                      + (r.status === 404 ? ' (no checkpoint loaded?)' : ''));
    }
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for(;;){
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      // SSE frames are separated by a blank line, and a frame can arrive
      // split across two reads, so anything after the last separator stays
      // in buf.
      const frames = buf.split('\n\n');
      buf = frames.pop();
      for (const f of frames){
        if (!f.startsWith('data: ')) continue;
        const d = JSON.parse(f.slice(6));
        if (d.t !== undefined){
          // the frame is one byte; its probability and its own value ride
          // along so the reply can be annotated afterwards without asking
          // the server a second time
          CUR.ps.push(d.p === undefined ? null : d.p);
          CUR.bs.push(d.b === undefined ? (d.t || '').charCodeAt(0) : d.b);
          // a byte that is not a printable character still counts, and is
          // drawn as a placeholder -- the reader should never see a shorter
          // reply than the model produced without being told why
          const ch = d.t || '\u00b7';
          reply += ch;
          // CUR.reply is what the Esc handler and the annotation read, so it
          // has to track the same string -- when it did not, Esc wiped the
          // whole reply and left only "[stopped]", which is destroying the
          // work rather than stopping it
          CUR.reply = reply;
          CUR.was = atBottom();
          CUR.tail += ch;
          paint();
          MECH.pending++; drawLearn();
        }
        else if (d.swap){
          // the swap frame carries ONLY a dim list. drawChips needs nothing
          // else, so nothing else is invented here: the earlier version
          // passed n_long/n_live derived from the list length under a
          // hardcoded threshold, which is a count the server never sent.
          if (d.memory) drawChips({dims: d.memory, threshold: MECH.longBytes});
        }
        else if (d.learning){
          // the bytes are done; the weights are not, and this is the window
          // in which Esc can no longer stop anything
          CUR.learning = true;
          body.classList.remove('caret');
          setBusy();
          // the pass is the long part of the turn, so the meter is handed
          // over to it now rather than after the fact
          MECH.training = null;
          $('m-learn').classList.add('wide');
          drawLearn();
          stat.textContent = 'reply done — training on the exchange...';
        }
        else if (d.train){
          // Progress through the pass, on the same stream the bytes came in
          // on. This is not a poll: it is the training loop reporting, which
          // is the only honest way to show a pass that is 97% of the turn and
          // used to be a blinking caret and nothing else.
          MECH.training = d.train;
          drawLearn();
          if (CUR.body) stat.textContent = 'training on the exchange — '
            + d.train.i + '/' + d.train.total + ' bytes'
            + (d.train.ce !== null && d.train.ce !== undefined
               ? '  ·  ce ' + d.train.ce.toFixed(3) : '')
            + (d.train.elapsed !== undefined ? '  ·  ' + d.train.elapsed + 's'
                                              : '');
        }
        else if (d.note){ note(d.note); }
        else if (d.error){ body.classList.add('err');
                           finishBody('\n\n[' + d.error + ']'); }
        else if (d.done){
          // the server's counters are authoritative: a step resets pending to
          // something the page cannot work out on its own
          const L = d.learned;
          applyState(d);
          // kept so folding the panels away and back can redraw them: a
          // canvas sized while its panel was display:none comes back blank
          LAST.stops = d.stops; LAST.latent = d.latent;
          LAST.heat = d.heat;   LAST.delta = d.delta;
          LAST.conf = d.conf || null;
          LAST.trace = (L && L.trace) ? L : null;
          // a cursor left over from the previous turn indexes bytes that mean
          // something else now, so it goes before the new turn is drawn
          setMark(null);
          // the server owns the threshold, for the same reason it owns the
          // byte cap: a rule drawn at a number the page made up is a rule
          // that can disagree with the count printed beside it
          if (d.conf && d.conf.u !== undefined) UNSURE = d.conf.u;
          drawStop(d.stops, d.threshold);
          drawLatent(d.latent);
          drawHeat(d.heat, MECH.layers);
          drawDelta(d.delta, MODE.learn);
          drawConf(LAST.conf);
          drawTrace(LAST.trace);
          // the pass is over, so the meter goes back to counting toward the
          // next step; a meter left reading 100% claims one is due now
          MECH.training = null;
          $('m-learn').classList.remove('wide');
          drawLearn();
          // the reply is finished, so it can be annotated; the caret comes
          // off first so the placeholder does not sit inside the last span
          body.classList.remove('caret');
          finishBody('');
          // The aggregate, in the byline rather than on every byte: a screen
          // reader should learn that a third of this reply was guesswork
          // without being made to read four hundred characters to find out.
          // The pin record is built HERE, on the message it belongs to, and
          // not into a single global slot: a global would be overwritten by
          // the next turn, so pinning the first of two exchanges would file
          // the second one's numbers under the first. A comparison built that
          // way is worse than no comparison, because it looks right.
          const msgEl = body.closest('.msg');
          if (msgEl){
            const rec = pinRecord(d, L);
            const sure = CUR.ps.filter(v => v != null);
            if (sure.length){
              const mean = sure.reduce((a, b) => a + b, 0) / sure.length;
              const low = sure.filter(v => v < UNSURE).length;
              msgEl.querySelector('.who').textContent =
                'TMT  \u00b7  ' + sure.length + ' bytes  \u00b7  '
                + 'mean p ' + mean.toFixed(2)
                + (low ? '  \u00b7  ' + low + ' below ' + UNSURE : '');
            }
            offerAgain(msgEl, CUR.prompt || '', rec);
          }
          $('export').disabled = false;
          if (L){
            if (L.error) note('learning failed: ' + L.error);
            else if (L.bytes && L.pending === 0){
              flare($('m-learn'));
              note('weight update' + (L.ce !== null && L.ce !== undefined
                                      ? ', ce ' + L.ce : ''));
            }
          }
          let line = d.n + ' bytes, ' + d.cps + '/s writing'
                   + (d.nonprint ? ' (' + d.nonprint + ' non-printable)' : '')
                   + (d.gen_s !== undefined ? ' in ' + d.gen_s + 's' : '');
          if (d.learn_s !== undefined && d.learn_s > 0.05)
            line += '  ·  +' + d.learn_s + 's learning on it';
          if (!L) line += '  ·  not learning';
          else if (L.error) line += '  ·  learning failed';
          else line += '  ·  ' + (L.pending || 0) + '/' + L.chunk
                    + ' bytes to the next step  ·  ' + L.total + ' total';
          stat.textContent = line;
        }
      }
    }
  } catch (err) {
    // a dropped connection is the server going away, not the model failing;
    // the browser reports both as "Failed to fetch", so say which is likely
    const gone = String(err).includes('Failed to fetch')
              || String(err).includes('aborted');
    body.classList.add('err');
    body.classList.remove('caret');
    // A connection that dies mid-pass leaves the training meter frozen at
    // whatever percentage it had reached, which is the one number on the page
    // that looks like it is still moving. Nothing is training any more.
    MECH.training = null;
    $('m-learn').classList.remove('wide');
    drawLearn();
    // the bytes that DID arrive are annotated rather than discarded, and the
    // error is appended after them rather than replacing them
    finishBody(gone ? '' : '\n\n[' + err + ']');
    // The Esc path was fixed so it could not leave the status reading
    // "thinking..."; the transport-drop path had the same failure and was
    // left with it. A dropped connection is the SERVER going away -- often a
    // training run holding the card -- so say that rather than leaving a
    // spinner nobody can stop.
    if (gone && !CUR.stopped){
      setConn('bad', 'server unreachable');
      stat.textContent = 'connection lost — is the server still running?'
                       + (reply ? ' (' + reply.length + ' bytes kept)' : '');
    }
  }
  body.classList.remove('caret');
  // Keep the turn for the DISPLAY. The server only reads the last user turn
  // each request -- the model's own replies live in its recurrent state, not
  // in the prompt -- so this history is for you, not for it.
  messages.push({role:'bot', content:reply});
  // "ask again" and "pin" are offered together, and only on the exchange that
  // just finished: an older reply has a different model state behind it, and
  // a button that cannot explain which one would be a trap.
  busy = false;
  CUR.learning = false;
  setBusy();
  q.focus();
};

empty(); q.focus();
drawPins();
// after layout, not during it: at the end of the script the flex row has no
// resolved width, so the placeholder wraps and scrollHeight reports a tall box
requestAnimationFrame(grow);
