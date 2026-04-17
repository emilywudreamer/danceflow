// audio-align.js — Audio extraction, chromagram, cross-correlation alignment (v2)
// Pure browser, zero dependencies. Uses Web Audio API + hand-rolled FFT.

const AudioAlign = {

  // ─── FFT (Cooley-Tukey radix-2) ───────────────────────────────
  _fft(re, im) {
    const n = re.length;
    if (n <= 1) return;
    // bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const angle = -2 * Math.PI / len;
      const wRe = Math.cos(angle), wIm = Math.sin(angle);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let j = 0; j < half; j++) {
          const a = i + j, b = a + half;
          const tRe = curRe * re[b] - curIm * im[b];
          const tIm = curRe * im[b] + curIm * re[b];
          re[b] = re[a] - tRe; im[b] = im[a] - tIm;
          re[a] += tRe;         im[a] += tIm;
          const nextRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nextRe;
        }
      }
    }
  },

  /** Magnitude spectrum of a real signal window (length must be power of 2) */
  _magnitudeSpectrum(signal, fftSize) {
    const re = new Float64Array(fftSize);
    const im = new Float64Array(fftSize);
    // apply Hann window
    for (let i = 0; i < signal.length && i < fftSize; i++) {
      const w = 0.5 * (1 - Math.cos(2 * Math.PI * i / (fftSize - 1)));
      re[i] = signal[i] * w;
    }
    this._fft(re, im);
    const mag = new Float64Array(fftSize / 2);
    for (let i = 0; i < mag.length; i++) {
      mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    }
    return mag;
  },

  // ─── Audio extraction ─────────────────────────────────────────
  /**
   * Extract mono audio at 22050 Hz from a video Blob.
   * Returns { data: Float32Array, sampleRate: 22050 } or null if silent/no audio.
   */
  async extractAudio(videoBlob) {
    const TARGET_SR = 22050;
    try {
      const arrayBuf = await videoBlob.arrayBuffer();
      // Decode with a temporary context
      const tmpCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 1, 44100);
      const decoded = await tmpCtx.decodeAudioData(arrayBuf);

      // Resample to TARGET_SR mono
      const outLen = Math.ceil(decoded.duration * TARGET_SR);
      const offCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, outLen, TARGET_SR);
      const src = offCtx.createBufferSource();
      src.buffer = decoded;
      src.connect(offCtx.destination);
      src.start();
      const rendered = await offCtx.startRendering();
      const data = rendered.getChannelData(0);

      // Silence check: RMS energy
      let sumSq = 0;
      for (let i = 0; i < data.length; i++) sumSq += data[i] * data[i];
      const rms = Math.sqrt(sumSq / data.length);
      const dbRMS = 20 * Math.log10(rms + 1e-12);
      if (dbRMS < -40) return null; // silent

      return { data, sampleRate: TARGET_SR };
    } catch (e) {
      console.warn('AudioAlign.extractAudio failed:', e);
      return null;
    }
  },

  // ─── Chromagram ────────────────────────────────────────────────
  /**
   * 12-dim chroma features.  hopSize in samples (default 512 @ 22050 Hz ≈ 23 ms).
   * Returns Array<Float64Array[12]>
   */
  computeChromagram(audioData, sampleRate, hopSize) {
    hopSize = hopSize || 512;
    const fftSize = 4096;
    const frames = [];

    for (let i = 0; i + fftSize <= audioData.length; i += hopSize) {
      const seg = audioData.subarray(i, i + fftSize);
      const mag = this._magnitudeSpectrum(seg, fftSize);

      const chroma = new Float64Array(12);
      for (let k = 1; k < mag.length; k++) {
        const freq = k * sampleRate / fftSize;
        if (freq < 60 || freq > 5000) continue;
        const midi = 12 * Math.log2(freq / 440) + 69;
        const bin = ((Math.round(midi) % 12) + 12) % 12;
        chroma[bin] += mag[k] * mag[k];
      }
      // L2 normalise
      let norm = 0;
      for (let b = 0; b < 12; b++) norm += chroma[b] * chroma[b];
      norm = Math.sqrt(norm) || 1;
      for (let b = 0; b < 12; b++) chroma[b] /= norm;
      frames.push(chroma);
    }
    return frames;
  },

  // ─── Helpers ───────────────────────────────────────────────────
  _cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < 12; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return (na && nb) ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
  },

  _resampleChroma(chroma, ratio) {
    // ratio > 1 means student is faster → fewer output frames
    const outLen = Math.round(chroma.length / ratio);
    const out = [];
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * ratio;
      const lo = Math.floor(srcIdx), hi = Math.min(lo + 1, chroma.length - 1);
      const frac = srcIdx - lo;
      const v = new Float64Array(12);
      for (let b = 0; b < 12; b++) v[b] = chroma[lo][b] * (1 - frac) + chroma[hi][b] * frac;
      out.push(v);
    }
    return out;
  },

  // ─── Cross-Correlation alignment ──────────────────────────────
  /**
   * Try 17 tempo ratios × sliding offset.
   * Returns { offset, tempoRatio, confidence }
   */
  alignAudio(teacherChroma, studentChroma) {
    const tempoRatios = [
      0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
      1.0,
      1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5, 2.0
    ];

    let bestScore = -Infinity, bestOffset = 0, bestRatio = 1.0;

    for (const ratio of tempoRatios) {
      const resampled = this._resampleChroma(studentChroma, ratio);
      const tLen = teacherChroma.length;
      const sLen = resampled.length;
      const maxLag = Math.max(tLen, sLen);
      // coarse step first
      const step = Math.max(1, Math.floor(Math.min(tLen, sLen) / 80));

      for (let lag = -maxLag; lag < maxLag; lag += step) {
        let score = 0, count = 0;
        for (let i = 0; i < tLen; i += 2) { // stride 2 for speed
          const j = i + lag;
          if (j >= 0 && j < sLen) {
            score += this._cosineSim(teacherChroma[i], resampled[j]);
            count++;
          }
        }
        if (count < 4) continue;
        score /= count;
        if (score > bestScore) {
          bestScore = score; bestOffset = lag; bestRatio = ratio;
        }
      }
    }

    // Refine: fine search ±step around best
    {
      const resampled = this._resampleChroma(studentChroma, bestRatio);
      const tLen = teacherChroma.length;
      const sLen = resampled.length;
      const step = Math.max(1, Math.floor(Math.min(tLen, sLen) / 80));
      for (let lag = bestOffset - step; lag <= bestOffset + step; lag++) {
        let score = 0, count = 0;
        for (let i = 0; i < tLen; i++) {
          const j = i + lag;
          if (j >= 0 && j < sLen) { score += this._cosineSim(teacherChroma[i], resampled[j]); count++; }
        }
        if (count < 4) continue;
        score /= count;
        if (score > bestScore) { bestScore = score; bestOffset = lag; }
      }
    }

    return { offset: bestOffset, tempoRatio: bestRatio, confidence: Math.max(0, Math.min(1, bestScore)) };
  },

  // ─── Frame remapping ──────────────────────────────────────────
  /**
   * Given alignment result + both frame arrays, produce matched pairs.
   * hopSeconds converts chroma-frame offset to seconds.
   */
  remapStudentFrames(studentFrames, alignment, teacherFrames, hopSeconds) {
    hopSeconds = hopSeconds || (512 / 22050);
    const { offset, tempoRatio } = alignment;
    const timeOffset = offset * hopSeconds;

    const pairs = [];
    for (const tFrame of teacherFrames) {
      const studentTime = (tFrame.ts - timeOffset) * tempoRatio;
      let bestIdx = 0, bestDist = Infinity;
      for (let i = 0; i < studentFrames.length; i++) {
        const d = Math.abs(studentFrames[i].ts - studentTime);
        if (d < bestDist) { bestDist = d; bestIdx = i; }
      }
      if (bestDist < 0.5) {
        pairs.push({
          teacher: tFrame,
          student: studentFrames[bestIdx],
          teacherTime: tFrame.ts,
          studentTime: studentFrames[bestIdx].ts
        });
      }
    }
    return pairs;
  }
};
