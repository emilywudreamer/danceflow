// scoring.js — v2: weighted Euclidean + jerk-ratio fluency + rhythm from audio alignment
// Depends on: utils.js (DF.*), dtw.js (FastDTW — fallback only), audio-align.js (AudioAlign)

const Scoring = {

  // MediaPipe 33-point weights
  JOINT_WEIGHTS: [
    // 0-10: head/face
    0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3,
    // 11-12: shoulders
    0.6, 0.6,
    // 13-14: elbows
    1.0, 1.0,
    // 15-16: wrists
    1.3, 1.3,
    // 17-22: hand points
    0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
    // 23-24: hips
    0.5, 0.5,
    // 25-26: knees
    1.0, 1.0,
    // 27-28: ankles
    1.3, 1.3,
    // 29-32: feet
    0.8, 0.8, 0.8, 0.8
  ],

  // ─── Helpers ──────────────────────────────────────────────────
  _mean(arr) {
    if (!arr.length) return 0;
    let s = 0; for (let i = 0; i < arr.length; i++) s += arr[i];
    return s / arr.length;
  },

  /** Per-joint movement range across a frame sequence (max-min per coord) */
  _computeJointRanges(frames) {
    const n = frames[0].landmarks.length;
    const ranges = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
      for (const f of frames) {
        const l = f.landmarks[j];
        if (l.x < minX) minX = l.x; if (l.x > maxX) maxX = l.x;
        if (l.y < minY) minY = l.y; if (l.y > maxY) maxY = l.y;
        if (l.z < minZ) minZ = l.z; if (l.z > maxZ) maxZ = l.z;
      }
      // Use the diagonal of the bounding box as the range
      ranges[j] = Math.sqrt((maxX - minX) ** 2 + (maxY - minY) ** 2 + (maxZ - minZ) ** 2);
    }
    return ranges;
  },

  _landmarkDist(lmsA, lmsB) {
    let s = 0;
    for (let j = 0; j < lmsA.length; j++) {
      s += (lmsA[j].x - lmsB[j].x) ** 2 + (lmsA[j].y - lmsB[j].y) ** 2 + (lmsA[j].z - lmsB[j].z) ** 2;
    }
    return Math.sqrt(s);
  },

  // ─── Accuracy (v2) ────────────────────────────────────────────
  _computeAccuracy(mappedPairs, teacherFrames) {
    const jointRanges = this._computeJointRanges(teacherFrames);
    const W = this.JOINT_WEIGHTS;
    const frameScores = [];

    for (const { teacher, student } of mappedPairs) {
      let weightedError = 0, totalWeight = 0;
      const tLms = teacher.landmarks, sLms = student.landmarks;
      for (let j = 0; j < 33; j++) {
        const tl = tLms[j], sl = sLms[j];
        if ((tl.visibility || 0) < 0.5 || (sl.visibility || 0) < 0.5) continue;
        const dist = Math.sqrt((tl.x - sl.x) ** 2 + (tl.y - sl.y) ** 2 + (tl.z - sl.z) ** 2);
        const range = Math.max(jointRanges[j], 0.01);
        const normErr = Math.min(dist / range, 1.0);
        weightedError += W[j] * normErr;
        totalWeight += W[j];
      }
      frameScores.push(totalWeight > 0 ? (1 - weightedError / totalWeight) * 100 : 50);
    }
    return { overall: this._mean(frameScores), perFrame: frameScores };
  },

  // ─── Fluency (v2, jerk-ratio) ─────────────────────────────────
  _smoothArray(arr, windowSize) {
    // Moving average to reduce noise
    const w = windowSize || 3;
    const result = [];
    for (let i = 0; i < arr.length; i++) {
      let sum = 0, count = 0;
      for (let j = Math.max(0, i - w); j <= Math.min(arr.length - 1, i + w); j++) {
        sum += arr[j]; count++;
      }
      result.push(sum / count);
    }
    return result;
  },

  _jerkProfile(frames) {
    const vel = [], acc = [], jerk = [];
    for (let i = 1; i < frames.length; i++) {
      vel.push(this._landmarkDist(frames[i].landmarks, frames[i - 1].landmarks));
    }
    // Smooth velocity before computing derivatives (reduces MediaPipe jitter)
    const smoothVel = this._smoothArray(vel, 2);
    for (let i = 1; i < smoothVel.length; i++) acc.push(Math.abs(smoothVel[i] - smoothVel[i - 1]));
    const smoothAcc = this._smoothArray(acc, 2);
    for (let i = 1; i < smoothAcc.length; i++) jerk.push(Math.abs(smoothAcc[i] - smoothAcc[i - 1]));
    return jerk;
  },

  _computeFluency(mappedPairs) {
    const tFrames = mappedPairs.map(p => p.teacher);
    const sFrames = mappedPairs.map(p => p.student);
    const tJerks = this._jerkProfile(tFrames);
    const sJerks = this._jerkProfile(sFrames);
    const frameScores = [];
    // sigma=1.5 (was 0.5) — much more forgiving; ratio 1.0=perfect, 0.3-3.0 still scores well
    const sigma = 1.5;
    for (let i = 0; i < Math.min(tJerks.length, sJerks.length); i++) {
      const ratio = sJerks[i] / Math.max(tJerks[i], 0.001);
      // Clamp extreme ratios from noise
      const clampedRatio = Math.max(0.1, Math.min(ratio, 10));
      // Log-space gaussian: penalize multiplicative deviation, not additive
      const logDev = Math.log(clampedRatio);
      frameScores.push(Math.exp(-0.5 * (logDev / sigma) ** 2) * 100);
    }
    return { overall: this._mean(frameScores), perFrame: frameScores };
  },

  // ─── Extension (v2, joint angles + limb extension ratio) ──────
  LIMB_DEFS: [
    { name: 'left_arm',  joints: [11, 13, 15], weight: 1.2 },
    { name: 'right_arm', joints: [12, 14, 16], weight: 1.2 },
    { name: 'left_leg',  joints: [23, 25, 27], weight: 1.2 },
    { name: 'right_leg', joints: [24, 26, 28], weight: 1.2 },
    { name: 'left_body',  points: [15, 11, 23, 27], weight: 0.8 },
    { name: 'right_body', points: [16, 12, 24, 28], weight: 0.8 },
  ],

  _jointAngle(lms, a, b, c) {
    const va = { x: lms[a].x - lms[b].x, y: lms[a].y - lms[b].y, z: (lms[a].z||0) - (lms[b].z||0) };
    const vc = { x: lms[c].x - lms[b].x, y: lms[c].y - lms[b].y, z: (lms[c].z||0) - (lms[b].z||0) };
    const dot = va.x*vc.x + va.y*vc.y + va.z*vc.z;
    const magA = Math.sqrt(va.x**2 + va.y**2 + va.z**2);
    const magC = Math.sqrt(vc.x**2 + vc.y**2 + vc.z**2);
    if (magA < 1e-6 || magC < 1e-6) return 0;
    return Math.acos(Math.min(1, Math.max(-1, dot / (magA * magC)))) * 180 / Math.PI;
  },

  _pointDist3(a, b) {
    return Math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + ((a.z||0)-(b.z||0))**2);
  },

  _limbExtensionRatio(lms, points) {
    let totalSeg = 0;
    for (let i = 1; i < points.length; i++) totalSeg += this._pointDist3(lms[points[i-1]], lms[points[i]]);
    const endToEnd = this._pointDist3(lms[points[0]], lms[points[points.length-1]]);
    return totalSeg > 0 ? endToEnd / totalSeg : 0;
  },

  _computeExtension(mappedPairs) {
    const frameScores = mappedPairs.map(({ teacher, student }) => {
      let totalScore = 0, totalWeight = 0;
      for (const limb of this.LIMB_DEFS) {
        if (limb.joints) {
          const [a, b, c] = limb.joints;
          const tAngle = this._jointAngle(teacher.landmarks, a, b, c);
          const sAngle = this._jointAngle(student.landmarks, a, b, c);
          if (tAngle > 10) {
            const ratio = Math.min(sAngle / tAngle, 1.2);
            totalScore += limb.weight * Math.min(ratio, 1.0) * 100;
            totalWeight += limb.weight;
          }
        } else if (limb.points) {
          const tExt = this._limbExtensionRatio(teacher.landmarks, limb.points);
          const sExt = this._limbExtensionRatio(student.landmarks, limb.points);
          if (tExt > 0.3) {
            const ratio = Math.min(sExt / tExt, 1.2);
            totalScore += limb.weight * Math.min(ratio, 1.0) * 100;
            totalWeight += limb.weight;
          }
        }
      }
      return totalWeight > 0 ? totalScore / totalWeight : 70;
    });
    return { overall: this._mean(frameScores), perFrame: frameScores };
  },

  // ─── Rhythm (v2, audio-based) ─────────────────────────────────
  _computeRhythm(alignment, mappedPairs) {
    // Audio confidence: remap 0.2-0.8 → 40-100 (chromagram rarely exceeds 0.8)
    const rawConf = alignment.confidence;
    const audioSync = Math.min(100, Math.max(0, (rawConf - 0.15) / 0.65 * 60 + 40));

    // Time sync: how consistently does student match teacher timing?
    // Tolerance 0.3s ≈ 1 beat at 120BPM. Half-beat off = significant deduction
    const timeDiffs = mappedPairs.map(p =>
      Math.abs(p.teacherTime - p.studentTime / alignment.tempoRatio)
    );
    const avgDiff = this._mean(timeDiffs);
    // Gaussian penalty: 0s=100, 0.15s=90, 0.3s=60, 0.5s=30, >0.8s≈0
    const timeSync = Math.exp(-4.0 * avgDiff * avgDiff) * 100;

    // Tempo match: soft curve instead of hard thresholds
    // 1.0 = perfect (100), 0.8/1.2 = still good (85), 0.5/2.0 = low (40)
    const tr = alignment.tempoRatio;
    const logDeviation = Math.abs(Math.log(tr)); // 0 at ratio=1, ~0.22 at 0.8/1.2
    const tempoScore = Math.exp(-2.0 * logDeviation * logDeviation) * 100;

    // Velocity profile correlation (movement speed shape similarity)
    const velCorr = this._velocityCorrelation(mappedPairs);

    // Blend: time sync (on-beat) + velocity shape are both critical
    const overall = timeSync * 0.35 + velCorr * 0.35 + audioSync * 0.15 + tempoScore * 0.15;

    return {
      overall: Math.round(overall),
      audioSync: Math.round(audioSync),
      timeSync: Math.round(timeSync),
      tempoRatio: alignment.tempoRatio,
      velCorr: Math.round(velCorr)
    };
  },

  _velocityCorrelation(mappedPairs) {
    // Compare smoothed velocity profiles of teacher vs student
    const tVel = [], sVel = [];
    for (let i = 1; i < mappedPairs.length; i++) {
      tVel.push(this._landmarkDist(
        mappedPairs[i].teacher.landmarks, mappedPairs[i-1].teacher.landmarks));
      sVel.push(this._landmarkDist(
        mappedPairs[i].student.landmarks, mappedPairs[i-1].student.landmarks));
    }
    // Smooth
    const tSmooth = this._smoothArray(tVel, 3);
    const sSmooth = this._smoothArray(sVel, 3);
    const minLen = Math.min(tSmooth.length, sSmooth.length);
    if (minLen < 3) return 50;
    
    // Pearson correlation
    const tM = this._mean(tSmooth.slice(0, minLen));
    const sM = this._mean(sSmooth.slice(0, minLen));
    let num = 0, dt = 0, ds = 0;
    for (let i = 0; i < minLen; i++) {
      const a = tSmooth[i] - tM, b = sSmooth[i] - sM;
      num += a * b; dt += a * a; ds += b * b;
    }
    const corr = (dt && ds) ? num / (Math.sqrt(dt) * Math.sqrt(ds)) : 0;
    // Remap: corr 0.3→60, 0.6→80, 0.9→98 (power curve, more generous)
    return Math.max(0, Math.min(100, 40 + 60 * Math.max(0, corr)));
  },

  // ─── Segment scoring ──────────────────────────────────────────
  _computeSegments(mappedPairs, accPerFrame, extPerFrame, fluPerFrame, rhythmOverall) {
    const numPairs = mappedPairs.length;
    if (numPairs < 3) return [];

    const tStart = mappedPairs[0].teacherTime;
    const tEnd = mappedPairs[numPairs - 1].teacherTime;
    const numSegs = Math.min(10, Math.max(3, Math.floor(numPairs / 5)));
    const segFrameLen = Math.floor(numPairs / numSegs);

    const segments = [];
    for (let s = 0; s < numSegs; s++) {
      const start = s * segFrameLen;
      const end = s === numSegs - 1 ? numPairs : (s + 1) * segFrameLen;

      const segAcc = this._mean(accPerFrame.slice(start, end));
      const segExt = extPerFrame.length >= end
        ? this._mean(extPerFrame.slice(start, end))
        : this._mean(extPerFrame);
      const segFlu = fluPerFrame.length >= end
        ? this._mean(fluPerFrame.slice(Math.max(0, start - 3), Math.max(0, end - 3)))
        : this._mean(fluPerFrame);
      const segRhythm = rhythmOverall;

      const segStartTime = mappedPairs[start].teacherTime;
      const segEndTime = mappedPairs[Math.min(end, numPairs) - 1].teacherTime;

      segments.push({
        startTime: Math.round(segStartTime * 10) / 10,
        endTime: Math.round(segEndTime * 10) / 10,
        accuracy: Math.round(segAcc),
        extension: Math.round(segExt),
        fluency: Math.round(segFlu),
        rhythm: Math.round(segRhythm),
        score: Math.round(segAcc * 0.5 + segExt * 0.2 + segFlu * 0.15 + segRhythm * 0.15)
      });
    }
    return segments;
  },

  // ─── DTW fallback (wraps v1 path into mappedPairs) ────────────
  _fallbackDTW(teacherFrames, studentFrames) {
    const tSeq = teacherFrames.map(f => DF.flattenLandmarks(f.landmarks));
    const sSeq = studentFrames.map(f => DF.flattenLandmarks(f.landmarks));
    const dtw = FastDTW.compute(tSeq, sSeq);
    return dtw.path.map(([ti, si]) => ({
      teacher: teacherFrames[ti],
      student: studentFrames[si],
      teacherTime: teacherFrames[ti].ts,
      studentTime: studentFrames[si].ts
    }));
  },

  // ─── Main entry (v2) ──────────────────────────────────────────
  /**
   * compute(teacherFrames, studentFrames, alignment?)
   *   alignment: result from AudioAlign.alignAudio (or null for DTW fallback)
   */
  compute(teacherFrames, studentFrames, alignment) {
    let mappedPairs;
    let usedAudioAlign = false;
    let fallbackReason = null;

    if (alignment && alignment.confidence > 0.15) {
      mappedPairs = AudioAlign.remapStudentFrames(studentFrames, alignment, teacherFrames);
      if (mappedPairs.length < 3) {
        fallbackReason = '音频对齐后有效帧过少，已切换为动作对齐';
        mappedPairs = this._fallbackDTW(teacherFrames, studentFrames);
      } else {
        usedAudioAlign = true;
      }
    } else {
      if (alignment) {
        fallbackReason = '未检测到相同音乐，使用动作对齐模式';
      } else {
        fallbackReason = '无音频或音频静音，使用动作对齐模式';
      }
      mappedPairs = this._fallbackDTW(teacherFrames, studentFrames);
    }

    // Scoring dimensions
    const acc = this._computeAccuracy(mappedPairs, teacherFrames);
    const flu = this._computeFluency(mappedPairs);

    // Rhythm: if no audio alignment, synthesize a basic rhythm from DTW timing
    let rhythmResult;
    if (usedAudioAlign) {
      rhythmResult = this._computeRhythm(alignment, mappedPairs);
    } else {
      // fallback: velocity correlation with improved mapping
      const velScore = this._velocityCorrelation(mappedPairs);
      rhythmResult = { overall: Math.round(velScore), audioSync: 0, timeSync: Math.round(velScore), tempoRatio: 1.0, velCorr: Math.round(velScore) };
    }

    const ext = this._computeExtension(mappedPairs);
    const total = acc.overall * 0.5 + ext.overall * 0.2 + flu.overall * 0.15 + rhythmResult.overall * 0.15;

    // Segments — independent per-segment 3-dim scoring
    const segments = this._computeSegments(mappedPairs, acc.perFrame, ext.perFrame, flu.perFrame, rhythmResult.overall);

    // Effective comparison duration
    const effectiveDuration = mappedPairs.length > 1
      ? mappedPairs[mappedPairs.length - 1].teacherTime - mappedPairs[0].teacherTime
      : 0;
    const teacherDuration = teacherFrames.length > 1
      ? teacherFrames[teacherFrames.length - 1].ts - teacherFrames[0].ts
      : 0;
    const studentDuration = studentFrames.length > 1
      ? studentFrames[studentFrames.length - 1].ts - studentFrames[0].ts
      : 0;

    return {
      total: Math.round(total),
      accuracy: Math.round(acc.overall),
      extension: Math.round(ext.overall),
      fluency: Math.round(flu.overall),
      rhythm: Math.round(rhythmResult.overall),
      segments,
      // v2 audio alignment metadata
      audioAlign: {
        used: usedAudioAlign,
        tempoRatio: alignment ? alignment.tempoRatio : 1.0,
        confidence: alignment ? Math.round(alignment.confidence * 100) : 0,
        fallbackReason
      },
      effectiveDuration: Math.round(effectiveDuration * 10) / 10,
      teacherDuration: Math.round(teacherDuration * 10) / 10,
      studentDuration: Math.round(studentDuration * 10) / 10,
      pairCount: mappedPairs.length
    };
  }
};
