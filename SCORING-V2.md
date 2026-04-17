# DanceFlow 评分体系 v2 — 音频对齐 + 科学评分

## 核心改进: 音频先行对齐

### 当前问题
V0.1 直接对两段视频的姿态序列做 DTW 对齐，但没有考虑:
1. 两段视频可能从歌曲不同位置开始（一个从副歌开始，一个从前奏）
2. 练习时音乐可能变速（0.5x/0.8x/1.2x）
3. 视频开头可能有准备动作、结尾可能多录了

### v2 Pipeline

```
Teacher Video ─→ 音频提取 ─→ ┐
                               ├─→ 音频对齐（找时间偏移+变速比）─→ 时间映射函数
Student Video ─→ 音频提取 ─→ ┘                                        ↓
                                                               重采样学生帧
Teacher Video ─→ 姿态提取 ─→ ┐                                       ↓
                               ├─→ 逐帧对比评分（已对齐）─→ 结果
Student Video ─→ 姿态提取 ─→ 重映射后的帧序列 ─→ ┘
```

### Step 1: 音频提取 (Web Audio API)

```javascript
async function extractAudio(videoBlob) {
  const audioCtx = new OfflineAudioContext(1, 1, 44100);
  const arrayBuffer = await videoBlob.arrayBuffer();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  
  // 重采样到统一采样率 (22050 Hz, 单声道)
  const offCtx = new OfflineAudioContext(1, 
    Math.ceil(audioBuffer.duration * 22050), 22050);
  const source = offCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offCtx.destination);
  source.start();
  const resampled = await offCtx.startRendering();
  return resampled.getChannelData(0); // Float32Array
}
```

### Step 2: 音频指纹 — Chromagram

将音频转换为 12 维色度特征向量（每个代表一个音高 C, C#, D, ..., B），对音乐内容不变，对变速有鲁棒性。

```javascript
function computeChromagram(audioData, sampleRate, hopSize=512) {
  const fftSize = 4096;
  const numBins = 12; // C, C#, D, ..., B
  const frames = [];
  
  for (let i = 0; i + fftSize <= audioData.length; i += hopSize) {
    const frame = audioData.slice(i, i + fftSize);
    const spectrum = fft(frame); // 使用 Web Audio AnalyserNode 或手写 FFT
    
    // 将频谱映射到 12 个色度 bin
    const chroma = new Float32Array(numBins);
    for (let k = 0; k < spectrum.length / 2; k++) {
      const freq = k * sampleRate / fftSize;
      if (freq < 60 || freq > 5000) continue; // 只看音乐频率范围
      const midi = 12 * Math.log2(freq / 440) + 69;
      const bin = ((Math.round(midi) % 12) + 12) % 12;
      chroma[bin] += spectrum[k] * spectrum[k]; // 能量
    }
    // 归一化
    const maxVal = Math.max(...chroma, 1e-8);
    frames.push(chroma.map(v => v / maxVal));
  }
  return frames; // Array<Float32Array[12]>
}
```

### Step 3: 音频对齐 — Cross-Correlation + 变速检测

```javascript
function alignAudio(teacherChroma, studentChroma) {
  // 尝试多个变速比
  const tempoRatios = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 
                        1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.5, 2.0];
  
  let bestScore = -Infinity;
  let bestOffset = 0;
  let bestRatio = 1.0;
  
  for (const ratio of tempoRatios) {
    // 按 ratio 重采样 student chromagram
    const resampled = resampleChroma(studentChroma, ratio);
    
    // 滑动窗口 cross-correlation
    const maxLag = Math.max(teacherChroma.length, resampled.length);
    for (let lag = -maxLag; lag < maxLag; lag += 4) { // 步长 4 加速
      let score = 0;
      let count = 0;
      for (let i = 0; i < teacherChroma.length; i++) {
        const j = i + lag;
        if (j >= 0 && j < resampled.length) {
          score += cosineSim(teacherChroma[i], resampled[j]);
          count++;
        }
      }
      if (count > 0) score /= count;
      if (score > bestScore) {
        bestScore = score;
        bestOffset = lag;
        bestRatio = ratio;
      }
    }
  }
  
  return {
    offset: bestOffset,      // 学生视频相对教师的帧偏移
    tempoRatio: bestRatio,   // 学生音乐的变速比 (1.0=同速, 0.8=慢速练习)
    confidence: bestScore    // 对齐置信度 (0~1)
  };
}
```

### Step 4: 帧重映射

```javascript
function remapStudentFrames(studentFrames, alignment, teacherFrames) {
  const { offset, tempoRatio } = alignment;
  const hopSeconds = 512 / 22050; // chromagram 的时间分辨率
  const timeOffset = offset * hopSeconds; // 转换为秒
  
  // 对于教师的每一帧，找到学生的对应帧
  const mappedPairs = [];
  for (const tFrame of teacherFrames) {
    // 教师时间 → 学生时间 (考虑偏移和变速)
    const studentTime = (tFrame.ts - timeOffset) * tempoRatio;
    
    // 找最近的学生帧
    let bestIdx = 0;
    let bestDist = Infinity;
    for (let i = 0; i < studentFrames.length; i++) {
      const dist = Math.abs(studentFrames[i].ts - studentTime);
      if (dist < bestDist) { bestDist = dist; bestIdx = i; }
    }
    
    if (bestDist < 0.5) { // 0.5 秒容差
      mappedPairs.push({
        teacher: tFrame,
        student: studentFrames[bestIdx],
        teacherTime: tFrame.ts,
        studentTime: studentFrames[bestIdx].ts
      });
    }
  }
  return mappedPairs;
}
```

### Step 5: 对齐后评分

用 Step 4 的 `mappedPairs` 替代 DTW path，然后按 v2 评分算法评分。

---

## v2 完整评分算法

### 关键点权重 (MediaPipe Pose 33 点)

```javascript
const JOINT_WEIGHTS = [
  // 0-10: 头部/面部 (低权重)
  0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3,
  // 11-12: 肩膀 (中权重)
  0.6, 0.6,
  // 13-14: 肘部 (高权重)
  1.0, 1.0,
  // 15-16: 手腕 (最高权重 — 手部动作是舞蹈差异最大的地方)
  1.3, 1.3,
  // 17-22: 手指 (中权重)
  0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
  // 23-24: 髋部 (低权重)
  0.5, 0.5,
  // 25-26: 膝盖 (高权重)
  1.0, 1.0,
  // 27-28: 脚踝 (最高权重)
  1.3, 1.3,
  // 29-32: 脚尖/脚跟 (中权重)
  0.8, 0.8, 0.8, 0.8
];
```

### 动作准确度 v2

```javascript
function computeAccuracy(mappedPairs, teacherFrames) {
  // 先计算教师每个关键点的运动范围（用于归一化）
  const jointRanges = computeJointRanges(teacherFrames); // 每个关节的 max-min
  
  const frameScores = mappedPairs.map(({ teacher, student }) => {
    let weightedError = 0;
    let totalWeight = 0;
    
    for (let j = 0; j < 33; j++) {
      const tl = teacher.landmarks[j];
      const sl = student.landmarks[j];
      
      // 跳过低置信度关键点
      if ((tl.visibility || 0) < 0.5 || (sl.visibility || 0) < 0.5) continue;
      
      // 欧氏距离
      const dist = Math.sqrt(
        (tl.x - sl.x) ** 2 + (tl.y - sl.y) ** 2 + (tl.z - sl.z) ** 2
      );
      
      // 用教师运动范围归一化
      const range = Math.max(jointRanges[j], 0.01);
      const normalizedError = Math.min(dist / range, 1.0);
      
      weightedError += JOINT_WEIGHTS[j] * normalizedError;
      totalWeight += JOINT_WEIGHTS[j];
    }
    
    return totalWeight > 0 ? (1 - weightedError / totalWeight) * 100 : 50;
  });
  
  return {
    overall: mean(frameScores),
    perFrame: frameScores
  };
}
```

### 流畅度 v2

```javascript
function computeFluency(mappedPairs) {
  // 计算教师和学生的 jerk (加速度变化率)
  const tJerks = computeJerkProfile(mappedPairs.map(p => p.teacher));
  const sJerks = computeJerkProfile(mappedPairs.map(p => p.student));
  
  const frameScores = [];
  for (let i = 0; i < Math.min(tJerks.length, sJerks.length); i++) {
    const ratio = sJerks[i] / Math.max(tJerks[i], 0.001);
    // 高斯映射: ratio=1 → 100, 偏离越多分越低
    const score = Math.exp(-0.5 * ((ratio - 1) / 0.5) ** 2) * 100;
    frameScores.push(score);
  }
  
  return {
    overall: mean(frameScores),
    perFrame: frameScores
  };
}

function computeJerkProfile(frames) {
  // velocity → acceleration → jerk
  const vel = [], acc = [], jerk = [];
  for (let i = 1; i < frames.length; i++) {
    vel.push(landmarkDistance(frames[i].landmarks, frames[i-1].landmarks));
  }
  for (let i = 1; i < vel.length; i++) {
    acc.push(Math.abs(vel[i] - vel[i-1]));
  }
  for (let i = 1; i < acc.length; i++) {
    jerk.push(Math.abs(acc[i] - acc[i-1]));
  }
  return jerk;
}
```

### 节奏感 v2

```javascript
function computeRhythm(alignment, mappedPairs) {
  // 维度 1: 音频对齐置信度 (音乐是否匹配)
  const audioSync = alignment.confidence * 100;
  
  // 维度 2: 时间映射的均匀性 (动作是否跟拍)
  const timeDiffs = mappedPairs.map(p => 
    Math.abs(p.teacherTime - p.studentTime / alignment.tempoRatio)
  );
  const avgDiff = mean(timeDiffs);
  const timeSync = Math.max(0, (1 - avgDiff / 0.5)) * 100; // 0.5秒容差
  
  // 维度 3: 变速检测惩罚 (如果学生用了不同速度)
  const tempoMatch = alignment.tempoRatio;
  const tempoPenalty = tempoMatch >= 0.95 && tempoMatch <= 1.05 ? 100 :
                       tempoMatch >= 0.8 && tempoMatch <= 1.2 ? 70 : 40;
  
  return {
    overall: audioSync * 0.4 + timeSync * 0.4 + tempoPenalty * 0.2,
    audioSync,
    timeSync,
    tempoRatio: alignment.tempoRatio
  };
}
```

### 延展性 (Extension) v2

衡量动作是否充分伸展到位——区分"做到了"和"做到位了"。

```javascript
function computeExtension(mappedPairs, teacherFrames) {
  // 定义肢体段
  const LIMB_PAIRS = [
    { name: 'left_arm', joints: [11, 13, 15], weight: 1.2 },   // 肩→肘→腕
    { name: 'right_arm', joints: [12, 14, 16], weight: 1.2 },
    { name: 'left_leg', joints: [23, 25, 27], weight: 1.2 },   // 髋→膝→踝
    { name: 'right_leg', joints: [24, 26, 28], weight: 1.2 },
    { name: 'left_body_line', points: [15, 11, 23, 27], weight: 0.8 },
    { name: 'right_body_line', points: [16, 12, 24, 28], weight: 0.8 },
  ];
  
  const frameScores = mappedPairs.map(({ teacher, student }) => {
    let totalScore = 0, totalWeight = 0;
    for (const limb of LIMB_PAIRS) {
      if (limb.joints) {
        // 三点关节角度：角度大=伸展充分
        const tAngle = jointAngle(teacher.landmarks, limb.joints);
        const sAngle = jointAngle(student.landmarks, limb.joints);
        if (tAngle > 10) {
          const ratio = Math.min(sAngle / tAngle, 1.2);
          totalScore += limb.weight * Math.min(ratio, 1.0) * 100;
          totalWeight += limb.weight;
        }
      } else if (limb.points) {
        // 多点延展比：端到端直线距离 / 各段总长
        const tExt = limbExtensionRatio(teacher.landmarks, limb.points);
        const sExt = limbExtensionRatio(student.landmarks, limb.points);
        if (tExt > 0.3) {
          const ratio = Math.min(sExt / tExt, 1.2);
          totalScore += limb.weight * Math.min(ratio, 1.0) * 100;
          totalWeight += limb.weight;
        }
      }
    }
    return totalWeight > 0 ? totalScore / totalWeight : 70;
  });
  return { overall: mean(frameScores), perFrame: frameScores };
}
```

### 总分 (新四维)

```javascript
total = accuracy * 0.5 + extension * 0.2 + fluency * 0.15 + rhythm * 0.15
```

**权重理由**：
- 准确度 50%：动作做对是基础
- 延展性 20%：舞蹈质感核心（区分"做到了"和"做到位了"）
- 流畅度 15%：动作衔接自然度
- 节奏感 15%：是否跟上音乐节拍

---

## 结果展示增强

### 新增展示项
1. **音频对齐信息**: "检测到音乐变速 0.8x，已自动对齐" 或 "音乐匹配度 95%"
2. **有效对比时长**: "共对比 28.5 秒（教师 30s，学生 35s 中 28.5s 有效匹配）"
3. **逐段评分**: 每段独立计算三维度，不再用全局混入
4. **关键帧高亮**: 标记分数最低的 3 个时间点，建议重点练习

---

## 实施路线

| Phase | 内容 | 工作量 |
|-------|------|--------|
| **Phase 1** | 音频提取 + Chromagram | 3h |
| **Phase 2** | 音频 Cross-Correlation 对齐 + 变速检测 | 4h |
| **Phase 3** | 帧重映射 + 加权欧氏距离准确度 | 2h |
| **Phase 4** | Jerk-ratio 流畅度 + 节奏感 | 2h |
| **Phase 5** | UI 更新（对齐信息、逐段三维评分） | 2h |
| **Phase 6** | Evals 测试集 | 3h |
| **总计** | | **~16h** |

---

## 降级策略

如果两段视频没有音频（静音），或音频完全不同（一个有音乐一个没有）:
1. 检测音频能量，如果一方 < -40dB，标记为"静音"
2. 静音情况下，fallback 到 DTW 动作对齐（当前方案）
3. 音频完全不同时，提示用户"未检测到相同音乐，使用动作对齐模式"
