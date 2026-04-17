# DanceFlow 评分体系评估与改进

## 1. 当前评分算法分析

### 总分公式
```
总分 = 动作准确度 × 0.5 + 流畅度 × 0.3 + 节奏感 × 0.2
```

### 各维度算法

#### 动作准确度 (Accuracy) — 权重 50%
```
方法: DTW 对齐后，沿 alignment path 逐帧计算余弦相似度
输入: teacher 33个关键点 flatten → 99维向量, student 同理
计算: cosine_similarity(teacher_frame, student_frame)
映射: (sim - 0.5) × 2 → 0~1  (即 sim=0.5→0%, sim=1.0→100%)
聚合: 所有帧的平均值 × 100
```

**问题分析:**
- ✅ 余弦相似度衡量姿态方向，合理
- ❌ 映射函数 `(sim-0.5)*2` 太宽松。实测中两个完全不同的人做不同动作，余弦相似度通常也在 0.7-0.9（因为人体结构相似，大部分关键点位置相对固定）
- ❌ 没有对关键点加权——手指末梢和躯干中心权重一样，但舞蹈中手臂、腿部动作差异远大于躯干
- ❌ 没有考虑关键点置信度（visibility），低置信度的点不应参与评分

#### 流畅度 (Fluency) — 权重 30%
```
方法: 计算 student 的逐帧速度变化（加速度），越小越流畅
计算: velocity[i] = euclidean(frame[i], frame[i-1])
      accel[i] = |velocity[i] - velocity[i-1]|
      avgAccel = mean(accel)
映射: (1 - avgAccel / 0.5) × 100
```

**问题分析:**
- ❌ **致命问题**: 归一化阈值 `0.5` 是随意设定的。MediaPipe 关键点坐标是 [0,1] 归一化的，全身 flatten 后的欧氏距离取决于运动幅度。跳舞动作大 → 速度大 → 加速度大 → 流畅度低。这意味着**动作幅度越大的舞蹈，流畅度评分越低**，完全错误
- ❌ 流畅度应该与教师对比，而不是绝对值。教师的舞蹈也有速度变化（这是正常的），学生的流畅度应该是"运动曲线是否平滑"，而非"是否不动"
- ❌ 只看 student 不看 teacher，无法区分"学生动作生硬"和"舞蹈本身就有急停急起"

#### 节奏感 (Rhythm) — 权重 20%
```
方法: 沿 DTW path 计算 teacher 和 student 的速度曲线，做 Pearson 相关系数
映射: (correlation + 1) / 2 × 100  (即 corr=-1→0%, corr=0→50%, corr=1→100%)
```

**问题分析:**
- ✅ Pearson 相关系数衡量速度曲线形状的一致性，思路正确
- ❌ DTW path 上提取速度曲线有问题：DTW 会把一帧映射到多帧（多对一），导致速度曲线被扭曲
- ❌ 没有处理 DTW path 中重复索引的情况（`ti1===ti0` 时跳过，导致 teacher 和 student 速度序列长度不同）
- ❌ `minLen` 截断导致丢弃大量数据

### 逐段评分
```
方法: 将 accuracy 序列等分为 N 段，每段计算准确度均值
      然后混入全局 fluency 和 rhythm: segScore = accuracy*0.5 + fluency*0.3 + rhythm*0.2
```

**问题分析:**
- ❌ 用全局 fluency/rhythm 混入每段，导致每段分数差异很小（只有 accuracy 在变）
- ❌ 应该计算每段自己的 fluency 和 rhythm

---

## 2. 改进方案 (v2)

### 核心原则
1. **一切以教师为参照** — 不评判绝对好坏，只衡量与教师的差距
2. **关键点加权** — 四肢末端（手、脚）权重高于躯干
3. **置信度过滤** — 低置信度关键点不参与评分
4. **归一化基于教师** — 用教师自身的运动范围做归一化基准
5. **逐段独立评分** — 每段有自己的三维度分

### 新权重
```
总分 = 动作准确度 × 0.6 + 流畅度 × 0.2 + 节奏感 × 0.2
```
理由: 对于舞蹈学习，"动作对不对"比"流不流畅"更重要。初学者先把动作做对，再追求流畅和节奏。

### 新算法

#### v2 动作准确度

```javascript
// 关键点权重（MediaPipe 33 点）
const JOINT_WEIGHTS = {
  // 躯干 (权重低，人体结构相似性高)
  hips: 0.5, spine: 0.5, shoulders: 0.6,
  // 上肢 (权重高，舞蹈差异大)
  elbows: 1.0, wrists: 1.2, hands: 0.8,
  // 下肢 (权重高)
  knees: 1.0, ankles: 1.2, feet: 0.8,
  // 头部 (权重中)
  head: 0.7
};

// 1. 逐关键点计算欧氏距离（不用余弦相似度）
// 2. 按权重加权
// 3. 用教师自身的关键点运动范围归一化
for each aligned frame pair (teacher_i, student_i):
  for each joint j:
    if visibility[j] < 0.5: skip
    dist = euclidean(teacher_j, student_j)  // 单个关键点的位置差
    range_j = max_range_of_joint_j_in_teacher_video  // 教师该关节的运动幅度
    normalized_error = dist / max(range_j, epsilon)
    weighted_error += JOINT_WEIGHTS[j] * min(normalized_error, 1.0)
  
  frame_accuracy = (1 - weighted_error / sum_weights) * 100

accuracy = mean(all frame_accuracies)
```

**改进点:**
- 用欧氏距离而非余弦相似度（更直观地衡量"位置差"）
- 关键点加权（手脚权重 > 躯干）
- 用教师运动范围归一化（消除舞蹈类型差异）
- 过滤低置信度关键点

#### v2 流畅度

```javascript
// 不再看绝对加速度，而是对比教师和学生的 jerk 比率
// jerk = 加速度的变化率（三阶导数）

teacher_jerk = compute_jerk(teacher_velocity_profile)
student_jerk = compute_jerk(student_velocity_profile)

// 逐段对比：学生的 jerk 是否比教师大（生硬）或小（太慢）
for each segment:
  jerk_ratio = student_jerk / max(teacher_jerk, epsilon)
  // 理想值是 1.0（和教师一样流畅）
  // > 1.5 说明学生动作生硬
  // < 0.5 说明学生动作迟缓/没跟上
  fluency_score = gaussian(jerk_ratio, mu=1.0, sigma=0.5) * 100

fluency = mean(all segment_fluency_scores)
```

**改进点:**
- 与教师对比，而非绝对值
- 用 jerk ratio 而非加速度绝对值
- 高斯映射：ratio=1 时满分，偏离越多分越低

#### v2 节奏感

```javascript
// 方法1: DTW warping path 的偏离程度
// 理想情况：对角线（1:1 同步）
// 实际：偏离对角线 = 节奏不同步

diagonal = [0, 1, 2, ..., min(T, S)]
warping_deviation = mean(|path[i] - diagonal[i]| for i in path)
max_deviation = max(T, S) / 2
rhythm = (1 - warping_deviation / max_deviation) * 100

// 方法2: 速度曲线的 cross-correlation (替代 Pearson)
// 允许小幅时移，找最大相关
for lag in [-max_lag, ..., max_lag]:
  corr[lag] = cross_correlation(teacher_speed, student_speed, lag)
best_corr = max(corr)
rhythm = (best_corr + 1) / 2 * 100
```

**改进点:**
- 方法 1 直接衡量时间同步性
- 方法 2 用 cross-correlation 替代 Pearson，容忍小幅时移
- 两者取平均更鲁棒

---

## 3. 评估标准 (Evals)

### 测试用例设计

| Case | Teacher | Student | 预期总分 | 预期准确度 | 预期流畅度 | 预期节奏 |
|------|---------|---------|----------|-----------|-----------|---------|
| 1. 同一视频 | A | A (复制) | 95-100 | 98-100 | 95-100 | 95-100 |
| 2. 同一人重复 | A | A (重录) | 80-95 | 85-95 | 80-90 | 80-90 |
| 3. 不同人相似动作 | A | B (模仿A) | 60-85 | 65-85 | 60-80 | 60-80 |
| 4. 不同人不同动作 | A | C (完全不同) | 10-40 | 15-40 | 30-60 | 10-30 |
| 5. 静止 vs 运动 | A (跳舞) | D (站着不动) | 5-20 | 10-25 | N/A | 5-15 |
| 6. 同动作不同速度 | A | E (A的2倍速) | 50-70 | 75-90 | 40-60 | 20-40 |

### 合理性检查规则

1. **同视频 ≥ 95**: 同一个视频对比自己，得分必须接近满分
2. **不同动作 ≤ 40**: 完全不同的舞蹈，得分不应超过 40
3. **准确度 ≥ 逐段平均**: 全局准确度不应低于逐段准确度的加权平均
4. **总分 ∈ [min(三维度), max(三维度)]**: 总分必须在三个维度之间
5. **逐段评分单调性**: 相邻段的分差不应超过 30（防止噪声导致跳变）

### 持续评估流程

```
1. 维护 5+ 对标注视频（已知难度等级）
2. 每次修改评分算法后，跑所有测试用例
3. 检查是否满足合理性规则
4. 记录分数变化的 diff
5. 如果任何规则违反，回滚修改
```

---

## 4. 实施优先级

| 优先级 | 改进项 | 影响 | 工作量 |
|--------|--------|------|--------|
| P0 | 修复流畅度算法（对比教师） | 解决流畅度虚低问题 | 2h |
| P0 | 关键点加权 | 提升准确度区分度 | 1h |
| P1 | 置信度过滤 | 减少噪声 | 0.5h |
| P1 | 教师运动范围归一化 | 适配不同舞蹈类型 | 1h |
| P1 | 逐段独立三维评分 | 解决段分与总分矛盾 | 1h |
| P2 | DTW warping deviation 节奏评分 | 更准确的节奏衡量 | 1h |
| P2 | 建立评估测试集 | 持续质量保证 | 3h |
