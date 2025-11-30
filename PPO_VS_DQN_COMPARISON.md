# PPO vs DQN 详细对比

## 核心问题回答

### 1️⃣ 观察空间（Observation Space）

**答案：✅ 完全一样**

两者都使用相同的观察空间处理方式：

```python
# 共同的观察处理流程（在 BaseAgent 中定义）

# 1. 观察字符串格式
observation_strings = [
    "You are in a kitchen. You see a apple 1, a fridge 1, ..."
]

# 2. 任务描述
task_desc_strings = [
    "Your task is to: put a clean apple in fridge."
]

# 3. 文本编码（使用 DistilBERT）
h_obs, obs_mask = self.encode(observation_strings, use_model="online")
h_td, td_mask = self.encode(task_desc_strings, use_model="online")

# 4. 信息聚合
aggregated_obs = self.online_net.aggretate_information(
    h_obs, h_td, obs_mask, td_mask
)
```

#### 详细说明

| 组成部分 | DQN | PPO | 是否相同 |
|---------|-----|-----|---------|
| 原始输入 | 文本字符串 | 文本字符串 | ✅ 相同 |
| 编码器 | DistilBERT embeddings | DistilBERT embeddings | ✅ 相同 |
| 编码维度 | 768 → 64 (block_hidden_dim) | 768 → 64 (block_hidden_dim) | ✅ 相同 |
| 观察历史 | ObservationPool (容量=3) | ObservationPool (容量=3) | ✅ 相同 |
| 循环状态 | GRU (如果 recurrent=True) | GRU (如果 recurrent=True) | ✅ 相同 |

---

### 2️⃣ 动作空间（Action Space）

**答案：✅ 基本一样（但选择机制不同）**

#### 动作候选集生成

```python
# 两者完全相同
action_candidate_list = [
    ["go to fridge 1", "open fridge 1", "take apple 1"],  # 环境1的可选动作
    ["examine apple 1", "put apple 1 in fridge 1"],      # 环境2的可选动作
]
```

#### 动作评分（Scoring）

```python
# DQN 和 PPO 都使用相同的 action_scoring 方法（继承自 BaseAgent）
action_scores, action_masks, current_dynamics = self.action_scoring(
    action_candidate_list, h_obs, obs_mask, h_td, td_mask,
    previous_dynamics, use_model="online"
)
# 输出：
# action_scores: [batch_size, max_num_actions]  # 每个动作的分数
# action_masks:  [batch_size, max_num_actions]  # 有效动作掩码（1=有效，0=无效）
```

#### 关键差异：动作选择机制

| 方面 | DQN | PPO |
|-----|-----|-----|
| **训练时选择** | ε-greedy（贪婪 + 随机） | 随机采样（按概率分布） |
| **评估时选择** | Argmax（选最大Q值） | Argmax（选最大概率） |
| **输出解释** | Q 值（动作价值） | Logits → 概率分布 |

##### DQN 动作选择（训练时）

```python
# DQN: ε-greedy 策略
# text_dqn_agent.py: 84-113

# 1. 计算最大Q值动作
action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
# argmax(Q(s,a))

# 2. 随机选择动作
action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

# 3. ε-greedy 混合
rand_num = np.random.uniform(0.0, 1.0, size=(batch_size,))
less_than_epsilon = (rand_num < self.epsilon)  # 概率 ε

chosen_indices = (
    less_than_epsilon * action_indices_random +      # ε 概率随机
    (1 - less_than_epsilon) * action_indices_maxq    # 1-ε 概率贪婪
)
```

##### PPO 动作选择（训练时）

```python
# PPO: 随机策略采样
# text_ppo_agent.py: 125-180

# 1. 将动作分数转换为概率
action_probs = masked_softmax_with_temperature(action_scores, action_masks, temperature=1.0)
# softmax(logits) → 概率分布

# 2. 按概率分布采样
for i in range(batch_size):
    valid_actions = action_masks[i].nonzero(as_tuple=True)[0]
    probs = action_probs[i][valid_actions]
    probs = probs / probs.sum()  # 重新归一化

    # 多项式采样（而非 argmax）
    action_idx = torch.multinomial(probs, 1).item()
    log_prob = torch.log(action_probs[i][action_idx] + 1e-10)

    action_indices.append(action_idx)
    log_probs.append(log_prob)  # PPO 需要记录对数概率！
```

#### 动作空间类型配置

```yaml
# ppo_config.yaml 和 dqn_config.yaml 中都有
action_space: "admissible"  # 从环境提供的可执行动作中选择

# 其他选项（两者都支持但本实现主要用 admissible）：
# - "generation": 使用 seq2seq 生成动作文本
# - "beam_search_choice": 使用 beam search 生成候选再选择
```

#### 对比总结

| 组成部分 | DQN | PPO | 是否相同 |
|---------|-----|-----|---------|
| 候选动作来源 | 环境提供 (admissible commands) | 环境提供 (admissible commands) | ✅ 相同 |
| 动作编码 | BERT + 评分网络 | BERT + 评分网络 | ✅ 相同 |
| 动作掩码 | 使用掩码过滤无效动作 | 使用掩码过滤无效动作 | ✅ 相同 |
| 输出含义 | Q 值 | 策略 logits | ❌ 不同 |
| 选择策略（训练） | ε-greedy | 随机采样 | ❌ 不同 |
| 选择策略（评估） | Argmax Q | Argmax probability | ✅ 基本相同 |
| 需要记录概率 | 否 | 是（log_prob） | ❌ 不同 |

---

### 3️⃣ 文本处理方法

**答案：✅ 完全一样**

两者共享完全相同的文本处理流程（都继承自 `BaseAgent`）：

#### 完整文本处理流程

```python
# 1. 预处理（BaseAgent 中定义）
# base_agent.py: 397-418

# 任务描述预处理
task_desc_strings = agent.preprocess_task(task_desc_strings)
# 例：移除多余空格，标准化格式

# 观察字符串预处理
observation_strings = agent.preprocess_observation(observation_strings)
# 例：清理文本，统一格式

# 动作候选预处理
action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
# 例：标准化动作文本


# 2. Tokenization（BaseAgent 中定义）
# base_agent.py: 283-290

word_id_list = [
    self.tokenizer.encode(item, add_special_tokens=False)
    for item in input_strings
]
# 使用 DistilBERT tokenizer 将文本转换为 token IDs

input_word = pad_sequences(word_id_list, maxlen=max_len(word_id_list) + 3)
# 填充到相同长度


# 3. BERT Embedding（Policy 模型中）
# model.py: 91-109

embeddings = self.bert_model.embeddings(input_words)
# 使用预训练 DistilBERT 的 embedding 层（冻结参数）
# 输出维度：[batch, seq_len, 768]

embeddings = self.word_embedding_prj(embeddings)
# 投影到 block_hidden_dim (默认 64)
# 输出维度：[batch, seq_len, 64]


# 4. Encoder（Policy 模型中）
# model.py: 118-127

for i in range(self.encoder_layers):
    encoding_sequence = self.encoder[i](
        encoding_sequence,
        input_word_masks,
        squared_mask,
        i * (self.encoder_conv_num + 2) + 1,
        self.encoder_layers
    )
# 使用卷积 + 自注意力的 Encoder Blocks


# 5. 聚合观察和任务（Policy 模型中）
# model.py: 129-134

aggregated_obs_representation = self.aggregation_attention(
    h_obs, h_td, obs_mask, td_mask
)
# 使用 CQAttention（类似 BiDAF）融合观察和任务信息
```

#### 文本处理架构对比

```
                 DQN                            PPO
                  │                              │
                  └──────────┬───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  BaseAgent 基类  │
                    │  (共享代码)      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         Tokenizer      Preprocessing   Encoding
         (DistilBERT)    (标准化)      (共享网络)
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Policy Network  │
                    │  (共享架构)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  动作评分层      │
                    │  (共享)         │
                    └─────────────────┘
```

#### 详细对比表

| 文本处理环节 | DQN | PPO | 代码位置 | 是否相同 |
|------------|-----|-----|---------|---------|
| **Tokenizer** | DistilBERT | DistilBERT | BaseAgent.__init__ | ✅ 相同 |
| **词汇表** | 28996 个 token | 28996 个 token | BaseAgent.__init__ | ✅ 相同 |
| **预处理** | preproc() | preproc() | BaseAgent | ✅ 相同 |
| **Embedding** | BERT embeddings (768维) | BERT embeddings (768维) | Policy.get_bert_embeddings | ✅ 相同 |
| **投影层** | Linear(768→64) | Linear(768→64) | Policy.word_embedding_prj | ✅ 相同 |
| **Encoder** | EncoderBlock × N | EncoderBlock × N | Policy.encoder | ✅ 相同 |
| **注意力** | CQAttention | CQAttention | Policy.aggregation_attention | ✅ 相同 |
| **循环单元** | GRUCell (可选) | GRUCell (可选) | Policy.rnncell | ✅ 相同 |

---

## 核心差异总结

### 相同的部分 ✅

```python
# 这些组件在 DQN 和 PPO 中完全一样：

1. 观察空间处理
   - 文本输入格式
   - BERT tokenization
   - Embedding 和 encoding
   - 观察历史池（ObservationPool）

2. 动作空间结构
   - 候选动作来源（admissible commands）
   - 动作文本编码
   - 动作掩码机制

3. 文本处理
   - Tokenizer（DistilBERT）
   - 预处理流程
   - 编码器架构
   - 注意力机制

4. 网络架构（共享部分）
   - Policy Network（除了最后的 head）
   - Encoder Blocks
   - Aggregation layers
```

### 不同的部分 ❌

```python
# 主要差异在于：

1. 动作选择机制
   DQN:  ε-greedy（训练时）
         - ε 概率随机
         - 1-ε 概率选最大Q值

   PPO:  随机采样（训练时）
         - 按策略概率分布采样
         - 需要记录 log_prob

2. 网络输出头
   DQN:  Q 值头
         - action_scorer_linear_2: (hid → 1)
         - 输出每个动作的 Q(s,a) 值

   PPO:  策略头 + 价值头
         - 策略：action_scorer_linear_2 (hid → 1) → softmax → 概率
         - 价值：value_head (hid → hid → 1) → V(s)

3. 训练目标
   DQN:  最小化 TD error
         loss = (Q(s,a) - (r + γ * max Q(s',a')))²

   PPO:  最大化 clipped surrogate objective
         loss = -min(ratio * A, clip(ratio) * A)
              + value_loss - entropy_bonus

4. 记忆机制
   DQN:  Prioritized Replay Buffer (容量 500k)
         - 存储 (s, a, r, s', done)
         - 重复采样训练

   PPO:  Rollout Buffer
         - 存储完整轨迹
         - 用完即弃

5. 探索策略
   DQN:  ε 线性退火（0.3 → 0.1）
         - epsilon_anneal_episodes: 1000

   PPO:  策略熵正则化
         - entropy_coef: 0.01
         - 自然探索（随机策略）
```

---

## 代码对比示例

### 示例 1：动作选择对比

```python
# ============ DQN 动作选择 ============
# text_dqn_agent.py: admissible_commands_act()

# 获取 Q 值
action_scores, _, _ = self.action_scoring(...)  # Q 值

# ε-greedy
if random.uniform(0, 1) < self.epsilon:
    chosen_idx = random.choice(valid_actions)  # 随机
else:
    chosen_idx = argmax(action_scores)  # 贪婪（最大Q值）


# ============ PPO 动作选择 ============
# text_ppo_agent.py: choose_action_with_log_prob()

# 获取策略 logits
action_scores, _, _ = self.action_scoring(...)  # logits

# 转换为概率
probs = softmax(action_scores)  # 概率分布

# 随机采样（按概率）
chosen_idx = multinomial(probs)  # 根据概率采样
log_prob = log(probs[chosen_idx])  # 记录对数概率
```

### 示例 2：网络前向传播对比

```python
# ============ 共同部分（两者相同）============

# 1. 编码
h_obs, obs_mask = self.encode(observation_strings)  # BERT + Encoder
h_td, td_mask = self.encode(task_desc_strings)

# 2. 聚合
aggregated = self.online_net.aggretate_information(h_obs, h_td, obs_mask, td_mask)

# 3. 评分
action_scores = self.online_net.score_actions(...)  # [batch, num_actions]


# ============ DQN 独有 ============

# 输出 Q 值
q_values = action_scores  # 直接使用
action = argmax(q_values)  # 选择最大Q值


# ============ PPO 独有 ============

# 输出策略 + 价值
state_repr = self.online_net.masked_mean(aggregated)
value = self.value_head(state_repr)  # 价值函数 V(s)

probs = softmax(action_scores)  # 策略 π(a|s)
action = multinomial(probs)  # 采样
```

### 示例 3：文本处理（完全相同）

```python
# ============ DQN 和 PPO 都使用这个流程 ============
# base_agent.py

# 原始文本
obs = "You are in a kitchen. You see apple 1, fridge 1."
task = "Your task is to: put apple in fridge."

# Tokenization
obs_tokens = tokenizer.encode(obs)
# [2017, 2024, 1999, 1037, 3829, ...]

# Embedding
obs_emb = bert_embeddings(obs_tokens)
# [1, seq_len, 768]

# Projection
obs_proj = linear(obs_emb)
# [1, seq_len, 64]

# Encoding
obs_encoded = encoder_blocks(obs_proj)
# [1, seq_len, 64]
```

---

## 配置对比

### DQN 配置（dqn_config.yaml）
```yaml
rl:
  action_space: "admissible"
  epsilon_greedy:
    epsilon_anneal_episodes: 1000
    epsilon_anneal_from: 0.3
    epsilon_anneal_to: 0.1
  replay:
    replay_memory_capacity: 500000
    replay_batch_size: 64
    discount_gamma_game_reward: 0.9
```

### PPO 配置（ppo_config.yaml）
```yaml
ppo:
  action_space: "admissible"  # 相同！
  clip_epsilon: 0.2
  ppo_epochs: 4
  entropy_coef: 0.01
  discount_gamma: 0.99  # 稍有不同
  normalize_advantages: True
```

---

## 结论

| 组件 | 相似度 | 说明 |
|------|-------|------|
| **观察空间** | 100% | 完全相同的文本处理流程 |
| **动作空间结构** | 95% | 候选集相同，选择机制不同 |
| **文本处理** | 100% | 完全共享 BaseAgent 的实现 |
| **网络编码器** | 100% | 共享 Policy Network 的编码部分 |
| **输出头** | 30% | DQN 是 Q 值头，PPO 是策略+价值双头 |
| **训练算法** | 0% | 完全不同的 RL 算法 |

**总结**：PPO 和 DQN 在**输入处理**（观察、文本、动作候选）上**完全一致**，差异主要在**输出解释**（Q值 vs 概率）和**学习算法**（TD learning vs Policy gradient）。

这种设计使得两种算法可以在相同的环境和网络架构上公平对比！
