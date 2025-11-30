# PPO 快速开始指南

## 概述

本项目现已实现基于 **PPO (Proximal Policy Optimization)** 算法的训练方法。PPO 是一种稳定且高效的强化学习算法，相比原有的 DQN 方法具有以下特点：

- ✅ **更稳定**：使用 clipped surrogate objective 防止策略更新过大
- ✅ **更简单**：不需要经验回放缓冲区和目标网络
- ✅ **更易调试**：超参数相对容易调整

## 新增文件

```
alfworld_meta_dqn/
├── alfworld/agents/agent/
│   └── text_ppo_agent.py          # PPO Agent 实现
├── scripts/
│   ├── train_ppo.py                # PPO 训练脚本
│   ├── ppo_config.yaml             # PPO 配置文件
│   └── test_ppo_import.py          # 测试脚本
├── PPO_GUIDE.md                    # 详细文档（英文）
└── PPO_QUICKSTART.md               # 本文件
```

## 快速开始

### 1. 准备环境

确保已安装所有依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. 设置数据路径

```bash
export ALFWORLD_DATA=/path/to/your/alfworld/data
```

### 3. 运行训练

```bash
cd scripts
python train_ppo.py
```

### 4. 自定义训练

指定 GPU 和输出目录：

```bash
python train_ppo.py --cuda_device 0 --output_dir ./my_ppo_results
```

## 主要配置参数

在 `scripts/ppo_config.yaml` 中可以调整以下关键参数：

### 训练相关
```yaml
general:
  training:
    batch_size: 10              # 并行环境数量
    max_episode: 50000          # 总训练轮数
    optimizer:
      learning_rate: 0.0003     # 学习率（PPO 通常使用较小值）
```

### PPO 超参数
```yaml
ppo:
  clip_epsilon: 0.2             # PPO 裁剪参数（0.1-0.3）
  ppo_epochs: 4                 # 每次更新的训练轮数
  minibatch_size: 4             # 小批量大小
  value_loss_coef: 0.5          # 价值损失系数
  entropy_coef: 0.01            # 熵奖励系数（鼓励探索）
  gae_lambda: 0.95              # GAE lambda 参数
  discount_gamma: 0.99          # 折扣因子
```

## 与 DQN 的对比

| 特性 | DQN (原实现) | PPO (新实现) |
|------|-------------|-------------|
| 学习方式 | Off-policy | On-policy |
| 内存需求 | 高（需要大型回放缓冲区）| 低（仅存储当前轨迹）|
| 训练稳定性 | 中等 | 高 |
| 样本效率 | 高 | 中等 |
| 探索策略 | ε-greedy | 随机策略采样 |
| 适用场景 | 离线学习、大数据 | 在线学习、连续任务 |

## 训练监控

训练过程中会输出：

```
Episode: 100 | time spent: 0:15:32 | step: 5000 | loss: 0.234 0.567 | reward: 0.45 | step: 12.3
         ↑              ↑              ↑           ↑       ↑           ↑           ↑
      轮数         已用时间        总步数      策略损失 价值损失      平均奖励    平均步数
```

### 启用可视化（可选）

在配置文件中设置：
```yaml
general:
  visdom: True
```

然后启动 visdom 服务器：
```bash
python -m visdom.server
```

访问 http://localhost:8097 查看训练曲线。

## 常见问题

### Q1: CUDA Out of Memory
**解决方法**：
- 减小 `batch_size`（如从 10 改为 5）
- 减小 `minibatch_size`（如从 4 改为 2）
- 减小 `block_hidden_dim`（在 model 配置中）

### Q2: 训练不收敛
**解决方法**：
- 降低学习率（如改为 1e-4）
- 增加 `entropy_coef` 以促进探索（如改为 0.02）
- 检查奖励信号是否正常

### Q3: 环境变量未设置
```bash
# 如果提示 ALFWORLD_DATA 未设置
export ALFWORLD_DATA=/path/to/alfworld/data
```

### Q4: 导入错误
确保已安装包：
```bash
pip install -e .
```

## 测试安装

运行测试脚本验证安装：

```bash
cd scripts
python test_ppo_import.py
```

如果看到 "All tests passed!"，说明安装成功。

## 性能建议

### 推荐配置（快速实验）
```yaml
general:
  training:
    batch_size: 5
    max_episode: 10000
ppo:
  ppo_epochs: 3
  minibatch_size: 2
```

### 推荐配置（最佳性能）
```yaml
general:
  training:
    batch_size: 20
    max_episode: 50000
ppo:
  ppo_epochs: 4
  minibatch_size: 8
```

## 高级用法

### 1. 多 GPU 训练

暂不支持，但可以通过运行多个实例实现：

```bash
# Terminal 1
python train_ppo.py --cuda_device 0 --output_dir ./run_1

# Terminal 2
python train_ppo.py --cuda_device 1 --output_dir ./run_2
```

### 2. 继续训练（TODO）

目前需要手动实现，可以在配置中设置：
```yaml
general:
  checkpoint:
    load_pretrained: True
    load_from_tag: 'ppo_alfworld_episode_10000'
```

### 3. 评估模型

可以参考 `adapt_dqn.py` 实现类似的 PPO 评估脚本。

## 代码结构

### PPO Agent 核心方法

- `choose_action_with_log_prob()`: 根据策略采样动作并计算对数概率
- `compute_gae()`: 计算广义优势估计
- `update()`: 使用 PPO 算法更新策略和价值网络
- `finish_of_episode()`: 每轮结束时调用，执行策略更新

### 训练流程

```python
1. 初始化环境和 Agent
2. for each episode:
    3. 重置环境
    4. for each step:
        5. 根据当前策略采样动作
        6. 执行动作，获得奖励
        7. 存储经验到 PPO 缓冲区
    8. 计算 GAE 优势
    9. 使用 PPO 更新策略（多个 epoch）
    10. 清空缓冲区
```

## 预期结果

在 ALFWorld 任务上：
- **收敛时间**：约 20,000 - 40,000 轮
- **最终性能**：平均成功率 > 50%（取决于任务难度）
- **训练时长**：单 GPU 约 2-3 天

## 下一步

1. ✅ 基础 PPO 实现完成
2. ⏳ 实现 PPO 评估脚本
3. ⏳ 添加课程学习
4. ⏳ 优化内存使用
5. ⏳ 支持多任务训练

## 反馈与贡献

如遇到问题或有改进建议，欢迎：
- 提交 Issue
- 发起 Pull Request
- 查看详细文档：[PPO_GUIDE.md](PPO_GUIDE.md)

## 许可证

与原项目相同（MIT License）
