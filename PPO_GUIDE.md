# PPO Training Guide for ALFWorld

This guide explains how to use the newly implemented PPO (Proximal Policy Optimization) agent for training on ALFWorld tasks.

## What's New

### 1. PPO Agent Implementation
- **File**: `alfworld/agents/agent/text_ppo_agent.py`
- **Features**:
  - Actor-Critic architecture with shared policy and value networks
  - Generalized Advantage Estimation (GAE)
  - Clipped surrogate objective
  - Entropy regularization
  - Mini-batch updates with multiple epochs

### 2. Training Script
- **File**: `scripts/train_ppo.py`
- **Command-line arguments**:
  - `--config`: Path to config file (default: `ppo_config.yaml`)
  - `--cuda_device`: CUDA device ID (default: `"0"`)
  - `--output_dir`: Directory to save models and logs (default: `./ppo_output`)

### 3. Configuration File
- **File**: `scripts/ppo_config.yaml`
- **Key PPO parameters**:
  ```yaml
  ppo:
    clip_epsilon: 0.2          # PPO clipping parameter
    ppo_epochs: 4              # number of epochs per update
    minibatch_size: 4          # minibatch size
    value_loss_coef: 0.5       # value loss coefficient
    entropy_coef: 0.01         # entropy bonus coefficient
    max_grad_norm: 0.5         # gradient clipping
    gae_lambda: 0.95           # GAE lambda
    discount_gamma: 0.99       # discount factor
    normalize_advantages: True # normalize advantages
  ```

## How to Use

### Basic Training

Navigate to the scripts directory and run:

```bash
cd scripts
python train_ppo.py
```

### Advanced Usage

**Specify GPU device:**
```bash
python train_ppo.py --cuda_device 0
```

**Custom output directory:**
```bash
python train_ppo.py --output_dir ./my_ppo_results
```

**Custom configuration:**
```bash
python train_ppo.py --config my_custom_ppo_config.yaml
```

**Complete example:**
```bash
python train_ppo.py \
    --config ppo_config.yaml \
    --cuda_device 1 \
    --output_dir ./experiments/ppo_run_001
```

## Key Differences from DQN

| Aspect | DQN | PPO |
|--------|-----|-----|
| **Learning Type** | Off-policy | On-policy |
| **Memory** | Prioritized replay buffer | Rollout buffer (cleared after each update) |
| **Action Selection** | ε-greedy exploration | Stochastic policy |
| **Value Estimation** | Q-values only | Both policy and value function |
| **Update Frequency** | Every k steps | After each episode rollout |
| **Sample Efficiency** | Higher (reuses old data) | Lower (only uses recent data) |
| **Stability** | Can be unstable | More stable with clipping |

## PPO Hyperparameter Guide

### Critical Parameters

1. **clip_epsilon (0.1 - 0.3)**
   - Controls how much the policy can change in one update
   - Smaller values → more conservative updates
   - Recommended: 0.2

2. **learning_rate (1e-4 - 5e-4)**
   - PPO typically uses smaller learning rates than DQN
   - Recommended: 3e-4

3. **ppo_epochs (3 - 10)**
   - Number of gradient steps per data batch
   - More epochs → better data utilization but risk overfitting
   - Recommended: 4

4. **gae_lambda (0.9 - 0.99)**
   - Trade-off between bias and variance in advantage estimation
   - Higher values → less bias but more variance
   - Recommended: 0.95

5. **entropy_coef (0.001 - 0.01)**
   - Encourages exploration
   - Too high → random behavior
   - Too low → premature convergence
   - Recommended: 0.01

### Tuning Tips

**If training is unstable:**
- Decrease learning rate (e.g., 1e-4)
- Decrease clip_epsilon (e.g., 0.1)
- Increase minibatch_size
- Decrease ppo_epochs

**If training is too slow:**
- Increase learning rate (but carefully)
- Increase batch_size (number of parallel environments)
- Increase entropy_coef for more exploration

**If agent gets stuck in local optimum:**
- Increase entropy_coef
- Decrease value_loss_coef
- Try different random seeds

## Architecture Details

### Network Structure

```
Input: Observation + Task Description (text)
  ↓
BERT Embeddings (frozen)
  ↓
Encoder Blocks (with attention)
  ↓
Aggregation (observation + task)
  ↓
GRU Cell (if recurrent=True)
  ↓
├─→ Policy Head → Action Scores → Softmax → Action Selection
└─→ Value Head → State Value
```

### Loss Function

```
Total Loss = Policy Loss + value_loss_coef × Value Loss - entropy_coef × Entropy

where:
  Policy Loss = -min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage)
  Value Loss = MSE(predicted_value, target_return)
  Entropy = -Σ(p × log(p))
```

## Monitoring Training

The training script reports:
- **Episode**: Current episode number
- **time spent**: Total training time
- **step**: Total environment steps
- **loss**: Policy loss and value loss
- **reward**: Average reward per episode
- **step**: Average steps per episode

### Using Visdom (Optional)

Enable visualization by setting in config:
```yaml
general:
  visdom: True
```

Then start visdom server:
```bash
python -m visdom.server
```

Access at: http://localhost:8097

## Expected Performance

### Sample Complexity
- PPO typically requires **more samples** than DQN due to on-policy nature
- However, PPO often achieves **more stable** learning curves
- Convergence time: ~20k-40k episodes (depending on task complexity)

### Computational Requirements
- **Memory**: ~8-16 GB RAM (less than DQN due to no replay buffer)
- **GPU**: NVIDIA GPU with 4+ GB VRAM recommended
- **Training Time**: ~2-5 days on single GPU for full training

## Troubleshooting

### Common Issues

**1. Import Error: "No module named 'alfworld'"**
```bash
# Install alfworld package
pip install -e .
```

**2. CUDA Out of Memory**
- Reduce `batch_size` in config
- Reduce `minibatch_size` in config
- Use smaller `block_hidden_dim`

**3. NaN losses**
- Check learning rate (might be too high)
- Ensure proper normalization of advantages
- Check for division by zero in advantage computation

**4. Policy not learning**
- Check that rewards are being received
- Verify action candidates are not empty
- Increase entropy_coef for more exploration
- Check if value estimates are reasonable

**5. "ALFWORLD_DATA environment variable not set"**
```bash
export ALFWORLD_DATA=/path/to/alfworld/data
```

## File Structure

```
alfworld_meta_dqn/
├── alfworld/
│   └── agents/
│       └── agent/
│           ├── text_ppo_agent.py    # PPO agent implementation
│           └── __init__.py           # Updated to export PPO agent
├── scripts/
│   ├── train_ppo.py                 # PPO training script
│   ├── ppo_config.yaml              # PPO configuration
│   ├── train_dqn.py                 # Original DQN training
│   └── dqn_config.yaml              # Original DQN config
└── PPO_GUIDE.md                      # This file
```

## Comparison with Meta-DQN

The original project implements Meta-DQN. Here's how PPO compares:

### Advantages of PPO:
- ✅ More stable training
- ✅ Simpler implementation (no target network, replay buffer)
- ✅ Better for continuous learning scenarios
- ✅ More sample efficient in some environments

### Advantages of Meta-DQN:
- ✅ Better sample efficiency with experience replay
- ✅ Off-policy learning allows for more data reuse
- ✅ Meta-learning capabilities for fast adaptation

**Recommendation**:
- Use **PPO** for initial experiments and stable baselines
- Use **Meta-DQN** for few-shot adaptation and transfer learning

## References

1. Schulman et al. (2017). "Proximal Policy Optimization Algorithms" [[arXiv]](https://arxiv.org/abs/1707.06347)
2. Schulman et al. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation" [[arXiv]](https://arxiv.org/abs/1506.02438)
3. Shridhar et al. (2021). "ALFWorld: Aligning Text and Embodied Environments for Interactive Learning"
4. Fakoor et al. (2020). "Meta-Q-Learning"

## Next Steps

After implementing PPO, you might want to:

1. **Implement PPO evaluation script** (similar to `adapt_dqn.py`)
2. **Add curriculum learning** for better sample efficiency
3. **Implement multi-task PPO** for transfer learning
4. **Add intrinsic motivation** (curiosity, RND, etc.)
5. **Experiment with different network architectures**

## Contributing

If you improve the PPO implementation or fix bugs, please:
1. Test thoroughly on at least one task type
2. Document any hyperparameter changes
3. Update this guide if needed

## License

Same as the parent project (MIT License)
