import copy
import operator
import logging
from queue import PriorityQueue

import numpy as np
import torch
import torch.nn.functional as F
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

from alfworld.agents.agent import BaseAgent
from alfworld.agents.modules.generic import to_np, to_pt, _words_to_ids, pad_sequences, preproc, max_len, ez_gather_dim_1, LinearSchedule, BeamSearchNode
from alfworld.agents.modules.layers import NegativeLogLoss, masked_mean, compute_mask, GetGenerationQValue


class PPOMemory:
    """
    Simple rollout buffer for PPO that stores trajectories from multiple episodes.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.observations = []
        self.task_descs = []
        self.actions = []
        self.action_indices = []
        self.action_candidates = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.dynamics = []

    def push(self, observation, task_desc, action, action_idx, action_candidate_list,
             reward, done, value, log_prob, dynamic):
        self.observations.append(observation)
        self.task_descs.append(task_desc)
        self.actions.append(action)
        self.action_indices.append(action_idx)
        self.action_candidates.append(action_candidate_list)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dynamics.append(dynamic)

    def get_batches(self):
        """Return all stored data as lists"""
        return {
            'observations': self.observations,
            'task_descs': self.task_descs,
            'actions': self.actions,
            'action_indices': self.action_indices,
            'action_candidates': self.action_candidates,
            'rewards': self.rewards,
            'dones': self.dones,
            'values': self.values,
            'log_probs': self.log_probs,
            'dynamics': self.dynamics
        }

    def __len__(self):
        return len(self.rewards)


class TextPPOAgent(BaseAgent):
    '''
    TextAgent trained with PPO (Proximal Policy Optimization)
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.training_method == "ppo"

        # PPO-specific parameters
        self.ppo_memory = PPOMemory()
        self.clip_epsilon = self.config['ppo']['clip_epsilon']
        self.ppo_epochs = self.config['ppo']['ppo_epochs']
        self.value_loss_coef = self.config['ppo']['value_loss_coef']
        self.entropy_coef = self.config['ppo']['entropy_coef']
        self.max_grad_norm = self.config['ppo']['max_grad_norm']
        self.gae_lambda = self.config['ppo']['gae_lambda']
        self.discount_gamma = self.config['ppo']['discount_gamma']
        self.normalize_advantages = self.config['ppo']['normalize_advantages']
        self.minibatch_size = self.config['ppo']['minibatch_size']

        # Value network (we'll add a value head to the policy network)
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(self.online_net.block_hidden_dim, self.online_net.block_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.online_net.block_hidden_dim, 1)
        )
        if self.use_cuda:
            self.value_head.cuda()

        # Update optimizer to include value head parameters
        self.optimizer = torch.optim.Adam(
            list(self.online_net.parameters()) + list(self.value_head.parameters()),
            lr=self.config['general']['training']['optimizer']['learning_rate']
        )

    def compute_value(self, observation_strings, task_desc_strings, previous_dynamics):
        """
        Compute state value using the value head.
        """
        with torch.no_grad():
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")

            # Aggregate observation and task description
            aggregated_obs = self.online_net.aggretate_information(h_obs, h_td, obs_mask, td_mask)
            state_representation = self.online_net.masked_mean(aggregated_obs, obs_mask)

            if self.recurrent and previous_dynamics is not None:
                state_representation = self.online_net.dynamics_aggregation(
                    torch.cat([state_representation, previous_dynamics], -1)
                )
                state_representation = torch.relu(state_representation)

            # Compute value
            value = self.value_head(state_representation)
            return value.squeeze(-1)

    def choose_action_with_log_prob(self, observation_strings, task_desc_strings,
                                    action_candidate_list, previous_dynamics):
        """
        Choose action and return action, index, log probability, value, and current dynamics.
        """
        batch_size = len(observation_strings)

        # Encode observations and task descriptions
        h_obs, obs_mask = self.encode(observation_strings, use_model="online")
        h_td, td_mask = self.encode(task_desc_strings, use_model="online")

        # Score actions
        action_scores, action_masks, current_dynamics = self.action_scoring(
            action_candidate_list, h_obs, obs_mask, h_td, td_mask,
            previous_dynamics, use_model="online"
        )

        # Compute action probabilities
        action_probs = masked_softmax_with_temperature(action_scores, action_masks, temperature=1.0)

        # Sample actions
        action_indices = []
        log_probs = []

        for i in range(batch_size):
            valid_actions = action_masks[i].nonzero(as_tuple=True)[0]
            if len(valid_actions) == 0:
                # No valid actions, choose randomly
                action_idx = 0
                log_prob = torch.tensor(0.0)
            else:
                probs = action_probs[i][valid_actions]
                probs = probs / probs.sum()  # Renormalize

                # Sample action
                action_idx = torch.multinomial(probs, 1).item()
                action_idx = valid_actions[action_idx].item()
                log_prob = torch.log(action_probs[i][action_idx] + 1e-10)

            action_indices.append(action_idx)
            log_probs.append(log_prob)

        action_indices = np.array(action_indices)
        chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, action_indices)]
        log_probs = torch.stack(log_probs)

        # Compute state value
        aggregated_obs = self.online_net.aggretate_information(h_obs, h_td, obs_mask, td_mask)
        state_representation = self.online_net.masked_mean(aggregated_obs, obs_mask)

        if self.recurrent and previous_dynamics is not None:
            state_representation = self.online_net.dynamics_aggregation(
                torch.cat([state_representation, previous_dynamics], -1)
            )
            state_representation = torch.relu(state_representation)

        value = self.value_head(state_representation).squeeze(-1)

        return chosen_actions, action_indices, log_probs, value, current_dynamics

    def admissible_commands_act(self, observation_strings, task_desc_strings,
                               action_candidate_list, previous_dynamics, random=False):
        """
        Act using PPO policy. During training, this samples from the policy.
        During evaluation, this acts greedily.
        """
        if self.mode == "eval":
            # Greedy action selection for evaluation
            return self.admissible_commands_act_greedy(observation_strings, task_desc_strings,
                                                      action_candidate_list, previous_dynamics)

        with torch.no_grad():
            chosen_actions, action_indices, log_probs, value, current_dynamics = \
                self.choose_action_with_log_prob(observation_strings, task_desc_strings,
                                                action_candidate_list, previous_dynamics)

            return chosen_actions, action_indices, current_dynamics

    def admissible_commands_act_greedy(self, observation_strings, task_desc_strings,
                                      action_candidate_list, previous_dynamics):
        """
        Greedy action selection (for evaluation).
        """
        with torch.no_grad():
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring(
                action_candidate_list, h_obs, obs_mask, h_td, td_mask,
                previous_dynamics, use_model="online"
            )

            # Choose max Q action
            action_rank = action_scores - torch.min(action_scores, -1, keepdim=True)[0] + 1e-2
            if action_masks is not None:
                action_rank = action_rank * action_masks
            action_indices = torch.argmax(action_rank, -1)
            action_indices = to_np(action_indices).astype(int)
            chosen_actions = [item[idx] for item, idx in zip(action_candidate_list, action_indices)]

            return chosen_actions, action_indices, current_dynamics

    def compute_gae(self, rewards, values, dones, last_value):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of rewards for each timestep
            values: List of value estimates for each timestep
            dones: List of done flags
            last_value: Value estimate for the last state

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0

        # Convert to tensors if needed
        if not isinstance(values, torch.Tensor):
            values = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards)
        if not isinstance(dones, torch.Tensor):
            dones = torch.tensor(dones)

        # Append last value for bootstrapping
        values_extended = torch.cat([values, last_value.unsqueeze(0)])

        # Compute GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values_extended[t + 1]

            delta = rewards[t] + self.discount_gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.discount_gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + values

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, advantages, returns, old_log_probs, observation_strings_list,
               task_desc_strings_list, action_candidate_list_list, action_indices_list,
               dynamics_list):
        """
        Update policy using PPO.

        Args:
            advantages: Computed advantages
            returns: Computed returns
            old_log_probs: Log probabilities from the old policy
            observation_strings_list: List of observation strings
            task_desc_strings_list: List of task description strings
            action_candidate_list_list: List of action candidate lists
            action_indices_list: List of chosen action indices
            dynamics_list: List of dynamics states
        """
        num_samples = len(observation_strings_list)

        # Convert to appropriate format
        if isinstance(advantages, list):
            advantages = torch.stack(advantages)
        if isinstance(returns, list):
            returns = torch.stack(returns)
        if isinstance(old_log_probs, list):
            old_log_probs = torch.stack(old_log_probs)

        if self.use_cuda:
            advantages = advantages.cuda()
            returns = returns.cuda()
            old_log_probs = old_log_probs.cuda()

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0

        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Create minibatches
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, self.minibatch_size):
                end_idx = min(start_idx + self.minibatch_size, num_samples)
                mb_indices = indices[start_idx:end_idx]

                # Get minibatch data
                mb_obs = [observation_strings_list[i] for i in mb_indices]
                mb_task = [task_desc_strings_list[i] for i in mb_indices]
                mb_action_candidates = [action_candidate_list_list[i] for i in mb_indices]
                mb_action_indices = [action_indices_list[i] for i in mb_indices]
                mb_dynamics = [dynamics_list[i] for i in mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]

                # Encode observations
                h_obs, obs_mask = self.encode(mb_obs, use_model="online")
                h_td, td_mask = self.encode(mb_task, use_model="online")

                # Score actions
                action_scores, action_masks, _ = self.action_scoring(
                    mb_action_candidates, h_obs, obs_mask, h_td, td_mask,
                    mb_dynamics[0] if len(mb_dynamics) > 0 else None, use_model="online"
                )

                # Compute action probabilities
                action_probs = masked_softmax_with_temperature(action_scores, action_masks, temperature=1.0)

                # Compute log probs and entropy for chosen actions
                new_log_probs = []
                entropies = []
                for i, action_idx in enumerate(mb_action_indices):
                    prob = action_probs[i][action_idx]
                    log_prob = torch.log(prob + 1e-10)
                    new_log_probs.append(log_prob)

                    # Entropy
                    valid_probs = action_probs[i][action_masks[i] > 0]
                    entropy = -(valid_probs * torch.log(valid_probs + 1e-10)).sum()
                    entropies.append(entropy)

                new_log_probs = torch.stack(new_log_probs)
                entropies = torch.stack(entropies)

                # Compute value estimates
                aggregated_obs = self.online_net.aggretate_information(h_obs, h_td, obs_mask, td_mask)
                state_representation = self.online_net.masked_mean(aggregated_obs, obs_mask)

                if self.recurrent and mb_dynamics[0] is not None:
                    # Stack dynamics for batch processing
                    dynamics_stacked = torch.stack(mb_dynamics) if isinstance(mb_dynamics[0], torch.Tensor) else mb_dynamics[0]
                    state_representation = self.online_net.dynamics_aggregation(
                        torch.cat([state_representation, dynamics_stacked], -1)
                    )
                    state_representation = torch.relu(state_representation)

                values = self.value_head(state_representation).squeeze(-1)

                # Compute PPO loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropies.mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.online_net.parameters()) + list(self.value_head.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropies.mean().item()
                update_count += 1

        avg_policy_loss = total_policy_loss / update_count if update_count > 0 else 0
        avg_value_loss = total_value_loss / update_count if update_count > 0 else 0
        avg_entropy = total_entropy / update_count if update_count > 0 else 0

        return avg_policy_loss, avg_value_loss, avg_entropy

    def finish_of_episode(self, episode_no, batch_size):
        """
        Called at the end of each episode to update the policy.
        """
        # Get all data from memory
        memory_data = self.ppo_memory.get_batches()

        if len(memory_data['rewards']) == 0:
            return None

        # Compute last values for bootstrapping
        last_obs = memory_data['observations'][-batch_size:]
        last_task = memory_data['task_descs'][-batch_size:]
        last_dynamics = memory_data['dynamics'][-batch_size:] if len(memory_data['dynamics']) > 0 else [None] * batch_size

        with torch.no_grad():
            last_value = self.compute_value(last_obs, last_task, last_dynamics[0] if last_dynamics[0] is not None else None)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            memory_data['rewards'],
            memory_data['values'],
            memory_data['dones'],
            last_value
        )

        # Update policy
        policy_loss, value_loss, entropy = self.update(
            advantages,
            returns,
            memory_data['log_probs'],
            memory_data['observations'],
            memory_data['task_descs'],
            memory_data['action_candidates'],
            memory_data['action_indices'],
            memory_data['dynamics']
        )

        # Clear memory
        self.ppo_memory.reset()

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }


def masked_softmax_with_temperature(logits, mask, temperature=1.0):
    """
    Compute masked softmax with temperature.
    """
    logits = logits / temperature
    # Mask invalid actions with very negative value
    masked_logits = logits.masked_fill(mask == 0, -1e10)
    probs = F.softmax(masked_logits, dim=-1)
    # Zero out invalid actions
    probs = probs * mask
    return probs
