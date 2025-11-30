import datetime
import os
import copy
import json
import pathlib
import numpy as np
import sys
import argparse

import alfworld.agents.environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.agent.text_ppo_agent import TextPPOAgent
from alfworld.agents.eval import evaluate_dqn
from alfworld.agents.modules.generic import HistoryScoreCache, EpisodicCountingMemory, ObjCentricEpisodicMemory
from alfworld.agents.utils.misc import extract_admissible_commands

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ppo_config.yaml",
                       help="Path to config file")
    parser.add_argument("--cuda_device", type=str, default="0",
                       help="CUDA device ID")
    parser.add_argument("--output_dir", type=str, default="./ppo_output",
                       help="Directory to save models and logs")
    args = parser.parse_args()

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    # Load config
    config_path = os.path.join(pathlib.Path(__file__).parent.absolute(), args.config)
    sys.argv = [sys.argv[0], config_path]

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    agent = TextPPOAgent(config)

    env_type = config["env"]["type"]
    id_eval_env, num_id_eval_game = None, 0
    ood_eval_env, num_ood_eval_game = None, 0

    # Initialize training environment
    alfred_env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval="train")
    env = alfred_env.init_env(batch_size=agent.batch_size)

    # Initialize evaluation environments
    if agent.run_eval:
        # In distribution
        if config['dataset']['eval_id_data_path'] is not None:
            alfred_env = getattr(alfworld.agents.environment, config["general"]["evaluate"]["env"]["type"])(
                config, train_eval="eval_in_distribution"
            )
            id_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
            num_id_eval_game = alfred_env.num_games
        # Out of distribution
        if config['dataset']['eval_ood_data_path'] is not None:
            alfred_env = getattr(alfworld.agents.environment, config["general"]["evaluate"]["env"]["type"])(
                config, train_eval="eval_out_of_distribution"
            )
            ood_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
            num_ood_eval_game = alfred_env.num_games

    output_dir = args.output_dir
    data_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Visdom setup
    if config["general"]["visdom"]:
        import visdom
        viz = visdom.Visdom()
        reward_win, step_win = None, None
        loss_win = None
        viz_game_points, viz_step, viz_overall_rewards = [], [], []
        viz_id_eval_game_points, viz_id_eval_step = [], []
        viz_ood_eval_game_points, viz_ood_eval_step = [], []
        viz_policy_loss, viz_value_loss, viz_entropy = [], [], []

    step_in_total = 0
    episode_no = 0
    running_avg_game_points = HistoryScoreCache(capacity=500)
    running_avg_overall_rewards = HistoryScoreCache(capacity=500)
    running_avg_game_steps = HistoryScoreCache(capacity=500)
    running_avg_policy_loss = HistoryScoreCache(capacity=500)
    running_avg_value_loss = HistoryScoreCache(capacity=500)
    running_avg_entropy = HistoryScoreCache(capacity=500)

    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_performance_so_far, best_ood_performance_so_far = 0.0, 0.0
    episodic_counting_memory = EpisodicCountingMemory()
    obj_centric_episodic_counting_memory = ObjCentricEpisodicMemory()

    # Training loop
    while True:
        if episode_no > agent.max_episode:
            break

        np.random.seed(episode_no)
        env.seed(episode_no)
        obs, infos = env.reset()
        batch_size = len(obs)

        agent.train()
        agent.init(batch_size)
        episodic_counting_memory.reset()
        obj_centric_episodic_counting_memory.reset()
        previous_dynamics = None

        chosen_actions = []
        prev_step_dones, prev_rewards = [], []
        for _ in range(batch_size):
            chosen_actions.append("restart")
            prev_step_dones.append(0.0)
            prev_rewards.append(0.0)

        observation_strings = list(obs)
        task_desc_strings, observation_strings = agent.get_task_and_obs(observation_strings)
        task_desc_strings = agent.preprocess_task(task_desc_strings)
        observation_strings = agent.preprocess_observation(observation_strings)
        first_sight_strings = copy.deepcopy(observation_strings)
        agent.observation_pool.push_first_sight(first_sight_strings)
        action_candidate_list = list(infos["admissible_commands"])
        action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
        observation_only = observation_strings
        observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, chosen_actions)]

        # Episode metrics
        still_running_mask = []
        game_rewards, game_points, game_step = [], [], []
        print_actions = []
        report = agent.report_frequency > 0 and (episode_no % agent.report_frequency <= (episode_no - batch_size) % agent.report_frequency)

        for _ in range(batch_size):
            still_running_mask.append(1.0)
            game_rewards.append(0.0)
            game_points.append(0.0)
            game_step.append(0.0)
            print_actions.append([])

        # Episode rollout
        for step_no in range(agent.max_nb_steps_per_episode):
            # Get task and observation strings
            curr_obs_strings = agent.observation_pool.get()
            curr_task_desc_strings = copy.deepcopy(task_desc_strings)

            # Choose action
            chosen_actions, chosen_indices, current_dynamics = agent.admissible_commands_act(
                curr_obs_strings, curr_task_desc_strings,
                action_candidate_list, previous_dynamics
            )

            # Also get log probs and values for PPO
            with torch.no_grad():
                import torch
                _, _, log_probs, values, _ = agent.choose_action_with_log_prob(
                    curr_obs_strings, curr_task_desc_strings,
                    action_candidate_list, previous_dynamics
                )

            # Execute actions
            obs, rewards, dones, infos = env.step(chosen_actions)
            rewards = np.array(rewards, dtype=np.float32)

            # Store transitions in PPO memory
            for b in range(batch_size):
                if still_running_mask[b] == 1.0:
                    agent.ppo_memory.push(
                        observation=curr_obs_strings[b],
                        task_desc=curr_task_desc_strings[b],
                        action=chosen_actions[b],
                        action_idx=chosen_indices[b],
                        action_candidate_list=action_candidate_list[b],
                        reward=rewards[b],
                        done=float(dones[b]),
                        value=values[b] if isinstance(values, torch.Tensor) else values,
                        log_prob=log_probs[b] if isinstance(log_probs, torch.Tensor) else log_probs,
                        dynamic=current_dynamics
                    )

            # Process observations
            observation_strings = list(obs)
            observation_strings = agent.preprocess_observation(observation_strings)
            agent.observation_pool.push_batch(observation_only)
            observation_only = observation_strings
            observation_strings = [item + " [SEP] " + a for item, a in zip(observation_strings, chosen_actions)]

            # Update action candidates
            action_candidate_list = list(infos["admissible_commands"])
            action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)

            # Update previous dynamics
            previous_dynamics = current_dynamics

            # Update metrics
            for b in range(batch_size):
                if still_running_mask[b] == 1.0:
                    game_rewards[b] += rewards[b]
                    game_points[b] += rewards[b]
                    game_step[b] += 1
                    print_actions[b].append(chosen_actions[b])

                    if dones[b]:
                        still_running_mask[b] = 0.0

            # Check if all games are done
            if np.sum(still_running_mask) == 0:
                break

        # End of episode - update policy
        update_info = agent.finish_of_episode(episode_no, batch_size)

        if update_info is not None:
            running_avg_policy_loss.push(update_info['policy_loss'])
            running_avg_value_loss.push(update_info['value_loss'])
            running_avg_entropy.push(update_info['entropy'])

        # Update statistics
        step_in_total += np.sum(game_step)
        episode_no += batch_size

        for b in range(batch_size):
            running_avg_game_points.push(game_points[b])
            running_avg_overall_rewards.push(game_rewards[b])
            running_avg_game_steps.push(game_step[b])

        # Reporting
        if report:
            time_2 = datetime.datetime.now()
            print("Episode: {:3d} | time spent: {:s} | step: {:6d} | loss: {:.4f}/{:.4f} | success: {:.2f}% | reward: {:.4f} | steps: {:.2f}".format(
                episode_no,
                str(time_2 - time_1).rsplit(".")[0],
                step_in_total,
                running_avg_policy_loss.get_avg() if len(running_avg_policy_loss) > 0 else 0.0,
                running_avg_value_loss.get_avg() if len(running_avg_value_loss) > 0 else 0.0,
                running_avg_game_points.get_avg() * 100,  # 成功率（0-1转换为百分比）
                running_avg_overall_rewards.get_avg(),
                running_avg_game_steps.get_avg()
            ))

            # Visdom plotting
            if config["general"]["visdom"]:
                viz_game_points.append(running_avg_game_points.get_avg())
                viz_step.append(running_avg_game_steps.get_avg())
                viz_overall_rewards.append(running_avg_overall_rewards.get_avg())
                viz_policy_loss.append(running_avg_policy_loss.get_avg() if len(running_avg_policy_loss) > 0 else 0.0)
                viz_value_loss.append(running_avg_value_loss.get_avg() if len(running_avg_value_loss) > 0 else 0.0)
                viz_entropy.append(running_avg_entropy.get_avg() if len(running_avg_entropy) > 0 else 0.0)
                viz_id_eval_game_points.append(id_eval_game_points)
                viz_id_eval_step.append(id_eval_game_step)
                viz_ood_eval_game_points.append(ood_eval_game_points)
                viz_ood_eval_step.append(ood_eval_game_step)
                viz_x = list(range(len(viz_game_points)))

                # Success rate plot (training + eval)
                if reward_win is None:
                    reward_win = viz.line(
                        X=viz_x,
                        Y=viz_game_points,
                        opts=dict(title=agent.experiment_tag + "_success_rate", ylabel='Success Rate'),
                        win=reward_win,
                        name="train"
                    )
                    viz.line(X=viz_x, Y=viz_id_eval_game_points,
                            opts=dict(title=agent.experiment_tag + "_success_rate"),
                            win=reward_win, update='append', name="id_eval")
                    viz.line(X=viz_x, Y=viz_ood_eval_game_points,
                            opts=dict(title=agent.experiment_tag + "_success_rate"),
                            win=reward_win, update='append', name="ood_eval")
                else:
                    viz.line(X=[len(viz_game_points) - 1], Y=[viz_game_points[-1]],
                            win=reward_win, update='append', name="train")
                    viz.line(X=[len(viz_id_eval_game_points) - 1], Y=[viz_id_eval_game_points[-1]],
                            win=reward_win, update='append', name="id_eval")
                    viz.line(X=[len(viz_ood_eval_game_points) - 1], Y=[viz_ood_eval_game_points[-1]],
                            win=reward_win, update='append', name="ood_eval")

                # Steps plot
                if step_win is None:
                    step_win = viz.line(
                        X=viz_x,
                        Y=viz_step,
                        opts=dict(title=agent.experiment_tag + "_steps", ylabel='Steps'),
                        win=step_win,
                        name="train"
                    )
                    viz.line(X=viz_x, Y=viz_id_eval_step,
                            win=step_win, update='append', name="id_eval")
                    viz.line(X=viz_x, Y=viz_ood_eval_step,
                            win=step_win, update='append', name="ood_eval")
                else:
                    viz.line(X=[len(viz_step) - 1], Y=[viz_step[-1]],
                            win=step_win, update='append', name="train")
                    viz.line(X=[len(viz_id_eval_step) - 1], Y=[viz_id_eval_step[-1]],
                            win=step_win, update='append', name="id_eval")
                    viz.line(X=[len(viz_ood_eval_step) - 1], Y=[viz_ood_eval_step[-1]],
                            win=step_win, update='append', name="ood_eval")

                # Loss plot
                if loss_win is None:
                    loss_win = viz.line(
                        X=viz_x,
                        Y=viz_policy_loss,
                        opts=dict(title=agent.experiment_tag + "_losses", ylabel='Loss'),
                        win=loss_win,
                        name="policy_loss"
                    )
                    viz.line(X=viz_x, Y=viz_value_loss,
                            win=loss_win, update='append', name="value_loss")
                    viz.line(X=viz_x, Y=viz_entropy,
                            win=loss_win, update='append', name="entropy")
                else:
                    viz.line(X=[len(viz_policy_loss) - 1], Y=[viz_policy_loss[-1]],
                            win=loss_win, update='append', name="policy_loss")
                    viz.line(X=[len(viz_value_loss) - 1], Y=[viz_value_loss[-1]],
                            win=loss_win, update='append', name="value_loss")
                    viz.line(X=[len(viz_entropy) - 1], Y=[viz_entropy[-1]],
                            win=loss_win, update='append', name="entropy")

        # Evaluation
        id_eval_game_points, id_eval_game_step, id_eval_game_gc = 0.0, 0.0, 0.0
        ood_eval_game_points, ood_eval_game_step, ood_eval_game_gc = 0.0, 0.0, 0.0

        if agent.run_eval:
            if episode_no % agent.report_frequency == 0:
                # Save current model
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_episode_" + str(episode_no) + ".pt")

                # Evaluate on in-distribution data
                if id_eval_env is not None:
                    id_eval_res = evaluate_dqn(id_eval_env, agent, num_id_eval_game)
                    id_eval_game_points = id_eval_res['average_points']
                    id_eval_game_step = id_eval_res['average_steps']
                    id_eval_game_gc = id_eval_res['average_goal_condition_points']
                    print("ID Eval: {:s} | success: {:2.1f}% | game points: {:2.3f} | GC: {:2.3f} | steps: {:2.3f}".format(
                        json_file_name,
                        id_eval_game_points * 100,  # 转换为百分比
                        id_eval_game_points,
                        id_eval_game_gc,
                        id_eval_game_step
                    ))

                    if id_eval_game_points > best_performance_so_far:
                        best_performance_so_far = id_eval_game_points
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_best_id.pt")
                        print("  >>> New best ID performance: {:.2f}%".format(best_performance_so_far * 100))

                # Evaluate on out-of-distribution data
                if ood_eval_env is not None:
                    ood_eval_res = evaluate_dqn(ood_eval_env, agent, num_ood_eval_game)
                    ood_eval_game_points = ood_eval_res['average_points']
                    ood_eval_game_step = ood_eval_res['average_steps']
                    ood_eval_game_gc = ood_eval_res['average_goal_condition_points']
                    print("OOD Eval: {:s} | success: {:2.1f}% | game points: {:2.3f} | GC: {:2.3f} | steps: {:2.3f}".format(
                        json_file_name,
                        ood_eval_game_points * 100,  # 转换为百分比
                        ood_eval_game_points,
                        ood_eval_game_gc,
                        ood_eval_game_step
                    ))

                    if ood_eval_game_points > best_ood_performance_so_far:
                        best_ood_performance_so_far = ood_eval_game_points
                        agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_best_ood.pt")
                        print("  >>> New best OOD performance: {:.2f}%".format(best_ood_performance_so_far * 100))

                # Save training statistics to JSON file
                _s = json.dumps({
                    "episode": episode_no,
                    "time_spent": str(time_2 - time_1).rsplit(".")[0],
                    "steps": step_in_total,
                    "policy_loss": running_avg_policy_loss.get_avg() if len(running_avg_policy_loss) > 0 else 0.0,
                    "value_loss": running_avg_value_loss.get_avg() if len(running_avg_value_loss) > 0 else 0.0,
                    "entropy": running_avg_entropy.get_avg() if len(running_avg_entropy) > 0 else 0.0,
                    "train_success_rate": running_avg_game_points.get_avg(),
                    "train_reward": running_avg_overall_rewards.get_avg(),
                    "train_steps": running_avg_game_steps.get_avg(),
                    "id_eval_success_rate": id_eval_game_points,
                    "id_eval_gc": id_eval_game_gc,
                    "id_eval_steps": id_eval_game_step,
                    "ood_eval_success_rate": ood_eval_game_points,
                    "ood_eval_gc": ood_eval_game_gc,
                    "ood_eval_steps": ood_eval_game_step,
                    "best_id_performance": best_performance_so_far,
                    "best_ood_performance": best_ood_performance_so_far
                })
                with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
                    outfile.write(_s + '\n')
                    outfile.flush()

    # Save final model
    agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + "_final.pt")
    print("Training completed!")


if __name__ == '__main__':
    train()
