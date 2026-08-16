"""
train_ppo.py — CleanRL-style IPPO script for HeterogeneousWarehouse

Trains a shared AgentNetwork across all robots using Proximal Policy Optimization.
"""

import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rware.warehouse import Warehouse, RewardType
from hetero_wrapper import HeterogeneousWarehouse
from network import AgentNetwork


@dataclass
class Args:
    exp_name: str = "ppo_hetero"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    
    # Environment
    n_agents: int = 4
    max_steps: int = 500
    
    # PPO Parameters
    total_timesteps: int = 200000
    learning_rate: float = 2.5e-4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


def make_env(args):
    base = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=1,
        n_agents=args.n_agents,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=max(args.n_agents, 2),
        max_inactivity_steps=None,
        max_steps=args.max_steps,
        reward_type=RewardType.INDIVIDUAL,
    )
    env = HeterogeneousWarehouse(base)
    return env


def main(args: Args):
    run_name = f"{args.exp_name}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = make_env(args)
    obs_dim = env.observation_space[0].shape[0]
    
    agent = AgentNetwork(obs_dim=obs_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # We treat n_agents as our batch dimension (vectorized env equivalent)
    obs = torch.zeros((args.num_steps, args.n_agents, obs_dim)).to(device)
    actions_move = torch.zeros((args.num_steps, args.n_agents)).to(device)
    actions_comm = torch.zeros((args.num_steps, args.n_agents)).to(device)
    logprobs = torch.zeros((args.num_steps, args.n_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.n_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.n_agents)).to(device)
    values = torch.zeros((args.num_steps, args.n_agents)).to(device)
    
    # Metrics
    global_step = 0
    start_time = time.time()
    
    # TRY NOT TO MODIFY: start the game
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = torch.Tensor(np.array(next_obs)).to(device)
    next_done = torch.zeros(args.n_agents).to(device)
    
    num_updates = args.total_timesteps // (args.num_steps * args.n_agents)
    batch_size = args.n_agents * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    
    episode_rewards = []
    
    print(f"Starting training on {device}...")
    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Tracking for this rollout
        step_bids = []

        for step in range(0, args.num_steps):
            global_step += args.n_agents
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                move_act, comm_act, bid, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                
            actions_move[step] = move_act
            actions_comm[step] = comm_act
            logprobs[step] = logprob
            
            step_bids.append(bid.cpu().numpy())

            # Convert to CPU for env
            move_np = move_act.cpu().numpy()
            comm_np = comm_act.cpu().numpy()
            
            # Action space requires [move, comm] per agent
            env_actions = np.stack([move_np, comm_np], axis=-1)
            
            next_obs_np, reward, done_env, truncated, info = env.step(env_actions)
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            # done is a single bool in RWARE, but cleanRL treats it as an array if vectorized.
            # We copy the single done flag to all agents.
            next_obs = torch.Tensor(np.array(next_obs_np)).to(device)
            next_done = torch.full((args.n_agents,), done_env, dtype=torch.float32).to(device)
            
            # If done, reset
            if done_env or truncated:
                episode_rewards.append(np.sum(reward))
                writer.add_scalar("charts/episodic_return", np.sum(reward), global_step)
                next_obs_np, _ = env.reset()
                next_obs = torch.Tensor(np.array(next_obs_np)).to(device)
                next_done = torch.zeros(args.n_agents).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions_move = actions_move.reshape(-1)
        b_actions_comm = actions_comm.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions_move[mb_inds],
                    b_actions_comm[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log training metrics
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # Log custom metrics
        avg_bid = np.mean(step_bids)
        comm_rate = actions_comm.mean().item()
        writer.add_scalar("metrics/avg_bid", avg_bid, global_step)
        writer.add_scalar("metrics/comm_rate", comm_rate, global_step)

        if update % 5 == 0:
            print(f"Update: {update}/{num_updates} | Global Step: {global_step} | SPS: {int(global_step / (time.time() - start_time))}")
            if episode_rewards:
                print(f"  -> Last episode reward sum: {episode_rewards[-1]:.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{run_name}.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    writer.close()


if __name__ == "__main__":
    main(Args())
