#!/usr/bin/env python3
"""DAgger iteration for the multi-lane simulation.

Procedure (Ross et al., 2010 + this project's round-1 result 53% -> 74%):
  1. Roll out the current policy (deterministic) in the multi-lane env.
  2. At every visited state, record the EXPERT's ideal hierarchical target
     (lane offset + speed) as the correction label.
  3. Fine-tune the policy on the corrective set (negative log-likelihood),
     tightening output variance so stochastic rollouts stay on-manifold.
  4. Save and broad-evaluate on 19 fixed seeds.

Usable as an iterative loop: feed the previous output back in as --base.

Usage:
    python autonomous_car/run_dagger.py --base models/bc_base_dagger1.pt \
        --out models/bc_dagger_next.pt --episodes 14 --steps 4000
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description="DAgger iteration for multi-lane driving")
    ap.add_argument("--base", default="autonomous_car/models/bc_base_dagger1.pt",
                    help="Source policy checkpoint (SAC/BC format)")
    ap.add_argument("--out", default="autonomous_car/models/bc_dagger_next.pt",
                    help="Output checkpoint")
    ap.add_argument("--episodes", type=int, default=14, help="Rollout episodes per round")
    ap.add_argument("--steps", type=int, default=4000, help="NLL fine-tune steps")
    ap.add_argument("--seed-offset", type=int, default=30, help="First rollout seed")
    ap.add_argument("--num-lanes", type=int, default=3)
    ap.add_argument("--obstacles", type=int, default=4)
    ap.add_argument("--traffic", type=int, default=2)
    ap.add_argument("--broad-eval", action="store_true",
                    help="Run 19-seed evaluation at the end")
    args = ap.parse_args()

    from autonomous_car.env.multilane_env import MultiLaneEnv
    from autonomous_car.controllers.multilane_expert import MultiLaneExpertController
    from autonomous_car.controllers.sac import SAC, SACConfig, evaluate_sac_metrics

    env = MultiLaneEnv(num_lanes=args.num_lanes, num_obstacles=args.obstacles,
                       num_traffic_vehicles=args.traffic, action_repeat=10,
                       render_mode=None, max_episode_steps=2000)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    asc = (env.action_space.high - env.action_space.low) / 2
    ab = (env.action_space.high + env.action_space.low) / 2

    cfg = SACConfig(hidden_dim=256, num_hidden_layers=2, learning_rate=3e-4,
                    batch_size=256, buffer_size=200000, warmup_steps=0)
    agent = SAC(obs_dim, act_dim, cfg, asc, ab, 'cpu')
    ck = torch.load(args.base, map_location='cpu', weights_only=False)
    agent.policy.load_state_dict(ck['policy_state_dict'])
    with torch.no_grad():
        agent.policy.log_std_head.bias.fill_(math.log(0.05))
        agent.policy.log_std_head.weight.mul_(0.05)
    print(f"[DAgger] base={args.base} obs={obs_dim} act={act_dim}")

    expert = MultiLaneExpertController(
        track=env.base_track, num_lanes=args.num_lanes, lane_width=env.lane_width,
        v_ref=10.0, action_noise=0.0)

    def rollout(seed):
        obs, info = env.reset(seed=seed)
        expert.reset()
        expert.set_obstacles(env.obstacles)
        expert.set_traffic(env.traffic_vehicles)
        pairs = []
        done = False
        while not done:
            state = info['state']
            a_learn = agent.select_action(obs, deterministic=True)
            expert.set_traffic(env.traffic_vehicles)
            expert.compute_action(state, dt=env.dt)
            target = np.array([expert.last_target_offset,
                               expert.last_target_speed], dtype=np.float32)
            pairs.append((obs, target))
            for _ in range(10):
                obs, r, term, trunc, info = env.step(a_learn)
                if term or trunc:
                    done = True
                    break
            done = done or term or trunc
        return pairs

    all_obs, all_tgt = [], []
    for s in range(args.episodes):
        seed = args.seed_offset + s
        pairs = rollout(seed)
        all_obs += [p[0] for p in pairs]
        all_tgt += [p[1] for p in pairs]
        print(f"  rollout seed {seed}: {len(pairs)} transitions "
              f"(cum {len(all_obs)})")

    obs_t = torch.FloatTensor(np.array(all_obs, dtype=np.float32))
    tgt_t = torch.FloatTensor(((np.array(all_tgt, dtype=np.float32) - ab) / asc))
    print(f"[DAgger] corrective set: {len(all_obs)} transitions")

    opt = torch.optim.Adam(agent.policy.parameters(), lr=1e-3)
    n = min(256, len(all_obs))
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(all_obs), (n,))
        opt.zero_grad()
        loss = agent._bc_loss(obs_t[idx], tgt_t[idx].clamp(-1 + 1e-6, 1 - 1e-6))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            agent.policy.log_std_head.bias.clamp_(min=-3.5, max=-1.0)
        if step % max(1, args.steps // 5) == 0:
            print(f"  [fit] step {step} loss {loss.item():.3f}")
    agent.save(args.out)
    print(f"[DAgger] saved {args.out}")

    if args.broad_eval:
        seeds = list(range(1, 17)) + [700, 701, 702]
        comps, colls, margs = [], [], []
        for seed in seeds:
            m = evaluate_sac_metrics(agent, env, num_episodes=1, seed=seed)
            comps.append(m['completion_rate'])
            colls.append(m['collision_rate'])
            margs.append(m['mean_min_hazard'])
            print(f"  seed {seed}: comp={m['completion_rate']:.2f} "
                  f"coll={m['collision_rate']:.2f} margin={m['mean_min_hazard']:.1f}")
        print(f"[DAgger] BROAD EVAL: comp={np.mean(comps):.2f} "
              f"coll={np.mean(colls):.2f} margin={np.mean(margs):.1f}")

    env.close()


if __name__ == '__main__':
    main()