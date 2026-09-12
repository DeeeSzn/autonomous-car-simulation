#!/usr/bin/env python3
"""
Run multi-lane environment with obstacles and traffic.

Usage:
    python run_multilane.py demo --episodes 5
    python run_multilane.py visualize --episodes 1 --traffic 5
    python run_multilane.py train_lwr --demos data/multilane_demos.npz --eval
"""

import argparse
import copy
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def run_demo(args):
    """Collect demonstrations with expert controller."""
    from autonomous_car.env.multilane_env import MultiLaneEnv
    from autonomous_car.controllers.multilane_expert import MultiLaneExpertController
    
    print("=" * 50)
    print("Collecting Multi-Lane Demonstrations")
    print("=" * 50)
    
    env = MultiLaneEnv(
        num_lanes=args.lanes,
        num_obstacles=args.obstacles,
        num_traffic_vehicles=args.traffic,
        action_repeat=args.action_repeat,
        obstacle_seed=args.seed,
        render_mode=None,
        speed_clamp=args.speed_clamp,
        clamp_margin=args.clamp_margin,
        max_episode_steps=getattr(args, "max_steps", 2000),
    )
    
    expert = MultiLaneExpertController(
        track=env.base_track,
        num_lanes=args.lanes,
            lane_width=env.lane_width,
            v_ref=10.0,
            action_noise=0.0
        )
    
    all_obs = []
    all_actions = []
    all_requested_actions = []
    all_lane_shield_reasons = []
    all_planner_phases = []
    all_states = []
    all_episode_ids = []
    episode_completed = []
    episode_statuses = []
    episode_end_steps = []
    episode_net_progress = []
    
    rng = np.random.default_rng(args.seed or 42)
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep if args.seed else None)
        expert.reset()
        expert.set_obstacles(env.obstacles)
        expert.set_traffic(env.traffic_vehicles)
        
        episode_obs = []
        episode_actions = []
        episode_states = []
        
        done = False
        total_reward = 0
        
        while not done:
            state = info["state"]
            
            # Compute one hierarchical expert decision at the same cadence used
            # by SAC. The target is then held for the full action-repeat window.
            expert.set_traffic(env.traffic_vehicles)
            action = expert.compute_action(state, dt=env.dt)
            # The environment action is hierarchical [lane_offset, target_speed].
            # `compute_action()` also computes low-level tracking controls for
            # diagnostics, but those controls must not be passed to env.step().
            expert_action = np.array([
                expert.last_target_offset,
                expert.last_target_speed,
            ], dtype=np.float32)
            env.set_planner_context(expert.phase, expert.target_lane)
            requested_action = expert_action.copy()

            # Sample one perturbation per decision, not once per held frame.
            perturbed_action = expert_action.copy()
            if rng.random() < args.perturb_rate:
                perturbed_action[0] += rng.uniform(
                    -args.perturb_magnitude, args.perturb_magnitude
                )
                perturbed_action[0] = np.clip(
                    perturbed_action[0], -env.lane_width, env.lane_width
                )
            
            decision_obs = obs.copy()
            decision_state = state.copy()

            for repeat_index in range(env.action_repeat):
                obs, reward, terminated, truncated, info = env.step(perturbed_action)
                total_reward += reward
                done = terminated or truncated
                if repeat_index == 0:
                    effective_action = np.array([
                        info.get("target_lane_offset", requested_action[0]),
                        info.get("target_speed", requested_action[1]),
                    ], dtype=np.float32)
                    episode_obs.append(decision_obs)
                    episode_actions.append(effective_action)
                    all_requested_actions.append(requested_action)
                    all_lane_shield_reasons.append(info.get("lane_shield_reason", ""))
                    all_planner_phases.append(expert.phase)
                    episode_states.append(decision_state)
                if done:
                    break
                if (
                    info.get("hazard_distance", float("inf")) < 35.0
                    or info.get("planner_phase") in {"PREPARE", "COMMIT", "BRAKE", "ESCAPE"}
                    or info.get("lane_shield_active", False)
                ):
                    break
        
        all_obs.extend(episode_obs)
        all_actions.extend(episode_actions)
        all_states.extend(episode_states)
        all_episode_ids.extend([ep] * len(episode_obs))
        episode_completed.append(info.get("episode_status") == "LAP_COMPLETED")
        episode_statuses.append(info.get("episode_status", "UNKNOWN"))
        episode_end_steps.append(int(info.get("step", len(episode_obs))))
        episode_net_progress.append(float(info.get("net_progress", 0.0)))
        
        status = info.get("episode_status", "UNKNOWN")
        print(f"Episode {ep + 1}/{args.episodes}: {status}, steps={len(episode_obs)}, reward={total_reward:.1f}")
    
    env.close()
    
    # Save demonstrations
    X = np.array(all_obs)
    y = np.array(all_actions)
    states = np.array(all_states)
    
    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        observations=X,
        actions=y,
        requested_actions=np.asarray(all_requested_actions, dtype=np.float32),
        lane_shield_reasons=np.asarray(all_lane_shield_reasons),
        planner_phases=np.asarray(all_planner_phases),
        states=states,
        episode_ids=np.asarray(all_episode_ids, dtype=np.int32),
        episode_completed=np.asarray(episode_completed, dtype=bool),
        episode_status=np.asarray(episode_statuses),
        episode_end_steps=np.asarray(episode_end_steps, dtype=np.int32),
        episode_net_progress=np.asarray(episode_net_progress, dtype=np.float32),
        collection_obstacles=args.obstacles,
        collection_traffic=args.traffic,
        collection_action_repeat=args.action_repeat,
    )
    
    print(f"\nSaved {len(X)} demonstrations to {save_path}")


def run_curriculum(args):
    """Collect and merge fresh demonstrations across driving difficulty levels."""
    levels = ((4, 4), (8, 8), (12, 12))
    output = Path(args.output)
    parts = []
    for index, (obstacles, traffic) in enumerate(levels):
        level_args = copy.copy(args)
        level_args.obstacles = obstacles
        level_args.traffic = traffic
        level_args.seed = args.seed + index * args.episodes
        level_args.output = str(output.with_name(
            f"{output.stem}_{obstacles}obs_{traffic}traffic.npz"
        ))
        run_demo(level_args)
        parts.append(np.load(level_args.output, allow_pickle=True))

    merged = {}
    for key in parts[0].files:
        values = [part[key] for part in parts]
        if key == "episode_ids":
            offset = 0
            adjusted = []
            for value in values:
                adjusted.append(value + offset)
                offset += int(value.max()) + 1 if value.size else 0
            merged[key] = np.concatenate(adjusted)
        elif values[0].ndim == 0:
            merged[key] = values[0]
        elif key.startswith("collection_"):
            continue
        else:
            merged[key] = np.concatenate(values)
    merged["curriculum_levels"] = np.asarray(levels, dtype=np.int32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **merged)
    print(f"Saved merged curriculum demonstrations to {output}")


def run_visualize(args):
    """Visualize expert or LWR controller."""
    from autonomous_car.env.multilane_env import MultiLaneEnv
    from autonomous_car.controllers.multilane_expert import MultiLaneExpertController
    
    print("=" * 50)
    print("Multi-Lane Visualization")
    print("=" * 50)
    
    env = MultiLaneEnv(
        num_lanes=args.lanes,
        num_obstacles=args.obstacles,
        num_traffic_vehicles=args.traffic,
        action_repeat=args.action_repeat,
        obstacle_seed=args.seed,
        render_mode="human",
        max_episode_steps=args.max_steps,
        speed_clamp=getattr(args, 'speed_clamp', False),
        clamp_margin=getattr(args, 'clamp_margin', 10.0),
    )
    
    if args.policy == "lwr" and args.model:
        from autonomous_car.controllers.lwr import LWRController
        controller = LWRController()
        controller.load(args.model)
        print(f"Loaded LWR model from {args.model}")
        use_expert = False
    elif args.policy == "hybrid" and args.model:
        import torch
        from autonomous_car.controllers.hybrid_bc import LayeredHybridController
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        controller = LayeredHybridController(
            args.model,
            env,
            device=device,
        )
        print(f"Loaded hybrid lane/speed policy from {args.model}")
        use_expert = False
    else:
        controller = MultiLaneExpertController(
            track=env.base_track,
            num_lanes=args.lanes,
            lane_width=env.lane_width,
            v_ref=10.0,
            verbose=args.verbose
        )
        use_expert = True
        print("Using expert controller")
    
    print(f"Environment: {args.obstacles} obstacles, {args.traffic} traffic vehicles")
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep if args.seed else None)
        
        if hasattr(controller, "reset"):
            controller.reset()
        if hasattr(controller, "set_obstacles"):
            controller.set_obstacles(env.obstacles)
        if hasattr(controller, "set_traffic"):
            controller.set_traffic(env.traffic_vehicles)
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < args.max_steps:
            if use_expert:
                controller.set_traffic(env.traffic_vehicles)
                controller.compute_action(
                    info["state"], dt=env.dt
                )
                env.set_planner_context(controller.phase, controller.target_lane)
                action = np.array([
                    controller.last_target_offset,
                    controller.last_target_speed,
                ], dtype=np.float32)
            else:
                if hasattr(controller, "set_traffic"):
                    controller.set_traffic(env.traffic_vehicles)
                action = controller.predict(obs)

            for _ in range(env.action_repeat):
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1
                if done or steps >= args.max_steps:
                    break
                # Keep long action holds only on a genuinely clear road.
                # Hazards and lane transitions require a fresh planner/safety
                # decision on the next simulator frame.
                hazard = info.get("hazard_distance", float("inf"))
                phase = info.get("planner_phase", "CRUISE")
                if (
                    hazard < 35.0
                    or phase in {"PREPARE", "COMMIT", "BRAKE", "ESCAPE"}
                    or info.get("lane_shield_active", False)
                ):
                    break

            if args.verbose and not use_expert and (steps % 20 == 0 or done):
                metrics = getattr(controller, "metrics", {})
                print(
                    f"[Step {steps}] lane={env.current_lane} "
                    f"proposed={metrics.get('proposed_lane')} "
                    f"target={metrics.get('target_lane')} "
                    f"phase={metrics.get('phase')} "
                    f"raw_speed={metrics.get('raw_speed', action[1]):.1f} "
                    f"safe_speed={metrics.get('safe_speed', action[1]):.1f} "
                    f"hazard_lane={info.get('hazard_lane')} "
                    f"hazard={info.get('hazard_distance', float('inf')):.1f} "
                    f"ttc={info.get('shield_ttc', float('inf')):.1f} "
                    f"shield={info.get('shield_active', False)}"
                )
        
        status = info.get("episode_status", "RUNNING")
        print(
            f"Episode {ep + 1}: {status}, steps={steps}, "
            f"reward={total_reward:.1f}, net_progress={info.get('net_progress', 0.0):.1f}, "
            f"completed_lap={info.get('completed_lap', False)}"
        )
    
    env.close()


def run_train_lwr(args):
    """Train LWR on multi-lane demonstrations."""
    from autonomous_car.controllers.lwr import LWRController
    from autonomous_car.env.multilane_env import MultiLaneEnv
    
    print("=" * 50)
    print("Training LWR on Multi-Lane Demos")
    print("=" * 50)
    
    # Load demos
    data = np.load(args.demos)
    X = data["observations"]
    y = data["actions"]
    print(f"Loaded {len(X)} demonstrations from {args.demos}")
    
    # Train LWR
    lwr = LWRController(k=100, tau=0.5, reg_lambda=1e-4)
    lwr.fit(X, y)
    
    # Save model
    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    lwr.save(str(save_path))
    print(f"Model saved to {save_path}")
    
    # Evaluate if requested
    if args.eval:
        print("\nEvaluating LWR policy...")
        env = MultiLaneEnv(
            num_lanes=args.lanes,
            num_obstacles=args.obstacles,
            num_traffic_vehicles=args.traffic,
            obstacle_seed=42,
            render_mode=None
        )
        
        successes = 0
        crashes = 0
        off_roads = 0
        traffic_hits = 0
        timeouts = 0
        
        for ep in range(10):
            # Use same seed range as training (42+ep) to test if LWR learned
            obs, info = env.reset(seed=42 + ep)
            done = False
            steps = 0
            
            while not done and steps < 2000:
                action = lwr.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            if terminated:
                if info.get("obstacle_hit"):
                    crashes += 1
                elif info.get("traffic_hit"):
                    traffic_hits += 1
                elif info.get("off_road"):
                    off_roads += 1
            elif truncated:
                timeouts += 1  # Ran full episode = success
                successes += 1
        
        print(f"Results over 10 episodes:")
        print(f"  Completed (timeout): {timeouts}")
        print(f"  Crashed (obstacle):  {crashes}")
        print(f"  Traffic collision:   {traffic_hits}")
        print(f"  Off-road:            {off_roads}")
        
        env.close()


def run_train_bc(args):
    """Train a pure behavior-cloning policy before RL fine-tuning."""
    import torch
    from autonomous_car.controllers.sac import SACConfig, train_bc
    from autonomous_car.env.multilane_env import MultiLaneEnv

    env = MultiLaneEnv(
        num_lanes=args.lanes,
        num_obstacles=args.obstacles,
        num_traffic_vehicles=args.traffic,
        action_repeat=args.action_repeat,
        render_mode=None,
    )
    config = SACConfig(
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_layers,
        learning_rate=args.lr,
        bc_learning_rate=args.bc_lr,
        batch_size=args.batch_size,
        buffer_size=10_000,
        warmup_steps=0,
    )
    if args.cpu:
        device = 'cpu'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    train_bc(env, config, args.demos, args.steps, args.output, device, args.batch_size)
    env.close()


def run_train_sac(args):
    """Train SAC (Soft Actor-Critic) on multi-lane environment.
    
    This is TRUE reinforcement learning that optimizes reward,
    unlike LWR which only imitates expert demonstrations.
    """
    from autonomous_car.env.multilane_env import MultiLaneEnv
    from autonomous_car.controllers.sac import SAC, SACConfig, train_sac
    
    print("=" * 60)
    print("SAC REINFORCEMENT LEARNING TRAINING")
    print("=" * 60)
    print("\nSAC optimizes: J(π) = E[Σ γᵗ(rₜ + α·H(π))]")
    print("This is real RL, not imitation learning!\n")
    
    # Create environment
    if args.speed_clamp:
        raise ValueError("--speed-clamp is inference-only; use it with visualize, not train_sac")

    env = MultiLaneEnv(
        num_lanes=args.lanes,
        num_obstacles=args.obstacles,
        num_traffic_vehicles=args.traffic,
        action_repeat=args.action_repeat,
        obstacle_seed=args.seed if args.seed else None,
        render_mode=None,
        use_decision_layer=args.decision_layer
    )
    
    print(f"Environment setup:")
    print(f"  - {args.lanes} lanes")
    print(f"  - {args.obstacles} static obstacles")
    print(f"  - {args.traffic} traffic vehicles")
    print(f"  - Observation dim: {env.observation_space.shape[0]}")
    print(f"  - Action dim: {env.action_space.shape[0]}")
    if args.decision_layer:
        print(f"  - Decision Layer: ENABLED (+2 obs dims)")
    
    # SAC configuration
    config = SACConfig(
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_layers,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        warmup_steps=args.warmup
    )
    if getattr(args, 'q_lr', None) is not None:
        config.q_lr = args.q_lr
    if getattr(args, 'q_clip_actor', None) is not None:
        config.q_clip_actor = args.q_clip_actor
    if getattr(args, 'safety_coef', None) is not None:
        config.safety_value_coef = args.safety_coef
    if getattr(args, 'alpha_start', None) is not None:
        config.alpha_start = args.alpha_start
        config.alpha_end = args.alpha_end
        config.alpha_anneal_steps = args.alpha_anneal_steps
    if getattr(args, 'lr_anneal_steps', None) is not None:
        config.lr_anneal_steps = args.lr_anneal_steps
    if getattr(args, 'lr_anneal_end_frac', None) is not None:
        config.lr_anneal_end_frac = args.lr_anneal_end_frac
    if args.bc_coef is not None:
        config.bc_coef = args.bc_coef
    if args.bc_coef_start is not None:
        config.bc_coef_start = args.bc_coef_start
    if args.bc_anneal_steps is not None:
        config.bc_anneal_steps = args.bc_anneal_steps
    config.bc_coef_floor = max(config.bc_coef_floor, args.bc_coef_floor)
    config.bc_update_interval = args.bc_update_interval
    config.bc_learning_rate = args.bc_lr
    config.q_scale_floor = args.q_scale_floor
    
    # Determine device (CUDA for NVIDIA, MPS for Apple Silicon, else CPU)
    import torch
    if not args.cpu and torch.cuda.is_available():
        device = 'cuda'
    elif not args.cpu and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"  - Device: {device}")
    
    # Train
    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    agent = train_sac(
        env=env,
        config=config,
        total_timesteps=args.timesteps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_path=str(save_path),
        device=device,
        domain_randomization=args.randomize,
        traffic_range=(args.traffic_min, args.traffic_max),
        obstacles_range=(args.obstacles_min, args.obstacles_max),
        use_decision_layer=args.decision_layer,
        demos_path=args.demos,
        demos_completed_only=args.demos_completed_only,
        resume_path=args.resume,
        early_stop_patience=args.early_stop_patience,
        resume_max_timesteps=args.resume_max_timesteps,
        eval_seed=args.eval_seed,
        eval_num_episodes=args.eval_episodes,
        eval_n_seeds=args.eval_seeds,
    )
    
    env.close()
    print("\nSAC training complete!")


def run_visualize_sac(args):
    """Visualize trained SAC policy."""
    from autonomous_car.env.multilane_env import MultiLaneEnv
    from autonomous_car.controllers.sac import SACController
    import numpy as np
    
    print("=" * 50)
    print("SAC Policy Visualization")
    print("=" * 50)
    
    # Check if decision layer is enabled
    use_decision_layer = getattr(args, 'decision_layer', False)
    decision_layer = None
    if use_decision_layer:
        from autonomous_car.controllers.decision_layer import DecisionLayer
        decision_layer = DecisionLayer(num_lanes=args.lanes, verbose=args.verbose)
        print("Decision Layer: ENABLED")
    
    env = MultiLaneEnv(
        num_lanes=args.lanes,
        num_obstacles=args.obstacles,
        num_traffic_vehicles=args.traffic,
        action_repeat=args.action_repeat,
        obstacle_seed=args.seed,
        render_mode="human",
        max_episode_steps=args.max_steps,
        use_decision_layer=use_decision_layer
    )
    
    # Enable kinematic speed clamp if requested
    if getattr(args, 'speed_clamp', False):
        env.speed_clamp = True
        env.clamp_margin = getattr(args, 'clamp_margin', 10.0)
        print(f"Kinematic speed clamp: ENABLED (margin={env.clamp_margin}m, a_max={env.a_max} m/s²)")
    else:
        print("Kinematic speed clamp: disabled")
    
    # Compute action scaling
    action_high = env.action_space.high
    action_low = env.action_space.low
    action_scale = (action_high - action_low) / 2
    action_bias = (action_high + action_low) / 2
    
    controller = SACController(
        model_path=args.model,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        action_scale=action_scale,
        action_bias=action_bias
    )
    
    print(f"Loaded SAC model from {args.model}")
    print(f"Environment: {args.obstacles} obstacles, {args.traffic} traffic vehicles")
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep if args.seed else None)
        
        # Initialize decision layer for new episode
        if decision_layer is not None:
            decision_layer.reset(start_lane=env.current_lane)
            dl_input = env.get_decision_layer_input()
            target_lane, desired_speed, _ = decision_layer.decide(**dl_input, dt=env.dt)
            env.set_decision_targets(target_lane, desired_speed)
            obs = env._get_obs()
        
        done = False
        total_reward = 0
        steps = 0
        prev_lane = env.current_lane
        lane_changes = 0
        
        # Print header for verbose mode
        if args.verbose:
            print(f"\n{'='*120}")
            print(f"EPISODE {ep + 1} - Decision Log (Hierarchical: targets -> tracking)")
            print(f"{'='*120}")
            header = f"{'Step':>5} | {'Lane':>4} | {'TgtLane':>7} | {'TgtSpd':>6} | {'ActV':>6} | {'Accel':>6} | {'Steer':>7} | {'Hazard':>7} | {'TTC':>6} | {'Emerg':>7} | {'Clamp':>7} | {'Intent'}"
            if use_decision_layer:
                header += f" | {'DL_Lane':>6} | {'DL_Spd':>5} | {'Reason'}"
            print(header)
            print(f"{'-'*130}")
        
        while not done and steps < args.max_steps:
            # Update decision layer BEFORE action selection
            if decision_layer is not None:
                dl_input = env.get_decision_layer_input()
                target_lane, desired_speed, decision_reason = decision_layer.decide(**dl_input, dt=env.dt)
                env.set_decision_targets(target_lane, desired_speed)
                # Re-get observation with updated decision targets
                obs = env._get_obs()
            
            action = controller.get_action(obs, deterministic=not args.stochastic)
            
            # Get decision info BEFORE step
            current_lane = env.current_lane
            per_lane = env._get_per_lane_info(env.prev_s, 60.0)
            obs_dists = per_lane['obstacle_dist']
            traffic_dists = per_lane['traffic_dist']
            min_dist = min(obs_dists[current_lane], traffic_dists[current_lane])
            
            # Find safest lane
            safest_lane = current_lane
            safest_dist = min_dist
            for lane in range(len(obs_dists)):
                if obs_dists[lane] > safest_dist:
                    safest_dist = obs_dists[lane]
                    safest_lane = lane
            
            emergency = min_dist < 35.0 and safest_lane != current_lane
            
            # Match SAC training: hold one hierarchical target for the full
            # action-repeat window, while counting simulator frames.
            for repeat_index in range(env.action_repeat):
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                steps += 1

                # Track lane changes inside the repeated-action window.
                if env.current_lane != prev_lane:
                    lane_changes += 1
                    if args.verbose:
                        print(f"  *** LANE CHANGE: {prev_lane} -> {env.current_lane} ***")
                prev_lane = env.current_lane
                if done or steps >= args.max_steps:
                    break
            
            # Verbose logging
            if args.verbose:
                tgt_lane_offset = action[0]
                tgt_speed = action[1]
                track_accel = info.get('tracking_accel', 0.0)
                track_steer = info.get('tracking_steer', 0.0)
                obs_str = "O[" + ",".join([f"{dist:.0f}" for dist in obs_dists]) + "]"
                
                # Probe clamp decision for the action just applied
                clamp_active = getattr(env, '_clamp_active', False)
                clamp_hazard = getattr(env, '_clamp_hazard_dist', float('inf'))
                clamp_str = f"@{clamp_hazard:.0f}m" if clamp_active else "  -  "
                ttc = info.get('shield_ttc', float('inf'))
                ttc_str = f"{ttc:.1f}" if np.isfinite(ttc) else "  - "
                actual_speed = info.get('actual_speed', 0.0)
                
                # Determine intention description
                half_road = env.total_road_width / 2 if hasattr(env, 'total_road_width') else 5.25
                target_lane_idx = int(round((tgt_lane_offset + half_road) / env.lane_width - 0.5))
                target_lane_idx = max(0, min(target_lane_idx, env.num_lanes - 1))
                intents = {
                    0: "RT", 1: "CTR", 2: "LT"
                }
                intent_desc = intents.get(target_lane_idx, f"L{target_lane_idx}")
                intent_desc += f"@{tgt_speed:.0f}m/s"
                
                # Only print key moments (every 20 steps, emergencies, or near obstacles)
                if (steps % 20 == 0 or emergency or min_dist < 20 or done or
                        clamp_active) and (repeat_index == env.action_repeat - 1 or done):
                    emerg_str = "YES!" if emergency else "no"
                    line = f"{steps:>5} | {current_lane:>4} | {tgt_lane_offset:>+7.2f} | {tgt_speed:>6.1f} | {actual_speed:>6.1f} | {track_accel:>+6.2f} | {track_steer:>+7.3f} | {min_dist:>7.1f} | {ttc_str:>6} | {emerg_str:>7} | {clamp_str:>7} | {intent_desc}"
                    if use_decision_layer:
                        line += f" | {target_lane:>6} | {desired_speed:>5.1f} | {decision_reason}"
                    print(line)
        
        status = "CRASHED" if terminated else "COMPLETED"
        if info.get("traffic_hit"):
            status = "TRAFFIC COLLISION"
        elif info.get("obstacle_hit"):
            status = "OBSTACLE COLLISION"
        print(f"\nEpisode {ep + 1}: {status}, steps={steps}, reward={total_reward:.1f}, lane_changes={lane_changes}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Multi-Lane Autonomous Car Simulation")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Demo collection
    demo_parser = subparsers.add_parser("demo", help="Collect demonstrations")
    demo_parser.add_argument("--episodes", type=int, default=20)
    demo_parser.add_argument("--lanes", type=int, default=3)
    demo_parser.add_argument("--obstacles", type=int, default=8)
    demo_parser.add_argument("--traffic", type=int, default=0, help="Number of traffic vehicles")
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument("--output", type=str, default="data/multilane_demos.npz")
    demo_parser.add_argument("--perturb-rate", type=float, default=0.2)
    demo_parser.add_argument("--perturb-magnitude", type=float, default=0.25)
    demo_parser.add_argument("--action-repeat", type=int, default=10)
    demo_parser.add_argument("--speed-clamp", action="store_true")
    demo_parser.add_argument("--clamp-margin", type=float, default=10.0)
    demo_parser.add_argument("--max-steps", type=int, default=2000)

    curriculum_parser = subparsers.add_parser(
        "collect_curriculum", help="Collect and merge 4/4, 8/8, and 12/12 demos"
    )
    curriculum_parser.add_argument("--episodes", type=int, default=50,
                                   help="Episodes per curriculum level")
    curriculum_parser.add_argument("--seed", type=int, default=100)
    curriculum_parser.add_argument("--output", type=str,
                                   default="data/multilane_curriculum_fresh.npz")
    curriculum_parser.add_argument("--lanes", type=int, default=3)
    curriculum_parser.add_argument("--perturb-rate", type=float, default=0.2)
    curriculum_parser.add_argument("--perturb-magnitude", type=float, default=0.25)
    curriculum_parser.add_argument("--action-repeat", type=int, default=10)
    curriculum_parser.add_argument("--speed-clamp", action="store_true")
    curriculum_parser.add_argument("--clamp-margin", type=float, default=5.0)
    curriculum_parser.add_argument("--max-steps", type=int, default=2000)
    
    # Visualization
    viz_parser = subparsers.add_parser("visualize", help="Visualize policy")
    viz_parser.add_argument("--policy", choices=["expert", "lwr", "hybrid", "sac"], default="expert")
    viz_parser.add_argument("--model", type=str, help="Path to model (LWR or SAC)")
    viz_parser.add_argument("--episodes", type=int, default=1)
    viz_parser.add_argument("--max-steps", type=int, default=800)  # ~1 lap at 10 m/s
    viz_parser.add_argument("--lanes", type=int, default=3)
    viz_parser.add_argument("--obstacles", type=int, default=8)
    viz_parser.add_argument("--traffic", type=int, default=0, help="Number of traffic vehicles")
    viz_parser.add_argument("--seed", type=int, default=42)
    viz_parser.add_argument("--verbose", action="store_true", help="Print lane changes and decisions")
    viz_parser.add_argument("--stochastic", action="store_true", help="Use stochastic SAC policy")
    viz_parser.add_argument("--decision-layer", action="store_true", 
                           help="Enable decision layer (must match training config)")
    viz_parser.add_argument("--speed-clamp", action="store_true",
                            help="Enable kinematic target_speed safety clamp (brake before obstacle)")
    viz_parser.add_argument("--clamp-margin", type=float, default=10.0,
                            help="Reaction margin (m) beyond kinematic stop distance for the clamp")
    viz_parser.add_argument("--action-repeat", type=int, default=10)
    
    # Train LWR
    train_parser = subparsers.add_parser("train_lwr", help="Train LWR controller")
    train_parser.add_argument("--demos", type=str, required=True)
    train_parser.add_argument("--output", type=str, default="models/multilane_lwr.npz")
    train_parser.add_argument("--lanes", type=int, default=3)
    train_parser.add_argument("--obstacles", type=int, default=8)
    train_parser.add_argument("--traffic", type=int, default=0, help="Number of traffic vehicles for eval")
    train_parser.add_argument("--eval", action="store_true")

    # Pure behavior cloning diagnostic
    bc_parser = subparsers.add_parser("train_bc", help="Train pure behavior cloning policy")
    bc_parser.add_argument("--demos", type=str, required=True)
    bc_parser.add_argument("--output", type=str, default="models/bc_multilane.pt")
    bc_parser.add_argument("--steps", type=int, default=5000)
    bc_parser.add_argument("--lanes", type=int, default=3)
    bc_parser.add_argument("--obstacles", type=int, default=4)
    bc_parser.add_argument("--traffic", type=int, default=2)
    bc_parser.add_argument("--hidden-dim", type=int, default=256)
    bc_parser.add_argument("--num-layers", type=int, default=2)
    bc_parser.add_argument("--lr", type=float, default=3e-4)
    bc_parser.add_argument("--bc-lr", type=float, default=5e-5)
    bc_parser.add_argument("--batch-size", type=int, default=256)
    bc_parser.add_argument("--cpu", action="store_true")
    bc_parser.add_argument("--action-repeat", type=int, default=10)
    
    # Train SAC (reinforcement learning)
    sac_parser = subparsers.add_parser("train_sac", help="Train SAC (reinforcement learning)")
    sac_parser.add_argument("--timesteps", type=int, default=500_000, help="Total training steps")
    sac_parser.add_argument("--lanes", type=int, default=3)
    sac_parser.add_argument("--obstacles", type=int, default=8)
    sac_parser.add_argument("--traffic", type=int, default=0, help="Number of traffic vehicles")
    sac_parser.add_argument("--seed", type=int, default=None)
    sac_parser.add_argument("--output", type=str, default="models/sac_multilane.pt")
    sac_parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer size")
    sac_parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers")
    sac_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    sac_parser.add_argument("--q-lr", type=float, default=None,
                            help="Separate critic LR (value-head stability; default = lr)")
    sac_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    sac_parser.add_argument("--batch-size", type=int, default=256)
    sac_parser.add_argument("--buffer-size", type=int, default=1_000_000)
    sac_parser.add_argument("--warmup", type=int, default=10_000, help="Random actions before training")
    sac_parser.add_argument("--eval-interval", type=int, default=10_000)
    sac_parser.add_argument("--save-interval", type=int, default=50_000)
    sac_parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available")
    
    # Domain randomization arguments
    sac_parser.add_argument("--randomize", action="store_true", help="Enable domain randomization")
    sac_parser.add_argument("--traffic-min", type=int, default=4, help="Min traffic vehicles (randomization)")
    sac_parser.add_argument("--traffic-max", type=int, default=12, help="Max traffic vehicles (randomization)")
    sac_parser.add_argument("--obstacles-min", type=int, default=4, help="Min obstacles (randomization)")
    sac_parser.add_argument("--obstacles-max", type=int, default=8, help="Max obstacles (randomization)")
    
    # BC-SAC / curriculum / safety args
    sac_parser.add_argument("--demos", type=str, default=None, help="Path to BC demo npz (keys: observations, actions, episode_ids, episode_completed)")
    sac_parser.add_argument("--demos-completed-only", action="store_true", help="Only use demos from completed episodes")
    sac_parser.add_argument("--q-clip-actor", type=float, default=None,
                            help="Clamp |normalized Q| in actor objective (bound critic influence)")
    sac_parser.add_argument("--safety-coef", type=float, default=None,
                            help="Weight of hazard-margin safety anchor in actor objective")
    sac_parser.add_argument("--alpha-start", type=float, default=None,
                            help="Start alpha for deterministic anneal (stability)")
    sac_parser.add_argument("--alpha-end", type=float, default=0.05,
                            help="Alpha anneal target (stability)")
    sac_parser.add_argument("--alpha-anneal-steps", type=int, default=0,
                            help="Env steps over which alpha anneals; 0 = fixed")
    sac_parser.add_argument("--lr-anneal-end-frac", type=float, default=0.3,
                            help="Final effective LR as a fraction (decay target)")
    sac_parser.add_argument("--lr-anneal-steps", type=int, default=None,
                            help="Env steps for LR decay (defaults to alpha anneal window)")
    sac_parser.add_argument("--bc-coef", type=float, default=None, help="BC lambda target (default: config value)")
    sac_parser.add_argument("--bc-coef-start", type=float, default=None, help="BC lambda at env_steps=0 (curriculum start)")
    sac_parser.add_argument("--bc-anneal-steps", type=int, default=None, help="Env steps over which lambda anneals; 0 = fixed")
    sac_parser.add_argument("--bc-coef-floor", type=float, default=0.5, help="Minimum BC lambda floor")
    sac_parser.add_argument("--bc-update-interval", type=int, default=8,
                            help="Separate BC update every N RL updates")
    sac_parser.add_argument("--bc-lr", type=float, default=5e-5,
                            help="Learning rate for separate BC optimizer")
    sac_parser.add_argument("--q-scale-floor", type=float, default=1.0,
                            help="Minimum Q scale used by actor normalization")
    sac_parser.add_argument("--resume", type=str, default=None, help="Resume training from a checkpoint")
    sac_parser.add_argument("--resume-max-timesteps", type=int, default=150_000,
                            help="Maximum continuation steps when resuming")
    sac_parser.add_argument("--early-stop-patience", type=int, default=3,
                            help="Stop after this many evals without improvement")
    sac_parser.add_argument("--eval-seed", type=int, default=42,
                            help="First fixed seed for checkpoint evaluation")
    sac_parser.add_argument("--eval-episodes", type=int, default=4,
                            help="Episodes per eval seed (cheap evals = faster sweeps)")
    sac_parser.add_argument("--eval-seeds", type=int, default=3,
                            help="Number of fixed seeds per eval")
    sac_parser.add_argument("--action-repeat", type=int, default=10)
    sac_parser.add_argument("--speed-clamp", action="store_true",
                            help="Deprecated for training; use inference-only visualize flag")
    sac_parser.add_argument("--clamp-margin", type=float, default=10.0,
                            help="Unused by training; visualize-only clamp margin")
    
    # Decision layer arguments
    sac_parser.add_argument("--decision-layer", action="store_true", 
                           help="Enable decision layer for strategic decisions (lane selection, speed targets)")
    
    args = parser.parse_args()
    
    if args.command == "demo":
        run_demo(args)
    elif args.command == "collect_curriculum":
        run_curriculum(args)
    elif args.command == "visualize":
        if args.policy == "sac" and args.model:
            run_visualize_sac(args)
        else:
            run_visualize(args)
    elif args.command == "train_lwr":
        run_train_lwr(args)
    elif args.command == "train_bc":
        run_train_bc(args)
    elif args.command == "train_sac":
        run_train_sac(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
