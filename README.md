# Multi-Lane Autonomous Driving — Imitation (BC/DAgger) + Simulation

A lightweight, numpy/PyTorch **multi-lane driving simulation** with a trained
**DAgger policy** (`autonomous_car/models/bc_base_dagger1.pt`) you can run in the browser-free
matplotlib renderer and watch **escape traffic / avoid obstacles**.

Measured on 19 broad seeds (4 obstacles + 2 traffic):
- **DAgger policy (this repo)**: 74% lap completion, 26% collision
- Expert oracle: 100% completion, 0% collision
- Pure behavior cloning base: 53% completion

## Install

```bash
pip install -r requirements.txt        # numpy, scipy, scikit-learn, gymnasium, matplotlib, torch
```

## Run the simulation (matplotlib)

Watch the trained DAgger policy drive and avoid obstacles/traffic, with a full
step-by-step decision log:

```bash
python autonomous_car/run_multilane.py visualize \
    --policy sac \
    --model autonomous_car/models/bc_base_dagger1.pt \
    --obstacles 4 --traffic 2 \
    --episodes 2 \
    --verbose
```

Verbose columns: lane, target lane offset, target/actual speed, acceleration,
steer, nearest hazard distance, time-to-collision, emergency, shield/clamp,
and intent. Run 5 episodes and you'll see it complete laps **and** occasionally
fail — that's the honest 74%/26% make-up the paper-readme should echo.

You can also test how it handles heavier traffic:

```bash
python autonomous_car/run_multilane.py visualize \
    --policy sac --model autonomous_car/models/bc_base_dagger1.pt --obstacles 6 --traffic 4 --episodes 3 --verbose
```

## Re-train your own policy with DAgger

1. Collect expert demonstrations (the expert uses Pure Pursuit + PID + a lane
   shield, and 20% steering perturbation is injected for robustness):

   ```bash
   python autonomous_car/run_multilane.py demo --episodes 30 --obstacles 8
   ```

2. Behavior clone from the completed demos:

   ```bash
   python autonomous_car/run_multilane.py train_bc --demos data/multilane_demos.npz --steps 12000 --output autonomous_car/models/bc_base.pt
   ```

3. Run DAgger iterations — each round rolls out the current policy, records the
   expert's ideal lane/speed target at every state, fine-tunes, and re-evaluates:

   ```bash
   python autonomous_car/run_dagger.py --base autonomous_car/models/bc_base_dagger1.pt \
       --out autonomous_car/models/bc_round2.pt --episodes 14 --steps 4000 --broad-eval
   ```

4. Watch the new policy:

   ```bash
   python autonomous_car/run_multilane.py visualize --policy sac --model autonomous_car/models/bc_round2.pt --obstacles 4 --traffic 2 --verbose
   ```

## Why imitation, not RL

In this simulator, reinforcement-learning (SAC) fine-tuning repeatedly ended
below its own BC warm-start (the critic loses its hazard-gradient under
off-policy drift). Expert-corrected self-rollouts (DAgger) reliably beat both
plain behavior cloning and RL: 53% -> 74% completion in one round. The
literature (NVIDIA end-to-end, Waymo ChauffeurNet, Learning-by-Cheating, and
CARLA end-to-end surveys) is consistent: imitation is the deployable backbone;
RL is a gated fine-tuning layer on top.

## Repository layout

```
autonomous_car/
  env/multilane_env.py    Multi-lane driving environment (gymnasium)
  env/track.py            Track geometry (oval/circular)
  controllers/sac.py      Policy network, SACController, BC training
  controllers/multilane_expert.py  Expert oracle (Pure Pursuit + PID + shield)
  run_multilane.py        CLI: visualize / demo / train_bc
  run_dagger.py           DAgger round script (retrain with expert corrections)
  autonomous_car/models/bc_base_dagger1.pt  The trained DAgger policy (1.8 MB)
```

## License

MIT — see [LICENSE](LICENSE).