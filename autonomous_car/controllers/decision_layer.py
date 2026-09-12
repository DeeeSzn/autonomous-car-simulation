"""
Decision Layer for Hierarchical Autonomous Driving.

This module implements a rule-based strategic decision layer that separates
WHAT to do (lane selection, speed targets) from HOW to do it (SAC execution).

Architecture:
    Observation → Decision Layer → [target_lane, desired_speed]
                       ↓
    Augmented Obs → SAC → [acceleration, steering]

The decision layer handles:
- Lane selection based on obstacle/traffic analysis
- Speed targets based on threat proximity
- Tactical decisions (when to change lanes)

SAC handles:
- Smooth execution of lane changes
- Fine-grained steering control
- Acceleration/braking control

This separation makes each component easier to train and debug.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class DecisionState:
    """Internal state for decision layer."""
    target_lane: int
    desired_speed: float
    decision_confidence: float  # 0-1, how confident in current decision
    time_in_state: float  # How long in current decision
    last_decision: str  # For debugging


class DecisionLayer:
    """
    Strategic decision-making for lane selection and speed control.
    
    This is a rule-based controller that makes high-level decisions,
    leaving low-level execution to the SAC policy.
    
    Decision Logic:
    1. Scan all lanes for threats (obstacles + traffic)
    2. Evaluate safety of each lane
    3. Choose safest lane that requires minimal change
    4. Set speed based on closest threat
    
    Key Design Principles:
    - Simple, interpretable rules
    - Works for any traffic density (not learned)
    - Provides consistent targets for SAC to follow
    """
    
    def __init__(
        self,
        num_lanes: int = 2,
        cruise_speed: float = 10.0,
        cautious_speed: float = 7.0,
        slow_speed: float = 4.0,
        emergency_speed: float = 5.0,  # V12 fix: min 5m/s to ensure lane changes complete
        # Thresholds
        safe_distance: float = 50.0,
        caution_distance: float = 30.0,
        danger_distance: float = 15.0,
        critical_distance: float = 8.0,
        # Lane change parameters
        min_lane_advantage: float = 10.0,  # Must be 10m+ better to change
        decision_cooldown: float = 1.0,  # Min seconds between lane decisions
        # Debug
        verbose: bool = False
    ):
        self.num_lanes = num_lanes
        self.cruise_speed = cruise_speed
        self.cautious_speed = cautious_speed
        self.slow_speed = slow_speed
        self.emergency_speed = emergency_speed
        
        self.safe_distance = safe_distance
        self.caution_distance = caution_distance
        self.danger_distance = danger_distance
        self.critical_distance = critical_distance
        
        self.min_lane_advantage = min_lane_advantage
        self.decision_cooldown = decision_cooldown
        self.verbose = verbose
        
        # State
        self.current_target_lane: int = 0
        self.current_desired_speed: float = cruise_speed
        self.time_since_decision: float = 0.0
        self.step_count: int = 0
        
    def reset(self, start_lane: int = 0):
        """Reset decision layer state."""
        self.current_target_lane = start_lane
        self.current_desired_speed = self.cruise_speed
        # Allow immediate decision on reset (set to cooldown so first decide() can act)
        self.time_since_decision = self.decision_cooldown
        self.step_count = 0
        
    def decide(
        self,
        current_lane: int,
        obstacle_dists: np.ndarray,
        traffic_dists: np.ndarray,
        relative_velocities: np.ndarray,
        lane_blocked: np.ndarray,
        ego_speed: float,
        dt: float = 0.05
    ) -> Tuple[int, float, str]:
        """
        Make strategic decision based on environment state.
        
        Args:
            current_lane: Current lane index (0 = rightmost)
            obstacle_dists: Distance to nearest obstacle in each lane [N lanes]
            traffic_dists: Distance to nearest traffic in each lane [N lanes]
            relative_velocities: Closing rate per lane (positive = closing gap)
            lane_blocked: Boolean flags for blocked lanes
            ego_speed: Current vehicle speed
            dt: Time step
            
        Returns:
            target_lane: Which lane to be in
            desired_speed: Target speed in m/s
            decision_reason: String explaining the decision (for debugging)
        """
        self.step_count += 1
        self.time_since_decision += dt
        
        # Combine obstacle and traffic into single threat distance per lane
        threat_dists = np.minimum(obstacle_dists, traffic_dists)
        
        # Current lane threat
        current_threat = threat_dists[current_lane]
        current_closing = relative_velocities[current_lane]
        
        # Adjust threat based on closing rate
        # If gap is closing fast, treat it as closer than it is
        effective_threat = current_threat
        if current_closing > 0:  # Gap is closing
            # Time to collision estimate
            ttc = current_threat / (current_closing + 0.1)
            if ttc < 3.0:  # Less than 3 seconds to collision
                effective_threat = min(effective_threat, ttc * 5)  # Treat as 5x closer
        
        # ===== SPEED DECISION =====
        if effective_threat > self.safe_distance:
            desired_speed = self.cruise_speed
            speed_reason = "clear"
        elif effective_threat > self.caution_distance:
            desired_speed = self.cautious_speed
            speed_reason = "caution"
        elif effective_threat > self.danger_distance:
            desired_speed = self.slow_speed
            speed_reason = "slow"
        else:
            desired_speed = self.emergency_speed
            speed_reason = "emergency"
            
        # ===== LANE DECISION =====
        target_lane = current_lane
        lane_reason = "stay"
        
        # Only consider lane change if cooldown expired
        if self.time_since_decision >= self.decision_cooldown:
            # Find best lane
            best_lane = current_lane
            best_threat = threat_dists[current_lane]
            
            for lane in range(self.num_lanes):
                if lane == current_lane:
                    continue
                    
                # Check if this lane is significantly better
                lane_threat = threat_dists[lane]
                lane_closing = relative_velocities[lane]
                
                # Adjust for closing rate in target lane too
                effective_lane_threat = lane_threat
                if lane_closing > 0:
                    ttc = lane_threat / (lane_closing + 0.1)
                    if ttc < 3.0:
                        effective_lane_threat = min(effective_lane_threat, ttc * 5)
                
                # Only change if significantly better
                if effective_lane_threat > best_threat + self.min_lane_advantage:
                    # Additional check: don't change into blocked lane
                    if not lane_blocked[lane]:
                        best_lane = lane
                        best_threat = effective_lane_threat
            
            if best_lane != current_lane:
                target_lane = best_lane
                lane_reason = f"change_to_{best_lane}"
                self.time_since_decision = 0.0  # Reset cooldown
                
        # Check if we should stay with previous target (commitment)
        # If we're mid-lane-change, keep the target
        if self.current_target_lane != current_lane:
            # We're in the middle of a lane change
            target_lane = self.current_target_lane
            lane_reason = "committing"
            
        # Emergency override: if current path is critically blocked, force decision
        if effective_threat < self.critical_distance and lane_blocked[current_lane]:
            # Find any safe lane immediately
            for lane in range(self.num_lanes):
                if lane != current_lane and threat_dists[lane] > self.danger_distance:
                    target_lane = lane
                    lane_reason = "emergency_change"
                    self.time_since_decision = 0.0
                    break
        
        # Update state
        self.current_target_lane = target_lane
        self.current_desired_speed = desired_speed
        
        decision_reason = f"{lane_reason}|{speed_reason}"
        
        if self.verbose and self.step_count % 20 == 0:
            print(f"  [Decision] Step {self.step_count}: "
                  f"threat={effective_threat:.1f}m, "
                  f"target_lane={target_lane}, speed={desired_speed:.1f}, "
                  f"reason={decision_reason}")
        
        return target_lane, desired_speed, decision_reason
    
    def decide_from_obs(
        self,
        obs: np.ndarray,
        num_lanes: int,
        dt: float = 0.05
    ) -> Tuple[int, float, str]:
        """
        Make decision directly from observation array.
        
        Parses the observation to extract relevant features and calls decide().
        
        Observation layout (for 2 lanes, 23 dims):
        [0]: lane_offset_norm
        [1]: heading_error
        [2]: speed_norm (speed / v_max, where v_max=20)
        [3]: yaw_rate
        [4]: lookahead_curv
        [5]: prev_accel_norm
        [6]: prev_steer_norm
        [7-11]: distance_sensors (5)
        [12-13]: current_lane_onehot (2 lanes)
        [14]: lane_deviation_norm
        [15-16]: obstacle_dist_norm per lane (2)
        [17-18]: traffic_dist_norm per lane (2)
        [19-20]: relative_velocity_norm per lane (2)
        [21-22]: lane_blocked per lane (2)
        
        For 3 lanes (28 dims), indices shift accordingly.
        """
        # Parse observation
        speed_norm = obs[2]
        ego_speed = speed_norm * 20.0  # Denormalize (v_max = 20)
        
        # Current lane from one-hot encoding
        lane_onehot_start = 12
        lane_onehot = obs[lane_onehot_start:lane_onehot_start + num_lanes]
        current_lane = int(np.argmax(lane_onehot))
        
        # Per-lane features start after lane_deviation
        per_lane_start = lane_onehot_start + num_lanes + 1
        
        # Obstacle distances (normalized by 60m)
        obs_dist_norm = obs[per_lane_start:per_lane_start + num_lanes]
        obstacle_dists = obs_dist_norm * 60.0
        
        # Traffic distances
        traffic_start = per_lane_start + num_lanes
        traffic_dist_norm = obs[traffic_start:traffic_start + num_lanes]
        traffic_dists = traffic_dist_norm * 60.0
        
        # Relative velocities (normalized by 15 m/s)
        rel_vel_start = traffic_start + num_lanes
        rel_vel_norm = obs[rel_vel_start:rel_vel_start + num_lanes]
        relative_velocities = rel_vel_norm * 15.0
        
        # Lane blocked flags
        blocked_start = rel_vel_start + num_lanes
        lane_blocked = obs[blocked_start:blocked_start + num_lanes]
        
        return self.decide(
            current_lane=current_lane,
            obstacle_dists=obstacle_dists,
            traffic_dists=traffic_dists,
            relative_velocities=relative_velocities,
            lane_blocked=lane_blocked,
            ego_speed=ego_speed,
            dt=dt
        )


def augment_observation(
    obs: np.ndarray,
    target_lane: int,
    desired_speed: float,
    num_lanes: int,
    max_speed: float = 20.0
) -> np.ndarray:
    """
    Augment observation with decision layer outputs.
    
    Adds normalized target_lane and desired_speed to observation.
    
    Args:
        obs: Original observation array
        target_lane: Target lane from decision layer
        desired_speed: Desired speed from decision layer
        num_lanes: Number of lanes for normalization
        max_speed: Max speed for normalization
        
    Returns:
        Augmented observation with 2 additional features
    """
    # Normalize
    target_lane_norm = target_lane / max(1, num_lanes - 1)  # 0-1
    desired_speed_norm = desired_speed / max_speed  # 0-1
    
    # Concatenate
    return np.concatenate([obs, [target_lane_norm, desired_speed_norm]])


# For testing
if __name__ == "__main__":
    # Test decision layer
    dl = DecisionLayer(num_lanes=2, verbose=True)
    dl.reset(start_lane=0)
    
    # Simulate some scenarios
    print("\n=== Test 1: Clear road ===")
    target, speed, reason = dl.decide(
        current_lane=0,
        obstacle_dists=np.array([60.0, 60.0]),
        traffic_dists=np.array([60.0, 60.0]),
        relative_velocities=np.array([0.0, 0.0]),
        lane_blocked=np.array([0.0, 0.0]),
        ego_speed=10.0
    )
    print(f"Decision: lane={target}, speed={speed}, reason={reason}")
    
    print("\n=== Test 2: Obstacle ahead in current lane ===")
    target, speed, reason = dl.decide(
        current_lane=0,
        obstacle_dists=np.array([25.0, 60.0]),
        traffic_dists=np.array([60.0, 60.0]),
        relative_velocities=np.array([0.0, 0.0]),
        lane_blocked=np.array([0.0, 0.0]),
        ego_speed=10.0,
        dt=2.0  # Simulate time passing
    )
    print(f"Decision: lane={target}, speed={speed}, reason={reason}")
    
    print("\n=== Test 3: Both lanes have traffic ===")
    dl.time_since_decision = 2.0  # Reset cooldown
    target, speed, reason = dl.decide(
        current_lane=0,
        obstacle_dists=np.array([60.0, 60.0]),
        traffic_dists=np.array([20.0, 40.0]),
        relative_velocities=np.array([5.0, 2.0]),  # Closing on both
        lane_blocked=np.array([0.0, 0.0]),
        ego_speed=10.0
    )
    print(f"Decision: lane={target}, speed={speed}, reason={reason}")
