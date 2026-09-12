"""
Expert Controller for Multi-Lane Environment with Obstacle and Traffic Avoidance.

Uses Pure Pursuit + lane change decisions based on obstacle/traffic detection.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

from autonomous_car.env.track import Track
from autonomous_car.env.multilane_env import Obstacle, Vehicle


class MultiLaneExpertController:
    """
    Expert controller for multi-lane driving with obstacle and traffic avoidance.
    
    Strategy:
    1. Detect obstacles and traffic ahead in current lane
    2. If obstacle/traffic detected, plan lane change if safe
    3. Use Pure Pursuit to track target lane centerline
    4. PID for speed control with obstacle-aware braking
    """
    
    def __init__(
        self,
        track: Track,
        num_lanes: int = 3,
        lane_width: float = 3.5,
        L: float = 2.5,
        lookahead_dist: float = 10.0,
        v_ref: float = 10.0,
        obstacle_detect_dist: float = 40.0,
        lane_change_threshold: float = 25.0,
        # PID gains
        kp: float = 1.5,
        ki: float = 0.1,
        kd: float = 0.05,
        # Limits
        a_max: float = 3.0,
        delta_max: float = 0.5,
        # Optional noise
        action_noise: float = 0.0,
        # Logging
        verbose: bool = False,
    ):
        self.track = track
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.L = L
        self.lookahead_dist = lookahead_dist
        self.v_ref = v_ref
        self.obstacle_detect_dist = obstacle_detect_dist
        self.lane_change_threshold = lane_change_threshold
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.a_max = a_max
        self.delta_max = delta_max
        self.action_noise = action_noise
        self.verbose = verbose
        
        # Step counter for logging
        self.step_count = 0
        
        # PID state
        self._integral_error = 0.0
        self._prev_speed_error = 0.0
        
        # Lane change state
        self.target_lane: int = num_lanes // 2
        self.lane_change_cooldown: float = 0.0
        self.emergency_brake: bool = False
        self.phase: str = "CRUISE"
        self.recovery_target_lane: Optional[int] = None
        # Brake-deadlock tracking: escalate to a forced escape when braking
        # cannot clear the obstacle (stopped or timed out in BRAKE).
        self._brake_time: float = 0.0
        self._low_speed_streak: int = 0
        self._decision_dt: float = 0.5
        self._commit_elapsed: float = 0.0
        # Liveness oracle (RSS safe-state / SOTIF triggering-condition /
        # STCLocker deadlock-oracle pattern): detects the trap state where
        # the planner believes CRUISE is fine while the env safety layer has
        # pinned achieved speed to ~0 - a planner-belief vs safety-reality
        # disagreement invisible to clearance metrics alone.
        self.liveness_pinned_time: float = 0.0
        self.liveness_attempts: int = 0
        self.liveness_cooldown: float = 0.0
        self.liveness_blocked: bool = False
        self._liveness_hold: bool = False
        self.liveness_max_attempts: int = 3
        self.liveness_pin_threshold: float = 1.0   # s of pinned v before trigger
        self.liveness_cooldown_time: float = 2.0   # s between attempts
        
        # Obstacles and traffic reference (set by environment)
        self.obstacles: List[Obstacle] = []
        self.traffic_vehicles: List[Vehicle] = []
        
        # Last computed hierarchical target [target_lane_offset, target_speed]
        # Used for BC-SAC demo collection in the hierarchical action space.
        self.last_target_offset: float = 0.0
        self.last_target_speed: float = v_ref
        self.last_escape_distance: float = float("inf")
        self.last_escape_feasible: bool = False
        self._ego_state: Optional[np.ndarray] = None
    
    def reset(self):
        """Reset controller state."""
        self._integral_error = 0.0
        self._prev_speed_error = 0.0
        self.target_lane = self.num_lanes // 2
        self.lane_change_cooldown = 0
        self.emergency_brake = False
        self.phase = "CRUISE"
        self.recovery_target_lane = None
        self._brake_time = 0.0
        self._low_speed_streak = 0
        self._commit_elapsed = 0.0
        self.liveness_pinned_time = 0.0
        self.liveness_attempts = 0
        self.liveness_cooldown = 0.0
        self.liveness_blocked = False
        self._liveness_hold = False
        self.step_count = 0
        self.last_target_offset = self._get_lane_center_offset(self.target_lane)
        self.last_target_speed = self.v_ref
        self.last_escape_distance = float("inf")
        self.last_escape_feasible = False
        self._ego_state = None
    
    def set_obstacles(self, obstacles: List[Obstacle]):
        """Set current obstacles."""
        self.obstacles = obstacles
    
    def set_traffic(self, traffic_vehicles: List[Vehicle]):
        """Set current traffic vehicles."""
        self.traffic_vehicles = traffic_vehicles
    
    def _get_lane_center_offset(self, lane: int) -> float:
        """Get lateral offset for lane center from track centerline."""
        return (lane - (self.num_lanes - 1) / 2) * self.lane_width
    
    def _get_current_lane(self, lateral_offset: float) -> int:
        """Determine current lane from lateral offset."""
        for lane in range(self.num_lanes):
            lane_center = self._get_lane_center_offset(lane)
            if abs(lateral_offset - lane_center) < self.lane_width / 2:
                return lane
        return max(0, min(self.num_lanes - 1,
                         int((lateral_offset / self.lane_width) + (self.num_lanes - 1) / 2 + 0.5)))

    def _vehicle_occupies_lane(self, vehicle: Vehicle, lane: int) -> bool:
        """Treat a transitioning vehicle as occupying its swept footprint."""
        vehicle_center = (
            self._get_lane_center_offset(vehicle.lane) + vehicle.lateral_offset
        )
        lane_center = self._get_lane_center_offset(lane)
        return abs(vehicle_center - lane_center) <= (
            self.lane_width / 2.0 + vehicle.width / 2.0
        )

    def _physical_lane_clearance(self, lane: int, current_s: float) -> float:
        """Estimate clearance using continuous traffic footprints and gaps."""
        track_length = self.track.get_total_length()
        clearance = 100.0
        for obstacle in self.obstacles:
            if obstacle.lane != lane:
                continue
            ahead = (obstacle.s - current_s) % track_length
            if 0.0 < ahead < clearance:
                clearance = ahead - obstacle.length / 2.0
        for vehicle in self.traffic_vehicles:
            if not self._vehicle_occupies_lane(vehicle, lane):
                continue
            signed = (vehicle.s - current_s) % track_length
            if signed > track_length / 2.0:
                signed -= track_length
            clearance = min(clearance, abs(signed) - vehicle.length / 2.0)
        return max(0.0, clearance)

    def _select_physical_escape(self, current_lane: int, current_s: float,
                                speed: float) -> Optional[int]:
        """Choose an escape lane before adjacent traffic becomes a contact."""
        candidates = []
        for lane in range(self.num_lanes):
            if lane == current_lane:
                continue
            clearance = self._physical_lane_clearance(lane, current_s)
            if clearance <= 8.0:
                continue
            if not self._is_lane_safe(lane, current_s, max(10.0, speed + 5.0)):
                continue
            path = range(min(current_lane, lane), max(current_lane, lane) + 1)
            path_clearance = min(
                self._physical_lane_clearance(path_lane, current_s)
                for path_lane in path
            )
            if path_clearance > 5.0:
                candidates.append((clearance - abs(lane - current_lane) * 2.0, lane))
        return max(candidates)[1] if candidates else None

    def _lane_change_required_distance(self, lateral_distance: float,
                                       speed: float) -> float:
        """Estimate distance needed to finish a steerable lane transition."""
        speed = max(float(speed), 0.5)
        max_lateral_rate = max(speed * np.sin(self.delta_max), 0.5)
        transition_time = lateral_distance / max_lateral_rate
        reaction_distance = speed * max(self._decision_dt, 0.0)
        vehicle_clearance = 2.25 + 1.0 + 0.3
        return (
            speed * transition_time
            + reaction_distance
            + vehicle_clearance
        )

    @staticmethod
    def _boxes_overlap(center_a, heading_a, half_a, center_b, heading_b, half_b,
                       margin: float = 0.0) -> bool:
        """Exact oriented-box overlap via SAT (mirrors MultiLaneEnv._obb_overlap)."""
        axes = []
        for heading in (heading_a, heading_b):
            axes.extend((
                np.array([np.cos(heading), np.sin(heading)]),
                np.array([-np.sin(heading), np.cos(heading)]),
            ))
        delta = np.asarray(center_b) - np.asarray(center_a)
        for axis in axes:
            axis = axis / np.linalg.norm(axis)
            basis_a = (
                np.array([np.cos(heading_a), np.sin(heading_a)]),
                np.array([-np.sin(heading_a), np.cos(heading_a)]),
            )
            basis_b = (
                np.array([np.cos(heading_b), np.sin(heading_b)]),
                np.array([-np.sin(heading_b), np.cos(heading_b)]),
            )
            radius_a = sum(size * abs(np.dot(basis, axis))
                           for size, basis in zip(half_a, basis_a))
            radius_b = sum(size * abs(np.dot(basis, axis))
                           for size, basis in zip(half_b, basis_b))
            if abs(float(np.dot(delta, axis))) > radius_a + radius_b + margin:
                return False
        return True

    def _trajectory_is_clear(self, target_lane: int, speed: float) -> bool:
        """Roll out the bicycle tracker and predicted traffic to target lane."""
        if self._ego_state is None or speed < 0.5:
            return False
        track_length = self.track.get_total_length()
        state = self._ego_state.astype(np.float64).copy()
        target_offset = self._get_lane_center_offset(target_lane)
        reached_target = False
        # Bound the rollout sampling rate. decision_dt tracks the caller's sim
        # step (env.dt=0.05) -> a naive 4.0/decision_dt rollout runs 80 steps
        # x 32 entities (~322ms) per feasibility check, freezing the window in
        # boxed states (profiled: _trajectory_is_clear avg 322ms). Sample at
        # <=0.2s: 20 samples over the 4s horizon. Max relative move per sample
        # = 0.2s * 19 m/s = 3.8m < min 4.5m box length -> no tunneling, still
        # collision-valid.
        rollout_dt = max(self._decision_dt, 0.2)
        max_steps = max(1, int(np.ceil(4.0 / rollout_dt)))

        for step in range(max_steps):
            x, y, theta, v = state
            closest, lateral = self.track.find_closest_point(x, y)
            lookahead = max(3.0, v)
            point = self.track.get_centerline_point(
                (closest.s + lookahead) % track_length
            )
            nx, ny = -np.sin(point.heading), np.cos(point.heading)
            target_x = point.x + target_offset * nx
            target_y = point.y + target_offset * ny
            alpha = np.arctan2(target_y - y, target_x - x) - theta
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
            steer = np.clip(
                np.arctan2(2.0 * self.L * np.sin(alpha), lookahead),
                -self.delta_max,
                self.delta_max,
            )
            # Integrate the rollout at a representative crawl speed: a
            # parked ego (v~0) would otherwise never translate in the
            # rollout and every escape would look infeasible forever.
            rollout_v = max(float(v), 1.5)
            next_state = np.array([
                x + rollout_v * np.cos(theta) * rollout_dt,
                y + rollout_v * np.sin(theta) * rollout_dt,
                theta + (rollout_v / self.L) * np.tan(steer) * rollout_dt,
                v,
            ])

            _, next_lateral = self.track.find_closest_point(
                next_state[0], next_state[1]
            )
            if abs(next_lateral - target_offset) < self.lane_width * 0.2:
                reached_target = True
            if abs(next_lateral) > self.lane_width * (self.num_lanes / 2.0 + 0.1):
                return False

            for obstacle in self.obstacles:
                if self._boxes_overlap(
                    next_state[:2], next_state[2], (2.25, 0.9),
                    (obstacle.x, obstacle.y), obstacle.heading,
                    (obstacle.length / 2.0, obstacle.width / 2.0),
                ):
                    return False
            for vehicle in self.traffic_vehicles:
                elapsed = (step + 1) * rollout_dt
                lateral_offset = vehicle.lateral_offset
                if lateral_offset > 0.0:
                    lateral_offset = max(0.0, lateral_offset - 1.5 * elapsed)
                elif lateral_offset < 0.0:
                    lateral_offset = min(0.0, lateral_offset + 1.5 * elapsed)
                point = self.track.get_centerline_point(
                    (vehicle.s + vehicle.speed * elapsed) % track_length
                )
                vehicle_offset = self._get_lane_center_offset(vehicle.lane) + lateral_offset
                vehicle_pose = (
                    point.x - np.sin(point.heading) * vehicle_offset,
                    point.y + np.cos(point.heading) * vehicle_offset,
                    point.heading,
                )
                if self._boxes_overlap(
                    next_state[:2], next_state[2], (2.25, 0.9),
                    vehicle_pose[:2], vehicle_pose[2],
                    (vehicle.length / 2.0, vehicle.width / 2.0),
                ):
                    return False
            state = next_state

        return reached_target

    def _lane_change_feasible(self, current_lane: int, target_lane: int,
                              current_s: float, speed: float) -> bool:
        """Require enough swept-path distance for a physically executable escape."""
        if target_lane == current_lane:
            self.last_escape_distance = float("inf")
            self.last_escape_feasible = False
            return False
        lateral_distance = abs(
            self._get_lane_center_offset(target_lane)
            - self._get_lane_center_offset(current_lane)
        )
        required = self._lane_change_required_distance(lateral_distance, speed)
        swept_clearance = min(
            self._lane_clearance(path_lane, current_s, horizon=100.0)
            for path_lane in range(
                min(current_lane, target_lane), max(current_lane, target_lane) + 1
            )
        )
        target_clearance = self._lane_clearance(target_lane, current_s, 100.0)
        self.last_escape_distance = required
        self.last_escape_feasible = (
            swept_clearance >= required
            and target_clearance >= required
            and self._is_lane_safe(target_lane, current_s, 28.0)
            and self._trajectory_is_clear(target_lane, speed)
        )
        return self.last_escape_feasible
    
    def _find_obstacles_ahead(
        self,
        current_s: float,
        lane: int,
        detect_dist: float
    ) -> List[Tuple[Union[Obstacle, Vehicle], float]]:
        """Find obstacles AND traffic ahead in given lane within detection distance.
        
        Returns list of (object, distance) tuples sorted by distance.
        Objects can be either static Obstacles or moving Vehicles.
        """
        track_length = self.track.get_total_length()
        objects_ahead = []
        
        # Check static obstacles
        for obs in self.obstacles:
            if obs.lane != lane:
                continue
            
            # Distance along track
            dist_s = obs.s - current_s
            if dist_s < 0:
                dist_s += track_length  # wrapped
            
            # Also check behind for safety
            if dist_s > track_length - 15:  # obstacle is behind but close
                continue
            
            if 0 < dist_s < detect_dist:
                objects_ahead.append((obs, dist_s))
        
        # Check traffic vehicles
        for veh in self.traffic_vehicles:
            if not self._vehicle_occupies_lane(veh, lane):
                continue
            
            dist_s = veh.s - current_s
            if dist_s < 0:
                dist_s += track_length
            
            if dist_s > track_length - 15:
                continue
            
            if 0 < dist_s < detect_dist:
                objects_ahead.append((veh, dist_s))
        
        return sorted(objects_ahead, key=lambda x: x[1])
    
    def _is_lane_safe(self, lane: int, current_s: float, safety_dist: float = 35.0) -> bool:
        """Check if lane is safe for lane change.
        
        Checks both ahead and behind for traffic. A lane clear ahead but
        occupied by an approaching vehicle from behind is unsafe to enter.
        """
        if lane < 0 or lane >= self.num_lanes:
            return False
        
        track_length = self.track.get_total_length()
        for obstacle in self.obstacles:
            if obstacle.lane != lane:
                continue
            ahead = (obstacle.s - current_s) % track_length
            if 0.0 < ahead < safety_dist:
                return False

        for vehicle in self.traffic_vehicles:
            if not self._vehicle_occupies_lane(vehicle, lane):
                continue
            ahead = (vehicle.s - current_s) % track_length
            behind = (current_s - vehicle.s) % track_length
            if min(ahead, behind) < safety_dist:
                return False

        return True

    def _lane_clearance(self, lane: int, current_s: float, horizon: float = 100.0,
                         ego_speed: Optional[float] = None) -> float:
        """Return the usable forward clearance in a lane over a horizon.

        ego_speed, when provided, makes the traffic model velocity-aware:
        a leading vehicle strictly faster than the ego has an opening gap
        and cannot be caught, so it does not reduce forward clearance.
        """
        track_length = self.track.get_total_length()
        clearance = horizon
        lane_pt = self.track.get_centerline_point(current_s % track_length)
        lane_offset = self._get_lane_center_offset(lane)
        ego_lane_x = lane_pt.x - np.sin(lane_pt.heading) * lane_offset
        ego_lane_y = lane_pt.y + np.cos(lane_pt.heading) * lane_offset
        for obstacle in self.obstacles:
            if obstacle.lane != lane:
                continue
            distance = (obstacle.s - current_s) % track_length
            if 0.0 < distance < clearance:
                clearance = distance
        for vehicle in self.traffic_vehicles:
            if not self._vehicle_occupies_lane(vehicle, lane):
                continue
            ahead = (vehicle.s - current_s) % track_length
            # Velocity-aware hazard (RSS catch-up): a leading vehicle strictly
            # faster than the ego has an opening gap and cannot be caught, so
            # it is not a forward-clearance hazard. Trailing and slower/equal
            # vehicles keep the existing conservative behavior.
            if (ego_speed is not None
                    and 0.0 < ahead < track_length / 2.0
                    and vehicle.speed > ego_speed):
                continue
            world_gap = np.hypot(vehicle.x - ego_lane_x, vehicle.y - ego_lane_y)
            physical_gap = world_gap - (2.25 + vehicle.length / 2.0 + 0.3)
            if world_gap < 30.0:
                clearance = min(clearance, physical_gap)
            behind = (current_s - vehicle.s) % track_length
            if behind < 20.0:
                clearance = min(clearance, 0.0)
            elif 0.0 < ahead < clearance:
                closing = max(0.0, self.v_ref - vehicle.speed)
                clearance = min(clearance, ahead - min(10.0, closing * 1.5))
        return max(0.0, clearance)

    def _choose_horizon_lane(self, current_lane: int, current_s: float,
                             speed: float = 0.0) -> int:
        """Choose the safest lane using 100m clearance, not nearest object only.

        Sweep-path aware: a candidate is only viable if every lane between the
        current lane and the candidate is itself passable at the current speed.
        """
        stop_dist = speed ** 2 / (2.0 * self.a_max)
        veto_threshold = stop_dist + 5.0
        candidates = []
        for lane in range(self.num_lanes):
            clearance = self._lane_clearance(lane, current_s, horizon=100.0)
            if lane != current_lane:
                swept = range(min(current_lane, lane), max(current_lane, lane) + 1)
                path_worst = min(
                    self._lane_clearance(path_lane, current_s, horizon=50.0)
                    for path_lane in swept
                )
                if path_worst < min(veto_threshold, clearance):
                    clearance = 0.0
                elif not self._is_lane_safe(lane, current_s, 28.0):
                    clearance = 0.0
            lane_change_cost = abs(lane - current_lane) * 2.0
            candidates.append((clearance - lane_change_cost, clearance, lane))
        candidates.sort(reverse=True)
        return candidates[0][2]

    def _forced_escape_lane(self, current_lane: int, current_s: float,
                            speed: float = 0.0) -> Optional[int]:
        """Pick the best escape lane using desperate-mode gap acceptance.

        When staying in the origin lane is already inside the stopping
        envelope, the origin lane's own hazard must not poison the swept-path
        check - otherwise a boxed vehicle rejects every candidate and sits
        until contact. Destination and intermediate (non-origin) lanes are
        judged with lateral-aware physical footprint gaps, and the surviving
        candidate must pass full trajectory validation.
        """
        speed = max(float(speed), 0.0)
        stay_lethal = (
            self._lane_clearance(current_lane, current_s, 100.0)
            <= speed ** 2 / (2.0 * self.a_max)
        )
        best_lane = None
        best_score = float("-inf")
        adjacent = [
            lane for lane in (current_lane - 1, current_lane + 1)
            if 0 <= lane < self.num_lanes
        ]
        for lane in adjacent:
            clearance = self._physical_lane_clearance(lane, current_s)
            if clearance <= 2.2:
                continue
            path_lanes = [
                path_lane
                for path_lane in range(min(current_lane, lane), max(current_lane, lane) + 1)
                if path_lane != current_lane or not stay_lethal
            ]
            path_clearance = min(
                self._physical_lane_clearance(path_lane, current_s)
                for path_lane in path_lanes
            ) if path_lanes else clearance
            if path_clearance <= 1.0:
                continue
            if not self._trajectory_is_clear(lane, max(speed, 1.5)):
                continue
            score = min(clearance, 30.0) - abs(lane - current_lane) * 2.0
            if score > best_score:
                best_score = score
                best_lane = lane
        return best_lane

    def _update_liveness_oracle(
        self, current_lane: int, current_s: float, speed: float,
        current_clearance: float,
    ) -> Optional[int]:
        """Black-box deadlock oracle for the CRUISE trap state.

        Detection: commanded cruise speed while the ACHIEVED speed stays
        pinned near zero - regardless of why (shield clamp, unseen hazard).
        Standards-shaped response:
          1. evaluate against safe-state criteria (_forced_escape_lane),
          2. bounded retries with cooldown (livelock prevention),
          3. declared degraded state after exhaustion (MRC analog).

        Returns an escape lane to commit to, or None (keep current lane).
        """
        pinned_now = (
            self.phase == "CRUISE"
            and self.last_target_speed > 2.0
            and speed < 0.3
        )
        if pinned_now:
            self.liveness_pinned_time += self._decision_dt
        else:
            self.liveness_pinned_time = 0.0

        self.liveness_cooldown = max(
            0.0, self.liveness_cooldown - self._decision_dt
        )

        self._liveness_hold = False
        # Declared failure: hold the honest degraded state, no more retries.
        if self.liveness_blocked:
            self.phase = "BRAKE"
            self.emergency_brake = True
            self._liveness_hold = True
            return None

        if (
            self.liveness_pinned_time < self.liveness_pin_threshold
            or self.liveness_cooldown > 0.0
        ):
            return None

        # Trigger: bounded re-evaluation.
        self.liveness_attempts += 1
        self.liveness_cooldown = self.liveness_cooldown_time
        escape_lane = self._forced_escape_lane(current_lane, current_s, speed)
        if escape_lane is not None and escape_lane != current_lane:
            self.target_lane = escape_lane
            self.recovery_target_lane = escape_lane
            self.phase = "COMMIT"
            self.emergency_brake = False
            self.lane_change_cooldown = 0.0
            self._brake_time = 0.0
            self._low_speed_streak = 0
            if self.verbose:
                print(
                    f"  [Step {self.step_count}] \u26d1 LIVENESS ESCAPE: "
                    f"Lane {current_lane} -> Lane {escape_lane} "
                    f"(pinned={self.liveness_pinned_time:.1f}s, "
                    f"attempt {self.liveness_attempts}/"
                    f"{self.liveness_max_attempts})"
                )
            return escape_lane

        # No viable escape: enter the honest degraded state immediately.
        if self.verbose:
            print(
                f"  [Step {self.step_count}] \u26d4 LIVENESS: no viable "
                f"escape from Lane {current_lane} (attempt "
                f"{self.liveness_attempts}/{self.liveness_max_attempts})"
            )
        self.phase = "BRAKE"
        self.emergency_brake = True
        self._liveness_hold = True
        if self.liveness_attempts >= self.liveness_max_attempts:
            self.liveness_blocked = True
            if self.verbose:
                print(
                    f"  [Step {self.step_count}] \u26d4 LIVENESS EXHAUSTED: "
                    f"declaring blocked (minimal-risk condition)"
                )
        return None

    def _update_recovery_state(self, current_lane: int, current_s: float, speed: float) -> int:
        """Maintain a lane-change recovery state until the maneuver completes."""
        current_clearance = self._lane_clearance(current_lane, current_s, 100.0, ego_speed=speed)

        # Liveness oracle runs BEFORE belief-based branches: it keys off
        # observed motion, so it catches disagreements the clearance metric
        # cannot see (smoke test seed 42 step ~160: expert said CRUISE with
        # 80m+ clearance while the env clamp held v=0 until stall).
        liveness_escape = self._update_liveness_oracle(
            current_lane, current_s, speed, current_clearance
        )
        if self._liveness_hold:
            # Degraded state was just declared this pass; do not let the
            # belief-based branches below overwrite it back to CRUISE.
            self._liveness_hold = False
            return current_lane
        if liveness_escape is not None:
            return liveness_escape

        # Commitment hysteresis: a COMMIT must either complete or escalate to a
        # forced escape within a bounded time. Proposals may not flip the target
        # back and forth while the maneuver is pending.
        if self.phase == "COMMIT" and current_lane != self.target_lane:
            self._commit_elapsed += self._decision_dt
        else:
            self._commit_elapsed = 0.0

        commit_stalled = (
            self.phase == "COMMIT"
            and current_lane != self.target_lane
            and self._commit_elapsed >= 2.5
            and current_clearance < 30.0
        )
        if commit_stalled:
            escape_lane = self._forced_escape_lane(self.target_lane, current_s, speed)
            if escape_lane is not None and escape_lane != self.target_lane:
                self.target_lane = escape_lane
                self.recovery_target_lane = escape_lane
                self._commit_elapsed = 0.0
                if self.verbose:
                    print(
                        f"  [Step {self.step_count}] ⛑ EARLY ESCAPE: "
                        f"stalled COMMIT → Lane {escape_lane} "
                        f"(elapsed={self._commit_elapsed:.1f}s, "
                        f"clearance={current_clearance:.1f}m)"
                    )
            elif escape_lane is None and speed < 3.0:
                self.phase = "BRAKE"

        if self.phase == "COMMIT" and current_lane != self.target_lane:
            if (
                self._lane_clearance(self.target_lane, current_s, 30.0) > 8.0
                and self._lane_change_feasible(
                    current_lane, self.target_lane, current_s, speed
                )
            ):
                return self.target_lane
            self.phase = "REPLAN"
        if self.phase == "COMMIT" and current_lane == self.target_lane:
            self.phase = "CRUISE"
            self.recovery_target_lane = None

        if self.phase in ("PREPARE", "COMMIT") and self.target_lane != current_lane:
            if self._lane_clearance(self.target_lane, current_s, 50.0) > 15.0:
                self.phase = "COMMIT"
                return self.target_lane
            self.phase = "REPLAN"

        if current_clearance < 80.0:
            # Early escape trigger (fix 5): if the predicted stopping point
            # leaves less than a 15 m execution window and no maneuver is
            # already in progress, escape NOW while still moving instead of
            # braking into the unrecoverable zone first.
            stop_dist_now = speed ** 2 / (2.0 * self.a_max)
            if (current_clearance < stop_dist_now + 15.0
                    and self.phase not in ("PREPARE", "COMMIT")):
                escape_lane = self._forced_escape_lane(current_lane, current_s, speed)
                if escape_lane is not None:
                    self.target_lane = escape_lane
                    self.recovery_target_lane = escape_lane
                    self.phase = "COMMIT"
                    self.emergency_brake = False
                    self.lane_change_cooldown = 0.0
                    self._brake_time = 0.0
                    self._low_speed_streak = 0
                    if self.verbose:
                        print(
                            f"  [Step {self.step_count}] ⛑ EARLY ESCAPE: "
                            f"Lane {current_lane} → Lane {escape_lane} "
                            f"(clearance={current_clearance:.1f}m, "
                            f"v={speed:.1f}m/s)"
                        )
                    return escape_lane

            chosen_lane = self._choose_horizon_lane(
                current_lane, current_s, speed
            )
            chosen_clearance = self._lane_clearance(chosen_lane, current_s, 100.0)
            if (
                chosen_lane != current_lane
                and chosen_clearance > current_clearance
                and self._lane_change_feasible(
                    current_lane, chosen_lane, current_s, speed
                )
            ):
                self.target_lane = chosen_lane
                self.recovery_target_lane = chosen_lane
                self.phase = "PREPARE"
                if speed > 1.0:
                    self.phase = "COMMIT"
                self._brake_time = 0.0
                self._low_speed_streak = 0
                return chosen_lane

            # Deadlock detection: braking is not clearing the obstacle. After a
            # brake timeout or repeated near-stop with tiny clearance, force an
            # escape commitment instead of sitting in front of the hazard.
            self._brake_time += self._decision_dt
            if speed < 1.0:
                self._low_speed_streak += 1
            else:
                self._low_speed_streak = 0

            brake_timed_out = self._brake_time >= 3.0
            stopped_blocked = (
                self._low_speed_streak >= 3 and current_clearance < 10.0
            )
            if brake_timed_out or stopped_blocked:
                escape_lane = self._forced_escape_lane(current_lane, current_s, speed)
                if escape_lane is not None:
                    self.target_lane = escape_lane
                    self.recovery_target_lane = escape_lane
                    self.phase = "COMMIT"
                    self.emergency_brake = False
                    self.lane_change_cooldown = 0.0
                    self._brake_time = 0.0
                    self._low_speed_streak = 0
                    if self.verbose:
                        print(
                            f"  [Step {self.step_count}] ⛑ FORCED ESCAPE: "
                            f"Lane {current_lane} → Lane {escape_lane} "
                            f"(brake_time={self._brake_time:.1f}s)"
                        )
                    return escape_lane
                if self.verbose:
                    print(
                        f"  [Step {self.step_count}] ⛔ ESCAPE BLOCKED: "
                        f"no viable lane from Lane {current_lane} "
                        f"(clearance={current_clearance:.1f}m, "
                        f"brake_time={self._brake_time:.1f}s)"
                    )

            self.phase = "BRAKE"
            self.emergency_brake = True
            return current_lane

        self.phase = "CRUISE"
        self.emergency_brake = False
        self._brake_time = 0.0
        self._low_speed_streak = 0
        return current_lane

    def _hazard_safe_speed(self, lane: int, current_s: float, speed: float) -> float:
        """Compute a target speed respecting stopping distance and traffic TTC."""
        track_length = self.track.get_total_length()
        available_distance = self._lane_clearance(lane, current_s, horizon=100.0)
        safe_speed = np.sqrt(2.0 * self.a_max * max(0.0, available_distance - 8.0))
        for vehicle in self.traffic_vehicles:
            if vehicle.lane != lane:
                continue
            distance = (vehicle.s - current_s) % track_length
            closing = max(speed - vehicle.speed, 0.0)
            if 0.0 < distance < 100.0 and closing > 0.1:
                safe_speed = min(safe_speed, vehicle.speed + distance / 4.0)
        return float(np.clip(safe_speed, 0.0, self.v_ref))
    
    def _plan_lane_change(
        self,
        current_lane: int,
        current_s: float,
        obstacles_ahead: List[Tuple[Union[Obstacle, 'Vehicle'], float]],
        current_speed: float = 10.0
    ) -> int:
        """Decide target lane based on obstacles and slow traffic."""
        if not obstacles_ahead:
            # No obstacle, stay in lane or return to center
            return current_lane
        
        nearest_obj, nearest_dist = obstacles_ahead[0]
        
        # Check if it's slow traffic we should overtake
        is_slow_traffic = False
        if hasattr(nearest_obj, 'speed'):  # It's a Vehicle, not a static Obstacle
            if nearest_obj.speed < current_speed - 1.0:  # Traffic is slower than us
                is_slow_traffic = True
        
        # Start lane change earlier for slow traffic (60m) vs obstacles (40m)
        trigger_dist = 60.0 if is_slow_traffic else 40.0
        
        if nearest_dist > trigger_dist:
            return current_lane
        
        # CRITICAL FIX: When stuck (very slow + close to obstacle), RESET cooldown
        # This allows continuous retry instead of waiting
        is_stuck = current_speed < 2.0 and nearest_dist < 20.0
        if is_stuck:
            self.lane_change_cooldown = 0  # Reset to allow immediate retry
        
        # Cooldown to prevent oscillation (but not when stuck)
        if self.lane_change_cooldown > 0:
            return self.target_lane
        
        # ADAPTIVE SAFETY MARGIN: When going slow or obstacle is close, accept smaller gaps
        # This prevents getting permanently stuck
        if current_speed < 2.0 or nearest_dist < 10.0:
            safety_dist = 10.0  # Ultra-desperate mode - very small gaps OK
        elif current_speed < 3.0 or nearest_dist < 15.0:
            safety_dist = 15.0  # Desperate mode - accept smaller gaps
        elif current_speed < 6.0:
            safety_dist = 20.0  # Cautious mode
        else:
            safety_dist = 28.0  # Normal mode
        
        # Find safe lanes to change into
        safe_lanes = []
        
        # When ultra-desperate, check ALL lanes
        if safety_dist <= 10.0:
            lanes_to_check = [l for l in range(self.num_lanes) if l != current_lane]
        else:
            lanes_to_check = [l for l in [current_lane - 1, current_lane + 1] if 0 <= l < self.num_lanes]
        
        for lane in lanes_to_check:
            if not self._is_lane_safe(lane, current_s, safety_dist):
                continue
            # IMPORTANT: Even in desperate mode, target lane must be safer than current lane
            # Check what's ahead in the target lane
            target_obstacles = self._find_obstacles_ahead(current_s, lane, 50.0)  # Look further
            
            if not target_obstacles:
                # Target lane is completely clear - definitely safe
                safe_lanes.append((lane, float('inf')))
            else:
                target_dist = target_obstacles[0][1]
                # Only consider if target lane obstacle is further than current + some margin
                if target_dist > nearest_dist + 10.0:  # At least 10m improvement
                    safe_lanes.append((lane, target_dist))
        
        if safe_lanes:
            # Sort by: 1) Prefer adjacent lanes, 2) Then by distance (further is better)
            safe_lanes.sort(key=lambda x: (abs(x[0] - current_lane), -x[1]))
            self.lane_change_cooldown = 3.0  # seconds, independent of frame skip
            new_lane = safe_lanes[0][0]
            target_dist = safe_lanes[0][1]
            if self.verbose:
                obj_type = "TRAFFIC" if is_slow_traffic else "OBSTACLE"
                speed_info = f", speed={nearest_obj.speed:.1f}m/s" if is_slow_traffic else ""
                mode = "ULTRA-DESPERATE" if safety_dist <= 10 else ("DESPERATE" if safety_dist < 20 else ("CAUTIOUS" if safety_dist < 25 else "NORMAL"))
                dist_info = f"clear" if target_dist == float('inf') else f"{target_dist:.0f}m clear"
                print(f"  [Step {self.step_count}] 🔄 OVERTAKE {obj_type}: Lane {current_lane} → Lane {new_lane} (at {nearest_dist:.1f}m{speed_info}) [{mode}] → {dist_info}")
            return new_lane
        
        # No safe lane - but don't just give up, use "follow at distance" mode
        # Only true emergency brake if obstacle is VERY close
        if nearest_dist < 10.0:
            if self.verbose:
                obj_type = "TRAFFIC" if is_slow_traffic else "OBSTACLE"
                print(f"  [Step {self.step_count}] 🛑 HARD BRAKE! ({obj_type} at {nearest_dist:.1f}m)")
            self.emergency_brake = True
        else:
            # Follow mode - slow down but keep checking for gaps
            if self.verbose and self.step_count % 40 == 0:  # Don't spam
                obj_type = "TRAFFIC" if is_slow_traffic else "OBSTACLE"
                print(f"  [Step {self.step_count}] 🚗 FOLLOWING {obj_type} at {nearest_dist:.1f}m (waiting for gap)")
        return current_lane
    
    def compute_action(
        self,
        state: np.ndarray,
        dt: float = 0.05,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Compute control action with obstacle avoidance.
        
        Args:
            state: [x, y, theta, v]
            dt: time step
            rng: random generator for noise
        
        Returns:
            action: [acceleration, steering]
        """
        x, y, theta, v = state
        self._ego_state = np.asarray(state, dtype=np.float64).copy()
        self._decision_dt = dt

        # Find position on track
        closest, lateral_offset = self.track.find_closest_point(x, y)
        current_s = closest.s
        current_lane = self._get_current_lane(lateral_offset)

        # A traffic vehicle can be physically close while still carrying its
        # old lane index during a transition. Escape before braking traps the
        # ego beside that vehicle at non-steerable speed.
        nearest_physical_traffic = min(
            (
                np.hypot(vehicle.x - x, vehicle.y - y), vehicle
            )
            for vehicle in self.traffic_vehicles
        ) if self.traffic_vehicles else (float("inf"), None)
        if nearest_physical_traffic[0] < max(10.0, v + 4.0):
            escape_lane = self._select_physical_escape(current_lane, current_s, v)
            if escape_lane is not None:
                self.target_lane = escape_lane
                self.recovery_target_lane = escape_lane
                self.phase = "COMMIT"
                self.emergency_brake = False
        
        # Detect obstacles ahead in current lane
        obstacles_ahead = self._find_obstacles_ahead(
            current_s, current_lane, self.obstacle_detect_dist
        )
        
        # Also check target lane if different - but be smarter about aborting
        if self.target_lane != current_lane:
            target_obstacles = self._find_obstacles_ahead(
                current_s, self.target_lane, self.obstacle_detect_dist
            )
            
            # Check obstacle distance in CURRENT lane
            current_lane_dist = obstacles_ahead[0][1] if obstacles_ahead else float('inf')
            
            # Only abort if: target is blocked AND current lane is safer
            # Don't abort if we're desperate (current obstacle very close)
            if target_obstacles and target_obstacles[0][1] < 15:
                if current_lane_dist > target_obstacles[0][1] + 5:  # Current lane is safer
                    if self.verbose:
                        print(f"  [Step {self.step_count}] ❌ ABORT: Target lane {self.target_lane} blocked at {target_obstacles[0][1]:.1f}m")
                    self.target_lane = current_lane
                # If current lane is worse or same, COMMIT to lane change anyway
        
        # Plan and maintain a recovery maneuver over a 100m horizon.
        self.emergency_brake = False
        self._decision_dt = dt
        self._update_recovery_state(current_lane, current_s, v)
        
        self.lane_change_cooldown = max(0.0, self.lane_change_cooldown - dt)
        
        # Target lane centerline
        target_offset = self._get_lane_center_offset(self.target_lane)
        
        # Pure Pursuit to target lane
        lookahead_s = (current_s + self.lookahead_dist) % self.track.get_total_length()
        lookahead_pt = self.track.get_centerline_point(lookahead_s)
        
        # Offset lookahead point to target lane
        nx = -np.sin(lookahead_pt.heading)
        ny = np.cos(lookahead_pt.heading)
        target_x = lookahead_pt.x + target_offset * nx
        target_y = lookahead_pt.y + target_offset * ny
        
        # Pure pursuit steering
        dx = target_x - x
        dy = target_y - y
        
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        local_x = cos_t * dx + sin_t * dy
        local_y = -sin_t * dx + cos_t * dy
        
        ld = np.sqrt(dx**2 + dy**2)
        if ld > 0.1:
            alpha = np.arctan2(local_y, local_x)
            delta = np.arctan2(2 * self.L * np.sin(alpha), ld)
        else:
            delta = 0.0
        
        delta = np.clip(delta, -self.delta_max, self.delta_max)
        
        # Speed control with stopping-distance and TTC limits.
        target_speed = self.v_ref
        
        # KEY FIX: Only consider obstacles in TARGET lane for braking decisions
        # If we're changing lanes to avoid an obstacle, don't brake for the obstacle we're avoiding
        target_lane_obstacles = self._find_obstacles_ahead(current_s, self.target_lane, 50)
        
        if target_lane_obstacles:
            nearest_dist = target_lane_obstacles[0][1]
            # Progressive braking based on obstacle in TARGET lane
            if nearest_dist < 40:
                slow_factor = max(0.3, (nearest_dist - 5) / 35)
                target_speed = self.v_ref * slow_factor

        target_speed = min(
            target_speed,
            self._hazard_safe_speed(self.target_lane, current_s, v),
        )
        
        # If we're actively changing lanes AND target lane is clear, maintain speed!
        if self.target_lane != current_lane and not target_lane_obstacles:
            target_speed = self.v_ref * 0.8  # Slightly slower for safety during lane change
        
        # Emergency brake ONLY if all lanes blocked (emergency_brake flag)
        if self.emergency_brake:
            target_speed = min(target_speed, 2.0)
        
        # PID speed control
        speed_error = target_speed - v
        
        p_term = self.kp * speed_error
        self._integral_error = np.clip(self._integral_error + speed_error * dt, -10, 10)
        i_term = self.ki * self._integral_error
        d_term = self.kd * (speed_error - self._prev_speed_error) / dt
        self._prev_speed_error = speed_error
        
        a = np.clip(p_term + i_term + d_term, -self.a_max, self.a_max)
        
        # Emergency braking ONLY if obstacle very close in TARGET lane (not current lane we're leaving)
        if target_lane_obstacles and target_lane_obstacles[0][1] < 6:
            a = -self.a_max
        
        # Hard brake only if truly stuck (all lanes blocked)
        if self.emergency_brake and obstacles_ahead and obstacles_ahead[0][1] < 10:
            a = -self.a_max
        
        # Add noise
        if self.action_noise > 0 and rng is not None:
            a += rng.normal(0, self.action_noise * self.a_max)
            delta += rng.normal(0, self.action_noise * self.delta_max)
            a = np.clip(a, -self.a_max, self.a_max)
            delta = np.clip(delta, -self.delta_max, self.delta_max)
        
        # Verbose logging
        if self.verbose:
            # Always log lane changes and emergency situations
            # Every 20 steps, show status summary
            if self.step_count % 20 == 0:
                status_parts = [
                    f"\n[Step {self.step_count}] phase={self.phase} "
                    f"v={v:.1f}→{target_speed:.1f} m/s, Lane {current_lane}"
                ]
                if self.target_lane != current_lane:
                    status_parts.append(f"→{self.target_lane}")
                
                if obstacles_ahead:
                    obj = obstacles_ahead[0][0]
                    dist = obstacles_ahead[0][1]
                    if hasattr(obj, 'speed'):
                        status_parts.append(f" | TRAFFIC ahead: {dist:.0f}m @ {obj.speed:.1f}m/s")
                    else:
                        status_parts.append(f" | OBSTACLE ahead: {dist:.0f}m")
                
                if self.emergency_brake:
                    status_parts.append(" | 🛑 BRAKING")
                elif v < 3.0 and target_speed > v:
                    status_parts.append(" | ⚠️ STUCK")
                
                print("".join(status_parts))
        
        self.step_count += 1
        
        # Store the hierarchical target for BC-SAC demo collection
        self.last_target_offset = target_offset
        self.last_target_speed = target_speed
        
        return np.array([a, delta], dtype=np.float32)
