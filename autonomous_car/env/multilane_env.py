"""
MultiLaneEnv: Enhanced environment with multiple lanes and obstacles.

Features:
- Multiple lanes with lane change capability
- Static obstacles the car must avoid
- Enhanced observations for lane/obstacle awareness
- Improved visualization
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from autonomous_car.env.track import Track, OvalTrack


@dataclass
class Obstacle:
    """Static obstacle on the track."""
    s: float  # arc length position along track
    lane: int  # which lane (0 = rightmost)
    length: float = 4.0  # obstacle length in meters
    width: float = 2.0  # obstacle width in meters
    
    # Computed world position (set by environment)
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0


@dataclass
class Vehicle:
    """Moving traffic vehicle on the track."""
    s: float  # arc length position along track
    lane: int  # which lane (0 = rightmost)
    speed: float = 8.0  # current speed in m/s
    target_speed: float = 8.0  # desired cruise speed
    length: float = 4.5  # vehicle length in meters
    width: float = 1.8  # vehicle width in meters
    
    # Computed world position (updated by environment)
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    
    # Simple behavior parameters
    max_accel: float = 2.0  # max acceleration m/s²
    max_brake: float = 4.0  # max braking m/s²
    safe_distance: float = 15.0  # min following distance
    
    # Lane change behavior
    lane_change_tendency: int = 0  # -1=prefer left, 0=stay, 1=prefer right
    lane_change_cooldown: float = 0.0  # time until next lane change allowed
    lane_change_interval: float = 5.0  # min time between lane changes
    lateral_offset: float = 0.0  # current offset from lane center during transition
    target_lateral_offset: float = 0.0  # target lateral offset (for smooth transitions)


class MultiLaneEnv(gym.Env):
    """
    Multi-lane autonomous driving environment with obstacles and traffic.
    
    Features:
    - N lanes (configurable)
    - Static obstacles placed on lanes
    - Moving traffic vehicles with car-following behavior
    - Lane change actions or continuous lateral control
    - Enhanced observation space with obstacle detection
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    
    def __init__(
        self,
        track: Optional[Track] = None,
        num_lanes: int = 3,
        lane_width: float = 3.5,
        dt: float = 0.05,
        L: float = 2.5,  # wheelbase
        v_max: float = 20.0,
        v_ref: float = 10.0,
        a_max: float = 3.0,
        delta_max: float = 0.5,
        max_episode_steps: int = 2000,
        # Obstacle settings
        num_obstacles: int = 5,
        obstacle_seed: Optional[int] = None,
        # Traffic settings
        num_traffic_vehicles: int = 0,  # Number of moving traffic vehicles
        traffic_speed_range: Tuple[float, float] = (6.0, 9.0),  # Speed range for traffic
        # Sensor settings
        num_distance_sensors: int = 5,  # front-facing distance sensors
        sensor_range: float = 50.0,
        # Reward weights
        w_lane: float = 1.0,
        w_head: float = 1.0,
        w_speed: float = 0.5,
        w_progress: float = 0.5,
        w_comfort: float = 0.1,
        obstacle_penalty: float = 200.0,
        offroad_penalty: float = 100.0,
        # Decision layer integration
        use_decision_layer: bool = False,
        # Action repetition (frame-skip for temporal persistence)
        action_repeat: int = 10,  # Hold target constant for N steps
        # Kinematic speed clamp (safety override)
        speed_clamp: bool = False,  # Clamp target_speed to stay within braking distance
        clamp_margin: float = 10.0,  # Reaction margin (m) beyond kinematic stop distance
        shield_ttc_horizon: float = 4.0,
        completion_bonus: float = 200.0,
        # Longitudinal gating margin (seed-42 forensics): entities whose
        # center lies more than this far BEHIND the ego along its heading are
        # excluded from forward-hazard perception (speed clamp / emergency
        # brake). Collision DETECTION is untouched.
        hazard_behind_margin: float = 1.0,
        # Centralized safety calibration (Bottleneck 3). Defaults reproduce
        # the empirically validated values; changing them requires re-running
        # the brake-deficit health check.
        collision_radius: float = 2.2,
        creep_buffer: float = 0.8,
        creep_gain: float = 2.0,
        brake_tail_gain: float = 0.5,
        stall_speed_threshold: float = 0.5,
        stall_timeout: float = 8.0,
        stall_progress_window: float = 0.5,
        # Rendering
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        # Track setup
        self.base_track = track if track is not None else OvalTrack()
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.total_road_width = num_lanes * lane_width
        
        # Vehicle parameters
        self.dt = dt
        self.L = L
        self.v_max = v_max
        self.v_ref = v_ref
        self.a_max = a_max
        self.delta_max = delta_max
        self.max_episode_steps = max_episode_steps
        
        # Obstacle settings
        self.num_obstacles = num_obstacles
        self.obstacle_seed = obstacle_seed
        self.obstacles: List[Obstacle] = []
        
        # Traffic settings
        self.num_traffic_vehicles = num_traffic_vehicles
        self.traffic_speed_range = traffic_speed_range
        self.traffic_vehicles: List[Vehicle] = []
        
        # Sensor settings
        self.num_distance_sensors = num_distance_sensors
        self.sensor_range = sensor_range
        self.sensor_angles = np.linspace(-np.pi/4, np.pi/4, num_distance_sensors)
        
        # Action repetition (frame-skip)
        self.action_repeat = action_repeat
        self._action_repeat_count = 0
        self._stored_target = np.zeros(2, dtype=np.float32)
        
        # Kinematic speed clamp (safety override)
        self.speed_clamp = speed_clamp
        self.clamp_margin = clamp_margin
        self._clamp_active = False  # scratchpad for diagnostics
        self._clamp_hazard_dist = float('inf')  # hazard distance at clamp decision
        self._shield_requested_speed = 0.0
        self._shield_safe_speed = self.v_max
        self._shield_ttc = float('inf')
        self._shield_emergency = False
        self._shield_hazard_lane = num_lanes // 2
        self._nearest_traffic_distance = float('inf')
        self._nearest_traffic_vehicle = None
        self._planner_phase = "CRUISE"
        self._planner_target_lane = None
        self._lane_shield_active = False
        self._lane_shield_target_lane = None
        self._lane_shield_reason = ""
        self._last_safety_decision: Dict[str, Any] = {}
        self.completion_bonus = completion_bonus
        self.shield_ttc_horizon = shield_ttc_horizon
        self.hazard_behind_margin = hazard_behind_margin
        self.collision_radius = collision_radius
        self.creep_buffer = creep_buffer
        self.creep_gain = creep_gain
        self.brake_tail_gain = brake_tail_gain
        # Low-speed attitude protection (ep10 forensics): below this speed,
        # and outside committed transitions, steering switches from pure
        # pursuit (fixed lookahead -> drift/yaw at crawl speeds) to
        # tangent-aligned heading hold.
        self.tangent_hold_speed = 2.0
        self.tangent_hold_min_err = float(np.deg2rad(8.0))
        self.heading_hold_gain = 1.5
        # Hardened attitude-hold parameters (QA review FM-1..FM-3):
        # hysteresis release thresholds, transition-progress floor, and the
        # speed at which steering authority reaches full ramp.
        self.attitude_hold_release_err = float(np.deg2rad(5.0))
        self.attitude_stopped_release_speed = 0.45
        self.attitude_transition_motion_eps = 1e-3
        self.attitude_authority_speed = 0.5
        self._hold_engaged = False
        self._stopped_latch = False
        self._transition_ref_s = 0.0
        self._transition_ref_lat = 0.0
        self._transition_motion = 0.0
        self._attitude_frozen_steps = 0
        self._attitude_deadlock = False
        self.stall_speed_threshold = stall_speed_threshold
        self.stall_timeout = stall_timeout
        self.stall_progress_window = stall_progress_window
        self.max_brake_deficit = 0.0
        # Collision-check recorder (diagnostics only, default OFF). When a
        # list is assigned, every swept-pair check appends a context-tagged
        # record: 'real' for actual frame collision tests, 'admission' for
        # traffic-transition probes. Trajectory-validation probes live in
        # the expert's own box test and are counted, not recorded.
        self.collision_recorder = None
        self._rec_context = "real"

        
        # Decision layer integration
        self.use_decision_layer = use_decision_layer
        self.decision_target_lane: int = num_lanes // 2  # Target from decision layer
        self.decision_desired_speed: float = v_ref  # Speed target from decision layer
        
        # Reward weights
        self.w_lane = w_lane
        self.w_head = w_head
        self.w_speed = w_speed
        self.w_progress = w_progress
        self.w_comfort = w_comfort
        self.obstacle_penalty = obstacle_penalty
        self.offroad_penalty = offroad_penalty
        
        # Action space: [target_lane_offset, target_speed]
        # The env internally uses a tracking controller (Pure Pursuit + PD)
        # to convert these targets into [accel, steer] for the bicycle model.
        # target_lane_offset is constrained to ±lane_width so that 1σ Gaussian
        # exploration stays within the road (max target = lane center of far lane).
        self.action_space = spaces.Box(
            low=np.array([-self.lane_width, 0.0], dtype=np.float32),
            high=np.array([self.lane_width, self.v_max], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space (28 dimensions total):
        # Base features (7):
        # - lane_offset (from road center, normalized)
        # - heading_error
        # - speed (normalized)
        # - yaw_rate
        # - lookahead_curvature
        # - prev_accel, prev_steer
        # Distance sensors (5):
        # - distance_sensors (N sensors)
        # Current lane info (4):
        # - current_lane_onehot (N lanes) - one-hot encoding of current lane
        # - lane_deviation (1) - deviation from lane center
        # Per-lane features (12 = 4 x N lanes):
        # - per_lane_obstacle_dist (N lanes) - distance to nearest obstacle in each lane
        # - per_lane_traffic_dist (N lanes) - distance to nearest traffic in each lane
        # - per_lane_relative_velocity (N lanes) - V8: CLOSING RATE (ego_speed - traffic_speed)
        #   Positive = gap closing (urgent), Negative = gap opening (safe), 0 = no traffic
        # - per_lane_blocked (N lanes) - 1 if lane blocked within safety distance, 0 otherwise
        base_obs_dim = 7 + num_distance_sensors  # legacy features (12)
        current_lane_features = num_lanes + 1  # one-hot + deviation (4)
        per_lane_features = 4 * num_lanes  # 4 features per lane (12)
        prev_target_features = 2  # prev_target_lane_offset_norm, prev_target_speed_norm
        decision_features = 2 if use_decision_layer else 0  # [target_lane, desired_speed]
        obs_dim = base_obs_dim + current_lane_features + per_lane_features + prev_target_features + decision_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # State
        self.state: np.ndarray = np.zeros(4)
        self.prev_action: np.ndarray = np.zeros(2)
        self._prev_target: np.ndarray = np.zeros(2)  # [norm_target_lane_offset, norm_target_speed]
        self.step_count: int = 0
        self.prev_s: float = 0.0
        self.current_lane: int = num_lanes // 2  # start in middle lane
        
        # Rendering
        self.render_mode = render_mode
        self._fig = None
        self._ax = None
        
    def _generate_obstacles(self, rng: np.random.Generator) -> List[Obstacle]:
        """Generate obstacles in alternating pattern (2-1-2-1) on different lanes.
        
        Creates a fair challenge where the car always has at least one clear lane.
        """
        obstacles = []
        track_length = self.base_track.get_total_length()
        
        # Spacing between obstacle groups (adaptive based on number requested)
        # For more obstacles, reduce spacing to fit them all on the track
        if self.num_obstacles > 20:
            group_spacing = max(25.0, (track_length - 60) / (self.num_obstacles / 1.5))
        else:
            group_spacing = 50.0  # meters between groups
        
        # Available lanes shuffled
        all_lanes = list(range(self.num_lanes))
        
        # Place obstacles in groups along the track
        current_s = 30.0  # start position
        obstacle_count = 0
        group_size_pattern = [2, 1, 2, 1]  # alternating pattern
        pattern_idx = 0
        
        while obstacle_count < self.num_obstacles and current_s < track_length - 30:
            # How many obstacles in this group
            group_size = group_size_pattern[pattern_idx % len(group_size_pattern)]
            group_size = min(group_size, self.num_obstacles - obstacle_count)
            
            # IMPORTANT: Never block all lanes - leave at least one clear
            max_in_group = min(group_size, self.num_lanes - 1)
            
            # Shuffle lanes and pick which ones get obstacles
            rng.shuffle(all_lanes)
            lanes_for_group = all_lanes[:max_in_group]
            
            for lane in lanes_for_group:
                obs = Obstacle(s=current_s, lane=lane)
                self._update_obstacle_position(obs)
                obstacles.append(obs)
                obstacle_count += 1
                if obstacle_count >= self.num_obstacles:
                    break
            
            current_s += group_spacing
            pattern_idx += 1
        
        return obstacles
    
    def _generate_traffic(self, rng: np.random.Generator) -> List[Vehicle]:
        """Generate traffic vehicles spread across the track.
        
        Traffic vehicles drive at varied speeds and are distributed
        across ALL lanes to create realistic traffic scenarios.
        Some vehicles will change lanes periodically.
        """
        vehicles = []
        track_length = self.base_track.get_total_length()
        
        if self.num_traffic_vehicles == 0:
            return vehicles
        
        # Spread vehicles evenly around the track, avoiding the start area
        spacing = (track_length - 100) / max(1, self.num_traffic_vehicles)
        
        for i in range(self.num_traffic_vehicles):
            # Position along track (start after ego vehicle spawn area)
            s = 50.0 + i * spacing
            s = s % track_length  # Wrap around
            
            # Distribute across lanes more evenly (round-robin with some randomness)
            base_lane = i % self.num_lanes
            lane_offset = rng.choice([-1, 0, 0, 1])  # Small chance to shift
            lane = max(0, min(self.num_lanes - 1, base_lane + lane_offset))
            
            # Random target speed within range
            min_speed, max_speed = self.traffic_speed_range
            target_speed = rng.uniform(min_speed, max_speed)
            
            # Assign lane change tendencies - 50% of vehicles change lanes, 50% stay
            # Decide if this vehicle changes lanes at all
            changes_lanes = rng.choice([False, True])  # 50% chance
            
            if not changes_lanes:
                lane_change_tendency = 0  # Stay in current lane
            else:
                # Vehicle changes lanes - direction based on current lane
                # Left lane (lane=num_lanes-1): can only go right
                # Right lane (lane=0): can only go left  
                # Middle lanes: can go either way
                if lane == 0:
                    lane_change_tendency = 1  # Must go left (toward center)
                elif lane == self.num_lanes - 1:
                    lane_change_tendency = -1  # Must go right (toward center)
                else:
                    lane_change_tendency = rng.choice([-1, 1])  # Equal chance either direction
            
            # Vary the lane change intervals
            lane_change_interval = rng.uniform(3.0, 8.0)
            
            vehicle = Vehicle(
                s=s,
                lane=lane,
                speed=target_speed,  # Start at target speed
                target_speed=target_speed,
                lane_change_tendency=lane_change_tendency,
                lane_change_interval=lane_change_interval,
                lane_change_cooldown=rng.uniform(0, lane_change_interval),  # Stagger initial changes
            )
            self._update_vehicle_position(vehicle)
            vehicles.append(vehicle)
        
        return vehicles
    
    def _update_vehicle_position(self, veh: Vehicle):
        """Update vehicle world position from track position."""
        track_length = self.base_track.get_total_length()
        veh.s = veh.s % track_length  # Wrap around
        
        pt = self.base_track.get_centerline_point(veh.s)
        
        # Base lane offset plus any lateral transition offset
        lane_offset = (veh.lane - (self.num_lanes - 1) / 2) * self.lane_width
        total_lateral = lane_offset + veh.lateral_offset
        
        # Normal direction (perpendicular to track)
        nx = -np.sin(pt.heading)
        ny = np.cos(pt.heading)
        
        veh.x = pt.x + total_lateral * nx
        veh.y = pt.y + total_lateral * ny
        veh.heading = pt.heading
    
    def _update_traffic(self, ego_s: Optional[float] = None,
                        ego_lane: Optional[int] = None):
        """Update all traffic vehicle positions using car-following model with lane changes."""
        track_length = self.base_track.get_total_length()
        ego_s = self.prev_s if ego_s is None else ego_s
        ego_lane = self.current_lane if ego_lane is None else ego_lane
        
        for veh in self.traffic_vehicles:
            # Update lane change cooldown
            veh.lane_change_cooldown = max(0, veh.lane_change_cooldown - self.dt)
            
            # Find closest vehicle ahead in same lane (IGNORE static obstacles - traffic passes through them)
            min_dist_ahead = float('inf')
            
            # Check other traffic vehicles (not obstacles - ghost mode for traffic)
            for other in self.traffic_vehicles:
                if other is veh:
                    continue
                if other.lane == veh.lane:
                    dist = (other.s - veh.s) % track_length
                    if 0 < dist < min_dist_ahead:
                        min_dist_ahead = dist
            
            # Check ego vehicle
            if ego_lane == veh.lane:
                dist = (ego_s - veh.s) % track_length
                if 0 < dist < min_dist_ahead:
                    min_dist_ahead = dist
            
            # ===== LANE CHANGE LOGIC =====
            # Attempt lane change if:
            # 1) has tendency AND cooldown expired, OR
            # 2) STUCK (speed near zero with obstacle ahead) - escape mode!
            is_stuck = veh.speed < 1.0 and min_dist_ahead < veh.safe_distance * 1.5
            should_try_lane_change = (veh.lane_change_tendency != 0 and veh.lane_change_cooldown <= 0) or is_stuck
            
            if should_try_lane_change:
                # If stuck, try both directions; otherwise follow tendency
                if is_stuck:
                    possible_lanes = [veh.lane - 1, veh.lane + 1]
                else:
                    possible_lanes = [veh.lane + veh.lane_change_tendency]
                
                for target_lane in possible_lanes:
                    # Check if target lane is valid
                    if not (0 <= target_lane < self.num_lanes):
                        continue
                    
                    # Check if lane change is safe (no vehicle close in target lane)
                    # NOTE: Traffic ignores obstacles (ghost mode) - only checks other vehicles
                    safe_to_change = True
                    
                    for other in self.traffic_vehicles:
                        if other is veh:
                            continue
                        if self._vehicle_swept_occupies_lane(other, target_lane):
                            dist = abs((other.s - veh.s + track_length/2) % track_length - track_length/2)
                            if dist < 15:  # Need 15m gap for other vehicles
                                safe_to_change = False
                                break
                    
                    if not safe_to_change:
                        continue
                    
                    # Check ego vehicle in target lane
                    if ego_lane == target_lane:
                        dist = abs((ego_s - veh.s + track_length/2) % track_length - track_length/2)
                        world_dist = np.hypot(veh.x - self.state[0], veh.y - self.state[1])
                        if dist < 25 or world_dist < 25:  # Include curved-track proximity
                            safe_to_change = False

                    if safe_to_change and self._traffic_transition_conflicts_with_ego(
                        veh, target_lane, ego_lane
                    ):
                        safe_to_change = False
                    
                    if safe_to_change:
                        # Execute lane change
                        old_lane = veh.lane
                        veh.lane = target_lane
                        veh.lane_change_cooldown = veh.lane_change_interval
                        
                        # Set lateral offset to smoothly transition
                        veh.lateral_offset = (old_lane - target_lane) * self.lane_width
                        veh.target_lateral_offset = 0.0
                        break  # Successfully changed lane
            
            # Cancel an active transition before its next swept pose can enter
            # the ego footprint, preserving position while reversing toward
            # the source lane.
            if abs(veh.lateral_offset) > 0.01:
                predicted_offset = self._vehicle_predicted_lateral_offset(veh)
                predicted_pose = self._vehicle_pose(
                    veh,
                    veh.lane,
                    predicted_offset,
                    veh.s + veh.speed * self.dt,
                )
                ego_pose = (self.state[0], self.state[1], self.state[2])
                if self._swept_pair_collision(
                    (veh.x, veh.y, veh.heading),
                    predicted_pose,
                    (veh.length / 2.0, veh.width / 2.0),
                    ego_pose,
                    ego_pose,
                    (2.25, 0.9),
                ):
                    current_center = (
                        self._get_lane_center_offset(veh.lane)
                        + veh.lateral_offset
                    )
                    source_lane = veh.lane + (1 if veh.lateral_offset > 0 else -1)
                    source_lane = max(0, min(self.num_lanes - 1, source_lane))
                    veh.lane = source_lane
                    veh.lateral_offset = (
                        current_center - self._get_lane_center_offset(source_lane)
                    )
                    veh.lane_change_cooldown = veh.lane_change_interval

            # Smooth lateral transition (move towards lane center)
            if abs(veh.lateral_offset) > 0.01:
                lateral_speed = 1.5  # m/s lateral movement
                if veh.lateral_offset > 0:
                    veh.lateral_offset = max(0, veh.lateral_offset - lateral_speed * self.dt)
                else:
                    veh.lateral_offset = min(0, veh.lateral_offset + lateral_speed * self.dt)
            else:
                veh.lateral_offset = 0.0
            
            # ===== SPEED CONTROL =====
            # Simple car-following: accelerate/brake based on distance
            if min_dist_ahead < veh.safe_distance:
                # Too close - brake
                accel = -veh.max_brake * (1 - min_dist_ahead / veh.safe_distance)
            elif min_dist_ahead < veh.safe_distance * 2:
                # Getting close - coast or gentle brake
                accel = -veh.max_brake * 0.3
            else:
                # Clear ahead - accelerate to target speed
                speed_error = veh.target_speed - veh.speed
                accel = np.clip(speed_error * 2.0, -veh.max_brake, veh.max_accel)
            
            # Update speed and position
            veh.speed = np.clip(veh.speed + accel * self.dt, 0.0, self.v_max)
            veh.s = (veh.s + veh.speed * self.dt) % track_length
            self._update_vehicle_position(veh)
    
    def _update_obstacle_position(self, obs: Obstacle):
        """Update obstacle world position from track position."""
        pt = self.base_track.get_centerline_point(obs.s)
        
        # Lane offset from road center
        road_center_offset = 0.0  # track centerline is road center
        lane_offset = (obs.lane - (self.num_lanes - 1) / 2) * self.lane_width
        
        # Normal direction (perpendicular to track)
        nx = -np.sin(pt.heading)
        ny = np.cos(pt.heading)
        
        obs.x = pt.x + lane_offset * nx
        obs.y = pt.y + lane_offset * ny
        obs.heading = pt.heading
    
    def _get_lane_center_offset(self, lane: int) -> float:
        """Get lateral offset for lane center from road center."""
        return (lane - (self.num_lanes - 1) / 2) * self.lane_width
    
    def _get_current_lane(self, lateral_offset: float) -> int:
        """Determine which lane the car is in based on lateral offset."""
        # lateral_offset is from road center
        for lane in range(self.num_lanes):
            lane_center = self._get_lane_center_offset(lane)
            if abs(lateral_offset - lane_center) < self.lane_width / 2:
                return lane
        # Default to nearest lane
        return max(0, min(self.num_lanes - 1, 
                          int((lateral_offset / self.lane_width) + (self.num_lanes - 1) / 2 + 0.5)))

    def _vehicle_occupies_lane(self, vehicle: Vehicle, lane: int) -> bool:
        """Use continuous footprint overlap instead of vehicle.lane alone."""
        vehicle_center = (
            self._get_lane_center_offset(vehicle.lane) + vehicle.lateral_offset
        )
        lane_center = self._get_lane_center_offset(lane)
        return abs(vehicle_center - lane_center) <= (
            self.lane_width / 2.0 + vehicle.width / 2.0
        )

    def _obstacle_occupies_lane(self, obstacle: Obstacle, lane: int) -> bool:
        """Treat a static obstacle as occupying every lane its footprint touches.

        Discrete obs.lane hides side-swipe threats: an obstacle registered in
        an adjacent lane can still be clipped when the ego drifts toward the
        lane boundary. Hazard features must use the same continuous footprint
        model as the swept collision detector.
        """
        obstacle_center = self._get_lane_center_offset(obstacle.lane)
        lane_center = self._get_lane_center_offset(lane)
        return abs(obstacle_center - lane_center) <= (
            self.lane_width / 2.0 + obstacle.width / 2.0
        )

    def _vehicle_predicted_lateral_offset(self, vehicle: Vehicle) -> float:
        """Predict one simulator frame of an active lateral transition."""
        if abs(vehicle.lateral_offset) <= 0.01:
            return 0.0
        step = 1.5 * self.dt
        if vehicle.lateral_offset > 0.0:
            return max(0.0, vehicle.lateral_offset - step)
        return min(0.0, vehicle.lateral_offset + step)

    def _vehicle_swept_occupies_lane(self, vehicle: Vehicle, lane: int) -> bool:
        """Reserve lanes intersected by the current-to-next footprint sweep."""
        start = self._get_lane_center_offset(vehicle.lane) + vehicle.lateral_offset
        end = self._get_lane_center_offset(vehicle.lane) + (
            self._vehicle_predicted_lateral_offset(vehicle)
        )
        low, high = sorted((start, end))
        lane_center = self._get_lane_center_offset(lane)
        half_lane = self.lane_width / 2.0
        half_vehicle = vehicle.width / 2.0
        return high + half_vehicle >= lane_center - half_lane and low - half_vehicle <= lane_center + half_lane

    def _vehicle_pose(self, vehicle: Vehicle, lane: int, lateral_offset: float,
                      s: Optional[float] = None) -> Tuple[float, float, float]:
        """Return a traffic pose without mutating the vehicle."""
        track_length = self.base_track.get_total_length()
        point = self.base_track.get_centerline_point(
            vehicle.s if s is None else s % track_length
        )
        offset = self._get_lane_center_offset(lane) + lateral_offset
        return (
            point.x - np.sin(point.heading) * offset,
            point.y + np.cos(point.heading) * offset,
            point.heading,
        )

    def _traffic_transition_conflicts_with_ego(
        self, vehicle: Vehicle, target_lane: int, ego_lane: int
    ) -> bool:
        """Reject a traffic transition whose swept box approaches the ego."""
        start_pose = (vehicle.x, vehicle.y, vehicle.heading)
        start_offset = (
            (vehicle.lane - target_lane) * self.lane_width
            if target_lane != vehicle.lane else vehicle.lateral_offset
        )
        probe = Vehicle(s=vehicle.s, lane=target_lane, lateral_offset=start_offset)
        end_pose = self._vehicle_pose(
            vehicle,
            target_lane,
            self._vehicle_predicted_lateral_offset(probe),
            vehicle.s + vehicle.speed * self.dt,
        )
        ego_pose = (self.state[0], self.state[1], self.state[2])
        self._rec_context = "admission"
        conflict = self._swept_pair_collision(
            start_pose, end_pose, (vehicle.length / 2.0, vehicle.width / 2.0),
            ego_pose, ego_pose, (2.25, 0.9),
        )
        self._rec_context = "real"
        if conflict:
            return True
        # Continuous-lateral conflict test: a discrete ego lane index lags the
        # physical position whenever the ego is itself mid-transition, which
        # let traffic transitions be admitted into space the ego still
        # occupies. Compare the traffic sweep directly against the ego's
        # continuous lateral coordinate.
        _, ego_lateral = self.base_track.find_closest_point(
            float(self.state[0]), float(self.state[1])
        )
        half_width_sum = probe.width / 2.0 + 0.9 + 0.3
        lateral_conflict = any(
            abs(
                self._get_lane_center_offset(lane)
                - ego_lateral
            ) <= self.lane_width / 2.0 + half_width_sum
            for lane in range(self.num_lanes)
            if self._vehicle_swept_occupies_lane(probe, lane)
        )
        world_dist = np.hypot(
            vehicle.x - self.state[0], vehicle.y - self.state[1]
        )
        # A stopped ego cannot dodge: forbid transitions anywhere near it.
        stopped_ego_bubble = float(self.state[3]) < 0.5 and world_dist < 15.0
        return (lateral_conflict and world_dist < 25.0) or stopped_ego_bubble
    
    def _get_distance_readings(self, x: float, y: float, theta: float) -> np.ndarray:
        """Get distance sensor readings (raycast to obstacles and traffic vehicles)."""
        readings = np.full(self.num_distance_sensors, self.sensor_range)
        
        for i, angle in enumerate(self.sensor_angles):
            ray_angle = theta + angle
            ray_dx = np.cos(ray_angle)
            ray_dy = np.sin(ray_angle)
            
            # Check static obstacles
            for obs in self.obstacles:
                obs_radius = max(obs.length, obs.width) / 2
                to_obs_x = obs.x - x
                to_obs_y = obs.y - y
                proj = to_obs_x * ray_dx + to_obs_y * ray_dy
                
                if proj > 0:  # obstacle is in front
                    perp = abs(-to_obs_x * ray_dy + to_obs_y * ray_dx)
                    if perp < obs_radius:
                        dist = proj - np.sqrt(max(0, obs_radius**2 - perp**2))
                        readings[i] = min(readings[i], max(0, dist))
            
            # Check traffic vehicles
            for veh in self.traffic_vehicles:
                veh_radius = max(veh.length, veh.width) / 2
                to_veh_x = veh.x - x
                to_veh_y = veh.y - y
                proj = to_veh_x * ray_dx + to_veh_y * ray_dy
                
                if proj > 0:  # vehicle is in front
                    perp = abs(-to_veh_x * ray_dy + to_veh_y * ray_dx)
                    if perp < veh_radius:
                        dist = proj - np.sqrt(max(0, veh_radius**2 - perp**2))
                        readings[i] = min(readings[i], max(0, dist))
        
        return readings
    
    def _get_per_lane_info(self, current_s: float, ego_speed: float, lookahead: float = 60.0) -> Dict[str, np.ndarray]:
        """Get per-lane obstacle and traffic information for enhanced observations.
        
        V8 Enhancement: Returns RELATIVE velocity instead of absolute traffic speed.
        This tells the agent how fast the gap is closing, which is more actionable.
        
        Returns:
            Dict with:
            - obstacle_dist: [N lanes] distance to nearest obstacle in each lane
            - traffic_dist: [N lanes] distance to nearest traffic vehicle in each lane
            - relative_velocity: [N lanes] closing rate (ego_speed - traffic_speed)
              Positive = closing gap, Negative = gap opening, 0 = no traffic
            - lane_blocked: [N lanes] 1 if blocked within 25m, 0 otherwise
        """
        track_length = self.base_track.get_total_length()
        
        obstacle_dist = np.full(self.num_lanes, lookahead, dtype=np.float32)
        traffic_dist = np.full(self.num_lanes, lookahead, dtype=np.float32)
        traffic_speed = np.zeros(self.num_lanes, dtype=np.float32)  # Raw speed, used internally
        relative_velocity = np.zeros(self.num_lanes, dtype=np.float32)  # V8: closing rate
        lane_blocked = np.zeros(self.num_lanes, dtype=np.float32)
        
        # Find nearest obstacle in each lane (continuous footprint occupancy)
        for obs in self.obstacles:
            # Distance ahead (handling track wrap)
            dist = (obs.s - current_s) % track_length
            if dist > track_length / 2:  # Behind us (wrapped)
                continue

            for lane in range(self.num_lanes):
                if not self._obstacle_occupies_lane(obs, lane):
                    continue
                if dist < obstacle_dist[lane]:
                    obstacle_dist[lane] = dist
        
        # Find nearest traffic vehicle in each lane
        for veh in self.traffic_vehicles:
            dist = (veh.s - current_s) % track_length
            world_gap = np.hypot(
                veh.x - self.state[0], veh.y - self.state[1]
            ) - (2.25 + veh.length / 2.0 + 0.3)
            if dist > track_length / 2 and world_gap > 30.0:
                continue

            for lane in range(self.num_lanes):
                if not self._vehicle_swept_occupies_lane(veh, lane):
                    continue
                effective_dist = min(dist, max(0.0, world_gap))
                if effective_dist < traffic_dist[lane]:
                    traffic_dist[lane] = effective_dist
                    traffic_speed[lane] = veh.speed
                    relative_velocity[lane] = ego_speed - veh.speed
        
        # Determine if lane is blocked (obstacle or slow/stopped traffic within 25m)
        safety_dist = 25.0
        for lane in range(self.num_lanes):
            if obstacle_dist[lane] < safety_dist:
                lane_blocked[lane] = 1.0
            elif traffic_dist[lane] < safety_dist and traffic_speed[lane] < 3.0:
                lane_blocked[lane] = 1.0
        
        return {
            'obstacle_dist': obstacle_dist,
            'traffic_dist': traffic_dist,
            'traffic_speed': traffic_speed,  # Still available for lane_blocked logic
            'relative_velocity': relative_velocity,  # V8: closing rate for observations
            'lane_blocked': lane_blocked
        }
    
    def _check_obstacle_collision(self, x: float, y: float) -> bool:
        """Check current ego pose against all static oriented boxes."""
        ego_heading = float(self.state[2])
        for obs in self.obstacles:
            if self._obb_overlap(
                (x, y), ego_heading, (2.25, 0.9),
                (obs.x, obs.y), obs.heading,
                (obs.length / 2.0, obs.width / 2.0),
            ):
                return True
        return False
    
    def _check_traffic_collision(self, x: float, y: float) -> bool:
        """Check current ego pose against traffic oriented boxes."""
        ego_heading = float(self.state[2])
        self._nearest_traffic_distance = float('inf')
        self._nearest_traffic_vehicle = None
        for veh in self.traffic_vehicles:
            
            dx = veh.x - x
            dy = veh.y - y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < self._nearest_traffic_distance:
                self._nearest_traffic_distance = float(dist)
                self._nearest_traffic_vehicle = veh
            
            if self._obb_overlap(
                (x, y), ego_heading, (2.25, 0.9),
                (veh.x, veh.y), veh.heading,
                (veh.length / 2.0, veh.width / 2.0),
            ):
                return True
        return False

    @staticmethod
    def _obb_overlap(center_a, heading_a, half_a, center_b, heading_b, half_b,
                     margin: float = 0.0) -> bool:
        """Exact oriented-rectangle overlap via SAT.

        ``margin`` widens both boxes uniformly before the test; it must be
        passed explicitly by callers who want conservatism. The default is
        exact geometric contact so collision outcomes are calibrated against
        ground truth rather than an implicit safety pad.
        """
        axes = []
        for heading in (heading_a, heading_b):
            forward = np.array([np.cos(heading), np.sin(heading)])
            lateral = np.array([-np.sin(heading), np.cos(heading)])
            axes.extend((forward, lateral))

        delta = np.asarray(center_b, dtype=np.float64) - np.asarray(center_a, dtype=np.float64)
        for axis in axes:
            axis = axis / np.linalg.norm(axis)
            projection = abs(float(np.dot(delta, axis)))
            radius_a = (
                half_a[0] * abs(np.dot(
                    np.array([np.cos(heading_a), np.sin(heading_a)]), axis
                ))
                + half_a[1] * abs(np.dot(
                    np.array([-np.sin(heading_a), np.cos(heading_a)]), axis
                ))
            )
            radius_b = (
                half_b[0] * abs(np.dot(
                    np.array([np.cos(heading_b), np.sin(heading_b)]), axis
                ))
                + half_b[1] * abs(np.dot(
                    np.array([-np.sin(heading_b), np.cos(heading_b)]), axis
                ))
            )
            if projection > radius_a + radius_b + margin:
                return False
        return True

    @staticmethod
    def _obb_axis_report(center_a, heading_a, half_a,
                         center_b, heading_b, half_b):
        """Per-axis SAT breakdown: projection vs combined radius."""
        axes = []
        for heading in (heading_a, heading_b):
            axes.extend((
                np.array([np.cos(heading), np.sin(heading)]),
                np.array([-np.sin(heading), np.cos(heading)]),
            ))
        delta = (np.asarray(center_b, dtype=np.float64)
                 - np.asarray(center_a, dtype=np.float64))
        rows = []
        for idx, axis in enumerate(axes):
            axis = axis / np.linalg.norm(axis)
            fwd_a = np.array([np.cos(heading_a), np.sin(heading_a)])
            lat_a = np.array([-np.sin(heading_a), np.cos(heading_a)])
            fwd_b = np.array([np.cos(heading_b), np.sin(heading_b)])
            lat_b = np.array([-np.sin(heading_b), np.cos(heading_b)])
            r_a = (half_a[0] * abs(np.dot(fwd_a, axis))
                   + half_a[1] * abs(np.dot(lat_a, axis)))
            r_b = (half_b[0] * abs(np.dot(fwd_b, axis))
                   + half_b[1] * abs(np.dot(lat_b, axis)))
            rows.append({
                "axis": int(idx),
                "projection": float(abs(np.dot(delta, axis))),
                "radius_sum": float(r_a + r_b),
            })
        return rows

    def _swept_pair_collision(self, start_a, end_a, half_a, start_b, end_b, half_b) -> bool:
        """Check interpolated oriented boxes to prevent tunneling."""
        distance = max(
            np.linalg.norm(np.asarray(end_a[:2]) - np.asarray(start_a[:2])),
            np.linalg.norm(np.asarray(end_b[:2]) - np.asarray(start_b[:2])),
        )
        samples = max(2, int(np.ceil(distance / 0.25)) + 1)
        record = self.collision_recorder
        for fraction in np.linspace(0.0, 1.0, samples):
            pose_a = np.asarray(start_a) + fraction * (
                np.asarray(end_a) - np.asarray(start_a)
            )
            pose_b = np.asarray(start_b) + fraction * (
                np.asarray(end_b) - np.asarray(start_b)
            )
            overlap = self._obb_overlap(
                pose_a[:2], pose_a[2], half_a,
                pose_b[:2], pose_b[2], half_b,
            )
            if record is not None:
                record.append({
                    "context": self._rec_context,
                    "fraction": float(fraction),
                    "overlap": bool(overlap),
                    "pose_a": [float(v) for v in pose_a],
                    "pose_b": [float(v) for v in pose_b],
                    "axes": self._obb_axis_report(
                        pose_a[:2], pose_a[2], half_a,
                        pose_b[:2], pose_b[2], half_b,
                    ),
                })
            if overlap:
                return True
        return False

    def _check_swept_obstacle_collision(self, start_state, end_state) -> bool:
        """Check the ego swept box against static obstacle boxes."""
        for obs in self.obstacles:
            obstacle_pose = (obs.x, obs.y, obs.heading)
            if self._swept_pair_collision(
                start_state, end_state, (2.25, 0.9),
                obstacle_pose, obstacle_pose, (obs.length / 2.0, obs.width / 2.0),
            ):
                return True
        return False

    def _check_swept_traffic_collision(self, start_state, previous_traffic) -> bool:
        """Check relative ego/traffic swept boxes for side-swipe contacts."""
        for index, veh in enumerate(self.traffic_vehicles):
            start_vehicle = previous_traffic[index]
            end_vehicle = (veh.x, veh.y, veh.heading)
            if self._swept_pair_collision(
                start_state, self.state, (2.25, 0.9),
                start_vehicle, end_vehicle, (veh.length / 2.0, veh.width / 2.0),
            ):
                return True
        return False
    
    def _is_off_road(self, lateral_offset: float) -> bool:
        """Check if car is off the road."""
        half_road = self.total_road_width / 2
        return abs(lateral_offset) > half_road
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Domain randomization: Update obstacle/traffic counts if provided
        if options:
            if 'num_obstacles' in options:
                self.num_obstacles = options['num_obstacles']
            if 'num_traffic_vehicles' in options:
                self.num_traffic_vehicles = options['num_traffic_vehicles']
        
        # Generate obstacles and traffic
        obs_rng = np.random.default_rng(self.obstacle_seed if self.obstacle_seed else seed)
        self.obstacles = self._generate_obstacles(obs_rng)
        self.traffic_vehicles = self._generate_traffic(obs_rng)
        
        # Start position (in middle lane, after first obstacle zone)
        start_s = self.np_random.uniform(0, 15)
        pt = self.base_track.get_centerline_point(start_s)
        
        # Start in middle lane
        start_lane = self.num_lanes // 2
        lane_offset = self._get_lane_center_offset(start_lane)
        
        # Add small noise
        lateral_noise = self.np_random.uniform(-0.3, 0.3)
        heading_noise = self.np_random.uniform(-0.05, 0.05)
        
        nx = -np.sin(pt.heading)
        ny = np.cos(pt.heading)
        x = pt.x + (lane_offset + lateral_noise) * nx
        y = pt.y + (lane_offset + lateral_noise) * ny
        theta = pt.heading + heading_noise
        v = self.np_random.uniform(5.0, 8.0)
        
        self.state = np.array([x, y, theta, v], dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self._prev_target = np.array([
            lane_offset / self.lane_width,  # normalized target lane offset (center of current lane)
            self.v_ref / self.v_max         # normalized target speed (reference speed)
        ], dtype=np.float32)
        self.step_count = 0
        self.prev_s = start_s
        self.total_progress = 0.0
        self.max_brake_deficit = 0.0
        self.net_progress = 0.0
        self._stall_steps = 0
        self._progress_history: Deque[float] = deque(
            maxlen=max(1, int(np.ceil(self.stall_timeout / self.dt)))
        )
        self.episode_status = "RUNNING"
        self._clamp_active = False
        self._clamp_hazard_dist = float('inf')
        self._shield_requested_speed = 0.0
        self._shield_safe_speed = self.v_max
        self._shield_ttc = float('inf')
        self._shield_emergency = False
        self._shield_hazard_lane = start_lane
        self._nearest_traffic_distance = float('inf')
        self._nearest_traffic_vehicle = None
        self._planner_phase = "CRUISE"
        self._planner_target_lane = None
        self._lane_shield_active = False
        self._lane_shield_target_lane = None
        self._lane_shield_reason = ""
        self._last_safety_decision = {}
        self.current_lane = start_lane
        self._prev_lane = start_lane  # Track for lane change reward
        self._prev_lateral_offset = lane_offset  # V7: Track lateral offset for movement reward
        
        # Decision layer steering history (for oscillation detection)
        self._prev_steer = 0.0
        self._prev_prev_steer = 0.0
        # Attitude-hold latches and watchdog counters (QA FM-1..FM-3)
        self._hold_engaged = False
        self._stopped_latch = False
        self._transition_ref_s = start_s
        self._transition_ref_lat = lane_offset
        self._transition_motion = 0.0
        self._attitude_frozen_steps = 0
        self._attitude_deadlock = False
        
        return self._get_obs(), self._get_info()
    
    def _lane_clearance_for_shield(self, lane: int, current_s: float, ego_speed: float) -> float:
        """Estimate lane clearance after the current action-repeat window."""
        track_length = self.base_track.get_total_length()
        horizon = self.dt * self.action_repeat
        clearance = 100.0
        _, ego_lateral = self.base_track.find_closest_point(*self.state[:2])
        lane_center = self._get_lane_center_offset(lane)
        for obstacle in self.obstacles:
            if obstacle.lane != lane:
                continue
            distance = (obstacle.s - current_s) % track_length
            if 0.0 < distance < clearance:
                clearance = distance
        for vehicle in self.traffic_vehicles:
            if not self._vehicle_swept_occupies_lane(vehicle, lane):
                continue
            vehicle_lane_offset = (
                self._get_lane_center_offset(vehicle.lane)
                + vehicle.lateral_offset
            )
            world_gap = np.hypot(vehicle.x - self.state[0], vehicle.y - self.state[1])
            physical_gap = world_gap - (2.25 + vehicle.length / 2.0 + 0.3)
            if world_gap < 30.0:
                # Same longitudinal gate as _build_shield_context.
                dxg = vehicle.x - float(self.state[0])
                dyg = vehicle.y - float(self.state[1])
                fgx, fgy = (
                    float(np.cos(self.state[2])),
                    float(np.sin(self.state[2])),
                )
                if dxg * fgx + dyg * fgy > -self.hazard_behind_margin:
                    clearance = min(clearance, physical_gap)
            signed_distance = (vehicle.s - current_s) % track_length
            if signed_distance > track_length / 2.0:
                signed_distance -= track_length
            closing = max(ego_speed - vehicle.speed, 0.0)
            approaching = max(vehicle.speed - ego_speed, 0.0)
            required_front = 15.0 + closing * 1.5
            required_rear = 15.0 + approaching * 1.5
            if abs(vehicle_lane_offset - ego_lateral) < 2.4 and abs(signed_distance) < 25.0:
                clearance = min(clearance, abs(signed_distance) - 4.0)
            elif signed_distance >= 0.0 and signed_distance < 100.0:
                clearance = min(
                    clearance,
                    signed_distance - closing * horizon - required_front,
                )
            elif signed_distance < 0.0 and -signed_distance < 100.0:
                clearance = min(
                    clearance,
                    -signed_distance - approaching * horizon - required_rear,
                )
        return float(clearance)

    def _shield_hazard_lanes(self, per_lane: Dict[str, np.ndarray]) -> List[int]:
        """Lanes the emergency-brake check must respect (planner-aware)."""
        cur = self.current_lane
        lanes = {cur}
        if (self._planner_phase in ("PREPARE", "COMMIT")
                and self._planner_target_lane is not None
                and self._planner_target_lane != cur):
            tgt = self._planner_target_lane
            lanes.update(range(min(cur, tgt), max(cur, tgt) + 1))
        return sorted(lanes)

    def set_planner_context(self, phase: str, target_lane: Optional[int]) -> None:
        """Publish planner state so the speed shield can ignore unrelated lanes."""
        self._planner_phase = phase
        self._planner_target_lane = target_lane

    def _apply_lane_shield(self, target_lane_offset: float) -> float:
        """Project an unsafe lane target onto a dynamically safe lane."""
        self._lane_shield_active = False
        self._lane_shield_target_lane = None
        self._lane_shield_reason = ""
        if not self.speed_clamp:
            return target_lane_offset

        x, y, _, v = self.state
        closest, _ = self.base_track.find_closest_point(x, y)
        current_lane = self.current_lane
        proposed_xy = target_lane_offset + self.total_road_width / 2.0
        proposed_lane = int(round(proposed_xy / self.lane_width - 0.5))
        proposed_lane = max(0, min(proposed_lane, self.num_lanes - 1))
        path_lanes = range(
            min(current_lane, proposed_lane),
            max(current_lane, proposed_lane) + 1,
        )
        path_clear = min(
            self._lane_clearance_for_shield(lane, closest.s, v)
            for lane in path_lanes
        )
        required_clearance = self.collision_radius + self.creep_buffer
        if path_clear > required_clearance:
            return target_lane_offset

        candidates = []
        current_clearance = self._lane_clearance_for_shield(
            current_lane, closest.s, v
        )
        critical_current = current_clearance < 5.0
        for lane in range(self.num_lanes):
            if lane == current_lane and critical_current:
                continue
            clearance = self._lane_clearance_for_shield(lane, closest.s, v)
            path = range(min(current_lane, lane), max(current_lane, lane) + 1)
            path_clearance = min(
                self._lane_clearance_for_shield(path_lane, closest.s, v)
                for path_lane in path
            )
            path_is_unsafe_but_escaping = current_clearance <= 0.0 and lane != current_lane
            if clearance > required_clearance and (
                path_clearance > required_clearance
                or path_is_unsafe_but_escaping
            ):
                candidates.append((clearance - abs(lane - current_lane) * 2.0, lane))
        if not candidates:
            self._lane_shield_active = True
            self._lane_shield_target_lane = current_lane
            self._lane_shield_reason = "no_safe_lane_brake"
            return self._get_lane_center_offset(current_lane)

        _, safe_lane = max(candidates)
        self._lane_shield_active = safe_lane != proposed_lane
        self._lane_shield_target_lane = safe_lane
        self._lane_shield_reason = (
            "critical_escape" if critical_current else "unsafe_target_lane"
        )
        return self._get_lane_center_offset(safe_lane)

    def _lateral_aware_hazard(self, h_arc: float, lane: int) -> float:
        """Lateral-aware hazard for a swept (non-origin) lane (Bottleneck 2).

        Arc distance alone ignores that threat decays as ego drifts away from
        an obstacle laterally. Recovery is linear and bounded so the speed cap
        rises smoothly mid-crossing without ever exceeding raw arc clearance
        while still overlapped (conservative-by-construction, monotone in
        lateral separation -> no cap discontinuities).
        """
        _, ego_lateral = self.base_track.find_closest_point(*self.state[:2])
        obj_lateral = self._get_lane_center_offset(lane)
        separation = abs(ego_lateral - obj_lateral)
        same_lane_width = 1.9  # collision half-width sum
        recovery = self.creep_gain * max(0.0, separation - same_lane_width)
        return max(0.0, h_arc - recovery)

    def _build_shield_context(self) -> Dict[str, Any]:
        """Single per-frame shield snapshot shared by all safety consumers."""
        x, y, theta, v = self.state
        closest, _ = self.base_track.find_closest_point(x, y)
        current_s = closest.s
        per_lane = self._get_per_lane_info(current_s, ego_speed=v, lookahead=80.0)

        cur_lane = self.current_lane
        planner_target = self._planner_target_lane
        relevant_lanes = {cur_lane}
        if (self._planner_phase in ("PREPARE", "COMMIT") and
                planner_target is not None and planner_target != cur_lane):
            relevant_lanes.update(range(
                min(cur_lane, planner_target),
                max(cur_lane, planner_target) + 1,
            ))
        relevant_lanes = sorted(relevant_lanes)

        lane_hazards = {
            lane: min(per_lane['obstacle_dist'][lane], per_lane['traffic_dist'][lane])
            for lane in relevant_lanes
        }

        in_transition = (
            self._planner_phase in ("PREPARE", "COMMIT")
            and planner_target is not None
            and planner_target != cur_lane
        )
        if in_transition:
            # Bottleneck 2: swept (non-origin) lanes use lateral-aware hazards
            # so the speed cap recovers as the crossing progresses instead of
            # staying pinned until the lane index flips.
            for lane in relevant_lanes:
                if lane != cur_lane:
                    lane_hazards[lane] = self._lateral_aware_hazard(
                        lane_hazards[lane], lane
                    )

        hazard_lane = min(lane_hazards, key=lane_hazards.get)

        # Lane-independent physical proximity: an obstacle registered in an
        # adjacent discrete lane can still be struck when the ego drifts toward
        # the lane boundary. The speed clamp must react to the true Euclidean
        # footprint gap, not only lane-membership hazards.
        px, py = float(x), float(y)
        _, ego_lateral = self.base_track.find_closest_point(px, py)
        contact_half_width = self.collision_radius + 0.2
        # Longitudinal gate (RSS responsibility transfer): only entities
        # ahead of the ego along its heading are FORWARD hazards. A passed
        # obstacle 11 m behind must not pin the speed clamp to zero.
        fwd_x, fwd_y = float(np.cos(theta)), float(np.sin(theta))
        physical_gap = float("inf")
        for obstacle in self.obstacles:
            obstacle_lateral = self._get_lane_center_offset(obstacle.lane)
            if abs(obstacle_lateral - ego_lateral) > contact_half_width:
                continue
            dx = obstacle.x - px
            dy = obstacle.y - py
            if dx * fwd_x + dy * fwd_y <= -self.hazard_behind_margin:
                continue
            gap = (
                np.hypot(dx, dy)
                - (2.25 + obstacle.length / 2.0 + 0.3)
            )
            physical_gap = min(physical_gap, float(gap))
        for vehicle in self.traffic_vehicles:
            vehicle_lateral = (
                self._get_lane_center_offset(vehicle.lane) + vehicle.lateral_offset
            )
            if abs(vehicle_lateral - ego_lateral) > contact_half_width:
                continue
            dx = vehicle.x - px
            dy = vehicle.y - py
            if dx * fwd_x + dy * fwd_y <= -self.hazard_behind_margin:
                continue
            gap = (
                np.hypot(dx, dy)
                - (2.25 + vehicle.length / 2.0 + 0.3)
            )
            physical_gap = min(physical_gap, float(gap))

        # Attitude-aware widening (bounded, transition-gated): a vehicle yawed
        # relative to the track tangent reaches laterally beyond its half
        # width. Shrink the physical gap by that extra reach so the clamp
        # treats a badly-oriented vehicle as wider. Suppressed during
        # PREPARE/COMMIT so valid slow crossings are never re-pinned.
        heading_err = abs(self._normalize_angle(theta - closest.heading))
        if self._planner_phase not in ("PREPARE", "COMMIT"):
            extra_reach = max(
                0.0,
                2.25 * float(np.sin(heading_err))
                + 0.9 * float(np.cos(heading_err))
                - 0.9,
            )
            physical_gap = max(0.0, physical_gap - extra_reach)

        return {
            "per_lane": per_lane,
            "v": v,
            "current_s": current_s,
            "cur_lane": cur_lane,
            "planner_target": planner_target,
            "relevant_lanes": relevant_lanes,
            "lane_hazards": lane_hazards,
            "hazard_lane": hazard_lane,
            "hazard_dist": lane_hazards[hazard_lane],
            "physical_gap": physical_gap,
            "in_transition": in_transition,
        }

    def _apply_speed_clamp(self, target_lane_offset: float, target_speed: float,
                           ctx: Optional[Dict[str, Any]] = None) -> float:
        """Clamp target_speed so the car can always brake before reaching a hazard.

        Kinematic feasibility: with braking deceleration a_max, the distance to
        stop from speed v is v^2 / (2*a_max). We keep a clamp_margin on top so the
        controller isn't riding the exact limit.

        The hazard distance is taken as the min over the CURRENT lane and the lane
        the target_lane_offset is steering toward, so a lane-change into danger is
        also braked (not just the lane we're leaving).

        Returns the clamped target_speed and records engagement scratch for diagnostics.
        """
        ctx = ctx if ctx is not None else self._build_shield_context()
        v = ctx["v"]
        per_lane = ctx["per_lane"]

        # Which lane are we steering toward? target_lane_offset is an absolute
        # lateral offset from road center; convert to a lane index.
        target_xy = target_lane_offset + self.total_road_width / 2
        target_lane_idx = int(round(target_xy / self.lane_width - 0.5))
        target_lane_idx = max(0, min(target_lane_idx, self.num_lanes - 1))

        cur_lane = ctx["cur_lane"]
        relevant_lanes = ctx["relevant_lanes"]
        planner_target = ctx["planner_target"]
        lane_hazards = ctx["lane_hazards"]
        hazard_lane = ctx["hazard_lane"]
        hazard_dist = ctx["hazard_dist"]
        physical_gap = ctx.get("physical_gap", float("inf"))
        self._shield_hazard_lane = hazard_lane

        in_transition = ctx["in_transition"]
        # The physical gap overrides lane-membership hazards: contact geometry
        # does not care which discrete lane an obstacle was registered in.
        hazard_dist = min(hazard_dist, physical_gap)
        if in_transition:
            # Escape semantics: while a lane change is committed, cap speed by
            # the worst hazard across ALL not-yet-cleared swept lanes (origin
            # included until ego has actually left it).
            #
            # Geometry-aware creep cap (replaces the flat 1.5 m/s floor that
            # caused the 15%->60% collision regression): the cap shrinks
            # linearly with remaining gap and reaches zero before the obstacle
            # corner becomes reachable, so a stalled maneuver can still creep
            # but can never re-accelerate inside the lethal band.
            swept_min = min(lane_hazards[lane] for lane in relevant_lanes)
            safe_dist = max(0.0, swept_min - self.clamp_margin)
            safe_speed = np.sqrt(2.0 * self.a_max * safe_dist)
            creep_cap = self.creep_gain * max(
                0.0, swept_min - self.collision_radius - self.creep_buffer
            )
            safe_speed = max(safe_speed, min(creep_cap, self.v_max))
            hazard_dist_effective = swept_min
        else:
            # Max safe speed: distance available to stop = hazard_dist - margin
            safe_dist = max(0.0, hazard_dist - self.clamp_margin)
            safe_speed = np.sqrt(2.0 * self.a_max * safe_dist)
            hazard_dist_effective = hazard_dist
        safe_speed = min(safe_speed, self.v_max)

        ttc = float('inf')
        for lane_idx in relevant_lanes:
            traffic_dist = per_lane['traffic_dist'][lane_idx]
            closing = max(per_lane['relative_velocity'][lane_idx], 0.0)
            if traffic_dist < 80.0 and closing > 0.1:
                ttc = min(ttc, traffic_dist / closing)
                traffic_speed = max(0.0, v - per_lane['relative_velocity'][lane_idx])
                safe_speed = min(
                    safe_speed,
                    traffic_speed + traffic_dist / self.shield_ttc_horizon,
                )

        self._shield_requested_speed = float(target_speed)
        self._shield_safe_speed = float(safe_speed)
        self._shield_ttc = ttc
        self._shield_emergency = hazard_dist_effective < 35.0 or ttc < 4.0

        self._nearest_traffic_distance = float('inf')
        self._nearest_traffic_vehicle = None
        x, y = float(self.state[0]), float(self.state[1])
        for vehicle in self.traffic_vehicles:
            distance = float(np.hypot(vehicle.x - x, vehicle.y - y))
            if distance < self._nearest_traffic_distance:
                self._nearest_traffic_distance = distance
                self._nearest_traffic_vehicle = vehicle

        self._clamp_active = target_speed > safe_speed + 0.01
        self._clamp_hazard_dist = hazard_dist

        if not self.speed_clamp:
            self._clamp_active = False
            return target_speed
        # Unconditional per-frame enforcement: NEVER hand back a requested
        # speed above the safe curve, even on frames where the clamp latch was
        # previously off. The old latch-off early-return let stale fast
        # targets ride for up to action_repeat frames.
        return float(min(target_speed, safe_speed))

    def _update_stall_state(self, progress: float, speed: float) -> bool:
        """Detect a vehicle that is alive but no longer making progress."""
        if speed < self.stall_speed_threshold and progress < self.stall_progress_window * self.dt:
            self._stall_steps += 1
        else:
            self._stall_steps = 0

        self._progress_history.append(max(progress, 0.0))
        elapsed = len(self._progress_history) * self.dt
        recent_progress = sum(self._progress_history)

        return (
            elapsed >= self.stall_timeout
            and recent_progress < self.stall_progress_window
            and speed < self.stall_speed_threshold
        )

    def _compute_tracking_action(self, target_lane_offset: float, target_speed: float) -> np.ndarray:
        """Convert high-level target [lane_offset, speed] to low-level [accel, steer].

        Uses Pure Pursuit for lateral control and a P-controller for speed.
        This is the key change from flat RL: the policy outputs intentions,
        and this deterministic controller handles smooth execution.
        """
        x, y, theta, v = self.state

        closest, lateral_now = self.base_track.find_closest_point(x, y)
        closest_s = closest.s

        # Speed-dependent lookahead (longer = smoother lane changes)
        min_lookahead = 3.0
        lookahead_time = 1.0  # 1 second lookahead for gentle, smooth steering
        lookahead = max(min_lookahead, v * lookahead_time)

        # Target point on base track
        total_s = self.base_track.get_total_length()
        target_s = closest_s + lookahead
        base_target = self.base_track.get_centerline_point(target_s % total_s)

        # Offset target by target_lane_offset in normal direction
        nx = -np.sin(base_target.heading)
        ny = np.cos(base_target.heading)
        target_x = base_target.x + target_lane_offset * nx
        target_y = base_target.y + target_lane_offset * ny

        # Pure Pursuit steering
        dx = target_x - x
        dy = target_y - y
        alpha = np.arctan2(dy, dx) - theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        steer = np.arctan2(2.0 * self.L * np.sin(alpha), lookahead)
        steer = np.clip(steer, -self.delta_max, self.delta_max)
        # Low-speed attitude protection (ep10 root cause, hardened per QA
        # review). Four defenses:
        #   FM-1 anti-chatter: both gates are latched with hysteresis so
        #     control authority cannot flip frame-to-frame at a threshold.
        #   FM-2 stored-energy kick: steer command is ramped by speed so no
        #     saturated command survives a stop waiting to fire on release.
        #   FM-3 deadlock honesty: at v~0 nothing can rotate the body; the
        #     frozen state is exposed via attitude_deadlock instead of being
        #     silently "corrected".
        #   Transition protection: a committed transition keeps steering
        #     authority only while it makes measurable arc/lateral progress;
        #     a stalled one loses it, a live one is never hijacked.
        heading_err_t = self._normalize_angle(theta - closest.heading)
        if not np.isfinite(heading_err_t):
            heading_err_t = 0.0
        err_abs = abs(heading_err_t)
        in_transition = self._planner_phase in ("PREPARE", "COMMIT")

        # FM-1: heading-error hysteresis (engage > 8 deg, release < 5 deg).
        if self._hold_engaged:
            self._hold_engaged = (
                err_abs > self.attitude_hold_release_err
            )
        else:
            self._hold_engaged = err_abs > self.tangent_hold_min_err

        # FM-1: stopped latch (engage < 0.3 m/s, release > 0.45 m/s).
        if self._stopped_latch:
            self._stopped_latch = (
                v < self.attitude_stopped_release_speed
            )
        else:
            self._stopped_latch = v < 0.3

        # Transition-progress tracker: EMA of per-frame arc + lateral motion,
        # wrap-aware. Reset whenever outside a committed transition.
        if in_transition:
            d_arc = float(closest.s - self._transition_ref_s)
            if d_arc < -0.5 * self.base_track.get_total_length():
                d_arc += self.base_track.get_total_length()
            d_lat = abs(float(lateral_now) - self._transition_ref_lat)
            self._transition_motion = (
                0.9 * self._transition_motion + 0.1 * (abs(d_arc) + d_lat)
            )
            transition_alive = (
                self._transition_motion
                > self.attitude_transition_motion_eps
            )
        else:
            transition_alive = False
            self._transition_motion = 0.0
        self._transition_ref_s = float(closest.s)
        self._transition_ref_lat = float(lateral_now)

        hold_permitted = (not in_transition) or (
            self._stopped_latch and not transition_alive
        )

        if (self._hold_engaged and hold_permitted
                and v < self.tangent_hold_speed):
            # FM-2: speed-ramped authority. Yaw capability scales with v, so
            # must the command; nothing saturates through a standstill.
            authority = float(np.clip(
                v / self.attitude_authority_speed, 0.0, 1.0
            ))
            steer = float(np.clip(
                -self.heading_hold_gain * heading_err_t,
                -self.delta_max,
                self.delta_max,
            )) * authority
            if not np.isfinite(steer):
                steer = 0.0
            # FM-3: frozen-attitude watchdog (1 s at < 0.05 m/s).
            if v < 0.05:
                self._attitude_frozen_steps += 1
            else:
                self._attitude_frozen_steps = 0
        else:
            self._attitude_frozen_steps = 0
        self._attitude_deadlock = (
            self._attitude_frozen_steps >= int(round(1.0 / self.dt))
        )

        # Speed P-controller
        speed_error = target_speed - v
        accel = np.clip(speed_error * 2.0, -self.a_max, self.a_max)

        return np.array([accel, steer], dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        requested_lane_offset = float(action[0])
        target_lane_offset, target_speed = action
        previous_state = self.state.copy()
        previous_traffic = [
            (vehicle.x, vehicle.y, vehicle.heading)
            for vehicle in self.traffic_vehicles
        ]

        target_lane_offset = self._apply_lane_shield(target_lane_offset)

        # Single per-frame hazard snapshot (Bottleneck 1): computed once here,
        # consumed by the speed clamp, the emergency-brake override, and the
        # brake-deficit health metric. Previously each consumer re-scanned all
        # obstacles and traffic (~30 scans per decision).
        frame_ctx = self._build_shield_context()

        # Kinematic speed clamp (safety override) — brakes BEFORE the state update
        target_speed = self._apply_speed_clamp(
            target_lane_offset, target_speed, ctx=frame_ctx
        )
        action = np.array([target_lane_offset, target_speed], dtype=np.float32)

        # Compute tracking controls from high-level targets
        a, delta = self._compute_tracking_action(target_lane_offset, target_speed)

        # Emergency-brake feasibility override: the P-controller tracks the
        # safe-speed curve lazily near zero error (exponential tail), which
        # overshoots when the curve demands near-a_max deceleration. If the
        # current speed cannot be shed within the available gap (including the
        # P-controller tail allowance), saturate braking immediately.
        v_chk = float(self.state[3])
        haz_eff = frame_ctx["hazard_dist"]
        required_stop = (
            v_chk ** 2 / (2.0 * self.a_max)
            + self.brake_tail_gain * v_chk  # P-controller exponential-tail allowance
        )
        available_gap = max(0.0, haz_eff - self.collision_radius)
        deficit = required_stop - available_gap
        self.max_brake_deficit = max(self.max_brake_deficit, deficit)
        if deficit > 0.0:
            a = -self.a_max

        self._last_safety_decision = {
            "requested_lane_offset": requested_lane_offset,
            "requested_speed": float(self._shield_requested_speed),
            "effective_lane_offset": float(target_lane_offset),
            "effective_speed": float(target_speed),
            "effective_acceleration": float(a),
            "effective_steering": float(delta),
            "planner_phase": self._planner_phase,
            "hazard_lane": int(self._shield_hazard_lane),
            "hazard_distance": float(self._clamp_hazard_dist),
            "safety_reason": self._lane_shield_reason,
            "attitude_hold_engaged": bool(self._hold_engaged),
            "attitude_deadlock": bool(self._attitude_deadlock),
        }

        x, y, theta, v = self.state
        
        # Kinematic bicycle model
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + (v / self.L) * np.tan(delta) * self.dt
        v_new = np.clip(v + a * self.dt, 0.0, self.v_max)
        
        # Normalize heading
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        self.state = np.array([x_new, y_new, theta_new, v_new], dtype=np.float32)
        previous_s = self.prev_s
        previous_lane = self.current_lane

        # Update traffic vehicles (they move too!)
        self._update_traffic(ego_s=previous_s, ego_lane=previous_lane)
        
        # Track info
        closest, lateral_offset = self.base_track.find_closest_point(x_new, y_new)
        heading_error = self._normalize_angle(theta_new - closest.heading)
        self.current_lane = self._get_current_lane(lateral_offset)
        lateral_velocity = (
            lateral_offset - self._prev_lateral_offset
        ) / self.dt
        self._prev_lateral_offset = lateral_offset
        
        # Progress
        current_s = closest.s
        progress = current_s - self.prev_s
        if progress < -self.base_track.get_total_length() / 2:
            progress += self.base_track.get_total_length()
        self.prev_s = current_s
        
        # Comfort (jerk, steer rate)
        jerk = abs(a - self.prev_action[0]) / self.dt
        steer_rate = abs(delta - self.prev_action[1]) / self.dt
        
        # Get per-lane danger information for reward shaping
        per_lane_info = self._get_per_lane_info(current_s, ego_speed=v_new, lookahead=60.0)
        
        # === LANE CHANGE DETECTION & SAFETY CALCS ===
        # Detect if we're in an "emergency lane change" situation
        current_lane_obs_dist = per_lane_info['obstacle_dist'][self.current_lane]
        current_lane_traffic_dist = per_lane_info['traffic_dist'][self.current_lane]
        min_danger_dist = min(current_lane_obs_dist, current_lane_traffic_dist)
        
        # Check if there's a safer lane available
        safest_lane = self.current_lane
        safest_dist = min_danger_dist
        for lane in range(self.num_lanes):
            lane_obs_dist = per_lane_info['obstacle_dist'][lane]
            lane_traffic_dist = per_lane_info['traffic_dist'][lane]
            lane_min_dist = min(lane_obs_dist, lane_traffic_dist)
            if lane_min_dist > safest_dist:
                safest_dist = lane_min_dist
                safest_lane = lane
        
        # We're in "emergency mode" if danger is close AND there's a safer lane
        emergency_lane_change = (min_danger_dist < 35.0 and safest_lane != self.current_lane)

        # REWARD COMPUTATION
        reward = 0.0

        if self.use_decision_layer:
            # === V12 COMMAND FOLLOWER REWARD ===
            # Pure obedience to Decision Layer targets
            
            # 1. Lane Compliance (Primary)
            # Target is ALWAYS the decision layer's choice
            target_lane_offset = self._get_lane_center_offset(self.decision_target_lane)
            lat_error = abs(lateral_offset - target_lane_offset)
            
            # Continuous reward: 1.0 at error=0, decays to 0.01 at error=3.6m (one lane width)
            # Sigma=1.2 is sharp enough to be precise but wide enough to guide
            lane_compliance = np.exp(-(lat_error**2) / (1.2**2))
            
            # 2. Speed Compliance (Secondary)
            speed_error = abs(v_new - self.decision_desired_speed)
            speed_compliance = np.exp(-(speed_error**2) / (2.0**2))
            
            # 3. Stability (Heading) - Weak penalty to prevent zigzag
            heading_penalty = (heading_error ** 2) * 0.1
            
            # 4. Smoothness
            action_penalty = 0.001 * (jerk**2 + steer_rate**2)
            
            # Weighted Sum
            # 3.0 lane + 1.0 speed = Max 4.0 per step + 0.1 alive
            reward += 3.0 * lane_compliance
            reward += 1.0 * speed_compliance
            reward -= heading_penalty
            reward -= action_penalty
            reward += 0.1 # Alive bonus
            
            # Update state variables needed for next step (even if not used in reward)
            self._prev_lane = self.current_lane
            self._prev_lateral_offset = lateral_offset

        else:
            # Safety-focused reward: imitation supplies normal driving behavior;
            # RL only needs a small progress incentive and a dense hazard signal.
            reward += 0.1 * progress

            # TTC uses ego speed for static obstacles and measured closing speed
            # for traffic, so fast approaches are penalized before contact.
            obstacle_dist = per_lane_info['obstacle_dist'][self.current_lane]
            traffic_dist = per_lane_info['traffic_dist'][self.current_lane]
            if obstacle_dist <= traffic_dist:
                closing_speed = max(v_new, 0.0)
            else:
                closing_speed = max(
                    per_lane_info['relative_velocity'][self.current_lane], 0.0
                )
            ttc = min_danger_dist / max(closing_speed, 1e-3)
            if min_danger_dist < 60.0:
                reward -= 2.0 * np.clip((6.0 - ttc) / 6.0, 0.0, 1.0)

        # Termination checks
        terminated = False
        self.total_progress += max(progress, 0.0)
        self.net_progress += progress
        lap_reached = self.net_progress >= 0.95 * self.base_track.get_total_length()
        self._rec_context = "real"
        obstacle_hit = self._check_swept_obstacle_collision(
            previous_state, self.state
        )
        traffic_hit = self._check_swept_traffic_collision(
            previous_state, previous_traffic
        )
        off_road = self._is_off_road(lateral_offset)
        stalled = self._update_stall_state(progress, float(v_new))
        # BLOCKED (declared minimal-risk hold) vs STALL (liveness failure):
        # the label follows WHY the vehicle is stationary, read from the
        # planner's published intent. Env-shield or expert-declared BRAKE means
        # "no viable escape, holding on purpose" (MRC); a stationary vehicle the
        # planner still thinks should be moving (e.g. CRUISE) is a liveness bug.
        declared_safe_stop = (
            self._lane_shield_reason == "no_safe_lane_brake"
            or self._planner_phase == "BRAKE"
        )
        blocked = stalled and declared_safe_stop
        
        if obstacle_hit or traffic_hit:
            reward -= self.obstacle_penalty
            terminated = True
        if off_road:
            reward -= self.offroad_penalty
            terminated = True
        if stalled and not terminated:
            reward -= self.obstacle_penalty
            terminated = True
        if lap_reached and not terminated:
            reward += self.completion_bonus
            terminated = True
        
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps and not terminated
        completed_lap = (
            lap_reached
            and not terminated
            and not stalled
        )
        # A lap is a successful terminal outcome, not a timeout.
        if lap_reached and not (obstacle_hit or traffic_hit or off_road or stalled):
            completed_lap = True

        if obstacle_hit:
            self.episode_status = "COLLISION_OBSTACLE"
        elif traffic_hit:
            self.episode_status = "COLLISION_TRAFFIC"
        elif off_road:
            self.episode_status = "OFF_ROAD"
        elif blocked:
            self.episode_status = "BLOCKED"
        elif stalled:
            self.episode_status = "STALL"
        elif completed_lap:
            self.episode_status = "LAP_COMPLETED"
        elif truncated:
            self.episode_status = "TIMEOUT"
        
        self.prev_action = np.array([a, delta], dtype=np.float32)
        self._prev_target = np.array([
            target_lane_offset / self.lane_width,  # normalized to lane_width
            target_speed / self.v_max              # normalized to v_max
        ], dtype=np.float32)

        info = self._get_info()
        info.update({
            "max_brake_deficit": float(self.max_brake_deficit),
            "progress": progress,
            "lateral_offset": lateral_offset,
            "lateral_velocity": float(lateral_velocity),
            "heading_error": heading_error,
            "current_lane": self.current_lane,
            "obstacle_hit": obstacle_hit,
            "traffic_hit": traffic_hit,
            "nearest_traffic_distance": float(self._nearest_traffic_distance),
            "nearest_traffic_lane": (
                int(self._nearest_traffic_vehicle.lane)
                if self._nearest_traffic_vehicle is not None else None
            ),
            "nearest_traffic_speed": (
                float(self._nearest_traffic_vehicle.speed)
                if self._nearest_traffic_vehicle is not None else None
            ),
            "nearest_traffic_lateral_offset": (
                float(self._nearest_traffic_vehicle.lateral_offset)
                if self._nearest_traffic_vehicle is not None else None
            ),
            "off_road": off_road,
            "target_lane_offset": target_lane_offset,
            "target_speed": target_speed,
            "requested_target_speed": float(self._shield_requested_speed),
            "requested_target_lane_offset": requested_lane_offset,
            "lane_shield_active": bool(self._lane_shield_active),
            "lane_shield_target_lane": self._lane_shield_target_lane,
            "lane_shield_reason": self._lane_shield_reason,
            "shield_safe_speed": float(self._shield_safe_speed),
            "shield_ttc": float(self._shield_ttc),
            "shield_active": bool(self._clamp_active),
            "emergency": bool(self._shield_emergency),
            "hazard_distance": float(self._clamp_hazard_dist),
            "hazard_lane": int(self._shield_hazard_lane),
            "planner_phase": self._planner_phase,
            "planner_target_lane": self._planner_target_lane,
            "actual_speed": float(v_new),
            "tracking_accel": a,
            "tracking_steer": delta,
            "total_progress": self.total_progress,
            "net_progress": float(self.net_progress),
            "completed_lap": completed_lap,
            "episode_status": self.episode_status,
            "stalled": bool(stalled),
            "stall_steps": int(self._stall_steps),
            "stall_reason": "zero_speed_no_progress" if stalled else "",
            "blocked": bool(blocked),
            "attitude_deadlock": bool(self._attitude_deadlock),
            "attitude_hold_engaged": bool(self._hold_engaged),
            "required_stop_distance": float(required_stop),
            "brake_deficit": float(deficit),
            "effective_acceleration": float(a),
            "effective_steering": float(delta),
            "safety_decision": dict(self._last_safety_decision),
        })
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        x, y, theta, v = self.state
        
        closest, lateral_offset = self.base_track.find_closest_point(x, y)
        heading_error = self._normalize_angle(theta - closest.heading)
        
        # Normalized observations
        lane_offset_norm = lateral_offset / self.total_road_width
        speed_norm = v / self.v_max
        
        # Yaw rate
        yaw_rate = (v / self.L) * np.tan(self.prev_action[1]) if v > 0.1 else 0.0
        
        # Lookahead curvature
        lookahead_curv = self.base_track.get_lookahead_curvature(x, y, lookahead_dist=15.0)
        
        # Distance sensors
        distances = self._get_distance_readings(x, y, theta)
        distances_norm = distances / self.sensor_range
        
        # Per-lane features for SAC (enhanced observations)
        # V8: Now includes relative velocity instead of absolute traffic speed
        per_lane_info = self._get_per_lane_info(closest.s, ego_speed=v, lookahead=60.0)
        lane_obs_dist_norm = per_lane_info['obstacle_dist'] / 60.0
        lane_traffic_dist_norm = per_lane_info['traffic_dist'] / 60.0
        # V8: Relative velocity normalized by max reasonable closing rate (~15 m/s)
        # Positive = closing gap (urgent), Negative = opening gap (safe)
        lane_relative_vel_norm = per_lane_info['relative_velocity'] / 15.0
        lane_blocked = per_lane_info['lane_blocked']
        
        # One-hot encoding of current lane (helps agent know WHERE it is)
        current_lane_onehot = np.zeros(self.num_lanes, dtype=np.float32)
        current_lane_onehot[self.current_lane] = 1.0
        
        # Lane deviation from center of current lane (normalized)
        target_lane_offset = self._get_lane_center_offset(self.current_lane)
        lane_deviation = lateral_offset - target_lane_offset
        lane_deviation_norm = lane_deviation / self.lane_width
        
        obs = np.concatenate([
            # Base features (7)
            [lane_offset_norm, heading_error, speed_norm, yaw_rate, lookahead_curv,
             self.prev_action[0] / self.a_max, self.prev_action[1] / self.delta_max],
            # Distance sensors (5)
            distances_norm,
            # Current lane info (4)
            current_lane_onehot,       # [3] one-hot current lane
            [lane_deviation_norm],     # [1] deviation from lane center
            # Per-lane features (12 = 4 x 3 lanes)
            lane_obs_dist_norm,        # [3] obstacle distance per lane
            lane_traffic_dist_norm,    # [3] traffic distance per lane
            lane_relative_vel_norm,    # [3] V8: relative velocity (closing rate) per lane
            lane_blocked,              # [3] blocked flag per lane
            # Previous intention targets (2) — intention feedback
            self._prev_target,         # [2] prev_target_lane_offset_norm, prev_target_speed_norm
        ]).astype(np.float32)
        
        # Decision layer features (optional, +2 dims)
        if self.use_decision_layer:
            # Normalize decision targets
            target_lane_norm = self.decision_target_lane / max(1, self.num_lanes - 1)
            desired_speed_norm = self.decision_desired_speed / self.v_max
            obs = np.concatenate([obs, [target_lane_norm, desired_speed_norm]])
        
        return obs
    
    def set_decision_targets(self, target_lane: int, desired_speed: float):
        """Set decision layer targets for next observation.
        
        Called by external decision layer before step() to inject
        strategic decisions into the observation space.
        
        Args:
            target_lane: Which lane the agent should aim for
            desired_speed: Target speed in m/s
        """
        self.decision_target_lane = target_lane
        self.decision_desired_speed = desired_speed
    
    def get_decision_layer_input(self) -> dict:
        """Get the current state needed for decision layer.
        
        Returns dict with all info needed for DecisionLayer.decide().
        """
        x, y, theta, v = self.state
        closest, lateral_offset = self.base_track.find_closest_point(x, y)
        per_lane_info = self._get_per_lane_info(closest.s, ego_speed=v, lookahead=60.0)
        
        return {
            'current_lane': self.current_lane,
            'obstacle_dists': per_lane_info['obstacle_dist'],
            'traffic_dists': per_lane_info['traffic_dist'],
            'relative_velocities': per_lane_info['relative_velocity'],
            'lane_blocked': per_lane_info['lane_blocked'],
            'ego_speed': v
        }
    
    def _get_info(self) -> Dict[str, Any]:
        return {"state": self.state.copy(), "step": self.step_count}
    
    def _normalize_angle(self, angle: float) -> float:
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def render(self):
        if self.render_mode is None:
            return None
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            return None
        
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(1, 1, figsize=(12, 10))
        
        self._ax.clear()
        
        # Draw track with multiple lanes
        n_pts = 300
        s_vals = np.linspace(0, self.base_track.get_total_length(), n_pts)
        
        # Draw road boundary (outer edges)
        half_road = self.total_road_width / 2
        outer_left_x, outer_left_y = [], []
        outer_right_x, outer_right_y = [], []
        
        for s in s_vals:
            pt = self.base_track.get_centerline_point(s)
            nx = -np.sin(pt.heading)
            ny = np.cos(pt.heading)
            outer_left_x.append(pt.x + half_road * nx)
            outer_left_y.append(pt.y + half_road * ny)
            outer_right_x.append(pt.x - half_road * nx)
            outer_right_y.append(pt.y - half_road * ny)
        
        self._ax.plot(outer_left_x, outer_left_y, 'k-', linewidth=3)
        self._ax.plot(outer_right_x, outer_right_y, 'k-', linewidth=3)
        
        # Draw lane dividers (dashed lines)
        for lane in range(1, self.num_lanes):
            lane_x, lane_y = [], []
            lane_offset = self._get_lane_center_offset(lane) - self.lane_width / 2
            for s in s_vals:
                pt = self.base_track.get_centerline_point(s)
                nx = -np.sin(pt.heading)
                ny = np.cos(pt.heading)
                lane_x.append(pt.x + lane_offset * nx)
                lane_y.append(pt.y + lane_offset * ny)
            self._ax.plot(lane_x, lane_y, 'w--', linewidth=1.5, alpha=0.8)
        
        # Draw road surface (gray fill)
        road_x = outer_left_x + outer_right_x[::-1] + [outer_left_x[0]]
        road_y = outer_left_y + outer_right_y[::-1] + [outer_left_y[0]]
        self._ax.fill(road_x, road_y, color='#404040', alpha=0.5, zorder=0)
        
        # Draw obstacles (orange with X marker)
        for obs in self.obstacles:
            cos_h, sin_h = np.cos(obs.heading), np.sin(obs.heading)
            corners = np.array([
                [obs.length/2, obs.width/2],
                [obs.length/2, -obs.width/2],
                [-obs.length/2, -obs.width/2],
                [-obs.length/2, obs.width/2],
            ])
            rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            corners_rot = corners @ rot.T + np.array([obs.x, obs.y])
            obstacle_patch = plt.Polygon(corners_rot, color='orange', alpha=0.9, zorder=2)
            self._ax.add_patch(obstacle_patch)
            # Obstacle marker
            self._ax.plot(obs.x, obs.y, 'rx', markersize=10, markeredgewidth=2)
        
        # Draw traffic vehicles (cyan/green with speed indicator)
        for veh in self.traffic_vehicles:
            cos_h, sin_h = np.cos(veh.heading), np.sin(veh.heading)
            corners = np.array([
                [veh.length/2, veh.width/2],
                [veh.length/2, -veh.width/2],
                [-veh.length/2, -veh.width/2],
                [-veh.length/2, veh.width/2],
            ])
            rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            corners_rot = corners @ rot.T + np.array([veh.x, veh.y])
            # Color based on speed (slower=cyan, faster=green)
            speed_ratio = veh.speed / self.v_max
            veh_color = (0, 0.5 + 0.5 * speed_ratio, 1 - 0.5 * speed_ratio)
            veh_patch = plt.Polygon(corners_rot, color=veh_color, alpha=0.85, zorder=2)
            self._ax.add_patch(veh_patch)
            # Speed indicator arrow
            arrow_len = veh.speed * 0.2
            self._ax.arrow(veh.x, veh.y, arrow_len * cos_h, arrow_len * sin_h,
                          head_width=0.4, head_length=0.2, fc='white', ec='black', 
                          alpha=0.7, zorder=2.5)
        
        # Draw vehicle
        x, y, theta, v = self.state
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        car_length = self.L * 1.2
        car_width = 1.8
        corners = np.array([
            [car_length/2, car_width/2],
            [car_length/2, -car_width/2],
            [-car_length/2, -car_width/2],
            [-car_length/2, car_width/2],
        ])
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        corners_rot = corners @ rot.T + np.array([x, y])
        car_patch = plt.Polygon(corners_rot, color='blue', alpha=0.9, zorder=3)
        self._ax.add_patch(car_patch)
        
        # Draw heading arrow
        arrow_len = 3.0
        self._ax.arrow(x, y, arrow_len * cos_t, arrow_len * sin_t,
                       head_width=0.8, head_length=0.4, fc='yellow', ec='black', zorder=4)
        
        # Draw sensor rays
        distances = self._get_distance_readings(x, y, theta)
        for i, (angle, dist) in enumerate(zip(self.sensor_angles, distances)):
            ray_angle = theta + angle
            end_x = x + dist * np.cos(ray_angle)
            end_y = y + dist * np.sin(ray_angle)
            color = 'red' if dist < 20 else 'green'
            self._ax.plot([x, end_x], [y, end_y], color=color, alpha=0.3, linewidth=1)
        
        self._ax.set_aspect('equal')
        self._ax.set_title(f'Step: {self.step_count} | Speed: {v:.1f} m/s | Lane: {self.current_lane + 1}/{self.num_lanes}')
        
        # Set view limits centered on car
        view_range = 60
        self._ax.set_xlim(x - view_range, x + view_range)
        self._ax.set_ylim(y - view_range, y + view_range)
        
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        
        if self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return img
        
        return None

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None


# Register environment
gym.register(
    id="MultiLaneEnv-v0",
    entry_point="autonomous_car.env.multilane_env:MultiLaneEnv",
    max_episode_steps=2000,
)
