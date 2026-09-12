"""
Track geometry for the autonomous car simulation.

Provides road centerlines, lane offset calculations, and curvature information.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class TrackPoint:
    """A point on the track with geometric information."""
    x: float
    y: float
    heading: float  # tangent direction (radians)
    curvature: float  # 1/radius, positive = left turn
    s: float  # arc length from start


class Track(ABC):
    """Abstract base class for track geometry."""
    
    def __init__(self, lane_width: float = 3.5):
        self.lane_width = lane_width
    
    @abstractmethod
    def get_centerline_point(self, s: float) -> TrackPoint:
        """Get track point at arc length s."""
        pass
    
    @abstractmethod
    def get_total_length(self) -> float:
        """Total track length."""
        pass
    
    def find_closest_point(self, x: float, y: float) -> Tuple[TrackPoint, float]:
        """
        Find closest point on centerline to (x, y).
        
        Returns:
            (TrackPoint, lateral_offset): closest point and signed lateral offset
                                          (positive = left of centerline)
        """
        # Discretize and search (can be optimized with spatial indexing)
        n_samples = max(100, int(self.get_total_length() / 0.5))
        s_vals = np.linspace(0, self.get_total_length(), n_samples)
        
        min_dist = float('inf')
        best_s = 0.0
        
        for s in s_vals:
            pt = self.get_centerline_point(s)
            dist = np.sqrt((x - pt.x)**2 + (y - pt.y)**2)
            if dist < min_dist:
                min_dist = dist
                best_s = s
        
        # Refine with local search
        for _ in range(5):
            ds = 0.1
            pt_center = self.get_centerline_point(best_s)
            pt_left = self.get_centerline_point(max(0, best_s - ds))
            pt_right = self.get_centerline_point(min(self.get_total_length(), best_s + ds))
            
            dist_center = np.sqrt((x - pt_center.x)**2 + (y - pt_center.y)**2)
            dist_left = np.sqrt((x - pt_left.x)**2 + (y - pt_left.y)**2)
            dist_right = np.sqrt((x - pt_right.x)**2 + (y - pt_right.y)**2)
            
            if dist_left < dist_center:
                best_s = max(0, best_s - ds)
            elif dist_right < dist_center:
                best_s = min(self.get_total_length(), best_s + ds)
            else:
                break
        
        closest = self.get_centerline_point(best_s)
        
        # Compute signed lateral offset
        dx = x - closest.x
        dy = y - closest.y
        # Normal vector (perpendicular to heading, pointing left)
        nx = -np.sin(closest.heading)
        ny = np.cos(closest.heading)
        lateral_offset = dx * nx + dy * ny
        
        return closest, lateral_offset
    
    def get_heading_error(self, x: float, y: float, theta: float) -> float:
        """Get heading error (vehicle heading - track heading at closest point)."""
        closest, _ = self.find_closest_point(x, y)
        error = theta - closest.heading
        # Normalize to [-pi, pi]
        while error > np.pi:
            error -= 2 * np.pi
        while error < -np.pi:
            error += 2 * np.pi
        return error
    
    def get_lookahead_curvature(self, x: float, y: float, lookahead_dist: float = 5.0) -> float:
        """Get average curvature over lookahead distance."""
        closest, _ = self.find_closest_point(x, y)
        s_start = closest.s
        s_end = min(s_start + lookahead_dist, self.get_total_length())
        
        # Average curvature over lookahead
        n_samples = 10
        curvatures = []
        for s in np.linspace(s_start, s_end, n_samples):
            pt = self.get_centerline_point(s % self.get_total_length())
            curvatures.append(pt.curvature)
        
        return np.mean(curvatures)
    
    def is_off_track(self, x: float, y: float) -> bool:
        """Check if position is outside lane boundaries."""
        _, lateral_offset = self.find_closest_point(x, y)
        return abs(lateral_offset) > self.lane_width / 2


class CircularTrack(Track):
    """Simple circular track for basic testing."""
    
    def __init__(self, radius: float = 50.0, lane_width: float = 3.5):
        super().__init__(lane_width)
        self.radius = radius
        self._length = 2 * np.pi * radius
    
    def get_centerline_point(self, s: float) -> TrackPoint:
        # Wrap s to [0, length)
        s = s % self._length
        angle = s / self.radius
        
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        heading = angle + np.pi / 2  # tangent direction
        curvature = 1.0 / self.radius  # constant positive curvature
        
        return TrackPoint(x=x, y=y, heading=heading, curvature=curvature, s=s)
    
    def get_total_length(self) -> float:
        return self._length


class OvalTrack(Track):
    """
    Oval track with two straights and two semicircles.
    
    Layout:
        straight_length on top and bottom
        semicircles of given radius on left and right
    """
    
    def __init__(
        self,
        straight_length: float = 100.0,
        turn_radius: float = 30.0,
        lane_width: float = 3.5
    ):
        super().__init__(lane_width)
        self.straight_length = straight_length
        self.turn_radius = turn_radius
        
        # Total length: 2 straights + 2 semicircles
        self._length = 2 * straight_length + 2 * np.pi * turn_radius
        
        # Segment boundaries (arc length)
        self._s1 = straight_length  # end of bottom straight
        self._s2 = self._s1 + np.pi * turn_radius  # end of right semicircle
        self._s3 = self._s2 + straight_length  # end of top straight
        # s4 = total length (end of left semicircle)
    
    def get_centerline_point(self, s: float) -> TrackPoint:
        s = s % self._length
        
        if s < self._s1:
            # Bottom straight (going right)
            x = s
            y = 0.0
            heading = 0.0
            curvature = 0.0
        elif s < self._s2:
            # Right semicircle (turning left/counterclockwise)
            arc = s - self._s1
            angle = arc / self.turn_radius
            cx = self.straight_length  # center x
            cy = self.turn_radius  # center y
            x = cx + self.turn_radius * np.sin(angle)
            y = cy - self.turn_radius * np.cos(angle)
            heading = angle
            curvature = 1.0 / self.turn_radius
        elif s < self._s3:
            # Top straight (going left)
            dist = s - self._s2
            x = self.straight_length - dist
            y = 2 * self.turn_radius
            heading = np.pi
            curvature = 0.0
        else:
            # Left semicircle (turning left/counterclockwise)
            arc = s - self._s3
            angle = arc / self.turn_radius
            cx = 0.0  # center x
            cy = self.turn_radius  # center y
            x = -self.turn_radius * np.sin(angle)
            y = cy + self.turn_radius * np.cos(angle)
            heading = np.pi + angle
            curvature = 1.0 / self.turn_radius
        
        # Normalize heading to [-pi, pi]
        while heading > np.pi:
            heading -= 2 * np.pi
        while heading < -np.pi:
            heading += 2 * np.pi
        
        return TrackPoint(x=x, y=y, heading=heading, curvature=curvature, s=s)
    
    def get_total_length(self) -> float:
        return self._length


class WaypointTrack(Track):
    """
    Track defined by a sequence of waypoints.
    Uses cubic spline interpolation for smooth centerline.
    """
    
    def __init__(
        self,
        waypoints: np.ndarray,  # shape (N, 2) for (x, y)
        lane_width: float = 3.5,
        closed: bool = True
    ):
        super().__init__(lane_width)
        self.waypoints = waypoints
        self.closed = closed
        
        # Compute arc lengths
        if closed:
            pts = np.vstack([waypoints, waypoints[0]])
        else:
            pts = waypoints
        
        diffs = np.diff(pts, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self._arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        self._length = self._arc_lengths[-1]
        
        # Precompute spline (simplified linear interpolation for now)
        self._pts = pts
    
    def get_centerline_point(self, s: float) -> TrackPoint:
        if self.closed:
            s = s % self._length
        else:
            s = np.clip(s, 0, self._length)
        
        # Find segment
        idx = np.searchsorted(self._arc_lengths, s) - 1
        idx = max(0, min(idx, len(self._pts) - 2))
        
        # Interpolate within segment
        s_start = self._arc_lengths[idx]
        s_end = self._arc_lengths[idx + 1]
        t = (s - s_start) / (s_end - s_start + 1e-9)
        
        p0 = self._pts[idx]
        p1 = self._pts[idx + 1]
        
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        heading = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        
        # Estimate curvature from neighboring segments
        if idx > 0:
            p_prev = self._pts[idx - 1]
            heading_prev = np.arctan2(p0[1] - p_prev[1], p0[0] - p_prev[0])
            dheading = heading - heading_prev
            ds = (s_end - s_start) / 2
            curvature = dheading / (ds + 1e-9)
        else:
            curvature = 0.0
        
        return TrackPoint(x=x, y=y, heading=heading, curvature=curvature, s=s)
    
    def get_total_length(self) -> float:
        return self._length
