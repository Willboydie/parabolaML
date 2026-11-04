"""Random, high-arc parabolic target trajectories for the reaching experiment."""
from itertools import product

import numpy as np


class MovingTarget:
    """Follow a parabolic arc between random points on opposite cuboid faces, then hold position.

    The default cuboid is centred on the world origin;
    its horizontal footprint is 5 x 5 m and height spans 0.07--1.30 m. Positions and speed use metres and seconds.
    Scripted launch speed is sampled uniformly from 0.50 to 0.70 m/s.
    ``lower`` and ``upper`` can override the complete world-coordinate bounds.
    """

    def __init__(self, seed=None, speed=None, lower=None, upper=None):
        self.lower = np.array([-2.5, -2.5, 0.07] if lower is None else lower, dtype=float)
        self.upper = np.array([2.5, 2.5, 1.30] if upper is None else upper, dtype=float)
        if (self.lower.shape != (3,) or self.upper.shape != (3,)
                or not np.all(np.isfinite([self.lower, self.upper]))
                or not np.all(self.upper > self.lower)):
            raise ValueError("Cuboid bounds must be finite three-vectors with upper > lower")
        rng = np.random.default_rng(seed)
        if speed is None:
            speed = rng.uniform(0.50, 0.70)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Target speed must be finite and positive")
        self.speed = float(speed)
        self.start = rng.uniform(self.lower, self.upper)
        self.end = rng.uniform(self.lower, self.upper)
        self.face_axis = int(rng.integers(0, 2))
        self.start_on_upper = bool(rng.integers(0, 2))
        self.start[self.face_axis] = (self.upper if self.start_on_upper else self.lower)[self.face_axis]
        self.end[self.face_axis] = (self.lower if self.start_on_upper else self.upper)[self.face_axis]
        # Opposite vertical side faces allow an interior launch and roof re-entry.
        height = self.upper[2] - self.lower[2]
        self.start[2] = rng.uniform(self.lower[2] + 0.1 * height, self.upper[2] - 0.15 * height)
        self.end[2] = rng.uniform(self.lower[2] + 0.1 * height, self.upper[2] - 0.15 * height)
        midpoint = (self.start + self.end) / 2
        self.control = midpoint.copy()
        # Solve for the actual apex, including unequal endpoint heights.
        self.peak_clearance = float(rng.uniform(0.5, 1.5))
        peak_height = self.upper[2] + self.peak_clearance
        self.control[2] = peak_height + np.sqrt((peak_height - self.start[2]) * (peak_height - self.end[2]))
        launch_derivative = 2 * (self.control - self.start)
        self.duration = float(np.linalg.norm(launch_derivative)) / self.speed
        self.velocity = launch_derivative / self.duration
        self.acceleration = 2 * (self.end - 2 * self.control + self.start) / self.duration**2

    def position(self, t):
        """Return the world position at elapsed simulation time ``t``.

        Follow the scripted arc and hold at the opposite face after arrival.
        """
        if not np.isfinite(t):
            raise ValueError("Elapsed time must be finite")
        elapsed = np.clip(float(t), 0.0, self.duration)
        return self.start + self.velocity * elapsed + 0.5 * self.acceleration * elapsed**2

    def spawn(self, client, color=(1.0, 0.15, 0.0, 1.0)):
        """Create a visual-only sphere, unaffected by physical gravity or contact."""
        import pybullet as p
        visual = client.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=color)
        return client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                                      basePosition=self.start.tolist())

    def draw(self, client, draw_box=True, path_color=None):
        """Draw the twelve cuboid edges and the intended parabolic path in Bullet.

        Return debug-item IDs so callers can remove the drawing when needed.
        This also works with a BulletClient; no global PyBullet connection is used.
        """
        corners = list(product(*zip(self.lower, self.upper)))
        ids = []
        for i, start in enumerate(corners):
            for end in corners[i + 1:]:
                if draw_box and sum(a != b for a, b in zip(start, end)) == 1:
                    ids.append(client.addUserDebugLine(start, end, [0.2, 0.7, 1.0], 4.0))
        points = [self.position(t).tolist() for t in np.linspace(0, self.duration, 65)]
        for start, end in zip(points[:-1], points[1:]):
            ids.append(client.addUserDebugLine(start, end, [1.0, 0.65, 0.0] if path_color is None else path_color, 2.0))
        return ids


def make_target_pair(seed=None, speed=None, lower=None, upper=None):
    """Two independent reproducible trajectories when a seed is supplied."""
    seeds = np.random.SeedSequence(seed).spawn(2)
    return [MovingTarget(seed=s, speed=speed, lower=lower, upper=upper) for s in seeds]


def nearest_target_index(targets, robot_position, time=0.0):
    """Choose the closest current ball in 3D; keep the first on a tie."""
    if not targets:
        raise ValueError("At least one target is required")
    distances = [np.linalg.norm(target.position(time) - robot_position) for target in targets]
    return int(np.argmin(distances))
