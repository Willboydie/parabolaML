"""Run with python -m unittest discover -s final/robotnik_arm_task."""
import unittest

import numpy as np

from moving_target import MovingTarget, make_target_pair, nearest_target_index


class MovingTargetTests(unittest.TestCase):
    def test_seed_and_parabolic_acceleration(self):
        target = MovingTarget(seed=42)
        np.testing.assert_array_equal(target.start, MovingTarget(seed=42).start)
        np.testing.assert_allclose(np.linalg.norm(target.velocity), target.speed)
        t1, t2 = target.duration * 0.2, target.duration * 0.7
        np.testing.assert_allclose((target.position(t2) - target.position(t1)) / (t2 - t1), target.velocity + target.acceleration * (t1 + t2) / 2)
        self.assertGreater(np.linalg.norm(target.acceleration), 0)
        self.assertGreater(np.linalg.norm(target.position(target.duration/2) - (target.start + target.end)/2), 0)

    def test_roof_exit_reentry_and_opposite_face_crossing(self):
        for seed in range(30):
            target = MovingTarget(seed=seed)
            positions = np.array([target.position(t) for t in np.linspace(0, target.duration, 101)])
            self.assertTrue(np.all(positions[:, :2] >= target.lower[:2] - 1e-12))
            self.assertTrue(np.all(positions[:, :2] <= target.upper[:2] + 1e-12))
            self.assertGreater(positions[50, 2], target.upper[2])
            self.assertLess(positions[-1, 2], target.upper[2])
            self.assertGreater(positions[0, 2], target.lower[2])
            self.assertLess(target.acceleration[2], 0)
            self.assertGreater(target.velocity[2], 0)
            self.assertLess((target.velocity + target.acceleration * target.duration)[2], 0)
            self.assertTrue(np.any(np.isclose(target.end, target.lower) | np.isclose(target.end, target.upper)))
            np.testing.assert_allclose(target.position(-1), target.start)
            np.testing.assert_allclose(target.position(target.duration), target.end)

    def test_default_dimensions_and_opposite_faces(self):
        axes = set()
        for seed in range(60):
            target = MovingTarget(seed=seed)
            np.testing.assert_allclose(target.upper - target.lower, [5, 5, 1.23])
            axis = target.face_axis
            axes.add(axis)
            boundaries = sorted([target.start[axis], target.end[axis]])
            np.testing.assert_allclose(boundaries, [target.lower[axis], target.upper[axis]])
            midpoint = target.position(target.duration / 2)
            self.assertTrue(np.all(midpoint[:2] > target.lower[:2]))
            self.assertTrue(np.all(midpoint[:2] < target.upper[:2]))
            self.assertGreater(midpoint[2], target.upper[2])
        self.assertEqual(axes, {0, 1})

    def test_centred_defaults_and_translated_bounds(self):
        target = MovingTarget(seed=1)
        np.testing.assert_allclose((target.lower[:2] + target.upper[:2]) / 2, [0, 0])
        shifted = MovingTarget(seed=1, lower=[7.5, -5.5, .07], upper=[12.5, -.5, 1.3])
        np.testing.assert_allclose(shifted.start - target.start, [10, -3, 0])
        np.testing.assert_allclose(shifted.position(1) - shifted.position(0), shifted.velocity + .5 * shifted.acceleration)
        self.assertAlmostEqual(np.linalg.norm(shifted.velocity), shifted.speed)

    def test_random_launch_speed_and_endpoint_hold(self):
        speeds = []
        for seed in range(100):
            target = MovingTarget(seed=seed)
            self.assertGreaterEqual(target.speed, .5)
            self.assertLessEqual(target.speed, .7)
            self.assertAlmostEqual(np.linalg.norm(target.velocity), target.speed)
            self.assertEqual(target.speed, MovingTarget(seed=seed).speed)
            np.testing.assert_allclose(target.position(target.duration + 10), target.end)
            speeds.append(target.speed)
        self.assertGreater(max(speeds) - min(speeds), .17)
        self.assertEqual(MovingTarget(seed=1, speed=.6).speed, .6)

    def test_draw_has_twelve_edges_and_path(self):
        class Client:
            def __init__(self):
                self.lines = []

            def addUserDebugLine(self, *args):
                self.lines.append(args)
                return len(self.lines)

        client = Client()
        self.assertEqual(len(MovingTarget(seed=4).draw(client)), 76)
        self.assertEqual(len(client.lines), 76)
        self.assertTrue(all(line[3] == 4.0 for line in client.lines[:12]))

    def test_actual_apex_clearance_is_half_to_one_and_half_metres(self):
        heights = []
        for seed in range(100):
            target = MovingTarget(seed=seed)
            apex_time = -target.velocity[2] / target.acceleration[2]
            clearance = target.position(apex_time)[2] - target.upper[2]
            self.assertGreater(apex_time, 0)
            self.assertLess(apex_time, target.duration)
            self.assertAlmostEqual(clearance, target.peak_clearance)
            self.assertGreaterEqual(clearance, .5 - 1e-12)
            self.assertLessEqual(clearance, 1.5 + 1e-12)
            heights.append(clearance)
        self.assertLess(min(heights), .6)
        self.assertGreater(max(heights), 1.4)

    def test_two_independent_paths_and_nearest_selection(self):
        pair = make_target_pair(seed=12)
        repeated = make_target_pair(seed=12)
        self.assertEqual(len(pair), 2)
        self.assertFalse(np.allclose(pair[0].start, pair[1].start))
        for target, again in zip(pair, repeated):
            np.testing.assert_allclose(target.start, again.start)
            np.testing.assert_allclose(target.end, again.end)
        for index in (0, 1):
            self.assertEqual(nearest_target_index(pair, pair[index].start), index)
        origin = np.zeros(3)
        expected = int(np.argmin([np.linalg.norm(t.start) for t in pair]))
        self.assertEqual(nearest_target_index(pair, origin), expected)

    def test_visual_target_is_unaffected_by_gravity(self):
        import pybullet as p
        from pybullet_utils.bullet_client import BulletClient
        client = BulletClient(connection_mode=p.DIRECT)
        try:
            client.setGravity(0, 0, -9.81)
            target = MovingTarget(seed=0)
            body = target.spawn(client)
            self.assertEqual(client.getDynamicsInfo(body, -1)[0], 0)
            for _ in range(100):
                client.stepSimulation()
            np.testing.assert_allclose(client.getBasePositionAndOrientation(body)[0], target.start)
        finally:
            client.disconnect()

    def test_invalid_configuration(self):
        for speed in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                MovingTarget(speed=speed)
        with self.assertRaises(ValueError):
            MovingTarget(lower=[1, 2, 3], upper=[1, 3, 4])


if __name__ == "__main__":
    unittest.main()
