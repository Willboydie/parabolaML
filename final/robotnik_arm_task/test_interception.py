"""Checks for observed-motion estimation and interception timing."""
import unittest
import numpy as np
from moving_target import MovingTarget
from robotnik_arm_control import InterceptionExpert, BaseControllerParams, ArmControllerParams, make_catchable_pair


class InterceptionTests(unittest.TestCase):
    def test_both_generated_paths_have_moving_interceptions(self):
        bp, ap = BaseControllerParams(nav_timeout=240.), ArmControllerParams(reach_limit=.85)
        choices = set()
        for seed in range(6):
            targets, selected, plans = make_catchable_pair(seed, None, [-2.5, -2.5, .07], [2.5, 2.5, 1.3],
                                                         np.zeros(3), 0., np.array([.724, -.01, .73]), .001, bp, ap)
            self.assertEqual(len(targets), 2)
            choices.add(selected)
            for target, plan in zip(targets, plans):
                self.assertLess(plan.intercept_time, target.duration - .19)
                self.assertGreater(plan.intercept_time, plan.estimated_arrival_time - 1e-10)
                np.testing.assert_allclose(plan.intercept, target.position(plan.intercept_time), atol=1e-4)
                self.assertLess((target.velocity + target.acceleration * plan.intercept_time)[2], 0)
                self.assertGreaterEqual(target.speed, .5)
                self.assertLessEqual(target.speed, .7)
        self.assertEqual(choices, {0, 1})

    def test_waits_two_steps_and_estimates_motion(self):
        target = MovingTarget(seed=3)
        expert = InterceptionExpert(target.lower, target.upper)
        self.assertFalse(expert.observe(0.0, target.position(0.0)))
        self.assertFalse(expert.observe(0.001, target.position(0.001)))
        self.assertTrue(expert.observe(0.002, target.position(0.002)))
        np.testing.assert_allclose(expert.velocity, target.velocity + .002 * target.acceleration, atol=1e-7)
        np.testing.assert_allclose(expert.acceleration, target.acceleration, atol=1e-7)
        for delay in (1., 20., 1000.):
            np.testing.assert_allclose(expert.predict(target.position(.002), delay), target.position(.002 + delay), atol=1e-4)

    def test_plan_is_future_point_on_observed_path(self):
        for seed in range(25):
            target = MovingTarget(seed=seed)
            expert = InterceptionExpert(target.lower, target.upper)
            for time in (0., .001, .002):
                expert.observe(time, target.position(time))
            expert.plan(.002, target.position(.002), np.zeros(3), 0., np.array([.7, 0, .7]),
                        BaseControllerParams(nav_timeout=240.), ArmControllerParams(reach_limit=.85))
            self.assertGreater(expert.intercept_time, .002)
            self.assertTrue(expert.interception_allowed(target.position(.002), expert.intercept_time - .002))
            self.assertLess(expert.intercept[2], target.upper[2] + 1e-4)
            self.assertLess((target.velocity + target.acceleration * min(expert.intercept_time, target.duration))[2], 0)
            np.testing.assert_allclose(expert.intercept, target.position(expert.intercept_time), atol=1e-4)
            self.assertGreater(expert.intercept_time, np.linalg.norm(expert.goal_pos) / .25)


    def test_earliest_feasible_and_shortest_base_travel(self):
        target = MovingTarget(seed=2)
        expert = InterceptionExpert(target.lower, target.upper)
        for time in (0., .001, .002):
            expert.observe(time, target.position(time))
        base, ee = np.zeros(3), np.array([.7, 0, .7])
        bp, ap = BaseControllerParams(nav_timeout=240.), ArmControllerParams(reach_limit=.85)
        expert.plan(.002, target.position(.002), base, 0., ee, bp, ap)
        delay = expert.intercept_time - .002
        earlier = expert.predict(target.position(.002), delay - .1)
        if expert.interception_allowed(target.position(.002), delay - .1):
            self.assertFalse(any(o[0] <= delay - .1 for o in expert.arrival_options(earlier, base, 0., ee, bp, ap)))
        feasible = [o for o in expert.arrival_options(expert.intercept, base, 0., ee, bp, ap) if o[0] <= delay + 1e-10]
        self.assertAlmostEqual(expert.planned_base_distance, min(o[1] for o in feasible))
        self.assertLessEqual(expert.estimated_arrival_time, expert.intercept_time)

    def test_no_capture_during_launch_or_above_roof(self):
        target = MovingTarget(seed=0)
        expert = InterceptionExpert(target.lower, target.upper)
        for t in (0., .001, .002):
            expert.observe(t, target.position(t))
        self.assertFalse(expert.interception_allowed(target.position(.002), 0.1))
        self.assertFalse(expert.interception_allowed(target.position(.002), target.duration / 2 - .002))
        self.assertGreater(expert.predict(target.position(.002), target.duration / 2 - .002)[2], target.upper[2])
        self.assertTrue(expert.interception_allowed(target.position(.002), target.duration - .002))

    def test_reachable_stationary_target_avoids_base_motion(self):
        expert = InterceptionExpert([-2.5, -2.5, .07], [2.5, 2.5, 1.3])
        target = np.array([.65, 0., .7])
        expert.plan(0., target, np.zeros(3), 0., target.copy(),
                    BaseControllerParams(nav_timeout=240.), ArmControllerParams(reach_limit=.85))
        self.assertEqual(expert.planned_base_distance, 0.)
        np.testing.assert_allclose(expert.goal_pos, [0., 0.])

if __name__ == '__main__':
    unittest.main()
