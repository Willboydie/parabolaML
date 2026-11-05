"""Dataset alignment, split isolation and noisy-label checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess

import numpy as np
from collect_expert import (EpisodeRecorder, FEATURES, episode_jobs, dependency_check,
                            recommended_workers, simulation_env)
from moving_target import MovingTarget
from robotnik_arm_control import BaseControllerParams, ArmControllerParams, make_catchable_pair


class CollectionTests(unittest.TestCase):
    def test_dependency_check_reports_missing_module_and_interpreter(self):
        failure = subprocess.CompletedProcess([], 1, '', "ModuleNotFoundError: No module named 'matplotlib'")
        with patch('collect_expert.subprocess.run', return_value=failure):
            with self.assertRaisesRegex(RuntimeError, 'matplotlib'):
                dependency_check()

    def test_dependency_check_uses_headless_worker_environment(self):
        with patch('collect_expert.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')) as run:
            dependency_check()
            env = run.call_args.kwargs['env']
            self.assertEqual(env['ROBOENV_HEADLESS'], '1')
            self.assertEqual(env['OMP_NUM_THREADS'], '1')
            self.assertEqual(env['OPENBLAS_NUM_THREADS'], '1')
            self.assertEqual(env['MKL_NUM_THREADS'], '1')
            self.assertIn('import robotnik_arm_task', run.call_args.args[0])
            self.assertEqual(simulation_env()['OMP_WAIT_POLICY'], 'PASSIVE')

    def test_worker_count_follows_cores_and_memory(self):
        gib = 1024 ** 3
        self.assertEqual(recommended_workers(logical=24, physical=16, available_bytes=20 * gib), 16)
        self.assertEqual(recommended_workers(logical=24, physical=16, available_bytes=6 * gib), 4)
        self.assertEqual(recommended_workers(logical=8, physical=16, available_bytes=64 * gib), 8)
        self.assertEqual(recommended_workers(logical=4, physical=0, available_bytes=None), 4)
        self.assertEqual(recommended_workers(logical=16, physical=16, available_bytes=2 * gib), 1)

    def recorder(self, split='train_dart', balls=1):
        config = dict(sample_hz=100., wheel_noise_std=.08, arm_noise_std=.2,
                      noise_hold_seconds=.05, config_hash='test')
        job = dict(split=split, index=0, seed=1, noise_seed=2)
        rec = EpisodeRecorder(job, config)
        rec.start([MovingTarget(seed=i) for i in range(balls)], 0, .001,
                  [f'j{i}' for i in range(17)], [-2.5, -2.5, .07], [2.5, 2.5, 1.3])
        return rec

    def state(self, t=0.):
        return dict(time=t, base_pos=np.zeros(3), base_ori=np.array([0., 0., 0., 1.]),
                    base_bearing=0., base_velocity=np.zeros(6), joint_pos=np.zeros(17),
                    joint_vel=np.zeros(17), ee_pos=np.ones(3), targets=np.ones((1, 3)),
                    selected=0, phase='drive', nav_done=False, q_des=np.zeros(7), qd_des=np.zeros(7),
                    ready=True, intercept=np.ones(3), intercept_time=2., base_goal=np.ones(2),
                    base_goal_bearing=0., ee_error=1.)

    def test_split_seeds_and_defaults(self):
        jobs = episode_jobs([200, 200, 50, 50], 2026)
        self.assertEqual(len(jobs), 500)
        self.assertEqual(len({job['seed'] for job in jobs}), 500)
        self.assertEqual(jobs, episode_jobs([200, 200, 50, 50], 2026))
        self.assertEqual(sum(width for _, width in FEATURES), 96)

    def test_dart_labels_clean_and_applied_noise_held(self):
        rec = self.recorder()
        clean = np.zeros(17)
        bp, ap = BaseControllerParams(nav_timeout=240.), ArmControllerParams(reach_limit=.85)
        first = rec.command(self.state(), clean, bp, ap)
        np.testing.assert_array_equal(rec.pending['y'], clean[:11])
        self.assertGreater(np.linalg.norm(first), 0.)
        second = rec.command(self.state(.001), clean, bp, ap)
        np.testing.assert_allclose(first, second)
        self.assertTrue(np.all(np.abs(first[:4]) <= .24))
        self.assertTrue(np.all(np.abs(first[4:11]) <= .6))
        self.assertTrue(np.all(first[11:] == 0))
        self.assertEqual(rec.pending['X'].shape, (96,))
        np.testing.assert_array_equal(rec.pending['target_mask'], [1, 0])
        self.assertTrue(np.all(rec.pending['target_history'][:, 1] == 0))
        np.testing.assert_allclose(rec.pending['previous_applied_action'], first[:11])
        self.assertAlmostEqual(rec.pending['history_time_offsets'][-2], -.001)

    def test_clean_splits_and_two_target_padding(self):
        bp, ap = BaseControllerParams(nav_timeout=240.), ArmControllerParams(reach_limit=.85)
        for split in ('train_normal', 'validation', 'test'):
            rec = self.recorder(split, balls=2)
            state = self.state()
            state['targets'] = np.array([[1, 2, 3], [4, 5, 6]])
            clean = np.arange(17, dtype=float) / 10
            np.testing.assert_array_equal(rec.command(state, clean, bp, ap), clean)
            np.testing.assert_array_equal(rec.pending['target_mask'], [1, 1])
            np.testing.assert_array_equal(rec.pending['target_positions'], state['targets'])

    def test_episode_serialization_alignment_and_final_sample(self):
        rec = self.recorder('train_normal')
        bp, ap = BaseControllerParams(nav_timeout=240.), ArmControllerParams(reach_limit=.85)
        for step in range(14):
            rec.command(self.state(step * .001), np.zeros(17), bp, ap)
            rec.after_step(np.full(17, step + 1), np.zeros(3), np.ones((1, 3)))
        rec.finish('Reached step cap 14', .013, 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.npz'
            rec.save(path)
            with np.load(path, allow_pickle=False) as episode:
                np.testing.assert_array_equal(episode['simulation_step'], [0, 1, 2, 10, 13])
                np.testing.assert_array_equal(episode['next_joint_pos'][:, 0], [1, 2, 3, 11, 14])
                self.assertTrue(episode['truncated'][-1])
                self.assertFalse(np.any(episode['terminated']))
                self.assertEqual(json.loads(str(episode['metadata_json']))['outcome'], 'step_cap')

    def test_one_ball_generation_has_no_target_choice(self):
        targets, selected, plans = make_catchable_pair(2, None, [-2.5, -2.5, .07], [2.5, 2.5, 1.3],
                                                      np.zeros(3), 0., np.array([.724, -.01, .73]), .001,
                                                      BaseControllerParams(nav_timeout=240.),
                                                      ArmControllerParams(reach_limit=.85), target_count=1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(len(plans), 1)
        self.assertEqual(selected, 0)


if __name__ == '__main__':
    unittest.main()
