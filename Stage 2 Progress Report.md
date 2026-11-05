# Stage 2 Progress Report

Nov 5, 2026 · @Will

## Summary

The main Stage 2 runs are done, with one clear result and one failure.

**The averaging failure is real.** Told which ball to catch, the oracle catches 86.2%, the same as the one-ball robot on those throws (81.8%) within noise. The squared-error robot, left to choose for itself, catches 2.5%, and half its misses end between the two balls.

**The mixture density network (MDN), the proposed fix, does not catch at all.** It manages 0.5–1.0% on two balls, and 0.2% on one ball, where it misses by 1.7 m. That is worse than the squared-error robot, which averages two balls and misses by 814 mm. It fits its training data, so a miss of that size is a reason to look for a mechanical fault before a modelling one. The one-ball result bears on choosing only if the MDN trained on one-ball scenes. That check is first in [What comes next](#what-comes-next).

The scripts, checkpoints, and result files behind the numbers below are listed in [Files](#files).

## The big picture

The project tests one known weakness of learning by copying. Show a robot two balls, and an expert that catches either one at random. A standard copier trained by squared error learns the average of what the expert did. The average of "go left" and "go right" is "go to the middle", which catches nothing.

The question is whether that failure really shows up here, and whether a model that can hold several answers at once (a mixture density network, or MDN) fixes it.

The project runs in three stages:

1. **Stage 0: the expert.** A scripted robot that drives, parks and reaches to catch a ball almost perfectly. It provides all the training examples. Done.
2. **Stage 1: the one-ball backbone.** A learned robot that copies the expert with one ball, made as accurate as reasonably possible. Every Stage 2 model reuses it, so any difference between them comes from the two-ball problem alone. Done.
3. **Stage 2: two balls.** Four robots are compared on the same backbone: the one-ball ceiling, a two-ball squared-error copier, an oracle that is told which ball to catch, and the MDN. In progress.

&#91;embedded content: Stage 2 progress · 7 done, 2 now, 2 later\]

The experiment's main question is answered: averaging is real, and the oracle shows the gap can be closed. What's left is finding why the MDN breaks.

## How Stage 1 ended

Stage 1 cut the one-ball miss from 80.5 mm to 38.2 mm by finding and fixing one cause. The learned robot parked its base about 54 mm from where the expert would, and its arm didn't reach to make up the difference. So the hand missed by roughly the parking error.

| Fix | Mean miss | What it did |
| --- | --- | --- |
| Starting point | 80.5 mm | Arm ignores parking errors |
| Ball position measured from the robot's base | 65.7 mm | Part of the forward/back error corrected |
| 1,000 demos with deliberately off-target parking | 38.2 mm | Sideways error fully corrected |

The second fix worked because the expert, when parked off-target, simply reaches further with the arm. The robot had never seen this, because the expert always parked perfectly. Showing it off-target parks taught it the reach.

The forward/back error is still only partly corrected (0.43), and it makes up most of what's left. The earlier guess was that the forward/back offsets in the demos were too small. A direct test has now ruled that out (below). Two other approaches, generic noise injection (DART) and a first attempt at expert correction (DAgger), did not help.

**Where the forward bias comes from (new).** The learned robot ends up ahead of the expert on 70% of throws. That bias is created in the last open-loop chunk of the approach, and it stays because the expert never demonstrates reversing.

- **During the drive the two are level.** Two seconds before the expert stops, the along-track gap is about zero (median −1 mm).
- **The learner brakes slightly late.** The expert brakes from 0.25 m/s to 0 over about the last second. The learner follows the same ramp, but about 4.5 mm/s faster the whole way, so its zero crossing comes late.
- **The late stop lands inside a chunk it has already committed to.** The robot predicts 2 s and executes 1 s (25 steps) before it looks again. The expert's command reaches zero, on average, 52% of the way through the chunk that is open. At that instant the learner is still commanding 7.4 cm/s, the speed fixed when the chunk was issued. For the rest of the chunk it averages 2.1 cm/s while the expert is already stopped.
- **The next chunk stops but doesn't back up.** By the next replan the learner is 11 mm ahead, and 70% of episodes are already ahead.

One episode, with times relative to the expert's stop:

| Time | Learner command | Expert command | Along-track gap |
| --- | --- | --- | --- |
| −0.4 s | 0.140 m/s | 0.121 m/s | −5 mm |
| 0 s | 0.086 m/s | 0.081 m/s | +3 mm |
| +0.2 s (chunk step 23) | 0.041 m/s | 0 | +14 mm |
| +0.4 s (new chunk) | 0.002 m/s | 0 | +18 mm |

The robot never reverses because the expert never does. Across 200 training demos (84,832 policy ticks), the expert's net forward command is never negative; the minimum is exactly 0. Wheels spin backward during turns, but the base never drives backward. That is how the controller is written: forward speed is multiplied by max(0, cos(heading error)), and once the base has passed the goal it turns in place instead of reversing. The rule is `base_controller` in `robotnik_arm/robotnik_arm_control.py`.

*Plain version:* the robot brakes a fraction late. Because it only re-checks once a second, it rolls past the stopping point, and nothing in its examples ever shows it backing up. So an overshoot is permanent; an undershoot gets fixed by driving on.

This gives three possible fixes, none tested yet:

- **Re-plan more often near the stop.** Tried at 50 Hz with 0.5 s chunks. It improved the miss but didn't move the bias (below).
- **Demos parked ahead of the goal**, so the arm learns to reach back. Tried as part of the learner-matched offsets below; it didn't help.
- **Let the expert reverse when past the goal**, so the demos contain it. Untested, and now the main remaining fix.

**Testing the "offsets too small" explanation (new).** If the forward/back correction is stuck because the demos' parking offsets were too small or the wrong shape, then demos matched to where this learner actually parks should fix it. They didn't; the correction got worse.

**How it was tested:**

- **Measure where the learner parks.** The current DeepSets ceiling was rolled out on the 200 test scenes, 3 seeds, recording the base error at the closest approach in the expert's route frame. That gave 600 parking errors: median 55.3 mm from the expert, 39.6 mm forward/back and 28.4 mm sideways. Heading error was 0.8°, under the 10° gate, so the parking heading was left unchanged.
- **Draw offsets from that cloud.** 1,000 offsets, capped at 150 mm (24 were capped). The drawn spread is about 45 mm forward/back and 46 mm sideways. Both demo sets below used this same list and the same 1,000 training scenes.
- **Two ways of applying each offset:**
  - **Goal shift.** The expert plans its best park, the goal is moved by the offset, and the arm reaches from there. The expert caught every episode, median miss 0.27 mm.
  - **Planner.** The expert keeps the offset pose only if its own timing model says the base can still arrive in time. It accepted 493 poses and refused 507 as too late. It still caught every episode, median miss 0.18 mm.
- **Train and score.** Each training set is the original 1,000 demos plus 1,000 new ones. A new DeepSets model was trained on each (3 seeds, 20,000 steps, same configuration as the ceiling) and scored on the 200 test scenes with no offset applied. The pre-set bar: mean miss down by more than 2 mm, and forward/back correction above 0.55.

| Model | Mean miss | Catch at 50 mm | Forward/back correction | Sideways correction |
| --- | --- | --- | --- | --- |
| Current ceiling | 36.1 mm | 86.5% | 0.43 | 0.81 |
| Goal shift | 42.7 mm | 84.3% | 0.24 | 0.79 |
| Planner | 45.3 mm | 78.0% | 0.30 | 0.60 |

No seed cleared the bar. The seed spread is wide, about 4 mm for goal shift and 5 mm for the planner. The best goal-shift seed missed by 37.3 mm and corrected 0.49 of the forward/back error. Sideways correction held for goal shift and fell for the planner.

**What this means.** Matching the demos to how this learner really parks made the forward/back correction worse, and so did letting the planner refuse offsets it couldn't reach in time. The remaining forward/back miss is therefore not caused by the demo offsets being too small or too unlike the learner's.

The new timing below also rules out an explanation this report offered earlier, that the base might still be rolling at the catch. It isn't: the learner stops about 9 s before the catch, so its forward/back offset is a parked one, just like the demos'. Why the arm corrects sideways errors (0.81) but only part of the forward/back error (0.43), even after matched demos, is still open.

A quick next test: slide the expert's parked base 50 and 100 mm forward and back, as was done sideways, and see how the expert catches. If it catches by timing along the ball's path rather than by reaching, the "correction" measure may be undercounting what a good forward/back response looks like.

**Shorter chunks don't remove the bias (new).** Two checks were run: one on existing logs, to pin down the bias, and one 50 Hz training run with shorter chunks, to see whether the bias was overshoot from committed chunks.

*Where the bias comes from.* 50 learner rollouts were compared with the expert on the same seeds, plus 200 expert training demos. This confirms the mechanism described above:

- the learner is level with the expert during the drive, until about 2 s before the stop;
- the lead appears in the last 1 s chunk, which is already committed when the expert's stop falls about halfway through it;
- by the next re-plan the learner is about 11 mm ahead;
- the new chunk stops but doesn't back up (0.5% reverse commands after that);
- the catch comes about 9 s later, and the lead is still there (about 18–25 mm, 70% of episodes ahead).

So the bias is a late brake plus no reverse in the labels, not a heading error or random drift.

*The DeepSets ceiling has it too.* The bias was first measured on the older 80 mm policy. Over 3 seeds and the 200 test scenes, the current ceiling also ends ahead: 60% of episodes, with a median signed along-track error of +16.9 mm (mean miss 36 mm, 87% caught at 50 mm).

*The shorter-chunk test.* The hypothesis: if the robot commits to less, the overshoot tail should shrink. The 50 Hz version predicts 1 s, executes 0.5 s and uses 0.24 s of history, with the same DeepSets model shape. Seed 0 of each:

|  | 25 Hz ceiling | 50 Hz, 0.5 s execute |
| --- | --- | --- |
| Mean miss | 35.2 mm | 24.7 mm |
| Catch at 50 mm | 91% | 94% |
| Signed along-track | +18.9 mm | +19.4 mm |
| Fraction ahead | 62% | 67% |

The miss improved; the stop did not. The learner still finishes in front, slightly more often. That fits more correction during the drive, not a shorter overshoot. Seeds 1–2 and the control (25 Hz with a 0.5 s execute) weren't run, because the bias didn't move.

*Plain version:* checking the plan twice as often makes the robot more accurate overall, but it still brakes a touch late. Once it's past the goal, nothing in its examples tells it to back up, so it stays there.

**Bottom line.** The forward parking bias is real on the DeepSets ceiling. Querying twice as often with 0.5 s chunks doesn't fix it, because the expert never reverses: once past the goal, the policy has no action that would undo it.

The 50 Hz miss improvement (35.2 → 24.7 mm) is from a single seed, but it is large. It is worth confirming with seeds 1–2. If it holds, it would be a better backbone, though under the backbone rule every Stage 2 model would need re-running on it.

## Stage 2 so far

The set-up steps all passed, and the two-ball robots and the MDN have now been trained and tested (3 seeds each, 200 test scenes).

**New ball encoder (plan step 0).** The robot now reads balls through DeepSets. A small network looks at each ball on its own, and the results are added up. Ball order can't matter, and the same model works with one ball or two.

Because this changes the backbone, the one-ball robot was retrained with it (3 seeds, the same 200 test scenes). It came out slightly better than the old version, so the encoder costs nothing:

| Measure | Before DeepSets | With DeepSets |
| --- | --- | --- |
| Mean miss | 38.2 mm | 36.1 mm |
| Catch rate at 10 mm | 4.2% | 6.0% |

**Catch radius (plan step 1).** A tiny 10 mm target makes almost every model score near zero, which hides real differences. So a catch now counts if the hand comes within a set radius of the ball. The smallest radius where the DeepSets one-ball robot catches over 80% is 45 mm (80.3%). The chosen radius is 50 mm (86.5%); the next section explains why.

**Expert check (plan step 2a).** The random-choice expert was run on 100 two-ball scenes:

- It caught every episode at both 10 mm and 45 mm, with a median miss of 0.14 mm.
- It picked slot 1 on 53% of episodes. That is consistent with a fair 50/50 coin.
- Its choice never changed partway through an episode.
- It showed no preference for the nearer or the earlier ball.

So the training examples will contain a genuine, unbiased two-way choice, which is exactly what the experiment needs.

**Two-ball results (plan steps 2–3).** On the 200 two-ball test scenes:

| Robot | Catch at 50 mm | Mean miss | Misses between the balls |
| --- | --- | --- | --- |
| One-ball robot, shown only the indicated ball | 81.8% (seeds 75.5–89.0%) | 37 mm | — |
| Oracle, told which ball to catch | 86.2% | 39 mm | 0% |
| Two-ball squared error | 2.5% | 814 mm | 52% of misses |
| MDN, highest-weight rule | 0.5% | about 960 mm | 23% of misses |
| MDN, continuation rule | 1.0% | about 960 mm | 23% of misses |

The oracle catches only the ball it is told, split 41% / 45% across the two slots. In just over half of episodes it crosses once, at about 1.8 s, from the nearer landing point to the indicated one, and it never ends between the balls. So a second ball is catchable once the choice is made.

The squared-error robot changes its nearer landing point in about a third of episodes; in the typical episode it never changes. Half of its misses end nearer the midpoint than either ball. That is the averaging failure the project set out to show: it never really makes the choice. All three of its seeds catch exactly 5 scenes out of 200.

**The MDN (plan step 4).** The mixture fails even with one ball. On the same 200 one-ball test scenes as the ceiling:

| Robot | Catch at 50 mm | Mean miss |
| --- | --- | --- |
| One-ball ceiling | 86.5% | 36 mm |
| MDN, highest-weight rule | 0.2% | 1,706 mm |
| MDN, continuation rule | 0.3% | 1,632 mm |

What is known so far:

- **It fits the training data.** On training rows, the chosen component's squared error is about 0.01, and the other components hold genuinely different actions.
- **The selection rule doesn't matter.** The two rules give the same catch outcome on 99% of scenes, and all three seeds are at or below 1.5%.
- **In the loop, it is confident but unsteady.** The top weight is about 0.95, four of the five components get used, and the chosen component changes about three times per episode.

A one-ball miss of 1.6–1.7 m means the robot barely gets to the ball, a larger miss than the squared-error robot's 814 mm while averaging two balls. *Plain version:* the MDN learned the examples, but when it drives the robot itself, something goes wrong long before the catch. Reading that as evidence about choosing needs the premise check below. The training script loads the two-ball grid only. Full record: `robotnik_arm/experiments/ablation/results/stage2/mdn_compare.json`, written by `eval_mdn.py`.

## The 50 mm decision

The radius is 50 mm, not the 45 mm the plan's "smallest over 80%" rule gives. This puts the ceiling at 86.5% rather than 80.3%, leaving more room above and below it to see differences between models.

It changes two things:

- **Balls must now land at least 200 mm apart** (4 × 50 mm), up from 180 mm. If the balls land close together, a robot that aims at the midpoint can still get within 50 mm of one ball. That would hide the very failure Stage 2 is looking for.
- **The expert check still holds.** The expert caught everything at 45 mm, and a 50 mm target is easier, so it passes at 50 mm too.

`stage2_plan.txt` is not in the repo. The "smallest radius over 80%" rule is the docstring of `radius_curve.py`, and the decision table is `_reading` in `eval_two_ball.py`. Both still describe that rule. The 50 mm choice is the hardcoded `RADIUS_MM = 50.0` in `eval_two_ball.py` and `eval_mdn.py`. Those three places should be updated to record 50 mm and the 200 mm separation.

## What comes next

The two-ball comparison landed on the first row of the plan's decision table: the oracle matched the ceiling, and the squared-error robot fell far below with many misses between the balls. That called for the MDN, which was built and fails. A mixture that fits its training data should not miss by 1.7 m when there is one right answer, and that miss is larger than the squared-error robot's 814 mm while averaging two balls. Look for a mechanical cause first and a modelling cause second. The premise check takes about five minutes. The run-time check is the one most likely to find a fault of this size. The later tests are worth running once those two are clean.

0. **Check the premise.** Was the MDN trained on any one-ball scenes? `train_mdn.py` loads `data/grid50_two_ball.npz` and describes the run as two-ball data only. Confirm the grid has two balls in every row. If the MDN only saw two-ball scenes, a one-ball input is new to it. With DeepSets, adding up one ball's features instead of two shifts the input the head sees. In that case "it fails even with one ball, so the problem isn't choosing" does not follow.

   **Control.** Run the two-ball squared-error robot on the one-ball test scenes.
   - If that also collapses, one-ball scenes are not a clean test for either model. The argument has to rest on the two-ball results and the hand-off test.
   - If it still catches about as well as the ceiling, one-ball scenes are a fair test, and the MDN's 1.7 m miss belongs to the MDN.

1. **Check the run-time code against training.** This is the cheapest test and the most likely to find a 1.7 m fault. Take about 100 training observations, run both the PyTorch model and the NumPy rollout, and compare the maximum difference at each stage:
   - the raw logits and the weights after softmax;
   - the component means before un-standardising;
   - the final chunk after un-standardising.

   A gap that starts at one stage, with the stage before it still matching, names the fault. The usual culprits:
   - the output reshaped in a different order (component × time × action) in NumPy than in training;
   - softmax applied over the wrong axis;
   - un-standardising with the wrong statistics.

   If every stage matches, the rollout is a faithful copy of the trained model, and the 1.7 m miss is not a translation between the two.

   **Free test, same pass.** Compare the MDN's chunk with the ceiling's chunk for the same one-ball observation. They should be close.
   - If one-ball inputs are new to the MDN, a chunk far from the ceiling is the shifted sum from the premise check, which is what that check already predicts.
   - If one-ball inputs are familiar and the chunk is still far from the ceiling, the trained MDN itself is outputting the wrong motion. The rollout matching training would then mean the fault is in the learned weights, and a run-time translation bug is ruled out.
   - If the chunk is close to the ceiling here, while the robot still misses by 1.7 m in the loop, the fault shows up only once the chunk is executed. That points on to the hand-off test.

2. **MDN with one component (K = 1).** A one-component mixture is a squared-error model that also learns a spread. On one ball it should behave like the ceiling. Run it if the premise check says one-ball scenes are a fair test.
   - If K = 1 also fails, the problem is in the MDN code path, the part shared with the five-component model.
   - If K = 1 catches like the ceiling, several components are what breaks the robot, and tests 4–6 say which part.

3. **Hand-off test.** Let the expert drive until 0.5, 1, 2 and 4 s before the catch, then hand over to the MDN. If the premise check retires the one-ball scenes, this is where that argument moves.
   - If it misses even from 0.5 s, it can't act well from good states, which points to a bug or a poor fit.
   - If it is fine late and fails early, the error builds up over the episode, as in Stage 1, made worse by switching components.

4. **Check the weights on unseen states.** On held-out expert states, compare the error of the highest-weight component (what the robot uses) with the error of the best-matching component.
   - If the highest-weight one is often wrong, the weights are what's failing. The components themselves may still hold the right actions.
   - If the highest-weight component is also the closest, the weights are doing their job on expert states. A miss in the loop then comes from states the expert never visits, which the hand-off test and the rollout log cover.

5. **What do the components stand for?** Tag every training row with the expert's phase (align, drive, orient, done) and whether it had noise added. Then see which component wins for each tag. Do this on the two-ball training rows. The one-ball rollout, where four components get used with no ball to choose, is a separate input, and test 0 decides whether it counts.
   - If components line up with phases, the MDN is choosing between "spin", "drive" and "hold" at moments where the squared-error model averaged them into smooth driving. That would echo the DAgger failure.
   - If they line up with neither phase nor noise, the split is some other feature of the action, and the phase story is out.

6. **Watch the robot.** In rollouts, log the base speed and heading alongside the chosen component.
   - If speed or heading jumps when the component changes, switching is kicking the trajectory off.
   - If the motion stays smooth and the hand still misses by metres, the chosen actions are wrong even while the choice is steady.

If the premise check, the run-time check, or the one-component model points to a bug, fix it and rerun. If the MDN is working as designed and still fails, the fallbacks from the plan are:

- a head that first classifies which ball, then regresses the motion for that ball;
- one mixture over the ball choice only, with motion regressed separately;
- a diffusion policy, as the stretch goal.

The report is due on 06/11. If there's no time for a fix, the MDN result stands as a diagnosed negative result: the oracle shows the gap can be closed, and the MDN shows that a mixture head alone doesn't close it in this closed loop.

## Things to watch

**Small differences in catch rate are noise.** With 200 test scenes and a catch rate near 86%, chance alone moves the rate by about 2.4 points from one run to the next. Pooling all three seeds (600 episodes) narrows that, but a gap of under about 5 points between two models should not be read as real without more test scenes.

**The separation rule changes which scenes appear.** Throwing out scenes where the balls land less than 200 mm apart could favour certain throws, such as long or wide ones. Log how many scenes are rejected. If it's a large share, check that the kept throws still look like the one-ball throws.

**Like-for-like ceiling: done.** The one-ball robot was run on the two-ball test scenes with only the indicated ball. It catches 81.8%, so the oracle's 86.2% is the same result within noise.

**Checks from the plan with no result recorded yet:**

- **Identical seeds.** The squared-error robot catches exactly 5 of 200 on all three seeds. Check whether these are the same 5 scenes, and that the three checkpoints are genuinely different files (step 0e). Five common scenes are worth describing in the report; identical checkpoints would mean the seeds weren't independent.
- **Step 0a, answered.** Larger forward/back offsets were tested, matched to the learner's own parking, and made things worse (see How Stage 1 ended).
- Step 0c: the ball-swap test, showing that swapping the two balls' order leaves the output unchanged.

## Files

Ablation scripts live in `robotnik_arm/experiments/ablation` and are run from that directory. Results are under `results/`, data under `data/`. Checkpoints (`results/models/*.pt` and the matching `.npz`) are gitignored. Each logged run also has a copy under `results/stage2/runs/`, written by `stage2_log.py`.

`STATUS.md`, `STAGE1_RESULTS.md`, and `results/stage2_config.json` describe the Stage 1 winner at 80.6 mm, before DeepSets and before two balls. The ceiling in this report is the DeepSets retrain.

### Shared code

| Piece | Where |
| --- | --- |
| Expert base, including the no-reverse rule | `robotnik_arm/robotnik_arm_control.py`, `base_controller` |
| Two-ball scene draw | same file, `make_catchable_pair` |
| Sim, 42-D observation, rollout | `catch_env.py` (`CatchEnv`, `OBS_NAMES`, `rollout`, `ExpertDriver`) |
| Hyperparameters and checkpoint names | `config.py`, `Config.name` |
| Per-ball features (22-D ball, 33-D robot) | `ballset.py` |
| Torch policies and the mixture loss | `policy.py`, `DeepSetsPolicy`, `gaussian_mixture_nll` |
| Training step, oracle flag, mixture inputs | `train.py` (`deepsets_forward`, `apply_oracle_flag`, `mixture_inputs`) |
| NumPy rollout and component choice | `np_policy.py`, `choose_component`, `MDNDriver`, `SetPolicyDriver` |
| Expert collection | `robotnik_arm/collect_expert.py`, `robotnik_arm/docs/DATA_COLLECTION.md` |

Checkpoint names follow `Config.name`. The ones this report uses:

| Robot | Files under `results/models/` |
| --- | --- |
| Stage 1 start, 80.5 mm | `r25_p2_e0.5_hstack0.5s_d1000_s{0,1,2}.{pt,npz}` |
| Base-frame ball, 65.7 mm | `r25_p2_e0.5_hstack0.5s_d1000_rel_add_s{0,1,2}` (see `run_relative.py`) |
| Off-target parking, 38.2 mm | `r25_p2_e0.5_hstack0.5s_d2000_rel_add_s{0,1,2}` |
| DeepSets ceiling, 36.1 mm | `r25_p2_e0.5_hstack0.5s_d2000_deepsets_s{0,1,2}` |
| Goal-shift and planner retrains | `..._deepsets_matched_s{0,1,2}`, `..._deepsets_planned_s{0,1,2}` |
| 50 Hz, 0.5 s execute, seed 0 | `r50_p1_e0.5_hstack0.24s_d2000_deepsets_s0` |
| Two-ball squared error | `twoball_mse_s{0,1,2}` |
| Oracle | `twoball_oracle_s{0,1,2}` |
| MDN, K=5 | `twoball_mdn_s{0,1,2}` |

### How Stage 1 ended

| Claim | Script | Record |
| --- | --- | --- |
| 80.5 mm start | Stage 1 winner above | `results/relative_features.json` (`stage1`); per-seed `results/runs/r25_p2_e0.5_hstack0.5s_d1000_s{0,1,2}.json` |
| 65.7 mm, ball measured from the base | `run_relative.py`; features in `config.py` `relative_ball` | `results/relative_features.json` |
| 38.2 mm, 1,000 off-target parks | `gen_offset_parking.py`, `train_offset_parking.py` | `results/offset_parking_train.json`; demos `data/offset_parking/`; grid `data/grid50_offset.npz` |
| Along-track / sideways split of that miss | `parking_split.py` | `results/parking_split.json`. This script loads the `rel_add` checkpoints (`r25_p2_e0.5_hstack0.5s_d2000_rel_add`) |
| DART and the first DAgger attempt | `gen_quiet_dart.py`, `sweep_cov_noise.py`, `collect_dagger.py`, `compare_methods.py` | `results/method_compare.json`, `results/cov_sweep.json`, `results/recover_sweep.json` |
| Where the DeepSets ceiling parks (600 errors) | `measure_park_cloud.py` | `results/learner_park_offsets.json`, `results/learner_park_cloud.npz` |
| Goal-shift demos | `gen_offset_matched.py` | `data/offset_parking_matched/`, `data/grid50_offset_matched.npz` |
| Planner demos (493 accepted, 507 refused) | `gen_offset_planned.py`, `PlannedOffsetDriver` | `data/offset_parking_planned/`, `data/grid50_offset_planned.npz` |
| Paths shared by that comparison | `variety_data.py` | |
| Train and the table that missed the bar | `train_offset_variety.py` | `results/offset_variety_compare.json`. It only reads `oneball_deepsets.json` |
| 50 Hz, 0.5 s chunks, and the ceiling's along-track bias | `rate50_diagnostic.py` (`primary_cfg` is 50 Hz, 1 s predict, 0.5 s execute, 0.24 s history; `control_cfg` is the 25 Hz / 0.5 s execute run that was not started) | `results/stage2/rate50_diagnostic.json` |

One-ball test seeds are `data/seeds.json`. The DeepSets ceiling was trained on `data/grid50_offset.npz`.

### Stage 2

| Claim | Script | Record |
| --- | --- | --- |
| DeepSets retrain, 36.1 mm and 6.0% at 10 mm | `train_deepsets_oneball.py` | `results/stage2/oneball_deepsets.json` |
| Order does not change the output | `test_deepsets.py` | `results/stage2/deepsets_unit.json` |
| Radius curve; 45 mm is the smallest mean over 80% | `radius_curve.py` | `results/stage2/radius_curve.json` |
| Expert on 100 scenes | `verify_two_ball.py`. `SEPARATION = 0.180`, so this check still uses 180 mm, four times 45 mm | `results/stage2/expert_verify.json` |
| 2,000 demos and 200 test scenes at 200 mm | `gen_two_ball.py`. `SEPARATION = 0.200` | `results/stage2/two_ball_data.json`; grid `data/grid50_two_ball.npz`; scenes `data/two_ball/test_scenes.json`; summary `data/two_ball/summary.json` |
| Squared-error and oracle training | `train_two_ball.py` | `results/stage2/two_ball_train.json` |
| Catch, miss, between-miss, and the decision table | `eval_two_ball.py`, `_reading` | `results/stage2/two_ball_compare.json` |
| MDN training, five diagonal Gaussians | `train_mdn.py` (`n_components=5`, `sigma_floor=0.01`) | `results/stage2/two_ball_mdn_train.json` |
| MDN rollouts, both selection rules, one ball and two | `eval_mdn.py` | `results/stage2/mdn_compare.json` |

Rejected scenes are already counted in `two_ball_data.json`: `separation_rejects_total` is 25 on the training scenes and `test_separation_rejects_total` is 4.

`train_two_ball.py` exits if the three seeds of a role hash to the same file, and writes the hashes into `two_ball_train.json`. Whether the squared-error robot's five catches are the same five scenes is in the per-seed rows of `two_ball_compare.json`.

The ball-swap check is already code. `test_deepsets.py` permutes the balls, and `eval_mdn.check_mdn` checks ball-swap on the mixture plus agreement between the torch net and the NumPy export. `check_mdn` runs at the start of `train_mdn.py` and is a synthetic batch, not a saved Stage 2 table.

### Tests in "What comes next"

These are not scripts yet, apart from the pieces named below.

0. **Premise.** `train_mdn.py` loads `data/grid50_two_ball.npz`, and the run note says two-ball data only. Confirm every training row has two balls. The control rolls `twoball_mse_s{0,1,2}` out on the one-ball scenes in `data/seeds.json`. That scoring pass still has to be written. The ceiling's one-ball scoring in `train_deepsets_oneball.py` is the pattern.
1. **Training net against the NumPy rollout, three stages.** `policy.export_numpy` writes the `.npz`. `eval_mdn.check_mdn` compares torch and NumPy on a made-up batch, and only the weights and the normalised means. A comparison on about 100 real training rows, reporting the logits, the weights after softmax, the normalised means, and the chunk after undoing the standardisation, still has to be written. The chunk after undoing the standardisation is what `MDNDriver` stores, from `act_mean` and `act_std`. The ceiling chunk for the same observation is `NumpyPolicy.predict_set` on `r25_p2_e0.5_hstack0.5s_d2000_deepsets_s{0,1,2}`.
2. **K = 1.** `Config.n_components` in `config.py`. `train_mdn.py` passes `n_components=5`.
3. **Hand-off.** The one-ball pattern is `diagnose.py` and `run_relative.handoff_episode`. An MDN hand-off still has to be written; the driver to hand to is `MDNDriver`.
4. **Weights on held-out states.** Training-row diagnostics are on the checkpoint under `meta.diagnostics`, and the last row of each seed is copied into `two_ball_mdn_train.json`.
5. **What the components stand for.** Expert phases are `align`, `drive`, `orient`, `done` on `BaseNavState` in `robotnik_arm_control.py`. `expert_phase_tests.py` is an earlier Stage 1 probe, not this tagging job.
6. **Base motion beside the chosen component.** `MDNDriver.act` already appends `choices` and `max_pi` on each replan. Speed and heading still have to be logged beside those.

## Glossary

| Term | Plain meaning |
| --- | --- |
| Expert | The scripted robot whose catches the learned robots copy |
| Backbone | The shared model design every robot uses, so only the part under test differs |
| Ceiling | The one-ball robot's catch rate (86.5% at 50 mm), the best expected with one ball |
| Catch radius | How close the hand must get to the ball to count as a catch (now 50 mm) |
| Miss distance | The closest the hand came to the ball, in mm |
| Squared-error (MSE) copier | A model trained to match the expert's average action; with two valid choices it tends to average them |
| Oracle | The same model, but told which ball to catch; shows the best possible with two balls present |
| MDN | Mixture density network: a model that outputs several possible actions with a weight on each, so it can keep "left" and "right" apart |
| Component | One of the MDN's candidate actions (here five), each with a weight saying how likely it is |
| Highest-weight / continuation rule | Two ways to pick a component at run time: the most likely one, or the one that best carries on the motion already under way |
| Hand-off test | The expert drives until a set time before the catch, then the learned robot takes over; separates errors that build up from errors made at once |
| DeepSets | A way of reading balls one at a time with the same small network and adding the results, so ball order can't matter |
| Between-miss | A miss where the hand ends nearer the midpoint of the two balls than either ball, the sign of averaging |
| Commitment time | How long after launch the robot settles on one ball for good |
| DART | Adding random noise while the expert drives, so the demos include recoveries from small mistakes |
| Off-target parking demos | Demos where the expert parks deliberately off its goal, so the robot learns to reach with the arm |
| Along-track / cross-track | Parking error forward or back along the robot's route / sideways to it |
