# Progress Report

Nov 5, 2026 · @Will

## Summary

The project has one result so far and one open problem.

**The learned robot catches 86.5% within 50 mm.** The robot with the DeepSets encoder misses by 36 mm on average, down from 80.5 mm at the end of the first sweep. Most of that gain came from one diagnosis: the base parked off-target, and the arm didn't reach to make up for it.

**Open problem: the robot stops slightly past its goal.** It ends ahead of the expert on 60% of throws (median +16.9 mm). Given the expert's states, its predictions near the stop match the expert's almost exactly. Holding each command for a whole tick puts the base 2.7 mm ahead. Neither the first chunk nor the chunk issued at the stop adds the rest: both finish within about 1 mm of the expert. In the closed-loop hand-off the command is still about 72 mm/s when the expert's command is already zero, the wheels follow it, and the base finishes 9.5 mm ahead.

The scripts, checkpoints and result files behind the numbers below are listed under Files.

## The project

The project trains a mobile robot with an arm to catch a thrown ball by copying a scripted expert (learning by copying, also called behaviour cloning).

The question so far is how close a learned robot can get to the expert's catches, and what limits it.

The work so far has two stages:

1. **Stage 0: the expert.** A scripted robot that drives, parks and reaches to catch a ball almost perfectly. It provides all the training examples. Done.
2. **Stage 1: the learned robot.** A robot that copies the expert, made as accurate as reasonably possible. Done, with one stopping problem still under test.

One problem remains: why the robot stops slightly late.

## Stage 1: one-ball catching

Stage 1 cut the one-ball miss from 80.5 mm to 38.2 mm by finding and fixing one cause. The learned robot parked its base about 54 mm from where the expert would, and its arm didn't reach to make up the difference. So the hand missed by roughly the parking error.

| Fix | Mean miss | What it did |
| --- | --- | --- |
| Starting point | 80.5 mm | Arm ignores parking errors |
| Ball position measured from the robot's base | 65.7 mm | Part of the forward/back error corrected |
| 1,000 demos with deliberately off-target parking | 38.2 mm | Sideways error fully corrected |

The second fix worked because the expert, when parked off-target, simply reaches further with the arm. The robot had never seen this, because the expert always parked perfectly. Showing it off-target parks taught it the reach.

The forward/back error is still only partly corrected (0.43), and it makes up most of what's left. The earlier guess was that the forward/back offsets in the demos were too small; a direct test has now ruled that out (below). Two other approaches, generic noise injection (DART) and a first attempt at expert correction (DAgger), did not help.

### The forward parking bias

The learned robot ends up ahead of the expert on most throws. The lead is created in the last open-loop chunk of the approach, and it stays because the expert never demonstrates reversing.

First measurements, on the older 80 mm policy:

- **During the drive the two are level.** Two seconds before the expert stops, the along-track gap is about zero (median −1 mm).
- **On the robot, the learner brakes slightly late.** The expert brakes from 0.25 m/s to 0 over about the last second. In closed loop the learner follows the same ramp but about 4.5 mm/s faster, so its zero crossing comes late. Later tests show the network's own predictions match the expert's ramp; the extra speed appears only once the commands are carried out (see "Where the late stop is made").
- **The late stop lands inside a chunk it has already committed to.** The robot predicts 2s and executes 1 s (25 steps) before it looks again. The expert's command reaches zero, on average, 52% of the way through the chunk that is open. At that instant the learner is still commanding 7.4 cm/s, the speed fixed when the chunk was issued. For the rest of the chunk it averages 2.1 cm/s while the expert is already stopped.
- **The next chunk stops but doesn't back up.** By the next replan the learner is 11 mm ahead, and 70% of episodes are already ahead.

One episode, with times relative to the expert's stop:

| Time | Learner command | Expert command | Along-track gap |
| --- | --- | --- | --- |
| −0.4 s | 0.140 m/s | 0.121 m/s | −5 mm |
| 0 s | 0.086 m/s | 0.081 m/s | +3 mm |
| +0.2 s (chunk step 23) | 0.041 m/s | 0 | +14 mm |
| +0.4 s (new chunk) | 0.002 m/s | 0 | +18 mm |

The robot never reverses because the expert never does. Across 200 training demos (84,832 policy ticks), the expert's net forward command is never negative; the minimum is exactly 0. Wheels spin backward during turns, but the base never drives backward. That is how the controller is written: forward speed is multiplied by max(0, cos(heading error)), and once the base has passed the goal it turns in place instead of reversing. The rule is `base_controller` in `robotnik_arm/robotnik_arm_control.py`.

*Plain version:* the robot stops a fraction late, and nothing in its examples ever shows it backing up. So an overshoot is permanent; an undershoot gets fixed by driving on.

**The current ceiling has it too.** Comparing 50 learner rollouts with the expert on the same seeds, plus 200 expert demos, confirms the pattern. The learner is level until about 2 s before the stop. The lead appears in the last 1 s chunk, the new chunk stops but doesn't back up (0.5% reverse commands after that), and the lead is still there at the catch about 9 s later (about 18–25 mm, 70% ahead). Over 3 seeds and the 200 test scenes, the DeepSets ceiling ends ahead on 60% of episodes, with a median signed along-track error of +16.9 mm (mean miss 36 mm, 87% caught at 50 mm).

**Possible fixes, and where each stands:**

- **Re-plan more often near the stop.** Tried at 50 Hz with 0.5 s chunks, now on three seeds. Seeds 0 and 2 miss less; the three-seed mean is only a few millimetres under the ceiling, inside the seed spread, and the bias didn't move (below).
- **Demos parked ahead of the goal**, so the arm learns to reach back. Tried as part of the learner-matched offsets below; it didn't help.
- **Smooth each command within its tick.** Ramp linearly between chunk steps instead of holding each step for a whole tick. Untested. The hold is 2.7 mm of the lead, so smoothing it waits until the 9 mm is accounted for (see Next steps).
- **Let the expert reverse when past the goal**, so the demos contain it. Untested.

### Shorter chunks

The hypothesis: if the robot commits to less, the overshoot tail should shrink. The 50 Hz version predicts 1 s, executes 0.5 s and uses 0.24 s of history, with the same DeepSets model shape. Seeds 1 and 2 are now scored on the same 200 test scenes. The ceiling column is the three-seed mean already given above.

|  | 25 Hz ceiling | 50 Hz seed 0 | 50 Hz seed 1 | 50 Hz seed 2 | 50 Hz, 3 seeds |
| --- | --- | --- | --- | --- | --- |
| Mean miss | 36.0 mm | 24.7 mm | 42.1 mm | 26.7 mm | 31.2 mm |
| Median miss | 31.4 mm | 19.3 mm | 23.8 mm | 22.7 mm | 21.9 mm |
| Catch at 50 mm | 87% | 94% | 91% | 93.5% | 93% |
| Signed along-track | +16.9 mm | +19.4 mm | +9.0 mm | +20.4 mm | +16.3 mm |
| Fraction ahead | 60% | 67% | 57% | 69% | 64% |

The miss improved on seeds 0 and 2, and not on seed 1. Seed 1's median is 23.8 mm, so its 42.1 mm mean is a few bad episodes. Across three seeds the mean miss is 31.2 mm, about 5 mm under the ceiling, and the seeds themselves spread by about 8 mm. That is smaller than the 10 mm gain seed 0 suggested.

The stop did not move. The three-seed along-track lead is +16.3 mm, with 64% of episodes ahead, against +16.9 mm and 60% for the ceiling. Seed 1 finishes less far ahead (+9.0 mm, 57%) and seed 2 further ahead (+20.4 mm, 69%). The 25 Hz control with a 0.5 s execute was not run. This is not a better backbone.

### Where the late stop is made (new)

Three tests on the DeepSets ceiling narrow the late stop down to the execution of the last chunk. The network's plan is right; the robot's motion isn't.

**1. Given the expert's state, the network brakes on time.** On the 200 test scenes, each ceiling seed was shown the expert's own state in the 2 s before the base stops, and its immediate forward-speed command was compared with the expert's.

| Seed | Mean residual | 95% interval |
| --- | --- | --- |
| 0 | −0.39 mm/s | −0.57 to −0.21 |
| 1 | −0.14 mm/s | −0.35 to +0.07 |
| 2 | +0.36 mm/s | +0.09 to +0.63 |
| Pooled | −0.06 mm/s | −0.20 to +0.09 |

- The median residual is 0.00 mm/s, and 49.5% of episodes are positive. The same residual 2–4 s earlier is +0.03 mm/s, and the yaw residual is −0.0001 rad/s.
- The braking ramp matches. Both curves hold 250 mm/s until about 1.1 s before the stop, then fall together, within about ±2 mm/s.
- The only departure is the stop sample: the expert is already at 0 and the prediction is still at 28 mm/s. That is one 20 ms tick, about 0.6 mm of travel.
- Seed 2 leans positive by a third of a millimetre per second, not the 4.5 mm/s seen in closed loop.

The expert's labels here are one-sided (never negative), but the prediction doesn't sit above them. So "averaging one-sided labels pushes the speed up" is ruled out for this network.

**2. Chunks issued further ahead are one tick late, and no later.** On the same held-out expert states, the predicted stop time was compared with the true one for chunks issued now and 0.25, 0.5 and 1.0 s earlier.

- The median stop error is 40 ms at every one of those lookaheads, and about 95% of stops are late. Past the executed second it steps to 80 ms and stays there.
- The speed the chunk assigns to the true stop instant is 32–37 mm/s across the whole first second. So the lateness is a one-tick hedge, not a ramp that gets worse further ahead.
- The integral of predicted minus expert forward speed over the chunk open at the stop is +0.3, −0.3 and +0.1 mm on the three seeds, and each interval covers zero. The part at or after the true stop is +1.3 mm. A chunk issued exactly 1 s before the stop commands about −1 mm over the second it executes.
- That flat 40 ms is why re-planning at 50 Hz didn't move the bias. The error is fully present at a quarter-second of lookahead, which is the average age of a command in the 50 Hz run. The 50 Hz model is 20 ms late at 0 and 0.25 s and 40 ms late at 0.5 s, with an open-chunk lead of +0.1 mm.

**3. The lead appears even from the expert's own state.** The expert drove until 2 s, 1 s or 0.5 s before its own stop, then the policy took over. The along-track gap at the switch was 0 on all 200 episodes.

| Hand-off before the stop | At the catch | 1 s after the stop |
| --- | --- | --- |
| 2.0 s | +6.2 mm, 55% ahead | +8.6 mm |
| 1.0 s | +7.2 mm, 55% ahead | +9.5 mm |
| 0.5 s | +0.9 mm, 54% ahead | +1.7 mm |

One second after the stop the three seeds agree closely. With a 1–2 s hand-off they sit at +7.2 to +11.0 mm, with intervals of about ±1–2 mm and 68–94% of episodes ahead. With a 0.5 s hand-off they sit at +1.6 to +1.8 mm. At the catch the per-seed values are less settled.

**What the three tests show.** The lead doesn't depend on how the learner reached the brake; it appears from a perfect state. It is produced in the last second, the chunk the robot executes through the stop, and it is mostly gone when the expert still controls the base at 0.5 s.

The sizes don't line up. The chunk issued from the expert state commands about 0 mm of extra travel at the sample instants, yet the base finishes about 9 mm ahead. The replay below is that gap with the network taken out.

**Holding each command for a whole tick accounts for 2.7 mm, not the 9 mm.** The expert runs at 1 kHz. From its state 1 s before the stop, the commands it had just logged were applied again, open-loop, on all 200 test scenes. The hold grid starts at that instant. The along-track gap is the replay base minus the expert base, in the expert's route frame.

| Replay | 1 s after the stop | At the catch | Extra travel in the commands |
| --- | --- | --- | --- |
| 1 kHz, as the expert applied them | 0.0 mm | 0.0 mm | 0.0 mm |
| Held 40 ms, as at 25 Hz | +2.7 mm | +2.5 mm | +2.7 mm |
| Held 20 ms, as at 50 Hz | +1.3 mm | +1.4 mm | +1.3 mm |

The 1 kHz replay matches the expert on every scene, so the replay is deterministic. The 40 ms hold finishes ahead on 99% of scenes. The interval on the mean is about ±0.1 mm. The 20 ms hold is half of that, which is what a hold should do. The base moves by the amount the held commands asked for (2.7 mm asked, 2.7 mm travelled), so on these commands the wheels are not adding a further lag.

The earlier estimate was 6–7 mm: half a tick of a 230 mm/s² ramp, over 1.1 s, plus one late tick. The logged commands do not integrate to that. Held for 40 ms they are worth 2.7 mm, and the hand-off from the same state is still about 9.5 mm. Smoothing the hold would remove the 2.7 mm. The chunk replay below shows where the rest appears.

The 50 Hz result fits this size. Halving the hold removes about 1.3 mm, and the three 50 Hz seeds spread much wider than that at the catch.

**The policy's chunk, played open-loop, does not contain the 9 mm.** From the same expert state, one second before the stop, each ceiling seed was shown the expert's history and queried once. The whole 2 s prediction was held for 40 ms per step. There was no replan. The position at the stop is the end of the 1 s the robot would execute; the position one second after the stop is the end of the prediction.

|  | At the stop | 1 s after the stop |
| --- | --- | --- |
| Expert commands, held 40 ms | +2.7 mm | +2.7 mm |
| Policy chunk, seed 0 | +2.5 mm | +0.1 mm |
| Policy chunk, seed 1 | +1.4 mm | −1.5 mm |
| Policy chunk, seed 2 | +1.9 mm | −0.1 mm |
| Policy chunk, 3 seeds | +1.9 mm | −0.5 mm |
| Policy closed loop, from the same state |  | +9.5 mm |

At the stop the chunk is the hold again: the base is 1.9 mm ahead, and the commands over that second asked for 1.8 mm. One second later, with the rest of the prediction played and no replan, the base is back with the expert (53%, 43% and 48% of scenes ahead). The closed-loop hand-off from this state is still +9.5 mm at that same instant.

**The chunk issued at the stop does not contain the 9 mm either.** The first chunk was left running until the stop, so the base was the 1.9 mm ahead measured above (seeds +2.5, +1.4 and +1.9 mm, the same scenes as the chunk replay). The policy was queried there, once, and that second chunk was held for 40 ms per step. There was no further replan. One second later is the instant at which the closed-loop hand-off is +9.5 mm.

|  | At the replan | 1 s later | Command, that second |
| --- | --- | --- | --- |
| Seed 0 | +2.5 mm | +1.3 mm | +3.6 mm |
| Seed 1 | +1.4 mm | −1.6 mm | +1.9 mm |
| Seed 2 | +1.9 mm | −0.3 mm | +2.5 mm |
| 3 seeds | +1.9 mm | −0.2 mm | +2.7 mm |
| Closed-loop hand-off |  | +9.5 mm |  |

The second chunk asks for 2.7 mm more travel than the expert and the base does not take it. Seeds finish 57%, 40% and 44% of scenes ahead. The 9 mm is not in this plan.

**The wheels follow the command. The hand-off's command has not stopped.** The same hand-off was run again, expert at 25 Hz until one second before its stop, then each ceiling seed. Commanded body speed and actual body speed were logged every millisecond. The along-track gap one second after the stop came back as +11.0, +8.2 and +9.4 mm, which is the published +9.5 mm.

|  | Actual − command, the second before | Command at the stop | Actual at the stop | Coast over the next second | Time to fall under 20 mm/s |
| --- | --- | --- | --- | --- | --- |
| Expert | +1.9 mm/s | 0 | 81 mm/s | 4.9 mm | 71 ms |
| Policy, seed 0 | +2.0 mm/s | 75 mm/s | 76 mm/s | 8.8 mm | 151 ms |
| Policy, seed 1 | +2.0 mm/s | 70 mm/s | 74 mm/s | 7.2 mm | 138 ms |
| Policy, seed 2 | +2.0 mm/s | 73 mm/s | 77 mm/s | 7.8 mm | 134 ms |

Over the braking second the base runs about 2 mm/s faster than the command it was given, for the expert and for the policy alike. At the stop the expert's command is zero and the base is still at 81 mm/s; that residual falls through 20 mm/s in 71 ms and is worth 4.9 mm of coast. The policy's command at that instant is still about 72 mm/s, and the base is within a few mm/s of it. The extra coast in the next second is about 3 mm. The wheels are not where the 9 mm comes from. The policy is still asking to drive.

*Plain version:* keeping each speed for 40 ms makes the robot stop 2.7 mm late. The learned robot's plan, issued from the expert's position and carried out without a second look, does not stop any later than that, and neither does the next plan. The 9 mm lead is in the closed-loop hand-off, where the learned robot is still commanding about 72 mm/s after the expert has commanded a stop, and the base does what it is told.

### Testing the "offsets too small" explanation

If the forward/back correction is stuck because the demos' parking offsets were too small or the wrong shape, then demos matched to where this learner actually parks should fix it. They didn't; the correction got worse.

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

The learner stops about 9 s before the catch, so its forward/back offset is a parked one, just like the demos'. Why the arm corrects sideways errors (0.81) but only part of the forward/back error (0.43), even after matched demos, is still open. It is not because a good forward/back response looks different from a reach. The expert, slid forward or back after it parks, catches by reaching (below).

### DeepSets encoder and catch radius

**New ball encoder.** The robot now reads the ball through DeepSets: a small network encodes the ball before the rest of the policy uses it.

Because this changes the backbone, the robot was retrained with it (3 seeds, the same 200 test scenes). It came out slightly better than the old version, so the encoder costs nothing:

| Measure | Before DeepSets | With DeepSets |
| --- | --- | --- |
| Mean miss | 38.2 mm | 36.1 mm |
| Catch rate at 10 mm | 4.2% | 6.0% |

**Catch radius.** A tiny 10 mm target makes almost every model score near zero, which hides real differences. So a catch now counts if the hand comes within a set radius of the ball. The smallest radius where the DeepSets robot catches over 80% is 45 mm (80.3%). The chosen radius is 50 mm (86.5%); see "The 50 mm decision" below.

### The 50 mm decision

The radius is 50 mm, not the 45 mm that the "smallest radius over 80%" rule gives. This puts the ceiling at 86.5% rather than 80.3%, leaving more room above and below it to see differences between models.

The "smallest radius over 80%" rule is the docstring of `radius_curve.py`, which still describes that rule. It should be updated to record the 50 mm choice.

## Next steps

The late stop is in the closed-loop command. Played open-loop, the chunk issued at the stop finishes within a millimetre of the expert. In the hand-off that same instant is 9.5 mm ahead, the policy is still commanding about 72 mm/s, and the base is doing what the command says. The wheels add about 2 mm/s to the expert and the policy alike.

1. **The along-track gap at the stop, in the hand-off.** The extra coast after the stop is only about 3 mm (about 8 mm for the policy, 4.9 mm for the expert). The along-track gap at the stop itself was not logged. That number says how much of the 9.5 mm is already there when the expert's command hits zero.
2. **The chunk the hand-off actually issues.** The open-loop second chunk was queried from the 1 kHz replay, which is 1.9 mm ahead at the stop. The hand-off queries on its own 25 Hz state. Logging the forward speed of that chunk, at the expert's stop, checks whether it is the command that is still at about 72 mm/s.

Smoothing the 40 ms hold would address the 2.7 mm. It is not the 9 mm lead, so the smoothing waits.

**Expert forward/back slide, done.** The parked expert was slid 50 and 100 mm forward and back on the same 20 training scenes used for the sideways slide, at 1 kHz. All 80 catches landed, all of them with the base staying parked. The miss stays under 0.1 mm.

| Slide | Arm along the route | Base |
| --- | --- | --- |
| +50 mm (forward) | 29 mm back | 0.3 mm |
| −50 mm | 71 mm forward | 0.3 mm |
| +100 mm | 79 mm back | 0.5 mm |
| −100 mm | 121 mm forward | 0.4 mm |

The four arm numbers are the slide plus one 21 mm forward reach (50−21, 50+21, 100−21, 100+21). The ball at the catch is still on the planned intercept, and the catch is 19 ms before the planned time on every slide. The meeting point does not move. The sideways slide was a reach too: the arm undid 44–56 mm of a 50 mm slide and 94–106 mm of a 100 mm slide. A good forward/back response is an arm reach, so the correction measure is not missing a catch made by timing.

## Things to watch

**Small differences in catch rate are noise.** With 200 test scenes and a catch rate near 86%, chance alone moves the rate by about 2.4 points from one run to the next. Pooling all three seeds (600 episodes) narrows that, but a gap of under about 5 points between two models should not be read as real without more test scenes.

**Checks from the plan:**

- **Step 0a, answered.** Larger forward/back offsets were tested, matched to the learner's own parking, and made things worse (see Stage 1).

## Files

Ablation scripts live in `robotnik_arm/experiments/ablation` and are run from that directory. Results are under `results/`, data under `data/`. Checkpoints (`results/models/*.pt` and the matching `.npz`) are gitignored. Each logged run also has a copy under `results/stage2/runs/`, written by `stage2_log.py`.

`STATUS.md`, `STAGE1_RESULTS.md` and `results/stage2_config.json` describe the Stage 1 winner at 80.6 mm, before DeepSets. The ceiling in this report is the DeepSets retrain.

### Shared code

| Piece | Where |
| --- | --- |
| Expert base, including the no-reverse rule | `robotnik_arm/robotnik_arm_control.py`, `base_controller` |
| Sim, 42-D observation, rollout | `catch_env.py` (`CatchEnv`, `OBS_NAMES`, `rollout`, `ExpertDriver`) |
| Hyperparameters and checkpoint names | `config.py`, `Config.name` |
| Per-ball features (22-D ball, 33-D robot) | `ballset.py` |
| Torch policies | `policy.py`, `DeepSetsPolicy` |
| Training step | `train.py` (`deepsets_forward`) |
| NumPy rollout | `np_policy.py`, `SetPolicyDriver` |
| Expert collection | `robotnik_arm/collect_expert.py`, `robotnik_arm/docs/DATA_COLLECTION.md` |

Checkpoint names follow `Config.name`. The ones this report uses:

| Robot | Files under `results/models/` |
| --- | --- |
| Stage 1 start, 80.5 mm | `r25_p2_e0.5_hstack0.5s_d1000_s{0,1,2}.{pt,npz}` |
| Base-frame ball, 65.7 mm | `r25_p2_e0.5_hstack0.5s_d1000_rel_add_s{0,1,2}` (see `run_relative.py`) |
| Off-target parking, 38.2 mm | `r25_p2_e0.5_hstack0.5s_d2000_rel_add_s{0,1,2}` |
| DeepSets ceiling, 36.1 mm | `r25_p2_e0.5_hstack0.5s_d2000_deepsets_s{0,1,2}` |
| Goal-shift and planner retrains | `..._deepsets_matched_s{0,1,2}`, `..._deepsets_planned_s{0,1,2}` |
| 50 Hz, 0.5 s execute | `r50_p1_e0.5_hstack0.24s_d2000_deepsets_s{0,1,2}` |

### Stage 1

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
| Paths shared by that comparison | `variety_data.py` |  |
| Train and the table that missed the bar | `train_offset_variety.py` | `results/offset_variety_compare.json`. It only reads `oneball_deepsets.json` |
| 50 Hz, 0.5 s chunks, seed 0, and the ceiling's along-track bias | `rate50_diagnostic.py` (`primary_cfg` is 50 Hz, 1 s predict, 0.5 s execute, 0.24 s history; `control_cfg` is the 25 Hz / 0.5 s execute run that was not started) | `results/stage2/rate50_diagnostic.json` |
| 50 Hz seeds 1 and 2 | `rate50_seeds12.py` (same config as `primary_cfg`; trains seeds 1 and 2, then scores them) | `results/stage2/rate50_s1_s2.json` |
| Stop residual, stop lookahead, and hand-off before the stop | Not yet listed | Not yet listed |
| Open-loop replay of the expert's commands, 1 kHz / 40 ms / 20 ms | `hold_replay.py` | `results/stage2/hold_replay.json` |
| Open-loop replay of the ceiling's chunk from 1 s before the stop | `chunk_replay.py` | `results/stage2/chunk_replay.json` |
| Open-loop replay of the chunk issued at the stop | `replan_replay.py` | `results/stage2/replan_replay.json` |
| Commanded speed against actual speed in the 25 Hz hand-off | `command_vs_actual.py` | `results/stage2/command_vs_actual.json` |
| Expert slid 50 and 100 mm forward and back after parking | `along_parking.py` | `results/stage2/along_parking.json` |

One-ball test seeds are `data/seeds.json`. The DeepSets ceiling was trained on `data/grid50_offset.npz`.

### DeepSets and catch radius

| Claim | Script | Record |
| --- | --- | --- |
| DeepSets retrain, 36.1 mm and 6.0% at 10 mm | `train_deepsets_oneball.py` | `results/stage2/oneball_deepsets.json` |
| Radius curve; 45 mm is the smallest mean over 80% | `radius_curve.py` | `results/stage2/radius_curve.json` |

### Tests in Next steps

The command-hold replay is `hold_replay.py`, the chunk replay is `chunk_replay.py`, the second-chunk replay is `replan_replay.py`, the commanded-against-actual log is `command_vs_actual.py`, and the forward/back slide is `along_parking.py`.

## Glossary

| Term | Plain meaning |
| --- | --- |
| Expert | The scripted robot whose catches the learned robots copy |
| Backbone | The shared model design every robot uses, so only the part under test differs |
| Ceiling | The current best learned robot (86.5% caught at 50 mm) |
| Catch radius | How close the hand must get to the ball to count as a catch (now 50 mm) |
| Miss distance | The closest the hand came to the ball, in mm |
| Hand-off test | The expert drives until a set time before the catch, then the learned robot takes over; separates errors that build up from errors made at once |
| DeepSets | The ball encoder: a small network that reads the ball's features before the rest of the policy uses them |
| DART | Adding random noise while the expert drives, so the demos include recoveries from small mistakes |
| Off-target parking demos | Demos where the expert parks deliberately off its goal, so the robot learns to reach with the arm |
| Along-track / cross-track | Parking error forward or back along the robot's route / sideways to it |
| Command hold | Applying each chunk step unchanged for a whole policy tick (40 ms at 25 Hz), instead of easing it towards the next step |
