# Stage 2 Progress Report

Nov 5, 2026 · @Will

## Summary

The main Stage 2 runs are done, with one clear result and one failure.

**The averaging failure is real.** Told which ball to catch, the oracle catches 86.2%, the same as the one-ball robot on those throws (81.8%) within noise. The squared-error robot, left to choose for itself, catches 2.5%, and half its misses end between the two balls.

**The mixture density network (MDN), the proposed fix, does not catch at all.** It manages 0.5–1.0% on two balls, and only 0.2% on one ball, where there is nothing to choose between. So its failure is not about choosing a ball; something in the MDN itself breaks once it drives the robot. It fits its training data well, so the next job is a short set of diagnostic tests to find where it breaks.

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

The robot never reverses because the expert never does. Across 200 training demos (84,832 policy ticks), the expert's net forward command is never negative; the minimum is exactly 0. Wheels spin backward during turns, but the base never drives backward. That is how the controller is written: forward speed is multiplied by max(0, cos(heading error)), and once the base has passed the goal it turns in place instead of reversing.

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

A one-ball miss of 1.6–1.7 m means the robot barely gets to the ball. *Plain version:* the MDN learned the examples, but when it drives the robot itself, something goes wrong long before the catch. Because it fails with one ball, the cause isn't the choice between balls (full record: `mdn_compare.json`).

## The 50 mm decision

The radius is 50 mm, not the 45 mm the plan's "smallest over 80%" rule gives. This puts the ceiling at 86.5% rather than 80.3%, leaving more room above and below it to see differences between models.

It changes two things:

- **Balls must now land at least 200 mm apart** (4 × 50 mm), up from 180 mm. If the balls land close together, a robot that aims at the midpoint can still get within 50 mm of one ball. That would hide the very failure Stage 2 is looking for.
- **The expert check still holds.** The expert caught everything at 45 mm, and a 50 mm target is easier, so it passes at 50 mm too.

The plan file (stage2\_plan.txt) still says "smallest radius over 80%". It should be updated to record 50 mm and the 200 mm separation.

## What comes next

The two-ball comparison landed on the first row of the plan's decision table: the oracle matched the ceiling, and the squared-error robot fell far below with many misses between the balls. That called for the MDN, which was built and fails. The next step is finding out why. The tests are ordered cheapest first, and the first three need about an hour.

1. **MDN with one component (K = 1).** A one-component mixture is a squared-error model that also learns a spread. On one ball it should behave like the ceiling. If it also fails, the problem is in the MDN code path, not in having several components.
2. **Check the run-time code against training.** Feed the same observations to the training model and to the NumPy rollout version, and compare the chunk each produces: weights, component means, and undoing the standardisation. A 1.7 m miss is the size a bug would produce.
3. **Hand-off test.** Let the expert drive until 0.5, 1, 2 and 4 s before the catch, then hand over to the MDN.
   - If it misses even from 0.5 s, it can't act well from good states, which points to a bug or a poor fit.
   - If it is fine late and fails early, the error builds up over the episode, as in Stage 1, made worse by switching components.
4. **Check the weights on unseen states.** On held-out expert states, compare the error of the highest-weight component (what the robot uses) with the error of the best-matching component. If the highest-weight one is often wrong, the weights are what's failing, not the components.
5. **What do the components stand for?** With one ball there's no ball choice, yet four components get used. Tag every training row with the expert's phase (align, drive, orient, done) and whether it had noise added. Then see which component wins for each tag.
   - If components line up with phases, the MDN is choosing between "spin", "drive" and "hold" at moments where the squared-error model averaged them into smooth driving. That would echo the DAgger failure.
6. **Watch the robot.** In rollouts, log the base speed and heading alongside the chosen component. This shows whether the robot spins, stalls or holds when the component switches.

If tests 1–3 point to a bug, fix it and rerun. If the MDN is working as designed and still fails, the fallbacks from the plan are:

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
