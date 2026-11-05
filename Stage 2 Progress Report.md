# Stage 2 Progress Report

Nov 5, 2026 · @Will

## Summary

Stage 2 is ready to generate data. The one-ball robot now uses the DeepSets ball encoder that Stage 2 needs, and it is no worse for it: mean miss 36.1 mm, against 38.2 mm before. The catch radius is set at 50 mm, where the one-ball robot catches 86.5% of balls. That 86.5% is the bar the two-ball robots will be measured against.

The random-choice expert passed its check on 100 two-ball scenes. It caught every ball, picked each slot about equally often, and never changed its mind. The next job is to generate 2,000 two-ball episodes with the balls landing at least 200 mm apart, then train the first two-ball robots.

## The big picture

The project tests one known weakness of learning by copying. Show a robot two balls, and an expert that catches either one at random. A standard copier trained by squared error learns the average of what the expert did. The average of "go left" and "go right" is "go to the middle", which catches nothing.

The question is whether that failure really shows up here, and whether a model that can hold several answers at once (a mixture density network, or MDN) fixes it.

The project runs in three stages:

1. **Stage 0: the expert.** A scripted robot that drives, parks and reaches to catch a ball almost perfectly. It provides all the training examples. Done.
2. **Stage 1: the one-ball backbone.** A learned robot that copies the expert with one ball, made as accurate as reasonably possible. Every Stage 2 model reuses it, so any difference between them comes from the two-ball problem alone. Done.
3. **Stage 2: two balls.** Four robots are compared on the same backbone: the one-ball ceiling, a two-ball squared-error copier, an oracle that is told which ball to catch, and the MDN. In progress.

&#91;embedded content: Stage 2 progress · 5 done, 2 now, 3 later\]

Everything needed before the two-ball robots can be trained is in place; the data comes next.

## How Stage 1 ended

Stage 1 cut the one-ball miss from 80.5 mm to 38.2 mm by finding and fixing one cause. The learned robot parked its base about 54 mm from where the expert would, and its arm didn't reach to make up the difference. So the hand missed by roughly the parking error.

| Fix | Mean miss | What it did |
| --- | --- | --- |
| Starting point | 80.5 mm | Arm ignores parking errors |
| Ball position measured from the robot's base | 65.7 mm | Part of the forward/back error corrected |
| 1,000 demos with deliberately off-target parking | 38.2 mm | Sideways error fully corrected |

The second fix worked because the expert, when parked off-target, simply reaches further with the arm. The robot had never seen this, because the expert always parked perfectly. Showing it off-target parks taught it the reach.

The forward/back error is still only about 43% corrected, and it makes up most of what's left. The likely reason is that the forward/back parking offsets in the demos were too small. Two other approaches, generic noise injection (DART) and a first attempt at expert correction (DAgger), did not help.

## Stage 2 so far

Three of the first steps in the Stage 2 plan are done, and all three passed.

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

## The 50 mm decision

The radius is 50 mm, not the 45 mm the plan's "smallest over 80%" rule gives. This puts the ceiling at 86.5% rather than 80.3%, leaving more room above and below it to see differences between models.

It changes two things:

- **Balls must now land at least 200 mm apart** (4 × 50 mm), up from 180 mm. If the balls land close together, a robot that aims at the midpoint can still get within 50 mm of one ball. That would hide the very failure Stage 2 is looking for.
- **The expert check still holds.** The expert caught everything at 45 mm, and a 50 mm target is easier, so it passes at 50 mm too.

The plan file (stage2\_plan.txt) still says "smallest radius over 80%". It should be updated to record 50 mm and the 200 mm separation.

## What comes next

The next step is generating the two-ball data, then training the first two two-ball robots.

1. **Generate 2,000 two-ball episodes** with the random-choice expert. Use the same recipe as the one-ball data: half with noise injected (DART), off-target parking, labels taken from the commands actually sent. Keep only scenes where the balls land at least 200 mm apart. Also make 200 new two-ball test scenes that the robots never train on.
2. **Train two robots on two-ball data only**, 3 seeds each:
   - **Two-ball MSE:** the same model as the ceiling, which has to work out for itself which ball to chase.
   - **Oracle:** the same model, but told which ball to catch. It shows how well a robot can do with two balls around once the choice is made for it.
3. **Compare them with the 86.5% ceiling** using catch rate, miss distance, how often misses land between the balls, how often the robot switches target, and how early it commits.
4. **Build the MDN if the result calls for it**, then iterate until it matches the oracle.

The comparison in step 3 decides what happens next:

| What we see | What it means | Next |
| --- | --- | --- |
| Oracle near 86.5%, MSE well below, many misses between the balls | The averaging failure is real | Build the MDN |
| Oracle well below 86.5% | A second ball hurts for some other reason | Fix that before the MDN |
| MSE near 86.5% | The robot settles its choice early, so averaging barely bites | Rethink the setup |

The target for the MDN is the oracle, not the ceiling. The oracle shows the best possible with two balls present; the one-ball number may be out of reach for reasons that have nothing to do with choosing.

## Things to watch

**Small differences in catch rate are noise.** With 200 test scenes and a catch rate near 86%, chance alone moves the rate by about 2.4 points from one run to the next. Pooling all three seeds (600 episodes) narrows that, but a gap of under about 5 points between two models should not be read as real without more test scenes.

**The separation rule changes which scenes appear.** Throwing out scenes where the balls land less than 200 mm apart could favour certain throws, such as long or wide ones. Log how many scenes are rejected. If it's a large share, check that the kept throws still look like the one-ball throws.

**Compare like with like.** The 86.5% ceiling was measured on one-ball test scenes, but the two-ball robots will be tested on new two-ball scenes. To compare fairly, also run the one-ball robot on each two-ball test scene with only one ball in it (the ball the oracle is told to catch). That gives a ceiling on exactly the throws the two-ball robots face.

**Checks from the plan with no result recorded yet:**

- Step 0a: whether the forward/back offset round (larger along-track parking offsets) was run or skipped.
- Step 0c: the ball-swap test, showing that swapping the two balls' order leaves the output unchanged.
- Step 0e: that the three seeds' checkpoints are genuinely different files.

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
| DeepSets | A way of reading balls one at a time with the same small network and adding the results, so ball order can't matter |
| Between-miss | A miss where the hand ends nearer the midpoint of the two balls than either ball, the sign of averaging |
| Commitment time | How long after launch the robot settles on one ball for good |
| DART | Adding random noise while the expert drives, so the demos include recoveries from small mistakes |
| Off-target parking demos | Demos where the expert parks deliberately off its goal, so the robot learns to reach with the arm |
| Along-track / cross-track | Parking error forward or back along the robot's route / sideways to it |
