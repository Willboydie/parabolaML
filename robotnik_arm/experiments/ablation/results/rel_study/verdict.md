# Flat rel_add study

Encoder is flat. Features are rel_add. DeepSets weights are not inputs.

Ablation winner: `r25_p2_e0.5_hnone_d1000_rel_add` catch 0.032 miss 59.1 mm.

## Predictions against the sweep

| Factor | Prediction from the baseline | Outcome | Best level | Catch | Miss mm | Baseline miss mm |
| --- | --- | --- | --- | ---: | ---: | ---: |
| rate | higher_rate_lowers_miss | confirmed | r25_p2_e0.5_hnone_d1000_rel_add | 0.032 | 59.1 | 113.3 |
| history | not_indicated | no prediction | r10_p2_e0.5_hgru2s_d1000_rel_add | 0.007 | 111.7 | 113.3 |
| predict_horizon | shorter_predict_lowers_miss | refuted | r10_p5_e0.2_hnone_d1000_rel_add | 0.013 | 94.1 | 113.3 |
| execute_window | not_indicated | no prediction | r10_p2_e0.5_hnone_d1000_rel_add | 0.007 | 113.3 | 113.3 |
| short_chunk | short_chunk_increases_miss | confirmed | r10_p2_e0.5_hnone_d1000_rel_add | 0.007 | 113.3 | 113.3 |

A prediction is confirmed only by the inequality written in hypotheses.json.
A factor marked not_indicated was still swept. It has no confirmation to claim.

## Winner, high-amplitude DART, parking offsets

| Model | Miss mm | Catch at 10 mm | Along slope | Sideways slope |
| --- | ---: | ---: | ---: | ---: |
| winner | 59.1 ± 5.5 | 0.032 ± 0.010 | 0.62 | -0.23 |
| dart | 89.5 ± 9.6 | 0.023 ± 0.014 | 0.62 | 0.33 |
| offset | 27.9 ± 0.5 | 0.098 ± 0.016 | 0.45 | 1.00 |

Offset demos lower the miss and raise the parking-compensation slope relative to high-amplitude command noise. The noise matches the winner's per-dimension command error and does not move the parking goal.

DART and the offset set each add 1,000 episodes on the offset dataset's training seeds.
The ablation winner stays the 1,000-demo model.
