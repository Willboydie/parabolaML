"""Stage 0: can the expert still catch when it may only act at each policy rate?"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / 'data'
checks = json.loads((DATA / 'expert_rate_check.json').read_text())
lines = ['| Expert rate | Episodes | Catch rate (10 mm) | Mean miss mm | Median miss mm | Max miss mm | Catch at 5 mm | Catch at 2 mm |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|']
for rate in (5, 10, 25, 1000):
    rows = [r for r in checks if r['rate'] == rate]
    md = np.array([r['min_dist'] for r in rows]) * 1000
    lines.append(f"| {rate} Hz | {len(rows)} | {np.mean([r['success'] for r in rows]):.3f} | {md.mean():.2f} | "
                 f"{np.median(md):.2f} | {md.max():.2f} | {np.mean(md < 5):.3f} | {np.mean(md < 2):.3f} |")
text = '\n'.join(lines)
print(text)
(DATA / 'stage0_expert_rates.md').write_text(text + '\n')
