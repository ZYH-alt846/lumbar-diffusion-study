"""Read aggregate files only and recompute the primary comparison."""
from pathlib import Path
import csv
import json

root = Path(__file__).resolve().parents[1]
with (root/'data/aggregate/core_results.csv').open(encoding='utf-8', newline='') as f:
    rows = {r['method']: r for r in csv.DictReader(f)}
main = float(rows['CD-F']['dice'])
best = max(float(r['dice']) for k,r in rows.items() if k != 'CD-F')
diagnostics = json.loads((root/'data/aggregate/diagnostics.json').read_text(encoding='utf-8'))
output = {'cohort': 'internal_select', 'n_cases': 34,
          'CD-F_minus_DPR-R': main-float(rows['DPR-R']['dice']),
          'CD-F_minus_CR': main-float(rows['CR']['dice']),
          'CD-F_minus_CD-S': main-float(rows['CD-S']['dice']),
          'quality_dice_threshold': best+0.01, 'quality_dice_pass': main >= best+0.01,
          'joint_condition_mean_difference': diagnostics['donor_joint_condition']['mean_difference'],
          'translation_fraction_gap_reduced': diagnostics['translation']['fraction_gap_reduced'],
          'rigid_seed0_fraction_gap_reduced': diagnostics['rigid_seed0']['fraction_gap_reduced']}
print(json.dumps(output, indent=2))
