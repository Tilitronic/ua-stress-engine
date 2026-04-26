import sqlite3
import json
from collections import Counter

conn = sqlite3.connect('src/stress_prediction/data/stress_training.db')
cur = conn.cursor()

cur.execute('''
SELECT stress_indices
FROM training_entries
WHERE variant_type != "grammatical_homonym"
''')

stresses = []
for row in cur.fetchall():
    s = json.loads(row[0])
    if s:
        stresses.append(s[0])

dist = Counter(stresses)
total = len(stresses)

print('Class distribution (positions 0-5 only):')
for pos in sorted(dist.keys()):
    count = dist[pos]
    pct = count * 100 / total
    print(f'  Position {pos}: {count:7,} ({pct:5.1f}%)')

# Baseline: always predict most common (position 1)
baseline_acc = dist[1] / total * 100
print(f'\nBaseline accuracy (always predict pos 1): {baseline_acc:.2f}%')
print(f'Our model accuracy: 72.03%')
print(f'Improvement over baseline: {72.03 - baseline_acc:.2f} percentage points')

conn.close()
