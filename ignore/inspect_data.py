import sqlite3
import json
from pathlib import Path

conn = sqlite3.connect('src/stress_prediction/data/stress_training.db')
cur = conn.cursor()

# Sample some entries to understand data structure
cur.execute("""
SELECT lemma, form, stress_indices, pos, features_json, variant_type 
FROM training_entries 
WHERE variant_type != 'grammatical_homonym'
LIMIT 20
""")

print("Sample training entries:")
print("-" * 100)
for lemma, form, stress_idx, pos, feats_json, variant_type in cur.fetchall():
    stress = json.loads(stress_idx) if stress_idx else []
    feats = json.loads(feats_json) if feats_json else {}
    print(f"Lemma: {lemma:20} Form: {form:20} Stress: {stress} POS: {pos:6} Features: {feats} Variant: {variant_type}")

# Check if there are alternative stresses (multiple stress possibilities per form)
print("\n\nForms with multiple stress patterns (free/morphological variants):")
print("-" * 100)
cur.execute("""
SELECT form, GROUP_CONCAT(DISTINCT stress_indices), COUNT(*) as cnt
FROM training_entries 
WHERE variant_type != 'grammatical_homonym'
GROUP BY form
HAVING COUNT(DISTINCT stress_indices) > 1
LIMIT 10
""")

for form, stresses, cnt in cur.fetchall():
    print(f"Form: {form:20} Stresses: {stresses:40} Count: {cnt}")

conn.close()
