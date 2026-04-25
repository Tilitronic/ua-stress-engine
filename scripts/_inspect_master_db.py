import sqlite3, json

db = r'w:\Projects\poetykaAnalizerEngine\VersaSenseEngine\VersaSenseBackend\src\data_management\transform\cache\MERGEDSQL_33734b4d5785370f0db8c93c657d5f1d244e9e559f3a9f4e6bcaa914959db665.sqlite3'
con = sqlite3.connect(db)

cols = con.execute('PRAGMA table_info(word_form)').fetchall()
print('word_form columns:', [c[1] for c in cols])

rows = con.execute('SELECT * FROM word_form LIMIT 5').fetchall()
print('\nSample rows:')
for r in rows:
    print(' ', r)

has_stress = con.execute(
    "SELECT COUNT(*) FROM word_form WHERE stress_indices_json IS NOT NULL AND stress_indices_json NOT IN ('[]', '')"
).fetchone()[0]
print(f'\nWith stress data: {has_stress:,}')

single = con.execute(
    "SELECT COUNT(*) FROM word_form WHERE form NOT LIKE '% %'"
).fetchone()[0]
print(f'Single-word forms: {single:,}')

# Heteronym count — forms with more than one stress index
multi_stress = con.execute(
    "SELECT COUNT(*) FROM word_form WHERE length(stress_indices_json) - length(replace(stress_indices_json, ',', '')) >= 1"
).fetchone()[0]
print(f'Forms with multiple stress variants: {multi_stress:,}')

# Sample heteronym
het = con.execute(
    "SELECT form, pos, stress_indices_json FROM word_form WHERE length(stress_indices_json) - length(replace(stress_indices_json, ',', '')) >= 1 LIMIT 5"
).fetchall()
print('\nSample heteronyms:')
for h in het:
    print(' ', h)

# POS distribution
pos_dist = con.execute(
    "SELECT pos, COUNT(*) as cnt FROM word_form GROUP BY pos ORDER BY cnt DESC LIMIT 15"
).fetchall()
print('\nPOS distribution:')
for p in pos_dist:
    print(f'  {p[0]}: {p[1]:,}')

# feature table sample
feat_cols = con.execute('PRAGMA table_info(feature)').fetchall()
print('\nfeature columns:', [c[1] for c in feat_cols])
feat_sample = con.execute('SELECT * FROM feature LIMIT 5').fetchall()
for f in feat_sample:
    print(' ', f)

con.close()
