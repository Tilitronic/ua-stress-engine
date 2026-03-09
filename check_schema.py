import sqlite3

conn = sqlite3.connect('src/stress_prediction/data/stress_training.db')
cur = conn.cursor()

# Check schema
cur.execute("PRAGMA table_info(training_entries)")
columns = cur.fetchall()

print("Training entries schema:")
for col_id, name, type_, notnull, default, pk in columns:
    print(f"  {name:25} {type_:10} default={default}")

# Check data
cur.execute("""
SELECT pos, COUNT(*) as cnt, AVG(pos_confidence) as avg_pos_conf, AVG(features_confidence) as avg_feat_conf
FROM training_entries
WHERE variant_type != 'grammatical_homonym'
GROUP BY pos
ORDER BY cnt DESC
LIMIT 15
""")

print("\n\nPOS distribution with confidence scores:")
print(f"{'POS':10} {'Count':>10} {'Avg POS Conf':>12} {'Avg Feat Conf':>14}")
print("-" * 50)
for pos, cnt, pos_conf, feat_conf in cur.fetchall():
    print(f"{pos:10} {cnt:10,} {pos_conf:12.2f} {feat_conf:14.2f}")

conn.close()
