import sqlite3
import json

conn = sqlite3.connect('src/stress_prediction/data/stress_training.db')
cur = conn.cursor()

# Check: same form, different stress_indices values
cur.execute('''
SELECT form, GROUP_CONCAT(DISTINCT stress_indices), COUNT(DISTINCT stress_indices) as stress_variants
FROM training_entries
WHERE variant_type != "grammatical_homonym"
GROUP BY form
HAVING COUNT(DISTINCT stress_indices) > 1
LIMIT 15
''')

print('Forms with DIFFERENT stress labels (data quality issue):')
print("-" * 80)
for form, stresses, variants in cur.fetchall():
    print(f'{form:20} stresses: {stresses:40} variants: {variants}')

# Count how many forms have this issue
cur.execute('''
SELECT COUNT(*) as forms_with_conflict
FROM (
    SELECT form
    FROM training_entries
    WHERE variant_type != "grammatical_homonym"
    GROUP BY form
    HAVING COUNT(DISTINCT stress_indices) > 1
)
''')

conflict_count = cur.fetchone()[0]
print(f'\nTotal forms with conflicting stress labels: {conflict_count:,}')

# Total unique forms
cur.execute('SELECT COUNT(DISTINCT form) FROM training_entries')
total_forms = cur.fetchone()[0]
print(f'Total unique forms: {total_forms:,}')
print(f'Percentage with conflicts: {conflict_count/total_forms*100:.2f}%')

conn.close()
