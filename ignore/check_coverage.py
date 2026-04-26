import sqlite3

def main():
    conn = sqlite3.connect('src/stress_prediction/data/stress_training.db')
    cur = conn.cursor()

    cur.execute('''
    SELECT 
        SUM(CASE WHEN pos IS NULL OR pos = '' THEN 1 ELSE 0 END) as missing_pos,
        SUM(CASE WHEN pos = 'X' THEN 1 ELSE 0 END) as pos_X,
        SUM(CASE WHEN features_json IS NULL OR features_json = '' OR features_json = '{}' THEN 1 ELSE 0 END) as missing_feats,
        COUNT(*) as total
    FROM training_entries
    WHERE variant_type != 'grammatical_homonym'
    ''')
    missing_pos, pos_x, missing_feats, total = cur.fetchone()

    print('Coverage after enrichment:')
    print(f'  total rows:           {total:,}')
    print(f'  missing POS:          {missing_pos:,} ({missing_pos*100/total:.2f}%)')
    print(f'  POS = "X":            {pos_x:,} ({pos_x*100/total:.2f}%)')
    print(f'  missing morphology:   {missing_feats:,} ({missing_feats*100/total:.2f}%)')

    cur.execute('''
    SELECT pos, COUNT(*) as cnt FROM training_entries 
    WHERE variant_type != 'grammatical_homonym'
    GROUP BY pos ORDER BY cnt DESC LIMIT 10
    ''')
    print('\nPOS distribution (top 10):')
    for pos, cnt in cur.fetchall():
        print(f'  {pos:8} {cnt:8,} ({cnt*100/total:5.2f}%)')

    conn.close()

if __name__ == '__main__':
    main()
