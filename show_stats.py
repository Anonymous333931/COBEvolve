import sqlite3

conn = sqlite3.connect('cobevolve_full_run.db')
cur = conn.cursor()

print('=== OVERALL COUNTS ===')
cur.execute('SELECT COUNT(*) FROM translations')
print('Translations stored:', cur.fetchone()[0])
cur.execute('SELECT COUNT(*) FROM failures')
print('Failures logged:', cur.fetchone()[0])
cur.execute('SELECT COUNT(*) FROM migration_log')
print('Total log events:', cur.fetchone()[0])

print('\n=== TRANSLATION METHOD BREAKDOWN ===')
cur.execute('SELECT method, COUNT(*) FROM translations GROUP BY method')
for r in cur.fetchall():
    print(f'  {r[0]}: {r[1]}')

print('\n=== TRANSLATION SUCCESS RATE ===')
cur.execute('SELECT success, COUNT(*) FROM translations GROUP BY success')
for r in cur.fetchall():
    label = 'success' if r[0] == 1 else 'failed'
    print(f'  {label}: {r[1]}')

print('\n=== AVERAGE ACCURACY SCORE ===')
cur.execute('SELECT AVG(accuracy_score), MIN(accuracy_score), MAX(accuracy_score) FROM translations')
r = cur.fetchone()
print(f'  avg: {r[0]:.3f}  min: {r[1]:.3f}  max: {r[2]:.3f}')

print('\n=== FAILURES: RESOLVED VS UNRESOLVED ===')
cur.execute('SELECT resolved, COUNT(*) FROM failures GROUP BY resolved')
for r in cur.fetchall():
    label = 'resolved' if r[0] == 1 else 'unresolved'
    print(f'  {label}: {r[1]}')

print('\n=== VALIDATION OUTCOMES ===')
cur.execute("SELECT status, COUNT(*) FROM migration_log WHERE module_path='VALIDATE' GROUP BY status")
for r in cur.fetchall():
    print(f'  {r[0]}: {r[1]}')

print('\n=== REPAIR OUTCOMES ===')
cur.execute("SELECT status, COUNT(*) FROM migration_log WHERE module_path='REPAIR' GROUP BY status")
for r in cur.fetchall():
    print(f'  {r[0]}: {r[1]}')

print('\n=== CACHE HITS (self-evolution evidence) ===')
cur.execute("SELECT COUNT(*) FROM migration_log WHERE status='CACHE_HIT'")
print('Total cache hits across all passes:', cur.fetchone()[0])

print('\n=== PASS MARKERS ===')
cur.execute("SELECT id, notes FROM migration_log WHERE status='PASS_MARKER' ORDER BY id")
for r in cur.fetchall():
    print(f'  row {r[0]}:', r[1])

print('\n=== LEARNING OUTCOMES ===')
cur.execute("SELECT status, COUNT(*) FROM migration_log WHERE module_path='LEARN' GROUP BY status")
for r in cur.fetchall():
    print(f'  {r[0]}: {r[1]}')

conn.close()