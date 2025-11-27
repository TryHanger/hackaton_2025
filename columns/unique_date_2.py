import sqlite3
import collections

conn = sqlite3.connect('../perevody.db')
cursor = conn.cursor()
query = """
SELECT transdate, cst_dim_id
FROM transactions 
"""
cursor.execute(query)
rows = cursor.fetchall()
idx_dates = {}
number = 0
for row in rows:
    if row[1] not in idx_dates:
        idx_dates[row[1]] = []
        idx_dates[row[1]].append(row[0])
    else:
        idx_dates[row[1]].append(row[0])
    number += 1

# for key, values in idx_dates.items():
#     counts_dates = collections.Counter(values)
#     for value in counts_dates.values():
#         if value > 1:
#             print(key, value)

unique_dates_per_id = {}
for cst_dim_id, dates in idx_dates.items():
    date_counts = collections.Counter(dates)
    unique_dates = [date for date, count in date_counts.items() if count == 1]
    if unique_dates:
        unique_dates_per_id[cst_dim_id] = unique_dates

print(unique_dates_per_id)
print(number)