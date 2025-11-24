import sqlite3
import pandas as pd

# Имя базы и таблицы
db_path = "perevody.db"
table_name = "transactions"
column_name = "transdate"   # любой столбец

# Подключаемся к базе
conn = sqlite3.connect(db_path)

# Читаем только один нужный столбец
query = f"SELECT {column_name} FROM {table_name}"
df = pd.read_sql(query, conn)

conn.close()

# Количество уникальных значений
unique_count = df[column_name].nunique()

print(f"Уникальных значений в {column_name}: {unique_count}")