# import pandas as pd
# import sqlite3
#
# # 1️⃣ Читаем CSV
# df = pd.read_csv(
#     "perevod_1.csv",
#     encoding="cp1251",
#     sep=None,           # правильный разделитель для этого файла
#     engine="python",
#     header=2            # вторая строка — реальные названия колонок
# )
#
# # 2️⃣ Подключаемся к SQLite (создаст файл perevody.db)
# conn = sqlite3.connect("info_perevody.db")
#
# # 3️⃣ Записываем все данные в таблицу "transactions"
# df.to_sql(
#     "transactions",   # имя таблицы в SQLite
#     conn,
#     if_exists="replace",  # перезаписать таблицу, если существует
#     index=False          # не сохранять индекс pandas
# )
#
# conn.close()
#
# print("Готово! Все данные перенесены в SQLite.")

import sqlite3
import pandas as pd

# Загружаем первую таблицу
conn1 = sqlite3.connect("info_perevody.db")
df1 = pd.read_sql("SELECT * FROM transactions", conn1)
conn1.close()

# Загружаем вторую таблицу
conn2 = sqlite3.connect("perevody.db")
df2 = pd.read_sql("SELECT * FROM transactions", conn2)
conn2.close()

# Merge по cst_dim_id + transdate
df_merged = pd.merge(
    df2,
    df1,
    how="left",
    on=["cst_dim_id", "transdate"],
    suffixes=("", "_tbl1")
)

# Переупорядочиваем колонки: сначала df2, потом df1
cols_df2 = df2.columns.tolist()
cols_df1 = [c for c in df_merged.columns if c not in cols_df2]
df_merged = df_merged[cols_df2 + cols_df1]

# Сохраняем в новую базу с правильным именем таблицы
conn_new = sqlite3.connect("perevody_all.db")
df_merged.to_sql("merged_transactions", conn_new, if_exists="replace", index=False)
conn_new.close()

print("Готово! Объединение выполнено.")