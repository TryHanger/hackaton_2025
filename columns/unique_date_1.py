# import sqlite3
# import collections
#
# conn = sqlite3.connect('../info_perevody.db')
# cursor = conn.cursor()
# query = """
# SELECT transdate, cst_dim_id
# FROM transactions
# """
# cursor.execute(query)
# rows = cursor.fetchall()
# idx_dates = {}
# for row in rows:
#     if row[1] not in idx_dates:
#         idx_dates[row[1]] = []
#         idx_dates[row[1]].append(row[0])
#     else:
#         idx_dates[row[1]].append(row[0])
#
# # for key, values in idx_dates.items():
# #     counts_dates = collections.Counter(values)
# #     for value in counts_dates.values():
# #         if value > 1:
# #             print(key, value)
#
# unique_dates_per_id = {}
# for cst_dim_id, dates in idx_dates.items():
#     date_counts = collections.Counter(dates)
#     unique_dates = [date for date, count in date_counts.items() if count == 1]
#     if unique_dates:
#         unique_dates_per_id[cst_dim_id] = unique_dates
#
#
# print(unique_dates_per_id)
#
# total_days = sum(len(dates) for dates in unique_dates_per_id.values())
# print(total_days)
# # print(idx_dates)
# # print(number)

# 453148036.0: ["'2025-03-08 00:00:00.000'", "'2025-02-04 00:00:00.000'", "'2025-01-06 00:00:00.000'"]


import sqlite3
import os

def table_info_perevody():
    conn = sqlite3.connect('../info_perevody.db')
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(transactions)")
    columns = cursor.fetchall()

    # 2. Создаем новую таблицу с такой же структурой
    create_columns = []
    for col in columns:
        col_name = col[1]
        col_type = col[2]
        create_columns.append(f"{col_name} {col_type}")

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS unique_transactions (
        {', '.join(create_columns)}
    )
    """
    cursor.execute(create_table_sql)

    cursor.execute("DELETE FROM unique_transactions;")

    # 3. Вставляем только уникальные записи
    query2 = """
    INSERT INTO unique_transactions 
    SELECT t1.*
    FROM transactions t1
    INNER JOIN (
        SELECT cst_dim_id, transdate
        FROM transactions
        GROUP BY cst_dim_id, transdate
        HAVING COUNT(*) = 1
    ) AS unique_pairs
    ON t1.cst_dim_id = unique_pairs.cst_dim_id AND t1.transdate = unique_pairs.transdate;
    """
    cursor.execute(query2)

    conn.commit()

    conn.close()

def table_perevody():
    conn = sqlite3.connect('../perevody.db')
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(transactions)")
    columns = cursor.fetchall()

    # 2. Создаем новую таблицу с такой же структурой
    create_columns = []
    for col in columns:
        col_name = col[1]
        col_type = col[2]
        create_columns.append(f"{col_name} {col_type}")

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS unique_transactions (
        {', '.join(create_columns)}
    )
    """
    cursor.execute(create_table_sql)

    cursor.execute("DELETE FROM unique_transactions;")

    # 3. Вставляем только уникальные записи
    query2 = """
    INSERT INTO unique_transactions 
    SELECT t1.*
    FROM transactions t1
    INNER JOIN (
        SELECT cst_dim_id, transdate
        FROM transactions
        GROUP BY cst_dim_id, transdate
        HAVING COUNT(*) = 1
    ) AS unique_pairs
    ON t1.cst_dim_id = unique_pairs.cst_dim_id AND t1.transdate = unique_pairs.transdate;
    """
    cursor.execute(query2)

    conn.commit()

    conn.close()

# table_perevody()
# table_info_perevody()

import sqlite3

import sqlite3
import os


def combine_databases_from_two_dbs():
    """
    Объединяет таблицы unique_transactions из двух разных БД.
    """
    try:
        # Подключаемся к обеим БД
        conn1 = sqlite3.connect('../info_perevody.db')
        conn2 = sqlite3.connect('../perevody.db')

        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()

        # Проверяем существование таблиц
        cursor1.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='unique_transactions'")
        table1_exists = cursor1.fetchone() is not None

        cursor2.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='unique_transactions'")
        table2_exists = cursor2.fetchone() is not None

        if not table1_exists and not table2_exists:
            print("❌ Таблица unique_transactions не найдена в обеих БД!")
            return False

        # Создаем новую БД для результата (или используем первую)
        result_conn = sqlite3.connect('../combined_perevody.db')
        result_cursor = result_conn.cursor()

        # 1. Создаем структуру таблицы в новой БД
        print("Создаем структуру таблицы...")

        # Берем структуру из первой БД
        cursor1.execute("PRAGMA table_info(unique_transactions)")
        columns = cursor1.fetchall()

        create_columns = []
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            create_columns.append(f"{col_name} {col_type}")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS combined_unique_transactions (
            {', '.join(create_columns)}
        )
        """
        result_cursor.execute("DROP TABLE IF EXISTS combined_unique_transactions")
        result_cursor.execute(create_table_sql)

        # 2. Вставляем данные из первой БД
        print("Добавляем данные из info_perevody.db...")
        cursor1.execute("SELECT * FROM unique_transactions")
        rows1 = cursor1.fetchall()

        if rows1:
            placeholders = ','.join(['?'] * len(columns))
            insert_sql = f"INSERT INTO combined_unique_transactions VALUES ({placeholders})"
            result_cursor.executemany(insert_sql, rows1)

        # 3. Добавляем данные из второй БД, которых нет в первой
        print("Добавляем данные из perevody.db...")

        # Получаем полный список имен столбцов из целевой схемы (19 столбцов)
        all_column_names = [col[1] for col in columns]

        # Определяем 7 столбцов для SELECT (предполагаем, что это первые 7)
        columns_to_select = all_column_names[:7]
        select_cols_str = ', '.join(columns_to_select)

        # Получаем все существующие комбинации (cst_dim_id, transdate) из результата
        result_cursor.execute("SELECT cst_dim_id, transdate FROM combined_unique_transactions")
        existing_combinations = set((row[0], row[1]) for row in result_cursor.fetchall())

        # Выбираем данные из conn2, но только 7 столбцов
        cursor2.execute(f"SELECT {select_cols_str} FROM unique_transactions")
        rows2_partial = cursor2.fetchall()

        new_rows = []
        # Определяем количество NULL-заполнителей (19 - 7 = 12)
        null_padding = [None] * (len(all_column_names) - len(columns_to_select))

        for row in rows2_partial:
            # ❗❗ ИСПРАВЛЕНИЕ ОШИБКИ BINDINGS:
            # Собираем полную строку (7 данных + 12 NULL)
            full_row_data = list(row) + null_padding

            # ПРОВЕРКА НА ДУБЛИКАТЫ
            # Предполагаем, что cst_dim_id и transdate находятся на позициях 0 и 1
            cst_dim_id = row[0]
            transdate = row[1]

            if (cst_dim_id, transdate) not in existing_combinations:
                new_rows.append(full_row_data)

        if new_rows:
            # Используем полный список столбцов (19) для INSERT
            placeholders = ','.join(['?'] * len(all_column_names))
            insert_cols_str = ','.join(all_column_names)

            # Явно указываем столбцы для вставки, чтобы избежать ошибок порядка
            insert_sql = f"INSERT INTO combined_unique_transactions ({insert_cols_str}) VALUES ({placeholders})"
            result_cursor.executemany(insert_sql, new_rows)

        # 4. Статистика
        count1 = len(rows1) if rows1 else 0
        count2 = len(rows2_partial) if rows2_partial else 0
        result_cursor.execute("SELECT COUNT(*) FROM combined_unique_transactions")
        combined_count = result_cursor.fetchone()[0]

        print("\n📊 СТАТИСТИКА ОБЪЕДИНЕНИЯ:")
        print(f"info_perevody.db: {count1:,} записей")
        print(f"perevody.db: {count2:,} записей")
        print(f"Объединенная БД: {combined_count:,} записей")
        print(f"Дубликатов исключено: {count1 + count2 - combined_count:,} записей")

        # 5. Проверка на дубликаты
        result_cursor.execute("""
        SELECT cst_dim_id, transdate, COUNT(*) as cnt
        FROM combined_unique_transactions
        GROUP BY cst_dim_id, transdate
        HAVING COUNT(*) > 1
        """)
        duplicates = result_cursor.fetchall()

        if not duplicates:
            print("✅ В объединенной таблице нет дубликатов!")
        else:
            print(f"❌ Найдено дубликатов: {len(duplicates)}")

        result_conn.commit()
        print("✅ Объединение завершено успешно!")

    except sqlite3.Error as e:
        print(f"❌ Ошибка базы данных: {e}")
        return False
    finally:
        for conn in [conn1, conn2, result_conn]:
            if conn:
                conn.close()

    return True


# Запуск объединения
combine_databases_from_two_dbs()