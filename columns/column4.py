import sqlite3

import pandas as pd


def get_cst_dim_ids():
    result_dict = {}
    try:
        conn = sqlite3.connect('info_perevody.db')  # замените на имя вашего файла БД
        cursor = conn.cursor()

        for i in range(0,6):
            query = """
            SELECT DISTINCT cst_dim_id 
            FROM transactions
            WHERE monthly_phone_model_changes = ?
            """
            cursor.execute(query, (i,))
            results = cursor.fetchall()
            idx = [id[0] for id in results]
            print(idx)
            result_dict[i] = idx

        print(result_dict)
        return result_dict

        # print("cst_dim_id с monthly_phone_model_changes > 4:")
        # for row in results:
        #     print(row[0])
        #
        # return [row[0] for row in results]  # возвращаем список ID

    except sqlite3.Error as e:
        print(f"Ошибка при работе с SQLite: {e}")
        
    finally:
        # Закрываем соединение
        if conn:
            conn.close()
            print("Соединение с БД закрыто")

def get_target_by_ids(target_ids):
    fraud_percentages  = {}
    try:
        conn = sqlite3.connect('perevody.db')
        cursor = conn.cursor()
        for key, id_list in target_ids.items():
            percentages = []
            for id in id_list:

                if id is None:
                    print(f"Пропущен None в группе {key}")
                    continue

                query = """
                SELECT target
                FROM transactions
                WHERE cst_dim_id = ?
                """
                cursor.execute(query, (id,))
                results = cursor.fetchall()

                fraud_count = 0
                total_count =len(results)

                for target in results:
                    if target[0] == 1:
                        fraud_count += 1

                if total_count > 0:
                    fraud_percentage = (fraud_count / total_count) * 100
                else:
                    fraud_percentage = 0
                    print(id)
                    print("!!!!!!!!!!!!!NIKITA ZDES' OSCHIBKA!!!!!!!!!!")

                percentages.append(fraud_percentage)

            fraud_percentages[key] = percentages

        print(fraud_percentages)
        return fraud_percentages

    except sqlite3.Error as e:
        print(f"Ошибка при работе с SQLite: {e}")

    finally:
        # Закрываем соединение
        if conn:
            conn.close()
            print("Соединение с БД закрыто")

if __name__ == "__main__":
    cst_dim_ids = get_cst_dim_ids()
    target_ids = get_target_by_ids(cst_dim_ids)
    rows = []
    for changes, vals in target_ids.items():
        s = pd.Series(vals)
        rows.append({
            "changes": changes,
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(ddof=0),
            "min": s.min(),
            "max": s.max(),
            "q25": s.quantile(0.25),
            "q75": s.quantile(0.75),
            "pct_nonzero": (s > 0).mean()
        })

    summary = pd.DataFrame(rows).sort_values("changes")
    print(summary.to_string(index=False))


