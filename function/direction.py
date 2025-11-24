import sqlite3
import matplotlib.pyplot as plt

def get_direction(target):
    conn = sqlite3.connect('../perevody.db')
    cursor = conn.cursor()
    query = """
    SELECT DISTINCT direction
    FROM transactions
    WHERE target = ?
    """
    cursor.execute(query, (target,))
    result = cursor.fetchall()

    conn.close()

    return result


def get_count_unique(directions):
    conn = sqlite3.connect('../perevody.db')
    cursor = conn.cursor()
    number_unique = []

    for row in directions:
        direction = row[0]
        query = """
        SELECT direction
        FROM transactions
        WHERE direction = ?
        """
        cursor.execute(query, (direction,))
        result = cursor.fetchall()
        number_unique.append(len(result))

    conn.close()

    number_unique_kolvo = {}
    for row in number_unique:
        if row not in number_unique_kolvo:
            number_unique_kolvo[row] = number_unique.count(row)

    return number_unique_kolvo


def visual(dict_unique):
    sorted_keys = sorted(dict_unique.keys())
    sorted_values = [dict_unique[key] for key in sorted_keys]
    plt.figure(figsize=(20, 6))
    plt.bar(range(len(sorted_keys)), sorted_values)

    # Настройки оси X
    plt.xticks(range(len(sorted_keys)), sorted_keys, rotation=90)
    plt.xlabel('Ключи')
    plt.ylabel('Значения')
    plt.title('Распределение значений по ключам')

    # Добавляем сетку для удобства чтения
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

def get_last_transaction(direction):
    conn = sqlite3.connect('../perevody.db')
    cursor = conn.cursor()
    query = """
    SELECT direction
    FROM transactions
    WHERE direction = ?
    """
    cursor.execute(query, (direction,))
    result = cursor.fetchall()
    conn.close()

    return result

if __name__ == '__main__':
    # directions = get_direction(0)
    # dict_unique = get_count_unique(directions)
    # visual(dict_unique)
    # print(dict_unique)

    transactions = get_last_transaction('3c1a1b74e4299090614802493fd8eb2d')
    print(len(transactions))