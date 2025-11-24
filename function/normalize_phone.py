import sqlite3
import re

def normalize(phone):
    phone = re.sub('[ ,_.-]', '', phone)
    phone = phone.lower()
    return phone

def lastPhone(id):
    conn = sqlite3.connect('../info_perevody.db')
    cursor = conn.cursor()
    query = """
    SELECT last_phone_model_categorical
    FROM transactions
    WHERE cst_dim_id = ?
    ORDER BY transdate DESC
    """
    cursor.execute(query, (id,))
    phone = cursor.fetchone()

    return phone

def verificationNumber(cst_dim_id, phone):
    phoneInDB = normalize(lastPhone(cst_dim_id)[0])
    phone = normalize(phone)
    if phoneInDB == phone:
        return True
    else:
        return False


if __name__ == '__main__':
    print(verificationNumber(451851768, "Samsung SM-A256E"))