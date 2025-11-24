import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Загружаем данные ---
conn = sqlite3.connect("perevody.db")
df = pd.read_sql("SELECT transdatetime, target FROM transactions", conn)
conn.close()

# --- 2. Конвертация datetime ---
df["transdatetime"] = pd.to_datetime(df["transdatetime"], format="mixed")

# --- 3. Берём только часы (округление до часа) ---
df["hour"] = df["transdatetime"].dt.floor("H").dt.hour   # только час от 0 до 23

# Преобразуем в строку для оси
df["hour_str"] = df["hour"].astype(str)

# --- 4. Группировки ---
# Все транзакции
all_counts = df.groupby("hour_str").size()

# Fraud-транзакции
fraud_counts = df[df["target"] == 1].groupby("hour_str").size()

# --- 5. График: все транзакции + fraud ---
plt.figure(figsize=(14, 5))

plt.bar(all_counts.index, all_counts.values, color="lightgray", label="Все транзакции")
plt.bar(fraud_counts.index, fraud_counts.values, color="red", label="Fraud", alpha=0.8)

plt.title("Распределение транзакций по часам (Fraud выделены)")
plt.xlabel("Часы суток (0–23)")
plt.ylabel("Количество")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

# --- 6. График: только Fraud ---
plt.figure(figsize=(14, 5))

plt.bar(fraud_counts.index, fraud_counts.values, color="red")
plt.title("Распределение Fraud-транзакций по часам")
plt.xlabel("Часы суток (0–23)")
plt.ylabel("Количество Fraud")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
