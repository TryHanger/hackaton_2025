import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Загружаем данные ---
conn = sqlite3.connect("perevody.db")
df = pd.read_sql("SELECT transdate, target FROM transactions", conn)
conn.close()

# --- 2. Конвертируем даты ---
df["transdate"] = pd.to_datetime(df["transdate"], format="mixed")

# --- 3. Берём только день месяца ---
df["day_of_month"] = df["transdate"].dt.day   # от 1 до 31

# --- 4. Группировки ---
# Все транзакции
all_counts = df.groupby("day_of_month").size()

# Только fraud
fraud_counts = df[df["target"] == 1].groupby("day_of_month").size()

# --- 5. График: все транзакции + fraud поверх ---
plt.figure(figsize=(12, 5))

plt.bar(all_counts.index, all_counts.values, color="lightgray", label="Все транзакции")
plt.bar(fraud_counts.index, fraud_counts.values, color="red", label="Fraud", alpha=0.8)

plt.title("Распределение транзакций по дню месяца (Fraud выделены)")
plt.xlabel("День месяца")
plt.ylabel("Количество")
plt.xticks(range(1, 32))
plt.legend()
plt.tight_layout()
plt.show()

# --- 6. График: только fraud ---
plt.figure(figsize=(12, 5))

plt.bar(fraud_counts.index, fraud_counts.values, color="red")
plt.title("Распределение Fraud-транзакций по дню месяца")
plt.xlabel("День месяца")
plt.ylabel("Количество Fraud")
plt.xticks(range(1, 32))
plt.tight_layout()
plt.show()
