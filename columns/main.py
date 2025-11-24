import pandas as pd

target_id = 2933793914

df = pd.read_csv(
    "perevod_2.csv",
    encoding="cp1251",
    sep=";",           # автоопределение разделителя
    engine="python",
    header=1            # вторая строка — реальный header
)

# теперь колонка cst_dim_id существует
filtered = df[df["cst_dim_id"] == target_id]

print(filtered)
# если хотите сохранить в новый CSV
filtered.to_csv("perevod_2_filter.csv", index=False, encoding="utf-8")