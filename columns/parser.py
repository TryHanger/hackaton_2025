import pandas as pd

def pars_1():
    df = pd.read_csv(
        "perevod_1.csv",
        encoding="cp1251",
        sep=None,           # автоопределение разделителя
        engine="python",
        header=1            # вторая строка — реальный header
    )
    return df

def pars_2():
    df = pd.read_csv(
        "perevod_2.csv",
        encoding="cp1251",
        sep=";",
        engine="python",
        header=1            # вторая строка — реальный header
    )
    return df

df = pars_2()
print("Сырые имена колонок:")
for i, col in enumerate(df.columns):
    print(f"{i}: ->{col}<-   (len={len(col)})")