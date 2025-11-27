import sqlite3
import pandas as pd

def logins_last_7_days():
    conn = sqlite3.connect("../comparison_results.db")
    df = pd.read_sql_query("SELECT * FROM logins_last_7_days", conn)
    new_table_name = "features"
    df.to_sql(new_table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Все данные перенесены в таблицу '{new_table_name}' с fraud_rate ✅")

def logins_last_30_days():
    conn = sqlite3.connect("../comparison_results.db")
    df_features = pd.read_sql_query("SELECT * FROM features", conn)
    df = pd.read_sql_query("SELECT cst_dim_id, transdate, logins_last_30_days FROM logins_last_30_days", conn)
    df = df_features.merge(df, on=["cst_dim_id", "transdate"], how="left")
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()
    print("Колонка logins_last_30_days добавлена в таблицу features ✅")

def login_frequency_7d():
    conn = sqlite3.connect("../comparison_results.db")
    df_features = pd.read_sql_query("SELECT * FROM features", conn)
    df = pd.read_sql_query("SELECT cst_dim_id, transdate, login_frequency_7d FROM login_frequency_7d", conn)
    df = df_features.merge(df, on=["cst_dim_id", "transdate"], how="left")
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()
    print("Колонка login_frequency_7d добавлена в таблицу features ✅")

def login_frequency_30d():
    conn = sqlite3.connect("../comparison_results.db")
    df_features = pd.read_sql_query("SELECT * FROM features", conn)
    df = pd.read_sql_query("SELECT cst_dim_id, transdate, login_frequency_30d FROM login_frequency_30d", conn)
    df = df_features.merge(df, on=["cst_dim_id", "transdate"], how="left")
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()
    print("Колонка login_frequency_30d добавлена в таблицу features ✅")

def freq_change_7d_vs_mean():
    conn = sqlite3.connect("../comparison_results.db")
    df_features = pd.read_sql_query("SELECT * FROM features", conn)
    df = pd.read_sql_query("SELECT cst_dim_id, transdate, freq_change_7d_vs_mean FROM freq_change_7d_vs_mean", conn)
    df = df_features.merge(df, on=["cst_dim_id", "transdate"], how="left")
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()

def logins_7d_over_30d_ratio():
    conn = sqlite3.connect("../comparison_results.db")
    df_features = pd.read_sql_query("SELECT * FROM features", conn)
    df = pd.read_sql_query("SELECT cst_dim_id, transdate, logins_7d_over_30d_ratio FROM logins_7d_over_30d_ratio", conn)
    df = df_features.merge(df, on=["cst_dim_id", "transdate"], how="left")
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()

def avg_login_interval_30d():
    conn = sqlite3.connect("../comparison_results.db")
    df_features = pd.read_sql_query("SELECT * FROM features", conn)
    df = pd.read_sql_query("SELECT cst_dim_id, transdate, avg_login_interval_30d FROM avg_login_interval_30d", conn)
    df = df_features.merge(df, on=["cst_dim_id", "transdate"], how="left")
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()

def zscore_avg_login_interval_7d():
    conn = sqlite3.connect("../comparison_results.db")
    df_features = pd.read_sql_query("SELECT * FROM features", conn)
    df = pd.read_sql_query("SELECT cst_dim_id, transdate, zscore_avg_login_interval_7d FROM zscore_avg_login_interval_7d", conn)
    df = df_features.merge(df, on=["cst_dim_id", "transdate"], how="left")
    df.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()

logins_last_7_days()
logins_last_30_days()
login_frequency_7d()
login_frequency_30d()
freq_change_7d_vs_mean()
logins_7d_over_30d_ratio()
avg_login_interval_30d()
zscore_avg_login_interval_7d()