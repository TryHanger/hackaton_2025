import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.impute import SimpleImputer  # <-- Добавлено для обработки NaN
import warnings

warnings.filterwarnings('ignore')

# УДАЛЕНА ИЗБЫТОЧНАЯ ФУНКЦИЯ (логика встроена ниже)
# def transform_frequency_column(value):
#     ... (функция удалена, чтобы избежать дублирования и упростить код)

# Установите соединение с базой данных
conn = sqlite3.connect("../clean_data.db")
query = """
SELECT
    cst_dim_id, 
    transdate,
    transdatetime,
    amount,
    monthly_os_changes,
    monthly_phone_model_changes,
    logins_last_7_days,
    logins_last_30_days,
    login_frequency_7d,
    login_frequency_30d,
    freq_change_7d_vs_mean,
    logins_7d_over_30d_ratio,
    avg_login_interval_30d,
    std_login_interval_30d,
    var_login_interval_30d,
    ewm_login_interval_7d,
    burstiness_login_interval,
    fano_factor_login_interval,
    zscore_avg_login_interval_7d,
    target
FROM unique_transactions
"""
df = pd.read_sql_query(query, conn)
conn.close()

print("=== ЗАГРУЗКА ДАННЫХ ===")
print(f"Кол-во записей: {len(df)}")
print(f"Кол-во признаков: {len(df.columns)}")

target_counts = df['target'].value_counts()
target_percentage = df['target'].value_counts(normalize=True) * 100

print("\n=== РАСПРЕДЕЛЕНИЕ TARGET ===")
print(f"Class 0 (легитимные): {target_counts[0]} записей ({target_percentage[0]:.2f}%)")
print(f"Class 1 (мошенничество): {target_counts[1]} записей ({target_percentage[1]:.2f}%)")
print(f"Всего записей: {len(df)}")

# =========================================================================================================================== #
#               ПОДГОТОВКА И КОНСТРУИРОВАНИЕ ПРИЗНАКОВ
# =========================================================================================================================== #
df['transdatetime'] = pd.to_datetime(df['transdatetime'])
df['transdate_day'] = df['transdatetime'].dt.day
df['transdate_dayofweek'] = df['transdatetime'].dt.dayofweek
df['transdate_hour'] = df['transdatetime'].dt.hour
df['transdate_is_business_hours'] = ((df['transdatetime'].dt.hour >= 10) &
                                     (df['transdatetime'].dt.hour <= 18)).astype(int)

df['transdate_year'] = df['transdatetime'].dt.year
df['transdate_month'] = df['transdatetime'].dt.month
df['transdate_minute'] = df['transdatetime'].dt.minute
df['transdate_week'] = df['transdatetime'].dt.isocalendar().week.astype(int)  # Приводим к int
df['transdate_quarter'] = df['transdatetime'].dt.quarter

df['amount_log'] = np.log1p(df['amount'])

df['total_device_changes'] = df['monthly_os_changes'] + df['monthly_phone_model_changes']

month_map = {
    'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04',
    'май': '05', 'июн': '06', 'июл': '07', 'авг': '08',
    'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12'
}


# --- ОЧИСТКА login_frequency_30d ---
# Функция transform_frequency_column должна быть определена здесь
def transform_frequency_column(value):
    s = str(value)
    s = s.strip().lower()

    # Сначала пытаемся обработать формат "день.месяц"
    for text_month, num_month in month_map.items():
        if text_month in s:
            try:
                day_part = s.split('.')[0]
                new_value = f"{day_part}.{num_month}"
                return float(new_value)
            except:
                return np.nan # Если формат не сработал

    # Если это не дата, очищаем десятичный разделитель
    s = s.replace(',', '.')

    # Пытаемся преобразовать оставшуюся строку в число.
    # Если это строка (например, '01.май' или 'N/A'), она станет NaN.
    try:
        return float(s)
    except ValueError:
        return np.nan # Гарантированно возвращаем числовой NaN

df['login_frequency_30d'] = df['login_frequency_30d'].apply(transform_frequency_column)
df['login_frequency_30d'] = pd.to_numeric(df['login_frequency_30d'], errors='coerce')

df["freq_change_7d_vs_mean"] = (
    df["freq_change_7d_vs_mean"]
    .astype(str)
    .str.strip()
    .str.replace(',', '.') # Заменяем запятые на точки
)
df["freq_change_7d_vs_mean"] = pd.to_numeric(df["freq_change_7d_vs_mean"], errors='coerce')

# --- ОБРАБОТКА ИНТЕРВАЛОВ И СЕНТИНЕЛ-ЗНАЧЕНИЙ ---
# Функция для безопасной гибридной обработки числовых столбцов
def safe_hybrid_transform(df, col):
    # 1. Приведение к числу и обработка мусора (NaN)
    df[col] = df[col].astype(str).str.strip().str.replace(',', '.')
    df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Создание флага сентинела (-1). Сначала заполняем NaN нулем для корректного флага
    # *Важно: это временное заполнение, для финального заполнения используем импьютер.*
    temp_col = df[col].fillna(0)
    df[f'{col}_is_sentinel'] = (temp_col == -1).astype(int)

    # 3. Логарифмирование и очистка
    df[f'{col}_log'] = np.log1p(temp_col)  # Применяем log1p к временному столбцу
    df.loc[df[f'{col}_log'] < 0, f'{col}_log'] = 0  # Очищаем результат от значений, возникших из -1

    # 4. Восстанавливаем NaN в исходных столбцах, если они были до fillna(0)
    # Это важно, чтобы SimpleImputer заполнил их корректной статистикой на train set
    df.loc[df[col].isna(), f'{col}_log'] = np.nan
    df.loc[df[col].isna(), f'{col}_is_sentinel'] = np.nan  # *Опционально: можно оставить 0, если NaN должен быть 0*


# Применяем к столбцам, где ожидается -1 или текстовый формат:
safe_hybrid_transform(df, 'avg_login_interval_30d')
safe_hybrid_transform(df, 'std_login_interval_30d')
safe_hybrid_transform(df, 'var_login_interval_30d')
safe_hybrid_transform(df, 'ewm_login_interval_7d')
safe_hybrid_transform(df, 'fano_factor_login_interval')

# ⚠️ УДАЛЯЕМ старые, необработанные столбцы
df = df.drop(columns=['avg_login_interval_30d', 'std_login_interval_30d', 'var_login_interval_30d',
                      'ewm_login_interval_7d', 'fano_factor_login_interval', 'amount',
                      'cst_dim_id', 'transdate', 'transdatetime'])

print("=== ДИАГНОСТИКА ВСЕХ СТОЛБЦОВ ===")

for col in df.columns:
    dtype = df[col].dtype
    if dtype == 'object':
        # Покажем примеры строковых значений
        sample_values = df[col].dropna().head(3).tolist()
        print(f"Столбец {col} (object): примеры {sample_values}")

        # Проверим, содержит ли русские месяцы
        has_russian_months = any(any(month in str(val).lower() for month in month_map.keys())
                                 for val in df[col].dropna().head(10))
        if has_russian_months:
            print(f"  ⚠️  Содержит русские месяцы!")

    elif np.issubdtype(dtype, np.number):
        # Для числовых столбцов проверим на выбросы
        if df[col].notna().sum() > 0:
            print(f"Столбец {col} ({dtype}): min={df[col].min():.2f}, max={df[col].max():.2f}")

# =========================================================================================================================== #
#                   ОПРЕДЕЛЕНИЕ ПРИЗНАКОВ И ОБУЧЕНИЕ
# =========================================================================================================================== #

features_final = [
    # Временные
    "transdate_day", "transdate_dayofweek", "transdate_hour",
    "transdate_is_business_hours", "transdate_year", "transdate_month",
    "transdate_minute", "transdate_week", "transdate_quarter",

    # Сумма и Изменения Устройств
    "amount_log", "total_device_changes", "monthly_os_changes", "monthly_phone_model_changes",

    # Исходные Поведенческие (тоже могут содержать NaN после to_numeric)
    "logins_last_7_days", "logins_last_30_days", "login_frequency_7d",
    "login_frequency_30d", "freq_change_7d_vs_mean", "logins_7d_over_30d_ratio",
    "burstiness_login_interval", "zscore_avg_login_interval_7d",

    # Преобразованные Интервалы
    "avg_login_interval_30d_log", "std_login_interval_30d_log",
    "var_login_interval_30d_log", "ewm_login_interval_7d_log",
    "fano_factor_login_interval_log",

    # Флаги Маркеров (-1)
    "avg_login_interval_30d_is_sentinel", "std_login_interval_30d_is_sentinel",
    "var_login_interval_30d_is_sentinel", "ewm_login_interval_7d_is_sentinel",
    "fano_factor_login_interval_is_sentinel"
]

CATEGORICAL_FEATURES = [
    # Временные (цикличные/дискретные)
    "transdate_dayofweek",
    "transdate_hour",
    "transdate_week",
    "transdate_day",
    "transdate_month",
    "transdate_quarter",

    # Счетчики (дискретные значения 0-N)
    "monthly_os_changes",
    "monthly_phone_model_changes",
    "total_device_changes"
]

X = df[features_final]
y = df['target']

# 1. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 2. Обработка NaN с помощью SimpleImputer (медиана)
imputer = SimpleImputer(strategy='median')
# Обучаем импьютер только на обучающей выборке
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Возвращаем в DataFrame для работы с именами столбцов и типами данных
X_train_final = pd.DataFrame(X_train_imputed, columns=features_final)
X_test_final = pd.DataFrame(X_test_imputed, columns=features_final)

# 3. 🛑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ПРИВЕДЕНИЕ КАТЕГОРИАЛЬНЫХ ТИПОВ
# Imputer возвращает float, CatBoost требует int/str для категорий.
for col in CATEGORICAL_FEATURES:
    # Округляем до ближайшего целого и приводим к типу 'int'
    X_train_final[col] = X_train_final[col].round(0).astype(int)
    X_test_final[col] = X_test_final[col].round(0).astype(int)

# 4. Обучение CatBoost
print("\n=== НАЧАЛО ОБУЧЕНИЯ CATBOOST ===")
model_w = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    auto_class_weights='Balanced',
    verbose=200,
    random_seed=42,
    cat_features=CATEGORICAL_FEATURES,
)

model_w.fit(
    X_train_final,
    y_train,
    eval_set=(X_test_final, y_test)
)

preds = model_w.predict(X_test_final)
probs = model_w.predict_proba(X_test_final)[:, 1]

print("\n=== РЕЗУЛЬТАТЫ МОДЕЛИ ===")
print(f"AUC: {roc_auc_score(y_test, probs):.4f}")
print(classification_report(y_test, preds))