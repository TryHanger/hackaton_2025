import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

script_dir = Path(__file__).parent
db_path = script_dir.parent / 'clean_data.db'
conn = sqlite3.connect(db_path)
# conn = sqlite3.connect("../clean_data.db")
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
df= pd.read_sql_query(query, conn)
conn.close()

print("=== ЗАГРУЗКА ДАННЫХ ===")
print(f"Кол-во записей: {len(df)}")
print(f"Кол-во признаков: {len(df.columns)}")
print("Столбцы:", list(df.columns))

target_counts = df['target'].value_counts()
target_percentage = df['target'].value_counts(normalize=True) * 100

print("\n=== РАСПРЕДЕЛЕНИЕ TARGET ===")
print(f"Class 0 (легитимные): {target_counts[0]} записей ({target_percentage[0]:.2f}%)")
print(f"Class 1 (мошенничество): {target_counts[1]} записей ({target_percentage[1]:.2f}%)")
print(f"Всего записей: {len(df)}")


def transform_frequency_column(value):
    """
    Преобразует строку, содержащую дату (например, '01.мар'), в числовой формат (например, 1.03).
    Корректные числовые строки (например, '0.9333...') возвращает без изменений.
    """
    s = str(value)
    s = s.strip().lower()  # Удаляем пробелы и приводим к нижнему регистру для надежности

    # Проверка, содержит ли строка месяцы
    for text_month, num_month in month_map.items():
        if text_month in s:
            try:
                # 01.мар -> 01.03 (День.Месяц)
                day_part = s.split('.')[0]

                # Соединяем 'день' и 'номер месяца' через точку
                new_value = f"{day_part}.{num_month}"

                # Возвращаем числовое значение
                return float(new_value)
            except:
                # Если что-то пошло не так при парсинге, возвращаем NaN
                return np.nan

                # Обработка корректных числовых строк (с запятой или точкой)
    s = s.replace(',', '.')

    # Попытка преобразовать очищенную строку в число
    try:
        return float(s)
    except ValueError:
        return np.nan

# =========================================================================================================================== #
#               ПОДГОТОВКА ДАННЫХ
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
df['transdate_week'] = df['transdatetime'].dt.isocalendar().week
df['transdate_quarter'] = df['transdatetime'].dt.quarter

df['amount_log'] = np.log1p(df['amount'])

df['total_device_changes'] = df['monthly_os_changes'] + df['monthly_phone_model_changes']

month_map = {
    'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04',
    'май': '05', 'июн': '06', 'июл': '07', 'авг': '08',
    'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12'
}
df['login_frequency_30d'] = df['login_frequency_30d'].apply(transform_frequency_column)

df["freq_change_7d_vs_mean"] = (
    df["freq_change_7d_vs_mean"]
    .astype(str)
    .str.strip()
    .str.replace(',', '.') # Заменяем запятые на точки
)
df["freq_change_7d_vs_mean"] = pd.to_numeric(df["freq_change_7d_vs_mean"], errors='coerce')

df['avg_30d_is_sentinel'] = (df['avg_login_interval_30d'] == -1).astype(int)
df['avg_login_interval_30d_log'] = np.log1p(df['avg_login_interval_30d'])
df.loc[df['avg_login_interval_30d_log'] < 0, 'avg_login_interval_30d_log'] = 0

df['std_30d_is_sentinel'] = (df['std_login_interval_30d'] == -1).astype(int)
df['std_login_interval_30d_log'] = np.log1p(df['std_login_interval_30d'])
df.loc[df['std_login_interval_30d_log'] < 0, 'std_login_interval_30d_log'] = 0

df['var_login_interval_30d'] = (
    df['var_login_interval_30d']
    .astype(str)
    .str.strip()
    .str.replace(',', '.')
)
df['var_login_interval_30d'] = pd.to_numeric(
    df['var_login_interval_30d'],
    errors='coerce'
)
df['var_30d_is_sentinel'] = (df['var_login_interval_30d'] == -1).astype(int)
df['var_login_interval_30d_log'] = np.log1p(df['var_login_interval_30d'])
df.loc[df['var_login_interval_30d_log'] < 0, 'var_login_interval_30d_log'] = 0

df['ewm_7d_is_sentinel'] = (df['ewm_login_interval_7d'] == -1).astype(int)
df['ewm_login_interval_7d_log'] = np.log1p(df['ewm_login_interval_7d'])
df.loc[df['ewm_login_interval_7d_log'] < 0, 'ewm_login_interval_7d_log'] = 0

df['fano_factor_is_sentinel'] = (df['fano_factor_login_interval'] == -1).astype(int)
df['fano_factor_login_interval_log'] = np.log1p(df['fano_factor_login_interval'])
df.loc[df['fano_factor_login_interval_log'] < 0, 'fano_factor_login_interval_log'] = 0


df['amount_x_hour'] = df['amount_log'] * df['transdate_hour']
df['amount_x_is_business'] = df['amount_log'] * df['transdate_is_business_hours']
df['amount_x_weekend'] = df['amount_log'] * (df['transdate_dayofweek'] >= 5).astype(int)

# Сезонность активности
df['zscore_x_hour'] = df['zscore_avg_login_interval_7d'] * df['transdate_hour']
df['zscore_x_day'] = df['zscore_avg_login_interval_7d'] * df['transdate_day']



df['amount_x_hour_x_quarter'] = df['amount_log'] * df['transdate_hour'] * df['transdate_quarter']
df['amount_x_zscore'] = df['amount_log'] * df['zscore_avg_login_interval_7d']
df['activity_volatility'] = df['burstiness_login_interval'] * df['zscore_avg_login_interval_7d']
df['suspicious_behavior_device'] = (
    (df['zscore_avg_login_interval_7d'] > 2) &
    (df['monthly_os_changes'] > 0)
).astype(int)
df['risk_profile'] = (
    df['zscore_avg_login_interval_7d'] * 0.3 +
    df['burstiness_login_interval'] * 0.3 +
    df['fano_factor_login_interval_log'] * 0.2 +
    df['monthly_os_changes'] * 0.2
)
df['amount_x_burstiness'] = df['amount_log'] * df['burstiness_login_interval']
df['month_x_week_x_quarter'] = df['transdate_month'] * df['transdate_week'] * df['transdate_quarter']
# =========================================================================================================================== #
# =========================================================================================================================== #

features_final = [
    # Временные
    "transdate_day",
    # "transdate_dayofweek",
    "transdate_hour",
    # "transdate_is_business_hours",
    # "transdate_year",
    "transdate_month",
    # "transdate_minute",
    "transdate_week",
    "transdate_quarter",

    # Сумма и Изменения Устройств
    "amount_log",
    # "total_device_changes",
    "monthly_os_changes",
    "monthly_phone_model_changes",

    # Исходные Поведенческие (тоже могут содержать NaN после to_numeric)
    "logins_last_7_days",
    "logins_last_30_days",
    # "login_frequency_7d",
    # "login_frequency_30d",
    "freq_change_7d_vs_mean",
    # "logins_7d_over_30d_ratio",
    "burstiness_login_interval",
    "zscore_avg_login_interval_7d",

    # Преобразованные Интервалы
    "avg_login_interval_30d_log",
    "std_login_interval_30d_log",
    # "var_login_interval_30d_log",
    "ewm_login_interval_7d_log",
    "fano_factor_login_interval_log",

    # Флаги Маркеров (-1)
    # "avg_30d_is_sentinel",
    # "std_30d_is_sentinel",
    # "var_30d_is_sentinel",
    # "ewm_7d_is_sentinel",
    # "fano_factor_is_sentinel",

    'amount_x_hour',
    'amount_x_is_business',
    # 'amount_x_weekend',

    # ПРОВЕРИТЬ ПОНИЖАЮТ ИЛИ НЕТ
    'zscore_x_hour',
    # 'zscore_x_day',

    'amount_x_hour_x_quarter',
    # 'amount_x_zscore',
    # 'activity_volatility',
    # 'suspicious_behavior_device',
    # 'risk_profile',
    # 'amount_x_burstiness',
    'month_x_week_x_quarter'
]

CATEGORICAL_FEATURES = [
    # Временные (цикличные/дискретные)
    # "transdate_dayofweek",
    "transdate_hour",
    "transdate_week",
    "transdate_day",
    "transdate_month",
    "transdate_quarter",

    # Счетчики (дискретные значения 0-N)
    "monthly_os_changes",
    "monthly_phone_model_changes",
    # "total_device_changes"
]

# =========================================================================================================================== #
# =========================================================================================================================== #

X = df[features_final]
y = df['target']

# === 3. РАЗДЕЛЕНИЕ ДАННЫХ (Train / Validation / Test) ===
# Шаг 1: Разделение на Обучение+Валидация и Тест (80%/20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Шаг 2: Разделение на Обучение и Валидацию (70%/10% от исходного набора)
# 10% от 80% (X_train_val) = 12.5%
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val
)

print(f"\nРазделение данных: Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
# =========================================================================================================================== #

# === 4. ИМПЬЮТАЦИЯ NaN И ПРИВЕДЕНИЕ ТИПОВ (Train/Val/Test) ===
print("\n=== ОБРАБОТКА NaN И ПРИВЕДЕНИЕ ТИПОВ ===")
imputer = SimpleImputer(strategy='median')

# 1. Обучаем импьютер только на TRAIN
X_train_imputed = imputer.fit_transform(X_train)

# 2. Трансформируем TRAIN, VAL и TEST
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

# Возвращаем в DataFrame
X_train_final = pd.DataFrame(X_train_imputed, columns=features_final)
X_val_final = pd.DataFrame(X_val_imputed, columns=features_final)
X_test_final = pd.DataFrame(X_test_imputed, columns=features_final)

# 3. Приведение категориальных типов к INT
for col in CATEGORICAL_FEATURES:
    for df_final in [X_train_final, X_val_final, X_test_final]:
        df_final[col] = df_final[col].round(0).astype(int)

print("Типы данных успешно преобразованы.")
# =========================================================================================================================== #

# =========================================================================================================================== #
#               RFECV - для выбора признака
# =========================================================================================================================== #
# import pandas as pd
# from lightgbm import LGBMClassifier
# from sklearn.feature_selection import RFECV
# from sklearn.model_selection import StratifiedKFold # Для надежной CV
# from sklearn.metrics import roc_auc_score
# import numpy as np
#
# # 1. Определение списка ВСЕХ текущих признаков
# # Убедитесь, что здесь все ваши новые признаки взаимодействия и агрегации
# features_final = X_train_final.columns.tolist()
#
# # 2. Инициализация Базовой Модели (LGBM)
# # LGBM быстр, что критически важно для RFECV, который много раз переобучает модель.
# # Устанавливаем параметры, которые уже близки к оптимуму (низкий LR, средняя глубина)
# lgbm_model = LGBMClassifier(
#     n_estimators=500,           # Количество деревьев (итераций)
#     learning_rate=0.03,
#     max_depth=6,
#     # Применяем ручное взвешивание классов (для LGBM это weight, а не class_weights)
#     # 5925 / 88 ≈ 67.3. Округлим до 70.
#     scale_pos_weight=70,
#     random_state=42,
#     n_jobs=-1, # Используем все ядра
#     verbose=-1 # Отключаем вывод в цикле
# )
#
# # 3. Настройка RFECV
# # Используем StratifiedKFold для сохранения дисбаланса классов при CV
# # cv=5 означает 5-кратная кросс-валидация
# rfecv = RFECV(
#     estimator=lgbm_model,
#     step=1, # Удалять по одному признаку на каждой итерации
#     cv=StratifiedKFold(5),
#     scoring='roc_auc',
#     n_jobs=-1
# )
#
# # 4. Запуск RFECV на обучающей выборке
# # RFECV автоматически выполнит обучение, удаление и CV
# print("🤖 Запуск RFECV...")
# rfecv.fit(X_train_final[features_final], y_train)
#
# # 5. Получение Оптимального Набора
# selected_features = [f for f, selected in zip(features_final, rfecv.support_) if selected]
#
# selected_features = [
#     f for f, selected in zip(features_final, rfecv.support_) if selected
# ]
#
# print(f"--- 🎯 Отобранные {len(selected_features)} Признаков ---")
# for i, feature in enumerate(selected_features, 1):
#     print(f"{i}. {feature}")
#
# print("\n--- РЕЗУЛЬТАТ RFECV ---")
# print(f"🤖 RFECV отобрал {len(selected_features)} оптимальных признаков.")
# print(f"Максимальный средний AUC (CV): {rfecv.cv_results_['mean_test_score'].max():.4f}")
#
# # 6. Визуализация Кривой RFECV
# # Строится кривая зависимости AUC от количества признаков
#
#
# # (Этот график показывает, после какого числа признаков AUC начинает падать)
# =========================================================================================================================== #
# =========================================================================================================================== #

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

model_w = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.01,
    depth=7,
    loss_function='Logloss',
    eval_metric='AUC',
    l2_leaf_reg=7,
    random_strength=2,
    bagging_temperature=1.5,


    class_weights=[1, 100],
    # auto_class_weights='Balanced', #'SqrtBalanced' - тоже можно попробовать, но база 'Balanced'

    early_stopping_rounds=500,
    verbose=200,
    cat_features=CATEGORICAL_FEATURES,
    random_seed=42,

)

# model_w.fit(X_train_final, y_train)

model_w.fit(
        X_train_final, y_train,
        eval_set=(X_val_final, y_val),
        use_best_model=True,
        verbose=200
    )

# preds = model_w.predict(X_test_final)
# probs = model_w.predict_proba(X_test_final)[:, 1]

preds_val = model_w.predict(X_val_final)
probs_val = model_w.predict_proba(X_val_final)[:, 1]

# Метки ВАЛИДАЦИИ (сбросим индекс, чтобы совпал с вероятностями)
y_val_aligned = y_val.reset_index(drop=True)

results = pd.DataFrame({
    'Actual_Target': y_val_aligned, # Используем y_val
    'Prob_Fraud': probs_val
})

# Смотрим на 5 случаев с самой высокой предсказанной вероятностью мошенничества
print("\n=== ТОП-5 ПРОГНОЗОВ МОШЕННИЧЕСТВА (на VAL) ===")
print(results.sort_values(by='Prob_Fraud', ascending=False).head(5).to_string(index=False))

# =========================================================================================================================== #
#                   ИЩЕМ ПОРОГ ВХОДА ДЛЯ МОШЕННИЧЕСКОЙ ТРАНЗАКЦИИ
# =========================================================================================================================== #

# Находим лучший порог для максимизации F1-score (общего баланса)
best_f1 = 0
best_thresh = 0.5

for thresh in np.arange(0.005, 0.3, 0.005):
    current_preds = (probs_val > thresh).astype(int)  # Используем probs_val

    if 1 in current_preds:
        # ИСПРАВЛЕНИЕ: Сравниваем ВАЛ-ВЕРОЯТНОСТИ (probs_val) с ВАЛ-МЕТКАМИ (y_val_aligned)
        current_f1 = f1_score(y_val_aligned, current_preds, average='binary', pos_label=1)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

print(f"\nНаилучший F1-score ({best_f1:.4f}) достигнут на VAL при пороге: {best_thresh:.3f}")

# preds_new = (probs > best_thresh).astype(int)
#
# # =========================================================================================================================== #
# #                   Тест своего порога
# # =========================================================================================================================== #
# optimal_threshold = 0.012
# probs = model_w.predict_proba(X_test_final)[:, 1]
# preds_final = (probs >= optimal_threshold).astype(int)
# print("\n=== РЕЗУЛЬТАТЫ С СОБСТВЕННЫМ ПОРОГОМ ===")
# print("AUC:", roc_auc_score(y_test, probs))
# print(classification_report(y_test, preds_final))
# # =========================================================================================================================== #
# # =========================================================================================================================== #
#
# print("\n=== РЕЗУЛЬТАТЫ С ОПТИМАЛЬНЫМ ПОРОГОМ ===")
# print(classification_report(y_test, preds_new))
#
# print("\n=== РЕЗУЛЬТАТЫ С СТАНДАРТНЫМ ПОРОГОМ ===")
# print("AUC:", roc_auc_score(y_test, probs))
# print(classification_report(y_test, preds))

# =========================================================================================================================== #
#                   Тест своего порога
# =========================================================================================================================== #


probs_test = model_w.predict_proba(X_test_final)[:, 1] # Получаем вероятности на ТЕСТ
y_test_aligned = y_test.reset_index(drop=True) # Метки ТЕСТА для сравнения

# 1. ТЕСТ С СОБСТВЕННЫМ ПОРОГОМ (0.1)
optimal_threshold = 0.1
preds_custom = (probs_test >= optimal_threshold).astype(int)
print("\n=== РЕЗУЛЬТАТЫ С СОБСТВЕННЫМ ПОРОГОМ (0.1) ===")
print("AUC:", roc_auc_score(y_test_aligned, probs_test))
print(classification_report(y_test_aligned, preds_custom))
cm_custom = confusion_matrix(y_test_aligned, preds_custom)
print("Матрица ошибок (порог 0.1):\n", cm_custom)
print(f"Обнаружено мошенничества: {cm_custom[1, 1]}/{cm_custom[1].sum()}")
# =========================================================================================================================== #
# =========================================================================================================================== #
# 2. ТЕСТ С ОПТИМАЛЬНЫМ ПОРОГОМ (найденным на VAL)
preds_optimal = (probs_test > best_thresh).astype(int)
print("\n=== РЕЗУЛЬТАТЫ С ОПТИМАЛЬНЫМ ПОРОГОМ (VAL F1-score) ===")
print("AUC:", roc_auc_score(y_test_aligned, probs_test))
print(f"Оптимальный порог: {best_thresh:.4f}")
print(classification_report(y_test_aligned, preds_optimal))
cm_optimal = confusion_matrix(y_test_aligned, preds_optimal)
print("Матрица ошибок (оптимальный порог):\n", cm_optimal)
print(f"Обнаружено мошенничества: {cm_optimal[1, 1]}/{cm_optimal[1].sum()}")
# =========================================================================================================================== #
# =========================================================================================================================== #

all_features = X_train_final.columns.tolist()
feature_importances_array = model_w.get_feature_importance()
importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": feature_importances_array
}).sort_values(by="importance", ascending=False)
print("\n🔝 ТОП-20 ВАЖНЫХ ПРИЗНАКОВ:")
print(importance_df.head(20).to_string(index=False))