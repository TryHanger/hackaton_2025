import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

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
    "transdate_minute",
    "transdate_week",
    # "transdate_quarter",

    # Сумма и Изменения Устройств
    "amount_log",
    # "total_device_changes",
    # "monthly_os_changes",
    # "monthly_phone_model_changes",

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
    # "std_login_interval_30d_log",
    # "var_login_interval_30d_log",
    "ewm_login_interval_7d_log",
    "fano_factor_login_interval_log",

    # Флаги Маркеров (-1)
    # "avg_30d_is_sentinel", "std_30d_is_sentinel",
    # "var_30d_is_sentinel", "ewm_7d_is_sentinel",
    # "fano_factor_is_sentinel"
    
    'amount_x_hour',
    'amount_x_is_business',
    'amount_x_weekend',
    'zscore_x_hour',
    'zscore_x_day',
]

CATEGORICAL_FEATURES = [
    # Временные (цикличные/дискретные)
    # "transdate_dayofweek",
    "transdate_hour",
    "transdate_week",
    "transdate_day",
    "transdate_month",
    # "transdate_quarter",

    # Счетчики (дискретные значения 0-N)
    # "monthly_os_changes",
    # "monthly_phone_model_changes",
    # "total_device_changes"
]

# =========================================================================================================================== #
# =========================================================================================================================== #

X = df[features_final]
y = df['target']

from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
#
# smote = SMOTE(
#     sampling_strategy='auto',  # Стратегия выборки. 'auto' означает увеличение меньшего класса до размера большинственного.
#     random_state=None,         # Зерно для генератора случайных чисел.
#     k_neighbors=5,             # Количество ближайших соседей для создания синтетических примеров.
#     n_jobs=1                   # Количество ядер для параллельной работы. -1 означает использование всех доступных ядер.
# )

# Предположим, что X — все признаки, y — бинарная целевая переменная (0/1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# =========================================================================================================================== #
# 🛑 НОВЫЙ КРИТИЧЕСКИЙ БЛОК: ИМПЬЮТАЦИЯ (МЕДИАНА) И ПРИВЕДЕНИЕ ТИПОВ
# =========================================================================================================================== #
from sklearn.impute import SimpleImputer

print("\n=== ОБРАБОТКА NaN И ПРИВЕДЕНИЕ ТИПОВ ===")
# 1. Импьютация NaN с помощью SimpleImputer (медиана)
imputer = SimpleImputer(strategy='median')

# Обучаем импьютер только на обучающей выборке и трансформируем обе
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Возвращаем в DataFrame для удобства и сохранения имен столбцов
X_train_final = pd.DataFrame(X_train_imputed, columns=features_final)
X_test_final = pd.DataFrame(X_test_imputed, columns=features_final)


# 2. Приведение категориальных типов к INT
# Все столбцы после импьютера - float. CatBoost требует int/str для категорий.
for col in CATEGORICAL_FEATURES:
    # Округляем до ближайшего целого (чтобы 4.0 стало 4) и приводим к типу 'int'
    X_train_final[col] = X_train_final[col].round(0).astype(int)
    X_test_final[col] = X_test_final[col].round(0).astype(int)

print("Типы данных успешно преобразованы.")
print(X_train_final[CATEGORICAL_FEATURES].dtypes)
# =========================================================================================================================== #

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

model_w = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    # class_weights=[1, 67],
    # class_weights=[1, 85],
    class_weights=[1, 100],
    # auto_class_weights='Balanced', #'SqrtBalanced' - тоже можно попробовать, но база 'Balanced'
    verbose=200,
    cat_features=CATEGORICAL_FEATURES,
    # task_type='CPU',
    # thread_count=-1,
)

model_w.fit(X_train_final, y_train, early_stopping_rounds=100)

preds = model_w.predict(X_test_final)
probs = model_w.predict_proba(X_test_final)[:, 1]

results = pd.DataFrame({
    'Actual_Target': y_test.reset_index(drop=True),
    'Prob_Fraud': probs
})

# Смотрим на 5 случаев с самой высокой предсказанной вероятностью мошенничества
print("\n=== ТОП-5 ПРОГНОЗОВ МОШЕННИЧЕСТВА ===")
print(results.sort_values(by='Prob_Fraud', ascending=False))

# =========================================================================================================================== #
#                   ИЩЕМ ПОРОГ ВХОДА ДЛЯ МОШЕННИЧЕСКОЙ ТРАНЗАКЦИИ
# =========================================================================================================================== #

# Находим лучший порог для максимизации F1-score (общего баланса)
best_f1 = 0
best_thresh = 0.5

# Перебираем пороги от 0.005 до 0.3 с шагом 0.005
# Мы начали с очень низкого порога, чтобы учесть низкие Prob_Fraud
for thresh in np.arange(0.005, 0.3, 0.005):
    # Преобразование вероятностей в классы с новым порогом
    current_preds = (probs > thresh).astype(int)

    # Защита от деления на ноль, если текущий порог слишком высок и не находит класс 1
    if 1 in current_preds:
        # Вычисление F1-score для класса 1 (мошенничество)
        current_f1 = f1_score(y_test, current_preds, average='binary', pos_label=1)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

print(f"\nНаилучший F1-score ({best_f1:.4f}) достигнут при пороге: {best_thresh:.3f}")

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
optimal_threshold = 0.025
# optimal_threshold = 0.012
probs_custom = model_w.predict_proba(X_test_final)[:, 1]
preds_custom = (probs_custom >= optimal_threshold).astype(int)
print("\n=== РЕЗУЛЬТАТЫ С СОБСТВЕННЫМ ПОРОГОМ (0.025) ===")
print("AUC:", roc_auc_score(y_test, probs_custom))
print(classification_report(y_test, preds_custom))

# Матрица ошибок для собственного порога
cm_custom = confusion_matrix(y_test, preds_custom)
print("Матрица ошибок (порог 0.025):")
print(cm_custom)
print(f"Обнаружено мошенничества: {cm_custom[1, 1]}/{cm_custom[1].sum()}")
# =========================================================================================================================== #
# =========================================================================================================================== #
print("\n=== РЕЗУЛЬТАТЫ С ОПТИМАЛЬНЫМ ПОРОГОМ ===")
# Предполагаем, что best_thresh уже вычислен где-то ранее
probs_optimal = model_w.predict_proba(X_test_final)[:, 1]
preds_optimal = (probs_optimal > best_thresh).astype(int)
print("AUC:", roc_auc_score(y_test, probs_optimal))
print(f"Оптимальный порог: {best_thresh:.4f}")
print(classification_report(y_test, preds_optimal))

# Матрица ошибок для оптимального порога
cm_optimal = confusion_matrix(y_test, preds_optimal)
print("Матрица ошибок (оптимальный порог):")
print(cm_optimal)
print(f"Обнаружено мошенничества: {cm_optimal[1, 1]}/{cm_optimal[1].sum()}")
# =========================================================================================================================== #
# =========================================================================================================================== #
print("\n=== РЕЗУЛЬТАТЫ С СТАНДАРТНЫМ ПОРОГОМ (0.5) ===")
probs_standard = model_w.predict_proba(X_test_final)[:, 1]
preds_standard = (probs_standard > 0.5).astype(int)
print("AUC:", roc_auc_score(y_test, probs_standard))
print(classification_report(y_test, preds_standard))

# Матрица ошибок для стандартного порога
cm_standard = confusion_matrix(y_test, preds_standard)
print("Матрица ошибок (стандартный порог 0.5):")
print(cm_standard)
print(f"Обнаружено мошенничества: {cm_standard[1, 1]}/{cm_standard[1].sum()}")



all_features = X_train_final.columns.tolist()
feature_importances_array = model_w.get_feature_importance()
importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": feature_importances_array
}).sort_values(by="importance", ascending=False)
print("\n🔝 ТОП-20 ВАЖНЫХ ПРИЗНАКОВ:")
print(importance_df.head(20).to_string(index=False))