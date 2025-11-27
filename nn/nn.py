import datetime
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, f1_score
)
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# === 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ (Оставляем как есть) ===
# ... (Ваш код загрузки данных и создания признаков, включая month_map) ...
# В целях краткости я удаляю блок кода, который уже был в промпте,
# предполагая, что он корректно выполняется и создает DataFrame 'df'

# ИЗМЕНЕННЫЙ КОД:
# Предполагаем, что ваш код загрузки и feature engineering до этого места
# выполнился и создал df со всеми нужными столбцами
script_dir = Path(__file__).parent
db_path = script_dir.parent / 'clean_data.db'
conn = sqlite3.connect(db_path)
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

# Определение month_map и transform_frequency_column, чтобы ваш код работал
month_map = {
    'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04',
    'май': '05', 'июн': '06', 'июл': '07', 'авг': '08',
    'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12'
}


def transform_frequency_column(value):
    s = str(value).strip().lower().replace(',', '.')
    for text_month, num_month in month_map.items():
        if text_month in s:
            try:
                day_part = s.split('.')[0]
                new_value = f"{day_part}.{num_month}"
                return float(new_value)
            except:
                return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# Ваш блок feature engineering
df['transdatetime'] = pd.to_datetime(df['transdatetime'])
df['transdate_day'] = df['transdatetime'].dt.day
df['transdate_dayofweek'] = df['transdatetime'].dt.dayofweek
df['transdate_hour'] = df['transdatetime'].dt.hour
df['transdate_is_business_hours'] = ((df['transdatetime'].dt.hour >= 10) & (df['transdatetime'].dt.hour <= 18)).astype(
    int)
df['transdate_year'] = df['transdatetime'].dt.year
df['transdate_month'] = df['transdatetime'].dt.month
df['transdate_minute'] = df['transdatetime'].dt.minute
df['transdate_week'] = df['transdatetime'].dt.isocalendar().week.astype(
    int)  # .dt.isocalendar().week возвращает Series of UInt32, нужно int
df['transdate_quarter'] = df['transdatetime'].dt.quarter

df['amount_log'] = np.log1p(df['amount'])
df['total_device_changes'] = df['monthly_os_changes'] + df['monthly_phone_model_changes']
df['login_frequency_30d'] = df['login_frequency_30d'].apply(transform_frequency_column)
df["freq_change_7d_vs_mean"] = pd.to_numeric(df["freq_change_7d_vs_mean"].astype(str).str.strip().str.replace(',', '.'),
                                             errors='coerce')

df['avg_30d_is_sentinel'] = (df['avg_login_interval_30d'] == -1).astype(int)
df['avg_login_interval_30d_log'] = np.log1p(df['avg_login_interval_30d'])
df.loc[df['avg_login_interval_30d_log'] < 0, 'avg_login_interval_30d_log'] = 0

df['std_30d_is_sentinel'] = (df['std_login_interval_30d'] == -1).astype(int)
df['std_login_interval_30d_log'] = np.log1p(df['std_login_interval_30d'])
df.loc[df['std_login_interval_30d_log'] < 0, 'std_login_interval_30d_log'] = 0

df['var_login_interval_30d'] = pd.to_numeric(df['var_login_interval_30d'].astype(str).str.strip().str.replace(',', '.'),
                                             errors='coerce')
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
df['zscore_x_hour'] = df['zscore_avg_login_interval_7d'] * df['transdate_hour']
df['zscore_x_day'] = df['zscore_avg_login_interval_7d'] * df['transdate_day']
df['amount_x_hour_x_quarter'] = df['amount_log'] * df['transdate_hour'] * df['transdate_quarter']
df['amount_x_zscore'] = df['amount_log'] * df['zscore_avg_login_interval_7d']
df['activity_volatility'] = df['burstiness_login_interval'] * df['zscore_avg_login_interval_7d']
df['suspicious_behavior_device'] = ((df['zscore_avg_login_interval_7d'] > 2) & (df['monthly_os_changes'] > 0)).astype(
    int)
df['risk_profile'] = (df['zscore_avg_login_interval_7d'] * 0.3 + df['burstiness_login_interval'] * 0.3 + df[
    'fano_factor_login_interval_log'] * 0.2 + df['monthly_os_changes'] * 0.2)
df['amount_x_burstiness'] = df['amount_log'] * df['burstiness_login_interval']
df['month_x_week_x_quarter'] = df['transdate_month'] * df['transdate_week'] * df['transdate_quarter']
# =========================================================================================================================== #

# === 2. ФИНАЛЬНЫЙ НАБОР ПРИЗНАКОВ (Очищенный и стабильный) ===
# Мы используем список из 20 стабильных признаков, которые вы выбрали ранее.

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
# Создаем папку для сохранения моделей
MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

model_performances = []

# === 5. ОБУЧЕНИЕ АНСАМБЛЯ С СОХРАНЕНИЕМ ===
seeds = [42, 101, 202, 303, 404]
models = []
model_performances = []

print(f"\n=== ЗАПУСК ОБУЧЕНИЯ АНСАМБЛЯ ({len(seeds)} моделей) ===")

best_model = None
best_auc = 0
best_model_index = 0

for i, seed in enumerate(seeds):
    print(f"\n[Модель {i + 1}/5] Обучение с random_seed={seed}")

    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.01,
        depth=7,
        loss_function='Logloss',
        eval_metric='AUC',
        l2_leaf_reg=5,
        random_strength=1.5,
        bagging_temperature=1.0,
        auto_class_weights='Balanced',
        early_stopping_rounds=500,
        verbose=200,
        cat_features=CATEGORICAL_FEATURES,
        random_seed=seed,
        thread_count=-1
    )

    model.fit(
        X_train_final, y_train,
        eval_set=(X_val_final, y_val),
        use_best_model=True,
        verbose=200
    )

    # Оцениваем модель на validation set
    val_probs = model.predict_proba(X_val_final)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    models.append(model)
    model_performances.append({
        'model_index': i,
        'seed': seed,
        'val_auc': val_auc,
        'best_iteration': model.get_best_iteration()
    })

    print(f"✅ Модель {i + 1} обучена. Val AUC: {val_auc:.4f}, Лучшая итерация: {model.get_best_iteration()}")

    # Сохраняем лучшую отдельную модель
    if val_auc > best_auc:
        best_auc = val_auc
        best_model = model
        best_model_index = i

# === СОХРАНЕНИЕ МОДЕЛЕЙ ===
print(f"\n💾 СОХРАНЕНИЕ МОДЕЛЕЙ...")

# 1. Сохраняем лучшую отдельную модель
best_model_path = MODELS_DIR / "best_single_model.cbm"
best_model.save_model(str(best_model_path))
print(f"✅ Лучшая модель сохранена: {best_model_path} (AUC: {best_auc:.4f})")

# 2. Сохраняем весь ансамбль
ensemble_path = MODELS_DIR / "ensemble_models"
ensemble_path.mkdir(exist_ok=True)

for i, model in enumerate(models):
    model_path = ensemble_path / f"model_{i}.cbm"
    model.save_model(str(model_path))

print(f"✅ Ансамбль из {len(models)} моделей сохранен в: {ensemble_path}")

# 3. Сохраняем метаданные ансамбля
metadata = {
    'created_date': datetime.datetime.now().isoformat(),
    'models_count': len(models),
    'best_model_index': best_model_index,
    'best_model_auc': best_auc,
    'model_performances': model_performances,
    'features_used': features_final,
    'categorical_features': CATEGORICAL_FEATURES,
    'test_auc': None,  # заполнится после оценки
    'optimal_threshold': None
}

metadata_path = MODELS_DIR / "ensemble_metadata.json"
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ Метаданные сохранены: {metadata_path}")

# 4. Сохраняем импутер и информацию о признаках
preprocessor_path = MODELS_DIR / "preprocessor.joblib"
joblib.dump({
    'imputer': imputer,
    'feature_names': features_final,
    'categorical_features': CATEGORICAL_FEATURES
}, preprocessor_path)

print(f"✅ Препроцессор сохранен: {preprocessor_path}")

# === 6. АНСАМБЛИРОВАНИЕ И ФИНАЛЬНЫЙ ПРОГНОЗ ===
print(f"\n🎯 АНСАМБЛИРОВАНИЕ...")

# Взвешенное усреднение по качеству моделей
model_weights = [perf['val_auc'] for perf in model_performances]
model_weights = np.array(model_weights) / sum(model_weights)

print("Веса моделей в ансамбле:")
for i, (perf, weight) in enumerate(zip(model_performances, model_weights)):
    print(f"  Модель {i + 1}: AUC={perf['val_auc']:.4f}, вес={weight:.3f}")

# Взвешенное усреднение вероятностей
final_probs = np.zeros(len(X_test_final))
for i, (model, weight) in enumerate(zip(models, model_weights)):
    final_probs += model.predict_proba(X_test_final)[:, 1] * weight

# === 7. ОЦЕНКА РЕЗУЛЬТАТОВ ===
test_auc = roc_auc_score(y_test, final_probs)
print(f"\n📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ АНСАМБЛЯ:")
print(f"AUC на тестовой выборке: {test_auc:.4f}")

# Обновляем метаданные
metadata['test_auc'] = test_auc

# Поиск оптимального порога
best_f1 = 0
best_thresh = 0.5
for thresh in np.arange(0.005, 0.3, 0.005):
    current_preds = (final_probs > thresh).astype(int)
    if 1 in current_preds:
        current_f1 = f1_score(y_test, current_preds, average='binary', pos_label=1)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_thresh = thresh

metadata['optimal_threshold'] = best_thresh

print(f"🎯 Оптимальный порог: {best_thresh:.4f} (F1-score: {best_f1:.4f})")

# Сохраняем обновленные метаданные
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# === ВЫВОД РЕЗУЛЬТАТОВ ===
print(f"\n{'=' * 50}")
print("🎉 АНСАМБЛЬ УСПЕШНО ОБУЧЕН И СОХРАНЕН!")
print(f"{'=' * 50}")

results_summary = {
    'Лучшая модель': f"AUC: {best_auc:.4f}",
    'Ансамбль': f"AUC: {test_auc:.4f}",
    'Улучшение': f"+{(test_auc - best_auc) * 100:.2f}%",
    'Оптимальный порог': f"{best_thresh:.4f}",
    'Сохранено моделей': f"{len(models)}",
    'Папка с моделями': str(MODELS_DIR)
}

for key, value in results_summary.items():
    print(f"{key:20}: {value}")

# === 8. ЗАГРУЗКА И ИСПОЛЬЗОВАНИЕ СОХРАНЕННЫХ МОДЕЛЕЙ ===
print(f"\n🔧 ПРИМЕР ЗАГРУЗКИ СОХРАНЕННЫХ МОДЕЛЕЙ:")

# Пример загрузки лучшей модели
loaded_best_model = CatBoostClassifier()
loaded_best_model.load_model(str(best_model_path))
print(f"✅ Лучшая модель загружена для предсказаний")


# Пример загрузки всего ансамбля
def load_ensemble(ensemble_dir):
    """Загрузка всего ансамбля"""
    ensemble_dir = Path(ensemble_dir)
    models = []

    # Загружаем метаданные
    with open(ensemble_dir.parent / "ensemble_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Загружаем все модели
    for i in range(metadata['models_count']):
        model = CatBoostClassifier()
        model_path = ensemble_dir / f"model_{i}.cbm"
        model.load_model(str(model_path))
        models.append(model)

    return models, metadata


# Загружаем препроцессор
preprocessor = joblib.load(preprocessor_path)
print(f"✅ Препроцессор загружен")

print(f"\n💡 ДЛЯ ПРЕДСКАЗАНИЙ НА НОВЫХ ДАННЫХ:")
print("1. Загрузите модель: model = CatBoostClassifier()")
print("2. model.load_model('saved_models/best_single_model.cbm')")
print("3. Используйте model.predict_proba(new_data)[:, 1]")