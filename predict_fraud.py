"""
Скрипт для предсказания мошенничества на основе JSON транзакций.
Использует модель, обученную в CatBoost.py
"""

import joblib
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from pathlib import Path
from typing import List, Dict, Any

# --- Настройки ---
MODELS_DIR = Path("saved_models")

# --- КОНСТАНТЫ ИЗ CatBoost.py ---

# Отобранный финальный список признаков (должен совпадать с features_final из CatBoost.py)
FEATURES_FINAL = [
    "transdate_day",
    "transdate_hour",
    "transdate_month",
    "transdate_week",
    "transdate_quarter",
    "amount_log",
    "monthly_os_changes",
    "monthly_phone_model_changes",
    "logins_last_7_days",
    "logins_last_30_days",
    "freq_change_7d_vs_mean",
    "burstiness_login_interval",
    "zscore_avg_login_interval_7d",
    "avg_login_interval_30d_log",
    "std_login_interval_30d_log",
    "ewm_login_interval_7d_log",
    "fano_factor_login_interval_log",
    'amount_x_hour',
    'amount_x_is_business',
    'zscore_x_hour',
    'amount_x_hour_x_quarter',
    'month_x_week_x_quarter'
]

# Список категориальных признаков для приведения к int
CATEGORICAL_FEATURES_USED = [
    "transdate_hour",
    "transdate_week",
    "transdate_day",
    "transdate_month",
    "transdate_quarter",
    "monthly_os_changes",
    "monthly_phone_model_changes",
]

# Карта месяцев для преобразования строк
MONTH_MAP = {
    'янв': '01', 'фев': '02', 'мар': '03', 'апр': '04',
    'май': '05', 'июн': '06', 'июл': '07', 'авг': '08',
    'сен': '09', 'окт': '10', 'ноя': '11', 'дек': '12'
}


def load_latest_model_and_preprocessor():
    """Загружает последнюю сохраненную модель, импьютер и метаданные."""
    
    # 1. Поиск последней версии метаданных
    metadata_files = sorted(MODELS_DIR.glob("metadata_*.json"), reverse=True)
    if not metadata_files:
        raise FileNotFoundError(f"Метаданные не найдены в папке {MODELS_DIR.resolve()}. Запустите CatBoost.py для обучения модели.")
    
    metadata_path = metadata_files[0]
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Извлекаем временную метку: metadata_20251128_030038.json -> 20251128_030038
    # Убираем префикс "metadata_" и расширение ".json"
    time_str = metadata_path.stem.replace('metadata_', '')
    
    # 2. Формируем имена файлов на основе метки времени
    model_name = f"catboost_single_model_{time_str}.cbm"
    imputer_name = f"imputer_{time_str}.joblib"
    
    model_path = MODELS_DIR / model_name
    imputer_path = MODELS_DIR / imputer_name
    
    if not model_path.exists() or not imputer_path.exists():
        raise FileNotFoundError(f"Файлы модели или импьютера не найдены: {model_path.name}, {imputer_path.name}")
    
    # 3. Загрузка импьютера
    imputer = joblib.load(imputer_path)
    
    # 4. Загрузка CatBoost модели
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    
    print(f"OK: Модель '{model_path.name}' и препроцессор загружены успешно.")
    print(f"OK: Оптимальный порог из метаданных: {metadata.get('optimal_f1_threshold', 0.1):.4f}")
    
    return model, imputer, metadata.get('features_used', FEATURES_FINAL), metadata.get('optimal_f1_threshold', 0.1)


def transform_frequency_column(value):
    """
    Преобразует строку, содержащую дату (например, '01.мар'), в числовой формат (например, 1.03).
    Воспроизводит логику из CatBoost.py
    """
    if pd.isna(value):
        return np.nan
    
    s = str(value)
    s = s.strip().lower()
    
    # Проверка, содержит ли строка месяцы
    for text_month, num_month in MONTH_MAP.items():
        if text_month in s:
            try:
                day_part = s.split('.')[0]
                new_value = f"{day_part}.{num_month}"
                return float(new_value)
            except:
                return np.nan
    
    # Обработка корректных числовых строк (с запятой или точкой)
    s = s.replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return np.nan


def prepare_json_data(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Принимает список JSON-объектов (транзакций), выполняет все шаги
    Feature Engineering из CatBoost.py и возвращает предобработанный DataFrame.
    """
    
    if not isinstance(raw_data, list) or not raw_data:
        return pd.DataFrame(columns=FEATURES_FINAL)
    
    df = pd.DataFrame(raw_data)
    
    # 1. ОБРАБОТКА ДАТЫ И ВРЕМЕНИ (как в CatBoost.py строки 98-109)
    if 'transdatetime' in df.columns:
        df['transdatetime'] = pd.to_datetime(df['transdatetime'], errors='coerce')
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
    
    # 2. ПРЕОБРАЗОВАНИЕ СУММЫ (строка 111)
    if 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])
    
    # 3. ОБРАБОТКА ИЗМЕНЕНИЙ УСТРОЙСТВ (строка 113)
    if 'monthly_os_changes' in df.columns and 'monthly_phone_model_changes' in df.columns:
        df['total_device_changes'] = df['monthly_os_changes'] + df['monthly_phone_model_changes']
    
    # 4. ОБРАБОТКА 'login_frequency_30d' (строка 120)
    if 'login_frequency_30d' in df.columns:
        df['login_frequency_30d'] = df['login_frequency_30d'].apply(transform_frequency_column)
    
    # 5. ОБРАБОТКА 'freq_change_7d_vs_mean' (строки 122-128)
    if 'freq_change_7d_vs_mean' in df.columns:
        df["freq_change_7d_vs_mean"] = (
            df["freq_change_7d_vs_mean"]
            .astype(str)
            .str.strip()
            .str.replace(',', '.')
        )
        df["freq_change_7d_vs_mean"] = pd.to_numeric(df["freq_change_7d_vs_mean"], errors='coerce')
    
    # 6. ЛОГАРИФМИРОВАНИЕ ПОВЕДЕНЧЕСКИХ ИНТЕРВАЛОВ (строки 130-158)
    # Обработка sentinel значения -1: заменяем на NaN перед логарифмированием
    interval_cols = [
        ('avg_login_interval_30d', 'avg_login_interval_30d_log'),
        ('std_login_interval_30d', 'std_login_interval_30d_log'),
        ('ewm_login_interval_7d', 'ewm_login_interval_7d_log'),
        ('fano_factor_login_interval', 'fano_factor_login_interval_log'),
    ]
    
    for src, dst in interval_cols:
        if src in df.columns:
            # Обработка -1: заменяем на NaN (как в CatBoost.py, но там используется np.log1p напрямую)
            # В CatBoost.py: df['avg_login_interval_30d_log'] = np.log1p(df['avg_login_interval_30d'])
            # Но -1 после log1p даст log(0) = -inf, поэтому нужно обработать
            df[src] = df[src].replace(-1, np.nan)
            df[dst] = df[src].apply(lambda x: np.log1p(x) if pd.notna(x) and x >= 0 else np.nan)
            df.loc[df[dst] < 0, dst] = 0
    
    # Обработка var_login_interval_30d (строки 138-150) - не используется в features_final, но может понадобиться
    if 'var_login_interval_30d' in df.columns:
        df['var_login_interval_30d'] = (
            df['var_login_interval_30d']
            .astype(str)
            .str.strip()
            .str.replace(',', '.')
        )
        df['var_login_interval_30d'] = pd.to_numeric(df['var_login_interval_30d'], errors='coerce')
    
    # 7. ПРИЗНАКИ ВЗАИМОДЕЙСТВИЯ (строки 161-185)
    if all(col in df.columns for col in ['amount_log', 'transdate_hour']):
        df['amount_x_hour'] = df['amount_log'] * df['transdate_hour']
    
    if all(col in df.columns for col in ['amount_log', 'transdate_is_business_hours']):
        df['amount_x_is_business'] = df['amount_log'] * df['transdate_is_business_hours']
    
    if all(col in df.columns for col in ['amount_log', 'transdate_dayofweek']):
        df['amount_x_weekend'] = df['amount_log'] * (df['transdate_dayofweek'] >= 5).astype(int)
    
    if all(col in df.columns for col in ['zscore_avg_login_interval_7d', 'transdate_hour']):
        df['zscore_x_hour'] = df['zscore_avg_login_interval_7d'] * df['transdate_hour']
    
    if all(col in df.columns for col in ['zscore_avg_login_interval_7d', 'transdate_day']):
        df['zscore_x_day'] = df['zscore_avg_login_interval_7d'] * df['transdate_day']
    
    if all(col in df.columns for col in ['amount_log', 'transdate_hour', 'transdate_quarter']):
        df['amount_x_hour_x_quarter'] = df['amount_log'] * df['transdate_hour'] * df['transdate_quarter']
    
    if all(col in df.columns for col in ['transdate_month', 'transdate_week', 'transdate_quarter']):
        df['month_x_week_x_quarter'] = df['transdate_month'] * df['transdate_week'] * df['transdate_quarter']
        df['month_x_week_x_quarter'] = df['month_x_week_x_quarter'].astype(float)
    
    # 8. ФИНАЛЬНЫЙ ОТБОР ПРИЗНАКОВ И ПРИВЕДЕНИЕ ТИПОВ
    
    # Проверка: все ли признаки созданы
    missing_features = set(FEATURES_FINAL) - set(df.columns)
    if missing_features:
        print(f"ВНИМАНИЕ: Отсутствуют признаки перед финальным отбором: {missing_features}")
        # Заполняем отсутствующие признаки нулями
        for feat in missing_features:
            df[feat] = 0
    
    # Оставляем только те признаки, которые используются моделью
    X_processed = df.reindex(columns=FEATURES_FINAL)
    
    # Приведение категориальных признаков к INT (для CatBoost)
    for col in CATEGORICAL_FEATURES_USED:
        if col in X_processed.columns:
            X_processed[col] = X_processed[col].fillna(0).round(0).astype(int)
    
    return X_processed


def predict_fraud(transactions_json: List[Dict[str, Any]], 
                  threshold: float = None,
                  return_proba: bool = False) -> List[Dict[str, Any]]:
    """
    Главная функция для предсказания мошенничества на основе JSON транзакций.
    
    Параметры:
    ----------
    transactions_json : List[Dict[str, Any]]
        Список словарей с данными транзакций. Каждый словарь должен содержать поля:
        - transdatetime (str/datetime): Дата и время транзакции
        - amount (float): Сумма транзакции
        - monthly_os_changes (int): Количество изменений ОС за месяц
        - monthly_phone_model_changes (int): Количество изменений модели телефона
        - logins_last_7_days (int): Количество входов за последние 7 дней
        - logins_last_30_days (int): Количество входов за последние 30 дней
        - freq_change_7d_vs_mean (str/float): Изменение частоты за 7 дней
        - burstiness_login_interval (float): Показатель всплеска активности
        - zscore_avg_login_interval_7d (float): Z-score среднего интервала входа
        - avg_login_interval_30d (float): Средний интервал входа за 30 дней (в секундах)
        - std_login_interval_30d (float): Стандартное отклонение интервала (в секундах)
        - ewm_login_interval_7d (float): Экспоненциально взвешенное среднее (в секундах)
        - fano_factor_login_interval (float): Фактор Фано
    
    threshold : float, optional
        Порог для классификации (по умолчанию используется оптимальный из метаданных)
    
    return_proba : bool, default=False
        Если True, возвращает также вероятности мошенничества
    
    Возвращает:
    -----------
    List[Dict[str, Any]]
        Список словарей с результатами для каждой транзакции:
        - transaction_id (int): Индекс транзакции
        - is_fraud (int): 1 если мошенничество, 0 если нет
        - fraud_probability (float): Вероятность мошенничества (если return_proba=True)
    """
    
    # 1. Загружаем модель и препроцессор
    print("Загрузка модели и препроцессора...")
    model, imputer, features_used, optimal_threshold = load_latest_model_and_preprocessor()
    
    # Используем переданный порог или оптимальный из метаданных
    if threshold is None:
        threshold = optimal_threshold if optimal_threshold else 0.1
    
    print(f"Используется порог: {threshold:.4f}")
    
    # 2. Подготавливаем данные (feature engineering)
    print(f"Обработка {len(transactions_json)} транзакций...")
    X_processed = prepare_json_data(transactions_json)
    
    # Проверяем наличие всех необходимых признаков
    missing_features = set(FEATURES_FINAL) - set(X_processed.columns)
    if missing_features:
        print(f"ВНИМАНИЕ: Отсутствуют признаки: {missing_features}")
        # Заполняем отсутствующие признаки нулями
        for feat in missing_features:
            X_processed[feat] = 0
    
    # Убеждаемся, что порядок признаков совпадает
    X_processed = X_processed[FEATURES_FINAL]
    
    # 3. Применяем импьютацию (заполнение NaN) - как в CatBoost.py строки 286-298
    X_imputed = imputer.transform(X_processed)
    X_final = pd.DataFrame(X_imputed, columns=FEATURES_FINAL)
    
    # 4. Приведение категориальных признаков к INT (как в CatBoost.py строки 300-303)
    for col in CATEGORICAL_FEATURES_USED:
        if col in X_final.columns:
            X_final[col] = X_final[col].round(0).astype(int)
    
    # 5. Делаем предсказания
    print("Выполнение предсказаний...")
    print(f"Используемый порог для классификации: {threshold:.4f}")
    fraud_proba = model.predict_proba(X_final)[:, 1]
    fraud_pred = (fraud_proba >= threshold).astype(int)
    
    # Отладочный вывод вероятностей
    print(f"\nВероятности мошенничества:")
    for i, prob in enumerate(fraud_proba):
        pred_status = "МОШЕННИЧЕСТВО" if fraud_pred[i] == 1 else "ЛЕГИТИМНО"
        print(f"  Транзакция {i}: {prob:.4f} {'>=' if prob >= threshold else '<'} {threshold:.4f} -> {pred_status}")
    
    # 6. Формируем результат
    results = []
    for i, (pred, proba) in enumerate(zip(fraud_pred, fraud_proba)):
        result = {
            'transaction_id': i,
            'is_fraud': int(pred)
        }
        if return_proba:
            result['fraud_probability'] = float(proba)
        results.append(result)
    
    print(f"Предсказания завершены. Обнаружено мошеннических транзакций: {sum(fraud_pred)}/{len(transactions_json)}")
    
    return results


# =========================================================================================================================== #
#                   ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =========================================================================================================================== #

if __name__ == "__main__":
    # Пример JSON транзакций: 2 легитимные + 2 мошеннические
    example_transactions = [
        # ЛЕГИТИМНАЯ ТРАНЗАКЦИЯ 1 - нормальная активность
        {
            "transdatetime": "2024-01-15 14:30:00",
            "amount": 5000.0,
            "monthly_os_changes": 0,
            "monthly_phone_model_changes": 0,
            "logins_last_7_days": 5,
            "logins_last_30_days": 20,
            "freq_change_7d_vs_mean": "1.2",
            "burstiness_login_interval": 0.5,
            "zscore_avg_login_interval_7d": 0.8,
            "avg_login_interval_30d": 86400.0,  # 1 день в секундах
            "std_login_interval_30d": 43200.0,
            "ewm_login_interval_7d": 86400.0,
            "fano_factor_login_interval": 1.2
        },
        # ЛЕГИТИМНАЯ ТРАНЗАКЦИЯ 2 - активный пользователь
        {
            "transdatetime": "2024-01-16 10:15:00",
            "amount": 2000.0,
            "monthly_os_changes": 0,
            "monthly_phone_model_changes": 0,
            "logins_last_7_days": 10,
            "logins_last_30_days": 45,
            "freq_change_7d_vs_mean": "1.5",
            "burstiness_login_interval": 0.3,
            "zscore_avg_login_interval_7d": -0.5,
            "avg_login_interval_30d": 3600.0,  # 1 час
            "std_login_interval_30d": 1800.0,
            "ewm_login_interval_7d": 3600.0,
            "fano_factor_login_interval": 0.8
        },
        # МОШЕННИЧЕСКАЯ ТРАНЗАКЦИЯ 1 - подозрительная активность (усиленные признаки)
        {
            "transdatetime": "2024-01-15 23:45:00",  # Поздний час
            "amount": 100000.0,  # Очень большая сумма
            "monthly_os_changes": 5,  # Много изменений ОС (максимально подозрительно)
            "monthly_phone_model_changes": 4,  # Много изменений телефона
            "logins_last_7_days": 0,  # Нет входов за неделю
            "logins_last_30_days": 1,  # Минимальная активность
            "freq_change_7d_vs_mean": "0.001",  # Экстремальное падение активности
            "burstiness_login_interval": 10.0,  # Очень высокий всплеск
            "zscore_avg_login_interval_7d": 8.0,  # Экстремально высокий z-score
            "avg_login_interval_30d": 2592000.0,  # Очень долгий интервал (30 дней)
            "std_login_interval_30d": 1296000.0,  # Высокое стандартное отклонение
            "ewm_login_interval_7d": 2592000.0,
            "fano_factor_login_interval": 10.0  # Очень высокая волатильность
        },
        # МОШЕННИЧЕСКАЯ ТРАНЗАКЦИЯ 2 - аномальное поведение (максимально подозрительно)
        {
            "transdatetime": "2024-01-17 03:15:00",  # Ночное время
            "amount": 200000.0,  # Экстремально большая сумма
            "monthly_os_changes": 6,  # Максимальное количество изменений ОС
            "monthly_phone_model_changes": 5,  # Максимальное количество изменений телефона
            "logins_last_7_days": 0,  # Полное отсутствие активности
            "logins_last_30_days": 0,  # Полное отсутствие активности
            "freq_change_7d_vs_mean": "0.0001",  # Минимальная активность
            "burstiness_login_interval": 15.0,  # Экстремальный всплеск
            "zscore_avg_login_interval_7d": 10.0,  # Максимальный z-score
            "avg_login_interval_30d": 2592000.0,  # 30 дней между входами
            "std_login_interval_30d": 2592000.0,  # Максимальное стандартное отклонение
            "ewm_login_interval_7d": 2592000.0,
            "fano_factor_login_interval": 15.0  # Максимальная волатильность
        }
    ]
    
    print("=" * 60)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ ФУНКЦИИ ПРЕДСКАЗАНИЯ")
    print("=" * 60)
    
    # Выполняем предсказания с вероятностями (используем оптимальный порог из метаданных)
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ С ОПТИМАЛЬНЫМ ПОРОГОМ (из метаданных):")
    print("=" * 60)
    results_optimal = predict_fraud(example_transactions, return_proba=True)
    for i, result in enumerate(results_optimal):
        status = "МОШЕННИЧЕСТВО" if result['is_fraud'] == 1 else "ЛЕГИТИМНО"
        proba = result.get('fraud_probability', 0.0)
        transaction_type = "МОШЕННИЧЕСКАЯ" if i >= 2 else "ЛЕГИТИМНАЯ"
        print(f"Транзакция {result['transaction_id']} ({transaction_type}): {status} (вероятность: {proba:.4f})")
    
    # Тест с пониженным порогом для демонстрации
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ С ПОНИЖЕННЫМ ПОРОГОМ (0.025):")
    print("=" * 60)
    results_low_threshold = predict_fraud(example_transactions, threshold=0.025, return_proba=True)
    for i, result in enumerate(results_low_threshold):
        status = "МОШЕННИЧЕСТВО" if result['is_fraud'] == 1 else "ЛЕГИТИМНО"
        proba = result.get('fraud_probability', 0.0)
        transaction_type = "МОШЕННИЧЕСКАЯ" if i >= 2 else "ЛЕГИТИМНАЯ"
        print(f"Транзакция {result['transaction_id']} ({transaction_type}): {status} (вероятность: {proba:.4f})")
    
    print("\n" + "=" * 60)
    print("JSON формат результатов (с порогом 0.025):")
    print("=" * 60)
    print(json.dumps(results_low_threshold, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
    print("JSON формат результатов (с оптимальным порогом):")
    print("=" * 60)
    print(json.dumps(results_optimal, indent=2, ensure_ascii=False))

