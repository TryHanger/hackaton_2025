import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KernelDensity
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def compare_with_join():
    """
    Более эффективное сравнение с использованием JOIN между БД.
    """
    try:
        # Создаем временную БД для результатов
        result_conn = sqlite3.connect(':memory:')
        result_cursor = result_conn.cursor()

        # Присоединяем обе БД
        result_cursor.execute("ATTACH DATABASE '../info_perevody.db' AS info_db")
        result_cursor.execute("ATTACH DATABASE '../perevody.db' AS perevody_db")

        # Создаем таблицу с результатами
        result_cursor.execute("""
            CREATE TABLE logins_last_7_days AS
            SELECT 
                i.cst_dim_id,
                i.transdate,
                i.logins_last_7_days,
                p.target
            FROM info_db.unique_transactions i
            LEFT JOIN perevody_db.unique_transactions p
                ON i.cst_dim_id = p.cst_dim_id 
                AND i.transdate = p.transdate
            WHERE i.logins_last_7_days IS NOT NULL
        """)

        # Статистика
        result_cursor.execute("SELECT COUNT(*) FROM logins_last_7_days")
        total_count = result_cursor.fetchone()[0]

        result_cursor.execute("SELECT COUNT(*) FROM logins_last_7_days WHERE target IS NOT NULL")
        matched_count = result_cursor.fetchone()[0]

        result_cursor.execute("SELECT COUNT(*) FROM logins_last_7_days WHERE target = 1")
        target_1_count = result_cursor.fetchone()[0]

        result_cursor.execute("SELECT COUNT(*) FROM logins_last_7_days WHERE target = 0")
        target_0_count = result_cursor.fetchone()[0]

        print("📊 СТАТИСТИКА СРАВНЕНИЯ:")
        print(f"Всего записей: {total_count}")
        print(f"Найдено соответствий: {matched_count}")
        print(f"Не найдено соответствий: {total_count - matched_count}")
        print(f"Target = 1: {target_1_count}")
        print(f"Target = 0: {target_0_count}")

        # Выборка результатов
        result_cursor.execute("SELECT * FROM logins_last_7_days LIMIT 10")
        sample_results = result_cursor.fetchall()

        print("\n🔍 ПЕРВЫЕ 10 РЕЗУЛЬТАТОВ:")
        print("ID | Дата | logins_last_7_days | Target")
        print("-" * 50)
        for row in sample_results:
            print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")

        # Сохраняем результаты в файл если нужно
        result_cursor.execute("""
            ATTACH DATABASE '../comparison_results.db' AS results_db
        """)
        result_cursor.execute("""
            CREATE TABLE results_db.logins_last_7_days AS
            SELECT * FROM logins_last_7_days
        """)

        print("\n💾 Результаты сохранены в comparison_results.db")

        # Отсоединяем БД
        result_cursor.execute("DETACH DATABASE info_db")
        result_cursor.execute("DETACH DATABASE perevody_db")
        result_cursor.execute("DETACH DATABASE results_db")

        return total_count

    except sqlite3.Error as e:
        print(f"❌ Ошибка базы данных: {e}")
        return 0

def load_data():
    script_dir = Path(__file__).parent
    db_path = script_dir.parent / 'comparison_results.db'
    conn = sqlite3.connect(db_path)
    # conn = sqlite3.connect('../comparison_results.db')
    df = pd.read_sql_query(
        "SELECT logins_last_7_days, target FROM logins_last_7_days WHERE target IS NOT NULL",
        conn
    )
    conn.close()
    return df

def automatic_binning_analysis(df):
    """Автоматический биннинг с разными стратегиями."""

    print("\n" + "=" * 60)
    print("1. АВТОМАТИЧЕСКИЙ БИННИНГ")
    print("=" * 60)

    X = df['logins_last_7_days'].values.reshape(-1, 1)

    # Разные стратегии биннинга
    strategies = {
        'uniform': 'Равномерный',
        'quantile': 'Квантильный',
        'kmeans': 'K-средних'
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for idx, (strategy, name) in enumerate(strategies.items()):
        # Создаем бины
        discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy=strategy)
        bins_array = discretizer.fit_transform(X).flatten()

        # Анализ fraud rate по бинам
        df_binned = df.copy()
        df_binned['bin'] = bins_array
        bin_stats = df_binned.groupby('bin').agg({
            'logins_last_7_days': ['min', 'max', 'count'],
            'target': ['sum', 'mean']
        }).round(3)

        bin_stats.columns = ['min_login', 'max_login', 'total_count', 'fraud_count', 'fraud_rate']
        bin_stats['fraud_rate_pct'] = (bin_stats['fraud_rate'] * 100).round(2)

        # Визуализация
        ax = axes[idx]
        bins_centers = (bin_stats['min_login'] + bin_stats['max_login']) / 2
        ax.plot(bins_centers, bin_stats['fraud_rate_pct'], 'o-', linewidth=2, markersize=8)
        ax.set_title(f'{name} биннинг\n({strategy})', fontsize=14)
        ax.set_xlabel('Логины за 7 дней', fontsize=12)
        ax.set_ylabel('Fraud Rate (%)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Добавляем значения
        for i, (center, rate) in enumerate(zip(bins_centers, bin_stats['fraud_rate_pct'])):
            ax.annotate(f'{rate}%', (center, rate), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9)

        print(f"\n{name} биннинг:")
        print(bin_stats[['min_login', 'max_login', 'total_count', 'fraud_rate_pct']])

    # Ручной биннинг на основе анализа
    axes[3].axis('off')
    plt.tight_layout()
    plt.show()

    return df

def trend_analysis(df):
    """Статистический анализ тренда риска."""

    print("\n" + "=" * 60)
    print("2. СТАТИСТИЧЕСКИЙ АНАЛИЗ ТРЕНДА")
    print("=" * 60)

    # Создаем оптимальные бины на основе распределения
    bins = [0, 1, 2, 3, 5, 8, 12, 20, 35, 70, 181]
    labels = ['0', '1', '2', '3-4', '5-7', '8-11', '12-19', '20-34', '35-69', '70+']

    df['bin'] = pd.cut(df['logins_last_7_days'], bins=bins, labels=labels, right=False)
    trend_data = df.groupby('bin').agg({
        'target': ['count', 'sum', 'mean'],
        'logins_last_7_days': 'median'
    }).round(4)

    trend_data.columns = ['total', 'fraud_count', 'fraud_rate', 'median_logins']
    trend_data['fraud_rate_pct'] = (trend_data['fraud_rate'] * 100).round(2)

    # Статистические тесты
    X = np.arange(len(trend_data))
    Y = trend_data['fraud_rate'].values

    # Линейная регрессия для тренда
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

    # Тест Манна-Кендалла (монотонность)
    from scipy.stats import kendalltau
    tau, p_tau = kendalltau(X, Y)

    print("📈 СТАТИСТИКА ТРЕНДА:")
    print(f"Линейный тренд: slope = {slope:.6f}, R² = {r_value ** 2:.4f}")
    print(f"P-value линейности: {p_value:.6f}")
    print(f"Тест Кендалла: tau = {tau:.4f}, p-value = {p_tau:.6f}")
    print(f"Монотонность: {'ДА' if p_tau < 0.05 and tau > 0 else 'НЕТ'}")

    # Визуализация тренда
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # График тренда
    ax1.plot(X, Y * 100, 'o-', linewidth=3, markersize=8, label='Наблюдаемый fraud rate')

    # Линейный тренд
    trend_line = intercept + slope * X
    ax1.plot(X, trend_line * 100, '--', color='red', linewidth=2,
             label=f'Линейный тренд (R²={r_value ** 2:.3f})')

    ax1.set_title('Анализ тренда Fraud Rate', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Уровень активности (бин)', fontsize=12)
    ax1.set_ylabel('Fraud Rate (%)', fontsize=12)
    ax1.set_xticks(X)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Добавляем значения
    for i, (x, y) in enumerate(zip(X, Y * 100)):
        ax1.annotate(f'{y:.1f}%', (x, y), xytext=(0, 10),
                     textcoords='offset points', ha='center', fontweight='bold')

    # Детальная таблица
    ax2.axis('off')
    table_data = []
    for bin_name in labels:
        if bin_name in trend_data.index:
            row = trend_data.loc[bin_name]
            table_data.append([
                bin_name, row['total'], row['fraud_count'],
                f"{row['fraud_rate_pct']}%", f"{row['median_logins']:.1f}"
            ])

    table = ax2.table(cellText=table_data,
                      colLabels=['Бин', 'Всего', 'Fraud', 'Fraud%', 'Медиана'],
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.set_title('Детальная статистика по бинам', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return trend_data

def tail_analysis(df):
    """Анализ редких значений (хвостов распределения)."""

    print("\n" + "=" * 60)
    print("3. АНАЛИЗ ХВОСТОВ РАСПРЕДЕЛЕНИЯ")
    print("=" * 60)

    # Определяем выбросы (более 3 стандартных отклонений)
    mean_logins = df['logins_last_7_days'].mean()
    std_logins = df['logins_last_7_days'].std()
    outlier_threshold = mean_logins + 3 * std_logins

    outliers = df[df['logins_last_7_days'] > outlier_threshold]
    normal_data = df[df['logins_last_7_days'] <= outlier_threshold]

    print(f"Порог выбросов: > {outlier_threshold:.1f} логинов")
    print(f"Выбросов: {len(outliers)} записей ({len(outliers) / len(df) * 100:.2f}%)")
    print(f"Выбросы - Fraud Rate: {outliers['target'].mean() * 100:.2f}%")
    print(f"Нормальные значения - Fraud Rate: {normal_data['target'].mean() * 100:.2f}%")

    # Анализ экстремальных значений
    extreme_thresholds = [50, 100, 150]

    print("\n🔍 АНАЛИЗ ЭКСТРЕМАЛЬНЫХ ЗНАЧЕНИЙ:")
    for threshold in extreme_thresholds:
        extreme_data = df[df['logins_last_7_days'] >= threshold]
        if len(extreme_data) > 0:
            fraud_rate = extreme_data['target'].mean() * 100
            print(f"≥{threshold} логинов: {len(extreme_data)} записей, Fraud Rate: {fraud_rate:.2f}%")

    # Визуализация хвостов
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Распределение с выделением выбросов
    ax1.hist(normal_data['logins_last_7_days'], bins=50, alpha=0.7,
             color='blue', label=f'Нормальные (≤{outlier_threshold:.0f})')
    ax1.hist(outliers['logins_last_7_days'], bins=20, alpha=0.7,
             color='red', label=f'Выбросы (> {outlier_threshold:.0f})')
    ax1.axvline(outlier_threshold, color='black', linestyle='--', linewidth=2, label='Порог выбросов')
    ax1.set_title('Распределение с выделением выбросов', fontsize=14)
    ax1.set_xlabel('Логины за 7 дней', fontsize=12)
    ax1.set_ylabel('Частота', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fraud rate по перцентилям - ИСПРАВЛЕННАЯ ЧАСТЬ
    percentiles = np.arange(0, 101, 5)
    percentile_values = np.percentile(df['logins_last_7_days'], percentiles)
    percentile_fraud = []

    for p in percentiles:
        threshold = np.percentile(df['logins_last_7_days'], p)
        high_activity = df[df['logins_last_7_days'] >= threshold]
        if len(high_activity) > 0:
            fraud_rate = high_activity['target'].mean() * 100
            percentile_fraud.append(fraud_rate)
        else:
            percentile_fraud.append(0)

    ax2.plot(100 - percentiles, percentile_fraud, 'o-', linewidth=2, markersize=4)
    ax2.set_title('Fraud Rate по перцентилям активности\n(топ X% самых активных)', fontsize=14)
    ax2.set_xlabel('Топ X% самых активных пользователей', fontsize=12)
    ax2.set_ylabel('Fraud Rate (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # Добавляем ключевые точки - ИСПРАВЛЕННАЯ ЧАСТЬ
    key_percentiles = [95, 99, 99.9]
    for p in key_percentiles:
        # Находим ближайший перцентиль из нашего списка
        closest_p = min(percentiles, key=lambda x: abs(x - p))
        idx = list(percentiles).index(closest_p)

        # Получаем реальное значение перцентиля
        actual_threshold = np.percentile(df['logins_last_7_days'], p)

        ax2.annotate(f'{p}%: {percentile_fraud[idx]:.1f}%\n(≥{actual_threshold:.0f} логинов)',
                     (100 - closest_p, percentile_fraud[idx]),
                     xytext=(10, 5), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                     fontsize=9)

    plt.tight_layout()
    plt.show()

    # Дополнительный анализ самых активных пользователей
    print("\n📊 АНАЛИЗ САМЫХ АКТИВНЫХ ПОЛЬЗОВАТЕЛЕЙ:")
    for p in [99, 99.5, 99.9]:
        threshold = np.percentile(df['logins_last_7_days'], p)
        top_users = df[df['logins_last_7_days'] >= threshold]
        if len(top_users) > 0:
            fraud_rate = top_users['target'].mean() * 100
            print(
                f"Топ {100 - p:.1f}% (≥{threshold:.0f} логинов): {len(top_users)} записей, Fraud Rate: {fraud_rate:.2f}%")

    return outliers

def nonlinear_analysis(df):
    """Поиск нелинейных зависимостей."""

    print("\n" + "=" * 60)
    print("4. ПОИСК НЕЛИНЕЙНЫХ ЗАВИСИМОСТЕЙ")
    print("=" * 60)

    # Создаем копию данных и сортируем
    sorted_df = df.copy().sort_values('logins_last_7_days').reset_index(drop=True)

    # Скользящее среднее для сглаживания (убираем NaN в конце)
    sorted_df['rolling_fraud'] = sorted_df['target'].rolling(window=100, center=True).mean()

    # Удаляем строки с NaN в rolling_fraud
    valid_data = sorted_df.dropna(subset=['rolling_fraud']).copy()

    # Полиномиальная аппроксимация
    X_poly = valid_data['logins_last_7_days'].values
    Y_poly = valid_data['rolling_fraud'].values

    # Аппроксимации разной степени
    degrees = [1, 2, 3, 4]
    colors = ['red', 'blue', 'green', 'purple']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # График 1: Скользящее среднее
    ax1.scatter(sorted_df['logins_last_7_days'], sorted_df['target'],
                alpha=0.1, color='gray', s=1, label='Исходные данные')
    ax1.plot(valid_data['logins_last_7_days'], valid_data['rolling_fraud'] * 100,
             linewidth=3, color='black', label='Скользящее среднее (n=100)')
    ax1.set_title('Fraud Rate vs Логины\n(скользящее среднее)', fontsize=14)
    ax1.set_xlabel('Логины за 7 дней', fontsize=12)
    ax1.set_ylabel('Fraud Rate (%)', fontsize=12)
    ax1.set_xlim(0, 100)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Полиномиальная аппроксимация
    ax2.scatter(X_poly, Y_poly * 100, alpha=0.3, color='gray', s=10, label='Сглаженные данные')

    r_squared = {}
    for degree, color in zip(degrees, colors):
        try:
            coeffs = np.polyfit(X_poly, Y_poly, degree)
            polynomial = np.poly1d(coeffs)
            y_fit = polynomial(X_poly)

            # R²
            ss_res = np.sum((Y_poly - y_fit) ** 2)
            ss_tot = np.sum((Y_poly - np.mean(Y_poly)) ** 2)
            r_sq = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            r_squared[degree] = r_sq

            ax2.plot(X_poly, y_fit * 100, color=color, linewidth=2,
                     label=f'Полином {degree} степени (R²={r_sq:.3f})')
        except Exception as e:
            print(f"Ошибка при аппроксимации степени {degree}: {e}")
            r_squared[degree] = 0

    ax2.set_title('Полиномиальная аппроксимация', fontsize=14)
    ax2.set_xlabel('Логины за 7 дней', fontsize=12)
    ax2.set_ylabel('Fraud Rate (%)', fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Анализ оптимальной степени
    print("📊 КАЧЕСТВО АППРОКСИМАЦИИ:")
    for degree, r_sq in r_squared.items():
        print(f"Полином {degree} степени: R² = {r_sq:.4f}")

    # Локальные максимумы/минимумы для лучшей модели
    if r_squared:
        optimal_degree = max(r_squared, key=r_squared.get)
        try:
            coeffs = np.polyfit(X_poly, Y_poly, optimal_degree)
            polynomial = np.poly1d(coeffs)

            # Производная для поиска экстремумов
            derivative = polynomial.deriv()
            critical_points = derivative.roots

            # Фильтруем действительные точки в диапазоне данных
            real_critical_points = critical_points[np.isreal(critical_points)].real
            real_critical_points = real_critical_points[
                (real_critical_points >= X_poly.min()) &
                (real_critical_points <= X_poly.max())
                ]

            print(f"\n🔍 КРИТИЧЕСКИЕ ТОЧКИ (полином {optimal_degree} степени):")
            if len(real_critical_points) > 0:
                for point in real_critical_points:
                    fraud_at_point = polynomial(point) * 100
                    point_type = "МАКСИМУМ" if polynomial.deriv(2)(point) < 0 else "минимум"
                    print(f"  {point:.1f} логинов: Fraud Rate = {fraud_at_point:.2f}% ({point_type})")
            else:
                print("  Критические точки не найдены")

        except Exception as e:
            print(f"Ошибка при анализе критических точек: {e}")

    return r_squared

def kde_analysis(df):
    """Kernel Density Estimation для распределений."""

    print("\n" + "=" * 60)
    print("5. KERNEL DENSITY ESTIMATION")
    print("=" * 60)

    logins_0 = df[df['target'] == 0]['logins_last_7_days']
    logins_1 = df[df['target'] == 1]['logins_last_7_days']

    # Ограничиваем диапазон для лучшей визуализации
    max_val = 50
    logins_0_trimmed = logins_0[logins_0 <= max_val]
    logins_1_trimmed = logins_1[logins_1 <= max_val]

    # KDE с разными bandwidth
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    bandwidths = [0.5, 1.0, 2.0, 'auto']
    titles = ['Small bandwidth (0.5)', 'Medium bandwidth (1.0)',
              'Large bandwidth (2.0)', 'Auto bandwidth']

    for idx, (bw, title) in enumerate(zip(bandwidths, titles)):
        ax = axes[idx // 2, idx % 2]

        # KDE для каждого класса
        if bw == 'auto':
            kde_0 = stats.gaussian_kde(logins_0_trimmed)
            kde_1 = stats.gaussian_kde(logins_1_trimmed)
        else:
            kde_0 = stats.gaussian_kde(logins_0_trimmed, bw_method=bw)
            kde_1 = stats.gaussian_kde(logins_1_trimmed, bw_method=bw)

        x_range = np.linspace(0, max_val, 200)

        ax.plot(x_range, kde_0(x_range), label='Target=0 (Легитимные)',
                color='blue', linewidth=2)
        ax.plot(x_range, kde_1(x_range), label='Target=1 (Мошеннические)',
                color='red', linewidth=2)

        ax.set_title(f'KDE: {title}', fontsize=14)
        ax.set_xlabel('Логины за 7 дней', fontsize=12)
        ax.set_ylabel('Плотность вероятности', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_val)

    plt.tight_layout()
    plt.show()

    # Ratio of densities (сигнал для классификации)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Auto KDE для ratio анализа
    kde_0_auto = stats.gaussian_kde(logins_0_trimmed)
    kde_1_auto = stats.gaussian_kde(logins_1_trimmed)

    x_range = np.linspace(0, max_val, 200)
    density_0 = kde_0_auto(x_range)
    density_1 = kde_1_auto(x_range)

    # Отношение плотностей
    likelihood_ratio = np.where(density_0 > 0, density_1 / density_0, 0)

    ax1.plot(x_range, density_0, label='P(x|Target=0)', color='blue', linewidth=2)
    ax1.plot(x_range, density_1, label='P(x|Target=1)', color='red', linewidth=2)
    ax1.set_title('Вероятностные распределения', fontsize=14)
    ax1.set_xlabel('Логины за 7 дней', fontsize=12)
    ax1.set_ylabel('Плотность вероятности', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_range, likelihood_ratio, color='purple', linewidth=3)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Порог (ratio=1)')
    ax2.set_title('Likelihood Ratio: P(x|Target=1) / P(x|Target=0)', fontsize=14)
    ax2.set_xlabel('Логины за 7 дней', fontsize=12)
    ax2.set_ylabel('Likelihood Ratio', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Области где мошенники более вероятны
    fraud_preferred = x_range[likelihood_ratio > 1]
    if len(fraud_preferred) > 0:
        print(f"🔍 ОБЛАСТИ ПРЕИМУЩЕСТВА МОШЕННИКОВ:")
        print(f"  Логины от {fraud_preferred[0]:.1f} до {fraud_preferred[-1]:.1f}")

        # Пик отношения правдоподобия
        peak_idx = np.argmax(likelihood_ratio)
        peak_x = x_range[peak_idx]
        peak_ratio = likelihood_ratio[peak_idx]
        print(f"  Пик отношения: {peak_ratio:.2f} при {peak_x:.1f} логинах")

    plt.tight_layout()
    plt.show()

    return likelihood_ratio, x_range

def final_report(df, trend_data, r_squared, likelihood_ratio, x_range):
    """Финальный отчет с выводами."""

    print("\n" + "=" * 80)
    print("🎯 ИТОГОВЫЙ ОТЧЕТ АНАЛИЗА")
    print("=" * 80)

    # Ключевые метрики
    total_fraud_rate = df['target'].mean() * 100
    correlation = df['logins_last_7_days'].corr(df['target'])

    print("📈 КЛЮЧЕВЫЕ МЕТРИКИ:")
    print(f"  • Общий Fraud Rate: {total_fraud_rate:.2f}%")
    print(f"  • Корреляция логины-риск: {correlation:.4f}")
    print(f"  • Оптимальная степень полинома: {max(r_squared, key=r_squared.get)}")
    print(f"  • Максимальный likelihood ratio: {np.max(likelihood_ratio):.2f}")

    # Практические рекомендации
    print("\n💡 ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")

    # Оптимальные пороги
    high_risk_threshold = 20
    medium_risk_threshold = 5

    high_risk_data = df[df['logins_last_7_days'] >= high_risk_threshold]
    medium_risk_data = df[(df['logins_last_7_days'] >= medium_risk_threshold) &
                          (df['logins_last_7_days'] < high_risk_threshold)]

    print(f"  1. ВЫСОКИЙ РИСК (≥{high_risk_threshold} логинов):")
    print(f"     • {len(high_risk_data)} записей ({len(high_risk_data) / len(df) * 100:.1f}%)")
    print(f"     • Fraud Rate: {high_risk_data['target'].mean() * 100:.2f}%")

    print(f"  2. СРЕДНИЙ РИСК ({medium_risk_threshold}-{high_risk_threshold - 1} логинов):")
    print(f"     • {len(medium_risk_data)} записей ({len(medium_risk_data) / len(df) * 100:.1f}%)")
    print(f"     • Fraud Rate: {medium_risk_data['target'].mean() * 100:.2f}%")

    # Области максимального сигнала
    peak_signal_idx = np.argmax(likelihood_ratio)
    peak_signal_x = x_range[peak_signal_idx]

    print(f"  3. МАКСИМАЛЬНЫЙ СИГНАЛ:")
    print(f"     • {peak_signal_x:.1f} логинов (likelihood ratio = {np.max(likelihood_ratio):.2f})")

    # Статистическая значимость
    from scipy.stats import mannwhitneyu
    stat, p_value = mannwhitneyu(logins_0, logins_1, alternative='two-sided')

    print(f"\n📊 СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:")
    print(f"  • Тест Манна-Уитни: p-value = {p_value:.6f}")
    print(f"  • Различие распределений: {'ЗНАЧИМО' if p_value < 0.05 else 'НЕЗНАЧИМО'}")

    # Финальная визуализация
    plt.figure(figsize=(12, 8))

    # Композитный график
    plt.subplot(2, 1, 1)

    # KDE распределения
    kde_0 = stats.gaussian_kde(logins_0[logins_0 <= 50])
    kde_1 = stats.gaussian_kde(logins_1[logins_1 <= 50])
    x_kde = np.linspace(0, 50, 200)

    plt.plot(x_kde, kde_0(x_kde), label='Легитимные (Target=0)', color='blue', linewidth=2)
    plt.plot(x_kde, kde_1(x_kde), label='Мошеннические (Target=1)', color='red', linewidth=2)
    plt.title('Итоговый анализ: Распределения и Risk Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Логины за 7 дней', fontsize=12)
    plt.ylabel('Плотность вероятности', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)

    # Fraud rate тренд
    plt.plot(trend_data.index, trend_data['fraud_rate_pct'], 'o-',
             linewidth=3, markersize=8, color='green', label='Наблюдаемый Fraud Rate')
    plt.axhline(y=total_fraud_rate, color='black', linestyle='--',
                label=f'Общий Fraud Rate ({total_fraud_rate:.2f}%)')
    plt.title('Fraud Rate по уровням активности', fontsize=14)
    plt.xlabel('Уровень активности', fontsize=12)
    plt.ylabel('Fraud Rate (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")

def calculate_detailed_stats(df, value_column='logins_last_7_days', target_column='target'):
    """
    Рассчитывает детальную статистику по группам target, включая RRN и SE.
    Адаптировано под данные login_frequency_7d.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с данными
    value_column : str
        Название колонки с числовыми значениями для анализа
    target_column : str
        Название колонки с целевой переменной

    Returns:
    --------
    pandas.DataFrame
        DataFrame со статистиками по группам
    """

    # Проверяем, что колонки существуют
    if value_column not in df.columns:
        available_cols = df.columns.tolist()
        raise ValueError(f"Колонка {value_column} не найдена в DataFrame. Доступные колонки: {available_cols}")
    if target_column not in df.columns:
        available_cols = df.columns.tolist()
        raise ValueError(f"Колонка {target_column} не найдена в DataFrame. Доступные колонки: {available_cols}")

    # Убираем пропущенные значения
    df_clean = df[[value_column, target_column]].dropna()

    # Преобразуем target в целые числа, если нужно
    df_clean[target_column] = df_clean[target_column].astype(int)

    print(f"📊 Анализ данных:")
    print(f"   Колонка для анализа: {value_column}")
    print(f"   Всего записей: {len(df_clean)}")
    print(f"   Диапазон значений: {df_clean[value_column].min():.4f} - {df_clean[value_column].max():.4f}")

    # Группируем по target
    grouped = df_clean.groupby(target_column)

    stats_list = []

    for target_val, group in grouped:
        values = group[value_column]
        n = len(values)

        print(f"   Target {target_val}: {n} записей")

        # Основные статистики
        mean_val = values.mean()
        median_val = values.median()
        std_val = values.std()
        min_val = values.min()
        max_val = values.max()
        q25 = values.quantile(0.25)
        q75 = values.quantile(0.75)

        # Процент ненулевых значений
        pct_nonzero = (values != 0).sum() / n * 100 if n > 0 else 0

        # RRN (Robust Relative Norm) - устойчивая относительная норма
        if median_val != 0:
            mad = np.median(np.abs(values - median_val))  # Median Absolute Deviation
            rrn = mad / median_val
        else:
            # Если медиана равна 0, используем среднее абсолютное отклонение от 0
            rrn = np.mean(np.abs(values)) if len(values) > 0 else 0

        # SE (Standard Error) - стандартная ошибка среднего
        se = std_val / np.sqrt(n) if n > 0 else 0

        # Коэффициент вариации
        cv = (std_val / mean_val * 100) if mean_val != 0 else 0

        # Асимметрия и эксцесс
        skewness = stats.skew(values) if len(values) > 2 else 0
        kurtosis = stats.kurtosis(values) if len(values) > 2 else 0

        # Дополнительные метрики для частоты логинов
        # Процент очень низкой активности (<0.1)
        pct_very_low = (values < 0.1).sum() / n * 100
        # Процент высокой активности (>5)
        pct_high = (values > 5).sum() / n * 100
        # Прочень очень высокой активности (>10)
        pct_very_high = (values > 10).sum() / n * 100

        stats_list.append({
            'target': target_val,
            'n': n,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'q25': q25,
            'q75': q75,
            'pct_nonzero': pct_nonzero,
            'rrn': rrn,
            'se': se,
            'cv': cv,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'pct_very_low': pct_very_low,
            'pct_high': pct_high,
            'pct_very_high': pct_very_high
        })

    # Создаем DataFrame со статистиками
    stats_df = pd.DataFrame(stats_list)

    # Добавляем общую статистику (все данные)
    all_values = df_clean[value_column]
    n_total = len(all_values)
    mean_total = all_values.mean()
    median_total = all_values.median()
    std_total = all_values.std()

    if median_total != 0:
        mad_total = np.median(np.abs(all_values - median_total))
        rrn_total = mad_total / median_total
    else:
        rrn_total = np.mean(np.abs(all_values)) if len(all_values) > 0 else 0

    se_total = std_total / np.sqrt(n_total) if n_total > 0 else 0

    total_stats = {
        'target': 'all',
        'n': n_total,
        'mean': mean_total,
        'median': median_total,
        'std': std_total,
        'min': all_values.min(),
        'max': all_values.max(),
        'q25': all_values.quantile(0.25),
        'q75': all_values.quantile(0.75),
        'pct_nonzero': (all_values != 0).sum() / n_total * 100,
        'rrn': rrn_total,
        'se': se_total,
        'cv': (std_total / mean_total * 100) if mean_total != 0 else 0,
        'skewness': stats.skew(all_values) if len(all_values) > 2 else 0,
        'kurtosis': stats.kurtosis(all_values) if len(all_values) > 2 else 0,
        'pct_very_low': (all_values < 0.1).sum() / n_total * 100,
        'pct_high': (all_values > 5).sum() / n_total * 100,
        'pct_very_high': (all_values > 10).sum() / n_total * 100
    }

    stats_df = pd.concat([stats_df, pd.DataFrame([total_stats])], ignore_index=True)

    return stats_df

def print_detailed_stats(stats_df):
    """Красиво выводит статистики для login_frequency_7d."""

    print("\n" + "=" * 100)
    print("📊 ДЕТАЛИЗИРОВАННАЯ СТАТИСТИКА ДЛЯ logins_last_7_days")
    print("=" * 100)

    # Форматируем вывод
    formatted_df = stats_df.copy()

    # Форматируем числовые колонки
    float_format_1 = ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75']
    float_format_2 = ['pct_nonzero', 'cv', 'pct_very_low', 'pct_high', 'pct_very_high']
    float_format_3 = ['rrn', 'se', 'skewness', 'kurtosis']

    for col in float_format_1:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")

    for col in float_format_2:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "NaN")

    for col in float_format_3:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else "NaN")

    # Форматируем n как целое число
    formatted_df['n'] = formatted_df['n'].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "NaN")

    # Переименовываем колонки для красивого вывода
    display_columns = {
        'target': 'Target',
        'n': 'N',
        'mean': 'Mean',
        'median': 'Median',
        'std': 'Std',
        'min': 'Min',
        'max': 'Max',
        'q25': 'Q25',
        'q75': 'Q75',
        'pct_nonzero': 'NonZero%',
        'rrn': 'RRN',
        'se': 'SE',
        'cv': 'CV%',
        'skewness': 'Skewness',
        'kurtosis': 'Kurtosis',
        'pct_very_low': '<0.1%',
        'pct_high': '>5%',
        'pct_very_high': '>10%'
    }

    # Применяем переименование только к существующим колонкам
    existing_columns = {k: v for k, v in display_columns.items() if k in formatted_df.columns}
    formatted_df = formatted_df.rename(columns=existing_columns)

    # Выводим основные статистики
    print("\n🎯 ОСНОВНЫЕ СТАТИСТИКИ:")
    basic_columns = ['Target', 'N', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Q25', 'Q75', 'NonZero%']
    basic_columns = [col for col in basic_columns if col in formatted_df.columns]
    print(formatted_df[basic_columns].to_string(index=False))

    print("\n📈 МЕТРИКИ ИЗМЕНЧИВОСТИ:")
    variability_columns = ['Target', 'RRN', 'SE', 'CV%', 'Skewness', 'Kurtosis']
    variability_columns = [col for col in variability_columns if col in formatted_df.columns]
    print(formatted_df[variability_columns].to_string(index=False))

    print("\n🔍 РАСПРЕДЕЛЕНИЕ ПО УРОВНЯМ АКТИВНОСТИ:")
    distribution_columns = ['Target', '<0.1%', '>5%', '>10%']
    distribution_columns = [col for col in distribution_columns if col in formatted_df.columns]
    print(formatted_df[distribution_columns].to_string(index=False))

    return formatted_df


def analyze_manual_bins(df, feature, target, bins, bin_labels=None):
    """
    Выполняет статистический анализ ручных бинов для fraud detection.

    Parameters:
    -----------
    df : pandas.DataFrame
        Исходный датафрейм
    feature : str
        Название числовой колонки (например 'login_frequency_7d')
    target : str
        Колонка с таргетом (0/1)
    bins : list
        Список границ интервалов, например: [0, 0.5, 1, 2, 3, 5, 8, 100]
    bin_labels : list, optional
        Названия бинов. Если None, генерируются автоматически

    Returns:
    --------
    pandas.DataFrame
        Таблица статистик по каждому бину с метриками надежности
    """

    # Копируем датафрейм чтобы не модифицировать оригинал
    df_work = df.copy()

    # Базовый fraud rate для расчета RRN
    baseline_fraud_rate = df_work[target].mean()

    # Присваиваем значениям бин
    df_work['bin'] = pd.cut(
        df_work[feature],
        bins=bins,
        labels=bin_labels,
        include_lowest=True,
        right=False  # [a, b) - правая граница не включается
    )

    # Группировка по бинам
    groups = df_work.groupby('bin', observed=True)

    # Сводная таблица
    result = []

    for bin_name, group in groups:

        values = group[feature].values

        # Защита от пустых бинов
        if len(values) == 0:
            result.append({
                'bin': bin_name,
                'count': 0,
                'fraud_count': 0,
                'fraud_rate': 0,
                'RRN': np.nan,
                'SE': np.nan,
                'CI_lower': np.nan,
                'CI_upper': np.nan,
                'CI_width': np.nan,
                'reliability': 'NO_DATA',
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'CV%': np.nan,
                'Q25': np.nan,
                'Q75': np.nan,
                'IQR': np.nan,
                'pct_nonzero': np.nan
            })
            continue

        # === КРИТИЧЕСКИЕ МЕТРИКИ ДЛЯ FRAUD DETECTION ===

        count = len(values)
        fraud_count = group[target].sum()
        fraud_rate = fraud_count / count if count > 0 else 0

        # 🔴 ИСПРАВЛЕНИЕ 1: RRN = отношение fraud rates, а не квартилей!
        # RRN (Relative Risk Ratio) - во сколько раз риск отличается от baseline
        RRN = fraud_rate / baseline_fraud_rate if baseline_fraud_rate > 0 else np.nan

        # 🔴 ИСПРАВЛЕНИЕ 2: SE для биномиального распределения (fraud rate)
        # Standard Error для пропорции
        SE_fraud = np.sqrt(fraud_rate * (1 - fraud_rate) / count) if count > 0 else np.nan

        # 95% Confidence Interval для fraud rate
        CI_lower = max(0, fraud_rate - 1.96 * SE_fraud) if not np.isnan(SE_fraud) else np.nan
        CI_upper = min(1, fraud_rate + 1.96 * SE_fraud) if not np.isnan(SE_fraud) else np.nan
        CI_width = CI_upper - CI_lower if not np.isnan(CI_lower) else np.nan

        # Оценка надежности бина
        if count >= 500 and CI_width < 0.03:
            reliability = 'HIGH'
        elif count >= 200 and CI_width < 0.05:
            reliability = 'MEDIUM'
        elif count >= 50:
            reliability = 'LOW'
        else:
            reliability = 'VERY_LOW'

        # === ДЕСКРИПТИВНАЯ СТАТИСТИКА ПРИЗНАКА ===

        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values, ddof=1) if count > 1 else 0

        # SE для среднего значения признака (не fraud rate!)
        SE_mean = std / np.sqrt(count) if count > 1 else 0

        # Coefficient of Variation
        CV = (std / mean * 100) if mean != 0 else 0

        Q25 = np.percentile(values, 25)
        Q75 = np.percentile(values, 75)
        IQR = Q75 - Q25  # Interquartile Range

        # Процент ненулевых значений
        pct_nonzero = (values != 0).sum() / count * 100 if count > 0 else 0

        result.append({
            # Идентификация бина
            'bin': str(bin_name),

            # === FRAUD МЕТРИКИ (КРИТИЧНО!) ===
            'count': count,
            'fraud_count': int(fraud_count),
            'fraud_rate': round(fraud_rate * 100, 3),  # в процентах
            'RRN': round(RRN, 3),  # ✅ ПРАВИЛЬНЫЙ RRN
            'SE': round(SE_fraud * 100, 3),  # ✅ SE для fraud rate (в процентах)
            'CI_lower': round(CI_lower * 100, 3),  # в процентах
            'CI_upper': round(CI_upper * 100, 3),  # в процентах
            'CI_width': round(CI_width * 100, 3),  # в процентах
            'reliability': reliability,

            # === ДЕСКРИПТИВНАЯ СТАТИСТИКА ===
            'mean': round(mean, 3),
            'median': round(median, 3),
            'std': round(std, 3),
            'SE_mean': round(SE_mean, 3),  # SE для среднего признака
            'CV%': round(CV, 2),
            'Q25': round(Q25, 3),
            'Q75': round(Q75, 3),
            'IQR': round(IQR, 3),
            'pct_nonzero': round(pct_nonzero, 2)
        })

    result_df = pd.DataFrame(result)

    # Сортировка по порядку бинов (если bin - категория с порядком)
    if bin_labels is not None:
        result_df['bin'] = pd.Categorical(result_df['bin'], categories=bin_labels, ordered=True)
        result_df = result_df.sort_values('bin')

    return result_df

def visualize_bin_analysis(analysis_df, feature_name):
    """
    Выводит красивую визуализацию результатов анализа бинов
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("=" * 80)
    print(f"📊 АНАЛИЗ ПРИЗНАКА: {feature_name}")
    print("=" * 80)

    # Основная таблица
    print("\n🎯 FRAUD МЕТРИКИ ПО БИНАМ:\n")
    display_cols = ['bin', 'count', 'fraud_count', 'fraud_rate', 'RRN',
                    'CI_width', 'reliability']
    print(analysis_df[display_cols].to_string(index=False))

    # Предупреждения о ненадежных бинах
    unreliable = analysis_df[analysis_df['reliability'].isin(['LOW', 'VERY_LOW'])]
    if len(unreliable) > 0:
        print("\n⚠️  ВНИМАНИЕ: Ненадежные бины (требуют объединения):")
        print(unreliable[['bin', 'count', 'fraud_rate', 'CI_width', 'reliability']].to_string(index=False))

    # Выводы
    print("\n✅ ВЫВОДЫ:")

    high_risk = analysis_df[
        (analysis_df['RRN'] >= 1.5) &
        (analysis_df['reliability'].isin(['HIGH', 'MEDIUM']))
        ]
    if len(high_risk) > 0:
        print(f"\n🔴 Бины с ПОВЫШЕННЫМ риском (RRN ≥ 1.5, надежные):")
        for _, row in high_risk.iterrows():
            print(f"   • {row['bin']}: FR={row['fraud_rate']:.2f}%, RRN={row['RRN']:.2f}, n={row['count']}")

    low_risk = analysis_df[
        (analysis_df['RRN'] <= 0.7) &
        (analysis_df['reliability'].isin(['HIGH', 'MEDIUM']))
        ]
    if len(low_risk) > 0:
        print(f"\n🟢 Бины с ПОНИЖЕННЫМ риском (RRN ≤ 0.7, надежные):")
        for _, row in low_risk.iterrows():
            print(f"   • {row['bin']}: FR={row['fraud_rate']:.2f}%, RRN={row['RRN']:.2f}, n={row['count']}")

    # График
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Fraud Rate с доверительными интервалами
    ax1 = axes[0, 0]
    reliable = analysis_df[analysis_df['reliability'].isin(['HIGH', 'MEDIUM'])]
    unreliable = analysis_df[~analysis_df['reliability'].isin(['HIGH', 'MEDIUM'])]

    ax1.bar(range(len(reliable)), reliable['fraud_rate'],
            color='steelblue', alpha=0.7, label='Надежные')
    ax1.bar(range(len(reliable), len(analysis_df)), unreliable['fraud_rate'],
            color='lightcoral', alpha=0.5, label='Ненадежные')

    # Доверительные интервалы
    for i, row in analysis_df.iterrows():
        ax1.errorbar(i, row['fraud_rate'],
                     yerr=row['CI_width'] / 2,
                     fmt='none', color='black', capsize=5, alpha=0.5)

    ax1.set_xlabel('Bin')
    ax1.set_ylabel('Fraud Rate (%)')
    ax1.set_title('Fraud Rate по бинам с 95% CI')
    ax1.set_xticks(range(len(analysis_df)))
    ax1.set_xticklabels(analysis_df['bin'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. RRN
    ax2 = axes[0, 1]
    colors = ['green' if r <= 0.7 else 'red' if r >= 1.5 else 'gray'
              for r in analysis_df['RRN']]
    ax2.bar(range(len(analysis_df)), analysis_df['RRN'], color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='--', label='Baseline (RRN=1.0)')
    ax2.axhline(y=1.5, color='red', linestyle=':', alpha=0.5, label='High Risk (1.5)')
    ax2.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='Low Risk (0.7)')
    ax2.set_xlabel('Bin')
    ax2.set_ylabel('RRN (Relative Risk Ratio)')
    ax2.set_title('Относительный риск по бинам')
    ax2.set_xticks(range(len(analysis_df)))
    ax2.set_xticklabels(analysis_df['bin'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Размер выборки и надежность
    ax3 = axes[1, 0]
    reliability_colors = {
        'HIGH': 'green',
        'MEDIUM': 'orange',
        'LOW': 'red',
        'VERY_LOW': 'darkred',
        'NO_DATA': 'gray'
    }
    colors = [reliability_colors.get(r, 'gray') for r in analysis_df['reliability']]
    ax3.bar(range(len(analysis_df)), analysis_df['count'], color=colors, alpha=0.7)
    ax3.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='High reliability (500)')
    ax3.axhline(y=200, color='orange', linestyle='--', alpha=0.5, label='Medium reliability (200)')
    ax3.set_xlabel('Bin')
    ax3.set_ylabel('Sample Size')
    ax3.set_title('Размер выборки и надежность')
    ax3.set_xticks(range(len(analysis_df)))
    ax3.set_xticklabels(analysis_df['bin'], rotation=45, ha='right')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Ширина доверительного интервала
    ax4 = axes[1, 1]
    ax4.bar(range(len(analysis_df)), analysis_df['CI_width'],
            color='steelblue', alpha=0.7)
    ax4.axhline(y=3, color='green', linestyle='--', alpha=0.5, label='Узкий CI (3%)')
    ax4.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Приемлемый CI (5%)')
    ax4.set_xlabel('Bin')
    ax4.set_ylabel('CI Width (%)')
    ax4.set_title('Ширина доверительного интервала')
    ax4.set_xticks(range(len(analysis_df)))
    ax4.set_xticklabels(analysis_df['bin'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()

num = 2

if __name__ == '__main__':
    if num == 0:
        compare_with_join()
    elif num == 1:
        df = load_data()
        logins_0 = df[df['target'] == 0]['logins_last_7_days']
        logins_1 = df[df['target'] == 1]['logins_last_7_days']

        print(f"📊 ДАННЫЕ ДЛЯ АНАЛИЗА:")
        print(f"Target=0: {len(logins_0)} записей")
        print(f"Target=1: {len(logins_1)} записей")
        print(f"Общий fraud rate: {len(logins_1) / len(df) * 100:.2f}%")

        automatic_binning_analysis(df)
        trend_data = trend_analysis(df)
        outliers = tail_analysis(df)
        r_squared = nonlinear_analysis(df)
        likelihood_ratio, x_range = kde_analysis(df)

        final_report(df, trend_data, r_squared, likelihood_ratio, x_range)

        # stats_df = calculate_detailed_stats(df, value_column='logins_last_7_days', target_column='target')
        # formatted_stats = print_detailed_stats(stats_df)

    elif num == 2:
        df = load_data()
        bins = [0, 5, 20, 999]
        labels = ["0–4", "5–19", "20+"]
        result = analyze_manual_bins(
            df=df,
            feature='logins_last_7_days',
            target='target',
            bins=bins,
            bin_labels=labels
        )

        visualize_bin_analysis(result, 'logins_last_7_days')