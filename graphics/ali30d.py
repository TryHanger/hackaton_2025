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
            CREATE TABLE avg_login_interval_30d AS
            SELECT 
                i.cst_dim_id,
                i.transdate,
                i.avg_login_interval_30d,
                p.target
            FROM info_db.unique_transactions i
            LEFT JOIN perevody_db.unique_transactions p
                ON i.cst_dim_id = p.cst_dim_id 
                AND i.transdate = p.transdate
            WHERE i.avg_login_interval_30d IS NOT NULL
        """)

        # Статистика
        result_cursor.execute("SELECT COUNT(*) FROM avg_login_interval_30d")
        total_count = result_cursor.fetchone()[0]

        result_cursor.execute("SELECT COUNT(*) FROM avg_login_interval_30d WHERE target IS NOT NULL")
        matched_count = result_cursor.fetchone()[0]

        result_cursor.execute("SELECT COUNT(*) FROM avg_login_interval_30d WHERE target = 1")
        target_1_count = result_cursor.fetchone()[0]

        result_cursor.execute("SELECT COUNT(*) FROM avg_login_interval_30d WHERE target = 0")
        target_0_count = result_cursor.fetchone()[0]

        print("📊 СТАТИСТИКА СРАВНЕНИЯ:")
        print(f"Всего записей: {total_count}")
        print(f"Найдено соответствий: {matched_count}")
        print(f"Не найдено соответствий: {total_count - matched_count}")
        print(f"Target = 1: {target_1_count}")
        print(f"Target = 0: {target_0_count}")

        # Выборка результатов
        result_cursor.execute("SELECT * FROM avg_login_interval_30d LIMIT 10")
        sample_results = result_cursor.fetchall()

        print("\n🔍 ПЕРВЫЕ 10 РЕЗУЛЬТАТОВ:")
        print("ID | Дата | avg_login_interval_30d | Target")
        print("-" * 50)
        for row in sample_results:
            print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")

        # Сохраняем результаты в файл если нужно
        result_cursor.execute("""
            ATTACH DATABASE '../comparison_results.db' AS results_db
        """)
        result_cursor.execute("""
            CREATE TABLE results_db.avg_login_interval_30d AS
            SELECT * FROM avg_login_interval_30d
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
    import sqlite3
    import pandas as pd
    from pathlib import Path

    script_dir = Path(__file__).parent
    db_path = script_dir.parent / 'comparison_results.db'
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT avg_login_interval_30d, target FROM avg_login_interval_30d WHERE target IS NOT NULL AND avg_login_interval_30d IS NOT NULL",
        conn
    )
    conn.close()

    # --- ИСПРАВЛЕНИЕ: КОНВЕРТАЦИЯ СЕКУНД В ДНИ ---

    # 86400 секунд в сутках (24 * 60 * 60).
    SECONDS_IN_DAY = 86400

    # Сначала обрабатываем специальные значения (-1)
    df['has_no_data'] = (df['avg_login_interval_30d'] == -1).astype(int)

    # Преобразуем интервал из секунд в дни, используя маску, чтобы не трогать -1
    df.loc[df['avg_login_interval_30d'] != -1, 'avg_login_interval_30d'] = \
        df.loc[df['avg_login_interval_30d'] != -1, 'avg_login_interval_30d'] / SECONDS_IN_DAY

    # --- ДАЛЬНЕЙШАЯ ПРЕДОБРАБОТКА (ИЗМЕНЕНА ПОД ДНИ) ---

    print("🔧 ПРЕДОБРАБОТКА ДАННЫХ avg_login_interval_30d (ПЕРЕВЕДЕНО В ДНИ):")
    print(
        f"Исходный диапазон (ДНИ): {df['avg_login_interval_30d'].min():.1f} - {df['avg_login_interval_30d'].max():.1f}")

    # Обрезаем экстремальные значения для анализа (макс 30 дней)
    # Теперь max_login = 2712540.0 секунд (31.4 дня) обрежется до 30.0 дней.
    df['interval_processed'] = df['avg_login_interval_30d'].clip(upper=30)

    # Создаем инвертированный признак (чем меньше интервал = выше активность)
    df['activity_level'] = 30 - df['interval_processed']
    df['activity_level'] = df['activity_level'].clip(lower=0)

    print(
        f"После обработки (обрезка до 30 дней): {df['interval_processed'].min():.1f} - {df['interval_processed'].max():.1f}")
    print(f"Отрицательных значений: {df['has_no_data'].sum()}")

    return df


def automatic_binning_analysis(df):
    """Автоматический биннинг с разными стратегиями."""

    print("\n" + "=" * 60)
    print("1. АВТОМАТИЧЕСКИЙ БИННИНГ")
    print("=" * 60)

    # ИСПОЛЬЗУЕМ ОБРАБОТАННЫЕ ДАННЫЕ
    X = df['interval_processed'].values.reshape(-1, 1)

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
            'interval_processed': ['min', 'max', 'count'],
            'target': ['sum', 'mean']
        }).round(3)

        bin_stats.columns = ['min_interval', 'max_interval', 'total_count', 'fraud_count', 'fraud_rate']
        bin_stats['fraud_rate_pct'] = (bin_stats['fraud_rate'] * 100).round(2)

        # Визуализация
        ax = axes[idx]
        bins_centers = (bin_stats['min_interval'] + bin_stats['max_interval']) / 2
        ax.plot(bins_centers, bin_stats['fraud_rate_pct'], 'o-', linewidth=2, markersize=8)
        ax.set_title(f'{name} биннинг\n({strategy})', fontsize=14)
        ax.set_xlabel('Средний интервал между логинами (дни)', fontsize=12)  # ИСПРАВЛЕНА ПОДПИСЬ
        ax.set_ylabel('Fraud Rate (%)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Добавляем значения
        for i, (center, rate) in enumerate(zip(bins_centers, bin_stats['fraud_rate_pct'])):
            ax.annotate(f'{rate}%', (center, rate), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9)

        print(f"\n{name} биннинг:")
        print(bin_stats[['min_interval', 'max_interval', 'total_count', 'fraud_rate_pct']])

    # Ручной биннинг на основе анализа
    axes[3].axis('off')
    plt.tight_layout()
    plt.show()

    return df


def trend_analysis(df):
    """Статистический анализ тренда риска для avg_login_interval_30d."""

    print("\n" + "=" * 60)
    print("2. СТАТИСТИЧЕСКИЙ АНАЛИЗ ТРЕНДА")
    print("=" * 60)

    # ОПТИМАЛЬНЫЕ БИНЫ ДЛЯ ИНТЕРВАЛА МЕЖДУ ЛОГИНАМИ
    bins = [-2, -0.5, 1, 3, 7, 14, 30, 31]  # включаем -1 и обрезаем на 30
    labels = ['Нет_данных', 'Очень_часто', 'Ежедневно', 'Раз_в_3дня', 'Раз_в_неделю', 'Раз_в_2недели', 'Редко']

    df['bin'] = pd.cut(df['avg_login_interval_30d'], bins=bins, labels=labels, right=False)
    trend_data = df.groupby('bin').agg({
        'target': ['count', 'sum', 'mean'],
        'avg_login_interval_30d': 'median'
    }).round(4)

    trend_data.columns = ['total', 'fraud_count', 'fraud_rate', 'median_interval']
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

    ax1.set_title('Fraud Rate vs Интервал между логинами', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Уровень активности', fontsize=12)
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
                f"{row['fraud_rate_pct']}%", f"{row['median_interval']:.1f}"
            ])

    table = ax2.table(cellText=table_data,
                      colLabels=['Активность', 'Всего', 'Fraud', 'Fraud%', 'Медиана_интервала'],
                      loc='center',
                      cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax2.set_title('Детальная статистика по уровням активности', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return trend_data


def tail_analysis(df):
    """Анализ редких значений (хвостов распределения)."""

    print("\n" + "=" * 60)
    print("3. АНАЛИЗ ХВОСТОВ РАСПРЕДЕЛЕНИЯ")
    print("=" * 60)

    # ИСПОЛЬЗУЕМ ОБРАБОТАННЫЕ ДАННЫЕ
    mean_interval = df['interval_processed'].mean()
    std_interval = df['interval_processed'].std()
    outlier_threshold = mean_interval + 3 * std_interval

    outliers = df[df['interval_processed'] > outlier_threshold]
    normal_data = df[df['interval_processed'] <= outlier_threshold]

    print(f"Порог выбросов: > {outlier_threshold:.1f} дней")
    print(f"Выбросов: {len(outliers)} записей ({len(outliers) / len(df) * 100:.2f}%)")
    print(f"Выбросы - Fraud Rate: {outliers['target'].mean() * 100:.2f}%")
    print(f"Нормальные значения - Fraud Rate: {normal_data['target'].mean() * 100:.2f}%")

    # Анализ пользователей без данных
    no_data = df[df['has_no_data'] == 1]
    if len(no_data) > 0:
        print(f"\n📊 ПОЛЬЗОВАТЕЛИ БЕЗ ДАННЫХ:")
        print(f"Количество: {len(no_data)} записей")
        print(f"Fraud Rate: {no_data['target'].mean() * 100:.2f}%")

    # Визуализация хвостов
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Распределение с выделением выбросов
    ax1.hist(normal_data['interval_processed'], bins=30, alpha=0.7,
             color='blue', label=f'Нормальные (≤{outlier_threshold:.0f}дн)')
    ax1.hist(outliers['interval_processed'], bins=10, alpha=0.7,
             color='red', label=f'Выбросы (> {outlier_threshold:.0f}дн)')
    ax1.axvline(outlier_threshold, color='black', linestyle='--', linewidth=2, label='Порог выбросов')
    ax1.set_title('Распределение интервалов между логинами', fontsize=14)
    ax1.set_xlabel('Интервал между логинами (дни)', fontsize=12)
    ax1.set_ylabel('Частота', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fraud rate по перцентилям активности
    percentiles = np.arange(0, 101, 5)
    percentile_values = np.percentile(df['activity_level'], percentiles)  # используем активность!
    percentile_fraud = []

    for p in percentiles:
        threshold = np.percentile(df['activity_level'], p)
        high_activity = df[df['activity_level'] >= threshold]  # высокий уровень активности
        if len(high_activity) > 0:
            fraud_rate = high_activity['target'].mean() * 100
            percentile_fraud.append(fraud_rate)
        else:
            percentile_fraud.append(0)

    ax2.plot(percentiles, percentile_fraud, 'o-', linewidth=2, markersize=4)
    ax2.set_title('Fraud Rate по уровню активности', fontsize=14)
    ax2.set_xlabel('Уровень активности (перцентиль)', fontsize=12)
    ax2.set_ylabel('Fraud Rate (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Добавляем ключевые точки
    key_percentiles = [5, 25, 50, 75, 95]
    for p in key_percentiles:
        closest_p = min(percentiles, key=lambda x: abs(x - p))
        idx = list(percentiles).index(closest_p)
        actual_activity = np.percentile(df['activity_level'], p)

        ax2.annotate(f'{p}%: {percentile_fraud[idx]:.1f}%\n(активность≥{actual_activity:.1f})',
                     (p, percentile_fraud[idx]),
                     xytext=(10, 5), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                     fontsize=9)

    plt.tight_layout()
    plt.show()

    # Анализ самых активных пользователей
    print("\n📊 АНАЛИЗ САМЫХ АКТИВНЫХ ПОЛЬЗОВАТЕЛЕЙ:")
    for p in [90, 95, 99]:
        threshold = np.percentile(df['activity_level'], p)
        active_users = df[df['activity_level'] >= threshold]
        if len(active_users) > 0:
            fraud_rate = active_users['target'].mean() * 100
            interval = 30 - threshold  # переводим обратно в интервал
            print(
                f"Топ {100 - p}% (интервал≤{interval:.1f}дн): {len(active_users)} записей, Fraud Rate: {fraud_rate:.2f}%")

    return outliers


def nonlinear_analysis(df):
    """Поиск нелинейных зависимостей для avg_login_interval_30d."""

    print("\n" + "=" * 60)
    print("4. ПОИСК НЕЛИНЕЙНЫХ ЗАВИСИМОСТЕЙ")
    print("=" * 60)

    # Создаем копию данных и сортируем по интервалу
    sorted_df = df.copy().sort_values('interval_processed').reset_index(drop=True)

    # Адаптивное окно для скользящего среднего
    window_size = min(100, len(df) // 10)
    sorted_df['rolling_fraud'] = sorted_df['target'].rolling(window=window_size, center=True).mean()

    # Удаляем строки с NaN в rolling_fraud
    valid_data = sorted_df.dropna(subset=['rolling_fraud']).copy()

    # Полиномиальная аппроксимация
    X_poly = valid_data['interval_processed'].values
    Y_poly = valid_data['rolling_fraud'].values

    # Аппроксимации разной степени
    degrees = [1, 2, 3, 4]
    colors = ['red', 'blue', 'green', 'purple']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # График 1: Скользящее среднее
    ax1.scatter(sorted_df['interval_processed'], sorted_df['target'],
                alpha=0.1, color='gray', s=1, label='Исходные данные')
    ax1.plot(valid_data['interval_processed'], valid_data['rolling_fraud'] * 100,
             linewidth=3, color='black', label=f'Скользящее среднее (n={window_size})')
    ax1.set_title('Fraud Rate vs Интервал между логинами\n(скользящее среднее)', fontsize=14)
    ax1.set_xlabel('Интервал между логинами (дни)', fontsize=12)
    ax1.set_ylabel('Fraud Rate (%)', fontsize=12)
    ax1.set_xlim(0, 30)  # ограничиваем разумным диапазоном
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
    ax2.set_xlabel('Интервал между логинами (дни)', fontsize=12)
    ax2.set_ylabel('Fraud Rate (%)', fontsize=12)
    ax2.set_xlim(0, 30)
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
                    print(f"  {point:.1f} дней: Fraud Rate = {fraud_at_point:.2f}% ({point_type})")
            else:
                print("  Критические точки не найдены")

        except Exception as e:
            print(f"Ошибка при анализе критических точек: {e}")

    return r_squared


def kde_analysis(df):
    """Kernel Density Estimation для распределений интервалов."""

    print("\n" + "=" * 60)
    print("5. KERNEL DENSITY ESTIMATION")
    print("=" * 60)

    # ИСПОЛЬЗУЕМ ОБРАБОТАННЫЕ ДАННЫЕ
    interval_0 = df[df['target'] == 0]['interval_processed']
    interval_1 = df[df['target'] == 1]['interval_processed']

    # Ограничиваем диапазон для лучшей визуализации
    max_val = 30  # максимальный интервал в днях
    interval_0_trimmed = interval_0[interval_0 <= max_val]
    interval_1_trimmed = interval_1[interval_1 <= max_val]

    print(f"Визуализация KDE для диапазона 0-{max_val} дней")
    print(f"Охватывает {len(interval_0_trimmed) / len(interval_0) * 100:.1f}% легитимных операций")
    print(f"Охватывает {len(interval_1_trimmed) / len(interval_1) * 100:.1f}% мошеннических операций")

    # KDE с разными bandwidth
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    bandwidths = [0.5, 1.0, 2.0, 'auto']
    titles = ['Small bandwidth (0.5)', 'Medium bandwidth (1.0)',
              'Large bandwidth (2.0)', 'Auto bandwidth']

    for idx, (bw, title) in enumerate(zip(bandwidths, titles)):
        ax = axes[idx // 2, idx % 2]

        # KDE для каждого класса
        if bw == 'auto':
            kde_0 = stats.gaussian_kde(interval_0_trimmed)
            kde_1 = stats.gaussian_kde(interval_1_trimmed)
        else:
            kde_0 = stats.gaussian_kde(interval_0_trimmed, bw_method=bw)
            kde_1 = stats.gaussian_kde(interval_1_trimmed, bw_method=bw)

        x_range = np.linspace(0, max_val, 200)

        ax.plot(x_range, kde_0(x_range), label='Target=0 (Легитимные)',
                color='blue', linewidth=2)
        ax.plot(x_range, kde_1(x_range), label='Target=1 (Мошеннические)',
                color='red', linewidth=2)

        ax.set_title(f'KDE: {title}', fontsize=14)
        ax.set_xlabel('Интервал между логинами (дни)', fontsize=12)
        ax.set_ylabel('Плотность вероятности', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_val)

    plt.tight_layout()
    plt.show()

    # Ratio of densities (сигнал для классификации)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Auto KDE для ratio анализа
    kde_0_auto = stats.gaussian_kde(interval_0_trimmed)
    kde_1_auto = stats.gaussian_kde(interval_1_trimmed)

    x_range = np.linspace(0, max_val, 200)
    density_0 = kde_0_auto(x_range)
    density_1 = kde_1_auto(x_range)

    # Отношение плотностей
    likelihood_ratio = np.where(density_0 > 0, density_1 / density_0, 0)

    ax1.plot(x_range, density_0, label='P(x|Target=0)', color='blue', linewidth=2)
    ax1.plot(x_range, density_1, label='P(x|Target=1)', color='red', linewidth=2)
    ax1.set_title('Вероятностные распределения', fontsize=14)
    ax1.set_xlabel('Интервал между логинами (дни)', fontsize=12)
    ax1.set_ylabel('Плотность вероятности', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_range, likelihood_ratio, color='purple', linewidth=3)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Порог (ratio=1)')
    ax2.set_title('Likelihood Ratio: P(x|Target=1) / P(x|Target=0)', fontsize=14)
    ax2.set_xlabel('Интервал между логинами (дни)', fontsize=12)
    ax2.set_ylabel('Likelihood Ratio', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Области где мошенники более вероятны
    fraud_preferred = x_range[likelihood_ratio > 1]
    if len(fraud_preferred) > 0:
        print(f"🔍 ОБЛАСТИ ПРЕИМУЩЕСТВА МОШЕННИКОВ:")
        print(f"  Интервал от {fraud_preferred[0]:.1f} до {fraud_preferred[-1]:.1f} дней")

        # Пик отношения правдоподобия
        peak_idx = np.argmax(likelihood_ratio)
        peak_x = x_range[peak_idx]
        peak_ratio = likelihood_ratio[peak_idx]
        print(f"  Пик отношения: {peak_ratio:.2f} при {peak_x:.1f} днях")

    plt.tight_layout()
    plt.show()

    return likelihood_ratio, x_range


def final_report(df, trend_data, r_squared, likelihood_ratio, x_range):
    """Финальный отчет с выводами для avg_login_interval_30d."""

    print("\n" + "=" * 80)
    print("🎯 ИТОГОВЫЙ ОТЧЕТ АНАЛИЗА (avg_login_interval_30d)")
    print("=" * 80)

    # Ключевые метрики
    total_fraud_rate = df['target'].mean() * 100
    correlation = df['interval_processed'].corr(df['target'])

    print("📈 КЛЮЧЕВЫЕ МЕТРИКИ:")
    print(f"  • Общий Fraud Rate: {total_fraud_rate:.2f}%")
    print(f"  • Корреляция интервал-риск: {correlation:.4f}")
    print(f"  • Оптимальная степень полинома: {max(r_squared, key=r_squared.get)}")
    print(f"  • Максимальный likelihood ratio: {np.max(likelihood_ratio):.2f}")

    # Практические рекомендации
    print("\n💡 ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")

    # Анализ ключевых групп
    very_active = df[df['interval_processed'] <= 1]  # ежедневно
    active = df[(df['interval_processed'] > 1) & (df['interval_processed'] <= 7)]  # раз в неделю
    inactive = df[df['interval_processed'] > 14]  # реже чем раз в 2 недели

    print(f"  1. ОЧЕНЬ АКТИВНЫЕ (интервал ≤ 1 день):")
    print(f"     • {len(very_active)} записей ({len(very_active) / len(df) * 100:.1f}%)")
    print(f"     • Fraud Rate: {very_active['target'].mean() * 100:.2f}%")

    print(f"  2. АКТИВНЫЕ (1-7 дней):")
    print(f"     • {len(active)} записей ({len(active) / len(df) * 100:.1f}%)")
    print(f"     • Fraud Rate: {active['target'].mean() * 100:.2f}%")

    print(f"  3. НЕАКТИВНЫЕ (>14 дней):")
    print(f"     • {len(inactive)} записей ({len(inactive) / len(df) * 100:.1f}%)")
    print(f"     • Fraud Rate: {inactive['target'].mean() * 100:.2f}%")

    # Области максимального сигнала
    peak_signal_idx = np.argmax(likelihood_ratio)
    peak_signal_x = x_range[peak_signal_idx]

    print(f"  4. МАКСИМАЛЬНЫЙ СИГНАЛ:")
    print(f"     • {peak_signal_x:.1f} дней (likelihood ratio = {np.max(likelihood_ratio):.2f})")

    # Статистическая значимость
    from scipy.stats import mannwhitneyu
    interval_0 = df[df['target'] == 0]['interval_processed']
    interval_1 = df[df['target'] == 1]['interval_processed']

    stat, p_value = mannwhitneyu(interval_0, interval_1, alternative='two-sided')

    print(f"\n📊 СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ:")
    print(f"  • Тест Манна-Уитни: p-value = {p_value:.6f}")
    print(f"  • Различие распределений: {'ЗНАЧИМО' if p_value < 0.05 else 'НЕЗНАЧИМО'}")

    # Финальная визуализация
    plt.figure(figsize=(12, 8))

    # Композитный график
    plt.subplot(2, 1, 1)

    # KDE распределения
    max_kde = 30
    interval_0_trimmed = interval_0[interval_0 <= max_kde]
    interval_1_trimmed = interval_1[interval_1 <= max_kde]

    kde_0 = stats.gaussian_kde(interval_0_trimmed)
    kde_1 = stats.gaussian_kde(interval_1_trimmed)
    x_kde = np.linspace(0, max_kde, 200)

    plt.plot(x_kde, kde_0(x_kde), label='Легитимные (Target=0)', color='blue', linewidth=2)
    plt.plot(x_kde, kde_1(x_kde), label='Мошеннические (Target=1)', color='red', linewidth=2)
    plt.title('Распределение интервалов между логинами', fontsize=16, fontweight='bold')
    plt.xlabel('Интервал между логинами (дни)', fontsize=12)
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


num = 0

if __name__ == '__main__':
    if num == 0:
        compare_with_join()
    elif num == 1:
        df = load_data()
        interval_0 = df[df['target'] == 0]['interval_processed']
        interval_1 = df[df['target'] == 1]['interval_processed']

        print(f"📊 ДАННЫЕ ДЛЯ АНАЛИЗА:")
        print(f"Target=0: {len(interval_0)} записей")
        print(f"Target=1: {len(interval_1)} записей")
        print(f"Общий fraud rate: {len(interval_1) / len(df) * 100:.2f}%")

        automatic_binning_analysis(df)
        trend_data = trend_analysis(df)
        outliers = tail_analysis(df)
        r_squared = nonlinear_analysis(df)
        likelihood_ratio, x_range = kde_analysis(df)

        final_report(df, trend_data, r_squared, likelihood_ratio, x_range)

    elif num == 2:
        df = load_data()
        bins = [-2, 0.3, 1.0, 30.1]
        labels = ['Высокоактивные', 'Активные', 'Малоактивные']

        result = analyze_manual_bins(
            df=df,
            feature='avg_login_interval_30d',
            target='target',
            bins=bins,
            bin_labels=labels
        )

        visualize_bin_analysis(result, 'avg_login_interval_30d')