#!/usr/bin/env python3
"""
Этап 4: Проверка на стационарность и статистические тесты
Модуль для анализа стационарности временных рядов недвижимости
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import logging

logger = logging.getLogger(__name__)

class StationarityAnalyzer:
    """Класс для анализа стационарности временных рядов"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_stationarity(self, df, target_column='MA', date_column='saledate'):
        """
        Полный анализ стационарности временного ряда
        
        Args:
            df: DataFrame с данными
            target_column: название целевой переменной
            date_column: название колонки с датами
            
        Returns:
            Словарь с результатами анализа
        """
        logger.info("Начинаем анализ стационарности")
        
        results = {}
        
        # Подготавливаем данные
        if date_column in df.columns:
            df_ts = df.set_index(date_column)[target_column].dropna()
        else:
            df_ts = df[target_column].dropna()
        
        # Визуальный анализ
        results['visual_analysis'] = self._visual_stationarity_analysis(df_ts)
        
        # Скользящие статистики
        results['rolling_statistics'] = self._rolling_statistics_analysis(df_ts)
        
        # Статистические тесты
        results['statistical_tests'] = self._statistical_tests(df_ts)
        
        # Дифференцирование
        results['differencing'] = self._differencing_analysis(df_ts)
        
        self.results = results
        logger.info("Анализ стационарности завершен")
        return results
    
    def _visual_stationarity_analysis(self, ts):
        """Визуальный анализ стационарности"""
        logger.info("Выполняем визуальный анализ стационарности")
        
        # Проверяем наличие тренда
        trend_present = self._check_trend(ts)
        
        # Проверяем изменение дисперсии
        variance_stable = self._check_variance_stability(ts)
        
        return {
            'trend_present': trend_present,
            'variance_stable': variance_stable,
            'stationary_by_eye': not trend_present and variance_stable
        }
    
    def _check_trend(self, ts):
        """Проверка наличия тренда"""
        # Простая проверка: сравниваем средние значения первой и второй половины
        mid_point = len(ts) // 2
        first_half_mean = ts[:mid_point].mean()
        second_half_mean = ts[mid_point:].mean()
        
        # Если разница больше 10% от общего среднего, считаем тренд присутствующим
        overall_mean = ts.mean()
        trend_threshold = 0.1 * overall_mean
        
        return abs(second_half_mean - first_half_mean) > trend_threshold
    
    def _check_variance_stability(self, ts):
        """Проверка стабильности дисперсии"""
        # Разделяем ряд на части и сравниваем дисперсии
        n_parts = 4
        part_size = len(ts) // n_parts
        
        variances = []
        for i in range(n_parts):
            start_idx = i * part_size
            end_idx = (i + 1) * part_size if i < n_parts - 1 else len(ts)
            part_var = ts.iloc[start_idx:end_idx].var()
            variances.append(part_var)
        
        # Если максимальная дисперсия больше чем в 2 раза минимальной, считаем нестабильной
        max_var = max(variances)
        min_var = min(variances)
        
        return max_var / min_var < 2.0
    
    def _rolling_statistics_analysis(self, ts, window_sizes=[4, 8, 12]):
        """Анализ скользящих статистик"""
        logger.info("Вычисляем скользящие статистики")
        
        rolling_stats = {}
        
        for window in window_sizes:
            if len(ts) >= window:
                rolling_mean = ts.rolling(window=window).mean()
                rolling_std = ts.rolling(window=window).std()
                
                rolling_stats[f'window_{window}'] = {
                    'rolling_mean': rolling_mean,
                    'rolling_std': rolling_std,
                    'mean_stability': self._check_mean_stability(rolling_mean),
                    'std_stability': self._check_std_stability(rolling_std)
                }
        
        return rolling_stats
    
    def _check_mean_stability(self, rolling_mean):
        """Проверка стабильности скользящего среднего"""
        # Проверяем, насколько сильно колеблется скользящее среднее
        mean_range = rolling_mean.max() - rolling_mean.min()
        overall_mean = rolling_mean.mean()
        
        # Если диапазон больше 20% от общего среднего, считаем нестабильным
        return mean_range / overall_mean < 0.2
    
    def _check_std_stability(self, rolling_std):
        """Проверка стабильности скользящего стандартного отклонения"""
        # Аналогично для стандартного отклонения
        std_range = rolling_std.max() - rolling_std.min()
        overall_std = rolling_std.mean()
        
        return std_range / overall_std < 0.3
    
    def _statistical_tests(self, ts):
        """Статистические тесты на стационарность"""
        logger.info("Выполняем статистические тесты")
        
        # Тест Дики-Фуллера (ADF)
        adf_result = adfuller(ts, autolag='AIC')
        
        # Тест KPSS
        try:
            kpss_result = kpss(ts, regression='c')
        except Exception as e:
            logger.warning(f"Ошибка в тесте KPSS: {e}")
            kpss_result = (np.nan, np.nan, np.nan, {'10%': np.nan, '5%': np.nan, '2.5%': np.nan, '1%': np.nan})
        
        # Интерпретация результатов
        adf_stationary = adf_result[1] < 0.05
        kpss_stationary = kpss_result[1] > 0.05
        
        return {
            'adf': {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_stationary
            },
            'kpss': {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_stationary
            },
            'overall_stationary': adf_stationary and kpss_stationary
        }
    
    def _differencing_analysis(self, ts):
        """Анализ дифференцирования"""
        logger.info("Анализируем дифференцирование")
        
        differencing_results = {}
        
        # Первое дифференцирование
        ts_diff1 = ts.diff().dropna()
        if len(ts_diff1) > 0:
            diff1_tests = self._statistical_tests(ts_diff1)
            differencing_results['first_difference'] = {
                'data': ts_diff1,
                'tests': diff1_tests,
                'stationary': diff1_tests['overall_stationary']
            }
        
        # Второе дифференцирование (если первое не помогло)
        if not differencing_results.get('first_difference', {}).get('stationary', False):
            ts_diff2 = ts_diff1.diff().dropna()
            if len(ts_diff2) > 0:
                diff2_tests = self._statistical_tests(ts_diff2)
                differencing_results['second_difference'] = {
                    'data': ts_diff2,
                    'tests': diff2_tests,
                    'stationary': diff2_tests['overall_stationary']
                }
        
        return differencing_results
    
    def create_stationarity_plots(self, df, target_column='MA', date_column='saledate'):
        """Создание графиков для анализа стационарности"""
        logger.info("Создаем графики стационарности")
        
        if date_column in df.columns:
            ts = df.set_index(date_column)[target_column].dropna()
        else:
            ts = df[target_column].dropna()
        
        plots = {}
        
        # График исходного ряда
        fig_original = go.Figure()
        fig_original.add_trace(go.Scatter(y=ts.values, name='Исходный ряд'))
        fig_original.update_layout(title='Исходный временной ряд', xaxis_title='Время', yaxis_title=target_column)
        plots['original_series'] = fig_original
        
        # График с трендом
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(y=ts.values, name='Исходный ряд'))
        
        # Добавляем линейный тренд
        x = np.arange(len(ts))
        z = np.polyfit(x, ts.values, 1)
        p = np.poly1d(z)
        fig_trend.add_trace(go.Scatter(y=p(x), name='Линейный тренд', line=dict(dash='dash')))
        
        fig_trend.update_layout(title='Временной ряд с трендом', xaxis_title='Время', yaxis_title=target_column)
        plots['trend_analysis'] = fig_trend
        
        # График скользящих статистик
        rolling_mean = ts.rolling(window=4).mean()
        rolling_std = ts.rolling(window=4).std()
        
        fig_rolling = make_subplots(rows=2, cols=1, subplot_titles=('Скользящее среднее', 'Скользящее стандартное отклонение'))
        
        fig_rolling.add_trace(go.Scatter(y=ts.values, name='Исходный ряд'), row=1, col=1)
        fig_rolling.add_trace(go.Scatter(y=rolling_mean.values, name='Скользящее среднее'), row=1, col=1)
        
        fig_rolling.add_trace(go.Scatter(y=rolling_std.values, name='Скользящее стд. отклонение'), row=2, col=1)
        
        fig_rolling.update_layout(height=600, title_text='Скользящие статистики')
        plots['rolling_statistics'] = fig_rolling
        
        # График дифференцирования
        ts_diff1 = ts.diff().dropna()
        
        fig_diff = make_subplots(rows=2, cols=1, subplot_titles=('Исходный ряд', 'Первое дифференцирование'))
        
        fig_diff.add_trace(go.Scatter(y=ts.values, name='Исходный ряд'), row=1, col=1)
        fig_diff.add_trace(go.Scatter(y=ts_diff1.values, name='Первое дифференцирование'), row=2, col=1)
        
        fig_diff.update_layout(height=600, title_text='Дифференцирование')
        plots['differencing'] = fig_diff
        
        return plots
    
    def generate_stationarity_report(self):
        """Генерация отчета о стационарности"""
        if not self.results:
            return "Анализ стационарности не выполнен"
        
        report = []
        report.append("=== ОТЧЕТ АНАЛИЗА СТАЦИОНАРНОСТИ ===\n")
        
        # Визуальный анализ
        if 'visual_analysis' in self.results:
            visual = self.results['visual_analysis']
            report.append("=== ВИЗУАЛЬНЫЙ АНАЛИЗ ===")
            report.append(f"Тренд присутствует: {'Да' if visual['trend_present'] else 'Нет'}")
            report.append(f"Дисперсия стабильна: {'Да' if visual['variance_stable'] else 'Нет'}")
            report.append(f"Стационарен на глаз: {'Да' if visual['stationary_by_eye'] else 'Нет'}")
        
        # Статистические тесты
        if 'statistical_tests' in self.results:
            tests = self.results['statistical_tests']
            report.append("\n=== СТАТИСТИЧЕСКИЕ ТЕСТЫ ===")
            
            # ADF тест
            adf = tests['adf']
            report.append(f"Тест Дики-Фуллера (ADF):")
            report.append(f"  Статистика: {adf['statistic']:.4f}")
            report.append(f"  p-value: {adf['p_value']:.4f}")
            report.append(f"  Стационарен: {'Да' if adf['is_stationary'] else 'Нет'}")
            
            # KPSS тест
            kpss = tests['kpss']
            report.append(f"\nТест KPSS:")
            report.append(f"  Статистика: {kpss['statistic']:.4f}")
            report.append(f"  p-value: {kpss['p_value']:.4f}")
            report.append(f"  Стационарен: {'Да' if kpss['is_stationary'] else 'Нет'}")
            
            # Общий вывод
            report.append(f"\nОбщий вывод: {'Ряд стационарен' if tests['overall_stationary'] else 'Ряд нестационарен'}")
        
        # Дифференцирование
        if 'differencing' in self.results:
            diff_results = self.results['differencing']
            report.append("\n=== ДИФФЕРЕНЦИРОВАНИЕ ===")
            
            if 'first_difference' in diff_results:
                first_diff = diff_results['first_difference']
                report.append(f"Первое дифференцирование:")
                report.append(f"  Стационарен: {'Да' if first_diff['stationary'] else 'Нет'}")
            
            if 'second_difference' in diff_results:
                second_diff = diff_results['second_difference']
                report.append(f"Второе дифференцирование:")
                report.append(f"  Стационарен: {'Да' if second_diff['stationary'] else 'Нет'}")
        
        # Рекомендации
        report.append("\n=== РЕКОМЕНДАЦИИ ===")
        if self.results.get('statistical_tests', {}).get('overall_stationary', False):
            report.append("• Ряд стационарен, можно применять ARIMA модели")
        else:
            report.append("• Ряд нестационарен, рекомендуется:")
            report.append("  - Применить дифференцирование")
            report.append("  - Использовать ARIMA(p,d,q) модели")
            report.append("  - Рассмотреть другие методы стационаризации")
        
        return "\n".join(report)

def main():
    """Пример использования модуля анализа стационарности"""
    # Загружаем данные
    try:
        df = pd.read_csv('processed_real_estate_data.csv')
        df['saledate'] = pd.to_datetime(df['saledate'], utc=True)
    except FileNotFoundError:
        print("Файл processed_real_estate_data.csv не найден. Загружаем исходные данные...")
        df = pd.read_csv('ma_lga_12345.csv')
        df['saledate'] = pd.to_datetime(df['saledate'], format='%d/%m/%Y')
    
    print("Загружены данные для анализа стационарности:")
    print(f"Размер: {df.shape}")
    
    # Создаем анализатор
    analyzer = StationarityAnalyzer()
    
    # Выполняем анализ
    results = analyzer.analyze_stationarity(df)
    
    # Генерируем отчет
    report = analyzer.generate_stationarity_report()
    print("\n" + report)
    
    # Создаем графики
    plots = analyzer.create_stationarity_plots(df)
    print(f"\nСоздано {len(plots)} графиков для анализа стационарности")

if __name__ == "__main__":
    main()

