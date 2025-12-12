"""
Модуль приведения к стационарности и преобразований временных рядов.

Реализует:
- Лог-трансформацию (если данные > 0)
- Преобразование Бокса–Кокса с подбором оптимального λ
- Дифференцирование (1-го порядка, сезонное, комбинированное)
- Проверку стационарности после каждого преобразования (ADF/KPSS)
- Выбор оптимальной цепочки преобразований
- Обратное преобразование для оценки в исходных единицах
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from scipy import stats
from scipy.special import boxcox, inv_boxcox
from scipy.stats import boxcox_normmax
import warnings
import logging
import json
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





class StationarityTransformation:
    """Класс для приведения временных рядов к стационарности."""
    
    def __init__(self):
        self.transformations = {}
        self.stationarity_tests = {}
        self.best_transformation = None
        
    def load_data(self, file_path: str, target_column: str = 'Weekly_Sales') -> pd.DataFrame:
        """Загрузка данных временного ряда."""
        try:
            logger.info(f"Загрузка данных из {file_path}")
            df = pd.read_csv(file_path)
            
            # Предполагаем, что есть колонка с датами
            date_columns = ['Date', 'date', 'DATE', 'Date_Time', 'datetime', 'timestamp']
            date_col = None
            
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                logger.info(f"Установлен индекс даты: {date_col}")
            else:
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
                logger.warning("Колонка с датой не найдена, создан искусственный индекс")
            
            if target_column not in df.columns:
                available_cols = df.columns.tolist()
                logger.error(f"Колонка '{target_column}' не найдена. Доступные колонки: {available_cols}")
                raise ValueError(f"Колонка '{target_column}' не найдена. Доступные колонки: {available_cols}")
            
            # Удаляем строки с пропущенными значениями в целевой переменной
            initial_length = len(df)
            df = df.dropna(subset=[target_column])
            final_length = len(df)
            
            if initial_length != final_length:
                logger.warning(f"Удалено {initial_length - final_length} строк с пропущенными значениями в целевой переменной")
            
            logger.info(f"Загружено {len(df)} наблюдений")
            logger.info(f"Период: {df.index.min()} - {df.index.max()}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Файл не найден: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.error("Файл пуст или содержит только заголовки")
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return None
    
    def test_normality(self, series: pd.Series) -> dict:
        """Тест нормальности распределения (Шапиро-Уилка)."""
        logger.info("Тестирование нормальности распределения...")
        
        try:
            clean_series = series.dropna()
            
            if len(clean_series) == 0:
                logger.error('Недостаточно данных для тестирования нормальности')
                return {'error': 'Недостаточно данных'}
            
            # Для больших выборок используем случайную подвыборку (максимум 5000 точек)
            if len(clean_series) > 5000:
                np.random.seed(42)  # Для воспроизводимости
                sample_size = min(5000, len(clean_series))
                sample_indices = np.random.choice(len(clean_series), size=sample_size, replace=False)
                sample_series = clean_series.iloc[sample_indices]
                logger.info(f"Для теста Шапиро-Уилка использована случайная выборка из {sample_size} точек (из {len(clean_series)})")
            else:
                sample_series = clean_series
            
            shapiro_stat, shapiro_p = stats.shapiro(sample_series)
            
            result = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_pvalue': shapiro_p,
                'is_normal': shapiro_p > 0.05,
                'sample_size': len(sample_series),
                'original_size': len(clean_series)
            }
            
            logger.info(f"Тест нормальности Шапиро-Уилка: статистика={shapiro_stat:.4f}, p-value={shapiro_p:.4f}, выборка={len(sample_series)}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка в тесте нормальности: {e}")
            return {'error': str(e)}
    
    def test_stationarity(self, series: pd.Series, test_name: str = '') -> dict:
        """Тестирование стационарности временного ряда."""
        logger.info(f"Тестирование стационарности: {test_name}")
        
        try:
            results = {}
            
            # Удаляем NaN значения
            clean_series = series.dropna()
            
            if len(clean_series) == 0:
                logger.error('Недостаточно данных для тестирования стационарности')
                return {'error': 'Недостаточно данных для тестирования'}
            
            if len(clean_series) < 10:
                logger.warning(f'Мало данных для тестирования стационарности: {len(clean_series)}')
            
            # ADF тест
            try:
                adf_result = adfuller(clean_series, autolag='AIC')
                results['adf'] = {
                    'statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05,
                    'used_lag': adf_result[2]
                }
                logger.info(f"ADF тест: статистика={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}")
            except Exception as e:
                logger.error(f"Ошибка ADF теста: {e}")
                results['adf'] = {'error': str(e)}
            
            # KPSS тест
            try:
                kpss_result = kpss(clean_series, regression='c', nlags='auto')
                results['kpss'] = {
                    'statistic': kpss_result[0],
                    'p_value': kpss_result[1],
                    'critical_values': kpss_result[3],
                    'is_stationary': kpss_result[1] > 0.05,
                    'used_lag': kpss_result[2]
                }
                logger.info(f"KPSS тест: статистика={kpss_result[0]:.4f}, p-value={kpss_result[1]:.4f}")
            except Exception as e:
                logger.error(f"Ошибка KPSS теста: {e}")
                results['kpss'] = {'error': str(e)}
            
            # Общий вывод о стационарности
            adf_stationary = results.get('adf', {}).get('is_stationary', False)
            kpss_stationary = results.get('kpss', {}).get('is_stationary', False)
            
            results['overall_stationary'] = adf_stationary and kpss_stationary
            results['test_name'] = test_name
            
            logger.info(f"Общий результат стационарности: {results['overall_stationary']}")
            
            # Добавляем тест нормальности
            normality_result = self.test_normality(clean_series)
            results['normality'] = normality_result
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в тестировании стационарности: {e}")
            return {'error': str(e)}
    
    def log_transformation(self, series: pd.Series) -> tuple:
        """Лог-трансформация временного ряда."""
        logger.info("Применение лог-трансформации...")
        
        try:
            # Проверяем, что все значения положительные
            if (series <= 0).any():
                logger.warning("Есть неположительные значения, добавляем константу")
                min_val = series.min()
                if min_val <= 0:
                    series_shifted = series - min_val + 1
                else:
                    series_shifted = series
            else:
                series_shifted = series
            
            # Лог-трансформация
            log_series = np.log(series_shifted)
            
            # Сохраняем параметры для обратного преобразования
            transformation_params = {
                'type': 'log',
                'shift': series_shifted.min() - series.min() if (series <= 0).any() else 0,
                'original_min': series.min()
            }
            
            logger.info("Лог-трансформация применена успешно")
            return log_series, transformation_params
            
        except Exception as e:
            logger.error(f"Ошибка в лог-трансформации: {e}")
            return series, {'type': 'log', 'error': str(e)}
    
    def box_cox_transformation(self, series: pd.Series) -> tuple:
        """Преобразование Бокса–Кокса."""
        logger.info("Применение преобразования Бокса–Кокса...")
        
        try:
            # Проверяем, что все значения положительные
            if (series <= 0).any():
                logger.warning("Есть неположительные значения, добавляем константу")
                min_val = series.min()
                if min_val <= 0:
                    series_shifted = series - min_val + 1
                else:
                    series_shifted = series
            else:
                series_shifted = series
            
            # Подбор оптимального λ
            try:
                lambda_optimal = boxcox_normmax(series_shifted, method='mle')
                logger.info(f"Оптимальный λ: {lambda_optimal:.4f}")
            except Exception as e:
                logger.warning(f"Ошибка подбора λ: {e}, используем λ=0 (логарифм)")
                lambda_optimal = 0
            
            # Применение преобразования Бокса–Кокса
            try:
                boxcox_series, lambda_used = boxcox(series_shifted, lambda_optimal)
                logger.info(f"Использованный λ: {lambda_used:.4f}")
            except Exception as e:
                logger.warning(f"Ошибка преобразования Бокса–Кокса: {e}, используем лог-трансформацию")
                # Fallback к лог-трансформации
                boxcox_series = np.log(series_shifted)
                lambda_used = 0
            
            # Сохранение параметров для обратного преобразования
            transformation_params = {
                'type': 'box_cox',
                'lambda': lambda_used,
                'shift': series_shifted.min() - series.min() if (series <= 0).any() else 0,
                'original_min': series.min()
            }
            
            logger.info("Преобразование Бокса–Кокса применено успешно")
            return boxcox_series, transformation_params
            
        except Exception as e:
            logger.error(f"Ошибка в преобразовании Бокса–Кокса: {e}")
            return series, {'type': 'box_cox', 'error': str(e)}
    
    def first_difference(self, series: pd.Series) -> tuple:
        """Дифференцирование первого порядка."""
        logger.info("Применение дифференцирования первого порядка...")
        
        try:
            diff_series = series.diff().dropna()
            
            transformation_params = {
                'type': 'first_difference',
                'original_first_value': series.iloc[0]
            }
            
            logger.info("Дифференцирование первого порядка применено успешно")
            return diff_series, transformation_params
            
        except Exception as e:
            logger.error(f"Ошибка в дифференцировании первого порядка: {e}")
            return series, {'type': 'first_difference', 'error': str(e)}
    
    def seasonal_difference(self, series: pd.Series, period: int = 7) -> tuple:
        """Сезонное дифференцирование."""
        logger.info(f"Применение сезонного дифференцирования (период: {period})...")
        
        try:
            if len(series) < period + 1:
                logger.error(f"Недостаточно данных для сезонного дифференцирования. Требуется минимум {period + 1}, доступно {len(series)}")
                return series, {'type': 'seasonal_difference', 'error': 'Недостаточно данных'}
            
            seasonal_diff_series = series.diff(period).dropna()
            
            transformation_params = {
                'type': 'seasonal_difference',
                'period': period,
                'original_first_values': series.iloc[:period].tolist()
            }
            
            logger.info("Сезонное дифференцирование применено успешно")
            return seasonal_diff_series, transformation_params
            
        except Exception as e:
            logger.error(f"Ошибка в сезонном дифференцировании: {e}")
            return series, {'type': 'seasonal_difference', 'error': str(e)}
    
    def combined_difference(self, series: pd.Series, period: int = 7) -> tuple:
        """Комбинированное дифференцирование."""
        logger.info(f"Применение комбинированного дифференцирования (период: {period})...")
        
        try:
            if len(series) < period + 2:
                logger.error(f"Недостаточно данных для комбинированного дифференцирования. Требуется минимум {period + 2}, доступно {len(series)}")
                return series, {'type': 'combined_difference', 'error': 'Недостаточно данных'}
            
            # Сначала сезонное, затем первое
            seasonal_diff = series.diff(period)
            combined_diff = seasonal_diff.diff().dropna()
            
            transformation_params = {
                'type': 'combined_difference',
                'period': period,
                'original_first_values': series.iloc[:period+1].tolist()
            }
            
            logger.info("Комбинированное дифференцирование применено успешно")
            return combined_diff, transformation_params
            
        except Exception as e:
            logger.error(f"Ошибка в комбинированном дифференцировании: {e}")
            return series, {'type': 'combined_difference', 'error': str(e)}
    
    def inverse_transform(self, transformed_series: pd.Series, 
                         transformation_params: dict) -> pd.Series:
        """Обратное преобразование."""
        logger.info(f"Обратное преобразование: {transformation_params['type']}")
        
        try:
            if 'error' in transformation_params:
                logger.error(f"Ошибка в параметрах преобразования: {transformation_params['error']}")
                return transformed_series
            
            if transformation_params['type'] == 'log':
                # Обратная лог-трансформация
                original_series = np.exp(transformed_series)
                if transformation_params['shift'] > 0:
                    original_series = original_series - transformation_params['shift']
                
            elif transformation_params['type'] == 'box_cox':
                # Обратное преобразование Бокса–Кокса
                original_series = inv_boxcox(transformed_series, transformation_params['lambda'])
                if transformation_params['shift'] > 0:
                    original_series = original_series - transformation_params['shift']
                    
            elif transformation_params['type'] == 'first_difference':
                # Обратное дифференцирование первого порядка
                original_series = transformed_series.cumsum() + transformation_params['original_first_value']
                
            elif transformation_params['type'] == 'seasonal_difference':
                # Обратное сезонное дифференцирование
                period = transformation_params['period']
                original_values = transformation_params['original_first_values']
                
                # Восстанавливаем исходный ряд
                restored_series = pd.Series(index=transformed_series.index, dtype=float)
                
                # Заполняем первые значения
                for i in range(min(period, len(original_values))):
                    if i < len(restored_series):
                        restored_series.iloc[i] = original_values[i]
                
                # Восстанавливаем остальные значения
                for i in range(period, len(restored_series)):
                    restored_series.iloc[i] = restored_series.iloc[i-period] + transformed_series.iloc[i]
                
                original_series = restored_series
                
            elif transformation_params['type'] == 'combined_difference':
                # Обратное комбинированное дифференцирование
                period = transformation_params['period']
                original_values = transformation_params['original_first_values']
                
                # Сначала восстанавливаем сезонное дифференцирование
                seasonal_restored = transformed_series.cumsum()
                
                # Затем восстанавливаем исходный ряд
                restored_series = pd.Series(index=seasonal_restored.index, dtype=float)
                
                # Заполняем первые значения
                for i in range(min(period+1, len(original_values))):
                    if i < len(restored_series):
                        restored_series.iloc[i] = original_values[i]
                
                # Восстанавливаем остальные значения
                for i in range(period+1, len(restored_series)):
                    restored_series.iloc[i] = restored_series.iloc[i-period] + seasonal_restored.iloc[i]
                
                original_series = restored_series
            
            else:
                logger.warning(f"Неизвестный тип преобразования: {transformation_params['type']}")
                return transformed_series
            
            logger.info("Обратное преобразование выполнено успешно")
            return original_series
            
        except Exception as e:
            logger.error(f"Ошибка в обратном преобразовании: {e}")
            return transformed_series
    
    def test_all_transformations(self, series: pd.Series) -> dict:
        """Тестирование всех возможных преобразований."""
        logger.info("=== ТЕСТИРОВАНИЕ ВСЕХ ПРЕОБРАЗОВАНИЙ ===")
        
        try:
            results = {}
            
            # Исходный ряд
            logger.info("1. Исходный ряд")
            original_stationarity = self.test_stationarity(series, 'Original')
            results['original'] = {
                'series': series,
                'stationarity': original_stationarity,
                'transformation_params': None
            }
            
            # Лог-трансформация
            if (series > 0).all():
                logger.info("2. Лог-трансформация")
                log_series, log_params = self.log_transformation(series)
                log_stationarity = self.test_stationarity(log_series, 'Log')
                results['log'] = {
                    'series': log_series,
                    'stationarity': log_stationarity,
                    'transformation_params': log_params
                }
            else:
                logger.warning("Лог-трансформация пропущена (есть неположительные значения)")
            
            # Преобразование Бокса–Кокса
            if (series > 0).all():
                logger.info("3. Преобразование Бокса–Кокса")
                boxcox_series, boxcox_params = self.box_cox_transformation(series)
                boxcox_stationarity = self.test_stationarity(boxcox_series, 'Box-Cox')
                results['box_cox'] = {
                    'series': boxcox_series,
                    'stationarity': boxcox_stationarity,
                    'transformation_params': boxcox_params
                }
            else:
                logger.warning("Преобразование Бокса–Кокса пропущено (есть неположительные значения)")
            
            # Дифференцирование первого порядка
            logger.info("4. Дифференцирование первого порядка")
            diff_series, diff_params = self.first_difference(series)
            diff_stationarity = self.test_stationarity(diff_series, 'First Difference')
            results['first_difference'] = {
                'series': diff_series,
                'stationarity': diff_stationarity,
                'transformation_params': diff_params
            }
            
            # Сезонное дифференцирование
            logger.info("5. Сезонное дифференцирование")
            seasonal_diff_series, seasonal_diff_params = self.seasonal_difference(series)
            seasonal_diff_stationarity = self.test_stationarity(seasonal_diff_series, 'Seasonal Difference')
            results['seasonal_difference'] = {
                'series': seasonal_diff_series,
                'stationarity': seasonal_diff_stationarity,
                'transformation_params': seasonal_diff_params
            }
            
            # Комбинированное дифференцирование
            logger.info("6. Комбинированное дифференцирование")
            combined_diff_series, combined_diff_params = self.combined_difference(series)
            combined_diff_stationarity = self.test_stationarity(combined_diff_series, 'Combined Difference')
            results['combined_difference'] = {
                'series': combined_diff_series,
                'stationarity': combined_diff_stationarity,
                'transformation_params': combined_diff_params
            }
            
            logger.info(f"Протестировано {len(results)} преобразований")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в тестировании всех преобразований: {e}")
            return {}
    
    def select_best_transformation(self, transformation_results: dict) -> dict:
        """Выбор лучшего преобразования."""
        logger.info("Выбор лучшего преобразования...")
        
        try:
            best_transformation = None
            best_score = float('inf')
            
            scores = {}
            
            for transform_name, result in transformation_results.items():
                stationarity = result['stationarity']
                
                if 'error' in stationarity:
                    logger.warning(f"Преобразование {transform_name} содержит ошибки, пропускаем")
                    continue
                
                # Критерии оценки:
                # 1. Стационарность (приоритет)
                # 2. P-value ADF теста (чем меньше, тем лучше)
                # 3. P-value KPSS теста (чем больше, тем лучше)
                
                is_stationary = stationarity.get('overall_stationary', False)
                adf_pvalue = stationarity.get('adf', {}).get('p_value', 1.0)
                kpss_pvalue = stationarity.get('kpss', {}).get('p_value', 0.0)
                
                # Составной скор (чем меньше, тем лучше)
                score = adf_pvalue + (1 - kpss_pvalue)
                
                # Бонус за стационарность
                if is_stationary:
                    score *= 0.1
                    logger.info(f"Преобразование {transform_name}: бонус за стационарность")
                
                scores[transform_name] = {
                    'score': score,
                    'is_stationary': is_stationary,
                    'adf_pvalue': adf_pvalue,
                    'kpss_pvalue': kpss_pvalue
                }
                
                if score < best_score:
                    best_score = score
                    best_transformation = transform_name
            
            logger.info(f"Лучшее преобразование: {best_transformation} со скором {best_score:.6f}")
            return {
                'best_transformation': best_transformation,
                'best_score': best_score,
                'all_scores': scores
            }
            
        except Exception as e:
            logger.error(f"Ошибка в выборе лучшего преобразования: {e}")
            return {
                'best_transformation': None,
                'best_score': float('inf'),
                'all_scores': {}
            }
    
    def visualize_transformations(self, transformation_results: dict, 
                                original_series: pd.Series) -> None:
        """Визуализация всех преобразований."""
        logger.info("Создание визуализации преобразований...")
        
        try:
            n_transforms = len(transformation_results)
            if n_transforms == 0:
                logger.warning("Нет данных для визуализации")
                return
            
            fig, axes = plt.subplots(n_transforms, 2, figsize=(15, 4*n_transforms))
            
            if n_transforms == 1:
                axes = axes.reshape(1, -1)
            
            for i, (transform_name, result) in enumerate(transformation_results.items()):
                series = result['series']
                stationarity = result['stationarity']
                
                # Функция для подвыборки данных для визуализации
                def downsample_for_plot(series, max_points=5000):
                    """Подвыборка данных для визуализации."""
                    clean_series = series.dropna()
                    if len(clean_series) <= max_points:
                        return clean_series
                    step = len(clean_series) // max_points
                    indices = range(0, len(clean_series), step)
                    return clean_series.iloc[indices]
                
                plot_series = downsample_for_plot(series)
                
                # График временного ряда
                axes[i, 0].plot(plot_series.index, plot_series.values, alpha=0.7, linewidth=0.8)
                axes[i, 0].set_title(f'{transform_name.replace("_", " ").title()}')
                axes[i, 0].set_xlabel('Время')
                axes[i, 0].set_ylabel('Значение')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Гистограмма
                hist_data = series.dropna()
                if len(hist_data) > 10000:
                    # Для больших данных используем подвыборку для гистограммы
                    np.random.seed(42)
                    hist_data = hist_data.sample(n=10000)
                axes[i, 1].hist(hist_data, bins=50, alpha=0.7, edgecolor='black')
                axes[i, 1].set_title(f'Распределение: {transform_name.replace("_", " ").title()}')
                axes[i, 1].set_xlabel('Значение')
                axes[i, 1].set_ylabel('Частота')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Добавляем информацию о стационарности и нормальности
                if 'error' not in stationarity:
                    is_stationary = stationarity.get('overall_stationary', False)
                    adf_pvalue = stationarity.get('adf', {}).get('p_value', 1.0)
                    kpss_pvalue = stationarity.get('kpss', {}).get('p_value', 0.0)
                    normality = stationarity.get('normality', {})
                    
                    info_text = f'Стационарный: {is_stationary}\nADF p-value: {adf_pvalue:.4f}\nKPSS p-value: {kpss_pvalue:.4f}'
                    
                    if 'error' not in normality:
                        is_normal = normality.get('is_normal', False)
                        shapiro_p = normality.get('shapiro_pvalue', 1.0)
                        info_text += f'\nНормальный: {is_normal}\nShapiro p-value: {shapiro_p:.4f}'
                    
                    axes[i, 0].text(0.02, 0.98, 
                                   info_text,
                                   transform=axes[i, 0].transAxes, 
                                   verticalalignment='top',
                                   fontsize=8,
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                else:
                    axes[i, 0].text(0.02, 0.98, 
                                   f'Ошибка в данных',
                                   transform=axes[i, 0].transAxes, 
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('stationarity_transformations.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Визуализация сохранена в stationarity_transformations.png")
            
        except Exception as e:
            logger.error(f"Ошибка в визуализации преобразований: {e}")
    
    def create_summary_table(self, transformation_results: dict, 
                           best_selection: dict) -> pd.DataFrame:
        """Создание сводной таблицы результатов."""
        logger.info("Создание сводной таблицы результатов...")
        
        try:
            summary_data = []
            
            for transform_name, result in transformation_results.items():
                stationarity = result['stationarity']
                
                if 'error' in stationarity:
                    summary_data.append({
                        'Преобразование': transform_name.replace('_', ' ').title(),
                        'Стационарный': 'Ошибка',
                        'ADF p-value': 'Ошибка',
                        'KPSS p-value': 'Ошибка',
                        'ADF статистика': 'Ошибка',
                        'KPSS статистика': 'Ошибка',
                        'Нормальность': 'Ошибка',
                        'Shapiro p-value': None,
                        'Лучший': False
                    })
                    continue
                
                normality = stationarity.get('normality', {})
                normality_info = 'Ошибка' if 'error' in normality else (
                    f"Да (p={normality.get('shapiro_pvalue', 0):.4f})" if normality.get('is_normal', False) 
                    else f"Нет (p={normality.get('shapiro_pvalue', 0):.4f})"
                )
                
                summary_data.append({
                    'Преобразование': transform_name.replace('_', ' ').title(),
                    'Стационарный': stationarity.get('overall_stationary', False),
                    'ADF p-value': stationarity.get('adf', {}).get('p_value', 1.0),
                    'KPSS p-value': stationarity.get('kpss', {}).get('p_value', 0.0),
                    'ADF статистика': stationarity.get('adf', {}).get('statistic', 0),
                    'KPSS статистика': stationarity.get('kpss', {}).get('statistic', 0),
                    'Нормальность': normality_info,
                    'Shapiro p-value': normality.get('shapiro_pvalue', None) if 'error' not in normality else None,
                    'Лучший': transform_name == best_selection.get('best_transformation', '')
                })
            
            df = pd.DataFrame(summary_data)
            logger.info(f"Создана сводная таблица с {len(df)} строками")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка в создании сводной таблицы: {e}")
            return pd.DataFrame()
    
    def run_comprehensive_stationarity_analysis(self, file_path: str, 
                                               target_column: str = 'Weekly_Sales') -> dict:
        """Запуск комплексного анализа стационарности."""
        logger.info("=== КОМПЛЕКСНЫЙ АНАЛИЗ СТАЦИОНАРНОСТИ ===")
        
        try:
            # Загрузка данных
            df = self.load_data(file_path, target_column)
            if df is None:
                logger.error("Не удалось загрузить данные")
                return {}
            
            series = df[target_column]
            logger.info(f"Загружен ряд длиной {len(series)}")
            
            # Тестирование всех преобразований
            transformation_results = self.test_all_transformations(series)
            
            if not transformation_results:
                logger.error("Не удалось выполнить преобразования")
                return {}
            
            # Выбор лучшего преобразования
            best_selection = self.select_best_transformation(transformation_results)
            
            logger.info(f"Лучшее преобразование: {best_selection.get('best_transformation', 'Не определено')}")
            logger.info(f"Скор: {best_selection.get('best_score', float('inf')):.6f}")
            
            # Визуализация
            self.visualize_transformations(transformation_results, series)
            
            # Создание сводной таблицы
            summary_df = self.create_summary_table(transformation_results, best_selection)
            
            if not summary_df.empty:
                logger.info("=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
                logger.info(f"\n{summary_df.to_string(index=False)}")
                
                # Сохранение результатов
                summary_df.to_csv('stationarity_analysis_results.csv', index=False, encoding='utf-8')
                logger.info("Сводная таблица сохранена в stationarity_analysis_results.csv")
            
            final_results = {
                'transformation_results': transformation_results,
                'best_selection': best_selection,
                'summary': summary_df.to_dict('records') if not summary_df.empty else [],
                'series_info': {
                    'length': len(series),
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max())
                }
            }
            
            with open('stationarity_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("Результаты сохранены:")
            logger.info("- stationarity_analysis_results.csv")
            logger.info("- stationarity_analysis.json")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Ошибка в комплексном анализе стационарности: {e}")
            return {}


def main():
    """Основная функция для демонстрации работы модуля."""
    logger.info("Запуск анализа стационарности...")
    
    try:
        stationarity_analyzer = StationarityTransformation()
        
        # Запуск анализа
        results = stationarity_analyzer.run_comprehensive_stationarity_analysis(
            "../New_final.csv", 
            "Weekly_Sales"
        )
        
        if results:
            logger.info("=== АНАЛИЗ СТАЦИОНАРНОСТИ ЗАВЕРШЕН ===")
            logger.info(f"Лучшее преобразование: {results['best_selection']['best_transformation']}")
        else:
            logger.error("Анализ не удался. Проверьте путь к файлу.")
            
    except Exception as e:
        logger.error(f"Ошибка в main: {e}")


if __name__ == "__main__":
    main()

