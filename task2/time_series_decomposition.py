"""
Модуль декомпозиции временных рядов.

Реализует:
- Аддитивную и мультипликативную декомпозицию
- Подбор оптимального периода сезонности
- Анализ остатков (стационарность, ACF/PACF, нормальность)
- Выбор лучшей модели декомпозиции
- Визуализацию всех компонент
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings
import logging
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class TimeSeriesDecomposition:
    """Класс для декомпозиции временных рядов."""
    
    def __init__(self):
        self.decomposition_results = {}
        self.residual_analysis = {}
        
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
                logger.info(f"Установлен индекс по колонке {date_col}")
            else:
                # Если нет колонки с датами, создаем индекс
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
                logger.warning("Колонка с датами не найдена, создан индекс по умолчанию")
            
            # Проверяем наличие целевой переменной
            if target_column not in df.columns:
                available_cols = df.columns.tolist()
                logger.error(f"Колонка '{target_column}' не найдена. Доступные колонки: {available_cols}")
                raise ValueError(f"Колонка '{target_column}' не найдена. Доступные колонки: {available_cols}")
            
            # Удаляем пропущенные значения
            initial_len = len(df)
            df = df.dropna(subset=[target_column])
            final_len = len(df)
            
            if initial_len != final_len:
                logger.warning(f"Удалено {initial_len - final_len} строк с пропущенными значениями")
            
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
    
    def seasonal_decomposition(self, series: pd.Series, period: int, 
                              model: str = 'additive') -> dict:
        """Выполнение сезонной декомпозиции."""
        try:
            logger.info(f"Выполнение декомпозиции: период={period}, модель={model}")
            
            # Проверяем достаточность данных
            if len(series) < 2 * period:
                logger.warning(f"Недостаточно данных для периода {period}. Требуется минимум {2 * period} наблюдений")
                return None
            
            # Проверяем наличие пропущенных значений
            if series.isnull().any():
                logger.warning("Обнаружены пропущенные значения, они будут интерполированы")
                series = series.interpolate(method='linear')
            
            # Выполняем декомпозицию
            decomposition = seasonal_decompose(
                series, 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            
            result = {
                'original': series,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period,
                'model': model,
                'decomposition': decomposition
            }
            
            logger.info("Декомпозиция выполнена успешно")
            return result
            
        except ValueError as e:
            logger.error(f"Ошибка параметров декомпозиции для периода {period}: {e}")
            return None
        except Exception as e:
            logger.error(f"Ошибка декомпозиции для периода {period}: {e}")
            return None
    
    def test_stationarity(self, series: pd.Series, test_name: str = '') -> dict:
        """Тестирование стационарности временного ряда."""
        results = {}
        
        logger.info(f"Тестирование стационарности: {test_name}")
        
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
        
        return results
    
    def analyze_residuals(self, residuals: pd.Series) -> dict:
        """Анализ остатков декомпозиции."""
        logger.info("Анализ остатков декомпозиции")
        
        # Удаляем NaN значения
        clean_residuals = residuals.dropna()
        
        if len(clean_residuals) == 0:
            logger.error('Недостаточно данных для анализа остатков')
            return {'error': 'Недостаточно данных для анализа остатков'}
        
        analysis = {}
        
        # Статистики остатков
        analysis['statistics'] = {
            'mean': clean_residuals.mean(),
            'std': clean_residuals.std(),
            'skewness': stats.skew(clean_residuals),
            'kurtosis': stats.kurtosis(clean_residuals),
            'jarque_bera_stat': stats.jarque_bera(clean_residuals)[0],
            'jarque_bera_pvalue': stats.jarque_bera(clean_residuals)[1],
            'min': clean_residuals.min(),
            'max': clean_residuals.max(),
            'median': clean_residuals.median()
        }
        
        logger.info(f"Статистики остатков: mean={analysis['statistics']['mean']:.4f}, std={analysis['statistics']['std']:.4f}")
        
        # Тест нормальности Шапиро-Уилка
        # Для больших выборок используем случайную подвыборку (максимум 5000 точек)
        try:
            if len(clean_residuals) > 5000:
                # Для больших выборок используем случайную подвыборку
                np.random.seed(42)  # Для воспроизводимости
                sample_size = min(5000, len(clean_residuals))
                sample_indices = np.random.choice(len(clean_residuals), size=sample_size, replace=False)
                sample_residuals = clean_residuals.iloc[sample_indices]
                logger.info(f"Для теста Шапиро-Уилка использована случайная выборка из {sample_size} точек (из {len(clean_residuals)})")
            else:
                sample_residuals = clean_residuals
            
            shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
            analysis['normality_tests'] = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_pvalue': shapiro_p,
                'is_normal': shapiro_p > 0.05,
                'sample_size': len(sample_residuals),
                'original_size': len(clean_residuals)
            }
            logger.info(f"Тест нормальности Шапиро-Уилка: статистика={shapiro_stat:.4f}, p-value={shapiro_p:.4f}, выборка={len(sample_residuals)}")
        except Exception as e:
            logger.error(f"Ошибка теста нормальности: {e}")
            analysis['normality_tests'] = {'error': str(e)}
        
        # Тестирование стационарности остатков
        analysis['stationarity'] = self.test_stationarity(clean_residuals, 'Residuals')
        
        return analysis
    
    def find_optimal_period(self, series: pd.Series, 
                           periods_to_test: list = [7, 30, 365]) -> dict:
        """Поиск оптимального периода сезонности."""
        logger.info(f"Поиск оптимального периода сезонности для периодов: {periods_to_test}")
        results = {}
        
        for period in periods_to_test:
            if len(series) < 2 * period:
                logger.warning(f"Пропускаем период {period}: недостаточно данных (требуется минимум {2 * period})")
                continue
            
            logger.info(f"Тестирование периода {period}")
            
            # Тестируем аддитивную модель
            additive_result = self.seasonal_decomposition(series, period, 'additive')
            if additive_result:
                additive_residuals = additive_result['residual']
                additive_analysis = self.analyze_residuals(additive_residuals)
                
                results[f'additive_{period}'] = {
                    'decomposition': additive_result,
                    'residual_analysis': additive_analysis,
                    'residual_variance': additive_residuals.var(),
                    'residual_std': additive_residuals.std()
                }
                logger.info(f"Аддитивная модель для периода {period}: дисперсия остатков={additive_residuals.var():.4f}")
            
            # Тестируем мультипликативную модель (только если все значения > 0)
            if (series > 0).all():
                multiplicative_result = self.seasonal_decomposition(series, period, 'multiplicative')
                if multiplicative_result:
                    multiplicative_residuals = multiplicative_result['residual']
                    multiplicative_analysis = self.analyze_residuals(multiplicative_residuals)
                    
                    results[f'multiplicative_{period}'] = {
                        'decomposition': multiplicative_result,
                        'residual_analysis': multiplicative_analysis,
                        'residual_variance': multiplicative_residuals.var(),
                        'residual_std': multiplicative_residuals.std()
                    }
                    logger.info(f"Мультипликативная модель для периода {period}: дисперсия остатков={multiplicative_residuals.var():.4f}")
            else:
                logger.warning(f"Мультипликативная модель для периода {period} пропущена (есть неположительные значения)")
        
        logger.info(f"Протестировано {len(results)} комбинаций период/модель")
        return results
    
    def select_best_model(self, decomposition_results: dict) -> dict:
        """Выбор лучшей модели декомпозиции на основе анализа остатков."""
        logger.info("Выбор лучшей модели декомпозиции")
        
        best_model = None
        best_score = float('inf')
        
        scores = {}
        
        for model_name, result in decomposition_results.items():
            residual_analysis = result.get('residual_analysis', {})
            
            # Критерии оценки качества:
            # 1. Минимальная дисперсия остатков
            # 2. Стационарность остатков
            # 3. Нормальность остатков
            
            residual_variance = result.get('residual_variance', float('inf'))
            is_stationary = residual_analysis.get('stationarity', {}).get('overall_stationary', False)
            is_normal = residual_analysis.get('normality_tests', {}).get('is_normal', False)
            
            # Составной скор (чем меньше, тем лучше)
            score = residual_variance
            
            # Бонус за стационарность
            if is_stationary:
                score *= 0.8
                logger.info(f"Модель {model_name}: бонус за стационарность")
            
            # Бонус за нормальность
            if is_normal:
                score *= 0.9
                logger.info(f"Модель {model_name}: бонус за нормальность")
            
            scores[model_name] = {
                'score': score,
                'residual_variance': residual_variance,
                'is_stationary': is_stationary,
                'is_normal': is_normal
            }
            
            logger.info(f"Модель {model_name}: скор={score:.6f}, дисперсия={residual_variance:.6f}")
            
            if score < best_score:
                best_score = score
                best_model = model_name
        
        logger.info(f"Лучшая модель: {best_model} со скором {best_score:.6f}")
        
        return {
            'best_model': best_model,
            'best_score': best_score,
            'all_scores': scores
        }
    
    def visualize_decomposition(self, decomposition_result: dict, 
                               title: str = 'Декомпозиция временного ряда') -> None:
        """Визуализация компонент декомпозиции."""
        logger.info("Создание визуализации декомпозиции")
        
        try:
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            fig.suptitle(title, fontsize=16)
            
            # Функция для подвыборки данных для визуализации
            def downsample_data(series, max_points=5000):
                """Подвыборка данных для визуализации."""
                if len(series) <= max_points:
                    return series
                # Используем равномерную подвыборку
                step = len(series) // max_points
                indices = range(0, len(series), step)
                return series.iloc[indices]
            
            # Исходный ряд
            original_data = downsample_data(decomposition_result['original'])
            axes[0].plot(original_data.index, original_data.values, color='blue', linewidth=0.8, alpha=0.7)
            axes[0].set_title('Исходный ряд')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylabel('Значение')
            
            # Тренд
            trend_data = downsample_data(decomposition_result['trend'].dropna())
            if len(trend_data) > 0:
                axes[1].plot(trend_data.index, trend_data.values, color='green', linewidth=0.8, alpha=0.7)
            axes[1].set_title('Тренд')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylabel('Тренд')
            
            # Сезонная компонента
            seasonal_data = downsample_data(decomposition_result['seasonal'].dropna())
            if len(seasonal_data) > 0:
                axes[2].plot(seasonal_data.index, seasonal_data.values, color='red', linewidth=0.8, alpha=0.7)
            axes[2].set_title('Сезонная компонента')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylabel('Сезонность')
            
            # Остатки
            residual_data = downsample_data(decomposition_result['residual'].dropna())
            if len(residual_data) > 0:
                axes[3].plot(residual_data.index, residual_data.values, color='orange', linewidth=0.8, alpha=0.7)
            axes[3].set_title('Остатки')
            axes[3].grid(True, alpha=0.3)
            axes[3].set_ylabel('Остатки')
            axes[3].set_xlabel('Время')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализации: {e}")
    
    def visualize_residual_analysis(self, residuals: pd.Series, 
                                  model_name: str = '') -> None:
        """Визуализация анализа остатков."""
        logger.info(f"Создание визуализации анализа остатков для {model_name}")
        
        try:
            clean_residuals = residuals.dropna()
            
            if len(clean_residuals) == 0:
                logger.warning("Нет данных для визуализации остатков")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'Анализ остатков - {model_name}', fontsize=16)
            
            # Гистограмма остатков
            axes[0, 0].hist(clean_residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Распределение остатков')
            axes[0, 0].set_xlabel('Значение остатка')
            axes[0, 0].set_ylabel('Частота')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            stats.probplot(clean_residuals, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (нормальность)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Остатки во времени (с подвыборкой для больших данных)
            def downsample_for_plot(series, max_points=5000):
                """Подвыборка данных для визуализации."""
                if len(series) <= max_points:
                    return series
                step = len(series) // max_points
                indices = range(0, len(series), step)
                return series.iloc[indices]
            
            residual_plot_data = downsample_for_plot(clean_residuals)
            axes[0, 2].plot(residual_plot_data.index, residual_plot_data.values, alpha=0.7, linewidth=0.8)
            axes[0, 2].set_title('Остатки во времени')
            axes[0, 2].set_xlabel('Время')
            axes[0, 2].set_ylabel('Значение остатка')
            axes[0, 2].grid(True, alpha=0.3)
            
            # ACF остатков
            plot_acf(clean_residuals, ax=axes[1, 0], lags=min(40, len(clean_residuals)//4), alpha=0.05)
            axes[1, 0].set_title('ACF остатков')
            
            # PACF остатков
            plot_pacf(clean_residuals, ax=axes[1, 1], lags=min(40, len(clean_residuals)//4), alpha=0.05)
            axes[1, 1].set_title('PACF остатков')
            
            # Scatter plot остатков vs прогнозов
            if len(clean_residuals) > 1:
                fitted_values = clean_residuals.shift(1).dropna()
                residuals_for_plot = clean_residuals[1:]
                
                if len(fitted_values) > 0 and len(residuals_for_plot) > 0:
                    axes[1, 2].scatter(fitted_values, residuals_for_plot, alpha=0.6)
                    axes[1, 2].set_title('Остатки vs Прогнозы')
                    axes[1, 2].set_xlabel('Прогнозы')
                    axes[1, 2].set_ylabel('Остатки')
                    axes[1, 2].grid(True, alpha=0.3)
                else:
                    axes[1, 2].text(0.5, 0.5, 'Недостаточно данных', ha='center', va='center', transform=axes[1, 2].transAxes)
                    axes[1, 2].set_title('Остатки vs Прогнозы')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализации остатков: {e}")
    
    def run_comprehensive_decomposition(self, file_path: str, 
                                       target_column: str = 'Weekly_Sales') -> dict:
        """Запуск комплексного анализа декомпозиции."""
        logger.info("=== КОМПЛЕКСНЫЙ АНАЛИЗ ДЕКОМПОЗИЦИИ ВРЕМЕННОГО РЯДА ===")
        
        try:
            # Загрузка данных
            df = self.load_data(file_path, target_column)
            if df is None:
                logger.error("Не удалось загрузить данные")
                return {}
            
            series = df[target_column]
            
            # Поиск оптимального периода
            logger.info("1. Поиск оптимального периода сезонности...")
            periods_to_test = [7, 30, 90, 365]  # Адаптируем под данные
            decomposition_results = self.find_optimal_period(series, periods_to_test)
            
            if not decomposition_results:
                logger.error("Не удалось выполнить декомпозицию ни для одного периода")
                return {}
            
            # Выбор лучшей модели
            logger.info("2. Выбор лучшей модели...")
            best_model_selection = self.select_best_model(decomposition_results)
            
            logger.info(f"Лучшая модель: {best_model_selection['best_model']}")
            logger.info(f"Скор: {best_model_selection['best_score']:.6f}")
            
            # Визуализация лучшей модели
            if best_model_selection['best_model']:
                best_model_name = best_model_selection['best_model']
                best_result = decomposition_results[best_model_name]
                
                logger.info(f"3. Визуализация лучшей модели: {best_model_name}")
                self.visualize_decomposition(
                    best_result['decomposition'], 
                    f'Лучшая декомпозиция: {best_model_name}'
                )
                
                logger.info(f"4. Анализ остатков для {best_model_name}")
                self.visualize_residual_analysis(
                    best_result['decomposition']['residual'],
                    best_model_name
                )
            
            # Сохранение результатов
            results = {
                'series_info': {
                    'length': len(series),
                    'period': f"{series.index.min()} - {series.index.max()}",
                    'target_column': target_column,
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max()
                },
                'decomposition_results': decomposition_results,
                'best_model_selection': best_model_selection
            }
            
            logger.info("Комплексный анализ декомпозиции завершен успешно")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в комплексном анализе декомпозиции: {e}")
            return {}


def main():
    """Основная функция для демонстрации работы модуля."""
    decomposer = TimeSeriesDecomposition()
    
    # Запуск анализа (замените путь на реальный)
    results = decomposer.run_comprehensive_decomposition(
        "../time_series/New_final.csv", 
        "Weekly_Sales"
    )
    
    if results:
        print("\n=== АНАЛИЗ ЗАВЕРШЕН ===")
        print(f"Проанализировано {results['series_info']['length']} наблюдений")
        print(f"Лучшая модель: {results['best_model_selection']['best_model']}")
    else:
        print("Анализ не удался. Проверьте путь к файлу и название колонки.")


if __name__ == "__main__":
    main()

