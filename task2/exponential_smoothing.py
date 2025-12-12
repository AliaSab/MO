"""
Модуль моделей экспоненциального сглаживания.

Реализует:
- SES (Simple Exponential Smoothing)
- Хольт (аддитивный тренд)
- Хольт (мультипликативный тренд)
- Построение прогнозов на h шагов
- Доверительные интервалы
- Диагностику адекватности модели
- Сравнение с наивным прогнозом
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
import logging
import json

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exponential_smoothing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)




class ExponentialSmoothingModels:
    """Класс для моделей экспоненциального сглаживания."""
    
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.diagnostics = {}
        
    def load_data(self, file_path: str, target_column: str = 'Weekly_Sales') -> pd.DataFrame:
        """Загрузка данных временного ряда."""
        logger.info(f"Загрузка данных из {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                logger.error("Файл пуст")
                return None
            
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
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
                logger.warning("Колонка с датами не найдена, создан искусственный индекс")
            
            if target_column not in df.columns:
                available_cols = df.columns.tolist()
                logger.error(f"Колонка '{target_column}' не найдена. Доступные колонки: {available_cols}")
                return None
            
            # Удаляем строки с NaN в целевой колонке
            initial_len = len(df)
            df = df.dropna(subset=[target_column])
            final_len = len(df)
            
            if final_len < initial_len:
                logger.warning(f"Удалено {initial_len - final_len} строк с NaN в {target_column}")
            
            logger.info(f"Загружено {len(df)} наблюдений")
            logger.info(f"Период: {df.index.min()} - {df.index.max()}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Файл {file_path} не найден")
            return None
        except pd.errors.EmptyDataError:
            logger.error(f"Файл {file_path} пуст")
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return None
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    train_size: float = 0.8) -> tuple:
        """Подготовка данных для обучения и тестирования."""
        logger.info(f"Подготовка данных с train_size={train_size}")
        
        try:
            series = df[target_column]
            
            if len(series) < 10:
                logger.error("Недостаточно данных для разделения")
                return None, None
            
            # Разделение на обучающую и тестовую выборки
            split_idx = int(len(series) * train_size)
            
            train_series = series[:split_idx]
            test_series = series[split_idx:]
            
            logger.info(f"Обучающая выборка: {len(train_series)} наблюдений")
            logger.info(f"Тестовая выборка: {len(test_series)} наблюдений")
            
            return train_series, test_series
            
        except Exception as e:
            logger.error(f"Ошибка в подготовке данных: {e}")
            return None, None
    
    def fit_ses_model(self, train_series: pd.Series, 
                      optimized: bool = True) -> dict:
        """Обучение модели Simple Exponential Smoothing."""
        logger.info("Обучение модели Simple Exponential Smoothing...")
        
        try:
            if len(train_series) < 3:
                logger.error("Недостаточно данных для обучения SES модели")
                return None
            
            model = ExponentialSmoothing(
                train_series,
                trend=None,
                seasonal=None,
                initialization_method='estimated'
            )
            
            fitted_model = model.fit(optimized=optimized)
            
            result = {
                'model_name': 'SES',
                'fitted_model': fitted_model,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'params': fitted_model.params,
                'fitted_values': fitted_model.fittedvalues,
                'residuals': fitted_model.resid
            }
            
            logger.info(f"SES модель обучена. AIC: {fitted_model.aic:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обучения SES модели: {e}")
            return None
    
    def fit_holt_additive_model(self, train_series: pd.Series, 
                               optimized: bool = True) -> dict:
        """Обучение модели Хольт с аддитивным трендом."""
        logger.info("Обучение модели Хольт (аддитивный тренд)...")
        
        try:
            if len(train_series) < 4:
                logger.error("Недостаточно данных для обучения Хольт модели")
                return None
            
            model = ExponentialSmoothing(
                train_series,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )
            
            fitted_model = model.fit(optimized=optimized)
            
            result = {
                'model_name': 'Holt Additive',
                'fitted_model': fitted_model,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'params': fitted_model.params,
                'fitted_values': fitted_model.fittedvalues,
                'residuals': fitted_model.resid
            }
            
            logger.info(f"Хольт (аддитивный) модель обучена. AIC: {fitted_model.aic:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обучения Хольт (аддитивный) модели: {e}")
            return None
    
    def fit_holt_multiplicative_model(self, train_series: pd.Series, 
                                    optimized: bool = True) -> dict:
        """Обучение модели Хольт с мультипликативным трендом."""
        logger.info("Обучение модели Хольт (мультипликативный тренд)...")
        
        try:
            if len(train_series) < 4:
                logger.error("Недостаточно данных для обучения Хольт модели")
                return None
            
            # Проверяем, что все значения положительные
            if (train_series <= 0).any():
                logger.warning("Есть неположительные значения, мультипликативная модель может работать некорректно")
            
            model = ExponentialSmoothing(
                train_series,
                trend='mul',
                seasonal=None,
                initialization_method='estimated'
            )
            
            fitted_model = model.fit(optimized=optimized)
            
            result = {
                'model_name': 'Holt Multiplicative',
                'fitted_model': fitted_model,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'params': fitted_model.params,
                'fitted_values': fitted_model.fittedvalues,
                'residuals': fitted_model.resid
            }
            
            logger.info(f"Хольт (мультипликативный) модель обучена. AIC: {fitted_model.aic:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка обучения Хольт (мультипликативный) модели: {e}")
            return None
    
    def generate_forecast(self, model_result: dict, horizon: int = 7, 
                         confidence_level: float = 0.95) -> dict:
        """Генерация прогноза на h шагов."""
        model_name = model_result['model_name']
        fitted_model = model_result['fitted_model']
        
        logger.info(f"Генерация прогноза для {model_name} на {horizon} шагов...")
        
        try:
            if horizon <= 0:
                logger.error("Горизонт прогноза должен быть положительным")
                return None
            
            # Генерация прогноза
            forecast_result = fitted_model.forecast(steps=horizon)
            
            # Доверительные интервалы
            if hasattr(fitted_model, 'get_prediction'):
                pred_intervals = fitted_model.get_prediction(steps=horizon)
                conf_int = pred_intervals.conf_int(alpha=1-confidence_level)
                
                forecast_data = {
                    'forecast': forecast_result,
                    'lower_bound': conf_int.iloc[:, 0],
                    'upper_bound': conf_int.iloc[:, 1],
                    'confidence_level': confidence_level
                }
            else:
                # Упрощенные доверительные интервалы
                residuals_std = model_result['residuals'].std()
                margin_error = 1.96 * residuals_std  # 95% доверительный интервал
                
                forecast_data = {
                    'forecast': forecast_result,
                    'lower_bound': forecast_result - margin_error,
                    'upper_bound': forecast_result + margin_error,
                    'confidence_level': confidence_level
                }
            
            logger.info(f"Прогноз для {model_name} сгенерирован успешно")
            return forecast_data
            
        except Exception as e:
            logger.error(f"Ошибка генерации прогноза для {model_name}: {e}")
            return None
    
    def naive_forecast(self, train_series: pd.Series, horizon: int = 7) -> np.ndarray:
        """Наивный прогноз (последнее значение)."""
        logger.info(f"Генерация наивного прогноза на {horizon} шагов...")
        
        try:
            if len(train_series) == 0:
                logger.error("Обучающая выборка пуста")
                return np.array([])
            
            if horizon <= 0:
                logger.error("Горизонт прогноза должен быть положительным")
                return np.array([])
            
            last_value = train_series.iloc[-1]
            naive_pred = np.full(horizon, last_value)
            
            logger.info(f"Наивный прогноз сгенерирован (значение: {last_value:.2f})")
            return naive_pred
            
        except Exception as e:
            logger.error(f"Ошибка генерации наивного прогноза: {e}")
            return np.array([])
    
    def evaluate_forecast(self, forecast: np.ndarray, actual: pd.Series) -> dict:
        """Оценка качества прогноза."""
        logger.info("Оценка качества прогноза...")
        
        try:
            if len(forecast) == 0:
                logger.error("Прогноз пуст")
                return {}
            
            if len(actual) == 0:
                logger.error("Фактические значения пусты")
                return {}
            
            # Обрезаем actual до длины прогноза
            actual_values = actual.iloc[:len(forecast)].values
            
            if len(actual_values) == 0:
                logger.error("Нет фактических значений для сравнения")
                return {}
            
            mae = np.mean(np.abs(forecast - actual_values))
            mse = np.mean((forecast - actual_values) ** 2)
            rmse = np.sqrt(mse)
            
            # MAPE с защитой от деления на ноль
            if np.all(actual_values != 0):
                mape = np.mean(np.abs((actual_values - forecast) / actual_values)) * 100
            else:
                mape = float('inf')
                logger.warning("MAPE не может быть вычислен из-за нулевых значений")
            
            metrics = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape
            }
            
            logger.info(f"Метрики прогноза: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка оценки прогноза: {e}")
            return {}
    
    def diagnostic_tests(self, model_result: dict) -> dict:
        """Диагностические тесты адекватности модели."""
        model_name = model_result['model_name']
        residuals = model_result['residuals'].dropna()
        
        logger.info(f"Проведение диагностических тестов для {model_name}...")
        
        try:
            if len(residuals) < 3:
                logger.error("Недостаточно остатков для диагностических тестов")
                return {'error': 'Недостаточно остатков'}
            
            diagnostics = {}
            
            # Тест Льюнга–Бокса на автокорреляцию остатков
            try:
                lb_result = acorr_ljungbox(residuals, lags=min(10, len(residuals)//2), return_df=True)
                diagnostics['ljung_box'] = {
                    'statistic': lb_result['lb_stat'].iloc[-1],
                    'p_value': lb_result['lb_pvalue'].iloc[-1],
                    'is_adequate': lb_result['lb_pvalue'].iloc[-1] > 0.05
                }
            except Exception as e:
                logger.warning(f"Ошибка в тесте Льюнга-Бокса: {e}")
                diagnostics['ljung_box'] = {'error': str(e)}
            
            # Тест нормальности остатков (Шапиро-Уилка)
            # Для больших выборок используем случайную подвыборку (максимум 5000 точек)
            try:
                if len(residuals) > 5000:
                    # Для больших выборок используем случайную подвыборку
                    np.random.seed(42)  # Для воспроизводимости
                    sample_size = min(5000, len(residuals))
                    sample_indices = np.random.choice(len(residuals), size=sample_size, replace=False)
                    sample_residuals = residuals.iloc[sample_indices]
                    logger.info(f"Для теста Шапиро-Уилка использована случайная выборка из {sample_size} точек (из {len(residuals)})")
                else:
                    sample_residuals = residuals
                
                shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
                diagnostics['normality'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05,
                    'sample_size': len(sample_residuals),
                    'original_size': len(residuals)
                }
                logger.info(f"Тест нормальности Шапиро-Уилка: статистика={shapiro_stat:.4f}, p-value={shapiro_p:.4f}, выборка={len(sample_residuals)}")
            except Exception as e:
                logger.warning(f"Ошибка в тесте нормальности: {e}")
                diagnostics['normality'] = {'error': str(e)}
            
            # Статистики остатков
            try:
                diagnostics['residual_stats'] = {
                    'mean': residuals.mean(),
                    'std': residuals.std(),
                    'skewness': stats.skew(residuals),
                    'kurtosis': stats.kurtosis(residuals)
                }
            except Exception as e:
                logger.warning(f"Ошибка в вычислении статистик остатков: {e}")
                diagnostics['residual_stats'] = {'error': str(e)}
            
            logger.info(f"Диагностические тесты для {model_name} завершены")
            return diagnostics
            
        except Exception as e:
            logger.error(f"Ошибка в диагностических тестах для {model_name}: {e}")
            return {'error': str(e)}
    
    def visualize_model_results(self, model_results: dict, train_series: pd.Series, 
                               test_series: pd.Series, forecasts: dict) -> None:
        """Визуализация результатов моделей."""
        logger.info("Создание визуализации результатов...")
        
        try:
            n_models = len(model_results)
            if n_models == 0:
                logger.warning("Нет моделей для визуализации")
                return
            
            fig, axes = plt.subplots(n_models, 2, figsize=(15, 5*n_models))
            
            if n_models == 1:
                axes = axes.reshape(1, -1)
            
            for i, (model_name, model_result) in enumerate(model_results.items()):
                # График временного ряда с прогнозом (с подвыборкой для больших данных)
                def downsample_for_plot(series, max_points=5000):
                    """Подвыборка данных для визуализации."""
                    if len(series) <= max_points:
                        return series
                    step = len(series) // max_points
                    indices = range(0, len(series), step)
                    return series.iloc[indices]
                
                train_plot_data = downsample_for_plot(train_series)
                test_plot_data = downsample_for_plot(test_series)
                
                axes[i, 0].plot(train_plot_data.index, train_plot_data.values, 
                              label='Обучающие данные', alpha=0.7, linewidth=0.8)
                axes[i, 0].plot(test_plot_data.index, test_plot_data.values, 
                              label='Тестовые данные', alpha=0.7, linewidth=0.8)
                
                if model_name in forecasts:
                    forecast_data = forecasts[model_name]
                    forecast_steps = range(len(train_series), len(train_series) + len(forecast_data['forecast']))
                    
                    axes[i, 0].plot(forecast_steps, forecast_data['forecast'], 
                                  label='Прогноз', color='red', linewidth=2)
                    
                    # Доверительные интервалы
                    axes[i, 0].fill_between(forecast_steps, 
                                          forecast_data['lower_bound'], 
                                          forecast_data['upper_bound'],
                                          alpha=0.3, color='red', 
                                          label=f'{forecast_data["confidence_level"]*100:.0f}% ДИ')
                
                axes[i, 0].set_title(f'{model_name} - Прогноз')
                axes[i, 0].set_xlabel('Время')
                axes[i, 0].set_ylabel('Значение')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
                
                # График остатков (с подвыборкой для больших данных)
                residuals = model_result['residuals'].dropna()
                if len(residuals) > 0:
                    # Подвыборка для визуализации больших данных
                    def downsample_for_plot(series, max_points=5000):
                        """Подвыборка данных для визуализации."""
                        if len(series) <= max_points:
                            return series
                        step = len(series) // max_points
                        indices = range(0, len(series), step)
                        return series.iloc[indices]
                    
                    residual_plot_data = downsample_for_plot(residuals)
                    axes[i, 1].plot(residual_plot_data.index, residual_plot_data.values, 
                                  alpha=0.7, linewidth=0.8)
                    axes[i, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                    axes[i, 1].set_title(f'{model_name} - Остатки')
                    axes[i, 1].set_xlabel('Время')
                    axes[i, 1].set_ylabel('Остаток')
                    axes[i, 1].grid(True, alpha=0.3)
                else:
                    axes[i, 1].text(0.5, 0.5, 'Нет остатков', 
                                   ha='center', va='center', transform=axes[i, 1].transAxes)
                    axes[i, 1].set_title(f'{model_name} - Остатки')
            
            plt.tight_layout()
            plt.savefig('exponential_smoothing_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info("Визуализация сохранена в exponential_smoothing_results.png")
            
        except Exception as e:
            logger.error(f"Ошибка в визуализации результатов: {e}")
    
    def create_summary_table(self, model_results: dict, forecasts: dict, 
                           test_series: pd.Series, naive_forecast: np.ndarray) -> pd.DataFrame:
        """Создание сводной таблицы результатов."""
        logger.info("Создание сводной таблицы результатов...")
        
        try:
            summary_data = []
            
            # Оценка наивного прогноза
            if len(naive_forecast) > 0:
                naive_metrics = self.evaluate_forecast(naive_forecast, test_series)
                if naive_metrics:
                    summary_data.append({
                        'Модель': 'Наивный прогноз',
                        'AIC': None,
                        'BIC': None,
                        'MAE': naive_metrics['mae'],
                        'RMSE': naive_metrics['rmse'],
                        'MAPE': naive_metrics['mape']
                    })
            
            # Оценка моделей экспоненциального сглаживания
            for model_name, model_result in model_results.items():
                if model_name in forecasts:
                    forecast_metrics = self.evaluate_forecast(
                        forecasts[model_name]['forecast'], test_series
                    )
                    
                    if forecast_metrics:
                        summary_data.append({
                            'Модель': model_name,
                            'AIC': model_result['aic'],
                            'BIC': model_result['bic'],
                            'MAE': forecast_metrics['mae'],
                            'RMSE': forecast_metrics['rmse'],
                            'MAPE': forecast_metrics['mape']
                        })
            
            df = pd.DataFrame(summary_data)
            logger.info(f"Создана сводная таблица с {len(df)} строками")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка в создании сводной таблицы: {e}")
            return pd.DataFrame()
    
    def run_comprehensive_analysis(self, file_path: str, target_column: str = 'Weekly_Sales',
                                  horizon: int = 7) -> dict:
        """Запуск комплексного анализа моделей экспоненциального сглаживания."""
        logger.info("=== КОМПЛЕКСНЫЙ АНАЛИЗ ЭКСПОНЕНЦИАЛЬНОГО СГЛАЖИВАНИЯ ===")
        
        try:
            # Загрузка данных
            df = self.load_data(file_path, target_column)
            if df is None:
                logger.error("Не удалось загрузить данные")
                return {}
            
            # Подготовка данных
            train_series, test_series = self.prepare_data(df, target_column)
            if train_series is None or test_series is None:
                logger.error("Не удалось подготовить данные")
                return {}
            
            # Обучение моделей
            model_results = {}
            
            # SES модель
            logger.info("Обучение SES модели...")
            ses_result = self.fit_ses_model(train_series)
            if ses_result:
                model_results['SES'] = ses_result
            
            # Хольт аддитивный
            logger.info("Обучение Хольт аддитивной модели...")
            holt_add_result = self.fit_holt_additive_model(train_series)
            if holt_add_result:
                model_results['Holt Additive'] = holt_add_result
            
            # Хольт мультипликативный
            logger.info("Обучение Хольт мультипликативной модели...")
            holt_mul_result = self.fit_holt_multiplicative_model(train_series)
            if holt_mul_result:
                model_results['Holt Multiplicative'] = holt_mul_result
            
            if not model_results:
                logger.error("Не удалось обучить ни одной модели")
                return {}
            
            # Генерация прогнозов
            logger.info("Генерация прогнозов...")
            forecasts = {}
            for model_name, model_result in model_results.items():
                forecast_data = self.generate_forecast(model_result, horizon)
                if forecast_data:
                    forecasts[model_name] = forecast_data
            
            # Наивный прогноз
            logger.info("Генерация наивного прогноза...")
            naive_forecast = self.naive_forecast(train_series, horizon)
            
            # Диагностические тесты
            logger.info("Проведение диагностических тестов...")
            diagnostics = {}
            for model_name, model_result in model_results.items():
                diagnostics[model_name] = self.diagnostic_tests(model_result)
            
            # Визуализация
            self.visualize_model_results(model_results, train_series, test_series, forecasts)
            
            # Создание сводной таблицы
            summary_df = self.create_summary_table(model_results, forecasts, test_series, naive_forecast)
            
            if not summary_df.empty:
                logger.info("=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
                logger.info(f"\n{summary_df.to_string(index=False)}")
                
                # Сохранение результатов
                summary_df.to_csv('exponential_smoothing_results.csv', index=False, encoding='utf-8')
                logger.info("Сводная таблица сохранена в exponential_smoothing_results.csv")
            
            final_results = {
                'model_results': model_results,
                'forecasts': forecasts,
                'diagnostics': diagnostics,
                'naive_forecast': naive_forecast.tolist() if len(naive_forecast) > 0 else [],
                'summary': summary_df.to_dict('records') if not summary_df.empty else [],
                'series_info': {
                    'train_length': len(train_series),
                    'test_length': len(test_series),
                    'horizon': horizon
                }
            }
            
            with open('exponential_smoothing_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("Результаты сохранены:")
            logger.info("- exponential_smoothing_results.csv")
            logger.info("- exponential_smoothing_analysis.json")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Ошибка в комплексном анализе: {e}")
            return {}


def main():
    """Основная функция для демонстрации работы модуля."""
    logger.info("Запуск анализа экспоненциального сглаживания...")
    
    try:
        es_analyzer = ExponentialSmoothingModels()
        
        # Запуск анализа
        results = es_analyzer.run_comprehensive_analysis(
            "../New_final.csv", 
            "Weekly_Sales",
            horizon=7
        )
        
        if results:
            logger.info("=== АНАЛИЗ ЭКСПОНЕНЦИАЛЬНОГО СГЛАЖИВАНИЯ ЗАВЕРШЕН ===")
            logger.info("Обучены и протестированы модели экспоненциального сглаживания")
        else:
            logger.error("Анализ не удался. Проверьте путь к файлу.")
            
    except Exception as e:
        logger.error(f"Ошибка в main: {e}")


if __name__ == "__main__":
    main()

