"""
Модуль стратегий многопшагового прогнозирования.

Реализует:
- Рекурсивную стратегию (одна модель → итеративное использование прогнозов)
- Прямую стратегию (отдельная модель для каждого шага)
- Гибридную стратегию (рекурсивная для ближних, прямая для дальних шагов)
- Сравнение по точности, времени вычислений, накоплению ошибки
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import warnings
import logging
import json
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class MultiStepForecasting:
    """Класс для многопшагового прогнозирования."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.strategies = ['recursive', 'direct', 'hybrid']
        
    def load_data(self, file_path: str, target_column: str = 'Weekly_Sales') -> pd.DataFrame:
        """Загрузка данных."""
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
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    feature_columns: list = None) -> tuple:
        """Подготовка данных для обучения."""
        logger.info("Подготовка данных для обучения...")
        
        try:
            if feature_columns is None:
                # Автоматический выбор признаков
                feature_columns = [col for col in df.columns 
                                 if col != target_column and df[col].dtype in ['int64', 'float64']]
            
            logger.info(f"Выбрано {len(feature_columns)} признаков для обучения")
            
            # Удаляем строки с пропущенными значениями
            clean_df = df[feature_columns + [target_column]].dropna()
            
            if len(clean_df) == 0:
                logger.error("Нет данных после удаления пропущенных значений")
                return None, None, None
            
            X = clean_df[feature_columns]
            y = clean_df[target_column]
            
            logger.info(f"Подготовлено {len(X)} образцов с {len(feature_columns)} признаками")
            return X, y, feature_columns
            
        except Exception as e:
            logger.error(f"Ошибка подготовки данных: {e}")
            return None, None, None
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   train_size: float = 0.8) -> tuple:
        """Разделение данных на обучающую и тестовую выборки."""
        logger.info(f"Разделение данных на обучающую ({train_size*100:.1f}%) и тестовую выборки...")
        
        try:
            if len(X) < 10:
                logger.error("Недостаточно данных для разделения")
                return None, None, None, None
            
            split_idx = int(len(X) * train_size)
            
            if split_idx < 5:
                logger.warning("Обучающая выборка слишком мала, используем минимум 5 образцов")
                split_idx = 5
            
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"Обучающая выборка: {len(X_train)} образцов")
            logger.info(f"Тестовая выборка: {len(X_test)} образцов")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Ошибка разделения данных: {e}")
            return None, None, None, None
    
    def recursive_strategy(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          model, horizon: int = 7) -> dict:
        """Рекурсивная стратегия прогнозирования."""
        logger.info(f"Рекурсивная стратегия (горизонт: {horizon})")
        
        try:
            start_time = time.time()
            
            # Обучение модели на обучающих данных
            model.fit(X_train, y_train)
            
            # Инициализация прогнозов
            predictions = []
            current_X = X_test.iloc[0:1].copy()
            
            for step in range(horizon):
                # Прогноз на следующий шаг
                pred = model.predict(current_X)[0]
                predictions.append(pred)
                
                # Обновление признаков для следующего шага
                if step < horizon - 1:
                    # Сдвигаем лаговые признаки
                    for col in current_X.columns:
                        if 'lag_' in col:
                            lag_num = int(col.split('_')[1])
                            if lag_num == 1:
                                current_X[col] = pred
                            else:
                                # Обновляем другие лаги
                                new_lag_col = f'lag_{lag_num-1}'
                                if new_lag_col in current_X.columns:
                                    current_X[col] = current_X[new_lag_col]
                    
                    # Обновляем скользящие статистики (упрощенно)
                    for col in current_X.columns:
                        if 'rolling_mean_' in col:
                            window = int(col.split('_')[2])
                            # Упрощенное обновление скользящего среднего
                            current_X[col] = (current_X[col] * (window - 1) + pred) / window
            
            predictions = np.array(predictions)
            
            # Вычисление метрик для каждого шага
            step_metrics = {}
            for step in range(horizon):
                if step < len(y_test):
                    actual = y_test.iloc[step]
                    predicted = predictions[step]
                    
                    step_metrics[f'step_{step+1}'] = {
                        'mae': abs(actual - predicted),
                        'mse': (actual - predicted) ** 2,
                        'actual': actual,
                        'predicted': predicted
                    }
            
            # Общие метрики
            actual_values = y_test.iloc[:horizon].values
            mae = mean_absolute_error(actual_values, predictions)
            mse = mean_squared_error(actual_values, predictions)
            rmse = np.sqrt(mse)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Рекурсивная стратегия завершена за {processing_time:.2f} сек")
            
            return {
                'strategy': 'recursive',
                'predictions': predictions,
                'step_metrics': step_metrics,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'processing_time': processing_time,
                'horizon': horizon
            }
            
        except Exception as e:
            logger.error(f"Ошибка в рекурсивной стратегии: {e}")
            return {
                'strategy': 'recursive',
                'error': str(e),
                'predictions': [],
                'step_metrics': {},
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'processing_time': 0,
                'horizon': horizon
            }
    
    def direct_strategy(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       model_class, horizon: int = 7) -> dict:
        """Прямая стратегия прогнозирования."""
        logger.info(f"Прямая стратегия (горизонт: {horizon})")
        
        try:
            start_time = time.time()
            
            # Обучение отдельных моделей для каждого шага
            models = {}
            predictions = []
            step_metrics = {}
            
            for step in range(horizon):
                logger.info(f"Обучение модели для шага {step+1}")
                
                # Создание целевой переменной для текущего шага
                if step == 0:
                    y_step = y_train
                    X_step = X_train
                else:
                    # Сдвигаем целевую переменную на step шагов вперед
                    y_step = y_train.shift(-step).dropna()
                    X_step = X_train.iloc[:len(y_step)]
                
                if len(y_step) < 5:
                    logger.warning(f"Недостаточно данных для шага {step+1}, пропускаем")
                    predictions.append(0)
                    continue
                
                # Обучение модели
                model = model_class()
                model.fit(X_step, y_step)
                models[f'step_{step+1}'] = model
                
                # Прогноз на тестовых данных
                if step < len(X_test):
                    pred = model.predict(X_test.iloc[step:step+1])[0]
                    predictions.append(pred)
                else:
                    predictions.append(0)
                
                # Метрики для текущего шага
                if step < len(y_test):
                    actual = y_test.iloc[step]
                    predicted = predictions[step]
                    
                    step_metrics[f'step_{step+1}'] = {
                        'mae': abs(actual - predicted),
                        'mse': (actual - predicted) ** 2,
                        'actual': actual,
                        'predicted': predicted
                    }
            
            predictions = np.array(predictions)
            
            # Общие метрики
            actual_values = y_test.iloc[:horizon].values
            mae = mean_absolute_error(actual_values, predictions)
            mse = mean_squared_error(actual_values, predictions)
            rmse = np.sqrt(mse)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Прямая стратегия завершена за {processing_time:.2f} сек")
            
            return {
                'strategy': 'direct',
                'predictions': predictions,
                'step_metrics': step_metrics,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'processing_time': processing_time,
                'horizon': horizon,
                'models': models
            }
            
        except Exception as e:
            logger.error(f"Ошибка в прямой стратегии: {e}")
            return {
                'strategy': 'direct',
                'error': str(e),
                'predictions': [],
                'step_metrics': {},
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'processing_time': 0,
                'horizon': horizon,
                'models': {}
            }
    
    def hybrid_strategy(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series,
                       model_class, horizon: int = 7,
                       recursive_steps: int = 3) -> dict:
        """Гибридная стратегия прогнозирования."""
        logger.info(f"Гибридная стратегия (горизонт: {horizon}, рекурсивных шагов: {recursive_steps})")
        
        try:
            start_time = time.time()
            
            predictions = []
            step_metrics = {}
            
            # Рекурсивная часть для ближайших шагов
            if recursive_steps > 0:
                logger.info(f"Выполнение рекурсивной части для {recursive_steps} шагов")
                recursive_result = self.recursive_strategy(
                    X_train, y_train, X_test, y_test,
                    model_class(), min(recursive_steps, horizon)
                )
                
                if 'error' not in recursive_result:
                    predictions.extend(recursive_result['predictions'])
                    step_metrics.update(recursive_result['step_metrics'])
                else:
                    logger.error(f"Ошибка в рекурсивной части: {recursive_result['error']}")
                    return {
                        'strategy': 'hybrid',
                        'error': recursive_result['error'],
                        'predictions': [],
                        'step_metrics': {},
                        'mae': float('inf'),
                        'mse': float('inf'),
                        'rmse': float('inf'),
                        'processing_time': 0,
                        'horizon': horizon,
                        'recursive_steps': recursive_steps
                    }
            
            # Прямая часть для дальних шагов
            if horizon > recursive_steps:
                logger.info(f"Выполнение прямой части для {horizon - recursive_steps} шагов")
                
                # Подготовка данных для прямой стратегии
                remaining_steps = horizon - recursive_steps
                
                # Создаем новые признаки с учетом уже сделанных прогнозов
                X_test_updated = X_test.copy()
                
                # Обновляем лаговые признаки на основе прогнозов
                for step in range(recursive_steps):
                    if step < len(predictions):
                        pred_value = predictions[step]
                        
                        for col in X_test_updated.columns:
                            if 'lag_1' in col:
                                X_test_updated[col] = pred_value
                            elif 'lag_' in col:
                                lag_num = int(col.split('_')[1])
                                if lag_num > 1:
                                    prev_lag_col = f'lag_{lag_num-1}'
                                    if prev_lag_col in X_test_updated.columns:
                                        X_test_updated[col] = X_test_updated[prev_lag_col]
                
                # Прямая стратегия для оставшихся шагов
                direct_result = self.direct_strategy(
                    X_train, y_train, X_test_updated, y_test,
                    model_class, remaining_steps
                )
                
                if 'error' not in direct_result:
                    predictions.extend(direct_result['predictions'])
                    
                    # Обновляем метрики с правильными индексами
                    for step_key, metrics in direct_result['step_metrics'].items():
                        step_num = int(step_key.split('_')[1])
                        new_step_key = f'step_{step_num + recursive_steps}'
                        step_metrics[new_step_key] = metrics
                else:
                    logger.error(f"Ошибка в прямой части: {direct_result['error']}")
            
            predictions = np.array(predictions)
            
            # Общие метрики
            actual_values = y_test.iloc[:horizon].values
            mae = mean_absolute_error(actual_values, predictions)
            mse = mean_squared_error(actual_values, predictions)
            rmse = np.sqrt(mse)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Гибридная стратегия завершена за {processing_time:.2f} сек")
            
            return {
                'strategy': 'hybrid',
                'predictions': predictions,
                'step_metrics': step_metrics,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'processing_time': processing_time,
                'horizon': horizon,
                'recursive_steps': recursive_steps
            }
            
        except Exception as e:
            logger.error(f"Ошибка в гибридной стратегии: {e}")
            return {
                'strategy': 'hybrid',
                'error': str(e),
                'predictions': [],
                'step_metrics': {},
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'processing_time': 0,
                'horizon': horizon,
                'recursive_steps': recursive_steps
            }
    
    def compare_strategies(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series,
                          model_class, horizon: int = 7) -> dict:
        """Сравнение всех стратегий прогнозирования."""
        logger.info("=== СРАВНЕНИЕ СТРАТЕГИЙ МНОГОШАГОВОГО ПРОГНОЗИРОВАНИЯ ===")
        
        try:
            results = {}
            
            # Рекурсивная стратегия
            logger.info("1. Рекурсивная стратегия")
            results['recursive'] = self.recursive_strategy(
                X_train, y_train, X_test, y_test, model_class(), horizon
            )
            
            # Прямая стратегия
            logger.info("2. Прямая стратегия")
            results['direct'] = self.direct_strategy(
                X_train, y_train, X_test, y_test, model_class, horizon
            )
            
            # Гибридная стратегия
            logger.info("3. Гибридная стратегия")
            results['hybrid'] = self.hybrid_strategy(
                X_train, y_train, X_test, y_test, model_class, horizon
            )
            
            logger.info("Сравнение стратегий завершено")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в сравнении стратегий: {e}")
            return {}
    
    def analyze_error_accumulation(self, results: dict) -> dict:
        """Анализ накопления ошибки по шагам."""
        logger.info("Анализ накопления ошибки...")
        
        try:
            error_analysis = {}
            
            for strategy_name, result in results.items():
                if 'error' in result:
                    logger.warning(f"Стратегия {strategy_name} содержит ошибки, пропускаем анализ")
                    continue
                
                step_errors = []
                step_maes = []
                
                for step_key in sorted(result['step_metrics'].keys()):
                    step_num = int(step_key.split('_')[1])
                    metrics = result['step_metrics'][step_key]
                    
                    step_errors.append(metrics['mse'])
                    step_maes.append(metrics['mae'])
                
                if len(step_errors) > 1:
                    error_analysis[strategy_name] = {
                        'step_errors': step_errors,
                        'step_maes': step_maes,
                        'error_trend': np.polyfit(range(len(step_errors)), step_errors, 1)[0],
                        'mae_trend': np.polyfit(range(len(step_maes)), step_maes, 1)[0]
                    }
                else:
                    logger.warning(f"Недостаточно данных для анализа тренда ошибки в стратегии {strategy_name}")
                    error_analysis[strategy_name] = {
                        'step_errors': step_errors,
                        'step_maes': step_maes,
                        'error_trend': 0,
                        'mae_trend': 0
                    }
            
            logger.info("Анализ накопления ошибки завершен")
            return error_analysis
            
        except Exception as e:
            logger.error(f"Ошибка в анализе накопления ошибки: {e}")
            return {}
    
    def visualize_comparison(self, results: dict, y_test: pd.Series, 
                           horizon: int = 7) -> None:
        """Визуализация сравнения стратегий."""
        logger.info("Создание визуализации сравнения...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Сравнение стратегий многопшагового прогнозирования', fontsize=16)
            
            # График прогнозов
            actual_values = y_test.iloc[:horizon].values
            steps = range(1, horizon + 1)
            
            axes[0, 0].plot(steps, actual_values, 'o-', label='Фактические значения', linewidth=2)
            
            colors = ['red', 'blue', 'green']
            for i, (strategy_name, result) in enumerate(results.items()):
                if 'error' not in result and len(result['predictions']) > 0:
                    axes[0, 0].plot(steps, result['predictions'], 'o-', 
                                  label=f'{strategy_name.capitalize()}', 
                                  color=colors[i], alpha=0.7)
            
            axes[0, 0].set_title('Прогнозы по шагам')
            axes[0, 0].set_xlabel('Шаг прогноза')
            axes[0, 0].set_ylabel('Значение')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # График ошибок по шагам
            for i, (strategy_name, result) in enumerate(results.items()):
                if 'error' not in result and 'step_metrics' in result:
                    step_maes = []
                    for step_key in sorted(result['step_metrics'].keys()):
                        step_maes.append(result['step_metrics'][step_key]['mae'])
                    
                    if len(step_maes) > 0:
                        axes[0, 1].plot(steps[:len(step_maes)], step_maes, 'o-', 
                                      label=f'{strategy_name.capitalize()}', 
                                      color=colors[i], alpha=0.7)
            
            axes[0, 1].set_title('MAE по шагам')
            axes[0, 1].set_xlabel('Шаг прогноза')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Сравнение метрик
            strategies = []
            maes = []
            rmses = []
            times = []
            
            for strategy_name, result in results.items():
                if 'error' not in result:
                    strategies.append(strategy_name)
                    maes.append(result['mae'])
                    rmses.append(result['rmse'])
                    times.append(result['processing_time'])
            
            if strategies:
                x = np.arange(len(strategies))
                width = 0.25
                
                axes[1, 0].bar(x - width, maes, width, label='MAE', alpha=0.7)
                axes[1, 0].bar(x, rmses, width, label='RMSE', alpha=0.7)
                axes[1, 0].set_title('Сравнение метрик качества')
                axes[1, 0].set_xlabel('Стратегия')
                axes[1, 0].set_ylabel('Значение метрики')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels([s.capitalize() for s in strategies])
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Время вычислений
                axes[1, 1].bar(strategies, times, alpha=0.7, color=colors[:len(strategies)])
                axes[1, 1].set_title('Время вычислений')
                axes[1, 1].set_xlabel('Стратегия')
                axes[1, 1].set_ylabel('Время (сек)')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            logger.info("Визуализация сравнения создана успешно")
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализации сравнения: {e}")
    
    def create_summary_table(self, results: dict) -> pd.DataFrame:
        """Создание сводной таблицы результатов."""
        logger.info("Создание сводной таблицы результатов...")
        
        try:
            summary_data = []
            
            for strategy_name, result in results.items():
                if 'error' not in result:
                    summary_data.append({
                        'Стратегия': strategy_name.capitalize(),
                        'MAE': result['mae'],
                        'RMSE': result['rmse'],
                        'MSE': result['mse'],
                        'Время (сек)': result['processing_time'],
                        'Горизонт': result['horizon']
                    })
                else:
                    logger.warning(f"Стратегия {strategy_name} содержит ошибки, пропускаем в таблице")
            
            if not summary_data:
                logger.warning("Нет данных для создания сводной таблицы")
                return pd.DataFrame()
            
            df = pd.DataFrame(summary_data)
            logger.info(f"Создана сводная таблица с {len(df)} стратегиями")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка создания сводной таблицы: {e}")
            return pd.DataFrame()
    
    def run_comprehensive_analysis(self, file_path: str, target_column: str = 'Weekly_Sales',
                                  horizon: int = 7) -> dict:
        """Запуск комплексного анализа многопшагового прогнозирования."""
        logger.info("=== КОМПЛЕКСНЫЙ АНАЛИЗ МНОГОШАГОВОГО ПРОГНОЗИРОВАНИЯ ===")
        
        try:
            # Загрузка данных
            df = self.load_data(file_path, target_column)
            if df is None:
                logger.error("Не удалось загрузить данные")
                return {}
            
            # Подготовка данных
            X, y, feature_columns = self.prepare_data(df, target_column)
            if X is None or y is None:
                logger.error("Не удалось подготовить данные")
                return {}
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            if X_train is None:
                logger.error("Не удалось разделить данные")
                return {}
            
            # Сравнение стратегий
            logger.info("Сравнение стратегий прогнозирования...")
            results = self.compare_strategies(
                X_train, y_train, X_test, y_test, 
                LinearRegression, horizon
            )
            
            if not results:
                logger.error("Не удалось выполнить сравнение стратегий")
                return {}
            
            # Анализ накопления ошибки
            logger.info("Анализ накопления ошибки...")
            error_analysis = self.analyze_error_accumulation(results)
            
            # Визуализация
            logger.info("Создание визуализации...")
            self.visualize_comparison(results, y_test, horizon)
            
            # Создание сводной таблицы
            logger.info("Создание сводной таблицы...")
            summary_df = self.create_summary_table(results)
            
            if not summary_df.empty:
                logger.info("=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
                print(summary_df.to_string(index=False))
                
                # Сохранение результатов
                summary_df.to_csv('multi_step_forecasting_results.csv', index=False, encoding='utf-8')
                logger.info("Результаты сохранены в multi_step_forecasting_results.csv")
            
            final_results = {
                'results': results,
                'error_analysis': error_analysis,
                'summary': summary_df.to_dict('records') if not summary_df.empty else []
            }
            
            with open('multi_step_forecasting_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("Результаты сохранены:")
            logger.info("- multi_step_forecasting_results.csv")
            logger.info("- multi_step_forecasting_analysis.json")
            logger.info("Комплексный анализ многопшагового прогнозирования завершен успешно")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Ошибка в комплексном анализе: {e}")
            return {}


def main():
    """Основная функция для демонстрации работы модуля."""
    logger.info("Запуск демонстрации модуля многопшагового прогнозирования")
    
    try:
        forecaster = MultiStepForecasting()
        
        # Запуск анализа
        results = forecaster.run_comprehensive_analysis(
            "../New_final.csv", 
            "Weekly_Sales",
            horizon=7
        )
        
        if results:
            logger.info("=== АНАЛИЗ ЗАВЕРШЕН ===")
            logger.info("Сравнены три стратегии многопшагового прогнозирования")
        else:
            logger.error("Анализ не удался. Проверьте путь к файлу.")
            
    except Exception as e:
        logger.error(f"Ошибка в основной функции: {e}")


if __name__ == "__main__":
    main()

