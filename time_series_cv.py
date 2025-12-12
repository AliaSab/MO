"""
Модуль кросс-валидации для временных рядов.

Реализует:
- Скользящее окно (фиксированная длина обучения, сдвиг по времени)
- Расширяющееся окно (обучение растёт со временем)
- TimeSeriesSplit (sklearn.model_selection)
- Оценку среднего качества по фолдам
- Анализ стабильности метрик во времени
- Визуализацию динамики ошибки по фолдам
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import logging
import json
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)






class TimeSeriesCrossValidation:
    """Класс для кросс-валидации временных рядов."""
    
    def __init__(self):
        self.cv_results = {}
        self.models = {}
        
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
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    feature_columns: list = None) -> tuple:
        """Подготовка данных для обучения."""
        logger.info("Подготовка данных для обучения...")
        
        try:
            if feature_columns is None:
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
    
    def rolling_window_cv(self, X: pd.DataFrame, y: pd.Series, 
                         model, train_size: int = 100, 
                         test_size: int = 20, step: int = 10) -> dict:
        """Кросс-валидация со скользящим окном."""
        logger.info(f"Скользящее окно CV: train_size={train_size}, test_size={test_size}, step={step}")
        
        try:
            if len(X) < train_size + test_size:
                logger.error(f"Недостаточно данных для скользящего окна. Требуется минимум {train_size + test_size}, доступно {len(X)}")
                return {
                    'fold_scores': [],
                    'fold_metrics': [],
                    'fold_indices': [],
                    'cv_type': 'rolling_window',
                    'error': 'Недостаточно данных'
                }
            
            results = {
                'fold_scores': [],
                'fold_metrics': [],
                'fold_indices': [],
                'cv_type': 'rolling_window'
            }
            
            fold = 0
            start_idx = 0
            
            while start_idx + train_size + test_size <= len(X):
                # Определение индексов для текущего фолда
                train_end = start_idx + train_size
                test_start = train_end
                test_end = test_start + test_size
                
                train_indices = range(start_idx, train_end)
                test_indices = range(test_start, test_end)
                
                # Разделение данных
                X_train_fold = X.iloc[train_indices]
                y_train_fold = y.iloc[train_indices]
                X_test_fold = X.iloc[test_indices]
                y_test_fold = y.iloc[test_indices]
                
                # Обучение модели
                model.fit(X_train_fold, y_train_fold)
                
                # Прогноз
                y_pred_fold = model.predict(X_test_fold)
                
                # Вычисление метрик
                mae = mean_absolute_error(y_test_fold, y_pred_fold)
                mse = mean_squared_error(y_test_fold, y_pred_fold)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_fold, y_pred_fold)
                
                fold_metrics = {
                    'fold': fold,
                    'train_start': start_idx,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'train_size': len(X_train_fold),
                    'test_size': len(X_test_fold)
                }
                
                results['fold_scores'].append(mae)  # Используем MAE как основную метрику
                results['fold_metrics'].append(fold_metrics)
                results['fold_indices'].append((train_indices, test_indices))
                
                fold += 1
                start_idx += step
            
            # Общие метрики
            if results['fold_scores']:
                results['mean_score'] = np.mean(results['fold_scores'])
                results['std_score'] = np.std(results['fold_scores'])
                results['n_folds'] = len(results['fold_scores'])
                
                logger.info(f"Выполнено {results['n_folds']} фолдов")
                logger.info(f"Средний MAE: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
            else:
                logger.warning("Не удалось выполнить ни одного фолда")
                results['mean_score'] = float('inf')
                results['std_score'] = 0
                results['n_folds'] = 0
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в скользящем окне CV: {e}")
            return {
                'fold_scores': [],
                'fold_metrics': [],
                'fold_indices': [],
                'cv_type': 'rolling_window',
                'error': str(e)
            }
    
    def expanding_window_cv(self, X: pd.DataFrame, y: pd.Series, 
                           model, initial_train_size: int = 50,
                           test_size: int = 20, step: int = 10) -> dict:
        """Кросс-валидация с расширяющимся окном."""
        logger.info(f"Расширяющееся окно CV: initial_train_size={initial_train_size}, test_size={test_size}, step={step}")
        
        try:
            if len(X) < initial_train_size + test_size:
                logger.error(f"Недостаточно данных для расширяющегося окна. Требуется минимум {initial_train_size + test_size}, доступно {len(X)}")
                return {
                    'fold_scores': [],
                    'fold_metrics': [],
                    'fold_indices': [],
                    'cv_type': 'expanding_window',
                    'error': 'Недостаточно данных'
                }
            
            results = {
                'fold_scores': [],
                'fold_metrics': [],
                'fold_indices': [],
                'cv_type': 'expanding_window'
            }
            
            fold = 0
            train_size = initial_train_size
            
            while train_size + test_size <= len(X):
                # Определение индексов для текущего фолда
                train_indices = range(0, train_size)
                test_start = train_size
                test_end = test_start + test_size
                test_indices = range(test_start, test_end)
                
                # Разделение данных
                X_train_fold = X.iloc[train_indices]
                y_train_fold = y.iloc[train_indices]
                X_test_fold = X.iloc[test_indices]
                y_test_fold = y.iloc[test_indices]
                
                # Обучение модели
                model.fit(X_train_fold, y_train_fold)
                
                # Прогноз
                y_pred_fold = model.predict(X_test_fold)
                
                # Вычисление метрик
                mae = mean_absolute_error(y_test_fold, y_pred_fold)
                mse = mean_squared_error(y_test_fold, y_pred_fold)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_fold, y_pred_fold)
                
                fold_metrics = {
                    'fold': fold,
                    'train_start': 0,
                    'train_end': train_size,
                    'test_start': test_start,
                    'test_end': test_end,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'train_size': len(X_train_fold),
                    'test_size': len(X_test_fold)
                }
                
                results['fold_scores'].append(mae)
                results['fold_metrics'].append(fold_metrics)
                results['fold_indices'].append((train_indices, test_indices))
                
                fold += 1
                train_size += step
            
            # Общие метрики
            if results['fold_scores']:
                results['mean_score'] = np.mean(results['fold_scores'])
                results['std_score'] = np.std(results['fold_scores'])
                results['n_folds'] = len(results['fold_scores'])
                
                logger.info(f"Выполнено {results['n_folds']} фолдов")
                logger.info(f"Средний MAE: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
            else:
                logger.warning("Не удалось выполнить ни одного фолда")
                results['mean_score'] = float('inf')
                results['std_score'] = 0
                results['n_folds'] = 0
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в расширяющемся окне CV: {e}")
            return {
                'fold_scores': [],
                'fold_metrics': [],
                'fold_indices': [],
                'cv_type': 'expanding_window',
                'error': str(e)
            }
    
    def sklearn_timeseries_cv(self, X: pd.DataFrame, y: pd.Series, 
                             model, n_splits: int = 5) -> dict:
        """Кросс-валидация с использованием TimeSeriesSplit из sklearn."""
        logger.info(f"Sklearn TimeSeriesSplit CV: n_splits={n_splits}")
        
        try:
            if len(X) < n_splits * 2:
                logger.error(f"Недостаточно данных для TimeSeriesSplit. Требуется минимум {n_splits * 2}, доступно {len(X)}")
                return {
                    'fold_scores': [],
                    'fold_metrics': [],
                    'fold_indices': [],
                    'cv_type': 'sklearn_timeseries',
                    'error': 'Недостаточно данных'
                }
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            results = {
                'fold_scores': [],
                'fold_metrics': [],
                'fold_indices': [],
                'cv_type': 'sklearn_timeseries'
            }
            
            fold = 0
            
            for train_indices, test_indices in tscv.split(X):
                # Разделение данных
                X_train_fold = X.iloc[train_indices]
                y_train_fold = y.iloc[train_indices]
                X_test_fold = X.iloc[test_indices]
                y_test_fold = y.iloc[test_indices]
                
                # Обучение модели
                model.fit(X_train_fold, y_train_fold)
                
                # Прогноз
                y_pred_fold = model.predict(X_test_fold)
                
                # Вычисление метрик
                mae = mean_absolute_error(y_test_fold, y_pred_fold)
                mse = mean_squared_error(y_test_fold, y_pred_fold)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test_fold, y_pred_fold)
                
                fold_metrics = {
                    'fold': fold,
                    'train_start': train_indices[0],
                    'train_end': train_indices[-1],
                    'test_start': test_indices[0],
                    'test_end': test_indices[-1],
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'train_size': len(X_train_fold),
                    'test_size': len(X_test_fold)
                }
                
                results['fold_scores'].append(mae)
                results['fold_metrics'].append(fold_metrics)
                results['fold_indices'].append((train_indices, test_indices))
                
                fold += 1
            
            # Общие метрики
            if results['fold_scores']:
                results['mean_score'] = np.mean(results['fold_scores'])
                results['std_score'] = np.std(results['fold_scores'])
                results['n_folds'] = len(results['fold_scores'])
                
                logger.info(f"Выполнено {results['n_folds']} фолдов")
                logger.info(f"Средний MAE: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
            else:
                logger.warning("Не удалось выполнить ни одного фолда")
                results['mean_score'] = float('inf')
                results['std_score'] = 0
                results['n_folds'] = 0
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в sklearn TimeSeriesSplit CV: {e}")
            return {
                'fold_scores': [],
                'fold_metrics': [],
                'fold_indices': [],
                'cv_type': 'sklearn_timeseries',
                'error': str(e)
            }
    
    def compare_cv_methods(self, X: pd.DataFrame, y: pd.Series, 
                          model_class, model_name: str = 'LinearRegression') -> dict:
        """Сравнение различных методов кросс-валидации."""
        logger.info("=== СРАВНЕНИЕ МЕТОДОВ КРОСС-ВАЛИДАЦИИ ===")
        
        try:
            results = {}
            
            # Скользящее окно
            logger.info("1. Скользящее окно")
            rolling_results = self.rolling_window_cv(
                X, y, model_class(), train_size=100, test_size=20, step=20
            )
            results['rolling_window'] = rolling_results
            
            # Расширяющееся окно
            logger.info("2. Расширяющееся окно")
            expanding_results = self.expanding_window_cv(
                X, y, model_class(), initial_train_size=50, test_size=20, step=20
            )
            results['expanding_window'] = expanding_results
            
            # Sklearn TimeSeriesSplit
            logger.info("3. Sklearn TimeSeriesSplit")
            sklearn_results = self.sklearn_timeseries_cv(
                X, y, model_class(), n_splits=5
            )
            results['sklearn_timeseries'] = sklearn_results
            
            logger.info("Сравнение методов кросс-валидации завершено")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в сравнении методов CV: {e}")
            return {}
    
    def analyze_stability(self, cv_results: dict) -> dict:
        """Анализ стабильности метрик во времени."""
        logger.info("Анализ стабильности метрик...")
        
        try:
            stability_analysis = {}
            
            for method_name, results in cv_results.items():
                if 'error' in results:
                    logger.warning(f"Метод {method_name} содержит ошибки, пропускаем анализ стабильности")
                    continue
                
                fold_metrics = results['fold_metrics']
                
                if not fold_metrics:
                    logger.warning(f"Нет метрик для анализа стабильности в методе {method_name}")
                    continue
                
                # Извлечение метрик по фолдам
                maes = [fold['mae'] for fold in fold_metrics]
                rmses = [fold['rmse'] for fold in fold_metrics]
                r2s = [fold['r2'] for fold in fold_metrics]
                
                # Статистики стабильности
                stability_analysis[method_name] = {
                    'mae_stability': {
                        'mean': np.mean(maes),
                        'std': np.std(maes),
                        'cv': np.std(maes) / (np.mean(maes) + 1e-8),  # Коэффициент вариации
                        'range': np.max(maes) - np.min(maes),
                        'trend': np.polyfit(range(len(maes)), maes, 1)[0] if len(maes) > 1 else 0  # Тренд
                    },
                    'rmse_stability': {
                        'mean': np.mean(rmses),
                        'std': np.std(rmses),
                        'cv': np.std(rmses) / (np.mean(rmses) + 1e-8),
                        'range': np.max(rmses) - np.min(rmses),
                        'trend': np.polyfit(range(len(rmses)), rmses, 1)[0] if len(rmses) > 1 else 0
                    },
                    'r2_stability': {
                        'mean': np.mean(r2s),
                        'std': np.std(r2s),
                        'cv': np.std(r2s) / (np.mean(r2s) + 1e-8),
                        'range': np.max(r2s) - np.min(r2s),
                        'trend': np.polyfit(range(len(r2s)), r2s, 1)[0] if len(r2s) > 1 else 0
                    }
                }
            
            logger.info("Анализ стабильности метрик завершен")
            return stability_analysis
            
        except Exception as e:
            logger.error(f"Ошибка в анализе стабильности: {e}")
            return {}
    
    def visualize_cv_results(self, cv_results: dict) -> None:
        """Визуализация результатов кросс-валидации."""
        logger.info("Создание визуализации результатов CV...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Результаты кросс-валидации временных рядов', fontsize=16)
            
            # График ошибок по фолдам
            colors = ['red', 'blue', 'green']
            for i, (method_name, results) in enumerate(cv_results.items()):
                if 'error' not in results and 'fold_metrics' in results and results['fold_metrics']:
                    fold_numbers = [fold['fold'] for fold in results['fold_metrics']]
                    maes = [fold['mae'] for fold in results['fold_metrics']]
                    
                    axes[0, 0].plot(fold_numbers, maes, 'o-', 
                                  label=f'{method_name.replace("_", " ").title()}', 
                                  color=colors[i], alpha=0.7)
            
            axes[0, 0].set_title('MAE по фолдам')
            axes[0, 0].set_xlabel('Номер фолда')
            axes[0, 0].set_ylabel('MAE')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # График R² по фолдам
            for i, (method_name, results) in enumerate(cv_results.items()):
                if 'error' not in results and 'fold_metrics' in results and results['fold_metrics']:
                    fold_numbers = [fold['fold'] for fold in results['fold_metrics']]
                    r2s = [fold['r2'] for fold in results['fold_metrics']]
                    
                    axes[0, 1].plot(fold_numbers, r2s, 'o-', 
                                  label=f'{method_name.replace("_", " ").title()}', 
                                  color=colors[i], alpha=0.7)
            
            axes[0, 1].set_title('R² по фолдам')
            axes[0, 1].set_xlabel('Номер фолда')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Сравнение средних метрик
            methods = []
            mean_maes = []
            std_maes = []
            
            for method_name, results in cv_results.items():
                if 'error' not in results and 'mean_score' in results:
                    methods.append(method_name)
                    mean_maes.append(results['mean_score'])
                    std_maes.append(results['std_score'])
            
            if methods:
                x = np.arange(len(methods))
                axes[1, 0].bar(x, mean_maes, yerr=std_maes, alpha=0.7, color=colors[:len(methods)])
                axes[1, 0].set_title('Средний MAE по методам')
                axes[1, 0].set_xlabel('Метод CV')
                axes[1, 0].set_ylabel('MAE')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
                
                # Количество фолдов
                n_folds = [cv_results[method]['n_folds'] for method in methods]
                axes[1, 1].bar(methods, n_folds, alpha=0.7, color=colors[:len(methods)])
                axes[1, 1].set_title('Количество фолдов')
                axes[1, 1].set_xlabel('Метод CV')
                axes[1, 1].set_ylabel('Количество фолдов')
                axes[1, 1].set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            logger.info("Визуализация результатов CV создана успешно")
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализации результатов CV: {e}")
    
    def create_summary_table(self, cv_results: dict, stability_analysis: dict) -> pd.DataFrame:
        """Создание сводной таблицы результатов."""
        logger.info("Создание сводной таблицы результатов...")
        
        try:
            summary_data = []
            
            for method_name, results in cv_results.items():
                if 'error' not in results and method_name in stability_analysis:
                    stability = stability_analysis[method_name]
                    
                    summary_data.append({
                        'Метод CV': method_name.replace('_', ' ').title(),
                        'Количество фолдов': results['n_folds'],
                        'Средний MAE': results['mean_score'],
                        'Стд. отклонение MAE': results['std_score'],
                        'Коэффициент вариации MAE': stability['mae_stability']['cv'],
                        'Тренд MAE': stability['mae_stability']['trend'],
                        'Диапазон MAE': stability['mae_stability']['range']
                    })
                else:
                    logger.warning(f"Метод {method_name} содержит ошибки или отсутствует в анализе стабильности, пропускаем в таблице")
            
            if not summary_data:
                logger.warning("Нет данных для создания сводной таблицы")
                return pd.DataFrame()
            
            df = pd.DataFrame(summary_data)
            logger.info(f"Создана сводная таблица с {len(df)} методами")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка создания сводной таблицы: {e}")
            return pd.DataFrame()
    
    def run_comprehensive_cv_analysis(self, file_path: str, target_column: str = 'Weekly_Sales') -> dict:
        """Запуск комплексного анализа кросс-валидации."""
        logger.info("=== КОМПЛЕКСНЫЙ АНАЛИЗ КРОСС-ВАЛИДАЦИИ ===")
        
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
            
            # Сравнение методов CV
            logger.info("Сравнение методов кросс-валидации...")
            cv_results = self.compare_cv_methods(X, y, LinearRegression)
            
            if not cv_results:
                logger.error("Не удалось выполнить сравнение методов CV")
                return {}
            
            # Анализ стабильности
            logger.info("Анализ стабильности метрик...")
            stability_analysis = self.analyze_stability(cv_results)
            
            # Визуализация
            logger.info("Создание визуализации...")
            self.visualize_cv_results(cv_results)
            
            # Создание сводной таблицы
            logger.info("Создание сводной таблицы...")
            summary_df = self.create_summary_table(cv_results, stability_analysis)
            
            if not summary_df.empty:
                logger.info("=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
                print(summary_df.to_string(index=False))
                
                # Сохранение результатов
                summary_df.to_csv('time_series_cv_results.csv', index=False, encoding='utf-8')
                logger.info("Результаты сохранены в time_series_cv_results.csv")
            
            final_results = {
                'cv_results': cv_results,
                'stability_analysis': stability_analysis,
                'summary': summary_df.to_dict('records') if not summary_df.empty else []
            }
            
            with open('time_series_cv_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("Результаты сохранены:")
            logger.info("- time_series_cv_results.csv")
            logger.info("- time_series_cv_analysis.json")
            logger.info("Комплексный анализ кросс-валидации завершен успешно")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Ошибка в комплексном анализе кросс-валидации: {e}")
            return {}


def main():
    """Основная функция для демонстрации работы модуля."""
    logger.info("Запуск демонстрации модуля кросс-валидации временных рядов")
    
    try:
        cv_analyzer = TimeSeriesCrossValidation()
        
        # Запуск анализа
        results = cv_analyzer.run_comprehensive_cv_analysis(
            "../New_final.csv", 
            "Weekly_Sales"
        )
        
        if results:
            logger.info("=== АНАЛИЗ КРОСС-ВАЛИДАЦИИ ЗАВЕРШЕН ===")
            logger.info("Сравнены три метода кросс-валидации для временных рядов")
        else:
            logger.error("Анализ не удался. Проверьте путь к файлу.")
            
    except Exception as e:
        logger.error(f"Ошибка в основной функции: {e}")


if __name__ == "__main__":
    main()

