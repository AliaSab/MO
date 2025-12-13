"""
Модуль для интеграции AutoGluon-TimeSeries в пайплайн.
Поддерживает multiple presets, backtesting, leaderboard и сравнение с кастомными моделями.
"""

import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

# Попытка импорта AutoGluon
try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("⚠️ AutoGluon не установлен. Для установки: pip install autogluon.timeseries")


class AutoGluonWrapper:
    """Обёртка для удобной работы с AutoGluon-TimeSeries."""
    
    def __init__(self, prediction_length=7, eval_metric="MAE", freq="W"):
        """
        Инициализация.
        
        Parameters:
        -----------
        prediction_length : int
            Горизонт прогнозирования
        eval_metric : str
            Метрика оценки ('MAE', 'MAPE', 'MASE', 'RMSE', 'SMAPE')
        freq : str
            Частота временного ряда ('W' - неделя, 'D' - день, 'H' - час)
        """
        if not AUTOGLUON_AVAILABLE:
            raise ImportError("AutoGluon не установлен")
        
        self.prediction_length = prediction_length
        self.eval_metric = eval_metric
        self.freq = freq
        self.predictors = {}  # Хранит обученные предикторы для каждого пресета
        self.training_times = {}
        self.leaderboards = {}
    
    def prepare_data(self, series, date_index, item_id='series_1'):
        """
        Преобразует pandas Series в TimeSeriesDataFrame для AutoGluon.
        
        Parameters:
        -----------
        series : pd.Series or np.ndarray
            Временной ряд
        date_index : pd.DatetimeIndex
            Временные метки
        item_id : str
            Идентификатор ряда
        
        Returns:
        --------
        TimeSeriesDataFrame
        """
        if isinstance(series, pd.Series):
            values = series.values
        else:
            values = series
        
        df = pd.DataFrame({
            'item_id': item_id,
            'timestamp': date_index,
            'target': values
        })
        
        # Создаем TimeSeriesDataFrame
        ts_dataframe = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column='item_id',
            timestamp_column='timestamp'
        )
        
        return ts_dataframe
    
    def fit_with_preset(self, train_data, preset="medium_quality", 
                       time_limit=None, verbosity=2):
        """
        Обучает AutoGluon с заданным пресетом.
        
        Parameters:
        -----------
        train_data : TimeSeriesDataFrame
            Обучающие данные
        preset : str
            Пресет качества:
            - 'fast_training': быстрое обучение
            - 'medium_quality': средний баланс скорость/качество (по умолчанию)
            - 'good_quality': хорошее качество
            - 'high_quality': высокое качество
            - 'best_quality': максимальное качество
        time_limit : int, optional
            Ограничение времени в секундах
        verbosity : int
            Уровень детализации (0-4)
        
        Returns:
        --------
        TimeSeriesPredictor
        """
        print(f"\n🚀 Обучение AutoGluon с пресетом: {preset}")
        print(f"   Горизонт: {self.prediction_length}, Метрика: {self.eval_metric}")
        
        predictor = TimeSeriesPredictor(
            target='target',
            prediction_length=self.prediction_length,
            eval_metric=self.eval_metric,
            verbosity=verbosity
        )
        
        start_time = time.time()
        
        fit_kwargs = {
            'train_data': train_data,
            'presets': preset,
            'time_limit': time_limit
        }
        
        predictor.fit(**fit_kwargs)
        
        elapsed = time.time() - start_time
        
        self.predictors[preset] = predictor
        self.training_times[preset] = elapsed
        
        # Сохраняем leaderboard
        try:
            leaderboard = predictor.leaderboard(train_data, silent=True)
            self.leaderboards[preset] = leaderboard
            print(f"\n✅ Обучение завершено за {elapsed:.2f} сек")
            print(f"   Лучшая модель: {leaderboard.iloc[0]['model']}")
            print(f"   Лучший score: {leaderboard.iloc[0]['score_val']:.6f}")
        except Exception as e:
            print(f"⚠️ Не удалось получить leaderboard: {e}")
        
        return predictor
    
    def fit_multiple_presets(self, train_data, presets=None, time_limit_per_preset=None):
        """
        Обучает AutoGluon с несколькими пресетами для сравнения.
        
        Parameters:
        -----------
        train_data : TimeSeriesDataFrame
            Обучающие данные
        presets : list, optional
            Список пресетов (по умолчанию: medium, high, best_quality)
        time_limit_per_preset : int, optional
            Ограничение времени на каждый пресет
        
        Returns:
        --------
        dict
            Словарь {preset: predictor}
        """
        if presets is None:
            presets = ["medium_quality", "high_quality", "best_quality"]
        
        print(f"\n{'='*80}")
        print(f"ОБУЧЕНИЕ AUTOGLUON С МНОЖЕСТВЕННЫМИ ПРЕСЕТАМИ")
        print(f"{'='*80}")
        print(f"Пресеты: {presets}")
        
        for preset in presets:
            try:
                self.fit_with_preset(
                    train_data, 
                    preset=preset, 
                    time_limit=time_limit_per_preset,
                    verbosity=2
                )
            except Exception as e:
                print(f"\n❌ Ошибка при обучении пресета {preset}: {e}")
        
        return self.predictors
    
    def predict(self, test_data=None, preset="medium_quality", quantile_levels=None):
        """
        Делает прогноз.
        
        Parameters:
        -----------
        test_data : TimeSeriesDataFrame, optional
            Тестовые данные (если None, прогноз из последней точки обучения)
        preset : str
            Какой пресет использовать
        quantile_levels : list, optional
            Уровни квантилей для доверительных интервалов (например, [0.1, 0.9])
        
        Returns:
        --------
        pd.DataFrame
            Прогнозы
        """
        if preset not in self.predictors:
            raise ValueError(f"Пресет {preset} не обучен. Доступные: {list(self.predictors.keys())}")
        
        predictor = self.predictors[preset]
        
        if test_data is not None:
            predictions = predictor.predict(test_data, quantile_levels=quantile_levels)
        else:
            predictions = predictor.predict(quantile_levels=quantile_levels)
        
        return predictions
    
    def get_leaderboard(self, preset="medium_quality", data=None):
        """
        Возвращает таблицу лидеров для заданного пресета.
        
        Parameters:
        -----------
        preset : str
            Пресет
        data : TimeSeriesDataFrame, optional
            Данные для оценки (если None, используется train)
        
        Returns:
        --------
        pd.DataFrame
        """
        if preset not in self.predictors:
            raise ValueError(f"Пресет {preset} не обучен")
        
        predictor = self.predictors[preset]
        
        if data is not None:
            leaderboard = predictor.leaderboard(data, silent=True)
        else:
            leaderboard = self.leaderboards.get(preset, None)
        
        return leaderboard
    
    def backtesting(self, full_data, num_windows=3, preset="medium_quality"):
        """
        Выполняет backtesting валидацию.
        
        Parameters:
        -----------
        full_data : TimeSeriesDataFrame
            Полные данные для валидации
        num_windows : int
            Количество окон для backtesting
        preset : str
            Пресет для использования
        
        Returns:
        --------
        pd.DataFrame
            Результаты backtesting
        """
        print(f"\n🔄 Backtesting с {num_windows} окнами (пресет: {preset})...")
        
        if preset not in self.predictors:
            raise ValueError(f"Пресет {preset} не обучен")
        
        predictor = self.predictors[preset]
        
        # Используем встроенный backtesting
        results = []
        data_len = len(full_data)
        window_size = data_len // (num_windows + 1)
        
        for i in range(num_windows):
            train_end = window_size * (i + 1)
            test_start = train_end
            test_end = test_start + self.prediction_length
            
            if test_end > data_len:
                break
            
            # Разбиваем данные
            train_slice = full_data.iloc[:train_end]
            test_slice = full_data.iloc[test_start:test_end]
            
            # Переобучаем на новых данных
            temp_predictor = TimeSeriesPredictor(
                target='target',
                prediction_length=self.prediction_length,
                eval_metric=self.eval_metric,
                verbosity=0
            )
            temp_predictor.fit(train_slice, presets=preset)
            
            # Прогноз
            predictions = temp_predictor.predict(train_slice)
            
            # Оценка
            # Извлекаем значения из прогноза и факта
            pred_values = predictions['mean'].values if 'mean' in predictions.columns else predictions.values
            actual_values = test_slice['target'].values
            
            mae = np.mean(np.abs(pred_values[:len(actual_values)] - actual_values))
            rmse = np.sqrt(np.mean((pred_values[:len(actual_values)] - actual_values) ** 2))
            
            results.append({
                'window': i + 1,
                'train_end': train_end,
                'test_end': test_end,
                'MAE': mae,
                'RMSE': rmse
            })
            
            print(f"  Окно {i+1}/{num_windows}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self, preset="medium_quality"):
        """
        Возвращает важность признаков (если доступно).
        
        Parameters:
        -----------
        preset : str
            Пресет
        
        Returns:
        --------
        pd.DataFrame or None
        """
        if preset not in self.predictors:
            return None
        
        predictor = self.predictors[preset]
        
        try:
            importance = predictor.feature_importance()
            return importance
        except:
            print(f"⚠️ Feature importance недоступна для пресета {preset}")
            return None
    
    def compare_presets(self):
        """
        Сравнивает результаты всех обученных пресетов.
        
        Returns:
        --------
        pd.DataFrame
            Сравнительная таблица
        """
        if not self.predictors:
            print("⚠️ Нет обученных пресетов")
            return None
        
        results = []
        
        for preset in self.predictors.keys():
            leaderboard = self.leaderboards.get(preset)
            
            if leaderboard is not None and len(leaderboard) > 0:
                best_model = leaderboard.iloc[0]
                
                results.append({
                    'preset': preset,
                    'best_model': best_model['model'],
                    'score_val': best_model['score_val'],
                    'training_time': self.training_times.get(preset, np.nan),
                    'n_models': len(leaderboard)
                })
        
        comparison = pd.DataFrame(results)
        comparison = comparison.sort_values('score_val')
        
        return comparison
    
    def save_predictor(self, preset="medium_quality", path="autogluon_model"):
        """
        Сохраняет обученный предиктор.
        
        Parameters:
        -----------
        preset : str
            Пресет для сохранения
        path : str
            Путь для сохранения
        """
        if preset not in self.predictors:
            raise ValueError(f"Пресет {preset} не обучен")
        
        predictor = self.predictors[preset]
        predictor.save(path)
        print(f"✅ Предиктор {preset} сохранен в {path}")
    
    def load_predictor(self, path, preset="loaded"):
        """
        Загружает сохраненный предиктор.
        
        Parameters:
        -----------
        path : str
            Путь к сохраненной модели
        preset : str
            Название для загруженного пресета
        """
        predictor = TimeSeriesPredictor.load(path)
        self.predictors[preset] = predictor
        print(f"✅ Предиктор загружен как '{preset}'")
        return predictor


def compare_autogluon_vs_custom(autogluon_preds, custom_preds, y_true, 
                                 autogluon_time, custom_time):
    """
    Сравнивает AutoGluon с кастомными моделями.
    
    Parameters:
    -----------
    autogluon_preds : dict
        Словарь {preset: predictions} для AutoGluon
    custom_preds : dict
        Словарь {model_name: predictions} для кастомных моделей
    y_true : array-like
        Истинные значения
    autogluon_time : dict
        Время обучения AutoGluon {preset: time}
    custom_time : float
        Общее время обучения кастомных моделей
    
    Returns:
    --------
    pd.DataFrame
        Сравнительная таблица
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    results = []
    
    # AutoGluon модели
    for preset, preds in autogluon_preds.items():
        # Извлекаем mean predictions если это DataFrame
        if isinstance(preds, pd.DataFrame):
            pred_values = preds['mean'].values if 'mean' in preds.columns else preds.values.flatten()
        else:
            pred_values = preds
        
        # Выравниваем длины
        min_len = min(len(pred_values), len(y_true))
        pred_values = pred_values[:min_len]
        y_true_slice = y_true[:min_len] if hasattr(y_true, '__getitem__') else y_true
        
        mae = mean_absolute_error(y_true_slice, pred_values)
        rmse = np.sqrt(mean_squared_error(y_true_slice, pred_values))
        
        results.append({
            'model': f'AutoGluon_{preset}',
            'type': 'AutoML',
            'MAE': mae,
            'RMSE': rmse,
            'training_time': autogluon_time.get(preset, np.nan),
            'interpretability': 'Low',
            'flexibility': 'Low',
            'automation': 'High'
        })
    
    # Кастомные модели (берем топ-5 лучших по MAE)
    custom_results = []
    for model_name, preds in custom_preds.items():
        if len(preds) > 0:
            min_len = min(len(preds), len(y_true))
            pred_slice = preds[:min_len]
            y_true_slice = y_true[:min_len]
            
            mae = mean_absolute_error(y_true_slice, pred_slice)
            rmse = np.sqrt(mean_squared_error(y_true_slice, pred_slice))
            
            custom_results.append({
                'model': model_name,
                'type': 'Custom',
                'MAE': mae,
                'RMSE': rmse
            })
    
    # Берем топ-5 кастомных моделей
    custom_results = sorted(custom_results, key=lambda x: x['MAE'])[:5]
    
    for cr in custom_results:
        results.append({
            'model': cr['model'],
            'type': 'Custom',
            'MAE': cr['MAE'],
            'RMSE': cr['RMSE'],
            'training_time': custom_time / len(custom_preds),  # Усредненное время
            'interpretability': 'High' if 'Linear' in cr['model'] or 'Ridge' in cr['model'] else 'Medium',
            'flexibility': 'High',
            'automation': 'Low'
        })
    
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values('MAE')
    
    return comparison


def create_autogluon_recommendations(comparison_df):
    """
    Создает рекомендации по использованию AutoGluon vs кастомных моделей.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Результаты сравнения
    
    Returns:
    --------
    dict
        Рекомендации
    """
    # Лучшая AutoGluon модель
    ag_models = comparison_df[comparison_df['type'] == 'AutoML']
    best_ag = ag_models.iloc[0] if len(ag_models) > 0 else None
    
    # Лучшая кастомная модель
    custom_models = comparison_df[comparison_df['type'] == 'Custom']
    best_custom = custom_models.iloc[0] if len(custom_models) > 0 else None
    
    recommendations = {
        'summary': {},
        'use_autogluon_when': [],
        'use_custom_when': [],
        'production_strategy': {},
        'retraining_template': {}
    }
    
    if best_ag is not None and best_custom is not None:
        mae_diff = ((best_custom['MAE'] - best_ag['MAE']) / best_custom['MAE']) * 100
        time_diff = best_ag['training_time'] - best_custom['training_time']
        
        recommendations['summary'] = {
            'best_autogluon_model': best_ag['model'],
            'best_custom_model': best_custom['model'],
            'autogluon_mae_advantage': f"{-mae_diff:.2f}%" if mae_diff < 0 else f"{mae_diff:.2f}% worse",
            'autogluon_time': f"{best_ag['training_time']:.2f} sec",
            'custom_time': f"{best_custom['training_time']:.2f} sec"
        }
        
        # Рекомендации когда использовать AutoGluon
        recommendations['use_autogluon_when'].extend([
            "Нужен быстрый MVP или proof-of-concept",
            "Ограничены ресурсы на feature engineering",
            "Требуется автоматический подбор гиперпараметров",
            "Нужны встроенные ансамбли и продвинутые модели (DeepAR, TFT)"
        ])
        
        if mae_diff < -5:  # AutoGluon значительно лучше
            recommendations['use_autogluon_when'].append(
                "AutoGluon показал лучшее качество (>5%) - рекомендуется для продакшена"
            )
        
        # Рекомендации когда использовать кастомные модели
        recommendations['use_custom_when'].extend([
            "Требуется полная интерпретируемость (LIME, SHAP)",
            "Нужен контроль над каждым этапом пайплайна",
            "Специфические требования к обработке выбросов",
            "Ограничения по памяти/размеру модели"
        ])
        
        if mae_diff > 5:  # Кастомные модели лучше
            recommendations['use_custom_when'].append(
                "Кастомные модели показали лучшее качество (>5%)"
            )
        
        # Стратегия для продакшена
        if abs(mae_diff) < 5:  # Модели примерно равны
            recommendations['production_strategy'] = {
                'approach': 'Hybrid',
                'description': 'Использовать AutoGluon для MVP, затем перейти к кастомным при необходимости',
                'phase_1': 'AutoGluon для быстрого запуска',
                'phase_2': 'Анализ feature importance из AutoGluon',
                'phase_3': 'Разработка кастомных моделей на основе инсайтов',
                'phase_4': 'A/B тестирование обоих подходов'
            }
        elif mae_diff < -5:
            recommendations['production_strategy'] = {
                'approach': 'AutoGluon-first',
                'description': 'AutoGluon показал лучшее качество - использовать в продакшене',
                'monitoring': 'Отслеживать деградацию метрик',
                'fallback': 'Держать кастомные модели как backup'
            }
        else:
            recommendations['production_strategy'] = {
                'approach': 'Custom-first',
                'description': 'Кастомные модели показали лучшее качество',
                'optimization': 'Продолжить тюнинг гиперпараметров',
                'ensemble': 'Рассмотреть ансамблирование с AutoGluon'
            }
        
        # Шаблон переобучения
        recommendations['retraining_template'] = {
            'frequency': 'Weekly or when MASE degrades >10%',
            'steps': [
                'Load new data',
                'Update feature engineering pipeline',
                'Retrain AutoGluon with best preset',
                'Retrain top-3 custom models',
                'Compare on validation set',
                'Deploy best model',
                'Monitor MASE/MAE on production data'
            ],
            'boxcox_recalibration': 'Recalculate lambda on expanded training set',
            'feature_selection': 'Review feature importance quarterly'
        }
    
    return recommendations




