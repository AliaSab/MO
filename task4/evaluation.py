"""
Этап 7: Оценка качества
Метрики (MAE, RMSE, MAPE, MASE, RMSSE), Diebold-Mariano тест.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Класс для расчета метрик качества прогнозирования."""
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        if mask.sum() == 0:
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def mase(y_true, y_pred, y_train, seasonality=1):
        """
        Mean Absolute Scaled Error.
        
        Parameters:
        -----------
        y_true : array-like
            Фактические значения
        y_pred : array-like
            Предсказанные значения
        y_train : array-like
            Обучающие данные (для naive прогноза)
        seasonality : int
            Сезонность для seasonal naive
        """
        # Преобразуем в numpy массивы для надежности
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        y_train = np.asarray(y_train)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        mae_forecast = MetricsCalculator.mae(y_true, y_pred)
        
        # Проверяем, что y_train не пустой и не все значения одинаковые
        if len(y_train) < 2:
            return np.nan
        
        # Naive прогноз на обучающих данных
        if seasonality == 1:
            # Для seasonality=1 используем разности соседних значений
            naive_forecast = np.abs(np.diff(y_train))
        else:
            # Для seasonal naive используем разности с сезонным лагом
            if len(y_train) <= seasonality:
                # Если данных недостаточно, используем простой naive
                naive_forecast = np.abs(np.diff(y_train))
            else:
                naive_forecast = np.abs(y_train[seasonality:] - y_train[:-seasonality])
        
        # Проверяем, что есть различия в данных
        if len(naive_forecast) == 0:
            return np.nan
        
        mae_naive = np.mean(naive_forecast)
        
        # Если mae_naive слишком маленький (близок к нулю), значит данные почти постоянные
        # В этом случае используем альтернативный расчет
        if mae_naive < 1e-10:
            # Если данные почти постоянные, используем среднее абсолютное значение как масштаб
            mean_abs_value = np.mean(np.abs(y_train))
            if mean_abs_value < 1e-10:
                return np.nan
            return mae_forecast / mean_abs_value
        
        return mae_forecast / mae_naive
    
    @staticmethod
    def rmsse(y_true, y_pred, y_train, seasonality=1):
        """
        Root Mean Squared Scaled Error.
        
        Parameters:
        -----------
        y_true : array-like
            Фактические значения
        y_pred : array-like
            Предсказанные значения
        y_train : array-like
            Обучающие данные (для naive прогноза)
        seasonality : int
            Сезонность для seasonal naive
        """
        # Преобразуем в numpy массивы для надежности
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        y_train = np.asarray(y_train)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        rmse_forecast = MetricsCalculator.rmse(y_true, y_pred)
        
        # Проверяем, что y_train не пустой
        if len(y_train) < 2:
            return np.nan
        
        # Naive прогноз на обучающих данных
        if seasonality == 1:
            naive_forecast = np.diff(y_train) ** 2
        else:
            if len(y_train) <= seasonality:
                naive_forecast = np.diff(y_train) ** 2
            else:
                naive_forecast = (y_train[seasonality:] - y_train[:-seasonality]) ** 2
        
        if len(naive_forecast) == 0:
            return np.nan
        
        rmse_naive = np.sqrt(np.mean(naive_forecast))
        
        # Если rmse_naive слишком маленький, используем альтернативный расчет
        if rmse_naive < 1e-10:
            std_value = np.std(y_train)
            if std_value < 1e-10:
                return np.nan
            return rmse_forecast / std_value
        
        return rmse_forecast / rmse_naive
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, y_train=None, seasonality=1):
        """
        Вычисляет все метрики.
        
        Parameters:
        -----------
        y_true : array-like
            Фактические значения
        y_pred : array-like
            Предсказанные значения
        y_train : array-like, optional
            Обучающие данные (для MASE и RMSSE)
        seasonality : int
            Сезонность
        
        Returns:
        --------
        dict
            Словарь метрик
        """
        metrics = {
            'MAE': MetricsCalculator.mae(y_true, y_pred),
            'RMSE': MetricsCalculator.rmse(y_true, y_pred),
            'MAPE': MetricsCalculator.mape(y_true, y_pred)
        }
        
        if y_train is not None:
            metrics['MASE'] = MetricsCalculator.mase(y_true, y_pred, y_train, seasonality)
            metrics['RMSSE'] = MetricsCalculator.rmsse(y_true, y_pred, y_train, seasonality)
        
        return metrics


class DieboldMarianoTest:
    """Diebold-Mariano тест для сравнения прогнозов."""
    
    @staticmethod
    def dm_test(forecast_A, forecast_B, actual, h=1, test='two_sided', power=2):
        """
        Diebold-Mariano тест для сравнения двух прогнозов.
        
        Parameters:
        -----------
        forecast_A : array-like
            Прогноз модели A
        forecast_B : array-like
            Прогноз модели B
        actual : array-like
            Фактические значения
        h : int
            Горизонт прогнозирования
        test : str
            Тип теста: 'two_sided', 'less', 'greater'
        power : int
            Степень функции потерь (1 для MAE, 2 для MSE)
        
        Returns:
        --------
        dict
            Результаты теста
        """
        # Преобразуем в массивы
        actual = np.array(actual)
        forecast_A = np.array(forecast_A)
        forecast_B = np.array(forecast_B)
        
        # Вычисляем ошибки прогнозов
        error_A = actual - forecast_A
        error_B = actual - forecast_B
        
        # Функция потерь
        if power == 1:
            d = np.abs(error_A) - np.abs(error_B)
        elif power == 2:
            d = error_A ** 2 - error_B ** 2
        else:
            d = np.abs(error_A) ** power - np.abs(error_B) ** power
        
        # Проверяем, что есть данные
        if len(d) == 0:
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'result': 'Недостаточно данных',
                'd_mean': np.nan
            }
        
        # Среднее значение d
        d_mean = np.mean(d)
        n = len(d)
        
        # Проверяем минимальный размер выборки
        if n < 2:
            return {
                'statistic': np.nan,
                'pvalue': np.nan,
                'result': 'Недостаточно данных (n < 2)',
                'd_mean': d_mean
            }
        
        # Вычисляем автоковариацию для учета автокорреляции (важно для временных рядов)
        gamma = []
        max_lag = min(h - 1, n - 1, 10)  # Ограничиваем максимальный лаг
        
        if max_lag <= 0 or n < 3:
            # Если нет лагов или мало данных, используем простую дисперсию
            var_d = np.var(d, ddof=1) if n > 1 else 1e-10
        else:
            for k in range(max_lag + 1):
                if k == 0:
                    gamma_k = np.var(d, ddof=1) if n > 1 else 0
                else:
                    if len(d) > k:
                        # Автоковариация
                        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
                    else:
                        gamma_k = 0
                gamma.append(gamma_k)
            
            # Оценка дисперсии с учетом автокорреляции (Newey-West style)
            if len(gamma) > 0:
                # Используем только положительные лаги для стабильности
                var_d = gamma[0] + 2 * sum([g for g in gamma[1:] if g > 0]) if len(gamma) > 1 else gamma[0]
            else:
                var_d = np.var(d, ddof=1) if n > 1 else 1e-10
        
        # Ограничиваем var_d разумными значениями
        var_d = max(var_d, 1e-10)
        if var_d > 1e10:
            var_d = np.var(d, ddof=1) if n > 1 else 1e-10
        
        # DM статистика
        try:
            dm_stat = d_mean / np.sqrt(var_d / n)
            
            # p-value
            if test == 'two_sided':
                pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
            elif test == 'less':
                pvalue = stats.norm.cdf(dm_stat)
            else:  # 'greater'
                pvalue = 1 - stats.norm.cdf(dm_stat)
            
            # Интерпретация
            if pvalue < 0.05:
                if test == 'two_sided':
                    result = 'Значимое различие'
                elif test == 'less':
                    result = 'A лучше B'
                else:
                    result = 'B лучше A'
            else:
                result = 'Нет значимого различия'
                
        except (ZeroDivisionError, ValueError) as e:
            dm_stat = np.nan
            pvalue = np.nan
            result = f'Ошибка вычисления: {e}'
        
        return {
            'statistic': dm_stat,
            'pvalue': pvalue,
            'result': result,
            'd_mean': d_mean
        }


class ModelEvaluator:
    """Класс для оценки моделей."""
    
    def __init__(self, y_train=None, seasonality=1):
        """
        Parameters:
        -----------
        y_train : array-like, optional
            Обучающие данные (для MASE и RMSSE)
        seasonality : int
            Сезонность
        """
        self.y_train = y_train
        self.seasonality = seasonality
        self.metrics_calc = MetricsCalculator()
        self.dm_test = DieboldMarianoTest()
    
    def evaluate_model(self, y_true, y_pred, model_name='model'):
        """
        Оценивает модель по всем метрикам.
        
        Parameters:
        -----------
        y_true : array-like
            Фактические значения
        y_pred : array-like
            Предсказанные значения
        model_name : str
            Название модели
        
        Returns:
        --------
        dict
            Метрики модели
        """
        metrics = self.metrics_calc.calculate_all_metrics(
            y_true, y_pred, self.y_train, self.seasonality
        )
        metrics['model'] = model_name
        
        return metrics
    
    def evaluate_all_models(self, y_true, predictions_dict):
        """
        Оценивает все модели.
        
        Parameters:
        -----------
        y_true : array-like
            Фактические значения
        predictions_dict : dict
            Словарь {название_модели: предсказания}
        
        Returns:
        --------
        pd.DataFrame
            Таблица метрик
        """
        results = []
        
        for model_name, y_pred in predictions_dict.items():
            metrics = self.evaluate_model(y_true, y_pred, model_name)
            results.append(metrics)
        
        df_results = pd.DataFrame(results)
        
        # Сортируем по MASE (основной метрике)
        if 'MASE' in df_results.columns:
            df_results = df_results.sort_values('MASE')
        
        return df_results
    
    def compare_models_dm(self, y_true, predictions_dict, test='two_sided', h=1):
        """
        Сравнивает модели через Diebold-Mariano тест.
        
        Parameters:
        -----------
        y_true : array-like
            Фактические значения
        predictions_dict : dict
            Словарь {название_модели: предсказания}
        test : str
            Тип теста
        h : int
            Горизонт прогнозирования (важно для учета автокорреляции)
        
        Returns:
        --------
        pd.DataFrame
            Матрица сравнений
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        # Создаем матрицу результатов
        dm_results = pd.DataFrame(
            index=model_names,
            columns=model_names
        )
        
        for i, model_A in enumerate(model_names):
            for j, model_B in enumerate(model_names):
                if i == j:
                    dm_results.loc[model_A, model_B] = '-'
                else:
                    result = self.dm_test.dm_test(
                        predictions_dict[model_A],
                        predictions_dict[model_B],
                        y_true,
                        h=h,
                        test=test
                    )
                    # Проверяем на NaN
                    if pd.isna(result['pvalue']):
                        dm_results.loc[model_A, model_B] = 'N/A'
                    else:
                        dm_results.loc[model_A, model_B] = f"{result['pvalue']:.4f}"
        
        return dm_results
    
    def create_summary_table(self, y_true, predictions_dict, train_times=None, 
                           autogluon_rank=None):
        """
        Создает сводную таблицу результатов.
        
        Parameters:
        -----------
        y_true : array-like
            Фактические значения
        predictions_dict : dict
            Словарь {название_модели: предсказания}
        train_times : dict, optional
            Словарь времени обучения
        autogluon_rank : dict, optional
            Ранги AutoGluon моделей
        
        Returns:
        --------
        pd.DataFrame
            Сводная таблица
        """
        # Оцениваем все модели
        df_metrics = self.evaluate_all_models(y_true, predictions_dict)
        
        # Добавляем время обучения
        if train_times:
            df_metrics['train_time'] = df_metrics['model'].map(train_times)
        
        # Добавляем ранги AutoGluon
        if autogluon_rank:
            df_metrics['autogluon_rank'] = df_metrics['model'].map(autogluon_rank)
        
        # Сортируем по MASE
        if 'MASE' in df_metrics.columns:
            df_metrics = df_metrics.sort_values('MASE')
        
        return df_metrics

