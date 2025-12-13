"""
Модуль для оценки качества моделей.
Включает метрики (MAE, RMSE, MAPE, MASE и др.) и статистические тесты (Diebold-Mariano, Wilcoxon).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Класс для расчета метрик качества прогноза."""
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error."""
        errors = np.abs(y_true - y_pred)
        # Фильтруем NaN и Inf
        errors = errors[np.isfinite(errors)]
        if len(errors) == 0:
            return np.nan
        return np.mean(errors)
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error."""
        errors = (y_true - y_pred) ** 2
        # Фильтруем NaN и Inf
        errors = errors[np.isfinite(errors)]
        if len(errors) == 0:
            return np.nan
        return np.sqrt(np.mean(errors))
    
    @staticmethod
    def mape(y_true, y_pred, epsilon=1e-8):
        """Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    @staticmethod
    def mase(y_true, y_pred, y_train, seasonality=1):
        """
        Mean Absolute Scaled Error.
        MASE = MAE / (1/n * sum(|y_train[t] - y_train[t-seasonality]|))
        """
        if len(y_train) < seasonality:
            seasonality = 1
        
        # Наивный прогноз на основе сезонности
        naive_errors = np.abs(np.diff(y_train, n=seasonality))
        if len(naive_errors) == 0:
            naive_errors = np.abs(y_train - np.mean(y_train))
        
        scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        if scale == 0:
            scale = 1.0
        
        mae = MetricsCalculator.mae(y_true, y_pred)
        return mae / scale
    
    @staticmethod
    def rmsle(y_true, y_pred, epsilon=1e-8):
        """Root Mean Squared Logarithmic Error."""
        y_true_log = np.log(y_true + epsilon)
        y_pred_log = np.log(y_pred + epsilon)
        return np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))
    
    @staticmethod
    def smape(y_true, y_pred, epsilon=1e-8):
        """Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred) + epsilon) / 2
        return np.mean(numerator / denominator) * 100
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """R-squared score."""
        # Проверяем на константные значения
        if len(y_true) == 0:
            return np.nan
        
        y_true_mean = np.mean(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true_mean) ** 2)
        
        # Если все значения y_true одинаковые (ss_tot = 0), R2 не определен
        if ss_tot < 1e-10:
            return np.nan
        
        r2 = 1 - (ss_res / ss_tot)
        
        # Ограничиваем R2 в разумных пределах (может быть отрицательным)
        return max(-10.0, min(10.0, r2))
    
    @staticmethod
    def mda(y_true, y_pred):
        """Mean Directional Accuracy."""
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return np.mean(true_direction == pred_direction) * 100
    
    def calculate_all_metrics(self, y_true, y_pred, y_train=None, seasonality=7):
        """Вычисляет все метрики."""
        metrics = {
            'MAE': self.mae(y_true, y_pred),
            'RMSE': self.rmse(y_true, y_pred),
            'MAPE': self.mape(y_true, y_pred),
            'SMAPE': self.smape(y_true, y_pred),
            'RMSLE': self.rmsle(y_true, y_pred),
            'R2': self.r2_score(y_true, y_pred),
            'MDA': self.mda(y_true, y_pred),
        }
        
        if y_train is not None:
            metrics['MASE'] = self.mase(y_true, y_pred, y_train, seasonality)
        
        return metrics


class DieboldMarianoTest:
    """Тест Diebold-Mariano для сравнения моделей."""
    
    @staticmethod
    def test(y_true, y_pred1, y_pred2, h=1, power=2):
        """
        Тест Diebold-Mariano.
        
        Args:
            y_true: истинные значения
            y_pred1: прогнозы первой модели
            y_pred2: прогнозы второй модели
            h: горизонт прогноза
            power: степень для функции потерь (1 для MAE, 2 для MSE)
        """
        # Ошибки прогнозов
        e1 = y_true - y_pred1
        e2 = y_true - y_pred2
        
        # Функция потерь
        if power == 1:
            d = np.abs(e1) - np.abs(e2)
        else:
            d = e1 ** 2 - e2 ** 2
        
        # Среднее и стандартное отклонение
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        
        if d_var == 0:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False
            }
        
        # Статистика теста
        n = len(d)
        dm_stat = d_mean / np.sqrt(d_var / n)
        
        # p-value (двусторонний тест)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        
        return {
            'statistic': dm_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_diff': d_mean
        }


class WilcoxonTest:
    """Тест Wilcoxon signed-rank для непараметрического сравнения."""
    
    @staticmethod
    def test(y_true, y_pred1, y_pred2, alternative='two-sided'):
        """
        Тест Wilcoxon signed-rank.
        
        Args:
            y_true: истинные значения
            y_pred1: прогнозы первой модели
            y_pred2: прогнозы второй модели
            alternative: 'two-sided', 'greater', 'less'
        """
        # Ошибки
        e1 = np.abs(y_true - y_pred1)
        e2 = np.abs(y_true - y_pred2)
        
        # Разности
        d = e1 - e2
        
        # Удаляем нули
        d = d[d != 0]
        
        if len(d) == 0:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False
            }
        
        # Тест Wilcoxon
        statistic, p_value = stats.wilcoxon(e1, e2, alternative=alternative)
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


class ModelEvaluator:
    """Класс для комплексной оценки моделей."""
    
    def __init__(self):
        self.metrics_calc = MetricsCalculator()
        self.dm_test = DieboldMarianoTest()
        self.wilcoxon_test = WilcoxonTest()
    
    def evaluate_model(self, y_true, y_pred, y_train=None, seasonality=7):
        """Оценивает одну модель."""
        return self.metrics_calc.calculate_all_metrics(y_true, y_pred, y_train, seasonality)
    
    def compare_models(self, y_true, y_pred1, y_pred2, y_train=None, seasonality=7):
        """Сравнивает две модели."""
        metrics1 = self.evaluate_model(y_true, y_pred1, y_train, seasonality)
        metrics2 = self.evaluate_model(y_true, y_pred2, y_train, seasonality)
        
        # Статистические тесты
        dm_result = self.dm_test.test(y_true, y_pred1, y_pred2, h=1, power=2)
        wilcoxon_result = self.wilcoxon_test.test(y_true, y_pred1, y_pred2)
        
        return {
            'model1_metrics': metrics1,
            'model2_metrics': metrics2,
            'diebold_mariano': dm_result,
            'wilcoxon': wilcoxon_result,
            'improvement': {
                metric: metrics2[metric] - metrics1[metric] 
                for metric in metrics1.keys()
            }
        }
    
    def create_comparison_table(self, results_dict, sort_by='MASE'):
        """
        Создает сводную таблицу сравнения моделей.
        
        Args:
            results_dict: словарь {model_name: {'metrics': {...}, 'time': ..., 'memory': ...}}
            sort_by: метрика для сортировки
        """
        rows = []
        for model_name, result in results_dict.items():
            metrics = result.get('metrics', {})
            
            # Получаем значения метрик с обработкой NaN
            def format_metric(value, default=np.nan):
                if value is None:
                    return default
                if isinstance(value, (int, float)) and np.isnan(value):
                    return np.nan
                return value
            
            row = {
                'Модель': model_name,
                'MAE': format_metric(metrics.get('MAE')),
                'RMSE': format_metric(metrics.get('RMSE')),
                'MAPE': format_metric(metrics.get('MAPE')),
                'MASE': format_metric(metrics.get('MASE')),
                'SMAPE': format_metric(metrics.get('SMAPE')),
                'R2': format_metric(metrics.get('R2')),
                'Время (с)': format_metric(result.get('time')),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Сортируем по указанной метрике (игнорируя NaN)
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=True, na_position='last')
        
        return df




