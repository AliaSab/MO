"""
Этап 1: Инжиниринг признаков
Лаги, скользящие окна, экспоненциальное сглаживание, временные признаки,
сезонные компоненты, трансформации целевой переменной.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import boxcox, inv_boxcox
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Класс для создания признаков временных рядов."""
    
    def __init__(self):
        self.lags = [1, 2, 3, 7, 14, 30]
        self.windows = [7, 14, 30]
        self.alphas = [0.3, 0.5, 0.7]
        self.fourier_periods = [7, 30]
        self.boxcox_lambda = None
        self.scaler = StandardScaler()
        self.use_log = False
        
    def create_lags(self, series, lags=None):
        """Создает лаги целевой переменной."""
        if lags is None:
            lags = self.lags
        
        df = pd.DataFrame(index=series.index)
        for lag in lags:
            df[f'lag_{lag}'] = series.shift(lag)
        return df
    
    def create_rolling_features(self, series, windows=None):
        """Создает скользящие окна: mean, std, min, max, median."""
        if windows is None:
            windows = self.windows
        
        df = pd.DataFrame(index=series.index)
        for window in windows:
            rolling = series.rolling(window=window, min_periods=1)
            df[f'rolling_mean_{window}'] = rolling.mean()
            df[f'rolling_std_{window}'] = rolling.std()
            df[f'rolling_min_{window}'] = rolling.min()
            df[f'rolling_max_{window}'] = rolling.max()
            df[f'rolling_median_{window}'] = rolling.median()
        return df
    
    def create_exponential_smoothing(self, series, alphas=None):
        """Создает экспоненциальное сглаживание."""
        if alphas is None:
            alphas = self.alphas
        
        df = pd.DataFrame(index=series.index)
        for alpha in alphas:
            col_name = f'exp_smooth_{alpha}'
            df[col_name] = series.copy()
            for i in range(1, len(series)):
                df[col_name].iloc[i] = alpha * series.iloc[i] + (1 - alpha) * df[col_name].iloc[i-1]
        return df
    
    def create_temporal_features(self, date_index):
        """Создает временные признаки из индекса дат."""
        df = pd.DataFrame(index=date_index)
        df['day_of_week'] = date_index.dayofweek
        df['month'] = date_index.month
        df['quarter'] = date_index.quarter
        df['week_of_year'] = date_index.isocalendar().week
        
        # Циклическое кодирование для дня недели и месяца
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_holiday_features(self, date_index, is_holiday_series=None):
        """Создает признаки праздников и выходных."""
        df = pd.DataFrame(index=date_index)
        
        if is_holiday_series is not None:
            df['is_holiday'] = is_holiday_series.astype(int)
        else:
            df['is_holiday'] = 0
        
        df['is_weekend'] = (date_index.dayofweek >= 5).astype(int)
        
        return df
    
    def create_fourier_terms(self, date_index, periods=None):
        """Создает сезонные компоненты через ряды Фурье."""
        if periods is None:
            periods = self.fourier_periods
        
        df = pd.DataFrame(index=date_index)
        t = np.arange(len(date_index))
        
        for s in periods:
            df[f'fourier_sin_{s}'] = np.sin(2 * np.pi * t / s)
            df[f'fourier_cos_{s}'] = np.cos(2 * np.pi * t / s)
        
        return df
    
    def apply_log_transform(self, series):
        """Применяет логарифмическое преобразование, если все значения > 0."""
        if (series > 0).all():
            self.use_log = True
            return np.log1p(series)
        return series
    
    def apply_boxcox_transform(self, series):
        """Применяет преобразование Бокса-Кокса."""
        if (series > 0).all():
            series_positive = series + 1  # Сдвиг для положительных значений
            transformed, self.boxcox_lambda = stats.boxcox(series_positive)
            return pd.Series(transformed, index=series.index)
        return series
    
    def inverse_boxcox_transform(self, series):
        """Обратное преобразование Бокса-Кокса."""
        if self.boxcox_lambda is not None:
            series_positive = inv_boxcox(series, self.boxcox_lambda)
            return series_positive - 1
        return series
    
    def inverse_log_transform(self, series):
        """Обратное логарифмическое преобразование."""
        if self.use_log:
            return np.expm1(series)
        return series
    
    def create_all_features(self, series, date_index, is_holiday_series=None, 
                           apply_log=True, apply_boxcox=True):
        """Создает все признаки для временного ряда."""
        features_list = []
        
        # Лаги
        lag_features = self.create_lags(series)
        features_list.append(lag_features)
        
        # Скользящие окна
        rolling_features = self.create_rolling_features(series)
        features_list.append(rolling_features)
        
        # Экспоненциальное сглаживание
        exp_smooth_features = self.create_exponential_smoothing(series)
        features_list.append(exp_smooth_features)
        
        # Временные признаки
        temporal_features = self.create_temporal_features(date_index)
        features_list.append(temporal_features)
        
        # Признаки праздников
        holiday_features = self.create_holiday_features(date_index, is_holiday_series)
        features_list.append(holiday_features)
        
        # Ряды Фурье
        fourier_features = self.create_fourier_terms(date_index)
        features_list.append(fourier_features)
        
        # Объединяем все признаки
        all_features = pd.concat(features_list, axis=1)
        
        # Заполняем пропуски
        all_features = all_features.bfill().fillna(0)
        
        # Трансформация целевой переменной
        y_transformed = series.copy()
        transform_info = {}
        
        if apply_log and (series > 0).all():
            y_transformed = self.apply_log_transform(series)
            transform_info['log'] = True
        else:
            transform_info['log'] = False
        
        if apply_boxcox and (series > 0).all() and not apply_log:
            y_transformed = self.apply_boxcox_transform(series)
            transform_info['boxcox'] = True
            transform_info['boxcox_lambda'] = self.boxcox_lambda
        else:
            transform_info['boxcox'] = False
        
        return all_features, y_transformed, transform_info
    
    def get_feature_names(self):
        """Возвращает список всех создаваемых признаков."""
        feature_names = []
        
        # Лаги
        for lag in self.lags:
            feature_names.append(f'lag_{lag}')
        
        # Скользящие окна
        for window in self.windows:
            feature_names.extend([
                f'rolling_mean_{window}',
                f'rolling_std_{window}',
                f'rolling_min_{window}',
                f'rolling_max_{window}',
                f'rolling_median_{window}'
            ])
        
        # Экспоненциальное сглаживание
        for alpha in self.alphas:
            feature_names.append(f'exp_smooth_{alpha}')
        
        # Временные признаки
        feature_names.extend([
            'day_of_week', 'month', 'quarter', 'week_of_year',
            'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos'
        ])
        
        # Признаки праздников
        feature_names.extend(['is_holiday', 'is_weekend'])
        
        # Ряды Фурье
        for period in self.fourier_periods:
            feature_names.extend([f'fourier_sin_{period}', f'fourier_cos_{period}'])
        
        return feature_names

