"""
Модуль для создания признаков временных рядов для глубокого обучения.
Включает временные фичи, лаги, скользящие статистики, Fourier features, positional encoding.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Класс для создания признаков временных рядов для DL моделей."""
    
    def __init__(self):
        self.features_created = []
        
    def _normalize_dates(self, dates):
        """Нормализует даты, убирая информацию о часовом поясе."""
        if isinstance(dates, pd.DatetimeIndex):
            if dates.tz is not None:
                return dates.tz_convert('UTC').tz_localize(None)
            return dates
        elif isinstance(dates, pd.Series):
            if isinstance(dates.index, pd.DatetimeIndex):
                if dates.index.tz is not None:
                    return dates.index.tz_convert('UTC').tz_localize(None)
                return dates.index
            else:
                dt_index = pd.to_datetime(dates.index, utc=True)
                if dt_index.tz is not None:
                    return dt_index.tz_convert('UTC').tz_localize(None)
                return dt_index
        else:
            dt_index = pd.to_datetime(dates, utc=True)
            if hasattr(dt_index, 'tz') and dt_index.tz is not None:
                return dt_index.tz_convert('UTC').tz_localize(None)
            return dt_index
    
    def create_temporal_features(self, dates, cyclic=True):
        """
        Создает временные признаки из дат.
        
        Args:
            dates: массив дат
            cyclic: использовать циклическое кодирование (sin/cos)
        """
        dt_index = self._normalize_dates(dates)
        features = pd.DataFrame(index=dt_index)
        
        # День недели
        dayofweek = dt_index.dayofweek
        if cyclic:
            features['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7)
            features['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7)
        else:
            features['dayofweek'] = dayofweek
        
        # Месяц
        month = dt_index.month
        if cyclic:
            features['month_sin'] = np.sin(2 * np.pi * month / 12)
            features['month_cos'] = np.cos(2 * np.pi * month / 12)
        else:
            features['month'] = month
        
        # День месяца
        day = dt_index.day
        if cyclic:
            features['day_sin'] = np.sin(2 * np.pi * day / 31)
            features['day_cos'] = np.cos(2 * np.pi * day / 31)
        else:
            features['day'] = day
        
        # Неделя года
        week = dt_index.isocalendar().week
        if cyclic:
            features['week_sin'] = np.sin(2 * np.pi * week / 52)
            features['week_cos'] = np.cos(2 * np.pi * week / 52)
        else:
            features['week'] = week
        
        # Квартал
        quarter = dt_index.quarter
        if cyclic:
            features['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
            features['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
        else:
            features['quarter'] = quarter
        
        # День года
        dayofyear = dt_index.dayofyear
        if cyclic:
            features['dayofyear_sin'] = np.sin(2 * np.pi * dayofyear / 365)
            features['dayofyear_cos'] = np.cos(2 * np.pi * dayofyear / 365)
        else:
            features['dayofyear'] = dayofyear
        
        # Является ли выходным (суббота или воскресенье)
        features['is_weekend'] = (dayofweek >= 5).astype(int)
        
        self.features_created.extend(features.columns.tolist())
        return features
    
    def create_fourier_seasonal_features(self, dates, periods=[7, 12, 24, 52]):
        """
        Создает сезонные Fourier features.
        
        Args:
            dates: массив дат
            periods: список периодов для сезонности (дни, месяцы, часы, недели)
        """
        dt_index = self._normalize_dates(dates)
        features = pd.DataFrame(index=dt_index)
        
        # Номер наблюдения (для временных рядов без явных дат)
        t = np.arange(len(dt_index))
        
        for period in periods:
            features[f'fourier_sin_{period}'] = np.sin(2 * np.pi * t / period)
            features[f'fourier_cos_{period}'] = np.cos(2 * np.pi * t / period)
            self.features_created.extend([f'fourier_sin_{period}', f'fourier_cos_{period}'])
        
        return features
    
    def create_lags(self, y, lags=[1, 7, 30]):
        """Создает лаги временного ряда."""
        features = pd.DataFrame(index=y.index if isinstance(y, pd.Series) else range(len(y)))
        
        y_values = y.values if isinstance(y, pd.Series) else y
        
        for lag in lags:
            if lag > 0:
                lag_values = np.concatenate([np.full(lag, np.nan), y_values[:-lag]])
                features[f'lag_{lag}'] = lag_values
                self.features_created.append(f'lag_{lag}')
        
        return features
    
    def create_rolling_features(self, y, windows=[7, 30], functions=['mean', 'std', 'min', 'max']):
        """Создает скользящие статистики."""
        if isinstance(y, pd.Series):
            features = pd.DataFrame(index=y.index)
            y_series = y
        else:
            features = pd.DataFrame(index=range(len(y)))
            y_series = pd.Series(y)
        
        for window in windows:
            rolling = y_series.rolling(window=window, min_periods=1)
            
            if 'mean' in functions:
                features[f'rolling_mean_{window}'] = rolling.mean()
                self.features_created.append(f'rolling_mean_{window}')
            
            if 'std' in functions:
                features[f'rolling_std_{window}'] = rolling.std()
                self.features_created.append(f'rolling_std_{window}')
            
            if 'min' in functions:
                features[f'rolling_min_{window}'] = rolling.min()
                self.features_created.append(f'rolling_min_{window}')
            
            if 'max' in functions:
                features[f'rolling_max_{window}'] = rolling.max()
                self.features_created.append(f'rolling_max_{window}')
        
        return features
    
    def create_positional_encoding(self, seq_length, d_model):
        """
        Создает positional encoding для трансформеров.
        
        Args:
            seq_length: длина последовательности
            d_model: размерность модели
            
        Returns:
            pos_encoding: (seq_length, d_model)
        """
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((seq_length, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def create_all_features(self, y, dates=None, lags=[1, 7, 30], 
                           rolling_windows=[7, 30], fourier_periods=[7, 12, 24, 52],
                           include_holiday=False, holiday_column=None):
        """
        Создает все признаки для временного ряда.
        
        Args:
            y: временной ряд
            dates: даты (опционально)
            lags: список лагов
            rolling_windows: окна для скользящих статистик
            fourier_periods: периоды для Fourier features
            include_holiday: включать ли информацию о праздниках
            holiday_column: колонка с информацией о праздниках
        """
        all_features = pd.DataFrame()
        
        # Временные признаки
        if dates is not None:
            temporal = self.create_temporal_features(dates, cyclic=True)
            all_features = pd.concat([all_features, temporal], axis=1)
            
            # Fourier features
            fourier = self.create_fourier_seasonal_features(dates, periods=fourier_periods)
            all_features = pd.concat([all_features, fourier], axis=1)
        
        # Лаги
        lags_features = self.create_lags(y, lags=lags)
        all_features = pd.concat([all_features, lags_features], axis=1)
        
        # Скользящие статистики
        rolling_features = self.create_rolling_features(y, windows=rolling_windows)
        all_features = pd.concat([all_features, rolling_features], axis=1)
        
        # Праздники
        if include_holiday and holiday_column is not None:
            if isinstance(holiday_column, pd.Series):
                all_features['is_holiday'] = holiday_column.astype(int)
                self.features_created.append('is_holiday')
        
        # Заполнение пропусков
        all_features = all_features.bfill().fillna(0)
        
        return all_features

