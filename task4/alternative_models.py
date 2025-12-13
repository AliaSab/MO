"""
Альтернативные модели для временных рядов (если AutoGluon недоступен).
Использует statsforecast, darts, Prophet и другие библиотеки.
"""

import pandas as pd
import numpy as np
import platform
import warnings
warnings.filterwarnings('ignore')

# Определяем оптимальное количество потоков для Windows
def get_n_jobs():
    """Возвращает оптимальное количество потоков (1 для Windows, -1 для других ОС)."""
    return 1 if platform.system() == 'Windows' else -1

# Проверяем доступность альтернативных библиотек
ALTERNATIVES = {}

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, Naive, SeasonalNaive
    STATSFORECAST_AVAILABLE = True
    ALTERNATIVES['statsforecast'] = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    ALTERNATIVES['statsforecast'] = False

try:
    from darts import TimeSeries
    from darts.models import (
        ExponentialSmoothing, Prophet, ARIMA, 
        RandomForest, XGBModel, LightGBMModel
    )
    DARTS_AVAILABLE = True
    ALTERNATIVES['darts'] = True
except ImportError:
    DARTS_AVAILABLE = False
    ALTERNATIVES['darts'] = False

try:
    from prophet import Prophet as FBProphet
    PROPHET_AVAILABLE = True
    ALTERNATIVES['prophet'] = True
except ImportError:
    PROPHET_AVAILABLE = False
    ALTERNATIVES['prophet'] = False

try:
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.arima import AutoARIMA as SktimeARIMA
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing as SktimeETS
    SKTIME_AVAILABLE = True
    ALTERNATIVES['sktime'] = True
except ImportError:
    SKTIME_AVAILABLE = False
    ALTERNATIVES['sktime'] = False


class AlternativeTimeSeriesModels:
    """Альтернативные модели для временных рядов."""
    
    def __init__(self):
        self.available_libs = [lib for lib, avail in ALTERNATIVES.items() if avail]
        print(f"Доступные альтернативные библиотеки: {self.available_libs}")
    
    def prepare_data_for_statsforecast(self, series, freq='W'):
        """
        Подготавливает данные для statsforecast.
        
        Parameters:
        -----------
        series : pd.Series
            Временной ряд с DatetimeIndex
        freq : str
            Частота данных ('W' для недельных, 'D' для дневных)
        
        Returns:
        --------
        pd.DataFrame
            Данные в формате statsforecast
        """
        if not STATSFORECAST_AVAILABLE:
            raise ImportError("statsforecast не установлен")
        
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        df['unique_id'] = 'series_1'
        return df[['unique_id', 'ds', 'y']]
    
    def fit_statsforecast_models(self, train_data, horizon=7):
        """
        Обучает модели из statsforecast.
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Обучающие данные в формате statsforecast
        horizon : int
            Горизонт прогнозирования
        
        Returns:
        --------
        dict
            Словарь с прогнозами моделей
        """
        if not STATSFORECAST_AVAILABLE:
            raise ImportError("statsforecast не установлен")
        
        models = [
            AutoARIMA(season_length=7),
            AutoETS(season_length=7),
            AutoTheta(season_length=7),
            Naive(),
            SeasonalNaive(season_length=7)
        ]
        
        sf = StatsForecast(
            models=models,
            freq='W',
            n_jobs=get_n_jobs()
        )
        
        sf.fit(train_data)
        forecasts = sf.predict(h=horizon)
        
        results = {}
        for model in models:
            model_name = type(model).__name__
            if model_name in forecasts.columns:
                results[f'StatsForecast_{model_name}'] = forecasts[model_name].values
        
        return results
    
    def fit_darts_models(self, train_series, val_series, horizon=7):
        """
        Обучает модели из darts.
        
        Parameters:
        -----------
        train_series : pd.Series
            Обучающий ряд
        val_series : pd.Series
            Валидационный ряд (для оценки)
        horizon : int
            Горизонт прогнозирования
        
        Returns:
        --------
        dict
            Словарь с прогнозами моделей
        """
        if not DARTS_AVAILABLE:
            raise ImportError("darts не установлен")
        
        results = {}
        
        # Преобразуем в TimeSeries
        train_ts = TimeSeries.from_series(train_series)
        val_ts = TimeSeries.from_series(val_series[:horizon])
        
        # Exponential Smoothing
        try:
            model = ExponentialSmoothing()
            model.fit(train_ts)
            forecast = model.predict(horizon)
            results['Darts_ExponentialSmoothing'] = forecast.values().flatten()
        except Exception as e:
            print(f"Ошибка ExponentialSmoothing: {e}")
        
        # ARIMA
        try:
            model = ARIMA()
            model.fit(train_ts)
            forecast = model.predict(horizon)
            results['Darts_ARIMA'] = forecast.values().flatten()
        except Exception as e:
            print(f"Ошибка ARIMA: {e}")
        
        # Prophet (если доступен)
        if PROPHET_AVAILABLE:
            try:
                model = Prophet()
                model.fit(train_ts)
                forecast = model.predict(horizon)
                results['Darts_Prophet'] = forecast.values().flatten()
            except Exception as e:
                print(f"Ошибка Prophet: {e}")
        
        return results
    
    def fit_prophet_model(self, train_series, horizon=7):
        """
        Обучает модель Prophet.
        
        Parameters:
        -----------
        train_series : pd.Series
            Обучающий ряд
        horizon : int
            Горизонт прогнозирования
        
        Returns:
        --------
        np.array
            Прогноз
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet не установлен")
        
        df = pd.DataFrame({
            'ds': train_series.index,
            'y': train_series.values
        })
        
        model = FBProphet(weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=horizon, freq='W')
        forecast = model.predict(future)
        
        return forecast['yhat'].tail(horizon).values
    
    def fit_sktime_models(self, train_series, horizon=7):
        """
        Обучает модели из sktime.
        
        Parameters:
        -----------
        train_series : pd.Series
            Обучающий ряд
        horizon : int
            Горизонт прогнозирования
        
        Returns:
        --------
        dict
            Словарь с прогнозами моделей
        """
        if not SKTIME_AVAILABLE:
            raise ImportError("sktime не установлен")
        
        results = {}
        fh = ForecastingHorizon(np.arange(1, horizon + 1))
        
        # AutoARIMA
        try:
            model = SktimeARIMA()
            model.fit(train_series)
            forecast = model.predict(fh)
            results['Sktime_AutoARIMA'] = forecast.values
        except Exception as e:
            print(f"Ошибка Sktime AutoARIMA: {e}")
        
        # Exponential Smoothing
        try:
            model = SktimeETS()
            model.fit(train_series)
            forecast = model.predict(fh)
            results['Sktime_ETS'] = forecast.values
        except Exception as e:
            print(f"Ошибка Sktime ETS: {e}")
        
        return results
    
    def get_available_models(self):
        """Возвращает список доступных альтернативных моделей."""
        available = []
        
        if STATSFORECAST_AVAILABLE:
            available.extend([
                'StatsForecast_AutoARIMA',
                'StatsForecast_AutoETS',
                'StatsForecast_AutoTheta',
                'StatsForecast_Naive',
                'StatsForecast_SeasonalNaive'
            ])
        
        if DARTS_AVAILABLE:
            available.extend([
                'Darts_ExponentialSmoothing',
                'Darts_ARIMA'
            ])
            if PROPHET_AVAILABLE:
                available.append('Darts_Prophet')
        
        if PROPHET_AVAILABLE:
            available.append('Prophet')
        
        if SKTIME_AVAILABLE:
            available.extend([
                'Sktime_AutoARIMA',
                'Sktime_ETS'
            ])
        
        return available


def install_instructions():
    """Возвращает инструкции по установке альтернативных библиотек."""
    instructions = {
        'statsforecast': 'pip install statsforecast',
        'darts': 'pip install darts',
        'prophet': 'pip install prophet  # Требует компилятор C++',
        'sktime': 'pip install sktime[all]'
    }
    
    return instructions

