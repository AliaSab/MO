"""
Этап 2: Валидация и разбиение данных
Хронологическое разбиение, TimeSeriesSplit, Purged Walk-Forward, AutoGluon backtesting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("AutoGluon не установлен. Установите: pip install autogluon.timeseries")


class DataValidator:
    """Класс для валидации и разбиения временных рядов."""
    
    def __init__(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """
        Инициализация валидатора.
        
        Parameters:
        -----------
        train_ratio : float
            Доля обучающей выборки (по умолчанию 0.6)
        val_ratio : float
            Доля валидационной выборки (по умолчанию 0.2)
        test_ratio : float
            Доля тестовой выборки (по умолчанию 0.2)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Сумма долей должна быть равна 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def chronological_split(self, data, date_index=None):
        """
        Хронологическое разбиение данных на train/val/test.
        
        Parameters:
        -----------
        data : pd.DataFrame, pd.Series или pd.DatetimeIndex
            Данные для разбиения
        date_index : pd.DatetimeIndex, optional
            Индекс дат (если data не имеет DatetimeIndex)
        
        Returns:
        --------
        train, val, test : tuple
            Разделенные данные
        """
        # Если data - это DatetimeIndex, обрабатываем его отдельно
        if isinstance(data, pd.DatetimeIndex):
            date_index = data
            # Сортируем по дате
            sorted_idx = date_index.argsort()
            date_sorted = date_index[sorted_idx]
            
            n = len(date_sorted)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)
            
            train = date_sorted[:train_end]
            val = date_sorted[train_end:val_end]
            test = date_sorted[val_end:]
            
            return train, val, test
        
        # Для DataFrame и Series
        if date_index is None:
            if isinstance(data.index, pd.DatetimeIndex):
                date_index = data.index
            else:
                raise ValueError("Необходим DatetimeIndex или параметр date_index")
        
        # Сортируем по дате
        sorted_idx = date_index.argsort()
        
        # Сортируем данные
        if isinstance(data, pd.Series):
            data_sorted = data.iloc[sorted_idx]
        elif isinstance(data, pd.DataFrame):
            data_sorted = data.iloc[sorted_idx]
        else:
            raise ValueError(f"Неподдерживаемый тип данных: {type(data)}")
        
        date_sorted = date_index[sorted_idx]
        
        n = len(data_sorted)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        train = data_sorted.iloc[:train_end]
        val = data_sorted.iloc[train_end:val_end]
        test = data_sorted.iloc[val_end:]
        
        return train, val, test
    
    def create_time_series_split(self, n_splits=5, max_train_size=365, gap=0):
        """
        Создает TimeSeriesSplit для кросс-валидации.
        
        Parameters:
        -----------
        n_splits : int
            Количество разбиений
        max_train_size : int
            Максимальный размер обучающей выборки
        gap : int
            Разрыв между train и test (для Purged Walk-Forward)
        
        Returns:
        --------
        tscv : TimeSeriesSplit
            Объект для кросс-валидации
        """
        return TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size,
            gap=gap
        )
    
    def purged_walk_forward_split(self, data, n_splits=5, gap=7, test_size=None):
        """
        Purged Walk-Forward разбиение с gap между train и test.
        
        Parameters:
        -----------
        data : pd.DataFrame или pd.Series
            Данные
        n_splits : int
            Количество разбиений
        gap : int
            Разрыв между train и test
        test_size : int, optional
            Размер тестовой выборки (по умолчанию len(data) // (n_splits + 1))
        
        Yields:
        -------
        train_idx, test_idx : tuple
            Индексы для train и test
        """
        n = len(data)
        if test_size is None:
            test_size = n // (n_splits + 1)
        
        for i in range(n_splits):
            test_start = n - (n_splits - i) * test_size
            test_end = test_start + test_size
            train_end = test_start - gap
            
            if train_end > 0:
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, min(test_end, n))
                yield train_idx, test_idx
    
    def prepare_autogluon_data(self, data, target_col='Weekly_Sales', 
                               date_col='Date', id_col=None):
        """
        Подготавливает данные для AutoGluon TimeSeriesPredictor.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные с колонками target_col и date_col
        target_col : str
            Название целевой переменной
        date_col : str
            Название колонки с датами
        id_col : str, optional
            Название колонки с идентификатором ряда (для мультирядов)
        
        Returns:
        --------
        ts_dataframe : TimeSeriesDataFrame
            Данные в формате AutoGluon
        """
        if not AUTOGLUON_AVAILABLE:
            raise ImportError("AutoGluon не установлен")
        
        df = data.copy()
        
        # Преобразуем дату
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            if df[date_col].dt.tz is not None:
                df[date_col] = df[date_col].dt.tz_localize(None)
        
        # Если есть id_col, используем его, иначе создаем фиктивный
        if id_col is None or id_col not in df.columns:
            df['item_id'] = 'series_1'
            id_col = 'item_id'
        
        # Переименовываем колонки для AutoGluon
        df_ts = df[[id_col, date_col, target_col]].copy()
        df_ts.columns = ['item_id', 'timestamp', 'target']
        
        # Сортируем
        df_ts = df_ts.sort_values(['item_id', 'timestamp'])
        
        # Создаем TimeSeriesDataFrame
        ts_dataframe = TimeSeriesDataFrame(df_ts)
        
        return ts_dataframe
    
    def autogluon_backtest(self, train_data, val_data=None, prediction_length=7,
                          presets=["medium_quality"], include_models=None,
                          eval_metric="MAE"):
        """
        Запускает AutoGluon backtesting.
        
        Parameters:
        -----------
        train_data : TimeSeriesDataFrame
            Обучающие данные
        val_data : TimeSeriesDataFrame, optional
            Валидационные данные
        prediction_length : int
            Горизонт прогнозирования
        presets : list
            Пресеты качества AutoGluon
        include_models : list, optional
            Список моделей для включения
        eval_metric : str
            Метрика оценки
        
        Returns:
        --------
        predictor : TimeSeriesPredictor
            Обученный предиктор
        """
        if not AUTOGLUON_AVAILABLE:
            raise ImportError("AutoGluon не установлен")
        
        predictor = TimeSeriesPredictor(
            target="target",
            prediction_length=prediction_length,
            eval_metric=eval_metric
        )
        
        fit_kwargs = {
            "presets": presets,
        }
        
        if include_models:
            fit_kwargs["include_models"] = include_models
        
        predictor.fit(train_data, **fit_kwargs)
        
        return predictor

