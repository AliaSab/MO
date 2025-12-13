"""
Этап 5: Построение моделей
Линейные, ансамбли, SVM, AutoML, бейзлайны.
"""

import numpy as np
import pandas as pd
import platform
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.svm import SVR, NuSVR
import warnings
warnings.filterwarnings('ignore')

# Определяем оптимальное количество потоков для Windows
def get_n_jobs():
    """Возвращает оптимальное количество потоков (1 для Windows, -1 для других ОС)."""
    return 1 if platform.system() == 'Windows' else -1

# Опциональные импорты
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from autogluon.timeseries import TimeSeriesPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


class BaselineModels:
    """Бейзлайны для временных рядов."""
    
    @staticmethod
    def naive_forecast(y, horizon=1):
        """Naive: последнее значение."""
        last_value = y.iloc[-1] if isinstance(y, pd.Series) else y[-1]
        return np.full(horizon, last_value)
    
    @staticmethod
    def seasonal_naive_forecast(y, seasonality=7, horizon=1):
        """Seasonal Naive: значение сезон назад."""
        if len(y) < seasonality:
            return BaselineModels.naive_forecast(y, horizon)
        
        seasonal_values = y.iloc[-seasonality:] if isinstance(y, pd.Series) else y[-seasonality:]
        predictions = []
        for h in range(horizon):
            idx = (h % seasonality)
            predictions.append(seasonal_values.iloc[idx] if isinstance(seasonal_values, pd.Series) else seasonal_values[idx])
        return np.array(predictions)
    
    @staticmethod
    def moving_average_forecast(y, window=7, horizon=1):
        """Moving Average: среднее за окно."""
        ma_value = y.iloc[-window:].mean() if isinstance(y, pd.Series) else y[-window:].mean()
        return np.full(horizon, ma_value)
    
    @staticmethod
    def linear_trend_forecast(y, horizon=1):
        """Linear Trend: линейный тренд."""
        n = len(y)
        x = np.arange(n)
        coeffs = np.polyfit(x, y.values if isinstance(y, pd.Series) else y, 1)
        future_x = np.arange(n, n + horizon)
        return np.polyval(coeffs, future_x)


def create_linear_models():
    """Создает линейные модели."""
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'BayesianRidge': BayesianRidge()
    }


def create_ensemble_models():
    """Создает модели ансамблей."""
    n_jobs = get_n_jobs()
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs)
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=n_jobs, verbose=-1)
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostRegressor(iterations=100, random_state=42, verbose=False)
    
    return models


def create_svm_models():
    """Создает SVM модели."""
    return {
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'NuSVR': NuSVR(kernel='rbf', nu=0.5, C=1.0)
    }


def create_stacking_model(base_models, meta_model=None):
    """
    Создает Stacking ансамбль.
    
    Parameters:
    -----------
    base_models : list of tuples (name, model)
        Базовые модели
    meta_model : sklearn model, optional
        Мета-модель (по умолчанию LinearRegression)
    
    Returns:
    --------
    StackingRegressor
    """
    if meta_model is None:
        meta_model = LinearRegression()
    
    return StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=get_n_jobs()
    )


def create_all_models():
    """Создает все доступные модели."""
    all_models = {}
    
    # Линейные
    all_models.update(create_linear_models())
    
    # Ансамбли
    all_models.update(create_ensemble_models())
    
    # SVM
    all_models.update(create_svm_models())
    
    return all_models


def create_autogluon_predictor(train_data, prediction_length=7, 
                               presets=["medium_quality", "high_quality", "best_quality"],
                               include_models=None, eval_metric="MAE"):
    """
    Создает и обучает AutoGluon TimeSeriesPredictor.
    
    Parameters:
    -----------
    train_data : TimeSeriesDataFrame
        Обучающие данные
    prediction_length : int
        Горизонт прогнозирования
    presets : list
        Пресеты качества
    include_models : list, optional
        Список моделей для включения
    eval_metric : str
        Метрика оценки
    
    Returns:
    --------
    TimeSeriesPredictor
    """
    if not AUTOGLUON_AVAILABLE:
        raise ImportError("AutoGluon не установлен")
    
    predictor = TimeSeriesPredictor(
        target="target",
        prediction_length=prediction_length,
        eval_metric=eval_metric
    )
    
    fit_kwargs = {"presets": presets}
    if include_models:
        fit_kwargs["include_models"] = include_models
    
    predictor.fit(train_data, **fit_kwargs)
    
    return predictor


class ModelTrainer:
    """Класс для обучения моделей."""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
    
    def add_model(self, name, model):
        """Добавляет модель."""
        self.models[name] = model
    
    def train_all(self, X_train, y_train):
        """Обучает все модели."""
        self.trained_models = {}
        
        for name, model in self.models.items():
            print(f"Обучаем модель: {name}")
            try:
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train, y_train)
                self.trained_models[name] = model_copy
            except Exception as e:
                print(f"Ошибка при обучении {name}: {e}")
        
        return self.trained_models
    
    def predict_all(self, X_test):
        """Предсказывает для всех моделей."""
        predictions = {}
        
        for name, model in self.trained_models.items():
            try:
                predictions[name] = model.predict(X_test)
            except Exception as e:
                print(f"Ошибка при предсказании {name}: {e}")
        
        return predictions
    
    def get_model(self, name):
        """Возвращает обученную модель."""
        return self.trained_models.get(name)

