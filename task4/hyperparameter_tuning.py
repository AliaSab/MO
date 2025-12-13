"""
Этап 3: Подбор гиперпараметров
GridSearchCV для линейных моделей, Optuna для градиентного бустинга, AutoGluon.
"""

import numpy as np
import platform
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import warnings
warnings.filterwarnings('ignore')

# Определяем оптимальное количество потоков для Windows
def get_n_jobs():
    """Возвращает оптимальное количество потоков (1 для Windows, -1 для других ОС)."""
    return 1 if platform.system() == 'Windows' else -1

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna не установлен. Установите: pip install optuna")

from validation import DataValidator


class HyperparameterTuner:
    """Класс для подбора гиперпараметров моделей."""
    
    def __init__(self, cv=None):
        """
        Инициализация тюнера.
        
        Parameters:
        -----------
        cv : cross-validation object, optional
            Объект кросс-валидации (например, TimeSeriesSplit)
        """
        self.cv = cv
        self.best_params_ = {}
        self.best_scores_ = {}
    
    def grid_search_linear(self, model, X, y, param_grid, scoring='neg_mean_absolute_error'):
        """
        GridSearchCV для линейных моделей.
        
        Parameters:
        -----------
        model : sklearn model
            Модель для обучения
        X : array-like
            Признаки
        y : array-like
            Целевая переменная
        param_grid : dict
            Сетка параметров
        scoring : str
            Метрика для оценки
        
        Returns:
        --------
        best_model : fitted model
            Лучшая модель
        """
        if self.cv is None:
            from sklearn.model_selection import TimeSeriesSplit
            self.cv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=self.cv,
            scoring=scoring,
            n_jobs=get_n_jobs(),
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_params_[type(model).__name__] = grid_search.best_params_
        self.best_scores_[type(model).__name__] = grid_search.best_score_
        
        return grid_search.best_estimator_
    
    def optuna_tune_lgbm(self, X_train, y_train, n_trials=100, cv=None):
        """
        Подбор гиперпараметров для LightGBM через Optuna.
        
        Parameters:
        -----------
        X_train : array-like
            Обучающие признаки
        y_train : array-like
            Обучающая целевая переменная
        n_trials : int
            Количество испытаний
        cv : cross-validation object, optional
            Объект кросс-валидации
        
        Returns:
        --------
        best_params : dict
            Лучшие параметры
        best_model : fitted model
            Лучшая модель
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna не установлен")
        
        if cv is None:
            from validation import DataValidator
            validator = DataValidator()
            cv = validator.create_time_series_split(n_splits=5)
        
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError("LightGBM не установлен. Установите: pip install lightgbm")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': get_n_jobs(),
                'verbose': -1
            }
            
            model = LGBMRegressor(**params)
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv, 
                scoring='neg_mean_absolute_error',
                n_jobs=get_n_jobs()
            )
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_model = LGBMRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        self.best_params_['LGBMRegressor'] = best_params
        self.best_scores_['LGBMRegressor'] = study.best_value
        
        return best_params, best_model
    
    def optuna_tune_xgboost(self, X_train, y_train, n_trials=100, cv=None):
        """
        Подбор гиперпараметров для XGBoost через Optuna.
        
        Parameters:
        -----------
        X_train : array-like
            Обучающие признаки
        y_train : array-like
            Обучающая целевая переменная
        n_trials : int
            Количество испытаний
        cv : cross-validation object, optional
            Объект кросс-валидации
        
        Returns:
        --------
        best_params : dict
            Лучшие параметры
        best_model : fitted model
            Лучшая модель
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna не установлен")
        
        if cv is None:
            from validation import DataValidator
            validator = DataValidator()
            cv = validator.create_time_series_split(n_splits=5)
        
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("XGBoost не установлен. Установите: pip install xgboost")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': get_n_jobs()
            }
            
            model = XGBRegressor(**params)
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv, 
                scoring='neg_mean_absolute_error',
                n_jobs=get_n_jobs()
            )
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_model = XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        self.best_params_['XGBRegressor'] = best_params
        self.best_scores_['XGBRegressor'] = study.best_value
        
        return best_params, best_model
    
    def get_best_params(self):
        """Возвращает лучшие параметры для всех моделей."""
        return self.best_params_
    
    def get_best_scores(self):
        """Возвращает лучшие оценки для всех моделей."""
        return self.best_scores_

