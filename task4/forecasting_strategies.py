"""
Этап 4: Стратегии прогнозирования
Direct, Recursive, Multi-output, DirRec стратегии для многошагового прогнозирования.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings('ignore')


class DirectStrategy(BaseEstimator, RegressorMixin):
    """
    Direct стратегия: обучение h отдельных моделей для каждого горизонта.
    """
    
    def __init__(self, base_model, horizon=7):
        """
        Parameters:
        -----------
        base_model : sklearn model
            Базовая модель
        horizon : int
            Горизонт прогнозирования
        """
        self.base_model = base_model
        self.horizon = horizon
        self.models = []
    
    def fit(self, X, y):
        """Обучает h моделей для каждого горизонта."""
        self.models = []
        
        for h in range(1, self.horizon + 1):
            # Создаем целевую переменную для горизонта h
            y_h = y.shift(-h) if isinstance(y, pd.Series) else np.roll(y, -h)
            
            # Убираем последние h наблюдений
            X_h = X.iloc[:-h] if isinstance(X, pd.DataFrame) else X[:-h]
            y_h = y_h.iloc[:-h] if isinstance(y_h, pd.Series) else y_h[:-h]
            
            # Обучаем модель
            model = type(self.base_model)(**self.base_model.get_params())
            model.fit(X_h, y_h)
            self.models.append(model)
        
        return self
    
    def predict(self, X):
        """Предсказывает для всех горизонтов."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)


class RecursiveStrategy(BaseEstimator, RegressorMixin):
    """
    Recursive стратегия: одна модель, предсказания используются как входы.
    """
    
    def __init__(self, base_model, horizon=7):
        """
        Parameters:
        -----------
        base_model : sklearn model
            Базовая модель
        horizon : int
            Горизонт прогнозирования
        """
        self.base_model = base_model
        self.horizon = horizon
        self.model = None
        self.feature_names_ = None
    
    def fit(self, X, y):
        """Обучает модель для одношагового прогноза."""
        # Обучаем на одношаговый прогноз
        y_1 = y.shift(-1) if isinstance(y, pd.Series) else np.roll(y, -1)
        
        X_1 = X.iloc[:-1] if isinstance(X, pd.DataFrame) else X[:-1]
        y_1 = y_1.iloc[:-1] if isinstance(y_1, pd.Series) else y_1[:-1]
        
        self.model = type(self.base_model)(**self.base_model.get_params())
        self.model.fit(X_1, y_1)
        
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        
        return self
    
    def predict(self, X):
        """Рекурсивно предсказывает для всех горизонтов."""
        predictions = np.zeros((len(X), self.horizon))
        X_current = X.copy()
        
        for h in range(self.horizon):
            # Предсказываем следующий шаг
            pred = self.model.predict(X_current)
            predictions[:, h] = pred
            
            # Обновляем признаки для следующего шага
            if h < self.horizon - 1:
                # Обновляем лаги (предполагаем, что первый признак - это lag_1)
                if isinstance(X_current, pd.DataFrame):
                    # Сдвигаем лаги
                    lag_cols = [col for col in X_current.columns if col.startswith('lag_')]
                    if lag_cols:
                        # Обновляем lag_1 значением текущего предсказания
                        if 'lag_1' in X_current.columns:
                            X_current['lag_1'] = pred
                        # Сдвигаем остальные лаги
                        for i in range(2, len(lag_cols) + 1):
                            if f'lag_{i}' in X_current.columns:
                                X_current[f'lag_{i}'] = X_current[f'lag_{i-1}']
                else:
                    # Для numpy массивов обновляем первый столбец (предполагаем lag_1)
                    if X_current.shape[1] > 0:
                        X_current[:, 0] = pred
        
        return predictions


class MultiOutputStrategy(BaseEstimator, RegressorMixin):
    """
    Multi-output стратегия: одна модель предсказывает вектор [y_{t+1}, ..., y_{t+h}].
    """
    
    def __init__(self, base_model, horizon=7):
        """
        Parameters:
        -----------
        base_model : sklearn model
            Базовая модель
        horizon : int
            Горизонт прогнозирования
        """
        self.base_model = base_model
        self.horizon = horizon
        self.model = None
    
    def fit(self, X, y):
        """Обучает модель для многошагового прогноза."""
        # Создаем целевую переменную: матрица [y_{t+1}, ..., y_{t+h}]
        y_multi = []
        
        for h in range(1, self.horizon + 1):
            y_h = y.shift(-h) if isinstance(y, pd.Series) else np.roll(y, -h)
            y_multi.append(y_h)
        
        y_multi = np.column_stack(y_multi)
        
        # Убираем последние horizon наблюдений
        X_multi = X.iloc[:-self.horizon] if isinstance(X, pd.DataFrame) else X[:-self.horizon]
        y_multi = y_multi[:-self.horizon]
        
        # Используем MultiOutputRegressor
        self.model = MultiOutputRegressor(
            type(self.base_model)(**self.base_model.get_params())
        )
        self.model.fit(X_multi, y_multi)
        
        return self
    
    def predict(self, X):
        """Предсказывает для всех горизонтов одновременно."""
        return self.model.predict(X)


class DirRecStrategy(BaseEstimator, RegressorMixin):
    """
    DirRec стратегия: гибрид - рекурсивная в пределах окна, прямая между окнами.
    """
    
    def __init__(self, base_model, horizon=7, window_size=3):
        """
        Parameters:
        -----------
        base_model : sklearn model
            Базовая модель
        horizon : int
            Горизонт прогнозирования
        window_size : int
            Размер окна для рекурсивного прогноза
        """
        self.base_model = base_model
        self.horizon = horizon
        self.window_size = window_size
        self.models = []
    
    def fit(self, X, y):
        """Обучает модели для каждого окна."""
        self.models = []
        n_windows = (self.horizon + self.window_size - 1) // self.window_size
        
        for w in range(n_windows):
            start_h = w * self.window_size + 1
            end_h = min((w + 1) * self.window_size + 1, self.horizon + 1)
            
            # Обучаем модель для первого шага в окне
            y_h = y.shift(-start_h) if isinstance(y, pd.Series) else np.roll(y, -start_h)
            
            X_h = X.iloc[:-start_h] if isinstance(X, pd.DataFrame) else X[:-start_h]
            y_h = y_h.iloc[:-start_h] if isinstance(y_h, pd.Series) else y_h[:-start_h]
            
            model = type(self.base_model)(**self.base_model.get_params())
            model.fit(X_h, y_h)
            self.models.append(model)
        
        return self
    
    def predict(self, X):
        """Предсказывает используя гибридную стратегию."""
        predictions = np.zeros((len(X), self.horizon))
        X_current = X.copy()
        
        n_windows = (self.horizon + self.window_size - 1) // self.window_size
        pred_idx = 0
        
        for w in range(n_windows):
            window_size = min(self.window_size, self.horizon - pred_idx)
            
            # Первый шаг в окне - прямое предсказание
            pred = self.models[w].predict(X_current)
            predictions[:, pred_idx] = pred
            pred_idx += 1
            
            # Остальные шаги в окне - рекурсивно
            for h in range(1, window_size):
                if pred_idx < self.horizon:
                    # Обновляем признаки
                    if isinstance(X_current, pd.DataFrame):
                        if 'lag_1' in X_current.columns:
                            X_current['lag_1'] = pred
                    else:
                        if X_current.shape[1] > 0:
                            X_current[:, 0] = pred
                    
                    # Предсказываем следующий шаг
                    pred = self.models[w].predict(X_current)
                    predictions[:, pred_idx] = pred
                    pred_idx += 1
        
        return predictions


def compare_strategies(strategies, X_train, y_train, X_test, y_test):
    """
    Сравнивает различные стратегии прогнозирования.
    
    Parameters:
    -----------
    strategies : dict
        Словарь {название: стратегия}
    X_train, y_train : array-like
        Обучающие данные
    X_test, y_test : array-like
        Тестовые данные
    
    Returns:
    --------
    results : dict
        Результаты сравнения
    """
    results = {}
    
    for name, strategy in strategies.items():
        print(f"Обучаем стратегию: {name}")
        
        # Обучение
        import time
        start_time = time.time()
        strategy.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Предсказание
        start_time = time.time()
        predictions = strategy.predict(X_test)
        predict_time = time.time() - start_time
        
        # Оценка накопления ошибки
        errors = []
        for h in range(predictions.shape[1]):
            if isinstance(y_test, pd.Series):
                y_true_h = y_test.iloc[h:len(y_test) - predictions.shape[1] + h + 1]
            else:
                y_true_h = y_test[h:len(y_test) - predictions.shape[1] + h + 1]
            
            y_pred_h = predictions[:len(y_true_h), h]
            mae = np.mean(np.abs(y_true_h - y_pred_h))
            errors.append(mae)
        
        results[name] = {
            'train_time': train_time,
            'predict_time': predict_time,
            'errors_by_horizon': errors,
            'mean_error': np.mean(errors),
            'cumulative_error': np.sum(errors)
        }
    
    return results


