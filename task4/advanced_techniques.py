"""
Этап 8: Продвинутые техники
Ансамблирование, обработка выбросов, сегментация.
"""

import numpy as np
import pandas as pd
import platform
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Определяем оптимальное количество потоков для Windows
def get_n_jobs():
    """Возвращает оптимальное количество потоков (1 для Windows, -1 для других ОС)."""
    return 1 if platform.system() == 'Windows' else -1


class EnsembleBuilder:
    """Класс для создания ансамблей моделей."""
    
    @staticmethod
    def weighted_average_ensemble(predictions_dict, weights=None):
        """
        Взвешенное усреднение прогнозов.
        
        Parameters:
        -----------
        predictions_dict : dict
            Словарь {название_модели: предсказания}
        weights : dict, optional
            Словарь весов (по умолчанию равные веса)
        
        Returns:
        --------
        np.array
            Взвешенное среднее прогнозов
        """
        predictions_list = list(predictions_dict.values())
        
        if weights is None:
            weights = {name: 1.0 / len(predictions_dict) for name in predictions_dict.keys()}
        
        # Нормализуем веса
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Вычисляем взвешенное среднее
        weighted_pred = np.zeros_like(predictions_list[0])
        
        for name, pred in predictions_dict.items():
            weight = weights.get(name, 0)
            weighted_pred += weight * pred
        
        return weighted_pred
    
    @staticmethod
    def weighted_average_by_mase(predictions_dict, mase_scores):
        """
        Взвешенное усреднение по обратным значениям MASE.
        
        Parameters:
        -----------
        predictions_dict : dict
            Словарь {название_модели: предсказания}
        mase_scores : dict
            Словарь {название_модели: MASE}
        
        Returns:
        --------
        np.array
            Взвешенное среднее прогнозов
        """
        # Веса обратно пропорциональны MASE
        weights = {name: 1.0 / (score + 1e-10) for name, score in mase_scores.items()}
        
        return EnsembleBuilder.weighted_average_ensemble(predictions_dict, weights)
    
    @staticmethod
    def create_stacking_ensemble(base_models, meta_model=None, cv=5):
        """
        Создает Stacking ансамбль.
        
        Parameters:
        -----------
        base_models : list of tuples (name, model)
            Базовые модели
        meta_model : sklearn model, optional
            Мета-модель
        cv : int
            Количество фолдов для кросс-валидации
        
        Returns:
        --------
        StackingRegressor
        """
        if meta_model is None:
            meta_model = LinearRegression()
        
        return StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=cv,
            n_jobs=get_n_jobs()
        )
    
    @staticmethod
    def create_voting_ensemble(base_models, weights=None):
        """
        Создает Voting ансамбль.
        
        Parameters:
        -----------
        base_models : list of tuples (name, model)
            Базовые модели
        weights : array-like, optional
            Веса моделей
        
        Returns:
        --------
        VotingRegressor
        """
        return VotingRegressor(
            estimators=base_models,
            weights=weights,
            n_jobs=get_n_jobs()
        )


class OutlierHandler:
    """Класс для обработки выбросов."""
    
    @staticmethod
    def isolation_forest_detection(X, contamination=0.1):
        """
        Обнаружение выбросов через Isolation Forest.
        
        Parameters:
        -----------
        X : array-like
            Признаки
        contamination : float
            Доля выбросов
        
        Returns:
        --------
        np.array
            Маска выбросов (True = выброс)
        """
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(X)
        return outliers == -1
    
    @staticmethod
    def winsorize(series, lower_percentile=5, upper_percentile=95):
        """
        Winsorization: ограничение значений перцентилями.
        
        Parameters:
        -----------
        series : pd.Series или np.array
            Данные
        lower_percentile : float
            Нижний перцентиль
        upper_percentile : float
            Верхний перцентиль
        
        Returns:
        --------
        np.array
            Winsorized данные
        """
        lower_bound = np.percentile(series, lower_percentile)
        upper_bound = np.percentile(series, upper_percentile)
        
        series_winsorized = series.copy()
        series_winsorized[series_winsorized < lower_bound] = lower_bound
        series_winsorized[series_winsorized > upper_bound] = upper_bound
        
        return series_winsorized
    
    @staticmethod
    def robust_scaler_transform(X):
        """
        Применяет RobustScaler (менее чувствителен к выбросам).
        
        Parameters:
        -----------
        X : array-like
            Данные
        
        Returns:
        --------
        np.array
            Масштабированные данные
        """
        scaler = RobustScaler()
        return scaler.fit_transform(X)


class SeriesSegmentation:
    """Класс для сегментации временных рядов."""
    
    @staticmethod
    def cluster_series_by_features(features, n_clusters=3):
        """
        Кластеризация рядов по признакам.
        
        Parameters:
        -----------
        features : array-like
            Признаки для кластеризации
        n_clusters : int
            Количество кластеров
        
        Returns:
        --------
        np.array
            Метки кластеров
        """
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        return labels
    
    @staticmethod
    def segment_by_season(date_index, season='winter'):
        """
        Сегментация по сезонам.
        
        Parameters:
        -----------
        date_index : pd.DatetimeIndex
            Индекс дат
        season : str
            Сезон: 'winter', 'spring', 'summer', 'fall'
        
        Returns:
        --------
        np.array
            Маска сезона
        """
        month = date_index.month
        
        season_map = {
            'winter': [12, 1, 2],
            'spring': [3, 4, 5],
            'summer': [6, 7, 8],
            'fall': [9, 10, 11]
        }
        
        return month.isin(season_map.get(season, []))
    
    @staticmethod
    def segment_by_regime(series, n_regimes=2, method='kmeans'):
        """
        Сегментация по режимам (высокий/низкий уровень).
        
        Parameters:
        -----------
        series : pd.Series или np.array
            Временной ряд
        n_regimes : int
            Количество режимов
        method : str
            Метод: 'kmeans', 'threshold'
        
        Returns:
        --------
        np.array
            Метки режимов
        """
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            
            X = series.values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            return labels
        
        elif method == 'threshold':
            threshold = np.median(series)
            return (series > threshold).astype(int)
        
        else:
            raise ValueError(f"Неизвестный метод: {method}")


class AdvancedTechniques:
    """Объединяющий класс для продвинутых техник."""
    
    def __init__(self):
        self.ensemble_builder = EnsembleBuilder()
        self.outlier_handler = OutlierHandler()
        self.segmentation = SeriesSegmentation()
    
    def create_ensemble(self, predictions_dict, method='weighted_mase', mase_scores=None):
        """
        Создает ансамбль прогнозов.
        
        Parameters:
        -----------
        predictions_dict : dict
            Словарь прогнозов
        method : str
            Метод: 'weighted_mase', 'equal', 'stacking'
        mase_scores : dict, optional
            MASE оценки для взвешивания
        
        Returns:
        --------
        np.array
            Ансамбль прогнозов
        """
        if method == 'weighted_mase' and mase_scores:
            return self.ensemble_builder.weighted_average_by_mase(predictions_dict, mase_scores)
        elif method == 'equal':
            return self.ensemble_builder.weighted_average_ensemble(predictions_dict)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def handle_outliers(self, X, method='isolation_forest', **kwargs):
        """
        Обрабатывает выбросы.
        
        Parameters:
        -----------
        X : array-like
            Данные
        method : str
            Метод: 'isolation_forest', 'winsorize', 'robust_scaler'
        **kwargs
            Дополнительные параметры
        
        Returns:
        --------
        processed_data : array-like
            Обработанные данные
        """
        if method == 'isolation_forest':
            contamination = kwargs.get('contamination', 0.1)
            outliers = self.outlier_handler.isolation_forest_detection(X, contamination)
            return X[~outliers]
        
        elif method == 'winsorize':
            lower = kwargs.get('lower_percentile', 5)
            upper = kwargs.get('upper_percentile', 95)
            return self.outlier_handler.winsorize(X, lower, upper)
        
        elif method == 'robust_scaler':
            return self.outlier_handler.robust_scaler_transform(X)
        
        else:
            raise ValueError(f"Неизвестный метод: {method}")

