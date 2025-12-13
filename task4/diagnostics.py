"""
Этап 6: Диагностика моделей
Остатки, feature importance, PDP, визуализация.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP не установлен. Установите: pip install shap")

try:
    from pdpbox import pdp
    PDPBOX_AVAILABLE = True
except ImportError:
    PDPBOX_AVAILABLE = False
    print("pdpbox не установлен. Установите: pip install pdpbox")


class ModelDiagnostics:
    """Класс для диагностики моделей временных рядов."""
    
    def __init__(self):
        self.residuals = {}
        self.feature_importance = {}
    
    def calculate_residuals(self, y_true, y_pred, model_name='model'):
        """Вычисляет остатки модели."""
        residuals = y_true - y_pred
        self.residuals[model_name] = residuals
        return residuals
    
    def test_stationarity(self, series, model_name='model'):
        """
        Тестирует стационарность остатков (ADF и KPSS тесты).
        
        Parameters:
        -----------
        series : array-like
            Временной ряд (остатки)
        model_name : str
            Название модели
        
        Returns:
        --------
        dict
            Результаты тестов
        """
        results = {}
        
        # ADF тест
        try:
            adf_result = adfuller(series.dropna() if isinstance(series, pd.Series) else series)
            results['ADF'] = {
                'statistic': adf_result[0],
                'pvalue': adf_result[1],
                'is_stationary': adf_result[1] < 0.05
            }
        except Exception as e:
            results['ADF'] = {'error': str(e)}
        
        # KPSS тест
        try:
            kpss_result = kpss(series.dropna() if isinstance(series, pd.Series) else series)
            results['KPSS'] = {
                'statistic': kpss_result[0],
                'pvalue': kpss_result[1],
                'is_stationary': kpss_result[1] > 0.05
            }
        except Exception as e:
            results['KPSS'] = {'error': str(e)}
        
        return results
    
    def plot_acf(self, residuals, model_name='model', lags=40, figsize=(10, 6)):
        """Строит ACF остатков."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(residuals, pd.Series):
            residuals = residuals.dropna()
        else:
            residuals = residuals[~np.isnan(residuals)]
        
        acf_values = acf(residuals, nlags=lags, fft=True)
        
        ax.bar(range(len(acf_values)), acf_values)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=1.96/np.sqrt(len(residuals)), color='red', linestyle='--', label='95% CI')
        ax.axhline(y=-1.96/np.sqrt(len(residuals)), color='red', linestyle='--')
        ax.set_xlabel('Lag')
        ax.set_ylabel('ACF')
        ax.set_title(f'ACF остатков: {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_feature_importance(self, model, feature_names=None, model_name='model'):
        """
        Получает важность признаков модели.
        
        Parameters:
        -----------
        model : sklearn model
            Обученная модель
        feature_names : list, optional
            Названия признаков
        model_name : str
            Название модели
        
        Returns:
        --------
        pd.DataFrame
            Важность признаков
        """
        importance = None
        
        # Для моделей с feature_importances_
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        
        # Для линейных моделей используем абсолютные значения коэффициентов
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            # Обрабатываем многомерные массивы (для регрессии обычно одномерный)
            if coef.ndim > 1:
                # Если это массив формы (1, n_features), берем первую строку
                if coef.shape[0] == 1:
                    coef = coef[0]
                # Если это массив формы (n_features, 1), берем первый столбец
                elif coef.shape[1] == 1:
                    coef = coef[:, 0]
                else:
                    # Для многоклассового случая берем среднее по классам
                    coef = np.mean(np.abs(coef), axis=0)
            importance = np.abs(coef)
        
        if importance is not None:
            # Убеждаемся, что importance одномерный массив
            importance = np.atleast_1d(importance).flatten()
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            df_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.feature_importance[model_name] = df_importance
            return df_importance
        
        return None
    
    def plot_feature_importance(self, importance_df, model_name='model', top_n=20, figsize=(10, 8)):
        """Визуализирует важность признаков."""
        fig, ax = plt.subplots(figsize=figsize)
        
        top_features = importance_df.head(top_n)
        
        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Важность')
        ax.set_title(f'Топ-{top_n} признаков: {model_name}')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def shap_analysis(self, model, X, model_name='model', max_samples=100):
        """
        Анализ важности признаков через SHAP.
        
        Parameters:
        -----------
        model : sklearn model
            Обученная модель
        X : array-like
            Признаки
        model_name : str
            Название модели
        max_samples : int
            Максимальное количество образцов для анализа
        
        Returns:
        --------
        shap_values : array
            SHAP значения
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP не установлен")
        
        # Ограничиваем количество образцов для скорости
        if len(X) > max_samples:
            if isinstance(X, pd.DataFrame):
                X_sample = X.sample(max_samples, random_state=42)
            else:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
        else:
            X_sample = X
        
        # Создаем explainer
        try:
            explainer = shap.TreeExplainer(model)
        except:
            try:
                explainer = shap.LinearExplainer(model, X_sample)
            except:
                explainer = shap.KernelExplainer(model.predict, X_sample)
        
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values, X_sample
    
    def plot_shap_summary(self, shap_values, X_sample, model_name='model', figsize=(10, 8)):
        """Визуализирует SHAP summary plot."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP не установлен")
        
        fig = plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f'SHAP Summary: {model_name}')
        plt.tight_layout()
        return fig
    
    def plot_residuals_vs_time(self, residuals, dates=None, model_name='model', figsize=(12, 6)):
        """Визуализирует остатки по времени."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if dates is None:
            dates = range(len(residuals))
        
        ax.plot(dates, residuals, alpha=0.6)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_xlabel('Время')
        ax.set_ylabel('Остатки')
        ax.set_title(f'Остатки по времени: {model_name}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions_vs_actual(self, y_true, y_pred, dates=None, model_name='model', figsize=(12, 6)):
        """Визуализирует предсказания vs факт."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if dates is None:
            dates = range(len(y_true))
        
        ax.plot(dates, y_true, label='Факт', alpha=0.7)
        ax.plot(dates, y_pred, label='Прогноз', alpha=0.7)
        ax.set_xlabel('Время')
        ax.set_ylabel('Значение')
        ax.set_title(f'Факт vs Прогноз: {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_errors_by_horizon(self, errors_by_horizon, model_name='model', figsize=(10, 6)):
        """Визуализирует ошибки по горизонтам прогнозирования."""
        fig, ax = plt.subplots(figsize=figsize)
        
        horizons = range(1, len(errors_by_horizon) + 1)
        ax.plot(horizons, errors_by_horizon, marker='o')
        ax.set_xlabel('Горизонт прогнозирования')
        ax.set_ylabel('Ошибка (MAE)')
        ax.set_title(f'Ошибки по горизонтам: {model_name}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


