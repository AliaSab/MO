"""
Модуль для диагностики моделей.
Включает визуализацию learning curves, attention maps, residual analysis и др.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelDiagnostics:
    """Класс для диагностики моделей."""
    
    def __init__(self):
        pass
    
    def plot_learning_curves(self, train_losses, val_losses, save_path=None):
        """Визуализация кривых обучения."""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_predictions(self, y_true, y_pred, dates=None, title='Прогнозы модели',
                        save_path=None, show_residuals=True):
        """Визуализация прогнозов и фактических значений."""
        n_plots = 2 if show_residuals else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        # Основной график
        ax = axes[0]
        if dates is not None:
            ax.plot(dates, y_true, 'b-', label='Факт', linewidth=2, alpha=0.7)
            ax.plot(dates, y_pred, 'r--', label='Прогноз', linewidth=2, alpha=0.7)
        else:
            ax.plot(y_true, 'b-', label='Факт', linewidth=2, alpha=0.7)
            ax.plot(y_pred, 'r--', label='Прогноз', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Время', fontsize=12)
        ax.set_ylabel('Значение', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # График остатков
        if show_residuals:
            ax2 = axes[1]
            residuals = y_true - y_pred
            if dates is not None:
                ax2.plot(dates, residuals, 'g-', linewidth=1.5, alpha=0.7)
            else:
                ax2.plot(residuals, 'g-', linewidth=1.5, alpha=0.7)
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
            ax2.set_xlabel('Время', fontsize=12)
            ax2.set_ylabel('Остатки', fontsize=12)
            ax2.set_title('Остатки модели', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_residual_analysis(self, residuals, save_path=None):
        """Анализ остатков: гистограмма, Q-Q plot, ACF."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Гистограмма остатков
        axes[0, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Остатки', fontsize=11)
        axes[0, 0].set_ylabel('Частота', fontsize=11)
        axes[0, 0].set_title('Распределение остатков', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (нормальность)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ACF остатков
        try:
            from statsmodels.tsa.stattools import acf
            lags = min(40, len(residuals) // 4)
            acf_values = acf(residuals, nlags=lags, fft=True)
            axes[1, 0].bar(range(len(acf_values)), acf_values, width=0.5, edgecolor='black', alpha=0.7)
            axes[1, 0].axhline(y=0, color='r', linestyle='-', linewidth=1)
            axes[1, 0].set_xlabel('Лаг', fontsize=11)
            axes[1, 0].set_ylabel('ACF', fontsize=11)
            axes[1, 0].set_title('Автокорреляция остатков', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        except:
            axes[1, 0].text(0.5, 0.5, 'ACF недоступен', ha='center', va='center')
        
        # Остатки vs прогнозы
        axes[1, 1].scatter(residuals[:-1], residuals[1:], alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Остатки(t)', fontsize=11)
        axes[1, 1].set_ylabel('Остатки(t+1)', fontsize=11)
        axes[1, 1].set_title('Корреляция остатков', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, results_dict, metric='MASE', save_path=None):
        """Сравнение моделей по метрике."""
        model_names = list(results_dict.keys())
        metric_values = [results_dict[name].get('metrics', {}).get(metric, np.nan) 
                         for name in model_names]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(model_names, metric_values, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'Сравнение моделей по {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Добавляем значения на столбцы
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            if not np.isnan(value):
                ax.text(value, i, f' {value:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_distribution(self, errors_dict, save_path=None):
        """Распределение ошибок для разных моделей."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model_name, errors in errors_dict.items():
            ax.hist(errors, bins=50, alpha=0.5, label=model_name, edgecolor='black')
        
        ax.set_xlabel('Ошибка', fontsize=12)
        ax.set_ylabel('Частота', fontsize=12)
        ax.set_title('Распределение ошибок моделей', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_nbeats_components(self, model, X_sample, save_path=None):
        """Анализ компонент N-BEATS (тренд/сезонность)."""
        # Это требует доступа к внутренним компонентам модели
        # Упрощенная версия
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Анализ компонент N-BEATS требует доступа к внутренним блокам модели',
                ha='center', va='center', fontsize=12)
        ax.set_title('Компоненты N-BEATS', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_diagnostic_report(self, model_name, y_true, y_pred, train_losses=None, 
                                val_losses=None, save_dir=None):
        """Создает полный диагностический отчет."""
        residuals = y_true - y_pred
        
        # Сохраняем графики
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # Learning curves
            if train_losses and val_losses:
                self.plot_learning_curves(train_losses, val_losses, 
                                        save_path=f"{save_dir}/{model_name}_learning_curves.png")
            
            # Прогнозы
            self.plot_predictions(y_true, y_pred, 
                                save_path=f"{save_dir}/{model_name}_predictions.png")
            
            # Анализ остатков
            self.plot_residual_analysis(residuals, 
                                       save_path=f"{save_dir}/{model_name}_residuals.png")
        
        return {
            'model_name': model_name,
            'residuals_mean': np.mean(residuals),
            'residuals_std': np.std(residuals),
            'residuals_skew': stats.skew(residuals),
            'residuals_kurtosis': stats.kurtosis(residuals),
        }


















