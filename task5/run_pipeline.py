"""
Основной скрипт для запуска полного пайплайна глубокого обучения для временных рядов.
"""

import os
os.environ['TCL_LIBRARY'] = "C:/Program Files/Python313/tcl/tcl8.6"
os.environ['TK_LIBRARY'] = "C:/Program Files/Python313/tcl/tk8.6"
import sys
import numpy as np
import pandas as pd
import torch
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Импорты наших модулей
from preprocessing import TimeSeriesPreprocessor
from feature_engineering import FeatureEngineer
from models import create_all_models
from training import train_model, ModelTrainer
from evaluation import MetricsCalculator, ModelEvaluator
from diagnostics import ModelDiagnostics

# Настройка
torch.manual_seed(42)
np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используется устройство: {device}")


class DeepLearningPipeline:
    """Полный пайплайн для глубокого обучения временных рядов."""
    
    def __init__(self, data_path, target_column='Weekly_Sales', date_column='Date',
                 output_dir='results', lookback=336, horizon=48):
        self.data_path = data_path
        self.target_column = target_column
        self.date_column = date_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.lookback = lookback
        self.horizon = horizon
        
        # Инициализация компонентов
        self.preprocessor = TimeSeriesPreprocessor(scaler_type='standard')
        self.feature_engineer = FeatureEngineer()
        self.metrics_calc = MetricsCalculator()
        self.evaluator = ModelEvaluator()
        self.diagnostics = ModelDiagnostics()
        
        # Данные
        self.df = None
        self.y = None
        self.dates = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.preprocessor_info = None
        
        # Результаты
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Загружает данные."""
        print("Загрузка данных...")
        self.df = pd.read_csv(self.data_path)
        
        # Обработка дат
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], utc=True)
        if self.df[self.date_column].dt.tz is not None:
            self.df[self.date_column] = self.df[self.date_column].dt.tz_localize(None)
        
        # Сортируем по дате
        self.df = self.df.sort_values(self.date_column)
        
        # Базовая информация о данных
        if self.target_column in self.df.columns:
            print(f"Диапазон {self.target_column}: [{self.df[self.target_column].min():.2f}, {self.df[self.target_column].max():.2f}]")
            print(f"Среднее: {self.df[self.target_column].mean():.2f}, Стандартное отклонение: {self.df[self.target_column].std():.2f}")
        
        # Группируем по Store и Dept
        if 'Store' in self.df.columns and 'Dept' in self.df.columns:
            n_stores = self.df['Store'].nunique()
            n_depts = self.df['Dept'].nunique()
            print(f"Группировка по дате: {n_stores} магазинов, {n_depts} отделов")
            
            # Агрегируем по всем магазинам и отделам (используем среднее)
            self.df = self.df.groupby(self.date_column)[self.target_column].mean().reset_index()
            self.df = self.df.set_index(self.date_column).sort_index()
            
            if len(self.df) < 200:
                print(f"⚠️ После группировки: {len(self.df)} наблюдений (мало данных)")
        else:
            if self.date_column in self.df.columns:
                self.df = self.df.set_index(self.date_column)
        
        self.y = self.df[self.target_column].dropna()
        self.dates = self.y.index
        
        print(f"Загружено {len(self.y)} наблюдений")
        print(f"Период: {self.dates.min()} - {self.dates.max()}")
        
        # Определяем частоту данных
        if len(self.dates) > 1:
            freq = pd.infer_freq(self.dates)
            if freq is None:
                # Пытаемся определить частоту по разнице дат
                time_diff = (self.dates[1] - self.dates[0]).days
                if time_diff == 7:
                    freq = 'W'
                elif time_diff == 1:
                    freq = 'D'
                else:
                    freq = f'{time_diff}D'
            print(f"Частота данных: {freq}")
        
        # Проверяем достаточность данных
        if len(self.y) < 100:
            warnings.warn(f"Мало данных: {len(self.y)} наблюдений. "
                        f"Рекомендуется минимум 1000 для глубокого обучения.")
        
        return self
    
    def preprocess_data(self, apply_transform='boxcox'):
        """Предобработка данных."""
        print("Предобработка данных...")
        
        # Адаптируем lookback в зависимости от объема данных
        n = len(self.y)
        min_required = self.lookback + self.horizon
        
        if n < min_required:
            # Автоматически уменьшаем lookback
            new_lookback = max(24, min(n - self.horizon - 10, self.lookback))
            if new_lookback != self.lookback:
                print(f"Адаптация lookback: {self.lookback} -> {new_lookback} "
                      f"(доступно {n} точек, нужно минимум {min_required})")
                self.lookback = new_lookback
        
        # Подготовка данных
        (X_train, y_train, train_dates), \
        (X_val, y_val, val_dates), \
        (X_test, y_test, test_dates), \
        self.preprocessor_info = self.preprocessor.prepare_data(
            self.y, 
            lookback=self.lookback,
            horizon=self.horizon,
            apply_transform=apply_transform
        )
        
        self.train_data = (X_train, y_train, train_dates)
        self.val_data = (X_val, y_val, val_dates)
        self.test_data = (X_test, y_test, test_dates)
        
        # Обновляем lookback и horizon из preprocessor_info (могут быть уменьшены автоматически)
        self.lookback = self.preprocessor_info.get('lookback', self.lookback)
        self.horizon = self.preprocessor_info.get('horizon', self.horizon)
        
        print(f"Train: {len(X_train)} последовательностей")
        print(f"Val: {len(X_val)} последовательностей")
        print(f"Test: {len(X_test)} последовательностей")
        print(f"Фактические параметры: lookback={self.lookback}, horizon={self.horizon}")
        
        return self
    
    def train_all_models(self, model_names=None, epochs=100, batch_size=32):
        """Обучает все модели."""
        if model_names is None:
            # Базовые и рекуррентные модели (быстрые)
            model_names = ['MLP', 'TCN', 'N-BEATS', 'N-HiTS', 'RNN', 'LSTM', 'GRU', 
                          'BiLSTM', 'BiGRU', 'Transformer', 'CNN-LSTM', 'CNN-GRU',
                          'DLinear', 'NLinear', 'Naive', 'SeasonalNaive']
            # Продвинутые модели (медленные, можно раскомментировать при необходимости)
            # model_names += ['Informer', 'Autoformer', 'PatchTST', 'TFT', 'TCN-Attention', 'LSTM-AE']
        
        X_train, y_train, _ = self.train_data
        X_val, y_val, _ = self.val_data
        
        input_size = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
        
        print(f"\nОбучение {len(model_names)} моделей...")
        print(f"Input size: {input_size}, Horizon: {self.horizon}")
        
        for model_name in model_names:
            try:
                print(f"\n{'='*60}")
                print(f"Обучение модели: {model_name}")
                print(f"{'='*60}")
                
                # Создаем модель
                model = create_all_models(input_size, horizon=self.horizon, lookback=self.lookback)[model_name]
                
                # Параметры обучения
                trainer_kwargs = {
                    'loss_fn': 'mse+mae',
                    'optimizer': 'adam',
                    'lr': 1e-3,
                    'weight_decay': 1e-4,
                    'gradient_clip': 1.0,
                    'early_stopping_patience': 15,
                    'reduce_lr_patience': 10,
                }
                
                # Обучаем
                start_time = time.time()
                trainer, train_losses, val_losses = train_model(
                    model, X_train, y_train, X_val, y_val,
                    batch_size=batch_size,
                    epochs=epochs,
                    device=device,
                    verbose=True,  # Включаем вывод для консольного режима
                    **trainer_kwargs
                )
                train_time = time.time() - start_time
                
                # Предсказания на валидации
                from training import TimeSeriesDataset
                from torch.utils.data import DataLoader
                
                # ОТЛАДКА: Проверяем исходные данные перед созданием DataLoader
                print(f"🔍 ОТЛАДКА для {model_name}:")
                print(f"  X_val форма: {X_val.shape}, диапазон: [{X_val.min():.6f}, {X_val.max():.6f}]")
                print(f"  y_val форма: {y_val.shape}, диапазон: [{y_val.min():.6f}, {y_val.max():.6f}]")
                print(f"  y_val первые 3 строки:\n{y_val[:3] if len(y_val) >= 3 else y_val}")
                print(f"  Уникальные значения в y_val (первые 10): {np.unique(y_val.flatten())[:10]}")
                
                val_dataset = TimeSeriesDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                y_pred_val, y_true_val = trainer.predict(val_loader)
                
                # Проверка на NaN в прогнозах
                if np.any(np.isnan(y_pred_val)) or np.any(np.isinf(y_pred_val)):
                    print(f"⚠️ ОШИБКА: NaN/inf в прогнозах для {model_name}!")
                    continue
                
                # Обратная трансформация
                if len(y_pred_val.shape) > 1:
                    y_pred_val_flat = y_pred_val[:, 0]  # Первый шаг горизонта
                    y_true_val_flat = y_true_val[:, 0]
                else:
                    y_pred_val_flat = y_pred_val
                    y_true_val_flat = y_true_val
                
                # Обратная нормализация
                y_pred_val_scaled = self.preprocessor.inverse_transform(y_pred_val_flat)
                y_true_val_scaled = self.preprocessor.inverse_transform(y_true_val_flat)
                
                if np.any(np.isnan(y_pred_val_scaled)) or np.any(np.isinf(y_pred_val_scaled)):
                    y_pred_val_scaled = np.nan_to_num(y_pred_val_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
                
                # Обратная трансформация (Box-Cox или log)
                if self.preprocessor_info['transform'] == 'boxcox':
                    lambda_val = self.preprocessor_info.get('lambda_boxcox')
                    y_pred_val_orig = self.preprocessor.inverse_boxcox(y_pred_val_scaled, lambda_val)
                    y_true_val_orig = self.preprocessor.inverse_boxcox(y_true_val_scaled, lambda_val)
                elif self.preprocessor_info['transform'] == 'log':
                    y_pred_val_orig = self.preprocessor.inverse_log(y_pred_val_scaled)
                    y_true_val_orig = self.preprocessor.inverse_log(y_true_val_scaled)
                else:
                    y_pred_val_orig = y_pred_val_scaled
                    y_true_val_orig = y_true_val_scaled
                
                # Подготовка train данных для метрик
                if len(self.train_data[1].shape) > 1:
                    y_train_flat = self.train_data[1][:, 0]
                else:
                    y_train_flat = self.train_data[1]
                
                y_train_scaled = self.preprocessor.inverse_transform(y_train_flat)
                
                if self.preprocessor_info['transform'] == 'boxcox':
                    y_train_orig = self.preprocessor.inverse_boxcox(
                        y_train_scaled, self.preprocessor_info['lambda_boxcox']
                    )
                elif self.preprocessor_info['transform'] == 'log':
                    y_train_orig = self.preprocessor.inverse_log(y_train_scaled)
                else:
                    y_train_orig = y_train_scaled
                
                # Вычисление метрик
                metrics = self.metrics_calc.calculate_all_metrics(
                    y_true_val_orig, y_pred_val_orig, y_train_orig, seasonality=7
                )
                
                # Финальная проверка на NaN
                if np.any(np.isnan(y_pred_val_orig)) or np.any(np.isinf(y_pred_val_orig)):
                    print(f"⚠️ ОШИБКА: NaN/inf в финальных прогнозах для {model_name}!")
                    continue
                
                # Сохраняем результаты
                self.models[model_name] = trainer
                self.results[model_name] = {
                    'metrics': metrics,
                    'time': train_time,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'y_pred': y_pred_val_orig,
                    'y_true': y_true_val_orig,
                }
                
                print(f"✅ {model_name}: MASE={metrics.get('MASE', 'N/A'):.4f}, "
                      f"MAE={metrics.get('MAE', 'N/A'):.2f}, "
                      f"RMSE={metrics.get('RMSE', 'N/A'):.2f} "
                      f"({train_time:.1f}с)")
                
            except Exception as e:
                print(f"Ошибка при обучении {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return self
    
    def evaluate_all_models(self):
        """Оценивает все модели на тестовых данных."""
        print("\nОценка моделей на тестовых данных...")
        
        X_test, y_test, _ = self.test_data
        
        for model_name, trainer in self.models.items():
            try:
                from training import TimeSeriesDataset
                from torch.utils.data import DataLoader
                test_dataset = TimeSeriesDataset(X_test, y_test)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
                
                y_pred_test, y_true_test = trainer.predict(test_loader)
                
                # Обратная трансформация (аналогично валидации)
                if len(y_pred_test.shape) > 1:
                    y_pred_test_flat = y_pred_test[:, 0]  # Первый шаг
                    y_true_test_flat = y_true_test[:, 0]  # Первый шаг
                else:
                    y_pred_test_flat = y_pred_test
                    y_true_test_flat = y_true_test
                
                if self.preprocessor_info['transform'] == 'boxcox':
                    y_pred_test_scaled = self.preprocessor.inverse_transform(y_pred_test_flat)
                    y_true_test_scaled = self.preprocessor.inverse_transform(y_true_test_flat)
                    y_pred_test_orig = self.preprocessor.inverse_boxcox(
                        y_pred_test_scaled,
                        self.preprocessor_info['lambda_boxcox']
                    )
                    y_true_test_orig = self.preprocessor.inverse_boxcox(
                        y_true_test_scaled,
                        self.preprocessor_info['lambda_boxcox']
                    )
                elif self.preprocessor_info['transform'] == 'log':
                    y_pred_test_scaled = self.preprocessor.inverse_transform(y_pred_test_flat)
                    y_true_test_scaled = self.preprocessor.inverse_transform(y_true_test_flat)
                    y_pred_test_orig = self.preprocessor.inverse_log(y_pred_test_scaled)
                    y_true_test_orig = self.preprocessor.inverse_log(y_true_test_scaled)
                else:
                    y_pred_test_orig = self.preprocessor.inverse_transform(y_pred_test_flat)
                    y_true_test_orig = self.preprocessor.inverse_transform(y_true_test_flat)
                
                # Метрики
                if len(self.train_data[1].shape) > 1:
                    y_train_flat = self.train_data[1][:, 0]  # Первый шаг
                else:
                    y_train_flat = self.train_data[1]
                
                y_train_scaled = self.preprocessor.inverse_transform(y_train_flat)
                
                if self.preprocessor_info['transform'] == 'boxcox':
                    y_train_orig = self.preprocessor.inverse_boxcox(
                        y_train_scaled, self.preprocessor_info['lambda_boxcox']
                    )
                elif self.preprocessor_info['transform'] == 'log':
                    y_train_orig = self.preprocessor.inverse_log(y_train_scaled)
                else:
                    y_train_orig = y_train_scaled
                
                metrics = self.metrics_calc.calculate_all_metrics(
                    y_true_test_orig, y_pred_test_orig, y_train_orig, seasonality=7
                )
                
                self.results[model_name]['test_metrics'] = metrics
                self.results[model_name]['test_y_pred'] = y_pred_test_orig
                self.results[model_name]['test_y_true'] = y_true_test_orig
                
            except Exception as e:
                print(f"Ошибка при оценке {model_name}: {e}")
        
        return self
    
    def create_diagnostics(self):
        """Создает диагностические графики."""
        print("\nСоздание диагностических графиков...")
        
        for model_name, result in self.results.items():
            try:
                # Learning curves
                if 'train_losses' in result and 'val_losses' in result:
                    self.diagnostics.plot_learning_curves(
                        result['train_losses'],
                        result['val_losses'],
                        save_path=str(self.output_dir / f"{model_name}_learning_curves.png")
                    )
                
                # Прогнозы
                if 'y_pred' in result and 'y_true' in result:
                    self.diagnostics.plot_predictions(
                        result['y_true'],
                        result['y_pred'],
                        save_path=str(self.output_dir / f"{model_name}_predictions.png")
                    )
                
                # Остатки
                if 'y_pred' in result and 'y_true' in result:
                    residuals = result['y_true'] - result['y_pred']
                    self.diagnostics.plot_residual_analysis(
                        residuals,
                        save_path=str(self.output_dir / f"{model_name}_residuals.png")
                    )
                
            except Exception as e:
                print(f"Ошибка при создании диагностики для {model_name}: {e}")
        
        # Сравнение моделей
        try:
            self.diagnostics.plot_model_comparison(
                self.results,
                metric='MASE',
                save_path=str(self.output_dir / "model_comparison.png")
            )
        except Exception as e:
            print(f"Ошибка при сравнении моделей: {e}")
        
        return self
    
    def save_results(self):
        """Сохраняет результаты."""
        print("\nСохранение результатов...")
        
        # Сводная таблица
        comparison_table = self.evaluator.create_comparison_table(self.results, sort_by='MASE')
        comparison_table.to_csv(self.output_dir / "model_comparison.csv", index=False)
        print(f"Сводная таблица сохранена: {self.output_dir / 'model_comparison.csv'}")
        
        # Детальные результаты
        results_summary = {}
        for model_name, result in self.results.items():
            results_summary[model_name] = {
                'metrics': result.get('metrics', {}),
                'test_metrics': result.get('test_metrics', {}),
                'time': result.get('time', 0),
            }
        
        with open(self.output_dir / "results_summary.json", 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Результаты сохранены: {self.output_dir / 'results_summary.json'}")
        
        return self


def main():
    """Главная функция."""
    # Параметры
    data_path = os.path.join(os.path.dirname(__file__), 'New_final.csv')
    output_dir = 'results'
    # Адаптивные параметры - будут скорректированы в зависимости от объема данных
    lookback = 336  # ~14 дней при недельной частоте (будет уменьшено, если данных мало)
    horizon = 48  # ~2 дня
    
    # Создаем пайплайн
    pipeline = DeepLearningPipeline(
        data_path=data_path,
        target_column='Weekly_Sales',
        date_column='Date',
        output_dir=output_dir,
        lookback=lookback,
        horizon=horizon
    )
    
    # Запускаем пайплайн
    pipeline.load_data() \
            .preprocess_data(apply_transform='boxcox') \
            .train_all_models(epochs=50, batch_size=32) \
            .evaluate_all_models() \
            .create_diagnostics() \
            .save_results()
    
    print("\n" + "="*60)
    print("Пайплайн завершен успешно!")
    print("="*60)


if __name__ == "__main__":
    main()

