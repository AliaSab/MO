"""
Модуль предобработки временных рядов для глубокого обучения.
Включает трансформации (лог, Бокс-Кокс), нормализацию и создание последовательностей.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import boxcox, boxcox_normmax
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesPreprocessor:
    """Класс для предобработки временных рядов для DL моделей."""
    
    def __init__(self, scaler_type='standard'):
        """
        Args:
            scaler_type: 'standard' или 'minmax'
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.lambda_boxcox = None
        self.transformations = []
        self.is_fitted = False
        
    def log_transform(self, y):
        """Применяет лог-трансформацию, если все значения > 0."""
        if (y > 0).all():
            y_log = np.log(y + 1e-8)  # Добавляем небольшое значение для стабильности
            self.transformations.append('log')
            return y_log, True
        else:
            warnings.warn("Не все значения > 0, лог-трансформация не применена")
            return y, False
    
    def boxcox_transform(self, y, brack=(-2, 2)):
        """Применяет преобразование Бокса-Кокса."""
        y_values = y.values if isinstance(y, pd.Series) else y
        
        if not (y_values > 0).all():
            warnings.warn("Не все значения > 0, Бокса-Кокса не применена")
            return y_values, None, False
        
        # Проверяем вариацию данных
        if np.std(y_values) < 1e-6 or np.var(y_values) < 1e-12:
            warnings.warn("Слишком малая вариация данных для Box-Cox, используем log-трансформацию")
            y_log = np.log(y_values + 1e-8)
            self.transformations.append('log')
            return y_log, None, False
        
        try:
            # Находим оптимальный lambda с обработкой ошибок
            # Ограничиваем диапазон lambda для стабильности (слишком отрицательные lambda могут быть проблематичными)
            try:
                # Ограничиваем диапазон lambda от -1 до 1 для стабильности
                lambda_opt = boxcox_normmax(y_values, brack=(-1, 1))
            except Exception as e:
                # Если не удалось найти оптимальный lambda, пробуем другие варианты
                warnings.warn(f"Не удалось найти оптимальный lambda для Box-Cox: {e}. Пробуем альтернативные методы.")
                
                # Пробуем более широкий диапазон, но все еще ограниченный
                try:
                    lambda_opt = boxcox_normmax(y_values, brack=(-2, 2))
                except:
                    # Если и это не работает, используем log-трансформацию
                    warnings.warn("Используем log-трансформацию вместо Box-Cox")
                    y_log = np.log(y_values + 1e-8)
                    self.transformations.append('log')
                    return y_log, None, False
            
            # Проверяем, что lambda не слишком экстремальный
            if abs(lambda_opt) > 2:
                warnings.warn(f"Lambda={lambda_opt:.4f} слишком экстремальный. Используем log-трансформацию.")
                y_log = np.log(y_values + 1e-8)
                self.transformations.append('log')
                return y_log, None, False
            
            # Применяем преобразование
            y_bc = boxcox(y_values, lmbda=lambda_opt)
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: Если после Box-Cox все значения стали одинаковыми, используем log
            if np.std(y_bc) < 1e-10 or len(np.unique(y_bc)) <= 3:
                warnings.warn(f"Box-Cox с lambda={lambda_opt:.4f} привел к потере вариации "
                            f"(std={np.std(y_bc):.10f}, unique={len(np.unique(y_bc))}). "
                            f"Используем log-трансформацию вместо Box-Cox.")
                y_log = np.log(y_values + 1e-8)
                self.transformations.append('log')
                return y_log, None, False
            
            # Проверяем, что вариация сохранилась
            if np.std(y_bc) / np.std(y_values) < 0.01:  # Вариация уменьшилась более чем в 100 раз
                warnings.warn(f"Box-Cox с lambda={lambda_opt:.4f} сильно уменьшил вариацию "
                            f"(было {np.std(y_values):.2f}, стало {np.std(y_bc):.2f}). "
                            f"Используем log-трансформацию вместо Box-Cox.")
                y_log = np.log(y_values + 1e-8)
                self.transformations.append('log')
                return y_log, None, False
            
            self.lambda_boxcox = lambda_opt
            self.transformations.append('boxcox')
            return y_bc, lambda_opt, True
            
        except Exception as e:
            warnings.warn(f"Ошибка при применении Box-Cox: {e}. Используем log-трансформацию.")
            y_log = np.log(y_values + 1e-8)
            self.transformations.append('log')
            return y_log, None, False
    
    def inverse_boxcox(self, y_transformed, lambda_val):
        """Обратное преобразование Бокса-Кокса."""
        if lambda_val is None:
            return y_transformed
        
        y_transformed = np.asarray(y_transformed)
        
        # Обработка NaN и inf
        if np.any(np.isnan(y_transformed)) or np.any(np.isinf(y_transformed)):
            warnings.warn("Обнаружены NaN или inf в данных для обратной Box-Cox трансформации")
            y_transformed = np.nan_to_num(y_transformed, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if lambda_val == 0:
            result = np.exp(y_transformed)
        else:
            # Вычисляем аргумент для обратной трансформации
            arg = lambda_val * y_transformed + 1
            
            # Проверяем на отрицательные значения
            if np.any(arg <= 0):
                warnings.warn("Отрицательные значения при обратном Box-Cox, используем exp")
                result = np.exp(y_transformed)
            else:
                result = np.power(arg, 1 / lambda_val)
        
        # Проверка на NaN и inf в результате
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            warnings.warn("NaN или inf после обратной Box-Cox трансформации, используем exp")
            result = np.exp(y_transformed)
            result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return result
    
    def inverse_log(self, y_log):
        """Обратное лог-преобразование."""
        return np.exp(y_log) - 1e-8
    
    def fit_scaler(self, y):
        """Обучает скейлер на данных."""
        y_2d = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        self.scaler.fit(y_2d)
        self.is_fitted = True
        
    def transform(self, y):
        """Применяет нормализацию."""
        if not self.is_fitted:
            raise ValueError("Скейлер не обучен. Сначала вызовите fit_scaler.")
        y_2d = y.values.reshape(-1, 1) if isinstance(y, pd.Series) else y.reshape(-1, 1)
        y_scaled = self.scaler.transform(y_2d)
        return y_scaled.flatten()
    
    def inverse_transform(self, y_scaled):
        """Обратная нормализация."""
        if not self.is_fitted:
            raise ValueError("Скейлер не обучен.")
        
        y_2d = y_scaled.reshape(-1, 1) if len(y_scaled.shape) == 1 else y_scaled
        y_original = self.scaler.inverse_transform(y_2d)
        return y_original.flatten()
    
    def create_sequences(self, data, lookback=336, horizon=48, stride=1):
        """
        Создает последовательности для обучения.
        
        Args:
            data: массив данных (1D или 2D)
            lookback: размер окна истории
            horizon: горизонт прогноза
            stride: шаг между последовательностями (автоматически увеличивается, если данных мало)
            
        Returns:
            X: (n_samples, lookback, n_features)
            y: (n_samples, horizon)
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Автоматически увеличиваем stride, если данных мало
        min_sequences_needed = 30
        n_samples_initial = (len(data) - lookback - horizon) // stride + 1
        
        if n_samples_initial < min_sequences_needed and len(data) > lookback + horizon:
            optimal_stride = max(1, (len(data) - lookback - horizon) // min_sequences_needed)
            max_reasonable_stride = max(1, lookback // 4)
            stride = min(optimal_stride, max_reasonable_stride)
            if stride > 1:
                print(f"Увеличен stride до {stride} для создания большего количества последовательностей")
        
        n_samples = (len(data) - lookback - horizon) // stride + 1
        if n_samples <= 0:
            raise ValueError(f"Недостаточно данных для создания последовательностей. "
                           f"Нужно минимум {lookback + horizon} точек, получено {len(data)}")
        
        X = np.zeros((n_samples, lookback, data.shape[1]))
        y = np.zeros((n_samples, horizon))
        
        for i in range(n_samples):
            start_idx = i * stride
            end_idx = start_idx + lookback
            
            if end_idx + horizon > len(data):
                available_horizon = len(data) - end_idx
                if available_horizon > 0:
                    X[i] = data[start_idx:end_idx]
                    y[i, :available_horizon] = data[end_idx:end_idx + available_horizon, 0]
                else:
                    if end_idx <= len(data):
                        X[i] = data[start_idx:end_idx]
                    else:
                        X[i] = data[-lookback:] if len(data) >= lookback else data
            else:
                X[i] = data[start_idx:end_idx]
                y[i] = data[end_idx:end_idx + horizon, 0]
        
        if n_samples > 0 and np.all(y == 0):
            print(f"⚠️ ВНИМАНИЕ: Все значения y равны нулю!")
        
        return X, y
    
    def prepare_data(self, y, lookback=336, horizon=48, apply_transform='boxcox', 
                     train_ratio=0.7, val_ratio=0.15):
        """
        Полная подготовка данных: трансформация, нормализация, создание последовательностей.
        
        Returns:
            train_data, val_data, test_data: кортежи (X, y, dates)
            preprocessor_info: словарь с информацией о трансформациях
        """
        # Сохраняем исходные данные
        y_original = y.copy()
        dates = y.index if isinstance(y, pd.Series) else None
        
        # Применяем трансформацию ко всему ряду
        if apply_transform == 'log':
            y_transformed, _ = self.log_transform(y)
            lambda_val = None
            actual_transform = 'log'
        elif apply_transform == 'boxcox':
            y_transformed, lambda_val, success = self.boxcox_transform(y)
            # Проверяем, что Box-Cox действительно применен (может быть fallback на log)
            if not success:
                # Box-Cox не применен, проверяем, что было применено
                if 'log' in self.transformations:
                    actual_transform = 'log'
                else:
                    actual_transform = 'none'
            else:
                actual_transform = 'boxcox'
        else:
            y_transformed = y.values if isinstance(y, pd.Series) else y
            lambda_val = None
            actual_transform = 'none'
        
        if len(np.unique(y_transformed)) <= 3:
            print(f"⚠️ После трансформации осталось только {len(np.unique(y_transformed))} уникальных значений!")
        
        # Проверяем достаточность данных
        n = len(y_transformed)
        min_required = lookback + horizon
        
        # Проверяем достаточность данных и автоматически корректируем параметры
        # Цель: создать минимум 30-50 последовательностей для нормального обучения
        min_sequences_target = 40  # Целевое количество последовательностей
        
        if n < min_required:
            # Автоматически уменьшаем lookback ИЛИ horizon, если данных недостаточно
            # Уменьшаем horizon в первую очередь, так как он меньше влияет на качество
            if n < lookback + 10:
                # Очень мало данных - уменьшаем и lookback, и horizon
                # Вычисляем оптимальные параметры для создания достаточного количества последовательностей
                # Формула: n_sequences = (n - lookback - horizon) // stride + 1
                # Хотим минимум min_sequences_target последовательностей
                # При stride=1: n - lookback - horizon >= min_sequences_target - 1
                # lookback + horizon <= n - min_sequences_target + 1
                max_total = n - min_sequences_target + 1
                if max_total < 25:
                    # Очень мало данных - используем минимальные параметры
                    new_lookback = max(12, int(n * 0.4))  # 40% данных для lookback
                    new_horizon = max(1, min(12, n - new_lookback - 5))  # Остальное для horizon
                else:
                    # Пытаемся сохранить разумное соотношение
                    new_lookback = max(24, int(max_total * 0.7))  # 70% для lookback
                    new_horizon = max(1, max_total - new_lookback)  # Остальное для horizon
                
                warnings.warn(f"Очень мало данных для lookback={lookback}, horizon={horizon}. "
                            f"Уменьшаем до lookback={new_lookback}, horizon={new_horizon} "
                            f"(доступно {n} точек, нужно минимум {min_required}, "
                            f"целевое количество последовательностей: {min_sequences_target})")
                lookback = new_lookback
                horizon = new_horizon
            else:
                # Уменьшаем только horizon, но проверяем, что будет достаточно последовательностей
                estimated_sequences = n - lookback - horizon + 1
                if estimated_sequences < min_sequences_target:
                    # Уменьшаем horizon еще больше, чтобы создать больше последовательностей
                    new_horizon = max(1, n - lookback - min_sequences_target + 1)
                    warnings.warn(f"Недостаточно данных для создания {min_sequences_target} последовательностей. "
                                f"Уменьшаем horizon до {new_horizon} (доступно {n} точек, lookback={lookback}, "
                                f"будет создано ~{n - lookback - new_horizon + 1} последовательностей)")
                    horizon = new_horizon
                else:
                    warnings.warn(f"Недостаточно данных для horizon={horizon}. "
                                f"Уменьшаем до {n - lookback - min_sequences_target + 1} "
                                f"(доступно {n} точек, lookback={lookback})")
                    horizon = max(1, n - lookback - min_sequences_target + 1)
            min_required = lookback + horizon
        
        # Обучаем скейлер только на train части
        train_end = int(n * train_ratio)
        y_train_for_scaler = y_transformed[:train_end]
        
        self.fit_scaler(pd.Series(y_train_for_scaler) if isinstance(y_transformed, np.ndarray) else y_train_for_scaler)
        y_scaled = self.transform(pd.Series(y_transformed) if isinstance(y_transformed, np.ndarray) else y_transformed)
        
        if len(np.unique(y_scaled)) <= 3:
            print(f"⚠️ После нормализации осталось только {len(np.unique(y_scaled))} уникальных значений!")
        
        # Создаем последовательности из всего нормализованного ряда
        X_all, y_all = self.create_sequences(y_scaled, lookback, horizon)
        
        # Разбиваем последовательности на train/val/test
        n_sequences = len(X_all)
        train_seq_end = int(n_sequences * train_ratio)
        val_seq_end = int(n_sequences * (train_ratio + val_ratio))
        
        X_train = X_all[:train_seq_end]
        y_train_seq = y_all[:train_seq_end]
        X_val = X_all[train_seq_end:val_seq_end]
        y_val_seq = y_all[train_seq_end:val_seq_end]
        X_test = X_all[val_seq_end:]
        y_test_seq = y_all[val_seq_end:]
        
        # Проверка на критические проблемы
        if len(y_val_seq) > 0 and np.all(y_val_seq == 0):
            print(f"⚠️ КРИТИЧЕСКАЯ ПРОБЛЕМА: Все значения y_val_seq равны нулю!")
        
        # Даты для последовательностей (приблизительные)
        if dates is not None:
            # Даты соответствуют началу каждой последовательности
            train_dates = dates[lookback:lookback + len(y_train_seq)] if len(dates) > lookback else None
            val_start_idx = lookback + len(y_train_seq)
            val_dates = dates[val_start_idx:val_start_idx + len(y_val_seq)] if len(dates) > val_start_idx else None
            test_start_idx = val_start_idx + len(y_val_seq)
            test_dates = dates[test_start_idx:test_start_idx + len(y_test_seq)] if len(dates) > test_start_idx else None
        else:
            train_dates = None
            val_dates = None
            test_dates = None
        
        preprocessor_info = {
            'transform': actual_transform,
            'requested_transform': apply_transform,
            'lambda_boxcox': lambda_val,
            'scaler_type': self.scaler_type,
            'lookback': lookback,
            'horizon': horizon
        }
        
        return (X_train, y_train_seq, train_dates), \
               (X_val, y_val_seq, val_dates), \
               (X_test, y_test_seq, test_dates), \
               preprocessor_info

