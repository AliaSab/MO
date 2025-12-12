"""
Модуль расширенного feature engineering для временных рядов.

Реализует:
- Временные признаки (день недели, месяц, квартал)
- Циклические признаки через sin/cos
- Лаговые признаки (lag_1, lag_7, lag_30)
- Скользящие статистики (среднее, std, min, max)
- Признаки волатильности
- Праздничные/событийные метки
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import logging
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




class TimeSeriesFeatureEngineering:
    """Класс для создания признаков временных рядов."""
    
    def __init__(self):
        self.features_created = []
        self.feature_importance = {}
        
    def load_data(self, file_path: str, target_column: str = 'Weekly_Sales') -> pd.DataFrame:
        """Загрузка данных временного ряда."""
        try:
            logger.info(f"Загрузка данных из {file_path}")
            df = pd.read_csv(file_path)
            
            # Предполагаем, что есть колонка с датами
            date_columns = ['Date', 'date', 'DATE', 'Date_Time', 'datetime', 'timestamp']
            date_col = None
            
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                logger.info(f"Установлен индекс по колонке {date_col}")
            else:
                # Если нет колонки с датами, создаем индекс
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
                logger.warning("Колонка с датами не найдена, создан индекс по умолчанию")
            
            # Проверяем наличие целевой переменной
            if target_column not in df.columns:
                available_cols = df.columns.tolist()
                logger.error(f"Колонка '{target_column}' не найдена. Доступные колонки: {available_cols}")
                raise ValueError(f"Колонка '{target_column}' не найдена. Доступные колонки: {available_cols}")
            
            # Удаляем пропущенные значения
            initial_len = len(df)
            df = df.dropna(subset=[target_column])
            final_len = len(df)
            
            if initial_len != final_len:
                logger.warning(f"Удалено {initial_len - final_len} строк с пропущенными значениями")
            
            logger.info(f"Загружено {len(df)} наблюдений")
            logger.info(f"Период: {df.index.min()} - {df.index.max()}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"Файл не найден: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            logger.error("Файл пуст или содержит только заголовки")
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return None
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание временных признаков."""
        logger.info("Создание временных признаков...")
        
        try:
            df_features = df.copy()
            
            # Базовые временные признаки
            df_features['year'] = df_features.index.year
            df_features['month'] = df_features.index.month
            df_features['day'] = df_features.index.day
            df_features['day_of_week'] = df_features.index.dayofweek
            df_features['day_of_year'] = df_features.index.dayofyear
            df_features['week_of_year'] = df_features.index.isocalendar().week
            df_features['quarter'] = df_features.index.quarter
            
            # Циклические признаки для месяца (0-11 -> 0-2π)
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
            
            # Циклические признаки для дня недели (0-6 -> 0-2π)
            df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
            
            # Циклические признаки для дня года (0-365 -> 0-2π)
            df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
            df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
            
            # Циклические признаки для квартала
            df_features['quarter_sin'] = np.sin(2 * np.pi * df_features['quarter'] / 4)
            df_features['quarter_cos'] = np.cos(2 * np.pi * df_features['quarter'] / 4)
            
            # Бинарные признаки
            df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
            df_features['is_month_start'] = (df_features['day'] <= 7).astype(int)
            df_features['is_month_end'] = (df_features['day'] >= 25).astype(int)
            df_features['is_quarter_start'] = df_features['month'].isin([1, 4, 7, 10]).astype(int)
            df_features['is_year_start'] = (df_features['month'] == 1).astype(int)
            
            temporal_features = [
                'year', 'month', 'day', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
                'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
                'day_of_year_sin', 'day_of_year_cos', 'quarter_sin', 'quarter_cos',
                'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_year_start'
            ]
            
            self.features_created.extend(temporal_features)
            
            logger.info(f"Создано {len(temporal_features)} временных признаков")
            return df_features
            
        except Exception as e:
            logger.error(f"Ошибка создания временных признаков: {e}")
            return df
    
    def create_lag_features(self, df: pd.DataFrame, target_column: str, 
                           lags: list = [1, 7, 14, 30, 90]) -> pd.DataFrame:
        """Создание лаговых признаков."""
        logger.info(f"Создание лаговых признаков для лагов: {lags}")
        
        try:
            df_features = df.copy()
            
            for lag in lags:
                lag_col_name = f'lag_{lag}'
                df_features[lag_col_name] = df_features[target_column].shift(lag)
                self.features_created.append(lag_col_name)
            
            # Лаги для других важных колонок (если есть)
            other_columns = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
            for col in other_columns:
                if col in df_features.columns:
                    for lag in [1, 7]:
                        lag_col_name = f'{col}_lag_{lag}'
                        df_features[lag_col_name] = df_features[col].shift(lag)
                        self.features_created.append(lag_col_name)
            
            lag_features_count = len([f for f in self.features_created if 'lag' in f])
            logger.info(f"Создано {lag_features_count} лаговых признаков")
            return df_features
            
        except Exception as e:
            logger.error(f"Ошибка создания лаговых признаков: {e}")
            return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_column: str,
                               windows: list = [7, 14, 30, 90]) -> pd.DataFrame:
        """Создание скользящих статистик."""
        logger.info(f"Создание скользящих статистик для окон: {windows}")
        
        try:
            df_features = df.copy()
            
            for window in windows:
                # Скользящее среднее
                df_features[f'rolling_mean_{window}'] = df_features[target_column].rolling(window=window).mean()
                
                # Скользящее стандартное отклонение
                df_features[f'rolling_std_{window}'] = df_features[target_column].rolling(window=window).std()
                
                # Скользящий минимум
                df_features[f'rolling_min_{window}'] = df_features[target_column].rolling(window=window).min()
                
                # Скользящий максимум
                df_features[f'rolling_max_{window}'] = df_features[target_column].rolling(window=window).max()
                
                # Скользящий коэффициент вариации
                rolling_mean = df_features[f'rolling_mean_{window}']
                rolling_std = df_features[f'rolling_std_{window}']
                df_features[f'rolling_cv_{window}'] = rolling_std / (rolling_mean + 1e-8)  # Добавляем малое значение для избежания деления на ноль
                
                # Скользящий диапазон
                df_features[f'rolling_range_{window}'] = (
                    df_features[f'rolling_max_{window}'] - df_features[f'rolling_min_{window}']
                )
                
                rolling_features = [
                    f'rolling_mean_{window}', f'rolling_std_{window}', 
                    f'rolling_min_{window}', f'rolling_max_{window}',
                    f'rolling_cv_{window}', f'rolling_range_{window}'
                ]
                
                self.features_created.extend(rolling_features)
            
            rolling_features_count = len([f for f in self.features_created if 'rolling' in f])
            logger.info(f"Создано {rolling_features_count} скользящих признаков")
            return df_features
            
        except Exception as e:
            logger.error(f"Ошибка создания скользящих признаков: {e}")
            return df
    
    def create_volatility_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Создание признаков волатильности."""
        logger.info("Создание признаков волатильности...")
        
        try:
            df_features = df.copy()
            
            # Скользящая волатильность (стандартное отклонение изменений)
            windows = [7, 14, 30]
            
            for window in windows:
                # Изменения (returns)
                returns = df_features[target_column].pct_change()
                
                # Скользящая волатильность
                df_features[f'volatility_{window}'] = returns.rolling(window=window).std()
                
                # Скользящий коэффициент вариации изменений
                rolling_mean_returns = returns.rolling(window=window).mean()
                rolling_std_returns = returns.rolling(window=window).std()
                df_features[f'returns_cv_{window}'] = rolling_std_returns / (rolling_mean_returns.abs() + 1e-8)
                
                self.features_created.extend([f'volatility_{window}', f'returns_cv_{window}'])
            
            # GARCH-подобные признаки (упрощенные)
            returns = df_features[target_column].pct_change()
            df_features['returns_squared'] = returns ** 2
            df_features['abs_returns'] = returns.abs()
            
            # Скользящие признаки для квадратов изменений
            for window in [7, 14]:
                df_features[f'rolling_squared_returns_{window}'] = df_features['returns_squared'].rolling(window=window).mean()
                df_features[f'rolling_abs_returns_{window}'] = df_features['abs_returns'].rolling(window=window).mean()
                
                self.features_created.extend([
                    f'rolling_squared_returns_{window}', f'rolling_abs_returns_{window}'
                ])
            
            self.features_created.extend(['returns_squared', 'abs_returns'])
            
            volatility_features_count = len([f for f in self.features_created if 'volatility' in f or 'returns' in f])
            logger.info(f"Создано {volatility_features_count} признаков волатильности")
            return df_features
            
        except Exception as e:
            logger.error(f"Ошибка создания признаков волатильности: {e}")
            return df
    
    def create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков праздников и событий."""
        logger.info("Создание признаков праздников...")
        
        try:
            df_features = df.copy()
            
            # Российские праздники (примерные даты)
            holidays_2020 = [
                '2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05', '2020-01-06', '2020-01-07', '2020-01-08',
                '2020-02-23', '2020-03-08', '2020-05-01', '2020-05-09', '2020-06-12', '2020-11-04'
            ]
            
            holidays_2021 = [
                '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07', '2021-01-08',
                '2021-02-23', '2021-03-08', '2021-05-01', '2021-05-09', '2021-06-12', '2021-11-04'
            ]
            
            holidays_2022 = [
                '2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08',
                '2022-02-23', '2022-03-08', '2022-05-01', '2022-05-09', '2022-06-12', '2022-11-04'
            ]
            
            all_holidays = holidays_2020 + holidays_2021 + holidays_2022
            
            # Создание признака праздника
            df_features['is_holiday'] = df_features.index.date.astype(str).isin(all_holidays).astype(int)
            
            # Признаки близости к праздникам
            holiday_dates = pd.to_datetime(all_holidays)
            
            # Расстояние до ближайшего праздника
            def distance_to_nearest_holiday(date):
                try:
                    distances = np.abs((holiday_dates - date).days)
                    return distances.min()
                except:
                    return 365  # Если ошибка, возвращаем максимальное расстояние
            
            df_features['days_to_nearest_holiday'] = df_features.index.map(distance_to_nearest_holiday)
            
            # Бинарные признаки для периодов перед/после праздников
            df_features['is_pre_holiday'] = (df_features['days_to_nearest_holiday'] <= 3).astype(int)
            df_features['is_post_holiday'] = (df_features['days_to_nearest_holiday'] <= 3).astype(int)
            
            # Сезонные события
            df_features['is_new_year_period'] = (
                (df_features['month'] == 12) & (df_features['day'] >= 25)
            ).astype(int)
            
            df_features['is_summer_season'] = df_features['month'].isin([6, 7, 8]).astype(int)
            df_features['is_winter_season'] = df_features['month'].isin([12, 1, 2]).astype(int)
            
            holiday_features = [
                'is_holiday', 'days_to_nearest_holiday', 'is_pre_holiday', 'is_post_holiday',
                'is_new_year_period', 'is_summer_season', 'is_winter_season'
            ]
            
            self.features_created.extend(holiday_features)
            
            logger.info(f"Создано {len(holiday_features)} праздничных признаков")
            return df_features
            
        except Exception as e:
            logger.error(f"Ошибка создания праздничных признаков: {e}")
            return df
    
    def create_interaction_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Создание признаков взаимодействия."""
        logger.info("Создание признаков взаимодействия...")
        
        try:
            df_features = df.copy()
            
            # Взаимодействие временных признаков с лагами
            if 'lag_7' in df_features.columns:
                df_features['month_lag7_interaction'] = df_features['month'] * df_features['lag_7']
                df_features['day_of_week_lag7_interaction'] = df_features['day_of_week'] * df_features['lag_7']
                
                self.features_created.extend(['month_lag7_interaction', 'day_of_week_lag7_interaction'])
            
            # Взаимодействие скользящих статистик
            if 'rolling_mean_7' in df_features.columns and 'rolling_std_7' in df_features.columns:
                df_features['mean_std_ratio_7'] = df_features['rolling_mean_7'] / (df_features['rolling_std_7'] + 1e-8)
                self.features_created.append('mean_std_ratio_7')
            
            # Взаимодействие с праздниками
            if 'is_holiday' in df_features.columns and 'lag_1' in df_features.columns:
                df_features['holiday_lag1_interaction'] = df_features['is_holiday'] * df_features['lag_1']
                self.features_created.append('holiday_lag1_interaction')
            
            interaction_features_count = len([f for f in self.features_created if 'interaction' in f or 'ratio' in f])
            logger.info(f"Создано {interaction_features_count} признаков взаимодействия")
            return df_features
            
        except Exception as e:
            logger.error(f"Ошибка создания признаков взаимодействия: {e}")
            return df
    
    def analyze_feature_importance(self, df: pd.DataFrame, target_column: str) -> dict:
        """Анализ важности признаков."""
        logger.info("Анализ важности признаков...")
        
        try:
            # Корреляция с целевой переменной
            available_features = [f for f in self.features_created if f in df.columns]
            if not available_features:
                logger.warning("Нет доступных признаков для анализа важности")
                return {}
            
            correlations = df[available_features + [target_column]].corr()[target_column].drop(target_column)
            correlations = correlations.abs().sort_values(ascending=False)
            
            # Статистики по признакам
            feature_stats = {}
            for feature in available_features:
                if feature in df.columns:
                    feature_stats[feature] = {
                        'correlation_with_target': correlations.get(feature, 0),
                        'missing_values': df[feature].isnull().sum(),
                        'missing_percentage': (df[feature].isnull().sum() / len(df)) * 100,
                        'unique_values': df[feature].nunique(),
                        'data_type': str(df[feature].dtype),
                        'mean': df[feature].mean() if df[feature].dtype in ['int64', 'float64'] else None,
                        'std': df[feature].std() if df[feature].dtype in ['int64', 'float64'] else None
                    }
            
            logger.info(f"Проанализировано {len(available_features)} признаков")
            
            return {
                'correlations': correlations,
                'feature_stats': feature_stats,
                'top_features': correlations.head(20).to_dict()
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа важности признаков: {e}")
            return {}
    
    def visualize_features(self, df: pd.DataFrame, target_column: str, 
                         top_n: int = 10) -> None:
        """Визуализация важных признаков."""
        logger.info(f"Создание визуализации признаков (топ-{top_n})...")
        
        try:
            # Анализ важности
            importance_analysis = self.analyze_feature_importance(df, target_column)
            if not importance_analysis:
                logger.warning("Нет данных для визуализации признаков")
                return
            
            correlations = importance_analysis['correlations']
            if correlations.empty:
                logger.warning("Нет корреляций для визуализации")
                return
            
            top_features = correlations.head(top_n)
            
            # График корреляций
            plt.figure(figsize=(12, 8))
            top_features.plot(kind='barh')
            plt.title(f'Топ-{top_n} признаков по корреляции с {target_column}')
            plt.xlabel('Абсолютная корреляция')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Распределение важных признаков
            n_features_to_plot = min(6, len(top_features))
            if n_features_to_plot > 0:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                axes = axes.flatten()
                
                for i, (feature, corr) in enumerate(top_features.head(n_features_to_plot).items()):
                    if i < len(axes) and feature in df.columns:
                        axes[i].scatter(df[feature], df[target_column], alpha=0.6)
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel(target_column)
                        axes[i].set_title(f'{feature} (корр: {corr:.3f})')
                        axes[i].grid(True, alpha=0.3)
                
                # Скрываем пустые subplot
                for i in range(n_features_to_plot, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.show()
            
            logger.info("Визуализация признаков создана успешно")
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализации признаков: {e}")
    
    def run_comprehensive_feature_engineering(self, file_path: str, 
                                            target_column: str = 'Weekly_Sales') -> pd.DataFrame:
        """Запуск комплексного создания признаков."""
        print("=== КОМПЛЕКСНОЕ СОЗДАНИЕ ПРИЗНАКОВ ===")
        
        # Загрузка данных
        df = self.load_data(file_path, target_column)
        if df is None:
            return None
        
        # Создание всех типов признаков
        df_features = df.copy()
        
        # 1. Временные признаки
        df_features = self.create_temporal_features(df_features)
        
        # 2. Лаговые признаки
        df_features = self.create_lag_features(df_features, target_column)
        
        # 3. Скользящие статистики
        df_features = self.create_rolling_features(df_features, target_column)
        
        # 4. Признаки волатильности
        df_features = self.create_volatility_features(df_features, target_column)
        
        # 5. Праздничные признаки
        df_features = self.create_holiday_features(df_features)
        
        # 6. Признаки взаимодействия
        df_features = self.create_interaction_features(df_features, target_column)
        
        # Анализ и визуализация
        print(f"\nВсего создано признаков: {len(self.features_created)}")
        
        # Анализ важности признаков
        importance_analysis = self.analyze_feature_importance(df_features, target_column)
        
        # Визуализация
        self.visualize_features(df_features, target_column)
        
        # Сохранение результатов
        df_features.to_csv('enhanced_dataset.csv', index=True)
        
        # Сохранение информации о признаках
        feature_info = {
            'total_features': len(self.features_created),
            'feature_list': self.features_created,
            'importance_analysis': importance_analysis
        }
        
        import json
        with open('feature_engineering_results.json', 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\nРезультаты сохранены:")
        print("- enhanced_dataset.csv")
        print("- feature_engineering_results.json")
        
        return df_features


def main():
    """Основная функция для демонстрации работы модуля."""
    fe = TimeSeriesFeatureEngineering()
    
    # Запуск создания признаков
    enhanced_df = fe.run_comprehensive_feature_engineering(
        "../New_final.csv", 
        "Weekly_Sales"
    )
    
    if enhanced_df is not None:
        print(f"\n=== СОЗДАНИЕ ПРИЗНАКОВ ЗАВЕРШЕНО ===")
        print(f"Исходных колонок: {len(enhanced_df.columns) - len(fe.features_created)}")
        print(f"Создано признаков: {len(fe.features_created)}")
        print(f"Общее количество колонок: {len(enhanced_df.columns)}")
    else:
        print("Создание признаков не удалось. Проверьте путь к файлу.")


if __name__ == "__main__":
    main()

