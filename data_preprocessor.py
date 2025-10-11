#!/usr/bin/env python3
"""
Этап 2: Предварительная очистка и предобработка данных
Модуль для обработки временных рядов недвижимости
"""
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Класс для предобработки данных временных рядов"""
    
    def __init__(self, timezone='Europe/Moscow'):
        self.timezone = timezone
        
    def preprocess_real_estate_data(self, df, date_column='saledate', target_column='MA'):
        """
        Предобработка данных о недвижимости
        
        Args:
            df: DataFrame с данными
            date_column: название колонки с датами
            target_column: название целевой переменной
            
        Returns:
            Обработанный DataFrame
        """
        logger.info("Начинаем предобработку данных недвижимости")
        
        df_clean = df.copy()
        
        # Этап 2.1: Приведение временных меток к единому формату
        df_clean = self._standardize_datetime(df_clean, date_column)
        
        # Этап 2.2: Удаление дубликатов по времени
        df_clean = self._remove_time_duplicates(df_clean, date_column)
        
        # Этап 2.3: Проверка монотонности временного ряда
        df_clean = self._ensure_time_monotonicity(df_clean, date_column)
        
        # Этап 2.4: Обработка пропусков
        df_clean = self._handle_missing_values(df_clean)
        
        # Этап 2.5: Обнаружение и обработка выбросов
        df_clean = self._handle_outliers(df_clean, target_column)
        
        # Этап 2.6: Ресемплирование до единой частоты
        df_clean = self._resample_to_frequency(df_clean, date_column, target_column)
        
        logger.info(f"Предобработка завершена. Размер данных: {df_clean.shape}")
        return df_clean
    
    def _standardize_datetime(self, df, date_column):
        """Приведение временных меток к единому формату"""
        logger.info("Стандартизация временных меток")
        
        if date_column in df.columns:
            # Преобразуем в datetime
            df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y', errors='coerce')
            
            # Устанавливаем временную зону
            df[date_column] = df[date_column].dt.tz_localize(self.timezone)
            
            logger.info(f"Временные метки стандартизированы. Диапазон: {df[date_column].min()} - {df[date_column].max()}")
        
        return df
    
    def _remove_time_duplicates(self, df, date_column):
        """Удаление дубликатов по времени"""
        logger.info("Удаление дубликатов по времени")
        
        initial_size = len(df)
        
        if date_column in df.columns:
            df = df.drop_duplicates(subset=[date_column], keep='first')
        
        removed_count = initial_size - len(df)
        logger.info(f"Удалено {removed_count} дубликатов")
        
        return df
    
    def _ensure_time_monotonicity(self, df, date_column):
        """Проверка монотонности временного ряда"""
        logger.info("Проверка монотонности временного ряда")
        
        if date_column in df.columns:
            df = df.sort_values(date_column)
            logger.info("Временной ряд отсортирован по возрастанию")
        
        return df
    
    def _handle_missing_values(self, df):
        """Обработка пропусков"""
        logger.info("Обработка пропусков")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            
            if missing_pct < 5:
                # Если пропусков меньше 5%, удаляем
                df = df.dropna(subset=[col])
                logger.info(f"Удалены строки с пропусками в {col} ({missing_pct:.1f}%)")
            else:
                # Иначе интерполируем
                df[col] = df[col].interpolate(method='linear')
                logger.info(f"Интерполированы пропуски в {col} ({missing_pct:.1f}%)")
        
        return df
    
    def _handle_outliers(self, df, target_column):
        """Обнаружение и обработка выбросов"""
        logger.info("Обработка выбросов")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in df.columns:
                # Метод IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Подсчитываем выбросы
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                # Заменяем выбросы на граничные значения
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                logger.info(f"Обработано {outliers_count} выбросов в {col}")
        
        return df
    
    def _resample_to_frequency(self, df, date_column, target_column):
        """Ресемплирование до единой частоты"""
        logger.info("Ресемплирование до квартальной частоты")
        
        if date_column in df.columns and target_column in df.columns:
            # Группируем по типу недвижимости и количеству спален
            grouping_cols = ['type', 'bedrooms'] if 'type' in df.columns and 'bedrooms' in df.columns else []
            
            if grouping_cols:
                # Группируем и агрегируем
                df_grouped = df.groupby(grouping_cols + [date_column])[target_column].mean().reset_index()
                df_grouped = df_grouped.set_index(date_column)
                
                # Ресемплируем каждую группу отдельно
                resampled_groups = []
                for name, group in df_grouped.groupby(grouping_cols):
                    group_resampled = group.resample('QE').mean(numeric_only=True).dropna()
                    group_resampled = group_resampled.reset_index()
                    
                    # Добавляем группировочные колонки
                    for i, col in enumerate(grouping_cols):
                        group_resampled[col] = name[i] if isinstance(name, tuple) else name
                    
                    resampled_groups.append(group_resampled)
                
                df = pd.concat(resampled_groups, ignore_index=True)
            else:
                # Простое ресемплирование
                df = df.set_index(date_column)
                df = df.resample('QE').mean(numeric_only=True).dropna()
                df = df.reset_index()
            
            logger.info(f"Ресемплирование завершено. Новый размер: {df.shape}")
        
        return df

def main():
    """Пример использования модуля предобработки"""
    # Загружаем данные
    df = pd.read_csv('ma_lga_12345.csv')
    
    print("Исходные данные:")
    print(f"Размер: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    print(f"Пропуски: {df.isnull().sum().sum()}")
    
    # Создаем предпроцессор
    preprocessor = DataPreprocessor()
    
    # Предобрабатываем данные
    processed_df = preprocessor.preprocess_real_estate_data(df)
    
    print("\nОбработанные данные:")
    print(f"Размер: {processed_df.shape}")
    print(f"Пропуски: {processed_df.isnull().sum().sum()}")
    
    # Сохраняем результат
    processed_df.to_csv('processed_real_estate_data.csv', index=False)
    print("\nОбработанные данные сохранены в processed_real_estate_data.csv")

if __name__ == "__main__":
    main()

