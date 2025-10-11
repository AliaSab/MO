#!/usr/bin/env python3
"""
Этап 2: Предварительная очистка и предобработка данных
Скрипт для обработки данных о недвижимости
"""
import pandas as pd
import numpy as np
import logging
from data_preprocessor import DataPreprocessor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Этап 2: Предобработка данных о недвижимости"""
    print("="*60)
    print("ЭТАП 2: ПРЕДВАРИТЕЛЬНАЯ ОЧИСТКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("="*60)
    print("Задачи:")
    print("• Приведение временных меток к единому формату")
    print("• Удаление дубликатов по времени")
    print("• Проверка монотонности временного ряда")
    print("• Обработка пропусков")
    print("• Обнаружение и обработка выбросов")
    print("• Ресемплирование до единой частоты")
    print()
    
    # Загрузка данных
    try:
        df = pd.read_csv('ma_lga_12345.csv')
        logger.info(f"Загружены исходные данные: {df.shape}")
    except FileNotFoundError:
        logger.error("Файл ma_lga_12345.csv не найден!")
        return False
    
    print(f"📁 Загружены исходные данные:")
    print(f"   Размер: {df.shape}")
    print(f"   Колонки: {df.columns.tolist()}")
    print(f"   Пропуски: {df.isnull().sum().sum()}")
    
    # Создание предпроцессора
    preprocessor = DataPreprocessor(timezone='Europe/Moscow')
    
    # Предобработка данных
    logger.info("Начинаем предобработку данных...")
    processed_df = preprocessor.preprocess_real_estate_data(df)
    
    if processed_df is None:
        logger.error("Ошибка при предобработке данных!")
        return False
    
    print(f"\n📊 Результаты предобработки:")
    print(f"   Исходный размер: {df.shape}")
    print(f"   Обработанный размер: {processed_df.shape}")
    print(f"   Пропуски после обработки: {processed_df.isnull().sum().sum()}")
    
    # Сохранение результатов
    output_file = 'processed_real_estate_data.csv'
    processed_df.to_csv(output_file, index=False)
    logger.info(f"Обработанные данные сохранены в {output_file}")
    
    # Статистика по группам
    if 'type' in processed_df.columns and 'bedrooms' in processed_df.columns:
        print(f"\n📈 Статистика по группам:")
        
        # По типам недвижимости
        type_stats = processed_df.groupby('type')['MA'].agg(['count', 'mean', 'median']).round(2)
        print(f"\nПо типам недвижимости:")
        for idx, row in type_stats.iterrows():
            print(f"   {idx}: {row['count']} записей, среднее = {row['mean']:.0f}, медиана = {row['median']:.0f}")
        
        # По количеству спален
        bedroom_stats = processed_df.groupby('bedrooms')['MA'].agg(['count', 'mean', 'median']).round(2)
        print(f"\nПо количеству спален:")
        for idx, row in bedroom_stats.iterrows():
            print(f"   {idx} спален: {row['count']} записей, среднее = {row['mean']:.0f}, медиана = {row['median']:.0f}")
    
    # Временной анализ
    if 'saledate' in processed_df.columns:
        print(f"\n📅 Временной анализ:")
        print(f"   Период: {processed_df['saledate'].min()} - {processed_df['saledate'].max()}")
        print(f"   Продолжительность: {(processed_df['saledate'].max() - processed_df['saledate'].min()).days} дней")
        
        # Анализ по годам
        # Убеждаемся, что колонка даты имеет правильный тип
        if not pd.api.types.is_datetime64_any_dtype(processed_df['saledate']):
            processed_df['saledate'] = pd.to_datetime(processed_df['saledate'], utc=True)
        processed_df['year'] = processed_df['saledate'].dt.year
        yearly_stats = processed_df.groupby('year')['MA'].agg(['count', 'mean']).round(2)
        print(f"\nПо годам:")
        for idx, row in yearly_stats.iterrows():
            print(f"   {idx}: {row['count']} записей, среднее = {row['mean']:.0f}")
    
    print(f"\n✅ Этап 2 завершен успешно!")
    print(f"📁 Обработанные данные сохранены в: {output_file}")
    print(f"📝 Следующий шаг: python step3.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Предобработка данных выполнена успешно!")
    else:
        print("\n❌ Предобработка данных завершилась с ошибкой!")
        exit(1)

