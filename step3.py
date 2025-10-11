#!/usr/bin/env python3
"""
Этап 3: Описательный статистический анализ и визуализация
Скрипт для анализа данных о недвижимости
"""
import pandas as pd
import numpy as np
import logging
from descriptive_analyzer import DescriptiveAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Этап 3: Описательный статистический анализ"""
    print("="*60)
    print("ЭТАП 3: ОПИСАТЕЛЬНЫЙ СТАТИСТИЧЕСКИЙ АНАЛИЗ И ВИЗУАЛИЗАЦИЯ")
    print("="*60)
    print("Задачи:")
    print("• Расчет дескриптивной статистики")
    print("• Анализ распределений")
    print("• Корреляционный анализ")
    print("• Анализ по группам")
    print("• Временной анализ")
    print()
    
    # Загрузка обработанных данных
    try:
        df = pd.read_csv('processed_real_estate_data.csv')
        df['saledate'] = pd.to_datetime(df['saledate'], utc=True)
        logger.info(f"Загружены обработанные данные: {df.shape}")
    except FileNotFoundError:
        logger.error("Файл processed_real_estate_data.csv не найден!")
        logger.info("Запустите сначала step2.py для предобработки данных")
        return False
    
    print(f"📁 Загружены обработанные данные:")
    print(f"   Размер: {df.shape}")
    print(f"   Колонки: {df.columns.tolist()}")
    print(f"   Типы данных:")
    for col in df.columns:
        print(f"     {col}: {df[col].dtype}")
    
    # Проверяем наличие целевой переменной
    if 'MA' not in df.columns:
        print(f"⚠️  ВНИМАНИЕ: Колонка 'MA' не найдена в данных!")
        print(f"   Доступные колонки: {df.columns.tolist()}")
        return False
    
    # Создание анализатора
    analyzer = DescriptiveAnalyzer()
    
    # Выполнение анализа
    logger.info("Начинаем описательный анализ...")
    results = analyzer.analyze_real_estate_data(df)
    
    if not results:
        logger.error("Ошибка при выполнении анализа!")
        return False
    
    print(f"\n📊 Результаты описательного анализа:")
    print(f"   Доступные разделы: {list(results.keys())}")
    
    # Отладочная информация о структуре результатов
    if 'descriptive_stats' in results:
        print(f"   Структура descriptive_stats: {list(results['descriptive_stats'].keys())}")
        if 'basic' in results['descriptive_stats']:
            basic_stats = results['descriptive_stats']['basic']
            print(f"   Basic stats - индексы: {basic_stats.index.tolist()}")
            print(f"   Basic stats - колонки: {basic_stats.columns.tolist()}")
        if 'additional' in results['descriptive_stats']:
            additional_stats = results['descriptive_stats']['additional']
            print(f"   Additional stats - индексы: {additional_stats.index.tolist()}")
            print(f"   Additional stats - колонки: {additional_stats.columns.tolist()}")
    
    # Базовая информация
    if 'basic_info' in results:
        info = results['basic_info']
        print(f"   Размер данных: {info['shape'][0]} строк, {info['shape'][1]} столбцов")
        print(f"   Использование памяти: {info['memory_usage'] / 1024**2:.2f} MB")
        print(f"   Общее количество пропусков: {sum(info['missing_values'].values())}")
    
    # Дескриптивная статистика
    if 'descriptive_stats' in results:
        stats = results['descriptive_stats']['basic']
        print(f"\n📈 Дескриптивная статистика:")
        
        # Проверяем, есть ли колонка MA в статистике
        if 'MA' in stats.columns:
            print(f"Целевая переменная (MA):")
            print(f"   Среднее: {stats.loc['mean', 'MA']:.2f}")
            print(f"   Медиана: {stats.loc['50%', 'MA']:.2f}")
            print(f"   Стандартное отклонение: {stats.loc['std', 'MA']:.2f}")
            print(f"   Минимум: {stats.loc['min', 'MA']:.2f}")
            print(f"   Максимум: {stats.loc['max', 'MA']:.2f}")
            print(f"   Диапазон: {stats.loc['max', 'MA'] - stats.loc['min', 'MA']:.2f}")
        else:
            print(f"Доступные числовые колонки: {stats.columns.tolist()}")
            # Показываем статистику для всех числовых колонок
            for col in stats.columns:
                print(f"\n{col}:")
                print(f"   Среднее: {stats.loc['mean', col]:.2f}")
                print(f"   Медиана: {stats.loc['50%', col]:.2f}")
                print(f"   Стандартное отклонение: {stats.loc['std', col]:.2f}")
                print(f"   Минимум: {stats.loc['min', col]:.2f}")
                print(f"   Максимум: {stats.loc['max', col]:.2f}")
        
        # Дополнительные метрики
        if 'additional' in results['descriptive_stats']:
            additional = results['descriptive_stats']['additional']
            print(f"\n📊 Дополнительные метрики:")
            
            # Проверяем структуру данных
            print(f"   Структура дополнительных метрик:")
            print(f"   Индексы: {additional.index.tolist()}")
            print(f"   Колонки: {additional.columns.tolist()}")
            
            if 'MA' in additional.columns:
                print(f"\nДля целевой переменной (MA):")
                if 'skewness' in additional.index:
                    print(f"   Асимметрия: {additional.loc['skewness', 'MA']:.3f}")
                if 'kurtosis' in additional.index:
                    print(f"   Эксцесс: {additional.loc['kurtosis', 'MA']:.3f}")
                if 'iqr' in additional.index:
                    print(f"   Межквартильный размах: {additional.loc['iqr', 'MA']:.2f}")
            else:
                print(f"Доступные колонки для дополнительных метрик: {additional.columns.tolist()}")
                for col in additional.columns:
                    print(f"\n{col}:")
                    if 'skewness' in additional.index:
                        print(f"   Асимметрия: {additional.loc['skewness', col]:.3f}")
                    if 'kurtosis' in additional.index:
                        print(f"   Эксцесс: {additional.loc['kurtosis', col]:.3f}")
                    if 'iqr' in additional.index:
                        print(f"   Межквартильный размах: {additional.loc['iqr', col]:.2f}")
    
    # Анализ по группам
    if 'group_analysis' in results:
        print(f"\n🏠 Анализ по группам:")
        
        # По типам недвижимости
        if 'by_type' in results['group_analysis'] and results['group_analysis']['by_type'] is not None:
            type_stats = results['group_analysis']['by_type']
            print(f"\nПо типам недвижимости:")
            for idx, row in type_stats.iterrows():
                print(f"   {idx}:")
                print(f"     Количество: {row['count']}")
                if 'mean' in row:
                    print(f"     Среднее: {row['mean']:.2f}")
                if 'median' in row:
                    print(f"     Медиана: {row['median']:.2f}")
                if 'std' in row:
                    print(f"     Стд. отклонение: {row['std']:.2f}")
        
        # По количеству спален
        if 'by_bedrooms' in results['group_analysis'] and results['group_analysis']['by_bedrooms'] is not None:
            bedroom_stats = results['group_analysis']['by_bedrooms']
            print(f"\nПо количеству спален:")
            for idx, row in bedroom_stats.iterrows():
                print(f"   {idx} спален:")
                print(f"     Количество: {row['count']}")
                if 'mean' in row:
                    print(f"     Среднее: {row['mean']:.2f}")
                if 'median' in row:
                    print(f"     Медиана: {row['median']:.2f}")
                if 'std' in row:
                    print(f"     Стд. отклонение: {row['std']:.2f}")
    
    # Корреляционный анализ
    if 'correlation_analysis' in results:
        corr_analysis = results['correlation_analysis']
        print(f"\n🔗 Корреляционный анализ:")
        
        if 'strong_correlations' in corr_analysis and corr_analysis['strong_correlations']:
            print(f"   Обнаружены сильные корреляции:")
            for corr in corr_analysis['strong_correlations']:
                print(f"     {corr['var1']} - {corr['var2']}: {corr['correlation']:.3f}")
        else:
            print(f"   Сильных корреляций не обнаружено")
    
    # Временной анализ
    if 'temporal_analysis' in results:
        temporal = results['temporal_analysis']
        print(f"\n📅 Временной анализ:")
        
        if 'date_range' in temporal:
            date_range = temporal['date_range']
            print(f"   Период: {date_range['start']} - {date_range['end']}")
            print(f"   Продолжительность: {date_range['duration_days']} дней")
        
        if 'yearly_stats' in temporal and temporal['yearly_stats'] is not None:
            yearly_stats = temporal['yearly_stats']
            print(f"\nПо годам:")
            for idx, row in yearly_stats.iterrows():
                print(f"   {idx}: {row['count']} записей", end="")
                if 'mean' in row:
                    print(f", среднее = {row['mean']:.0f}")
                else:
                    print()
        
        if 'quarterly_stats' in temporal and temporal['quarterly_stats'] is not None:
            quarterly_stats = temporal['quarterly_stats']
            print(f"\nПо кварталам:")
            for idx, row in quarterly_stats.iterrows():
                print(f"   Q{idx}: {row['count']} записей", end="")
                if 'mean' in row:
                    print(f", среднее = {row['mean']:.0f}")
                else:
                    print()
    
    # Создание визуализаций
    logger.info("Создаем визуализации...")
    visualizations = analyzer.create_visualizations(df)
    
    print(f"\n📊 Создано {len(visualizations)} визуализаций:")
    for name in visualizations.keys():
        print(f"   - {name}")
    
    # Генерация отчета
    try:
        report = analyzer.generate_summary_report()
        
        # Сохранение отчета
        with open('descriptive_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Отчет сохранен в: descriptive_analysis_report.txt")
    except Exception as e:
        logger.warning(f"Не удалось создать отчет: {e}")
        print(f"⚠️  Предупреждение: Не удалось создать отчет: {e}")
    
    # Сохранение результатов анализа
    import json
    try:
        # Конвертируем numpy типы и сложные структуры в Python типы для JSON
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                # Конвертируем DataFrame в словарь
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                # Конвертируем Series в словарь
                return obj.to_dict()
            elif isinstance(obj, tuple):
                # Конвертируем кортежи в строки
                return str(obj)
            elif isinstance(obj, dict):
                # Рекурсивно конвертируем словари
                return {str(k) if isinstance(k, tuple) else k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                # Рекурсивно конвертируем списки
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Рекурсивно конвертируем результаты
        converted_results = convert_numpy(results)
        
        with open('descriptive_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📊 Результаты анализа сохранены в: descriptive_analysis_results.json")
    except Exception as e:
        logger.error(f"Ошибка при сохранении результатов: {e}")
        print(f"❌ Ошибка при сохранении результатов: {e}")
        print(f"   Результаты анализа доступны в памяти, но не сохранены в файл")
    
    print(f"\n✅ Этап 3 завершен успешно!")
    print(f"📝 Следующий шаг: python step4.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Описательный анализ выполнен успешно!")
    else:
        print("\n❌ Описательный анализ завершился с ошибкой!")
        exit(1)

