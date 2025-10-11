#!/usr/bin/env python3
"""
Этап 4: Проверка на стационарность и статистические тесты
Скрипт для анализа стационарности временных рядов недвижимости
"""
import pandas as pd
import numpy as np
import logging
from stationarity_analyzer import StationarityAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Этап 4: Анализ стационарности"""
    print("="*60)
    print("ЭТАП 4: ПРОВЕРКА НА СТАЦИОНАРНОСТЬ И СТАТИСТИЧЕСКИЕ ТЕСТЫ")
    print("="*60)
    print("Задачи:")
    print("• Визуальный анализ стационарности")
    print("• Расчет скользящих статистик")
    print("• Статистические тесты (ADF, KPSS)")
    print("• Анализ дифференцирования")
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
    print(f"   Временной диапазон: {df['saledate'].min()} - {df['saledate'].max()}")
    
    # Создание анализатора
    analyzer = StationarityAnalyzer()
    
    # Выполнение анализа стационарности
    logger.info("Начинаем анализ стационарности...")
    results = analyzer.analyze_stationarity(df)
    
    if not results:
        logger.error("Ошибка при выполнении анализа стационарности!")
        return False
    
    print(f"\n📊 Результаты анализа стационарности:")
    
    # Визуальный анализ
    if 'visual_analysis' in results:
        visual = results['visual_analysis']
        print(f"\n👁️ Визуальный анализ:")
        print(f"   Тренд присутствует: {'Да' if visual['trend_present'] else 'Нет'}")
        print(f"   Дисперсия стабильна: {'Да' if visual['variance_stable'] else 'Нет'}")
        print(f"   Стационарен на глаз: {'Да' if visual['stationary_by_eye'] else 'Нет'}")
    
    # Скользящие статистики
    if 'rolling_statistics' in results:
        rolling_stats = results['rolling_statistics']
        print(f"\n📈 Скользящие статистики:")
        
        for window_name, stats in rolling_stats.items():
            print(f"   {window_name}:")
            print(f"     Среднее стабильно: {'Да' if stats['mean_stability'] else 'Нет'}")
            print(f"     Стд. отклонение стабильно: {'Да' if stats['std_stability'] else 'Нет'}")
    
    # Статистические тесты
    if 'statistical_tests' in results:
        tests = results['statistical_tests']
        print(f"\n🧮 Статистические тесты:")
        
        # ADF тест
        adf = tests['adf']
        print(f"\n   Тест Дики-Фуллера (ADF):")
        print(f"     Статистика: {adf['statistic']:.4f}")
        print(f"     p-value: {adf['p_value']:.4f}")
        print(f"     Критические значения:")
        for level, value in adf['critical_values'].items():
            print(f"       {level}: {value:.4f}")
        print(f"     Стационарен: {'Да' if adf['is_stationary'] else 'Нет'}")
        
        # KPSS тест
        kpss = tests['kpss']
        print(f"\n   Тест KPSS:")
        print(f"     Статистика: {kpss['statistic']:.4f}")
        print(f"     p-value: {kpss['p_value']:.4f}")
        print(f"     Критические значения:")
        for level, value in kpss['critical_values'].items():
            print(f"       {level}: {value:.4f}")
        print(f"     Стационарен: {'Да' if kpss['is_stationary'] else 'Нет'}")
        
        # Общий вывод
        print(f"\n   Общий вывод: {'Ряд стационарен' if tests['overall_stationary'] else 'Ряд нестационарен'}")
    
    # Дифференцирование
    if 'differencing' in results:
        diff_results = results['differencing']
        print(f"\n🔄 Анализ дифференцирования:")
        
        if 'first_difference' in diff_results:
            first_diff = diff_results['first_difference']
            print(f"\n   Первое дифференцирование:")
            print(f"     Стационарен: {'Да' if first_diff['stationary'] else 'Нет'}")
            
            if first_diff['stationary']:
                print(f"     Рекомендация: Использовать d=1 в ARIMA модели")
        
        if 'second_difference' in diff_results:
            second_diff = diff_results['second_difference']
            print(f"\n   Второе дифференцирование:")
            print(f"     Стационарен: {'Да' if second_diff['stationary'] else 'Нет'}")
            
            if second_diff['stationary']:
                print(f"     Рекомендация: Использовать d=2 в ARIMA модели")
    
    # Создание графиков
    logger.info("Создаем графики стационарности...")
    plots = analyzer.create_stationarity_plots(df)
    
    print(f"\n📊 Создано {len(plots)} графиков:")
    for name in plots.keys():
        print(f"   - {name}")
    
    # Генерация отчета
    report = analyzer.generate_stationarity_report()
    
    # Сохранение отчета
    with open('stationarity_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Отчет сохранен в: stationarity_analysis_report.txt")
    
    # Сохранение результатов анализа
    import json
    with open('stationarity_analysis_results.json', 'w', encoding='utf-8') as f:
        # Конвертируем результаты для JSON
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return obj
        
        converted_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                converted_results[key] = {k: convert_for_json(v) for k, v in value.items()}
            else:
                converted_results[key] = convert_for_json(value)
        
        json.dump(converted_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"📊 Результаты анализа сохранены в: stationarity_analysis_results.json")
    
    # Рекомендации
    print(f"\n💡 Рекомендации:")
    if results.get('statistical_tests', {}).get('overall_stationary', False):
        print(f"   ✅ Ряд стационарен, можно применять ARIMA модели")
        print(f"   📈 Рекомендуемые параметры: ARIMA(p,0,q)")
    else:
        print(f"   ⚠️ Ряд нестационарен, рекомендуется:")
        print(f"   📉 Применить дифференцирование")
        print(f"   🔧 Использовать ARIMA(p,d,q) модели с d>0")
        print(f"   📊 Рассмотреть другие методы стационаризации")
    
    print(f"\n✅ Этап 4 завершен успешно!")
    print(f"📝 Следующий шаг: python step5.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Анализ стационарности выполнен успешно!")
    else:
        print("\n❌ Анализ стационарности завершился с ошибкой!")
        exit(1)

