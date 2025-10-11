#!/usr/bin/env python3
"""
Главный скрипт для выполнения всех этапов анализа временных рядов недвижимости
Система анализа данных о ценах на недвижимость в Австралии
"""
import os
import sys
import subprocess
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_step(step_number: int, script_name: str) -> bool:
    """Запуск отдельного этапа"""
    print(f"\n{'='*60}")
    print(f"ЗАПУСК ЭТАПА {step_number}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_name):
        print(f"❌ Скрипт {script_name} не найден!")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(f"✅ Этап {step_number} выполнен успешно")
        if result.stdout:
            print("Вывод:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка в этапе {step_number}: {e}")
        if e.stdout:
            print("Вывод:")
            print(e.stdout)
        if e.stderr:
            print("Ошибки:")
            print(e.stderr)
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ Этап {step_number} прерван пользователем")
        return False

def check_dependencies():
    """Проверка зависимостей"""
    print("🔍 Проверка зависимостей...")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'scipy', 'statsmodels', 'pytz'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Отсутствуют пакеты: {', '.join(missing_packages)}")
        print("💡 Установите их командой: pip install -r requirements.txt")
        return False
    
    print("✅ Все основные зависимости установлены")
    return True

def check_data_file():
    """Проверка наличия исходного файла данных"""
    if not os.path.exists('ma_lga_12345.csv'):
        print("❌ Файл ma_lga_12345.csv не найден!")
        print("💡 Убедитесь, что файл находится в текущей директории")
        return False
    
    print("✅ Исходный файл данных найден")
    return True

def main():
    """Основная функция"""
    print("🏠 СИСТЕМА АНАЛИЗА ВРЕМЕННЫХ РЯДОВ НЕДВИЖИМОСТИ")
    print("=" * 60)
    print("Анализ данных о ценах на недвижимость в Австралии")
    print("Источник данных: ma_lga_12345.csv")
    print()
    print("Этапы анализа:")
    print("2. Предварительная очистка и предобработка данных")
    print("3. Описательный статистический анализ и визуализация")
    print("4. Проверка на стационарность и статистические тесты")
    print("5. Создание лаговых признаков и скользящих статистик")
    print("6. Анализ автокорреляции: ACF и PACF")
    print("7. Декомпозиция временного ряда")
    print("8. Разработка веб-интерфейса для интерактивного анализа")
    print()
    
    # Проверка зависимостей
    if not check_dependencies():
        return
    
    # Проверка файла данных
    if not check_data_file():
        return
    
    # Запуск этапов
    steps = [
        (2, "step2.py", "Предобработка данных"),
        (3, "step3.py", "Описательный анализ"),
        (4, "step4.py", "Анализ стационарности"),
    ]
    
    completed_steps = []
    
    for step_number, script_name, description in steps:
        print(f"\n📋 Этап {step_number}: {description}")
        
        if run_step(step_number, script_name):
            completed_steps.append(step_number)
        else:
            print(f"\n❌ Этап {step_number} завершился с ошибкой")
            break
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print(f"ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'='*60}")
    
    if len(completed_steps) == len(steps):
        print("🎉 Все этапы выполнены успешно!")
        print("\n📁 Созданные файлы:")
        files = [
            "processed_real_estate_data.csv",
            "descriptive_analysis_report.txt",
            "descriptive_analysis_results.json",
            "stationarity_analysis_report.txt",
            "stationarity_analysis_results.json"
        ]
        
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"   ✅ {file} ({size:,} байт)")
            else:
                print(f"   ❌ {file} (не найден)")
        
        print(f"\n🌐 Для веб-интерфейса запустите:")
        print(f"   streamlit run server.py")
        
        print(f"\n📊 Структура данных:")
        print(f"   • Целевая переменная: MA (Median Auction)")
        print(f"   • Временная переменная: saledate")
        print(f"   • Группировочные переменные: type, bedrooms")
        print(f"   • Частота: квартальная")
        
    else:
        print(f"⚠️ Выполнено этапов: {len(completed_steps)}/{len(steps)}")
        print(f"✅ Завершенные этапы: {completed_steps}")
        
        if completed_steps:
            print(f"\n💡 Для продолжения запустите:")
            next_step = completed_steps[-1] + 1
            if next_step <= len(steps):
                print(f"   python step{next_step}.py")
    
    print(f"\n📝 Дополнительные команды:")
    print(f"   • Предобработка только: python step2.py")
    print(f"   • Описательный анализ только: python step3.py")
    print(f"   • Анализ стационарности только: python step4.py")
    print(f"   • Веб-интерфейс: streamlit run server.py")
    
    print(f"\n📚 Документация:")
    print(f"   • README.md - подробное описание")
    print(f"   • requirements.txt - зависимости")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Выполнение прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Критическая ошибка: {e}")
        sys.exit(1)

