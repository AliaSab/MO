#!/usr/bin/env python3
"""
Скрипт для запуска Streamlit приложения анализа временных рядов недвижимости
"""
import subprocess
import sys
import os

def check_requirements():
    """Проверка установленных зависимостей"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import scipy
        import statsmodels
        import seaborn
        import matplotlib
        import pytz
        print("✅ Все зависимости установлены")
        return True
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        print("Установите зависимости командой: pip install -r requirements.txt")
        return False

def main():
    """Основная функция запуска"""
    print("🏠 Запуск приложения анализа временных рядов недвижимости...")
    
    # Проверяем зависимости
    if not check_requirements():
        sys.exit(1)
    
    # Проверяем наличие файла server.py
    if not os.path.exists('server.py'):
        print("❌ Файл server.py не найден в текущей директории")
        print("Убедитесь, что вы находитесь в директории Time_series_Alya")
        sys.exit(1)
    
    # Проверяем наличие файла ma_lga_12345.csv
    if not os.path.exists('ma_lga_12345.csv'):
        print("⚠️  Файл ma_lga_12345.csv не найден. Приложение будет работать с загруженными пользователем данными")
    
    print("🌐 Запуск Streamlit сервера...")
    print("📱 Приложение будет доступно по адресу: http://localhost:8501")
    print("⏹️  Для остановки нажмите Ctrl+C")
    print("-" * 50)
    
    try:
        # Запускаем Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "server.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Приложение остановлено пользователем")
    except Exception as e:
        print(f"❌ Ошибка при запуске: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

