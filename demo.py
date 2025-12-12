"""
Демонстрационный скрипт для быстрого тестирования основных функций анализа временных рядов.

Этот скрипт показывает основные возможности системы анализа временных рядов
без полного запуска всех этапов.
"""

import os
import sys
import logging
from time_series_decomposition import TimeSeriesDecomposition
from exponential_smoothing import ExponentialSmoothingModels

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_decomposition():
    """Демонстрация декомпозиции временного ряда."""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ: ДЕКОМПОЗИЦИЯ ВРЕМЕННОГО РЯДА")
    print("="*50)
    
    decomposer = TimeSeriesDecomposition()
    
    # Загрузка данных
    data_path = "../New_final.csv"
    if not os.path.exists(data_path):
        print(f"Файл данных не найден: {data_path}")
        return
    
    df = decomposer.load_data(data_path, "Weekly_Sales")
    if df is None:
        print("Не удалось загрузить данные")
        return
    
    series = df["Weekly_Sales"]
    print(f"Обработка временного ряда длиной {len(series)}")
    
    # Декомпозиция с периодом 7
    print("\n1. Декомпозиция с периодом 7:")
    result = decomposer.seasonal_decomposition(series, period=7, model='additive')
    
    if result:
        print(f"   Размерность исходного ряда: {len(result['original'])}")
        print(f"   Размерность тренда: {len(result['trend'].dropna())}")
        print(f"   Размерность сезонной компоненты: {len(result['seasonal'].dropna())}")
        print(f"   Размерность остатков: {len(result['residual'].dropna())}")
        
        # Анализ остатков
        residual_analysis = decomposer.analyze_residuals(result['residual'])
        print(f"   Среднее остатков: {residual_analysis['statistics']['mean']:.4f}")
        print(f"   Стд. отклонение остатков: {residual_analysis['statistics']['std']:.4f}")
        
        if residual_analysis['stationarity']['overall_stationary']:
            print("   ✅ Остатки стационарны")
        else:
            print("   ❌ Остатки не стационарны")
    
    return result


def demo_exponential_smoothing():
    """Демонстрация экспоненциального сглаживания."""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ: ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ")
    print("="*50)
    
    es_analyzer = ExponentialSmoothingModels()
    
    # Загрузка данных
    data_path = "../New_final.csv"
    if not os.path.exists(data_path):
        print(f"Файл данных не найден: {data_path}")
        return
    
    df = es_analyzer.load_data(data_path, "Weekly_Sales")
    if df is None:
        print("Не удалось загрузить данные")
        return
    
    # Подготовка данных
    train_series, test_series = es_analyzer.prepare_data(df, "Weekly_Sales")
    print(f"Обучающая выборка: {len(train_series)} наблюдений")
    print(f"Тестовая выборка: {len(test_series)} наблюдений")
    
    # SES модель
    print("\n1. Simple Exponential Smoothing:")
    ses_result = es_analyzer.fit_ses_model(train_series)
    
    if ses_result:
        print(f"   AIC: {ses_result['aic']:.2f}")
        print(f"   BIC: {ses_result['bic']:.2f}")
        
        # Прогноз на 7 шагов
        forecast_data = es_analyzer.generate_forecast(ses_result, horizon=7)
        if forecast_data:
            print(f"   Прогноз на 7 шагов: {forecast_data['forecast'][:3]}...")
            print(f"   Уровень доверия: {forecast_data['confidence_level']*100:.0f}%")
            
            # Оценка качества
            metrics = es_analyzer.evaluate_forecast(forecast_data['forecast'], test_series)
            print(f"   MAE: {metrics['mae']:.4f}")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   MAPE: {metrics['mape']:.2f}%")
    
    # Хольт аддитивный
    print("\n2. Хольт (аддитивный тренд):")
    holt_add_result = es_analyzer.fit_holt_additive_model(train_series)
    
    if holt_add_result:
        print(f"   AIC: {holt_add_result['aic']:.2f}")
        print(f"   BIC: {holt_add_result['bic']:.2f}")
        
        # Прогноз на 7 шагов
        forecast_data = es_analyzer.generate_forecast(holt_add_result, horizon=7)
        if forecast_data:
            metrics = es_analyzer.evaluate_forecast(forecast_data['forecast'], test_series)
            print(f"   MAE: {metrics['mae']:.4f}")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   MAPE: {metrics['mape']:.2f}%")
    
    return ses_result, holt_add_result


def demo_feature_engineering():
    """Демонстрация создания признаков."""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ: СОЗДАНИЕ ПРИЗНАКОВ")
    print("="*50)
    
    from feature_engineering import TimeSeriesFeatureEngineering
    
    fe = TimeSeriesFeatureEngineering()
    
    # Загрузка данных
    data_path = "../New_final.csv"
    if not os.path.exists(data_path):
        print(f"Файл данных не найден: {data_path}")
        return
    
    df = fe.load_data(data_path, "Weekly_Sales")
    if df is None:
        print("Не удалось загрузить данные")
        return
    
    print(f"Исходный датасет: {df.shape}")
    
    # Создание временных признаков
    print("\n1. Временные признаки:")
    df_features = fe.create_temporal_features(df)
    print(f"   Создано временных признаков: {len([f for f in fe.features_created if 'sin' in f or 'cos' in f or 'is_' in f])}")
    
    # Создание лаговых признаков
    print("\n2. Лаговые признаки:")
    df_features = fe.create_lag_features(df_features, "Weekly_Sales", lags=[1, 7])
    print(f"   Создано лаговых признаков: {len([f for f in fe.features_created if 'lag' in f])}")
    
    # Создание скользящих статистик
    print("\n3. Скользящие статистики:")
    df_features = fe.create_rolling_features(df_features, "Weekly_Sales", windows=[7, 14])
    print(f"   Создано скользящих признаков: {len([f for f in fe.features_created if 'rolling' in f])}")
    
    print(f"\nОбщее количество созданных признаков: {len(fe.features_created)}")
    print(f"Итоговый размер датасета: {df_features.shape}")
    
    return df_features


def main():
    """Основная функция демонстрации."""
    print("ДЕМОНСТРАЦИЯ СИСТЕМЫ АНАЛИЗА ВРЕМЕННЫХ РЯДОВ")
    print("="*60)
    
    try:
        # Демонстрация декомпозиции
        decomposition_result = demo_decomposition()
        
        # Демонстрация экспоненциального сглаживания
        es_results = demo_exponential_smoothing()
        
        # Демонстрация создания признаков
        enhanced_df = demo_feature_engineering()
        
        print("\n" + "="*60)
        print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
        print("="*60)
        
        print("\nДля полного анализа запустите:")
        print("python run_all_steps.py")
        
        print("\nДля веб-интерфейса запустите:")
        print("streamlit run web_interface.py")
        
    except Exception as e:
        print(f"Ошибка в демонстрации: {e}")
        logger.error(f"Ошибка в демонстрации: {e}")


if __name__ == "__main__":
    main()


