"""
Веб-интерфейс для интерактивного прогнозирования временных рядов.

Реализует:
- Загрузку CSV/Parquet файлов
- Выбор целевой переменной и горизонта прогноза
- Переключение между аддитивной/мультипликативной декомпозицией
- Настройку параметров преобразования (включая λ для Бокса–Кокса)
- Визуализацию прогнозов с доверительными интервалами
- Таблицу сравнения метрик для всех моделей и стратегий
- Экспорт прогноза и параметров модели
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_interface.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Импорт модулей анализа
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from time_series_decomposition import TimeSeriesDecomposition
    from feature_engineering import TimeSeriesFeatureEngineering
    from multi_step_forecasting import MultiStepForecasting
    from time_series_cv import TimeSeriesCrossValidation
    from stationarity_transformation import StationarityTransformation
    from exponential_smoothing import ExponentialSmoothingModels
    logger.info("Все модули анализа успешно импортированы")
except ImportError as e:
    logger.error(f"Ошибка импорта модулей: {e}")
    st.error(f"Ошибка импорта модулей: {e}")


class TimeSeriesWebInterface:
    """Класс для веб-интерфейса анализа временных рядов."""
    
    def __init__(self):
        self.data = None
        self.target_column = None
        self.analysis_results = {}
        
    def load_data_interface(self):
        """Интерфейс загрузки данных."""
        logger.info("Загрузка интерфейса данных")
        st.header("📁 Загрузка данных")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите CSV файл", 
            type=['csv'],
            help="Поддерживаются файлы CSV с временными рядами",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                logger.info(f"Загружен файл: {uploaded_file.name}")
                
                # Чтение файла
                if uploaded_file.name.endswith('.csv'):
                    self.data = pd.read_csv(uploaded_file)
                
                if self.data.empty:
                    st.error("Файл пуст")
                    logger.error("Загруженный файл пуст")
                    return False
                
                st.success(f"Файл загружен успешно! Размер: {self.data.shape}")
                logger.info(f"Файл загружен успешно, размер: {self.data.shape}")
                
                # Выбор колонки с датами
                date_columns = [col for col in self.data.columns 
                              if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower()]
                
                if date_columns:
                    date_col = st.selectbox("Выберите колонку с датами", date_columns, key="date_column")
                    try:
                        self.data[date_col] = pd.to_datetime(self.data[date_col])
                        self.data = self.data.set_index(date_col)
                        logger.info(f"Установлен индекс по колонке {date_col}")
                    except Exception as e:
                        st.error(f"Ошибка преобразования дат: {e}")
                        logger.error(f"Ошибка преобразования дат: {e}")
                        return False
                else:
                    st.warning("Колонка с датами не найдена. Будет создан индекс по умолчанию.")
                    logger.warning("Колонка с датами не найдена, создан искусственный индекс")
                    self.data.index = pd.date_range(start='2020-01-01', periods=len(self.data), freq='D')
                
                # Выбор целевой переменной
                numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    st.error("В данных нет числовых колонок")
                    logger.error("В данных нет числовых колонок")
                    return False
                
                self.target_column = st.selectbox("Выберите целевую переменную", numeric_columns, key="target_column")
                logger.info(f"Выбрана целевая переменная: {self.target_column}")
                
                # Предварительный просмотр данных
                st.subheader("Предварительный просмотр данных")
                st.dataframe(self.data.head(10))
                
                # Основная статистика
                st.subheader("Основная статистика")
                stats = self.data[self.target_column].describe()
                st.write(stats)
                
                # Проверка на пропущенные значения
                missing_count = self.data[self.target_column].isnull().sum()
                if missing_count > 0:
                    st.warning(f"Найдено {missing_count} пропущенных значений в целевой переменной")
                    logger.warning(f"Найдено {missing_count} пропущенных значений")
                
                return True
                
            except Exception as e:
                st.error(f"Ошибка загрузки файла: {e}")
                logger.error(f"Ошибка загрузки файла: {e}")
                return False
        
        return False
    
    def decomposition_interface(self):
        """Интерфейс декомпозиции временного ряда."""
        logger.info("Загрузка интерфейса декомпозиции")
        st.header("🔍 Декомпозиция временного ряда")
        
        if self.data is None:
            st.warning("Сначала загрузите данные")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            period = st.selectbox("Период сезонности", [7, 30, 90, 365], index=0, key="decomp_period")
        
        with col2:
            model_type = st.selectbox("Тип модели", ['additive', 'multiplicative'], key="decomp_model")
        
        if st.button("Выполнить декомпозицию", type="primary", key="decomp_button"):
            with st.spinner("Выполняется декомпозиция..."):
                try:
                    logger.info(f"Начало декомпозиции: период={period}, тип={model_type}")
                    
                    decomposer = TimeSeriesDecomposition()
                    series = self.data[self.target_column]
                    
                    # Проверка достаточности данных
                    if len(series) < period * 2:
                        st.error(f"Недостаточно данных для декомпозиции с периодом {period}")
                        logger.error(f"Недостаточно данных для декомпозиции: {len(series)} < {period * 2}")
                        return
                    
                    # Выполнение декомпозиции
                    result = decomposer.seasonal_decomposition(series, period, model_type)
                    
                    if result and 'error' not in result:
                        logger.info("Декомпозиция выполнена успешно")
                        
                        # Визуализация компонент
                        fig = make_subplots(
                            rows=4, cols=1,
                            subplot_titles=['Исходный ряд', 'Тренд', 'Сезонная компонента', 'Остатки'],
                            vertical_spacing=0.05
                        )
                        
                        # Исходный ряд
                        fig.add_trace(
                            go.Scatter(x=result['original'].index, y=result['original'].values,
                                     name='Исходный ряд', line=dict(color='blue')),
                            row=1, col=1
                        )
                        
                        # Тренд
                        fig.add_trace(
                            go.Scatter(x=result['trend'].index, y=result['trend'].values,
                                     name='Тренд', line=dict(color='green')),
                            row=2, col=1
                        )
                        
                        # Сезонная компонента
                        fig.add_trace(
                            go.Scatter(x=result['seasonal'].index, y=result['seasonal'].values,
                                     name='Сезонная', line=dict(color='red')),
                            row=3, col=1
                        )
                        
                        # Остатки
                        fig.add_trace(
                            go.Scatter(x=result['residual'].index, y=result['residual'].values,
                                     name='Остатки', line=dict(color='orange')),
                            row=4, col=1
                        )
                        
                        fig.update_layout(height=800, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Анализ остатков
                        st.subheader("Анализ остатков")
                        residual_analysis = decomposer.analyze_residuals(result['residual'])
                        
                        if 'error' not in residual_analysis:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Среднее", f"{residual_analysis['statistics']['mean']:.4f}")
                                st.metric("Стд. отклонение", f"{residual_analysis['statistics']['std']:.4f}")
                            
                            with col2:
                                st.metric("Асимметрия", f"{residual_analysis['statistics']['skewness']:.4f}")
                                st.metric("Эксцесс", f"{residual_analysis['statistics']['kurtosis']:.4f}")
                            
                            with col3:
                                if 'normality_tests' in residual_analysis:
                                    st.metric("Тест нормальности", 
                                             "Нормальные" if residual_analysis['normality_tests']['is_normal'] else "Не нормальные")
                                
                                if residual_analysis['stationarity']['overall_stationary']:
                                    st.success("Остатки стационарны")
                                else:
                                    st.warning("Остатки не стационарны")
                        else:
                            st.error(f"Ошибка анализа остатков: {residual_analysis['error']}")
                            logger.error(f"Ошибка анализа остатков: {residual_analysis['error']}")
                    
                    else:
                        error_msg = result.get('error', 'Неизвестная ошибка') if result else 'Результат декомпозиции пуст'
                        st.error(f"Ошибка декомпозиции: {error_msg}")
                        logger.error(f"Ошибка декомпозиции: {error_msg}")
                    
                except Exception as e:
                    st.error(f"Ошибка декомпозиции: {e}")
                    logger.error(f"Ошибка декомпозиции: {e}")
    
    def forecasting_interface(self):
        """Интерфейс прогнозирования."""
        logger.info("Загрузка интерфейса прогнозирования")
        st.header("📈 Прогнозирование")
        
        if self.data is None:
            st.warning("Сначала загрузите данные")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            horizon = st.selectbox("Горизонт прогноза", [7, 14, 30, 90], index=0, key="forecast_horizon")
        
        with col2:
            strategy = st.selectbox("Стратегия прогнозирования", 
                                  ['recursive', 'direct', 'hybrid'], key="forecast_strategy")
        
        if st.button("Выполнить прогнозирование", type="primary", key="forecast_button"):
            with st.spinner("Выполняется прогнозирование..."):
                try:
                    logger.info(f"Начало прогнозирования: горизонт={horizon}, стратегия={strategy}")
                    
                    forecaster = MultiStepForecasting()
                    
                    # Подготовка данных
                    X, y, feature_columns = forecaster.prepare_data(self.data, self.target_column)
                    
                    if X is None or y is None:
                        st.error("Ошибка подготовки данных для прогнозирования")
                        logger.error("Ошибка подготовки данных для прогнозирования")
                        return
                    
                    X_train, X_test, y_train, y_test = forecaster.split_data(X, y)
                    
                    if X_train is None or X_test is None:
                        st.error("Ошибка разделения данных")
                        logger.error("Ошибка разделения данных")
                        return
                    
                    # Выполнение прогнозирования
                    from sklearn.linear_model import LinearRegression
                    
                    if strategy == 'recursive':
                        result = forecaster.recursive_strategy(
                            X_train, y_train, X_test, y_test, 
                            LinearRegression(), horizon
                        )
                    elif strategy == 'direct':
                        result = forecaster.direct_strategy(
                            X_train, y_train, X_test, y_test, 
                            LinearRegression, horizon
                        )
                    else:  # hybrid
                        result = forecaster.hybrid_strategy(
                            X_train, y_train, X_test, y_test, 
                            LinearRegression, horizon
                        )
                    
                    if result and 'error' not in result:
                        logger.info("Прогнозирование выполнено успешно")
                        
                        # Визуализация прогноза
                        fig = go.Figure()
                        
                        # Обучающие данные
                        fig.add_trace(go.Scatter(
                            x=list(range(len(y_train))),
                            y=y_train.values,
                            name='Обучающие данные',
                            line=dict(color='blue')
                        ))
                        
                        # Тестовые данные
                        fig.add_trace(go.Scatter(
                            x=list(range(len(y_train), len(y_train) + len(y_test))),
                            y=y_test.values,
                            name='Тестовые данные',
                            line=dict(color='green')
                        ))
                        
                        # Прогноз
                        forecast_steps = list(range(len(y_train), len(y_train) + horizon))
                        fig.add_trace(go.Scatter(
                            x=forecast_steps,
                            y=result['predictions'],
                            name='Прогноз',
                            line=dict(color='red', width=3)
                        ))
                        
                        fig.update_layout(
                            title=f'Прогноз ({strategy} стратегия)',
                            xaxis_title='Время',
                            yaxis_title='Значение',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Метрики качества
                        st.subheader("Метрики качества")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("MAE", f"{result['mae']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{result['rmse']:.4f}")
                        with col3:
                            st.metric("Время (сек)", f"{result['processing_time']:.2f}")
                        
                        # Детальные метрики по шагам
                        if 'step_metrics' in result:
                            st.subheader("Метрики по шагам")
                            step_data = []
                            for step_key, metrics in result['step_metrics'].items():
                                step_data.append({
                                    'Шаг': step_key,
                                    'MAE': metrics['mae'],
                                    'MSE': metrics['mse'],
                                    'Фактическое': metrics['actual'],
                                    'Прогноз': metrics['predicted']
                                })
                            
                            step_df = pd.DataFrame(step_data)
                            st.dataframe(step_df, use_container_width=True)
                    
                    else:
                        error_msg = result.get('error', 'Неизвестная ошибка') if result else 'Результат прогнозирования пуст'
                        st.error(f"Ошибка прогнозирования: {error_msg}")
                        logger.error(f"Ошибка прогнозирования: {error_msg}")
                    
                except Exception as e:
                    st.error(f"Ошибка прогнозирования: {e}")
                    logger.error(f"Ошибка прогнозирования: {e}")
    
    def exponential_smoothing_interface(self):
        """Интерфейс экспоненциального сглаживания."""
        logger.info("Загрузка интерфейса экспоненциального сглаживания")
        st.header("📊 Экспоненциальное сглаживание")
        
        if self.data is None:
            st.warning("Сначала загрузите данные")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            horizon = st.selectbox("Горизонт прогноза", [7, 14, 30], index=0, key="es_horizon")
        
        with col2:
            model_type = st.selectbox("Тип модели", 
                                    ['SES', 'Holt Additive', 'Holt Multiplicative'], key="es_model")
        
        if st.button("Выполнить экспоненциальное сглаживание", type="primary", key="es_button"):
            with st.spinner("Выполняется экспоненциальное сглаживание..."):
                try:
                    logger.info(f"Начало экспоненциального сглаживания: горизонт={horizon}, модель={model_type}")
                    
                    es_analyzer = ExponentialSmoothingModels()
                    
                    # Подготовка данных
                    train_series, test_series = es_analyzer.prepare_data(self.data, self.target_column)
                    
                    if train_series is None or test_series is None:
                        st.error("Ошибка подготовки данных для экспоненциального сглаживания")
                        logger.error("Ошибка подготовки данных для экспоненциального сглаживания")
                        return
                    
                    # Обучение модели
                    if model_type == 'SES':
                        model_result = es_analyzer.fit_ses_model(train_series)
                    elif model_type == 'Holt Additive':
                        model_result = es_analyzer.fit_holt_additive_model(train_series)
                    else:  # Holt Multiplicative
                        model_result = es_analyzer.fit_holt_multiplicative_model(train_series)
                    
                    if model_result:
                        logger.info(f"Модель {model_type} обучена успешно")
                        
                        # Генерация прогноза
                        forecast_data = es_analyzer.generate_forecast(model_result, horizon)
                        
                        if forecast_data:
                            logger.info("Прогноз сгенерирован успешно")
                            
                            # Визуализация
                            fig = go.Figure()
                            
                            # Обучающие данные
                            fig.add_trace(go.Scatter(
                                x=train_series.index,
                                y=train_series.values,
                                name='Обучающие данные',
                                line=dict(color='blue')
                            ))
                            
                            # Тестовые данные
                            fig.add_trace(go.Scatter(
                                x=test_series.index,
                                y=test_series.values,
                                name='Тестовые данные',
                                line=dict(color='green')
                            ))
                            
                            # Прогноз
                            forecast_index = pd.date_range(
                                start=train_series.index[-1] + timedelta(days=1),
                                periods=horizon,
                                freq='D'
                            )
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_index,
                                y=forecast_data['forecast'],
                                name='Прогноз',
                                line=dict(color='red', width=3)
                            ))
                            
                            # Доверительные интервалы
                            fig.add_trace(go.Scatter(
                                x=forecast_index,
                                y=forecast_data['upper_bound'],
                                fill=None,
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                showlegend=False
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=forecast_index,
                                y=forecast_data['lower_bound'],
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(0,0,0,0)',
                                name=f'{forecast_data["confidence_level"]*100:.0f}% ДИ'
                            ))
                            
                            fig.update_layout(
                                title=f'{model_type} - Прогноз',
                                xaxis_title='Время',
                                yaxis_title='Значение',
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Метрики модели
                            st.subheader("Метрики модели")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("AIC", f"{model_result['aic']:.2f}")
                            with col2:
                                st.metric("BIC", f"{model_result['bic']:.2f}")
                            with col3:
                                st.metric("Уровень доверия", f"{forecast_data['confidence_level']*100:.0f}%")
                            
                            # Параметры модели
                            st.subheader("Параметры модели")
                            params_df = pd.DataFrame([
                                {'Параметр': k, 'Значение': v} 
                                for k, v in model_result['params'].items()
                            ])
                            st.dataframe(params_df, use_container_width=True)
                            
                            # Оценка качества прогноза
                            forecast_metrics = es_analyzer.evaluate_forecast(forecast_data['forecast'], test_series)
                            if forecast_metrics:
                                st.subheader("Качество прогноза")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("MAE", f"{forecast_metrics['mae']:.4f}")
                                with col2:
                                    st.metric("RMSE", f"{forecast_metrics['rmse']:.4f}")
                                with col3:
                                    st.metric("MAPE", f"{forecast_metrics['mape']:.2f}%")
                        
                        else:
                            st.error("Ошибка генерации прогноза")
                            logger.error("Ошибка генерации прогноза")
                    
                    else:
                        st.error(f"Ошибка обучения модели {model_type}")
                        logger.error(f"Ошибка обучения модели {model_type}")
                    
                except Exception as e:
                    st.error(f"Ошибка экспоненциального сглаживания: {e}")
                    logger.error(f"Ошибка экспоненциального сглаживания: {e}")
    
    def comparison_interface(self):
        """Интерфейс сравнения моделей."""
        logger.info("Загрузка интерфейса сравнения моделей")
        st.header("📊 Сравнение моделей")
        
        if self.data is None:
            st.warning("Сначала загрузите данные")
            return
        
        if st.button("Запустить полное сравнение моделей", type="primary", key="comparison_button"):
            with st.spinner("Выполняется сравнение моделей..."):
                try:
                    logger.info("Начало сравнения моделей")
                    
                    # Создание таблицы сравнения
                    comparison_data = []
                    
                    # Экспоненциальное сглаживание
                    es_analyzer = ExponentialSmoothingModels()
                    train_series, test_series = es_analyzer.prepare_data(self.data, self.target_column)
                    
                    if train_series is None or test_series is None:
                        st.error("Ошибка подготовки данных для сравнения")
                        logger.error("Ошибка подготовки данных для сравнения")
                        return
                    
                    # SES
                    logger.info("Обучение SES модели для сравнения")
                    ses_result = es_analyzer.fit_ses_model(train_series)
                    if ses_result:
                        ses_forecast = es_analyzer.generate_forecast(ses_result, 7)
                        if ses_forecast:
                            ses_metrics = es_analyzer.evaluate_forecast(ses_forecast['forecast'], test_series)
                            if ses_metrics:
                                comparison_data.append({
                                    'Модель': 'SES',
                                    'MAE': ses_metrics['mae'],
                                    'RMSE': ses_metrics['rmse'],
                                    'MAPE': ses_metrics['mape'],
                                    'AIC': ses_result['aic']
                                })
                    
                    # Хольт аддитивный
                    logger.info("Обучение Хольт аддитивной модели для сравнения")
                    holt_add_result = es_analyzer.fit_holt_additive_model(train_series)
                    if holt_add_result:
                        holt_add_forecast = es_analyzer.generate_forecast(holt_add_result, 7)
                        if holt_add_forecast:
                            holt_add_metrics = es_analyzer.evaluate_forecast(holt_add_forecast['forecast'], test_series)
                            if holt_add_metrics:
                                comparison_data.append({
                                    'Модель': 'Holt Additive',
                                    'MAE': holt_add_metrics['mae'],
                                    'RMSE': holt_add_metrics['rmse'],
                                    'MAPE': holt_add_metrics['mape'],
                                    'AIC': holt_add_result['aic']
                                })
                    
                    # Хольт мультипликативный
                    logger.info("Обучение Хольт мультипликативной модели для сравнения")
                    holt_mul_result = es_analyzer.fit_holt_multiplicative_model(train_series)
                    if holt_mul_result:
                        holt_mul_forecast = es_analyzer.generate_forecast(holt_mul_result, 7)
                        if holt_mul_forecast:
                            holt_mul_metrics = es_analyzer.evaluate_forecast(holt_mul_forecast['forecast'], test_series)
                            if holt_mul_metrics:
                                comparison_data.append({
                                    'Модель': 'Holt Multiplicative',
                                    'MAE': holt_mul_metrics['mae'],
                                    'RMSE': holt_mul_metrics['rmse'],
                                    'MAPE': holt_mul_metrics['mape'],
                                    'AIC': holt_mul_result['aic']
                                })
                    
                    # Наивный прогноз
                    logger.info("Генерация наивного прогноза для сравнения")
                    naive_forecast = es_analyzer.naive_forecast(train_series, 7)
                    if len(naive_forecast) > 0:
                        naive_metrics = es_analyzer.evaluate_forecast(naive_forecast, test_series)
                        if naive_metrics:
                            comparison_data.append({
                                'Модель': 'Наивный прогноз',
                                'MAE': naive_metrics['mae'],
                                'RMSE': naive_metrics['rmse'],
                                'MAPE': naive_metrics['mape'],
                                'AIC': None
                            })
                    
                    if not comparison_data:
                        st.error("Не удалось обучить ни одной модели для сравнения")
                        logger.error("Не удалось обучить ни одной модели для сравнения")
                        return
                    
                    # Создание таблицы сравнения
                    comparison_df = pd.DataFrame(comparison_data)
                    logger.info(f"Создана таблица сравнения с {len(comparison_df)} моделями")
                    
                    # Визуализация сравнения
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['MAE', 'RMSE', 'MAPE', 'AIC'],
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    models = comparison_df['Модель']
                    
                    # MAE
                    fig.add_trace(
                        go.Bar(x=models, y=comparison_df['MAE'], name='MAE'),
                        row=1, col=1
                    )
                    
                    # RMSE
                    fig.add_trace(
                        go.Bar(x=models, y=comparison_df['RMSE'], name='RMSE'),
                        row=1, col=2
                    )
                    
                    # MAPE
                    fig.add_trace(
                        go.Bar(x=models, y=comparison_df['MAPE'], name='MAPE'),
                        row=2, col=1
                    )
                    
                    # AIC (только для моделей с AIC)
                    aic_data = comparison_df[comparison_df['AIC'].notna()]
                    if not aic_data.empty:
                        fig.add_trace(
                            go.Bar(x=aic_data['Модель'], y=aic_data['AIC'], name='AIC'),
                            row=2, col=2
                        )
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Таблица сравнения
                    st.subheader("Таблица сравнения")
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Определение лучшей модели
                    if not comparison_df.empty:
                        best_model_mae = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Модель']
                        best_model_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Модель']
                        
                        st.success(f"Лучшая модель по MAE: {best_model_mae}")
                        st.success(f"Лучшая модель по RMSE: {best_model_rmse}")
                        
                        logger.info(f"Лучшая модель по MAE: {best_model_mae}")
                        logger.info(f"Лучшая модель по RMSE: {best_model_rmse}")
                    
                except Exception as e:
                    st.error(f"Ошибка сравнения моделей: {e}")
                    logger.error(f"Ошибка сравнения моделей: {e}")
    
    def export_interface(self):
        """Интерфейс экспорта результатов."""
        logger.info("Загрузка интерфейса экспорта")
        st.header("💾 Экспорт результатов")
        
        if self.data is None:
            st.warning("Сначала загрузите данные")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_format = st.selectbox("Формат экспорта", ['CSV', 'JSON'], key="export_format")
        
        with col2:
            include_forecast = st.checkbox("Включить прогноз", value=True, key="include_forecast")
        
        if st.button("Экспортировать результаты", type="primary", key="export_button"):
            try:
                logger.info(f"Начало экспорта: формат={export_format}, прогноз={include_forecast}")
                
                # Подготовка данных для экспорта
                export_data = {
                    'data_info': {
                        'target_column': self.target_column,
                        'data_shape': self.data.shape,
                        'date_range': f"{self.data.index.min()} - {self.data.index.max()}",
                        'export_timestamp': datetime.now().isoformat()
                    }
                }
                
                if include_forecast:
                    # Добавляем прогноз
                    es_analyzer = ExponentialSmoothingModels()
                    train_series, test_series = es_analyzer.prepare_data(self.data, self.target_column)
                    
                    if train_series is not None and test_series is not None:
                        # Простой прогноз на 7 дней
                        naive_forecast = es_analyzer.naive_forecast(train_series, 7)
                        if len(naive_forecast) > 0:
                            export_data['forecast'] = {
                                'horizon': 7,
                                'method': 'naive',
                                'values': naive_forecast.tolist()
                            }
                            logger.info("Добавлен прогноз в экспорт")
                    else:
                        logger.warning("Не удалось подготовить данные для прогноза")
                
                if export_format == 'CSV':
                    # Экспорт в CSV
                    csv_buffer = io.StringIO()
                    self.data.to_csv(csv_buffer)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="Скачать CSV",
                        data=csv_data,
                        file_name=f"time_series_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
                    logger.info("CSV файл готов к скачиванию")
                
                else:  # JSON
                    # Экспорт в JSON
                    json_data = json.dumps(export_data, ensure_ascii=False, indent=2, default=str)
                    
                    st.download_button(
                        label="Скачать JSON",
                        data=json_data,
                        file_name=f"time_series_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_json"
                    )
                    logger.info("JSON файл готов к скачиванию")
                
                st.success("Результаты готовы к скачиванию!")
                
            except Exception as e:
                st.error(f"Ошибка экспорта: {e}")
                logger.error(f"Ошибка экспорта: {e}")


def main():
    """Основная функция веб-интерфейса."""
    logger.info("Запуск веб-интерфейса")
    
    try:
        st.set_page_config(
            page_title="Анализ временных рядов",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("📈 Анализ временных рядов")
        st.markdown("Интерактивный инструмент для анализа и прогнозирования временных рядов")
        
        # Создание экземпляра интерфейса
        interface = TimeSeriesWebInterface()
        
        # Создание вкладок
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📁 Загрузка данных",
            "🔍 Декомпозиция", 
            "📈 Прогнозирование",
            "📊 Экспоненциальное сглаживание",
            "📊 Сравнение моделей",
            "💾 Экспорт"
        ])
        
        with tab1:
            interface.load_data_interface()
        
        with tab2:
            interface.decomposition_interface()
        
        with tab3:
            interface.forecasting_interface()
        
        with tab4:
            interface.exponential_smoothing_interface()
        
        with tab5:
            interface.comparison_interface()
        
        with tab6:
            interface.export_interface()
        
        # Информация о программе
        st.sidebar.header("ℹ️ О программе")
        st.sidebar.markdown("""
        Этот инструмент позволяет:
        
        - **Декомпозиция**: Разложение временного ряда на компоненты
        - **Прогнозирование**: Многопшаговое прогнозирование
        - **Экспоненциальное сглаживание**: Классические модели сглаживания
        - **Сравнение моделей**: Оценка качества различных подходов
        - **Экспорт**: Сохранение результатов анализа
        
        Загрузите CSV файл с временным рядом для начала работы.
        """)
        
        logger.info("Веб-интерфейс запущен успешно")
        
    except Exception as e:
        logger.error(f"Ошибка в main: {e}")
        st.error(f"Ошибка запуска интерфейса: {e}")


if __name__ == "__main__":
    main()

