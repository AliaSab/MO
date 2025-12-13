"""
Веб-интерфейс для визуального сравнения моделей глубокого обучения.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import warnings
import time
from pathlib import Path
warnings.filterwarnings('ignore')

# Импорты наших модулей
from preprocessing import TimeSeriesPreprocessor
from feature_engineering import FeatureEngineer
from models import create_all_models
from training import train_model, ModelTrainer, TimeSeriesDataset
from torch.utils.data import DataLoader
from evaluation import MetricsCalculator, ModelEvaluator
from diagnostics import ModelDiagnostics

# Настройка страницы
st.set_page_config(
    page_title="Глубокое обучение для временных рядов",
    page_icon="🧠",
    layout="wide"
)

# Инициализация сессии
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = TimeSeriesPreprocessor()
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'run_training' not in st.session_state:
    st.session_state.run_training = False
if 'training_params' not in st.session_state:
    st.session_state.training_params = {}
if 'device' not in st.session_state:
    st.session_state.device = 'cuda' if torch.cuda.is_available() else 'cpu'

device = st.session_state.device


def load_data(uploaded_file):
    """Загружает данные из файла."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Поддерживаются только CSV и Parquet файлы")
            return None
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        return None


def prepare_time_series(df, date_column, target_column):
    """Подготавливает временной ряд."""
    try:
        # Создаем копию, чтобы не изменять исходный DataFrame
        df = df.copy()
        
        df[date_column] = pd.to_datetime(df[date_column], utc=True)
        if df[date_column].dt.tz is not None:
            df[date_column] = df[date_column].dt.tz_localize(None)
        
        # Сортируем по дате
        df = df.sort_values(date_column)
        
        # Группируем по Store и Dept (как в run_pipeline.py)
        if 'Store' in df.columns and 'Dept' in df.columns:
            # Проверяем, сколько уникальных дат до группировки
            unique_dates_before = df[date_column].nunique()
            
            # Агрегируем по всем магазинам и отделам (используем среднее значение по датам)
            df = df.groupby(date_column)[target_column].mean().reset_index()
            df = df.set_index(date_column)
            # Сортируем по дате еще раз после группировки
            df = df.sort_index()
            
            unique_dates_after = len(df)
            # Если после группировки осталось очень мало данных, предупреждаем
            if unique_dates_after < 100:
                st.warning(f"⚠️ После группировки по дате осталось только {unique_dates_after} наблюдений "
                          f"(было {unique_dates_before} уникальных дат). "
                          f"Это может привести к малому количеству последовательностей для обучения.")
        else:
            df = df.set_index(date_column)
        
        if target_column not in df.columns:
            st.error(f"Колонка {target_column} не найдена")
            return None, None
        
        y = df[target_column].dropna()
        dates = y.index
        
        return y, dates
    except Exception as e:
        st.error(f"Ошибка при подготовке данных: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None


def plot_predictions_interactive(y_true, y_pred, dates=None, model_name="Model"):
    """Интерактивный график прогнозов."""
    fig = go.Figure()
    
    # Преобразуем в numpy массивы для надежности
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Проверяем, что длины совпадают
    min_len = min(len(y_true), len(y_pred))
    if len(y_true) != len(y_pred):
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    # Проверяем на NaN и Inf
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(valid_mask) == 0:
        # Нет валидных данных
        fig.add_annotation(
            text="Нет валидных данных для отображения",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Используем только валидные данные
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    # Обрабатываем даты
    if dates is not None:
        # Преобразуем в массив для индексации
        if hasattr(dates, 'tolist'):
            dates_list = np.array(dates.tolist())
        elif hasattr(dates, 'values'):
            dates_list = np.array(dates.values)
        elif isinstance(dates, (list, tuple, np.ndarray)):
            dates_list = np.array(dates)
        else:
            dates_list = np.array([dates])
        
        # Фильтруем даты по валидной маске
        if len(dates_list) == len(valid_mask):
            dates_list = dates_list[valid_mask]
        elif len(dates_list) > len(valid_mask):
            dates_list = dates_list[:len(valid_mask)][valid_mask[:len(dates_list)]]
        else:
            dates_list = np.array(list(range(len(y_true))))
        
        x_axis = dates_list
    else:
        x_axis = np.array(range(len(y_true)))
    
    # Определяем режим отображения в зависимости от количества точек
    if len(y_true) == 1:
        mode_true = 'markers'
        mode_pred = 'markers'
        marker_size = 10
    elif len(y_true) <= 5:
        mode_true = 'markers+lines'
        mode_pred = 'markers+lines'
        marker_size = 8
    else:
        mode_true = 'lines+markers'
        mode_pred = 'lines+markers'
        marker_size = 4
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=y_true,
        mode=mode_true,
        name='Факт',
        line=dict(color='blue', width=2) if len(y_true) > 1 else None,
        marker=dict(size=marker_size, color='blue', symbol='circle')
    ))
    
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=y_pred,
        mode=mode_pred,
        name='Прогноз',
        line=dict(color='red', width=2, dash='dash') if len(y_pred) > 1 else None,
        marker=dict(size=marker_size, color='red', symbol='diamond')
    ))
    
    # Настройка осей
    fig.update_layout(
        title=f'Прогнозы модели {model_name} (точек: {len(y_true)})',
        xaxis_title='Время',
        yaxis_title='Значение',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    # Настройка формата оси Y для больших чисел
    max_val = max(np.max(np.abs(y_true)), np.max(np.abs(y_pred))) if len(y_true) > 0 else 0
    if max_val > 1000:
        fig.update_layout(yaxis=dict(tickformat='.2s'))  # Научная нотация для больших чисел
    
    return fig


def plot_learning_curves_interactive(train_losses, val_losses, model_name="Model"):
    """Интерактивный график кривых обучения."""
    fig = go.Figure()
    
    epochs = range(1, len(train_losses) + 1)
    
    fig.add_trace(go.Scatter(
        x=list(epochs),
        y=train_losses,
        mode='lines',
        name='Train Loss',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(epochs),
        y=val_losses,
        mode='lines',
        name='Validation Loss',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f'Кривые обучения: {model_name}',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_model_comparison_interactive(results_dict, metric='MASE'):
    """Интерактивное сравнение моделей."""
    model_names = list(results_dict.keys())
    metric_values = []
    
    for name in model_names:
        metrics = results_dict[name].get('metrics', {})
        metric_values.append(metrics.get(metric, np.nan))
    
    # Сортируем
    sorted_data = sorted(zip(model_names, metric_values), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
    model_names, metric_values = zip(*sorted_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(metric_values),
        y=list(model_names),
        orientation='h',
        marker=dict(color='steelblue', opacity=0.7),
        text=[f'{v:.4f}' if not np.isnan(v) else 'N/A' for v in metric_values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Сравнение моделей по {metric}',
        xaxis_title=metric,
        yaxis_title='Модель',
        height=400 + len(model_names) * 30,
        showlegend=False
    )
    
    return fig


def main():
    st.title("🧠 Глубокое обучение для прогнозирования временных рядов")
    st.markdown("---")
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Загрузка данных
        st.subheader("1. Загрузка данных")
        uploaded_file = st.file_uploader("Загрузите CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            if st.session_state.data is None or st.button("Перезагрузить данные"):
                st.session_state.data = load_data(uploaded_file)
                if st.session_state.data is not None:
                    st.success("Данные загружены успешно!")
        
        if st.session_state.data is not None:
            st.subheader("2. Выбор колонок")
            date_columns = [col for col in st.session_state.data.columns 
                          if pd.api.types.is_datetime64_any_dtype(st.session_state.data[col]) or
                          'date' in col.lower() or 'time' in col.lower()]
            
            if not date_columns:
                date_columns = st.session_state.data.columns.tolist()
            
            date_column = st.selectbox("Колонка с датами", date_columns)
            target_column = st.selectbox("Целевая переменная", 
                                        st.session_state.data.columns.tolist())
            
            st.subheader("3. Параметры модели")
            lookback = st.slider("Lookback (окно истории)", 24, 500, 336, 
                               help="Рекомендуется 336 для недельных данных. Меньше = быстрее обучение")
            horizon = st.selectbox("Horizon (горизонт прогноза)", [24, 48, 168], index=1)
            
            st.subheader("4. Предобработка")
            transform_type = st.selectbox("Трансформация", 
                                         ['boxcox', 'log', 'none'], 
                                         index=0)
            scaler_type = st.selectbox("Нормализация", 
                                     ['standard', 'minmax'], 
                                     index=0)
            
            st.subheader("5. Параметры обучения")
            st.markdown("**💡 Для быстрого обучения используйте:** epochs=20-30, batch_size=64-128")
            epochs = st.slider("Эпохи", 5, 200, 20, 
                             help="Меньше эпох = быстрее обучение. 20-30 обычно достаточно с early stopping")
            batch_size = st.slider("Batch size", 16, 256, 64,
                                 help="Больше batch_size = быстрее обучение, но больше памяти")
            learning_rate = st.selectbox("Learning rate", 
                                        [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], 
                                        index=2)
            optimizer = st.selectbox("Оптимизатор", 
                                    ['adam', 'adamw'], 
                                    index=0)
            loss_fn = st.selectbox("Функция потерь", 
                                  ['mse', 'mae', 'huber', 'mse+mae'], 
                                  index=3)
            
            st.subheader("6. Выбор моделей")
            
            # Группируем модели по сложности
            basic_models = ['MLP', 'DLinear', 'NLinear', 'Naive', 'SeasonalNaive']
            rnn_models = ['RNN', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU']
            advanced_models = ['TCN', 'N-BEATS', 'N-HiTS', 'Transformer', 'CNN-LSTM', 'CNN-GRU']
            sota_models = ['Informer', 'Autoformer', 'PatchTST', 'TFT', 'TCN-Attention', 'LSTM-AE']
            
            model_group = st.radio(
                "Группа моделей:",
                ["Базовые (быстрые)", "RNN (средние)", "Продвинутые (медленные)", "SOTA (очень медленные)", "Все"],
                index=0,
                help="Базовые - для быстрого тестирования, SOTA - самые сложные модели"
            )
            
            if model_group == "Базовые (быстрые)":
                available_models = basic_models
                default_models = ['LSTM', 'DLinear']
            elif model_group == "RNN (средние)":
                available_models = rnn_models
                default_models = ['LSTM', 'GRU']
            elif model_group == "Продвинутые (медленные)":
                available_models = advanced_models
                default_models = ['N-BEATS', 'TCN']
            elif model_group == "SOTA (очень медленные)":
                available_models = sota_models
                default_models = ['Informer', 'TFT']
            else:
                available_models = basic_models + rnn_models + advanced_models + sota_models
                default_models = ['LSTM', 'DLinear']
            
            st.markdown("**💡 Для быстрого тестирования выберите 1-2 модели**")
            selected_models = st.multiselect("Выберите модели", 
                                            available_models, 
                                            default=[m for m in default_models if m in available_models],
                                            help="Меньше моделей = быстрее обучение. Рекомендуется начать с 1-2 моделей")
            
            if st.button("🚀 Запустить обучение", type="primary"):
                st.session_state.run_training = True
                st.session_state.training_params = {
                    'date_column': date_column,
                    'target_column': target_column,
                    'lookback': lookback,
                    'horizon': horizon,
                    'transform_type': transform_type,
                    'scaler_type': scaler_type,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'optimizer': optimizer,
                    'loss_fn': loss_fn,
                    'selected_models': selected_models,
                }
    
    # Основная область
    if st.session_state.data is None:
        st.info("👈 Загрузите данные в боковой панели")
        st.markdown("""
        ### Инструкция:
        1. Загрузите CSV файл с временным рядом (например, New_final.csv)
        2. Выберите колонку с датами и целевую переменную
        3. Настройте параметры модели (lookback, horizon)
        4. Выберите предобработку и параметры обучения
        5. Выберите модели для сравнения
        6. Нажмите "Запустить обучение"
        
        ### Поддерживаемые модели:
        - **Базовые**: MLP, TCN, N-BEATS
        - **Рекуррентные**: RNN, LSTM, GRU, BiLSTM, BiGRU
        - **Трансформеры**: Transformer
        - **Гибриды**: CNN-LSTM, CNN-GRU
        - **Бейзлайны**: DLinear, NLinear, Naive, SeasonalNaive
        """)
    else:
        # Просмотр данных
        with st.expander("📊 Просмотр данных", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(st.session_state.data.head(100))
            with col2:
                st.write(f"**Размер данных:** {st.session_state.data.shape}")
                st.write(f"**Колонки:** {list(st.session_state.data.columns)}")
        
        # Запуск обучения
        if st.session_state.get('run_training', False) and 'training_params' in st.session_state:
            params = st.session_state.training_params
            try:
                run_training_pipeline(params)
            except Exception as e:
                st.error(f"Ошибка при выполнении пайплайна: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                st.session_state.run_training = False
        
        # Отображение результатов
        if st.session_state.results:
            display_results()


def run_training_pipeline(params):
    """Запускает пайплайн обучения."""
    st.header("🔄 Обучение моделей")
    
    # Подготовка данных (без spinner для ускорения)
    try:
        st.write("📥 Подготовка данных...")
        y, dates = prepare_time_series(
            st.session_state.data,
            params['date_column'],
            params['target_column']
        )
        
        if y is None:
            st.error("Ошибка при подготовке данных")
            return
        
        if len(y) < 100:
            st.warning(f"⚠️ Мало данных: {len(y)} наблюдений. Рекомендуется минимум 1000 для глубокого обучения.")
        
        # Предобработка
        st.write("🔄 Предобработка данных...")
        preprocessor = TimeSeriesPreprocessor(scaler_type=params['scaler_type'])
        transform_type = params['transform_type'] if params['transform_type'] != 'none' else None
        
        (X_train, y_train, train_dates), \
        (X_val, y_val, val_dates), \
        (X_test, y_test, test_dates), \
        preprocessor_info = preprocessor.prepare_data(
            y,
            lookback=params['lookback'],
            horizon=params['horizon'],
            apply_transform=transform_type
        )
        
        st.success(f"✅ Данные подготовлены: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Проверяем фактические размеры данных
        st.info(f"📊 **Фактические размеры:** X_train={X_train.shape}, y_train={y_train.shape}")
        
    except Exception as e:
        st.error(f"Ошибка при подготовке данных: {e}")
        import traceback
        st.code(traceback.format_exc())
        return
    
    # Обучение моделей
    # input_size - это количество признаков (features), а не длина последовательности
    # Для моделей, которые работают с последовательностями, нужен lookback
    input_size = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
    # Используем фактические параметры из preprocessor_info (могут быть изменены автоматически)
    actual_lookback = preprocessor_info.get('lookback', params['lookback'])
    actual_horizon = preprocessor_info.get('horizon', params['horizon'])
    # Также проверяем фактические размеры в данных
    actual_seq_len = X_train.shape[1] if len(X_train.shape) == 3 else X_train.shape[0]
    actual_y_horizon = y_train.shape[1] if len(y_train.shape) > 1 else 1
    
    st.info(f"📊 **Параметры моделей:** input_size={input_size}, lookback={actual_lookback}, horizon={actual_horizon}")
    st.info(f"📊 **Фактические размеры данных:** X_train={X_train.shape}, y_train={y_train.shape}, "
            f"actual_seq_len={actual_seq_len}, actual_y_horizon={actual_y_horizon}")
    
    # Проверяем соответствие параметров
    if actual_seq_len != actual_lookback:
        st.warning(f"⚠️ Фактический размер последовательности ({actual_seq_len}) не совпадает с lookback ({actual_lookback}). Используем фактический размер.")
        actual_lookback = actual_seq_len
    
    if actual_y_horizon != actual_horizon:
        st.warning(f"⚠️ Фактический размер горизонта в y ({actual_y_horizon}) не совпадает с horizon ({actual_horizon}). Используем фактический размер.")
        actual_horizon = actual_y_horizon
    
    results = {}
    models = {}
    
    # Информация о параметрах (выводим один раз)
    st.info(f"📊 **Параметры обучения:** epochs={params['epochs']}, batch_size={params['batch_size']}, "
            f"lookback={actual_lookback}, horizon={actual_horizon}")
    
    # Создаем контейнер для результатов
    results_container = st.container()
    
    # Создаем статус-бар для отслеживания прогресса
    status_placeholder = st.empty()
    
    for idx, model_name in enumerate(params['selected_models']):
        try:
            # Обновляем статус
            status_placeholder.info(f"🔄 Обучение модели {idx+1}/{len(params['selected_models'])}: **{model_name}**")
            
            # Создаем модель с фактическими параметрами (lookback и horizon могут быть изменены автоматически)
            model = create_all_models(input_size, horizon=actual_horizon, lookback=actual_lookback)[model_name]
            
            # Параметры обучения (оптимизированы для скорости и предотвращения переобучения)
            # Для малого количества данных используем более агрессивный early stopping
            trainer_kwargs = {
                'loss_fn': params['loss_fn'],
                'optimizer': params['optimizer'],
                'lr': params['learning_rate'],
                'weight_decay': 1e-3,  # Увеличено для лучшей регуляризации
                'gradient_clip': 1.0,
                'early_stopping_patience': 5,  # Уменьшено для предотвращения переобучения
                'reduce_lr_patience': 3,  # Уменьшено для более быстрой адаптации
            }
            
            # Обучаем без verbose для ускорения (отключаем вывод в консоль)
            status_placeholder.info(f"⏳ Обучение {model_name}... (это может занять некоторое время)")
            start_time = time.time()
            
            # Выполняем обучение
            trainer, train_losses, val_losses = train_model(
                model, X_train, y_train, X_val, y_val,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                device=device,
                verbose=False,  # Отключаем вывод для ускорения
                **trainer_kwargs
            )
            train_time = time.time() - start_time
            
            # Выводим результат обучения
            status_placeholder.success(f"✅ {model_name} обучен за {train_time:.2f} сек")
            
            # Предсказания на валидации (оптимизировано для скорости)
            val_dataset = TimeSeriesDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], 
                                  shuffle=False, num_workers=0, pin_memory=False)
            y_pred_val, y_true_val = trainer.predict(val_loader)
            
            # Обратная трансформация (аналогично run_pipeline.py)
            # Используем первый шаг горизонта для метрик
            if len(y_pred_val.shape) > 1:
                y_pred_val_flat = y_pred_val[:, 0]  # Первый шаг горизонта
                y_true_val_flat = y_true_val[:, 0]  # Первый шаг горизонта
            else:
                y_pred_val_flat = y_pred_val
                y_true_val_flat = y_true_val
            
            # Информация о прогнозах (только для диагностики, если нужно)
            # st.write(f"🔍 **{model_name}:** Форма прогнозов: {y_pred_val.shape}")
            
            # Проверка на NaN/inf перед обратной трансформацией
            if np.any(np.isnan(y_pred_val_flat)) or np.any(np.isinf(y_pred_val_flat)):
                st.warning(f"⚠️ Обнаружены NaN/inf в прогнозах {model_name} перед обратной трансформацией")
                y_pred_val_flat = np.nan_to_num(y_pred_val_flat, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Сначала обратная нормализация
            y_pred_val_scaled = preprocessor.inverse_transform(y_pred_val_flat)
            y_true_val_scaled = preprocessor.inverse_transform(y_true_val_flat)
            
            # Проверка после обратной нормализации
            if np.any(np.isnan(y_pred_val_scaled)) or np.any(np.isinf(y_pred_val_scaled)):
                st.warning(f"⚠️ Обнаружены NaN/inf в прогнозах {model_name} после обратной нормализации")
                y_pred_val_scaled = np.nan_to_num(y_pred_val_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
            
            if preprocessor_info['transform'] == 'boxcox':
                y_pred_val_orig = preprocessor.inverse_boxcox(
                    y_pred_val_scaled, preprocessor_info['lambda_boxcox']
                )
                y_true_val_orig = preprocessor.inverse_boxcox(
                    y_true_val_scaled, preprocessor_info['lambda_boxcox']
                )
            elif preprocessor_info['transform'] == 'log':
                y_pred_val_orig = preprocessor.inverse_log(y_pred_val_scaled)
                y_true_val_orig = preprocessor.inverse_log(y_true_val_scaled)
            else:
                y_pred_val_orig = y_pred_val_scaled
                y_true_val_orig = y_true_val_scaled
            
            # Финальная проверка
            if np.any(np.isnan(y_pred_val_orig)) or np.any(np.isinf(y_pred_val_orig)):
                st.warning(f"⚠️ Обнаружены NaN/inf в финальных прогнозах {model_name}")
                y_pred_val_orig = np.nan_to_num(y_pred_val_orig, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Метрики
            metrics_calc = MetricsCalculator()
            if len(y_train.shape) > 1:
                y_train_flat = y_train[:, 0]  # Первый шаг
            else:
                y_train_flat = y_train
            
            y_train_scaled = preprocessor.inverse_transform(y_train_flat)
            if preprocessor_info['transform'] == 'boxcox':
                y_train_orig = preprocessor.inverse_boxcox(
                    y_train_scaled, preprocessor_info['lambda_boxcox']
                )
            elif preprocessor_info['transform'] == 'log':
                y_train_orig = preprocessor.inverse_log(y_train_scaled)
            else:
                y_train_orig = y_train_scaled
            
            metrics = metrics_calc.calculate_all_metrics(
                y_true_val_orig, y_pred_val_orig, y_train_orig, seasonality=7
            )
            
            # Проверка: почему MAE = RMSE (это возможно только если все ошибки одинаковые по модулю)
            # Это нормально для малого количества точек (например, 2 точки)
            if len(y_true_val_orig) > 0 and len(y_pred_val_orig) > 0:
                errors = np.abs(y_true_val_orig - y_pred_val_orig)
                errors = errors[np.isfinite(errors)]
                if len(errors) > 0 and len(errors) <= 2:
                    # Для 1-2 точек MAE может быть равен RMSE, если ошибки одинаковые по модулю
                    # Это нормально и не является ошибкой
                    pass
            
            # Сохраняем результаты
            # Для визуализации используем первый шаг горизонта (как и для метрик)
            # Это обеспечивает согласованность между метриками и графиками
            y_pred_val_viz = y_pred_val_orig
            y_true_val_viz = y_true_val_orig
            
            models[model_name] = trainer
            results[model_name] = {
                'metrics': metrics,
                'time': train_time,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'y_pred': y_pred_val_viz,  # Для визуализации
                'y_true': y_true_val_viz,  # Для визуализации
                'dates': val_dates,
            }
            
        except Exception as e:
            st.error(f"Ошибка при обучении {model_name}: {e}")
            import traceback
            st.code(traceback.format_exc())
            continue
    
    # Сохраняем результаты только если есть успешно обученные модели
    if models and results:
        st.session_state.models = models
        st.session_state.results = results
        st.success(f"✅ Обучение завершено! Успешно обучено {len(models)} моделей.")
    else:
        st.warning("⚠️ Не удалось обучить ни одной модели. Проверьте параметры и данные.")


def display_results():
    """Отображает результаты."""
    st.header("📊 Результаты сравнения моделей")
    
    results = st.session_state.results
    
    # Сводная таблица
    st.subheader("Сводная таблица метрик")
    evaluator = ModelEvaluator()
    comparison_table = evaluator.create_comparison_table(results, sort_by='MASE')
    
    # Форматируем таблицу для отображения (заменяем NaN на "N/A")
    display_table = comparison_table.copy()
    for col in display_table.columns:
        if col != 'Модель':
            display_table[col] = display_table[col].apply(
                lambda x: 'N/A' if (isinstance(x, float) and np.isnan(x)) or x is None else x
            )
    
    st.dataframe(display_table, use_container_width=True)
    
    # Графики сравнения
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Сравнение по MASE")
        fig_mase = plot_model_comparison_interactive(results, metric='MASE')
        st.plotly_chart(fig_mase, use_container_width=True)
    
    with col2:
        st.subheader("Сравнение по RMSE")
        fig_rmse = plot_model_comparison_interactive(results, metric='RMSE')
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Детальные результаты по моделям
    st.subheader("Детальные результаты по моделям")
    
    selected_model = st.selectbox("Выберите модель для детального просмотра", 
                                 list(results.keys()))
    
    if selected_model:
        model_result = results[selected_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("MASE", f"{model_result['metrics'].get('MASE', 'N/A'):.4f}")
            st.metric("MAE", f"{model_result['metrics'].get('MAE', 'N/A'):.4f}")
            st.metric("RMSE", f"{model_result['metrics'].get('RMSE', 'N/A'):.4f}")
        
        with col2:
            st.metric("MAPE", f"{model_result['metrics'].get('MAPE', 'N/A'):.2f}%")
            r2_value = model_result['metrics'].get('R2', np.nan)
            if isinstance(r2_value, (int, float)) and not np.isnan(r2_value):
                st.metric("R2", f"{r2_value:.4f}")
            else:
                st.metric("R2", "N/A")
            st.metric("Время обучения", f"{model_result['time']:.2f} сек")
        
        # Графики
        st.subheader(f"Прогнозы модели {selected_model}")
        fig_pred = plot_predictions_interactive(
            model_result['y_true'],
            model_result['y_pred'],
            model_result.get('dates'),
            selected_model
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Кривые обучения
        if 'train_losses' in model_result and 'val_losses' in model_result:
            st.subheader(f"Кривые обучения: {selected_model}")
            fig_learning = plot_learning_curves_interactive(
                model_result['train_losses'],
                model_result['val_losses'],
                selected_model
            )
            st.plotly_chart(fig_learning, use_container_width=True)


if __name__ == "__main__":
    main()

