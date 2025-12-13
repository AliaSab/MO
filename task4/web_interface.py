"""
Веб-интерфейс для сравнения моделей прогнозирования временных рядов.
Показывает результаты всех 9 этапов пайплайна.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Импорты наших модулей
from feature_engineering import FeatureEngineer
from validation import DataValidator
from models import create_all_models, BaselineModels, ModelTrainer
from diagnostics import ModelDiagnostics
from evaluation import MetricsCalculator, ModelEvaluator, DieboldMarianoTest
from advanced_techniques import AdvancedTechniques
import time

# Настройка страницы
st.set_page_config(
    page_title="Сравнение моделей временных рядов",
    page_icon="📊",
    layout="wide"
)

# Инициализация сессии
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


def load_data_from_file(file_path='../New_final.csv'):
    """Загружает данные из файла."""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Обработка дат
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
                if df['Date'].dt.tz is not None:
                    df['Date'] = df['Date'].dt.tz_localize(None)
            return df
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        return None


def prepare_time_series(df, target_col='Weekly_Sales', date_col='Date', 
                        group_cols=None, aggregate=False):
    """Подготавливает временной ряд."""
    if group_cols is None:
        group_cols = []
    
    if aggregate and group_cols and all(col in df.columns for col in group_cols):
        # Агрегация
        agg_dict = {target_col: 'sum'}
        if 'IsHoliday' in df.columns:
            agg_dict['IsHoliday'] = 'max'
        df_grouped = df.groupby(group_cols + [date_col]).agg(agg_dict).reset_index()
        df_grouped = df_grouped.sort_values(date_col)
        date_index = pd.DatetimeIndex(df_grouped[date_col])
        series = df_grouped[target_col]
        is_holiday = df_grouped['IsHoliday'] if 'IsHoliday' in df_grouped.columns else None
    else:
        # Берем первый Store и Dept
        if 'Store' in df.columns and 'Dept' in df.columns:
            first_store = df['Store'].iloc[0]
            first_dept = df['Dept'].iloc[0]
            df_filtered = df[(df['Store'] == first_store) & (df['Dept'] == first_dept)]
        else:
            df_filtered = df
        
        df_sorted = df_filtered.sort_values(date_col)
        date_index = pd.DatetimeIndex(df_sorted[date_col])
        series = df_sorted[target_col]
        is_holiday = df_sorted['IsHoliday'] if 'IsHoliday' in df_sorted.columns else None
    
    return series, date_index, is_holiday


def run_pipeline_quick(df, target_col='Weekly_Sales', date_col='Date', 
                      horizon=7):
    """Запускает упрощенный пайплайн для веб-интерфейса."""
    results = {
        'models': {},
        'metrics': {},
        'predictions': {},
        'train_times': {},
        'feature_importance': {},
        'residuals': {}
    }
    
    try:
        # Подготовка данных
        series, date_index, is_holiday = prepare_time_series(
            df, target_col, date_col, group_cols=['Store', 'Dept'], aggregate=False
        )
        
        # Этап 1: Инжиниринг признаков
        feature_engineer = FeatureEngineer()
        X, y_transformed, transform_info = feature_engineer.create_all_features(
            series, date_index, is_holiday, apply_log=True, apply_boxcox=False
        )
        
        # Этап 2: Разбиение
        validator = DataValidator(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        X_train, X_val, X_test = validator.chronological_split(X, date_index)
        y_train, y_val, y_test = validator.chronological_split(y_transformed, date_index)
        date_train, date_val, date_test = validator.chronological_split(date_index, date_index)
        
        # Этап 5: Обучение моделей (все модели)
        all_models = create_all_models()
        selected_models = all_models
        
        trainer = ModelTrainer()
        for name, model in selected_models.items():
            trainer.add_model(name, model)
        
        # Обучение
        train_start = time.time()
        trained_models = trainer.train_all(X_train, y_train)
        train_time_total = time.time() - train_start
        
        # Предсказания
        predictions_val = trainer.predict_all(X_val[:min(len(X_val), 100)])
        
        # Бейзлайны
        baseline_preds = {}
        baseline_preds['Naive'] = BaselineModels.naive_forecast(y_train, 1)
        baseline_preds['SeasonalNaive'] = BaselineModels.seasonal_naive_forecast(y_train, 7, 1)
        baseline_preds['MovingAverage'] = BaselineModels.moving_average_forecast(y_train, 7, 1)
        
        # Оценка
        evaluator = ModelEvaluator(y_train=y_train, seasonality=7)
        
        all_predictions = {}
        for name, pred in predictions_val.items():
            if len(pred) > 0:
                all_predictions[name] = pred
        
        # Добавляем бейзлайны
        y_val_slice = y_val.iloc[:min(len(y_val), 100)] if isinstance(y_val, pd.Series) else y_val[:min(len(y_val), 100)]
        for name, baseline_pred in baseline_preds.items():
            all_predictions[name] = np.full(len(y_val_slice), baseline_pred[0] if len(baseline_pred) > 0 else y_train.iloc[-1])
        
        # Метрики
        max_len = max([len(pred) for pred in all_predictions.values()] + [len(y_val_slice)])
        y_val_final = y_val.iloc[:min(max_len, len(y_val))] if isinstance(y_val, pd.Series) else y_val[:min(max_len, len(y_val))]
        
        metrics_df = evaluator.evaluate_all_models(y_val_final, all_predictions)
        
        # Сохраняем результаты
        results['metrics'] = metrics_df.to_dict('records')
        results['predictions'] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in all_predictions.items()}
        results['y_true'] = y_val_final.tolist() if isinstance(y_val_final, pd.Series) else y_val_final.tolist()
        results['y_train'] = y_train.tolist() if isinstance(y_train, pd.Series) else y_train.tolist()
        results['dates_val'] = date_val[:len(y_val_final)].strftime('%Y-%m-%d').tolist()
        results['dates_train'] = date_train.strftime('%Y-%m-%d').tolist()
        results['trained_models'] = list(trained_models.keys())
        
        # Feature importance для топ-3
        diagnostics = ModelDiagnostics()
        if 'MASE' in metrics_df.columns:
            top_3 = metrics_df.nsmallest(3, 'MASE')['model'].tolist()
            for model_name in top_3:
                if model_name in trained_models:
                    importance = diagnostics.get_feature_importance(
                        trained_models[model_name], 
                        X_train.columns.tolist(), 
                        model_name
                    )
                    if importance is not None:
                        results['feature_importance'][model_name] = importance.to_dict('records')
        
        # Diebold-Mariano тест
        if len(all_predictions) >= 2:
            model_names = list(all_predictions.keys())[:5]
            min_len = min([len(all_predictions[k]) for k in model_names] + [len(y_val_final)])
            dm_predictions = {k: all_predictions[k][:min_len] for k in model_names}
            # Передаем горизонт прогнозирования для правильного учета автокорреляции
            dm_results = evaluator.compare_models_dm(y_val_final[:min_len], dm_predictions, h=horizon)
            results['dm_test'] = dm_results.to_dict()
        
        results['success'] = True
        results['n_features'] = X.shape[1]
        results['train_size'] = len(X_train)
        results['val_size'] = len(y_val_final)
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        st.error(f"Ошибка при выполнении пайплайна: {e}")
    
    return results


def display_comparison(results):
    """Отображает сравнение моделей."""
    if not results.get('success', False):
        st.error("Пайплайн не выполнен успешно. Проверьте данные.")
        return
    
    st.header("📊 Сравнение моделей прогнозирования")
    
    # Метрики
    if 'metrics' in results and results['metrics']:
        metrics_df = pd.DataFrame(results['metrics'])
        
        st.subheader("📈 Метрики качества моделей")
        
        # Сортируем по MASE
        if 'MASE' in metrics_df.columns:
            metrics_df = metrics_df.sort_values('MASE')
        
        # Отображаем таблицу с форматированием
        display_metrics = metrics_df.copy()
        for col in ['MAE', 'RMSE', 'MAPE', 'MASE', 'RMSSE']:
            if col in display_metrics.columns:
                display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        
        st.dataframe(
            display_metrics[['model', 'MAE', 'RMSE', 'MAPE', 'MASE', 'RMSSE']].head(15),
            use_container_width=True,
            hide_index=True
        )
        
        # Визуализация метрик
        st.subheader("📉 Визуализация метрик")
        
        top_10 = metrics_df.head(10)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAE', 'RMSE', 'MASE', 'MAPE'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        metrics_to_plot = ['MAE', 'RMSE', 'MASE', 'MAPE']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, pos in zip(metrics_to_plot, positions):
            if metric in top_10.columns:
                fig.add_trace(
                    go.Bar(
                        x=top_10['model'],
                        y=top_10[metric],
                        name=metric,
                        text=[f"{v:.4f}" for v in top_10[metric]],
                        textposition='auto'
                    ),
                    row=pos[0], col=pos[1]
                )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Сравнение метрик топ-10 моделей"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # График прогнозов
        st.subheader("🔮 Прогнозы моделей")
        
        if 'predictions' in results and 'y_true' in results:
            predictions = results['predictions']
            y_true = results['y_true']
            dates_val = results.get('dates_val', [])
            
            # Топ-5 моделей
            top_5_models = metrics_df.head(5)['model'].tolist()
            
            fig = go.Figure()
            
            # Фактические значения
            if dates_val:
                fig.add_trace(go.Scatter(
                    x=dates_val[:len(y_true)],
                    y=y_true,
                    mode='lines',
                    name='Факт',
                    line=dict(color='black', width=3)
                ))
            else:
                fig.add_trace(go.Scatter(
                    y=y_true,
                    mode='lines',
                    name='Факт',
                    line=dict(color='black', width=3)
                ))
            
            # Прогнозы топ-5 моделей
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for idx, model_name in enumerate(top_5_models):
                if model_name in predictions:
                    pred = predictions[model_name]
                    if isinstance(pred, list):
                        pred = np.array(pred)
                    
                    min_len = min(len(pred), len(y_true))
                    if dates_val:
                        fig.add_trace(go.Scatter(
                            x=dates_val[:min_len],
                            y=pred[:min_len],
                            mode='lines+markers',
                            name=f'{model_name}',
                            line=dict(color=colors[idx % len(colors)], width=2, dash='dash')
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            y=pred[:min_len],
                            mode='lines+markers',
                            name=f'{model_name}',
                            line=dict(color=colors[idx % len(colors)], width=2, dash='dash')
                        ))
            
            fig.update_layout(
                title="Сравнение прогнозов топ-5 моделей",
                xaxis_title="Дата" if dates_val else "Индекс",
                yaxis_title="Значение",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        if 'feature_importance' in results and results['feature_importance']:
            st.subheader("🎯 Важность признаков (топ-3 модели)")
            
            for model_name, importance_data in list(results['feature_importance'].items())[:3]:
                importance_df = pd.DataFrame(importance_data)
                # Используем абсолютные значения для важности
                importance_df['abs_importance'] = importance_df['importance'].abs()
                # Сортируем по убыванию абсолютной важности и берем топ-15
                importance_df = importance_df.sort_values('abs_importance', ascending=False).head(15)
                # Сортируем для отображения (от большего к меньшему)
                importance_df = importance_df.sort_values('abs_importance', ascending=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance_df['abs_importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    name=model_name,
                    marker=dict(color='steelblue')
                ))
                
                fig.update_layout(
                    title=f"Топ-15 признаков: {model_name}",
                    xaxis_title="Абсолютная важность",
                    yaxis_title="Признак",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Diebold-Mariano тест
        if 'dm_test' in results and results['dm_test']:
            st.subheader("📊 Статистическое сравнение (Diebold-Mariano тест)")
            
            try:
                dm_df = pd.DataFrame(results['dm_test'])
                
                # Показываем исходную таблицу с p-values
                st.markdown("**Таблица p-values:**")
                st.dataframe(dm_df, use_container_width=True)
                
                # Добавляем пояснение
                st.markdown("""
                **Интерпретация:**
                - Значения в таблице - это **p-values** теста Diebold-Mariano
                - **p-value < 0.05**: статистически значимое различие между моделями (одна модель лучше другой)
                - **p-value ≥ 0.05**: нет статистически значимого различия между моделями (модели статистически эквивалентны)
                - **"-"** означает сравнение модели с самой собой
                
                **Примечание:** Высокие p-values (близкие к 1.0) означают, что модели дают статистически неразличимые прогнозы. 
                Это нормально, если модели используют похожие алгоритмы и данные.
                """)
                
                # Показываем статистику по таблице
                p_values_list = []
                for col in dm_df.columns:
                    for idx in dm_df.index:
                        val = dm_df.loc[idx, col]
                        if val != '-' and isinstance(val, str):
                            try:
                                pval = float(val)
                                if not pd.isna(pval):
                                    p_values_list.append(pval)
                            except:
                                pass
                
                if p_values_list:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        significant_count = sum(1 for p in p_values_list if p < 0.05)
                        st.metric("Значимых различий", f"{significant_count}/{len(p_values_list)}")
                    with col2:
                        avg_pvalue = np.mean(p_values_list)
                        st.metric("Средний p-value", f"{avg_pvalue:.4f}")
                    with col3:
                        min_pvalue = np.min(p_values_list)
                        st.metric("Минимальный p-value", f"{min_pvalue:.4f}")
                    
            except Exception as e:
                st.warning(f"Не удалось отобразить DM тест: {e}")
        
        # Сводная информация
        st.subheader("ℹ️ Сводная информация")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Количество моделей", len(metrics_df))
        with col2:
            st.metric("Количество признаков", results.get('n_features', 'N/A'))
        with col3:
            st.metric("Размер обучающей выборки", results.get('train_size', 'N/A'))
        with col4:
            best_model = metrics_df.iloc[0]['model'] if len(metrics_df) > 0 else 'N/A'
            st.metric("Лучшая модель", best_model)
        
        if 'MASE' in metrics_df.columns and len(metrics_df) > 0:
            best_mase = metrics_df.iloc[0]['MASE']
            st.metric("Лучший MASE", f"{best_mase:.4f}")
    
    else:
        st.warning("Метрики не доступны")


def main():
    st.title("📊 Сравнение моделей прогнозирования временных рядов")
    st.markdown("---")
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        st.subheader("1. Загрузка данных")
        data_path = st.text_input(
            "Путь к файлу данных",
            value="../New_final.csv",
            help="Относительный путь к файлу New_final.csv"
        )
        
        if st.button("🔄 Загрузить данные", type="primary"):
            with st.spinner("Загрузка данных..."):
                df = load_data_from_file(data_path)
                if df is not None:
                    st.session_state.data = df
                    st.session_state.data_loaded = True
                    st.success(f"Загружено {len(df)} строк")
                else:
                    st.error("Не удалось загрузить данные")
        
        if st.session_state.data_loaded and 'data' in st.session_state:
            st.subheader("2. Параметры пайплайна")
            
            target_col = st.selectbox(
                "Целевая переменная",
                ['Weekly_Sales'],
                index=0
            )
            
            horizon = st.selectbox(
                "Горизонт прогнозирования",
                [1, 7, 14, 30],
                index=0,
                help="Для веб-интерфейса используется горизонт 1 для скорости"
            )
            
            if st.button("🚀 Запустить пайплайн", type="primary"):
                with st.spinner("Выполнение пайплайна... Это может занять несколько минут."):
                    results = run_pipeline_quick(
                        st.session_state.data,
                        target_col=target_col,
                        horizon=horizon
                    )
                    st.session_state.pipeline_results = results
                    if results.get('success', False):
                        st.success("Пайплайн выполнен успешно!")
                    else:
                        st.error(f"Ошибка: {results.get('error', 'Неизвестная ошибка')}")
    
    # Основная область
    if not st.session_state.data_loaded:
        st.info("👈 Загрузите данные в боковой панели")
        st.markdown("""
        ### Инструкция:
        1. Укажите путь к файлу `New_final.csv` (по умолчанию `../New_final.csv`)
        2. Нажмите "Загрузить данные"
        3. Настройте параметры пайплайна
        4. Нажмите "Запустить пайплайн"
        5. Просмотрите результаты сравнения моделей
        
        ### Что показывается:
        - 📈 Метрики качества всех моделей (MAE, RMSE, MAPE, MASE, RMSSE)
        - 📉 Визуализация метрик
        - 🔮 Графики прогнозов топ-5 моделей
        - 🎯 Важность признаков для лучших моделей
        - 📊 Статистическое сравнение (Diebold-Mariano тест)
        """)
    else:
        if 'data' in st.session_state:
            with st.expander("📋 Просмотр данных", expanded=False):
                st.dataframe(st.session_state.data.head(100))
                st.write(f"Размер данных: {st.session_state.data.shape}")
        
        if st.session_state.pipeline_results is not None:
            display_comparison(st.session_state.pipeline_results)
        else:
            st.info("👆 Нажмите 'Запустить пайплайн' для начала анализа")


if __name__ == "__main__":
    main()





