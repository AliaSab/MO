"""
Главный скрипт для выполнения полного пайплайна прогнозирования временных рядов.
Объединяет все 9 этапов задания.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
os.environ['TCL_LIBRARY'] = "C:/Program Files/Python313/tcl/tcl8.6"
os.environ['TK_LIBRARY'] = "C:/Program Files/Python313/tcl/tk8.6"
import time
import warnings
warnings.filterwarnings('ignore')

# Импорты наших модулей
from feature_engineering import FeatureEngineer
from validation import DataValidator
from hyperparameter_tuning import HyperparameterTuner
from forecasting_strategies import DirectStrategy, RecursiveStrategy, MultiOutputStrategy, DirRecStrategy
from models import create_all_models, BaselineModels, ModelTrainer
from diagnostics import ModelDiagnostics
from evaluation import MetricsCalculator, ModelEvaluator, DieboldMarianoTest
from advanced_techniques import AdvancedTechniques

# Попытка импорта альтернативных моделей
try:
    from alternative_models import AlternativeTimeSeriesModels
    ALTERNATIVE_MODELS_AVAILABLE = True
except ImportError:
    ALTERNATIVE_MODELS_AVAILABLE = False
    print("Альтернативные модели временных рядов не доступны")

# Попытка импорта AutoGluon
try:
    from autogluon_integration import AutoGluonWrapper, compare_autogluon_vs_custom, create_autogluon_recommendations
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("AutoGluon не доступен")

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")


def load_data(file_path):
    """Загружает данные из CSV файла."""
    print(f"Загрузка данных из {file_path}...")
    df = pd.read_csv(file_path)
    
    # Обработка дат
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
    
    print(f"Загружено {len(df)} строк")
    return df


def prepare_time_series(df, target_col='Weekly_Sales', date_col='Date', 
                        group_cols=None):
    """
    Подготавливает временной ряд.
    
    Если есть группирующие колонки (Store, Dept), агрегируем по ним.
    """
    if group_cols is None:
        group_cols = []
    
    # Если есть группирующие колонки, агрегируем
    if group_cols and all(col in df.columns for col in group_cols):
        print(f"Агрегация по {group_cols}...")
        # Агрегируем целевую переменную
        agg_dict = {target_col: 'sum'}
        if 'IsHoliday' in df.columns:
            agg_dict['IsHoliday'] = 'max'  # Берем max (True если хотя бы один True)
        
        df_grouped = df.groupby(group_cols + [date_col]).agg(agg_dict).reset_index()
        df_grouped = df_grouped.sort_values(date_col)
        date_index = pd.DatetimeIndex(df_grouped[date_col])
        series = df_grouped[target_col]
        is_holiday = df_grouped['IsHoliday'] if 'IsHoliday' in df_grouped.columns else None
    else:
        # Простой случай - один ряд
        df_sorted = df.sort_values(date_col)
        date_index = pd.DatetimeIndex(df_sorted[date_col])
        series = df_sorted[target_col]
        is_holiday = df_sorted['IsHoliday'] if 'IsHoliday' in df_sorted.columns else None
    
    return series, date_index, is_holiday


def main():
    """Главная функция пайплайна."""
    print("=" * 80)
    print("ПАЙПЛАЙН ПРОГНОЗИРОВАНИЯ ВРЕМЕННЫХ РЯДОВ")
    print("=" * 80)
    
    # Параметры
    DATA_PATH = 'New_final.csv'
    TARGET_COL = 'Weekly_Sales'
    DATE_COL = 'Date'
    HORIZON = 7  # Горизонт прогнозирования
    RESULTS_DIR = 'results'
    
    # Для агрегации: если True, то агрегируем по Store и Dept
    # Если False, то берем первый Store и Dept для демонстрации
    AGGREGATE_BY_GROUP = False
    
    # Создаем директорию для результатов
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # ========== ЗАГРУЗКА ДАННЫХ ==========
    print("\n[ЭТАП 0] Загрузка данных...")
    df = load_data(DATA_PATH)
    
    # Для демонстрации берем один Store и Dept, или агрегируем
    if AGGREGATE_BY_GROUP:
        series, date_index, is_holiday = prepare_time_series(
            df, TARGET_COL, DATE_COL, group_cols=['Store', 'Dept']
        )
    else:
        # Берем первый Store и Dept для демонстрации
        first_store = df['Store'].iloc[0] if 'Store' in df.columns else None
        first_dept = df['Dept'].iloc[0] if 'Dept' in df.columns else None
        if first_store is not None and first_dept is not None:
            df_filtered = df[(df['Store'] == first_store) & (df['Dept'] == first_dept)]
            print(f"Используем Store={first_store}, Dept={first_dept}")
        else:
            df_filtered = df
        series, date_index, is_holiday = prepare_time_series(
            df_filtered, TARGET_COL, DATE_COL, group_cols=None
        )
    
    print(f"Длина ряда: {len(series)}")
    print(f"Период: {date_index.min()} - {date_index.max()}")
    
    # ========== ЭТАП 1: ИНЖИНИРИНГ ПРИЗНАКОВ ==========
    print("\n[ЭТАП 1] Инжиниринг признаков...")
    feature_engineer = FeatureEngineer()
    
    # Создаем признаки
    X, y_transformed, transform_info = feature_engineer.create_all_features(
        series, date_index, is_holiday, apply_log=True, apply_boxcox=False
    )
    
    print(f"Создано признаков: {X.shape[1]}")
    print(f"Трансформация: {transform_info}")
    
    # ========== ЭТАП 2: ВАЛИДАЦИЯ И РАЗБИЕНИЕ ==========
    print("\n[ЭТАП 2] Валидация и разбиение данных...")
    validator = DataValidator(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    
    # Хронологическое разбиение
    X_train, X_val, X_test = validator.chronological_split(X, date_index)
    y_train, y_val, y_test = validator.chronological_split(y_transformed, date_index)
    date_train, date_val, date_test = validator.chronological_split(date_index, date_index)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # TimeSeriesSplit для кросс-валидации
    tscv = validator.create_time_series_split(n_splits=5, max_train_size=365)
    
    # ========== ЭТАП 3: ПОДБОР ГИПЕРПАРАМЕТРОВ ==========
    print("\n[ЭТАП 3] Подбор гиперпараметров...")
    tuner = HyperparameterTuner(cv=tscv)
    
    # GridSearch для Ridge
    from sklearn.linear_model import Ridge
    ridge_param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    best_ridge = tuner.grid_search_linear(
        Ridge(), X_train, y_train, ridge_param_grid
    )
    print(f"Лучший Ridge: alpha={best_ridge.alpha}")
    
    # Optuna для LightGBM (если доступен)
    try:
        lgbm_params, best_lgbm = tuner.optuna_tune_lgbm(
            X_train, y_train, n_trials=50  # Уменьшено для скорости
        )
        print(f"Лучший LightGBM: {lgbm_params}")
    except Exception as e:
        print(f"Optuna tuning пропущен: {e}")
        best_lgbm = None
    
    # ========== ЭТАП 4: СТРАТЕГИИ ПРОГНОЗИРОВАНИЯ ==========
    print("\n[ЭТАП 4] Стратегии прогнозирования...")
    
    # Создаем базовую модель для стратегий
    from sklearn.linear_model import LinearRegression
    base_model = LinearRegression()
    
    strategies = {
        'Direct': DirectStrategy(base_model, horizon=HORIZON),
        'Recursive': RecursiveStrategy(base_model, horizon=HORIZON),
        'MultiOutput': MultiOutputStrategy(base_model, horizon=HORIZON),
        'DirRec': DirRecStrategy(base_model, horizon=HORIZON, window_size=3)
    }
    
    # Обучаем и сравниваем стратегии
    strategy_results = {}
    for name, strategy in strategies.items():
        print(f"  Тестируем стратегию: {name}")
        try:
            strategy.fit(X_train, y_train)
            preds = strategy.predict(X_val[:100])  # Ограничиваем для скорости
            strategy_results[name] = preds
        except Exception as e:
            print(f"    Ошибка: {e}")
    
    # ========== ЭТАП 5: ПОСТРОЕНИЕ МОДЕЛЕЙ ==========
    print("\n[ЭТАП 5] Построение моделей...")
    
    # Создаем все модели
    all_models = create_all_models()
    
    # Добавляем настроенные модели
    if best_ridge:
        all_models['Ridge_tuned'] = best_ridge
    if best_lgbm:
        all_models['LightGBM_tuned'] = best_lgbm
    
    # Обучаем модели
    trainer = ModelTrainer()
    for name, model in all_models.items():
        trainer.add_model(name, model)
    
    train_start = time.time()
    trained_models = trainer.train_all(X_train, y_train)
    train_time = time.time() - train_start
    
    print(f"Обучено моделей: {len(trained_models)}")
    print(f"Время обучения: {train_time:.2f} сек")
    
    # Бейзлайны
    baseline_preds = {}
    baseline_preds['Naive'] = BaselineModels.naive_forecast(y_train, HORIZON)
    baseline_preds['SeasonalNaive'] = BaselineModels.seasonal_naive_forecast(y_train, 7, HORIZON)
    baseline_preds['MovingAverage'] = BaselineModels.moving_average_forecast(y_train, 7, HORIZON)
    baseline_preds['LinearTrend'] = BaselineModels.linear_trend_forecast(y_train, HORIZON)
    
    # Предсказания на валидации (берем первые HORIZON точек для одношагового прогноза)
    # Для многошагового прогноза нужно использовать стратегии из этапа 4
    predictions_val = trainer.predict_all(X_val[:min(len(X_val), 100)])  # Ограничиваем для скорости
    
    # ========== ЭТАП 6: ДИАГНОСТИКА МОДЕЛЕЙ ==========
    print("\n[ЭТАП 6] Диагностика моделей...")
    
    diagnostics = ModelDiagnostics()
    
    # Диагностика для топ-3 моделей
    top_models = list(predictions_val.keys())[:3]
    
    for model_name in top_models:
        if model_name in predictions_val:
            y_pred = predictions_val[model_name]
            if len(y_pred) > 0:
                y_true_slice = y_val.iloc[:len(y_pred)] if isinstance(y_val, pd.Series) else y_val[:len(y_pred)]
                residuals = diagnostics.calculate_residuals(y_true_slice, y_pred, model_name)
                
                # ACF остатков
                try:
                    fig = diagnostics.plot_acf(residuals, model_name)
                    fig.savefig(f'{RESULTS_DIR}/acf_{model_name}.png', dpi=150, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"  Ошибка ACF для {model_name}: {e}")
                
                # Feature importance
                if model_name in trained_models:
                    model = trained_models[model_name]
                    importance = diagnostics.get_feature_importance(model, X_train.columns.tolist(), model_name)
                    if importance is not None:
                        fig = diagnostics.plot_feature_importance(importance, model_name)
                        fig.savefig(f'{RESULTS_DIR}/feature_importance_{model_name}.png', dpi=150, bbox_inches='tight')
                        plt.close(fig)
    
    # ========== ЭТАП 7: ОЦЕНКА КАЧЕСТВА ==========
    print("\n[ЭТАП 7] Оценка качества...")
    
    # Для MASE используем исходные данные (до трансформации), если они были трансформированы
    # Проверяем, была ли применена трансформация
    if transform_info.get('log', False):
        # Обратное преобразование для расчета MASE
        y_train_for_mase = feature_engineer.inverse_log_transform(y_train)
        print(f"Используем исходные данные (до логарифмирования) для MASE")
    else:
        y_train_for_mase = y_train
    
    evaluator = ModelEvaluator(y_train=y_train_for_mase, seasonality=7)
    
    # Оцениваем все модели
    all_predictions = {}
    for name, pred in predictions_val.items():
        if len(pred) > 0:
            y_true_slice = y_val.iloc[:len(pred)] if isinstance(y_val, pd.Series) else y_val[:len(pred)]
            all_predictions[name] = pred
    
    # Добавляем бейзлайны (для одношагового прогноза берем первое значение)
    for name, baseline_pred in baseline_preds.items():
        if len(baseline_pred) > 0:
            # Для одношагового сравнения берем первое значение бейзлайна
            all_predictions[name] = np.full(len(y_val[:min(len(y_val), 100)]), baseline_pred[0])
    
    # Метрики (берем соответствующий срез y_val)
    max_len = max([len(pred) for pred in all_predictions.values()] + [len(y_val)])
    y_val_slice = y_val.iloc[:min(max_len, len(y_val))] if isinstance(y_val, pd.Series) else y_val[:min(max_len, len(y_val))]
    metrics_df = evaluator.evaluate_all_models(y_val_slice, all_predictions)
    print("\nМетрики моделей:")
    print(metrics_df.to_string())
    
    # Сохраняем метрики
    metrics_df.to_csv(f'{RESULTS_DIR}/metrics.csv', index=False)
    metrics_df.to_json(f'{RESULTS_DIR}/metrics.json', orient='records', indent=2)
    
    # Diebold-Mariano тест
    if len(all_predictions) >= 2:
        model_names = list(all_predictions.keys())
        # Выравниваем длины для DM теста
        min_len = min([len(all_predictions[k]) for k in model_names[:5]] + [len(y_val_slice)])
        dm_predictions = {k: all_predictions[k][:min_len] for k in model_names[:5]}
        dm_results = evaluator.compare_models_dm(
            y_val_slice[:min_len], 
            dm_predictions
        )
        dm_results.to_csv(f'{RESULTS_DIR}/dm_test.csv')
        print("\nDiebold-Mariano тест:")
        print(dm_results.to_string())
    
    # ========== ЭТАП 7.5: AUTOGLUON ==========
    autogluon_predictions = {}
    autogluon_training_times = {}
    autogluon_leaderboards = {}
    
    if AUTOGLUON_AVAILABLE:
        print("\n" + "=" * 80)
        print("[ЭТАП 7.5] AUTOGLUON TIMESERIES")
        print("=" * 80)
        
        try:
            # Создаем wrapper
            ag_wrapper = AutoGluonWrapper(
                prediction_length=HORIZON,
                eval_metric="MAE",
                freq="W"
            )
            
            # Подготавливаем данные для AutoGluon
            # Объединяем train и val для обучения, тестируем на test
            train_val_series = pd.concat([y_train, y_val])
            train_val_dates = pd.concat([
                pd.Series(date_train),
                pd.Series(date_val)
            ])
            
            ag_train_data = ag_wrapper.prepare_data(
                train_val_series,
                pd.DatetimeIndex(train_val_dates)
            )
            
            # Обучаем с несколькими пресетами
            presets_to_test = ["medium_quality", "high_quality"]  # best_quality займет очень много времени
            
            # Ограничиваем время на каждый пресет (в секундах)
            time_limit_per_preset = 300  # 5 минут на пресет
            
            ag_wrapper.fit_multiple_presets(
                ag_train_data,
                presets=presets_to_test,
                time_limit_per_preset=time_limit_per_preset
            )
            
            # Получаем прогнозы для каждого пресета
            for preset in presets_to_test:
                if preset in ag_wrapper.predictors:
                    try:
                        predictions = ag_wrapper.predict(preset=preset, quantile_levels=[0.1, 0.9])
                        
                        # Извлекаем mean predictions
                        if isinstance(predictions, pd.DataFrame):
                            if 'mean' in predictions.columns:
                                pred_values = predictions['mean'].values
                            else:
                                pred_values = predictions.values.flatten()
                        else:
                            pred_values = predictions
                        
                        # Повторяем прогноз для всей длины test (для честного сравнения)
                        # В реальности AutoGluon дает h-шаговый прогноз
                        test_len = len(y_test)
                        if len(pred_values) < test_len:
                            # Расширяем прогноз повторением последнего значения
                            pred_extended = np.concatenate([
                                pred_values,
                                np.full(test_len - len(pred_values), pred_values[-1])
                            ])
                        else:
                            pred_extended = pred_values[:test_len]
                        
                        autogluon_predictions[f'AutoGluon_{preset}'] = pred_extended
                        autogluon_training_times[preset] = ag_wrapper.training_times[preset]
                        
                        # Получаем leaderboard
                        leaderboard = ag_wrapper.get_leaderboard(preset)
                        if leaderboard is not None:
                            autogluon_leaderboards[preset] = leaderboard
                            
                            print(f"\n📊 Leaderboard для {preset}:")
                            print(leaderboard[['model', 'score_val']].head(5).to_string())
                            
                            # Сохраняем leaderboard
                            leaderboard.to_csv(f'{RESULTS_DIR}/autogluon_leaderboard_{preset}.csv', index=False)
                    
                    except Exception as e:
                        print(f"❌ Ошибка при прогнозе для {preset}: {e}")
            
            # Добавляем AutoGluon прогнозы в общий пул для оценки
            for name, pred in autogluon_predictions.items():
                all_predictions[name] = pred
            
            # Backtesting валидация
            print("\n" + "-" * 80)
            print("BACKTESTING ВАЛИДАЦИЯ AUTOGLUON")
            print("-" * 80)
            
            for preset in presets_to_test:
                if preset in ag_wrapper.predictors:
                    try:
                        backtesting_results = ag_wrapper.backtesting(
                            ag_train_data,
                            num_windows=3,
                            preset=preset
                        )
                        
                        print(f"\n✅ Backtesting для {preset} завершен")
                        backtesting_results.to_csv(f'{RESULTS_DIR}/autogluon_backtesting_{preset}.csv', index=False)
                    
                    except Exception as e:
                        print(f"⚠️ Backtesting для {preset} пропущен: {e}")
            
            # Пересчитываем метрики с AutoGluon моделями
            print("\n" + "-" * 80)
            print("ОЦЕНКА КАЧЕСТВА С AUTOGLUON")
            print("-" * 80)
            
            # Используем test set для финальной оценки
            y_test_slice = y_test.iloc[:min(len(y_test), 100)] if isinstance(y_test, pd.Series) else y_test[:min(len(y_test), 100)]
            
            # Обрезаем все прогнозы до одинаковой длины
            for name in list(all_predictions.keys()):
                if len(all_predictions[name]) > len(y_test_slice):
                    all_predictions[name] = all_predictions[name][:len(y_test_slice)]
            
            metrics_df_with_ag = evaluator.evaluate_all_models(y_test_slice, all_predictions)
            
            print("\nМетрики моделей (включая AutoGluon):")
            # Показываем только топ-10
            print(metrics_df_with_ag.sort_values('MASE').head(10).to_string())
            
            # Сохраняем обновленные метрики
            metrics_df_with_ag.to_csv(f'{RESULTS_DIR}/metrics_with_autogluon.csv', index=False)
            metrics_df_with_ag.to_json(f'{RESULTS_DIR}/metrics_with_autogluon.json', orient='records', indent=2)
            
            # Сравнение AutoGluon vs Custom
            print("\n" + "=" * 80)
            print("СРАВНЕНИЕ AUTOGLUON VS КАСТОМНЫЕ МОДЕЛИ")
            print("=" * 80)
            
            # Готовим данные для сравнения
            custom_predictions_for_comparison = {k: v for k, v in all_predictions.items() 
                                                 if not k.startswith('AutoGluon_')}
            
            comparison_df = compare_autogluon_vs_custom(
                autogluon_preds=autogluon_predictions,
                custom_preds=custom_predictions_for_comparison,
                y_true=y_test_slice,
                autogluon_time=autogluon_training_times,
                custom_time=train_time
            )
            
            print("\nСравнительная таблица:")
            print(comparison_df.to_string())
            comparison_df.to_csv(f'{RESULTS_DIR}/autogluon_vs_custom.csv', index=False)
            
            # Создаем рекомендации
            recommendations = create_autogluon_recommendations(comparison_df)
            
            print("\n" + "=" * 80)
            print("РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ")
            print("=" * 80)
            
            print("\n📌 Краткая сводка:")
            for key, value in recommendations['summary'].items():
                print(f"  {key}: {value}")
            
            print("\n✅ Использовать AutoGluon когда:")
            for rec in recommendations['use_autogluon_when']:
                print(f"  - {rec}")
            
            print("\n✅ Использовать кастомные модели когда:")
            for rec in recommendations['use_custom_when']:
                print(f"  - {rec}")
            
            print(f"\n🚀 Стратегия для продакшена: {recommendations['production_strategy'].get('approach', 'N/A')}")
            print(f"   {recommendations['production_strategy'].get('description', '')}")
            
            # Сохраняем рекомендации
            with open(f'{RESULTS_DIR}/autogluon_recommendations.json', 'w', encoding='utf-8') as f:
                json.dump(recommendations, f, indent=2, ensure_ascii=False)
            
            # Сравнение пресетов AutoGluon
            preset_comparison = ag_wrapper.compare_presets()
            if preset_comparison is not None:
                print("\n📊 Сравнение пресетов AutoGluon:")
                print(preset_comparison.to_string())
                preset_comparison.to_csv(f'{RESULTS_DIR}/autogluon_presets_comparison.csv', index=False)
            
            # Обновляем metrics_df для дальнейшего использования
            metrics_df = metrics_df_with_ag
            
        except Exception as e:
            print(f"\n❌ ОШИБКА в AutoGluon: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️ AutoGluon не установлен, пропускаем этап")
        print("   Для установки: pip install autogluon.timeseries")
    
    # ========== ЭТАП 8: ПРОДВИНУТЫЕ ТЕХНИКИ ==========
    print("\n[ЭТАП 8] Продвинутые техники...")
    
    advanced = AdvancedTechniques()
    
    # Ансамблирование
    if len(all_predictions) >= 2:
        mase_scores = dict(zip(metrics_df['model'], metrics_df['MASE']))
        ensemble_pred = advanced.create_ensemble(all_predictions, 'weighted_mase', mase_scores)
        print("Создан взвешенный ансамбль")
    
    # Альтернативные модели (если AutoGluon недоступен)
    if ALTERNATIVE_MODELS_AVAILABLE:
        print("\n[ДОПОЛНИТЕЛЬНО] Тестирование альтернативных моделей...")
        alt_predictions = {}
        try:
            alt_models = AlternativeTimeSeriesModels()
            
            # StatsForecast
            if 'statsforecast' in alt_models.available_libs:
                try:
                    train_df = alt_models.prepare_data_for_statsforecast(y_train)
                    alt_predictions = alt_models.fit_statsforecast_models(train_df, horizon=HORIZON)
                    for name, pred in alt_predictions.items():
                        if len(pred) > 0:
                            all_predictions[name] = np.full(len(y_val_slice), pred[0] if len(pred) > 0 else y_train.iloc[-1])
                    print(f"  Добавлено моделей StatsForecast: {len(alt_predictions)}")
                except Exception as e:
                    print(f"  Ошибка StatsForecast: {e}")
            
            # Обновляем метрики с альтернативными моделями
            if alt_predictions:
                metrics_df_updated = evaluator.evaluate_all_models(y_val_slice, all_predictions)
                metrics_df = metrics_df_updated
                print(f"  Всего моделей после добавления альтернативных: {len(metrics_df)}")
        except Exception as e:
            print(f"Ошибка при работе с альтернативными моделями: {e}")
    
    # ========== ЭТАП 9: ИТОГОВЫЙ АНАЛИЗ ==========
    print("\n[ЭТАП 9] Итоговый анализ...")
    
    # Ранжирование моделей
    if 'MASE' in metrics_df.columns:
        metrics_df_sorted = metrics_df.sort_values('MASE')
        print("\nРанжирование по MASE:")
        print(metrics_df_sorted[['model', 'MASE', 'MAE', 'RMSE']].head(10).to_string())
        
        # Визуализация топ-3 моделей
        top_3_models = metrics_df_sorted.head(3)['model'].tolist()
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        for idx, model_name in enumerate(top_3_models):
            if model_name in all_predictions:
                y_pred = all_predictions[model_name]
                y_true_slice = y_val.iloc[:len(y_pred)] if isinstance(y_val, pd.Series) else y_val[:len(y_pred)]
                
                # Выравниваем длины
                min_len = min(len(y_pred), len(y_true_slice))
                y_pred_aligned = y_pred[:min_len]
                y_true_aligned = y_true_slice.iloc[:min_len] if isinstance(y_true_slice, pd.Series) else y_true_slice[:min_len]
                
                axes[idx].plot(y_true_aligned.values if isinstance(y_true_aligned, pd.Series) else y_true_aligned, 
                              label='Факт', alpha=0.7)
                axes[idx].plot(y_pred_aligned, label='Прогноз', alpha=0.7)
                mase_val = metrics_df_sorted[metrics_df_sorted["model"]==model_name]["MASE"].values[0] if len(metrics_df_sorted[metrics_df_sorted["model"]==model_name]) > 0 else 0
                axes[idx].set_title(f'{model_name} (MASE: {mase_val:.4f})')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/top3_models.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Сохраняем итоговый отчет
    summary = {
        'total_models': len(trained_models),
        'best_model': metrics_df_sorted.iloc[0]['model'] if 'MASE' in metrics_df.columns else None,
        'best_mase': metrics_df_sorted.iloc[0]['MASE'] if 'MASE' in metrics_df.columns else None,
        'train_time': train_time,
        'n_features': X.shape[1],
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test)
    }
    
    with open(f'{RESULTS_DIR}/summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("ПАЙПЛАЙН ЗАВЕРШЕН")
    print("=" * 80)
    print(f"\nРезультаты сохранены в директорию: {RESULTS_DIR}/")
    print(f"Лучшая модель: {summary['best_model']}")
    print(f"Лучший MASE: {summary['best_mase']:.6f}")


if __name__ == '__main__':
    main()

