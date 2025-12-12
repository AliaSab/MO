"""
Главный файл для запуска всех этапов анализа временных рядов.

Этапы:
1. Декомпозиция временного ряда
2. Расширенный feature engineering
3. Стратегии многопшагового прогнозирования
4. Кросс-валидация для временных рядов
5. Приведение к стационарности и преобразования
6. Модели экспоненциального сглаживания
7. Веб-интерфейс для интерактивного прогнозирования
8. Сравнительный анализ и выводы
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, Optional
import pandas as pd

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорт модулей с обработкой ошибок
try:
    from time_series_decomposition import TimeSeriesDecomposition
    from feature_engineering import TimeSeriesFeatureEngineering
    from multi_step_forecasting import MultiStepForecasting
    from time_series_cv import TimeSeriesCrossValidation
    from stationarity_transformation import StationarityTransformation
    from exponential_smoothing import ExponentialSmoothingModels
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('time_series_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TimeSeriesAnalysisPipeline:
    """Класс для управления пайплайном анализа временных рядов."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.results = {}
        
    def run_step1_decomposition(self) -> Dict[str, Any]:
        """Этап 1: Декомпозиция временного ряда."""
        logger.info("=== ЭТАП 1: ДЕКОМПОЗИЦИЯ ВРЕМЕННОГО РЯДА ===")
        
        try:
            decomposer = TimeSeriesDecomposition()
            results = decomposer.run_comprehensive_decomposition(self.data_path)
            
            if results:
                logger.info("Этап 1 завершен успешно")
                return results
            else:
                logger.warning("Этап 1 завершен с пустыми результатами")
                return {}
                
        except Exception as e:
            logger.error(f"Ошибка в этапе 1: {e}")
            return {'error': str(e)}
    
    def run_step2_feature_engineering(self) -> Dict[str, Any]:
        """Этап 2: Расширенный feature engineering."""
        logger.info("=== ЭТАП 2: РАСШИРЕННЫЙ FEATURE ENGINEERING ===")
        
        try:
            fe = TimeSeriesFeatureEngineering()
            enhanced_df = fe.run_comprehensive_feature_engineering(self.data_path)
            
            if enhanced_df is not None and not enhanced_df.empty:
                results = {
                    'enhanced_dataset': enhanced_df,
                    'features_created': getattr(fe, 'features_created', []),
                    'total_features': len(getattr(fe, 'features_created', []))
                }
                logger.info("Этап 2 завершен успешно")
                return results
            else:
                logger.warning("Этап 2 завершен с пустыми результатами")
                return {}
                
        except Exception as e:
            logger.error(f"Ошибка в этапе 2: {e}")
            return {'error': str(e)}
    
    def run_step3_multi_step_forecasting(self) -> Dict[str, Any]:
        """Этап 3: Стратегии многопшагового прогнозирования."""
        logger.info("=== ЭТАП 3: МНОГОШАГОВОЕ ПРОГНОЗИРОВАНИЕ ===")
        
        try:
            forecaster = MultiStepForecasting()
            results = forecaster.run_comprehensive_analysis(self.data_path, horizon=7)
            
            if results:
                logger.info("Этап 3 завершен успешно")
                return results
            else:
                logger.warning("Этап 3 завершен с пустыми результатами")
                return {}
                
        except Exception as e:
            logger.error(f"Ошибка в этапе 3: {e}")
            return {'error': str(e)}
    
    def run_step4_time_series_cv(self) -> Dict[str, Any]:
        """Этап 4: Кросс-валидация для временных рядов."""
        logger.info("=== ЭТАП 4: КРОСС-ВАЛИДАЦИЯ ВРЕМЕННЫХ РЯДОВ ===")
        
        try:
            cv_analyzer = TimeSeriesCrossValidation()
            results = cv_analyzer.run_comprehensive_cv_analysis(self.data_path)
            
            if results:
                logger.info("Этап 4 завершен успешно")
                return results
            else:
                logger.warning("Этап 4 завершен с пустыми результатами")
                return {}
                
        except Exception as e:
            logger.error(f"Ошибка в этапе 4: {e}")
            return {'error': str(e)}
    
    def run_step5_stationarity_transformation(self) -> Dict[str, Any]:
        """Этап 5: Приведение к стационарности и преобразования."""
        logger.info("=== ЭТАП 5: СТАЦИОНАРНОСТЬ И ПРЕОБРАЗОВАНИЯ ===")
        
        try:
            stationarity_analyzer = StationarityTransformation()
            results = stationarity_analyzer.run_comprehensive_stationarity_analysis(self.data_path)
            
            if results:
                logger.info("Этап 5 завершен успешно")
                return results
            else:
                logger.warning("Этап 5 завершен с пустыми результатами")
                return {}
                
        except Exception as e:
            logger.error(f"Ошибка в этапе 5: {e}")
            return {'error': str(e)}
    
    def run_step6_exponential_smoothing(self) -> Dict[str, Any]:
        """Этап 6: Модели экспоненциального сглаживания."""
        logger.info("=== ЭТАП 6: ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ ===")
        
        try:
            es_analyzer = ExponentialSmoothingModels()
            results = es_analyzer.run_comprehensive_analysis(self.data_path, horizon=7)
            
            if results:
                logger.info("Этап 6 завершен успешно")
                return results
            else:
                logger.warning("Этап 6 завершен с пустыми результатами")
                return {}
                
        except Exception as e:
            logger.error(f"Ошибка в этапе 6: {e}")
            return {'error': str(e)}
    
    def run_step7_web_interface(self) -> Dict[str, Any]:
        """Этап 7: Веб-интерфейс."""
        logger.info("=== ЭТАП 7: ВЕБ-ИНТЕРФЕЙС ===")
        
        try:
            logger.info("Для запуска веб-интерфейса выполните команду:")
            logger.info("streamlit run web_interface.py")
            
            # Проверяем наличие файла веб-интерфейса
            web_interface_path = os.path.join(os.path.dirname(__file__), 'web_interface.py')
            if os.path.exists(web_interface_path):
                logger.info("Файл веб-интерфейса найден")
                return {
                    'web_interface_available': True,
                    'web_interface_path': web_interface_path,
                    'instructions': 'streamlit run web_interface.py'
                }
            else:
                logger.warning("Файл веб-интерфейса не найден")
                return {
                    'web_interface_available': False,
                    'error': 'Файл web_interface.py не найден'
                }
                
        except Exception as e:
            logger.error(f"Ошибка в этапе 7: {e}")
            return {'error': str(e)}
    
    def run_step8_comparative_analysis(self) -> Dict[str, Any]:
        """Этап 8: Сравнительный анализ и выводы."""
        logger.info("=== ЭТАП 8: СРАВНИТЕЛЬНЫЙ АНАЛИЗ ===")
        
        try:
            # Сбор результатов всех этапов
            comparative_results = {
                'decomposition_summary': self.results.get('step1', {}),
                'feature_engineering_summary': self.results.get('step2', {}),
                'multi_step_forecasting_summary': self.results.get('step3', {}),
                'cv_summary': self.results.get('step4', {}),
                'stationarity_summary': self.results.get('step5', {}),
                'exponential_smoothing_summary': self.results.get('step6', {}),
                'web_interface_summary': self.results.get('step7', {})
            }
            
            # Анализ лучших моделей
            best_models = self.analyze_best_models(comparative_results)
            
            # Практические рекомендации
            recommendations = self.generate_recommendations(comparative_results)
            
            # Рефлексия
            reflection = self.generate_reflection()
            
            final_analysis = {
                'comparative_results': comparative_results,
                'best_models': best_models,
                'recommendations': recommendations,
                'reflection': reflection
            }
            
            # Сохранение итогового анализа
            self.save_final_analysis(final_analysis)
            
            logger.info("Этап 8 завершен успешно")
            return final_analysis
            
        except Exception as e:
            logger.error(f"Ошибка в этапе 8: {e}")
            return {'error': str(e)}
    
    def analyze_best_models(self, comparative_results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ лучших моделей по различным критериям."""
        logger.info("Анализ лучших моделей...")
        
        try:
            best_models = {
                'decomposition': None,
                'forecasting_strategy': None,
                'exponential_smoothing': None,
                'overall_best': None
            }
            
            # Лучшая модель декомпозиции
            if 'decomposition_summary' in comparative_results:
                decomp_results = comparative_results['decomposition_summary']
                if 'best_model_selection' in decomp_results and 'error' not in decomp_results:
                    best_models['decomposition'] = decomp_results['best_model_selection'].get('best_model', 'Не определено')
            
            # Лучшая стратегия прогнозирования
            if 'multi_step_forecasting_summary' in comparative_results:
                forecast_results = comparative_results['multi_step_forecasting_summary']
                if 'results' in forecast_results and 'error' not in forecast_results:
                    results = forecast_results['results']
                    if results:
                        try:
                            best_strategy = min(results.keys(), key=lambda k: results[k].get('mae', float('inf')))
                            best_models['forecasting_strategy'] = best_strategy
                        except (ValueError, TypeError):
                            best_models['forecasting_strategy'] = 'Не определено'
            
            # Лучшая модель экспоненциального сглаживания
            if 'exponential_smoothing_summary' in comparative_results:
                es_results = comparative_results['exponential_smoothing_summary']
                if 'summary' in es_results and 'error' not in es_results:
                    summary_data = es_results['summary']
                    if summary_data:
                        try:
                            summary_df = pd.DataFrame(summary_data)
                            if not summary_df.empty and 'MAE' in summary_df.columns:
                                best_es_model = summary_df.loc[summary_df['MAE'].idxmin(), 'Модель']
                                best_models['exponential_smoothing'] = best_es_model
                        except (KeyError, ValueError, TypeError):
                            best_models['exponential_smoothing'] = 'Не определено'
            
            logger.info("Анализ лучших моделей завершен")
            return best_models
            
        except Exception as e:
            logger.error(f"Ошибка в анализе лучших моделей: {e}")
            return {
                'decomposition': 'Ошибка',
                'forecasting_strategy': 'Ошибка',
                'exponential_smoothing': 'Ошибка',
                'overall_best': 'Ошибка'
            }
    
    def generate_recommendations(self, comparative_results: Dict[str, Any]) -> Dict[str, str]:
        """Генерация практических рекомендаций."""
        logger.info("Генерация рекомендаций...")
        
        recommendations = {
            'data_preprocessing': 'Используйте преобразование Бокса–Кокса для стабилизации дисперсии',
            'decomposition': 'Применяйте аддитивную декомпозицию для большинства временных рядов',
            'forecasting': 'Используйте гибридную стратегию для многопшагового прогнозирования',
            'model_selection': 'Экспоненциальное сглаживание показывает хорошие результаты для краткосрочных прогнозов',
            'validation': 'Применяйте кросс-валидацию с расширяющимся окном для оценки качества моделей',
            'production': 'Регулярно переобучайте модели с новыми данными для поддержания актуальности'
        }
        
        return recommendations
    
    def generate_reflection(self) -> Dict[str, str]:
        """Генерация рефлексии по проекту."""
        logger.info("Генерация рефлексии...")
        
        reflection = {
            'challenges': [
                'Сложность выбора оптимальных параметров для различных моделей',
                'Необходимость баланса между точностью и интерпретируемостью',
                'Обработка пропущенных значений и выбросов в данных',
                'Выбор подходящих метрик для оценки качества моделей'
            ],
            'insights': [
                'Комбинирование различных подходов дает лучшие результаты',
                'Предобработка данных критически важна для качества прогнозов',
                'Временные признаки значительно улучшают качество моделей',
                'Кросс-валидация помогает избежать переобучения'
            ],
            'lessons_learned': [
                'Важность визуализации для понимания структуры данных',
                'Необходимость тестирования множественных гипотез',
                'Значение автоматизации процессов анализа',
                'Важность документирования всех этапов анализа'
            ]
        }
        
        return reflection
    
    def save_final_analysis(self, final_analysis: Dict[str, Any]) -> None:
        """Сохранение итогового анализа."""
        logger.info("Сохранение итогового анализа...")
        
        with open('final_time_series_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(final_analysis, f, ensure_ascii=False, indent=2, default=str)
        
        # Создание краткого отчета
        report = f"""
# ИТОГОВЫЙ ОТЧЕТ ПО АНАЛИЗУ ВРЕМЕННЫХ РЯДОВ

## Лучшие модели:
- Декомпозиция: {final_analysis['best_models']['decomposition']}
- Стратегия прогнозирования: {final_analysis['best_models']['forecasting_strategy']}
- Экспоненциальное сглаживание: {final_analysis['best_models']['exponential_smoothing']}

## Практические рекомендации:
{chr(10).join([f"- {k}: {v}" for k, v in final_analysis['recommendations'].items()])}

## Основные вызовы:
{chr(10).join([f"- {challenge}" for challenge in final_analysis['reflection']['challenges']])}

## Ключевые инсайты:
{chr(10).join([f"- {insight}" for insight in final_analysis['reflection']['insights']])}

## Уроки:
{chr(10).join([f"- {lesson}" for lesson in final_analysis['reflection']['lessons_learned']])}
"""
        
        with open('final_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("Итоговый анализ сохранен в:")
        logger.info("- final_time_series_analysis.json")
        logger.info("- final_report.md")
    
    def run_all_steps(self) -> Dict[str, Any]:
        """Запуск всех этапов анализа временных рядов."""
        logger.info("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА АНАЛИЗА ВРЕМЕННЫХ РЯДОВ")
        
        start_time = time.time()
        successful_steps = 0
        failed_steps = 0
        
        try:
            # Этап 1: Декомпозиция
            logger.info("Запуск этапа 1...")
            self.results['step1'] = self.run_step1_decomposition()
            if 'error' not in self.results['step1']:
                successful_steps += 1
            else:
                failed_steps += 1
            
            # Этап 2: Feature Engineering
            logger.info("Запуск этапа 2...")
            self.results['step2'] = self.run_step2_feature_engineering()
            if 'error' not in self.results['step2']:
                successful_steps += 1
            else:
                failed_steps += 1
            
            # Этап 3: Многопшаговое прогнозирование
            logger.info("Запуск этапа 3...")
            self.results['step3'] = self.run_step3_multi_step_forecasting()
            if 'error' not in self.results['step3']:
                successful_steps += 1
            else:
                failed_steps += 1
            
            # Этап 4: Кросс-валидация
            logger.info("Запуск этапа 4...")
            self.results['step4'] = self.run_step4_time_series_cv()
            if 'error' not in self.results['step4']:
                successful_steps += 1
            else:
                failed_steps += 1
            
            # Этап 5: Стационарность
            logger.info("Запуск этапа 5...")
            self.results['step5'] = self.run_step5_stationarity_transformation()
            if 'error' not in self.results['step5']:
                successful_steps += 1
            else:
                failed_steps += 1
            
            # Этап 6: Экспоненциальное сглаживание
            logger.info("Запуск этапа 6...")
            self.results['step6'] = self.run_step6_exponential_smoothing()
            if 'error' not in self.results['step6']:
                successful_steps += 1
            else:
                failed_steps += 1
            
            # Этап 7: Веб-интерфейс
            logger.info("Запуск этапа 7...")
            self.results['step7'] = self.run_step7_web_interface()
            if 'error' not in self.results['step7']:
                successful_steps += 1
            else:
                failed_steps += 1
            
            # Этап 8: Сравнительный анализ
            logger.info("Запуск этапа 8...")
            self.results['step8'] = self.run_step8_comparative_analysis()
            if 'error' not in self.results['step8']:
                successful_steps += 1
            else:
                failed_steps += 1
            
            total_time = time.time() - start_time
            
            # Создание итогового отчёта
            self.create_final_report(total_time, successful_steps, failed_steps)
            
            logger.info(f"ПАЙПЛАЙН ЗАВЕРШЕН ЗА {total_time:.2f} СЕКУНД")
            logger.info(f"Успешных этапов: {successful_steps}, неудачных: {failed_steps}")
            
        except Exception as e:
            logger.error(f"Критическая ошибка в пайплайне: {e}")
            total_time = time.time() - start_time
            self.create_final_report(total_time, successful_steps, failed_steps)
            raise
        
        return self.results
    
    def create_final_report(self, total_time: float, successful_steps: int = 0, failed_steps: int = 0) -> None:
        """Создание итогового отчёта."""
        logger.info("Создание итогового отчёта")
        
        try:
            report = {
                'pipeline_summary': {
                    'total_execution_time': total_time,
                    'successful_steps': successful_steps,
                    'failed_steps': failed_steps,
                    'total_steps': successful_steps + failed_steps,
                    'success_rate': successful_steps / (successful_steps + failed_steps) if (successful_steps + failed_steps) > 0 else 0,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'data_path': self.data_path
                },
                'step_summaries': {}
            }
            
            # Сводка по каждому этапу
            step_names = {
                'step1': 'Декомпозиция временного ряда',
                'step2': 'Расширенный feature engineering',
                'step3': 'Стратегии многопшагового прогнозирования',
                'step4': 'Кросс-валидация для временных рядов',
                'step5': 'Приведение к стационарности и преобразования',
                'step6': 'Модели экспоненциального сглаживания',
                'step7': 'Веб-интерфейс для интерактивного прогнозирования',
                'step8': 'Сравнительный анализ и выводы'
            }
            
            for step, results in self.results.items():
                has_error = 'error' in results if isinstance(results, dict) else False
                report['step_summaries'][step] = {
                    'name': step_names.get(step, step),
                    'status': 'failed' if has_error else 'completed',
                    'has_results': bool(results) and not has_error,
                    'error': results.get('error') if has_error else None
                }
            
            # Сохранение отчёта
            with open('pipeline_final_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info("Итоговый отчёт сохранен в pipeline_final_report.json")
            
        except Exception as e:
            logger.error(f"Ошибка создания итогового отчёта: {e}")


def main():
    """Основная функция."""
    logger.info("ЗАПУСК ПАЙПЛАЙНА АНАЛИЗА ВРЕМЕННЫХ РЯДОВ")
    
    # Путь к данным
    data_path = "../New_final.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"Файл данных не найден: {data_path}")
        logger.info("Убедитесь, что файл New_final.csv находится в родительской директории")
        return
    
    # Создание и запуск пайплайна
    pipeline = TimeSeriesAnalysisPipeline(data_path)
    
    try:
        results = pipeline.run_all_steps()
        
        print("\n" + "="*60)
        print("ПАЙПЛАЙН АНАЛИЗА ВРЕМЕННЫХ РЯДОВ ЗАВЕРШЕН")
        print("="*60)
        
        print("\nСозданные файлы:")
        print("• classical_vectorization_results.json - результаты декомпозиции")
        print("• enhanced_dataset.csv - расширенный датасет с признаками")
        print("• multi_step_forecasting_results.csv - результаты многопшагового прогнозирования")
        print("• time_series_cv_results.csv - результаты кросс-валидации")
        print("• stationarity_analysis_results.csv - результаты анализа стационарности")
        print("• exponential_smoothing_results.csv - результаты экспоненциального сглаживания")
        print("• final_time_series_analysis.json - итоговый анализ")
        print("• final_report.md - итоговый отчет")
        print("• pipeline_final_report.json - отчет по пайплайну")
        
        print("\nДля запуска веб-интерфейса выполните:")
        print("streamlit run web_interface.py")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения пайплайна: {e}")
        print(f"\nОшибка: {e}")
        print("Проверьте логи в файле time_series_analysis.log")


if __name__ == "__main__":
    main()

