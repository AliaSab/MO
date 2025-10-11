#!/usr/bin/env python3
"""
Этап 3: Описательный статистический анализ и визуализация
Модуль для анализа временных рядов недвижимости
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

class DescriptiveAnalyzer:
    """Класс для описательного статистического анализа"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_real_estate_data(self, df, target_column='MA', date_column='saledate'):
        """
        Полный описательный анализ данных о недвижимости
        
        Args:
            df: DataFrame с данными
            target_column: название целевой переменной
            date_column: название колонки с датами
            
        Returns:
            Словарь с результатами анализа
        """
        logger.info("Начинаем описательный анализ данных недвижимости")
        
        results = {}
        
        # Базовая информация
        results['basic_info'] = self._get_basic_info(df)
        
        # Дескриптивная статистика
        results['descriptive_stats'] = self._calculate_descriptive_stats(df)
        
        # Анализ по группам
        results['group_analysis'] = self._analyze_by_groups(df, target_column)
        
        # Корреляционный анализ
        results['correlation_analysis'] = self._correlation_analysis(df)
        
        # Временной анализ
        if date_column in df.columns:
            results['temporal_analysis'] = self._temporal_analysis(df, date_column, target_column)
        
        self.results = results
        logger.info("Описательный анализ завершен")
        return results
    
    def _get_basic_info(self, df):
        """Получение базовой информации о данных"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        return info
    
    def _calculate_descriptive_stats(self, df):
        """Расчет дескриптивной статистики"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {}
        
        # Основная статистика
        basic_stats = df[numeric_cols].describe()
        
        # Дополнительные метрики
        additional_stats = pd.DataFrame({
            'skewness': df[numeric_cols].skew(),
            'kurtosis': df[numeric_cols].kurtosis(),
            'median': df[numeric_cols].median(),
            'mode': df[numeric_cols].mode().iloc[0] if len(df[numeric_cols].mode()) > 0 else None,
            'range': df[numeric_cols].max() - df[numeric_cols].min(),
            'iqr': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
        })
        
        return {
            'basic': basic_stats,
            'additional': additional_stats
        }
    
    def _analyze_by_groups(self, df, target_column):
        """Анализ по группам (тип недвижимости, количество спален)"""
        group_analysis = {}
        
        # Анализ по типу недвижимости
        if 'type' in df.columns:
            type_stats = df.groupby('type')[target_column].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            group_analysis['by_type'] = type_stats
        
        # Анализ по количеству спален
        if 'bedrooms' in df.columns:
            bedroom_stats = df.groupby('bedrooms')[target_column].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            group_analysis['by_bedrooms'] = bedroom_stats
        
        # Комбинированный анализ
        if 'type' in df.columns and 'bedrooms' in df.columns:
            combined_stats = df.groupby(['type', 'bedrooms'])[target_column].agg([
                'count', 'mean', 'median', 'std'
            ]).round(2)
            group_analysis['combined'] = combined_stats
        
        return group_analysis
    
    def _correlation_analysis(self, df):
        """Корреляционный анализ"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        # Корреляционная матрица
        corr_matrix = df[numeric_cols].corr()
        
        # Находим сильные корреляции
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Сильная корреляция
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_correlations
        }
    
    def _temporal_analysis(self, df, date_column, target_column):
        """Временной анализ"""
        temporal_info = {}
        
        # Убеждаемся, что колонка даты имеет правильный тип
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], utc=True)
        
        # Временной диапазон
        temporal_info['date_range'] = {
            'start': df[date_column].min(),
            'end': df[date_column].max(),
            'duration_days': (df[date_column].max() - df[date_column].min()).days
        }
        
        # Анализ по годам
        df['year'] = df[date_column].dt.year
        yearly_stats = df.groupby('year')[target_column].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2)
        temporal_info['yearly_stats'] = yearly_stats
        
        # Анализ по кварталам
        df['quarter'] = df[date_column].dt.quarter
        quarterly_stats = df.groupby('quarter')[target_column].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2)
        temporal_info['quarterly_stats'] = quarterly_stats
        
        return temporal_info
    
    def create_visualizations(self, df, target_column='MA', date_column='saledate'):
        """Создание визуализаций"""
        visualizations = {}
        
        # График временного ряда
        if date_column in df.columns:
            fig_time = px.line(df, x=date_column, y=target_column, 
                             title=f'Временной ряд: {target_column}')
            visualizations['time_series'] = fig_time
        
        # Распределение целевой переменной
        fig_dist = px.histogram(df, x=target_column, 
                               title=f'Распределение {target_column}')
        visualizations['distribution'] = fig_dist
        
        # Box plot по группам
        if 'type' in df.columns:
            fig_box = px.box(df, x='type', y=target_column,
                           title=f'{target_column} по типам недвижимости')
            visualizations['boxplot_by_type'] = fig_box
        
        if 'bedrooms' in df.columns:
            fig_box_bed = px.box(df, x='bedrooms', y=target_column,
                               title=f'{target_column} по количеству спален')
            visualizations['boxplot_by_bedrooms'] = fig_box_bed
        
        # Корреляционная матрица
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               title="Корреляционная матрица")
            visualizations['correlation_heatmap'] = fig_corr
        
        return visualizations
    
    def generate_summary_report(self):
        """Генерация сводного отчета"""
        if not self.results:
            return "Анализ не выполнен"
        
        report = []
        report.append("=== ОТЧЕТ ОПИСАТЕЛЬНОГО АНАЛИЗА ===\n")
        
        # Базовая информация
        if 'basic_info' in self.results:
            info = self.results['basic_info']
            report.append(f"Размер данных: {info['shape'][0]} строк, {info['shape'][1]} столбцов")
            report.append(f"Память: {info['memory_usage'] / 1024**2:.2f} MB")
            report.append(f"Пропуски: {sum(info['missing_values'].values())}")
        
        # Дескриптивная статистика
        if 'descriptive_stats' in self.results:
            report.append("\n=== ДЕСКРИПТИВНАЯ СТАТИСТИКА ===")
            stats = self.results['descriptive_stats']['basic']
            report.append(f"Среднее значение целевой переменной: {stats.iloc[1, 0]:.2f}")
            report.append(f"Медиана: {stats.iloc[5, 0]:.2f}")
            report.append(f"Стандартное отклонение: {stats.iloc[2, 0]:.2f}")
        
        # Анализ по группам
        if 'group_analysis' in self.results:
            report.append("\n=== АНАЛИЗ ПО ГРУППАМ ===")
            if 'by_type' in self.results['group_analysis']:
                report.append("Статистика по типам недвижимости:")
                type_stats = self.results['group_analysis']['by_type']
                for idx, row in type_stats.iterrows():
                    report.append(f"  {idx}: среднее = {row['mean']:.2f}, медиана = {row['median']:.2f}")
        
        # Корреляционный анализ
        if 'correlation_analysis' in self.results:
            report.append("\n=== КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ===")
            strong_corr = self.results['correlation_analysis']['strong_correlations']
            if strong_corr:
                report.append("Сильные корреляции:")
                for corr in strong_corr:
                    report.append(f"  {corr['var1']} - {corr['var2']}: {corr['correlation']:.3f}")
            else:
                report.append("Сильных корреляций не обнаружено")
        
        return "\n".join(report)

def main():
    """Пример использования модуля анализа"""
    # Загружаем обработанные данные
    try:
        df = pd.read_csv('processed_real_estate_data.csv')
    except FileNotFoundError:
        print("Файл processed_real_estate_data.csv не найден. Загружаем исходные данные...")
        df = pd.read_csv('ma_lga_12345.csv')
        df['saledate'] = pd.to_datetime(df['saledate'], format='%d/%m/%Y')
    
    print("Загружены данные:")
    print(f"Размер: {df.shape}")
    print(f"Колонки: {df.columns.tolist()}")
    
    # Создаем анализатор
    analyzer = DescriptiveAnalyzer()
    
    # Выполняем анализ
    results = analyzer.analyze_real_estate_data(df)
    
    # Генерируем отчет
    report = analyzer.generate_summary_report()
    print("\n" + report)
    
    # Создаем визуализации
    visualizations = analyzer.create_visualizations(df)
    
    print(f"\nСоздано {len(visualizations)} визуализаций:")
    for name in visualizations.keys():
        print(f"  - {name}")

if __name__ == "__main__":
    main()

