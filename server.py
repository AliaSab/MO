import os
os.environ['TCL_LIBRARY'] = "C:/Program Files/Python313/tcl/tcl8.6"
os.environ['TK_LIBRARY'] = "C:/Program Files/Python313/tcl/tk8.6"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Добавляем логирование для отладки
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Статистические тесты и анализ
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pytz

# Настройка страницы
st.set_page_config(
    page_title="Анализ временных рядов недвижимости",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RealEstateTimeSeriesApp:
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.target_column = None
        self.date_column = None
        
    def load_data(self, uploaded_file=None, use_sample=False):
        """Загрузка данных"""
        if use_sample:
            # Используем встроенный пример данных ma_lga_12345.csv
            try:
                # Получаем абсолютный путь к файлу
                script_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(script_dir, 'ma_lga_12345.csv')
                
                if not os.path.exists(csv_path):
                    st.error(f"Файл ma_lga_12345.csv не найден по пути: {csv_path}")
                    st.error("Убедитесь, что файл находится в директории Time_series_Alya")
                    return False
                
                self.df = pd.read_csv(csv_path)
                logger.info(f"Загружен файл: {csv_path}, размер: {self.df.shape}")
                
                # Преобразуем дату в правильный формат
                self.df['saledate'] = pd.to_datetime(self.df['saledate'], format='%d/%m/%Y')
                self.date_column = 'saledate'
                self.target_column = 'MA'  # Median Auction как целевая переменная
                
                logger.info("Данные успешно загружены и настроены")
                return True
            except Exception as e:
                st.error(f"Ошибка при загрузке примера данных: {e}")
                return False
        elif uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    self.df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.parquet'):
                    self.df = pd.read_parquet(uploaded_file)
                else:
                    st.error("Поддерживаются только CSV и Parquet файлы")
                    return False
                
                # Автоматическое определение колонок
                date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    self.date_column = date_cols[0]
                    self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], utc=True)
                else:
                    st.warning("Не найдена колонка с датами. Выберите вручную в настройках.")
                
                return True
            except Exception as e:
                st.error(f"Ошибка при загрузке файла: {e}")
                return False
        return False
    
    def preprocess_data(self):
        """Предобработка данных (Этап 2)"""
        if self.df is None:
            return None
            
        df_clean = self.df.copy()
        
        # Этап 2.1: Приведение временных меток к единому формату
        if self.date_column:
            df_clean[self.date_column] = pd.to_datetime(df_clean[self.date_column], utc=True)
        
        # Этап 2.2: Удаление дубликатов по времени
        if self.date_column:
            df_clean = df_clean.drop_duplicates(subset=[self.date_column], keep='first')
        
        # Этап 2.3: Проверка монотонности временного ряда
        if self.date_column:
            df_clean = df_clean.sort_values(self.date_column)
        
        # Этап 2.4: Обработка пропусков
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            missing_pct = df_clean[col].isnull().sum() / len(df_clean) * 100
            
            if missing_pct < 5:
                # Если пропусков меньше 5%, удаляем
                df_clean = df_clean.dropna(subset=[col])
            else:
                # Иначе интерполируем
                df_clean[col] = df_clean[col].interpolate(method='linear')
        
        # Этап 2.5: Обнаружение и обработка выбросов
        for col in numeric_cols:
            if col in df_clean.columns:
                # Метод IQR
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Заменяем выбросы на граничные значения
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Этап 2.6: Ресемплирование до единой частоты (квартально)
        if self.date_column and self.target_column:
            df_clean = df_clean.set_index(self.date_column)
            df_clean = df_clean.resample('QE').mean(numeric_only=True).dropna()
            df_clean = df_clean.reset_index()
        
        self.processed_df = df_clean
        return df_clean
    
    def descriptive_analysis(self):
        """Описательный статистический анализ (Этап 3)"""
        if self.processed_df is None:
            return None
        
        df = self.processed_df
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Дескриптивная статистика
        desc_stats = df[numeric_cols].describe()
        
        # Дополнительные статистики
        additional_stats = pd.DataFrame({
            'Асимметрия': df[numeric_cols].skew(),
            'Эксцесс': df[numeric_cols].kurtosis(),
            'Медиана': df[numeric_cols].median()
        })
        
        return desc_stats, additional_stats
    
    def stationarity_tests(self):
        """Проверка на стационарность (Этап 4)"""
        if self.processed_df is None or self.target_column is None:
            return None
        
        df = self.processed_df
        target_data = df[self.target_column].dropna()
        
        # Тест Дики-Фуллера
        adf_result = adfuller(target_data)
        
        # Тест KPSS
        try:
            kpss_result = kpss(target_data, regression='c')
        except:
            kpss_result = (np.nan, np.nan, np.nan, {'10%': np.nan, '5%': np.nan, '2.5%': np.nan, '1%': np.nan})
        
        # Скользящие статистики
        rolling_mean = target_data.rolling(window=4).mean()  # Квартальное окно
        rolling_std = target_data.rolling(window=4).std()
        
        return {
            'adf': adf_result,
            'kpss': kpss_result,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std
        }
    
    def create_lag_features(self):
        """Создание лаговых признаков (Этап 5)"""
        if self.processed_df is None or self.target_column is None:
            return None
        
        df = self.processed_df.copy()
        
        # Создаем лаги целевой переменной
        df[f'{self.target_column}_lag_1'] = df[self.target_column].shift(1)
        df[f'{self.target_column}_lag_4'] = df[self.target_column].shift(4)  # Годовой лаг для квартальных данных
        df[f'{self.target_column}_lag_8'] = df[self.target_column].shift(8)  # Двухлетний лаг
        
        # Скользящие статистики
        df[f'{self.target_column}_rolling_mean_4'] = df[self.target_column].rolling(window=4).mean()
        df[f'{self.target_column}_rolling_std_4'] = df[self.target_column].rolling(window=4).std()
        
        # Удаляем строки с NaN
        df = df.dropna()
        
        return df
    
    def acf_pacf_analysis(self):
        """Анализ автокорреляции (Этап 6)"""
        if self.processed_df is None or self.target_column is None:
            return None
        
        df = self.processed_df
        target_data = df[self.target_column].dropna()
        
        # Вычисляем ACF и PACF
        acf_values = acf(target_data, nlags=20, fft=False)
        pacf_values = pacf(target_data, nlags=20)
        
        return acf_values, pacf_values
    
    def decompose_time_series(self, model='additive'):
        """Декомпозиция временного ряда (Этап 7)"""
        if self.processed_df is None or self.target_column is None:
            return None
        
        df = self.processed_df
        target_data = df[self.target_column].dropna()
        
        # Устанавливаем индекс времени
        if self.date_column:
            target_series = pd.Series(target_data.values, index=df[self.date_column].iloc[:len(target_data)])
        else:
            target_series = target_data
        
        # Декомпозиция
        decomposition = seasonal_decompose(target_series, model=model, period=4)  # Квартальная сезонность
        
        return decomposition
    
    def generate_report(self):
        """Генерация HTML отчета"""
        if self.processed_df is None:
            return None
        
        html_content = f"""
        <html>
        <head>
            <title>Отчет анализа временного ряда недвижимости</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <h1>Отчет анализа временного ряда недвижимости</h1>
            <p>Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Общая информация</h2>
            <div class="metric">
                <strong>Размер датасета:</strong> {self.processed_df.shape[0]} строк, {self.processed_df.shape[1]} столбцов
            </div>
            <div class="metric">
                <strong>Целевая переменная:</strong> {self.target_column}
            </div>
            <div class="metric">
                <strong>Временной столбец:</strong> {self.date_column}
            </div>
            
            <h2>Статистические тесты</h2>
            <p>Результаты тестов на стационарность будут добавлены в полной версии.</p>
            
            <h2>Рекомендации</h2>
            <ul>
                <li>Данные готовы для дальнейшего анализа</li>
                <li>Рекомендуется проверить стационарность перед моделированием</li>
                <li>Рассмотрите возможность создания дополнительных признаков</li>
            </ul>
        </body>
        </html>
        """
        
        return html_content

def main():
    """Основная функция приложения"""
    st.title("🏠 Анализ временных рядов недвижимости")
    st.markdown("Интерактивный анализ данных о ценах на недвижимость в Австралии")
    
    # Создаем экземпляр приложения
    app = RealEstateTimeSeriesApp()
    
    # Боковая панель для настроек
    st.sidebar.header("⚙️ Настройки")
    
    # Загрузка данных
    st.sidebar.subheader("📁 Загрузка данных")
    use_sample = st.sidebar.checkbox("Использовать пример данных (ma_lga_12345.csv)", value=True)
    
    if not use_sample:
        uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл", type=['csv'])
        if uploaded_file:
            if app.load_data(uploaded_file=uploaded_file):
                st.success("✅ Данные успешно загружены!")
    else:
        if app.load_data(use_sample=True):
            st.success("✅ Пример данных загружен!")
    
    if app.df is not None:
        # Настройки анализа
        st.sidebar.subheader("🔧 Настройки анализа")
        
        # Выбор колонок
        if app.date_column is None:
            app.date_column = st.sidebar.selectbox("Выберите колонку с датами", app.df.columns)
        
        if app.target_column is None:
            numeric_cols = app.df.select_dtypes(include=[np.number]).columns.tolist()
            app.target_column = st.sidebar.selectbox("Выберите целевую переменную", numeric_cols)
        
        # Кнопка предобработки
        if st.sidebar.button("🔄 Предобработать данные"):
            with st.spinner("Обрабатываем данные..."):
                app.preprocess_data()
                st.success("✅ Данные предобработаны!")
        
        # Основные вкладки
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Обзор", "🔧 Предобработка", "📈 Описательная статистика", 
            "📉 Стационарность", "⏰ Лаги", "🔄 ACF/PACF", "🧩 Декомпозиция", "📄 Отчет"
        ])
        
        with tab1:
            st.header("📊 Обзор данных")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Первые 10 строк")
                st.dataframe(app.df.head(10))
            
            with col2:
                st.subheader("Информация о данных")
                st.write(f"**Размер:** {app.df.shape[0]} строк, {app.df.shape[1]} столбцов")
                st.write(f"**Дата:** {app.date_column}")
                st.write(f"**Целевая переменная:** {app.target_column}")
                
                # Основная статистика
                if app.target_column:
                    st.write("**Основная статистика целевой переменной:**")
                    st.write(app.df[app.target_column].describe())
        
        with tab2:
            st.header("🔧 Предобработка данных")
            
            if st.button("Выполнить предобработку"):
                with st.spinner("Выполняем предобработку..."):
                    processed_df = app.preprocess_data()
                    
                    if processed_df is not None:
                        st.success("✅ Предобработка завершена!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("До предобработки")
                            st.write(f"Размер: {app.df.shape}")
                            st.write("Пропуски:")
                            st.write(app.df.isnull().sum())
                        
                        with col2:
                            st.subheader("После предобработки")
                            st.write(f"Размер: {processed_df.shape}")
                            st.write("Пропуски:")
                            st.write(processed_df.isnull().sum())
                        
                        # График временного ряда
                        if app.date_column and app.target_column:
                            fig = px.line(processed_df, x=app.date_column, y=app.target_column, 
                                        title=f"Временной ряд: {app.target_column}")
                            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("📈 Описательная статистика")
            
            if app.processed_df is not None:
                desc_stats, additional_stats = app.descriptive_analysis()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Основная статистика")
                    st.dataframe(desc_stats)
                
                with col2:
                    st.subheader("Дополнительные метрики")
                    st.dataframe(additional_stats)
                
                # Корреляционная матрица
                numeric_cols = app.processed_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = app.processed_df[numeric_cols].corr()
                    
                    fig = px.imshow(corr_matrix, 
                                  text_auto=True, 
                                  aspect="auto",
                                  title="Корреляционная матрица")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Сначала выполните предобработку данных")
        
        with tab4:
            st.header("📉 Проверка стационарности")
            
            if app.processed_df is not None:
                stationarity_results = app.stationarity_tests()
                
                if stationarity_results:
                    # Результаты тестов
                    adf_result = stationarity_results['adf']
                    kpss_result = stationarity_results['kpss']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Тест Дики-Фуллера (ADF)")
                        st.write(f"**Статистика:** {adf_result[0]:.4f}")
                        st.write(f"**p-value:** {adf_result[1]:.4f}")
                        st.write(f"**Критические значения:**")
                        for key, value in adf_result[4].items():
                            st.write(f"  {key}: {value:.4f}")
                        
                        if adf_result[1] < 0.05:
                            st.success("✅ Ряд стационарен (p < 0.05)")
                        else:
                            st.warning("⚠️ Ряд нестационарен (p >= 0.05)")
                    
                    with col2:
                        st.subheader("Тест KPSS")
                        st.write(f"**Статистика:** {kpss_result[0]:.4f}")
                        st.write(f"**p-value:** {kpss_result[1]:.4f}")
                        
                        if kpss_result[1] > 0.05:
                            st.success("✅ Ряд стационарен (p > 0.05)")
                        else:
                            st.warning("⚠️ Ряд нестационарен (p <= 0.05)")
                    
                    # График скользящих статистик
                    rolling_mean = stationarity_results['rolling_mean']
                    rolling_std = stationarity_results['rolling_std']
                    
                    fig = make_subplots(rows=2, cols=1, 
                                      subplot_titles=('Скользящее среднее', 'Скользящее стандартное отклонение'))
                    
                    fig.add_trace(go.Scatter(y=rolling_mean, name='Скользящее среднее'), row=1, col=1)
                    fig.add_trace(go.Scatter(y=rolling_std, name='Скользящее стд. отклонение'), row=2, col=1)
                    
                    fig.update_layout(height=600, title_text="Анализ стационарности")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Сначала выполните предобработку данных")
        
        with tab5:
            st.header("⏰ Лаговые признаки")
            
            if app.processed_df is not None:
                lag_df = app.create_lag_features()
                
                if lag_df is not None:
                    st.subheader("Данные с лаговыми признаками")
                    st.dataframe(lag_df.head(10))
                    
                    # Корреляция лагов с целевой переменной
                    lag_cols = [col for col in lag_df.columns if 'lag' in col]
                    if lag_cols:
                        correlations = lag_df[lag_cols + [app.target_column]].corr()[app.target_column].drop(app.target_column)
                        
                        fig = px.bar(x=correlations.index, y=correlations.values,
                                   title="Корреляция лагов с целевой переменной")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Сначала выполните предобработку данных")
        
        with tab6:
            st.header("🔄 ACF/PACF анализ")
            
            if app.processed_df is not None:
                acf_pacf_results = app.acf_pacf_analysis()
                
                if acf_pacf_results:
                    acf_values, pacf_values = acf_pacf_results
                    
                    fig = make_subplots(rows=2, cols=1, 
                                      subplot_titles=('ACF', 'PACF'))
                    
                    # ACF
                    fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, name='ACF'), row=1, col=1)
                    
                    # PACF
                    fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name='PACF'), row=2, col=1)
                    
                    fig.update_layout(height=600, title_text="Автокорреляционные функции")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Сначала выполните предобработку данных")
        
        with tab7:
            st.header("🧩 Декомпозиция временного ряда")
            
            if app.processed_df is not None:
                model_type = st.selectbox("Тип модели", ["additive", "multiplicative"])
                
                if st.button("Выполнить декомпозицию"):
                    decomposition = app.decompose_time_series(model=model_type)
                    
                    if decomposition:
                        fig = make_subplots(rows=4, cols=1, 
                                          subplot_titles=('Исходный ряд', 'Тренд', 'Сезонность', 'Остатки'))
                        
                        # Исходный ряд
                        fig.add_trace(go.Scatter(y=decomposition.observed, name='Исходный ряд'), row=1, col=1)
                        
                        # Тренд
                        fig.add_trace(go.Scatter(y=decomposition.trend, name='Тренд'), row=2, col=1)
                        
                        # Сезонность
                        fig.add_trace(go.Scatter(y=decomposition.seasonal, name='Сезонность'), row=3, col=1)
                        
                        # Остатки
                        fig.add_trace(go.Scatter(y=decomposition.resid, name='Остатки'), row=4, col=1)
                        
                        fig.update_layout(height=800, title_text=f"Декомпозиция ({model_type})")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Сначала выполните предобработку данных")
        
        with tab8:
            st.header("📄 Отчет")
            
            if st.button("Сгенерировать отчет"):
                report_html = app.generate_report()
                
                if report_html:
                    st.success("✅ Отчет сгенерирован!")
                    
                    # Показываем отчет
                    st.components.v1.html(report_html, height=600, scrolling=True)
                    
                    # Кнопка скачивания
                    b64 = base64.b64encode(report_html.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="time_series_report.html">📥 Скачать отчет</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    else:
        st.info("👆 Загрузите данные для начала анализа")

if __name__ == "__main__":
    main()

