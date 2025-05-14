import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Функция для анализа и моделирования
def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Загрузка данных из CSV-файла
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Читаем CSV-файл
            data = pd.read_csv(uploaded_file)
            
            # Проверка наличия данных
            if len(data) == 0:
                st.error("Файл пуст или поврежден.")
                return
            
            # Проверка на наличие обязательных столбцов
            required_columns = {'UID', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Target'}
            if not set(required_columns).issubset(set(data.columns)):
                st.error("Файл не содержит всех необходимых столбцов.")
                return
            
            # Предобработка данных
            # Удаляем ненужные столбцы
            data = data.drop(columns=[
                'UID',
                'Product ID',
                'TWF',
                'HDF',
                'PWF',
                'OSF',
                'RNF'
            ])
            
            # Преобразуем категориальную переменную Type в числовую
            encoder = LabelEncoder()
            data['Type'] = encoder.fit_transform(data['Type'])
            
            # Проверка на пропущенные значения
            print(data.isnull().sum())
            
            # Разделение данных на обучающую и тестовую выборки
            X = data.drop(columns=['Target'], axis=1)
            y = data['Target']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Обучение модели логистической регрессии
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
            # Оценка модели
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            
            # Визуализация результатов
            st.header("Результаты обучения модели")
            st.write(f"Accuracy: {accuracy:.2f}")
            
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
            
            st.subheader("Classification Report")
            st.text(classification_report)
            
            st.header("Предсказание по новым данным")
            with st.form("prediction_form"):
                st.write("Введите значения признаков для предсказания:")
                productID = st.selectbox("productID", encoder.classes_)
                air_temp = st.number_input("air temperature [K]", value=300)
                process_temp = st.number_input("process temperature [K]", value=310)
                rotational_speed = st.number_input("rotational speed [rpm]", value=2860)
                torque = st.number_input("torque [Nm]", value=40)
                tool_wear = st.number_input("tool wear [min]", value=0)
                
                submit_button = st.form_submit_button("Предсказать")
                
                if submit_button:
                    input_data = pd.DataFrame({
                        'Type': encoder.transform([productID])[0],
                        'Air temperature': [air_temp],
                        'Process temperature': [process_temp],
                        'Rotational speed': [rotational_speed],
                        'Torque': [torque],
                        'Tool wear': [tool_wear]
                    })
                    
                    prediction = model.predict(input_data)
                    prediction_proba = model.predict_proba(input_data)[:, 1]
                    
                    st.write(f"Предсказание: {prediction[0]}")
                    st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")
        
        except Exception as e:
            st.error(f"Произошла ошибка при обработке данных: {e}")

# Презентация проекта
def presentation_page():
    st.title("Презентация проекта")
    
    # Содержимое презентации в формате Markdown
    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Описание задачи и датасета.
    - Цель: предсказать отказ оборудования (Target = 1) или его отсутствие (Target = 0).
    ---
    ## Этапы работы
    1. Загрузка данных.
    2. Предобработка данных.
    3. Обучение модели.
    4. Оценка модели.
    5. Визуализация результатов.
    ---
    ## Streamlit-приложение
    - Основная страница: анализ данных и предсказания.
    - Страница с презентацией: описание проекта.
    ---
    ## Заключение
    - Итоги и возможные улучшения.
    """
    
    # Настройки презентации
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multisectio