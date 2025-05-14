import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler

def analysis_and_model_page():
    st.title(" Анализ данных и модель прогнозирования отказов")
    
    # 1) Загрузка данных
    st.header("1. Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Показать данные
        if st.checkbox("Показать первые 5 строк данных"):
            st.write(data.head())
        
        # 2) Предобработка
        st.header("2. Предобработка данных")
        data_clean = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
        
        # Кодирование категориальных признаков
        le = LabelEncoder()
        data_clean['Type'] = le.fit_transform(data_clean['Type'])
        
        # Разделение на признаки и целевую переменную
        X = data_clean.drop(columns=['Machine failure'])
        y = data_clean['Machine failure']
        
        # Масштабирование числовых признаков
        num_cols = ['Air temperature [K]', 'Process temperature [K]', 
                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 3) Обучение модели
        st.header("3. Обучение модели")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # 4) Оценка модели
        st.header("4. Оценка модели")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Метрики
        st.subheader("Метрики качества")
        st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.3f}")
        st.write(f"**AUC-ROC**: {roc_auc_score(y_test, y_proba):.3f}")
        
        # Confusion Matrix
        st.subheader("Матрица ошибок")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Нет отказа', 'Отказ'],
                   yticklabels=['Нет отказа', 'Отказ'])
        ax.set_xlabel('Предсказание')
        ax.set_ylabel('Факт')
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("Отчет классификации")
        st.text(classification_report(y_test, y_pred))
        
        # ROC Curve
        st.subheader("ROC-кривая")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC кривая (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
        # 5) Предсказание
        st.header("5. Прогнозирование на новых данных")
        with st.form("prediction_form"):
            st.subheader("Введите параметры оборудования:")
            
            col1, col2 = st.columns(2)
            with col1:
                input_type = st.selectbox("Тип продукта", options=['L', 'M', 'H'], 
                                        help="L=0, M=1, H=2")
                input_air_temp = st.number_input("Температура воздуха [K]", 
                                               min_value=250.0, max_value=350.0, value=300.0)
                input_process_temp = st.number_input("Температура процесса [K]", 
                                                   min_value=250.0, max_value=350.0, value=310.0)
            
            with col2:
                input_speed = st.number_input("Скорость вращения [rpm]", 
                                            min_value=1000, max_value=3000, value=1500)
                input_torque = st.number_input("Крутящий момент [Nm]", 
                                             min_value=0.0, max_value=100.0, value=40.0)
                input_wear = st.number_input("Износ инструмента [min]", 
                                           min_value=0, max_value=300, value=0)
            
            submitted = st.form_submit_button("Сделать прогноз")
            
            if submitted:
                # Преобразование введенных данных
                type_map = {'L': 0, 'M': 1, 'H': 2}
                input_data = pd.DataFrame({
                    'Type': [type_map[input_type]],
                    'Air temperature [K]': [input_air_temp],
                    'Process temperature [K]': [input_process_temp],
                    'Rotational speed [rpm]': [input_speed],
                    'Torque [Nm]': [input_torque],
                    'Tool wear [min]': [input_wear]
                })
                
                # Масштабирование
                input_data[num_cols] = scaler.transform(input_data[num_cols])
                
                # Прогноз
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0][1]
                
                # Отображение результатов
                st.subheader("Результаты прогнозирования")
                if prediction == 1:
                    st.error(f" Прогнозируется отказ оборудования (вероятность: {prediction_proba:.1%})")
                else:
                    st.success(f" Оборудование в норме (вероятность отказа: {prediction_proba:.1%})")
                
                # Дополнительная информация
                st.write(f"**Точность модели (Accuracy)**: {accuracy_score(y_test, y_pred):.1%}")
                st.write(f"**Площадь под ROC-кривой (AUC)**: {roc_auc:.3f}")

if __name__ == "__main__":
    analysis_and_model_page()