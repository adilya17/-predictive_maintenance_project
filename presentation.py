import streamlit as st
import reveal_slides as rs
from PIL import Image  # Для добавления изображений

def presentation_page():
    st.title("🎯 Презентация проекта: Прогнозирование отказов оборудования")
    
    # Настройки презентации в сайдбаре
    with st.sidebar:
        st.header("⚙️ Настройки презентации")
        theme = st.selectbox(
            "Цветовая тема",
            ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"],
            index=0
        )
        transition = st.selectbox(
            "Анимация перехода",
            ["slide", "convex", "concave", "zoom", "none"],
            index=0
        )
        st.markdown("---")
        st.info("ℹ️ Используйте стрелки клавиатуры для навигации")

    # Содержание презентации с улучшенной разметкой
    presentation_markdown = """
    <!-- .slide: data-background-color="#2d3e50" -->
    # 🏭 Прогнозирование отказов оборудования
    ### Система предиктивного обслуживания
    ---
    ## 📌 Цели проекта
    - Разработка ML-модели для предсказания отказов
    - Снижение простоев оборудования на 20-30%
    - Оптимизация затрат на обслуживание
    
    ```python
    # Пример кода
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    ```
    ---
    <!-- .slide: data-background="#4b6cb7" -->
    ## 📊 Используемые данные
    **Датасет:** AI4I 2020 Predictive Maintenance  
    **Характеристики:**
    - 10,000 записей
    - 14 параметров оборудования
    - 5 типов отказов
    
    | Параметр          | Диапазон значений |
    |-------------------|------------------|
    | Температура       | 250-350K         |
    | Скорость вращения | 1000-3000 rpm    |
    ---
    ## 🛠️ Предобработка данных
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <div>
    **Этапы обработки:**
    1. Удаление ID-полей
    2. Кодирование категорий
    3. Масштабирование
    4. Балансировка классов
    </div>
    <div>
    ```python
    # Пример обработки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ```
    </div>
    </div>
    ---
    ## 🤖 Обучение моделей
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <div>
    **Тестируемые модели:**
    - Логистическая регрессия
    - Случайный лес
    - XGBoost
    - SVM
    </div>
    <div>
    **Лучшая модель:**
    ```python
    RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    ```
    </div>
    </div>
    ---
    ## 📈 Результаты
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <div>
    **Метрики:**
    - Accuracy: 97.4%
    - Precision: 0.89
    - Recall: 0.75
    - AUC-ROC: 0.98
    </div>
    <div>
    ```python
    print(classification_report(
        y_test, 
        y_pred,
        target_names=["Нет отказа", "Отказ"]
    ))
    ```
    </div>
    </div>
    ---
    ## 🚀 Дальнейшее развитие
    **Планы улучшения:**
    1. Интеграция с IoT-датчиками
    2. Реализация real-time прогнозирования
    3. Разработка API для интеграции
    
    **Бизнес-эффект:**
    - Снижение затрат на 15-25%
    - Увеличение uptime оборудования
    ---
    <!-- .slide: data-background-color="#2d3e50" -->
    ## 🙏 Благодарности
    ### Спасибо за внимание!
    ### Вопросы?
    """

    # Загрузка логотипа (пример)
    try:
        logo = Image.open("logo.png")
        st.sidebar.image(logo, width=200)
    except:
        st.sidebar.warning("Логотип не найден")

    # Настройка и отображение презентации
    rs.slides(
        presentation_markdown,
        height=700,
        theme=theme,
        config={
            "transition": transition,
            "controls": True,
            "progress": True,
            "center": True,
            "slideNumber": True
        },
        markdown_props={
            "data-separator": "^---$",
            "data-separator-vertical": "^--$",
            "data-separator-notes": "^Note:"
        }
    )

if __name__ == "__main__":
    presentation_page()