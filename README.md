 Проект: Прогнозирование отказов оборудования с помощью бинарной классификации

 О проекте
Проект направлен на создание ML-модели для предсказания вероятности выхода оборудования из строя. Модель классифицирует состояние оборудования:
- 1 - ожидается отказ
- 0 - оборудование работает нормально

Решение реализовано в виде интерактивного Streamlit-приложения с визуализацией результатов.

 Данные
В работе используется открытый датасет *AI4I 2020 Predictive Maintenance Dataset*:
- 10,000 записей
- 14 характеристик оборудования
- Сбалансированные классы

[Описание данных на UCI](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset)

Быстрый старт

1. Установите репозиторий:
```bash
git clone https://github.com/AmirRRR777/predictive_maintenance_project.git
cd predictive_maintenance_project
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Запустите приложение:
```bash
streamlit run app.py
```

## Структура проекта
```
predictive_maintenance_project/
├── app.py                 # Главный модуль приложения
├── analysis_and_model.py  # Анализ данных и ML-модель
├── presentation.py        # Презентационные материалы
├── requirements.txt       # Зависимости Python
├── data/                  # Исходные данные
└── README.md              # Документация