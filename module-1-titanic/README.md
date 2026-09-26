# Модуль 1 — Titanic: EDA и бинарная классификация

**Автор:** Мусаев Руслан Шамхалович, ПКТб-23-1<br>
**Дата обучения и проверки:** 23.09.2026<br>
**Данные:** [Kaggle Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

В [notebook.ipynb](./notebook.ipynb) показаны анализ данных, графики, обучение и проверка трёх моделей: Logistic Regression, Decision Tree и Random Forest. Ноутбук содержит 14 разделов; все кодовые ячейки выполнены. Метрики посчитаны на 179 строках из `train.csv`, которые не участвовали в обучении. Ответов для Kaggle `test.csv` у нас нет.

## Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,804 | 0,783 | 0,681 | 0,729 | **0,849** |
| Decision Tree | 0,782 | 0,742 | 0,667 | 0,702 | 0,813 |
| Random Forest | 0,788 | 0,804 | 0,594 | 0,683 | 0,846 |

Время обучения записано в [metrics.json](./models/metrics.json). На другом компьютере оно может отличаться. Для повторного разделения данных используются `test_size=0.2`, `random_state=42` и сохранение долей классов (`stratify`).

## Воспроизведение

Нужен Python 3.12. Запускайте ноутбук **из папки `module-1-titanic`**: пути к данным и моделям заданы относительно неё.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Откройте `notebook.ipynb` в Jupyter или VS Code и выполните ячейки по порядку. Вводить данные вручную не нужно. В разделе 11 модель скачивается из GitHub; три пробных прогноза сравниваются с локальной моделью.

## Файлы

- `data/train.csv`, `data/test.csv`, `data/titanic_info.md` — исходные данные и описание.
- `models/lr_model.pkl`, `dt_model.pkl`, `rf_model.pkl` — обученные модели.
- `models/scaler.pkl`, `le_sex.pkl`, `feature_cols.json`, `preprocessing_state.json` — настройки подготовки новых данных.
- `models/metrics.json`, `metadata.json` — метрики, время обучения и параметры запуска.
- `examples/` — графики EDA, ROC-кривые, матрицы ошибок и важность признаков.
- `submission.csv` — 418 предсказаний для Kaggle `test.csv` в требуемом формате.
- `requirements.txt` — нужные библиотеки и их версии.
