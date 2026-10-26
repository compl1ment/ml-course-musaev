# Модуль 3 — Customer Churn: ансамблевые методы

**Автор:** Мусаев Руслан Шамхалович, ПКТб-23-1. **Дата:** 26.10.2026.

В [notebook.ipynb](notebook.ipynb) выполнены 14 разделов лабораторной:
анализ Telco Customer Churn, сравнение RF/XGBoost/LightGBM/CatBoost,
30 испытаний Optuna, интерпретация SHAP и бонусная калибровка.
Для просмотра без запуска также сохранён [HTML-отчёт](report.html).

## Результаты

| Модель | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit, c |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.8454 | 0.7708 | 0.5532 | 0.7086 | 0.6213 | 0.8398 | 0.506 |
| XGBoost | 0.8318 | 0.7630 | 0.5426 | 0.6818 | 0.6043 | 0.8273 | 0.218 |
| LightGBM | 0.8391 | 0.7622 | 0.5379 | 0.7406 | 0.6232 | 0.8385 | 0.182 |
| CatBoost | 0.8476 | 0.7651 | 0.5394 | 0.7861 | 0.6398 | 0.8455 | 7.455 |
| LightGBM (Optuna) | 0.8500 | 0.8034 | 0.6702 | 0.5107 | 0.5797 | 0.8466 | 0.114 |
| Калибровка (бонус) | — | 0.8020 | 0.6621 | 0.5187 | 0.5817 | 0.8466 | 0.409 |


Основная модель — **LightGBM (Optuna)**, выбранная по максимальному среднему train-CV ROC-AUC **0.8500 ± 0.0121**. На 1409 отложенных клиентах: ROC-AUC **0.8466**, Accuracy **0.8034**, Precision **0.6702**, Recall **0.5107**, F1 **0.5797**. Optuna изменила CV ROC-AUC LightGBM на **+0.0109**, holdout ROC-AUC — на **+0.0081** (это абсолютные разности AUC, не относительные проценты). На этом holdout максимальный AUC у LightGBM (Optuna); это не меняет правило выбора по CV. Разница между близкими AUC не означает статистически доказанного превосходства. При пороге 0.5 Recall базового CatBoost составляет 0.7861, а итоговой модели — 0.5107: лучший AUC не гарантирует лучшую полноту при заданном пороге. Train ROC-AUC основной модели — 0.8619; разрыв с holdout показывает ограничение обобщения. Brier до / после sigmoid-калибровки: **0.1352 / 0.1351**, log loss: **0.4154 / 0.4150**. Меньшие значения лучше.

CV — 5 одинаковых стратифицированных фолдов на 5634 обучающих клиентах.
Holdout — 1409 клиентов (20%), seed 42; положительный класс — уход (`Yes`).
Порог для Precision/Recall/F1 зафиксирован равным 0.5. Время зависит от компьютера;
в `models/metrics.json` отдельно сохранены CV, поиск и финальный fit.
У калиброванного бонуса внешний CV ROC-AUC не оценивался, поэтому стоит прочерк.

## Подготовка и интерпретация

11 пустых строк TotalCharges преобразуются в NaN. Медианы и LabelEncoder
обучаются только внутри train/CV-фолдов. Созданы TenureYears, AvgMonthlyCharge,
IsNewClient и счётчик HasMultipleServices. `customerID` и `Churn` исключены из входа.
RF/XGBoost/LightGBM используют коды категорий с −1 для неизвестных значений;
CatBoost получает исходные строки. Scaler сохранён для состава артефактов,
но не применяется к деревьям. Веса классов вычисляются на обучающих данных.

Для LightGBM (Optuna) на 401 клиентах проверена аддитивность SHAP. Наибольшие средние абсолютные вклады: `Contract`, `MonthlyCharges`, `tenure`, `OnlineSecurity`, `TechSupport`. Клиент **5178-LMXOP** имеет прогноз риска **85.83%**. Вклады в увеличение выхода: `Contract` (+0.872), `tenure` (+0.595), `MonthlyCharges` (+0.292). Вклады в уменьшение: `PhoneService` (-0.001). Единицы вкладов: **log_odds**, базовое значение -1.5100. Сумма базы и всех вкладов воспроизводит выход модели; при log_odds применяется sigmoid. Ни один отдельный вклад не интерпретируется как прибавка стольких же процентных пунктов.

## Воспроизведение

Нужен Python 3.12. Из папки модуля:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Откройте notebook.ipynb в Jupyter/VS Code, выберите окружение и выполните все ячейки
сверху вниз. Подготовка и обучение находятся в `churn_pipeline.py` и `lab_workflow.py`;
их полный код включён в приложение ноутбука. Все пути к данным относительные.
30 испытаний и CV требуют времени. Для offline запуска задайте
`HOUSE_CHURN_CHECK_GITHUB=0` в окружении ядра. По умолчанию raw-проверка использует main;
ветку или коммит задаёт `HOUSE_CHURN_GITHUB_REF`.

## Сохранённая модель

Передавайте исходные 19 характеристик клиента без повторного кодирования.
Рядом с моделью должен быть доступен `churn_pipeline.py` и установленные зависимости.

```python
import json
import joblib
from churn_pipeline import predict_churn

model = joblib.load('models/best_model.pkl')
with open('examples/demo_clients.json', encoding='utf-8') as source:
    clients = json.load(source)
print(predict_churn(clients, model))
```

Joblib/pickle следует загружать только из доверенного источника. Сетевая проверка GitHub пока не подтверждена.

## Файлы

- `data/telco_churn.csv`, `data/churn_info.md` — исходные данные и словарь всех столбцов.
- `models/rf_model.pkl`, `xgb_model.pkl`, `lgbm_model.pkl`, `cat_model.pkl` — базовые Pipeline.
- `models/lgbm_optuna_model.pkl` — настроенный LightGBM; `best_model.pkl` — выбранный кандидат.
- `models/calibrated_model.pkl` — бонус с sigmoid-калибровкой.
- `models/scaler.pkl`, `label_encoders.pkl`, `feature_cols.json` — подготовка и схема.
- `models/metrics.json`, `metadata.json`, `optuna_study.json` — фактические результаты и параметры.
- `models/repository_verification.json`, `verification.json` — результаты проверок.
- `examples/` — графики, все 30 trials, SHAP, holdout-вероятности и демонстрационные клиенты.
- `verify_artifacts.py` — независимая проверка моделей, метрик и структуры отчёта.

Выводы описывают статистические связи и поведение моделей. Эффект мер удержания
и качество на будущих клиентах отдельно не проверены.
