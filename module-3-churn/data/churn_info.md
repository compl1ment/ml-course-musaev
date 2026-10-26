# Данные Telco Customer Churn

Источник: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Исходный пользовательский CSV извлечён из скачанного архива; байты сохранены без изменения.

7043 клиента, 21 столбец, 19 входных характеристик. SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`.

Churn Yes: 1869 (26.54%); No: 5174 (73.46%). ID уникальны, дубликатов строк нет. После числового преобразования TotalCharges получаются 11 пропусков; это клиенты с tenure=0. Для соблюдения задания пропуски заполняются медианой train, строки не удаляются.

| Столбец | Перевод | Тип / варианты | Пропусков после преобразования |
|---|---|---|---:|
| customerID | Идентификатор клиента | идентификатор | 0 |
| gender | Пол | Female, Male | 0 |
| SeniorCitizen | Пенсионер (0/1) | 0, 1 | 0 |
| Partner | Есть партнёр | No, Yes | 0 |
| Dependents | Есть иждивенцы | No, Yes | 0 |
| tenure | Стаж в месяцах | число | 0 |
| PhoneService | Подключена телефония | No, Yes | 0 |
| MultipleLines | Несколько телефонных линий | No, No phone service, Yes | 0 |
| InternetService | Тип интернет-подключения | DSL, Fiber optic, No | 0 |
| OnlineSecurity | Онлайн-защита | No, No internet service, Yes | 0 |
| OnlineBackup | Резервное копирование | No, No internet service, Yes | 0 |
| DeviceProtection | Защита устройства | No, No internet service, Yes | 0 |
| TechSupport | Техническая поддержка | No, No internet service, Yes | 0 |
| StreamingTV | Потоковое телевидение | No, No internet service, Yes | 0 |
| StreamingMovies | Потоковые фильмы | No, No internet service, Yes | 0 |
| Contract | Тип контракта | Month-to-month, One year, Two year | 0 |
| PaperlessBilling | Электронные счета | No, Yes | 0 |
| PaymentMethod | Способ оплаты | Bank transfer (automatic), Credit card (automatic), Electronic check, Mailed check | 0 |
| MonthlyCharges | Ежемесячный платёж, USD | число | 0 |
| TotalCharges | Общая сумма начислений, USD | число | 11 |
| Churn | Ушёл ли клиент (целевая) | No, Yes | 0 |

Новые признаки: стаж в годах, TotalCharges/(tenure+1), индикатор tenure<=6, число подключённых услуг. После подготовки — 23 входных столбца. No internet service и No phone service — смысловые категории, а не пропуски.
