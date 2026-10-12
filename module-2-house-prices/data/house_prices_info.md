# Данные House Prices

Источник: [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

| Файл | Строк | Столбцов | Назначение |
|---|---:|---:|---|
| train.csv | 1460 | 81 | Есть SalePrice |
| test.csv | 1459 | 80 | Цены неизвестны |

79 характеристик описывают дом; Id служит идентификатором. Площади даны в квадратных футах, длины — в футах, цена — в USD.

Цена: среднее $180,921.20, медиана $163,000.00, минимум $34,900, максимум $755,000. Асимметрия 1.8829, эксцесс 6.5363.

Полный словарь столбцов и значений категорий находится в [data_description.txt](data_description.txt).

| Столбец | Смысл | Пропусков train | Пропусков test |
|---|---|---:|---:|
| SalePrice | Цена продажи | 0 | Нет столбца |
| Id | Идентификатор | 0 | 0 |
| OverallQual | Качество материалов и отделки | 0 | 0 |
| OverallCond | Состояние дома | 0 | 0 |
| GrLivArea | Жилая площадь над землёй | 0 | 0 |
| YearBuilt | Год постройки | 0 | 0 |
| YearRemodAdd | Год реконструкции | 0 | 0 |
| Neighborhood | Район | 0 | 0 |
| LotArea | Площадь участка | 0 | 0 |
| LotFrontage | Длина границы вдоль улицы | 259 | 227 |
| TotalBsmtSF | Площадь подвала | 0 | 1 |
| GarageCars | Вместимость гаража | 0 | 1 |
| GarageArea | Площадь гаража | 0 | 1 |
| FullBath | Полные ванные над землёй | 0 | 0 |
| KitchenQual | Качество кухни | 0 | 1 |
| PoolQC | Качество бассейна | 1453 | 1456 |
| MasVnrType | Тип каменной или кирпичной облицовки | 872 | 894 |

## Пропуски во всех столбцах

| Столбец | Train | Test |
|---|---:|---:|
| Id | 0 | 0 |
| MSSubClass | 0 | 0 |
| MSZoning | 0 | 4 |
| LotFrontage | 259 | 227 |
| LotArea | 0 | 0 |
| Street | 0 | 0 |
| Alley | 1369 | 1352 |
| LotShape | 0 | 0 |
| LandContour | 0 | 0 |
| Utilities | 0 | 2 |
| LotConfig | 0 | 0 |
| LandSlope | 0 | 0 |
| Neighborhood | 0 | 0 |
| Condition1 | 0 | 0 |
| Condition2 | 0 | 0 |
| BldgType | 0 | 0 |
| HouseStyle | 0 | 0 |
| OverallQual | 0 | 0 |
| OverallCond | 0 | 0 |
| YearBuilt | 0 | 0 |
| YearRemodAdd | 0 | 0 |
| RoofStyle | 0 | 0 |
| RoofMatl | 0 | 0 |
| Exterior1st | 0 | 1 |
| Exterior2nd | 0 | 1 |
| MasVnrType | 872 | 894 |
| MasVnrArea | 8 | 15 |
| ExterQual | 0 | 0 |
| ExterCond | 0 | 0 |
| Foundation | 0 | 0 |
| BsmtQual | 37 | 44 |
| BsmtCond | 37 | 45 |
| BsmtExposure | 38 | 44 |
| BsmtFinType1 | 37 | 42 |
| BsmtFinSF1 | 0 | 1 |
| BsmtFinType2 | 38 | 42 |
| BsmtFinSF2 | 0 | 1 |
| BsmtUnfSF | 0 | 1 |
| TotalBsmtSF | 0 | 1 |
| Heating | 0 | 0 |
| HeatingQC | 0 | 0 |
| CentralAir | 0 | 0 |
| Electrical | 1 | 0 |
| 1stFlrSF | 0 | 0 |
| 2ndFlrSF | 0 | 0 |
| LowQualFinSF | 0 | 0 |
| GrLivArea | 0 | 0 |
| BsmtFullBath | 0 | 2 |
| BsmtHalfBath | 0 | 2 |
| FullBath | 0 | 0 |
| HalfBath | 0 | 0 |
| BedroomAbvGr | 0 | 0 |
| KitchenAbvGr | 0 | 0 |
| KitchenQual | 0 | 1 |
| TotRmsAbvGrd | 0 | 0 |
| Functional | 0 | 2 |
| Fireplaces | 0 | 0 |
| FireplaceQu | 690 | 730 |
| GarageType | 81 | 76 |
| GarageYrBlt | 81 | 78 |
| GarageFinish | 81 | 78 |
| GarageCars | 0 | 1 |
| GarageArea | 0 | 1 |
| GarageQual | 81 | 78 |
| GarageCond | 81 | 78 |
| PavedDrive | 0 | 0 |
| WoodDeckSF | 0 | 0 |
| OpenPorchSF | 0 | 0 |
| EnclosedPorch | 0 | 0 |
| 3SsnPorch | 0 | 0 |
| ScreenPorch | 0 | 0 |
| PoolArea | 0 | 0 |
| PoolQC | 1453 | 1456 |
| Fence | 1179 | 1169 |
| MiscFeature | 1406 | 1408 |
| MiscVal | 0 | 0 |
| MoSold | 0 | 0 |
| YrSold | 0 | 0 |
| SaleType | 0 | 1 |
| SaleCondition | 0 | 0 |
| SalePrice | 0 | Нет столбца |

Статистики для предобработки обучаются только на train-фолдах. Отложенные 292 дома не фильтруются по цене. Из обучающей части исключены два объекта (Id 524 и 1299) по сочетанию GrLivArea > 4000 и SalePrice < 300000. Границы IQR используются для анализа, а не для удаления всех крупных домов.
