"""Подготовка House Prices и прогноз цены с сохранённой моделью."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


# В этих полях отсутствие значения обычно означает отсутствие объекта.
ABSENT_CATEGORIES = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
]
ABSENT_NUMBERS = [
    'GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
    'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
]

# Порядок оценок берём из описания датасета, а не из алфавита.
QUALITY = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ORDINAL_MAPS = {
    name: QUALITY for name in [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
    ]
}
ORDINAL_MAPS.update({
    'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'Functional': {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7},
})


class AmesFeatureBuilder(TransformerMixin, BaseEstimator):
    """Учит медианы только на train и создаёт одинаковые признаки во всех выборках."""

    def _normalise(self, X):
        # Сохраняем схему train, чтобы новый дом мог иметь неполный набор полей.
        frame = X.reindex(columns=self.feature_names_in_).copy()
        for col in self.categorical_columns_:
            frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
        for col in ABSENT_CATEGORIES:
            frame[col] = frame[col].fillna('None')

        # Нулевая площадь облицовки помогает отличить отсутствие от неизвестного типа.
        no_veneer = frame['MasVnrArea'].fillna(0).eq(0)
        frame.loc[no_veneer & frame['MasVnrType'].isna(), 'MasVnrType'] = 'None'
        for col in ABSENT_NUMBERS:
            frame[col] = frame[col].fillna(0)
        return frame

    def fit(self, X, y=None):
        # Целевую переменную и идентификатор никогда не включаем в признаки.
        columns = [c for c in X.columns if c not in ('Id', 'SalePrice', 'SalePrice_log')]
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.n_features_in_ = len(columns)
        self.categorical_columns_ = X[columns].select_dtypes(exclude='number').columns.tolist()
        frame = self._normalise(X)

        # Для LotFrontage сохраняем отдельную медиану каждого района.
        self.frontage_by_neighborhood_ = frame.groupby('Neighborhood')['LotFrontage'].median().to_dict()
        self.numeric_medians_ = frame.select_dtypes('number').median().fillna(0).to_dict()
        return self

    def transform(self, X):
        check_is_fitted(self, 'numeric_medians_')
        frame = self._normalise(X)

        # В новом районе используем общую медиану train, не статистику новых домов.
        frontage = frame['Neighborhood'].map(self.frontage_by_neighborhood_)
        frame['LotFrontage'] = frame['LotFrontage'].fillna(frontage)
        for col, median in self.numeric_medians_.items():
            frame[col] = pd.to_numeric(frame[col], errors='coerce').fillna(median)

        # Агрегаты описывают площадь, удобства и возраст дома.
        frame['TotalSF'] = frame['TotalBsmtSF'] + frame['1stFlrSF'] + frame['2ndFlrSF']
        frame['TotalBath'] = frame['FullBath'] + 0.5 * frame['HalfBath']
        frame['TotalBath'] += frame['BsmtFullBath'] + 0.5 * frame['BsmtHalfBath']
        porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
        frame['TotalPorch'] = frame[porch_cols].sum(axis=1)
        frame['HouseAge'] = frame['YrSold'] - frame['YearBuilt']
        frame['RemodAge'] = frame['YrSold'] - frame['YearRemodAdd']

        # Бинарные признаки показывают наличие объектов, а не их размер.
        for name, source in {
            'HasPool': 'PoolArea', 'HasGarage': 'GarageArea', 'Has2ndFloor': '2ndFlrSF',
            'HasFireplace': 'Fireplaces', 'HasBsmt': 'TotalBsmtSF',
        }.items():
            frame[name] = frame[source].gt(0).astype(int)
        frame['IsRemodeled'] = frame['YearRemodAdd'].ne(frame['YearBuilt']).astype(int)
        frame['OverallQual_TotalSF'] = frame['OverallQual'] * frame['TotalSF']

        # Порядковые категории превращаем в осмысленные числовые оценки.
        for col, mapping in ORDINAL_MAPS.items():
            frame[col] = frame[col].map(mapping).astype(float)
        for col in ['MSSubClass', 'MoSold']:
            frame[col] = frame[col].astype(int).astype(str).astype(object)

        # Сжимаем длинные правые хвосты площадей; исходные значения неотрицательны.
        skewed = ['LotArea', 'LotFrontage', 'GrLivArea', 'TotalSF', 'TotalPorch', 'MasVnrArea']
        for col in skewed:
            frame[col] = np.log1p(frame[col].clip(lower=0))
        return frame


def make_price_pipeline(regressor):
    """Возвращает полный Pipeline, который принимает исходные столбцы CSV."""
    # Числа и категории обрабатываются по-разному, но внутри одного fit.
    numeric = SimpleImputer(strategy='median', keep_empty_features=True)
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                  min_frequency=5, sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('categorical', categorical, make_column_selector(dtype_exclude=np.number)),
    ], verbose_feature_names_out=False)

    # Масштабируем и числа, и dummy-столбцы: штрафы моделей сравнивают их веса.
    return Pipeline([
        ('features', AmesFeatureBuilder()),
        ('columns', columns),
        ('scaler', StandardScaler()),
        ('regressor', regressor),
    ])


def predict_price(house_data, model):
    """Возвращает цену в долларах по исходным, ещё не закодированным данным."""
    required = {'OverallQual', 'GrLivArea', 'YearBuilt', 'YrSold'}
    missing = required - house_data.keys()
    if missing:
        raise ValueError(f'Отсутствуют обязательные поля: {sorted(missing)}')

    # Проверяем очевидные ошибки ввода до вызова обученной модели.
    if not 1 <= float(house_data['OverallQual']) <= 10:
        raise ValueError('OverallQual должен быть от 1 до 10')
    if float(house_data['GrLivArea']) <= 0:
        raise ValueError('GrLivArea должен быть положительным')
    if float(house_data['YearBuilt']) > float(house_data['YrSold']):
        raise ValueError('Год продажи не может предшествовать постройке')

    # Pipeline повторяет предобработку, а expm1 возвращает логарифм в доллары.
    row = pd.DataFrame([house_data]).drop(columns=['SalePrice', 'SalePrice_log'], errors='ignore')
    price = float(np.expm1(model.predict(row)[0]))
    if not np.isfinite(price) or price <= 0:
        raise ValueError('Модель вернула некорректную цену')
    return price
