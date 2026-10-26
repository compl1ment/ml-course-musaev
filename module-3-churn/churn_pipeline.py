"""Подготовка Telco Churn и воспроизводимые полные Pipeline для четырёх моделей."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


NUMERIC = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
               'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
               'Contract', 'PaperlessBilling', 'PaymentMethod']
DERIVED = ['TenureYears', 'AvgMonthlyCharge', 'IsNewClient', 'HasMultipleServices']
FEATURES = NUMERIC + CATEGORICAL + DERIVED


class TelcoPreprocessor(TransformerMixin, BaseEstimator):
    """Учится только на переданном train; неизвестные категории получают код −1.

    CatBoost может получать исходные строки категорий. Для остальных моделей
    сохраняются отдельные LabelEncoder. Scaler сохраняется для требований задания,
    но к деревьям не применяется: им не требуется стандартизация.
    """
    def __init__(self, native_categories=False):
        self.native_categories = native_categories

    @staticmethod
    def _raw_frame(X):
        # Удаляем идентификатор и целевую посредством явного списка признаков.
        frame = pd.DataFrame(X).reindex(columns=NUMERIC + CATEGORICAL).copy()
        for col in NUMERIC:
            frame[col] = pd.to_numeric(frame[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        # Пенсионер — категория 0/1; строковое представление стабильно для новых входов.
        frame['SeniorCitizen'] = pd.to_numeric(frame['SeniorCitizen'], errors='coerce').fillna(-1).astype(int).astype(str)
        for col in CATEGORICAL:
            frame[col] = frame[col].fillna('Missing').astype(str).str.strip().replace('', 'Missing')
        return frame

    def _engineer(self, frame):
        # Медианы берутся из fit текущего фолда; test не участвует в их расчёте.
        frame = frame.copy()
        for col in NUMERIC:
            frame[col] = frame[col].fillna(self.numeric_medians_[col])
        frame['TenureYears'] = frame['tenure'] / 12
        frame['AvgMonthlyCharge'] = frame['TotalCharges'] / (frame['tenure'] + 1)
        frame['IsNewClient'] = (frame['tenure'] <= 6).astype(int)
        # Считаем 8 услуг со значением Yes и отдельное подключение к интернету.
        services = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        frame['HasMultipleServices'] = frame[services].eq('Yes').sum(axis=1)
        frame['HasMultipleServices'] += frame['InternetService'].isin(['DSL', 'Fiber optic']).astype(int)
        return frame[FEATURES]

    def fit(self, X, y=None):
        frame = self._raw_frame(X)
        self.feature_names_in_ = np.asarray(NUMERIC + CATEGORICAL, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.numeric_medians_ = frame[NUMERIC].median().to_dict()
        assert all(np.isfinite(list(self.numeric_medians_.values())))
        # Кодировщики видят только категории текущей обучающей части.
        self.label_encoders_ = {col: LabelEncoder().fit(frame[col]) for col in CATEGORICAL}
        encoded = self._encode(self._engineer(frame))
        self.scaler_ = StandardScaler().fit(encoded)
        return self

    def _encode(self, frame):
        encoded = frame.copy()
        for col, encoder in self.label_encoders_.items():
            mapping = dict(zip(encoder.classes_, range(len(encoder.classes_))))
            encoded[col] = frame[col].map(mapping).fillna(-1).astype(int)
        return encoded.astype(float)

    def transform(self, X):
        check_is_fitted(self, 'numeric_medians_')
        frame = self._engineer(self._raw_frame(X))
        return frame if self.native_categories else self._encode(frame)

    def display_frame(self, X):
        """Числовые коды нужны только для цветов SHAP-графиков CatBoost."""
        check_is_fitted(self, 'numeric_medians_')
        return self._encode(self._engineer(self._raw_frame(X)))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)


class BalancedXGBClassifier(XGBClassifier):
    """Соотношение классов пересчитывается в каждом fit, включая CV-фолды."""
    def fit(self, X, y, **kwargs):
        labels = np.asarray(y)
        self.set_params(scale_pos_weight=float((labels == 0).sum() / (labels == 1).sum()))
        return super().fit(X, y, **kwargs)


def model_pipeline(estimator, native_categories=False):
    # Полная предобработка является частью Pipeline и повторяется внутри CV.
    return Pipeline([('preprocess', TelcoPreprocessor(native_categories=native_categories)),
                     ('model', estimator)])


def baseline_models():
    return {
        'random_forest': model_pipeline(RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=2)),
        'xgboost': model_pipeline(BalancedXGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, tree_method='hist',
            eval_metric='logloss', random_state=42, n_jobs=2)),
        'lightgbm': model_pipeline(LGBMClassifier(
            n_estimators=200, num_leaves=31, max_depth=7, learning_rate=0.05,
            class_weight='balanced', subsample=0.9, subsample_freq=1,
            random_state=42, n_jobs=2, verbosity=-1, importance_type='gain')),
        'catboost': model_pipeline(CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights='Balanced', cat_features=tuple(CATEGORICAL),
            random_seed=42, thread_count=2, verbose=False,
            allow_writing_files=False), native_categories=True),
    }


def predict_churn(client, model, threshold=0.5):
    """Возвращает вероятность ухода и решение для исходных характеристик клиента."""
    if not 0 < threshold < 1:
        raise ValueError('Порог должен быть между 0 и 1.')
    frame = pd.DataFrame([client]) if isinstance(client, dict) else pd.DataFrame(client)
    if frame.empty or any(col not in frame for col in NUMERIC + CATEGORICAL):
        raise ValueError('Передайте все 19 исходных характеристик клиента.')
    probability = np.asarray(model.predict_proba(frame))[:, 1]
    return pd.DataFrame({'churn_probability': probability,
                         'churn_prediction': (probability >= threshold).astype(int)}, index=frame.index)
