"""Независимая проверка сохранённых моделей, метрик, подготовки и ноутбука."""
from pathlib import Path
import hashlib
import json

import joblib
import nbformat
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from churn_pipeline import CATEGORICAL, FEATURES, NUMERIC, TelcoPreprocessor, predict_churn


ROOT = Path(__file__).resolve().parent


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def main():
    metadata = load('models/metadata.json')
    metrics = load('models/metrics.json')
    frame = pd.read_csv(ROOT / 'data/telco_churn.csv')
    # Хеш проверяет именно исходный CSV, а не похожую копию из другого источника.
    assert hashlib.sha256((ROOT / 'data/telco_churn.csv').read_bytes()).hexdigest() == metadata['source_sha256']
    assert frame.shape == (7043, 21) and frame.customerID.is_unique
    train = frame.set_index('customerID').loc[metadata['train_ids']]
    test = frame.set_index('customerID').loc[metadata['test_ids']]
    assert len(train) == 5634 and len(test) == 1409
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(frame.customerID)
    # Повторяем только разделение, не обучение, и проверяем исходный порядок клиентов.
    expected_train, expected_test = train_test_split(frame.index, test_size=0.2,
        stratify=frame.Churn.eq('Yes').astype(int), random_state=42)
    assert frame.loc[expected_train, 'customerID'].tolist() == metadata['train_ids']
    assert frame.loc[expected_test, 'customerID'].tolist() == metadata['test_ids']
    assert frame.loc[pd.to_numeric(frame.TotalCharges, errors='coerce').isna(), 'tenure'].eq(0).all()

    predictions = pd.read_csv(ROOT / 'examples/holdout_predictions.csv')
    assert predictions.customerID.tolist() == metadata['test_ids']
    truth = test.Churn.eq('Yes').astype(int).to_numpy()
    assert np.array_equal(predictions.Churn, truth)
    files = {'random_forest': 'rf_model.pkl', 'xgboost': 'xgb_model.pkl',
        'lightgbm': 'lgbm_model.pkl', 'catboost': 'cat_model.pkl',
        'lightgbm_optuna': 'lgbm_optuna_model.pkl', 'calibrated_bonus': 'calibrated_model.pkl'}
    X = test.drop(columns='Churn')
    for name, filename in files.items():
        probability = predictions[name].to_numpy()
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        labels = probability >= 0.5
        # Метрики пересчитываются непосредственно из сохранённых вероятностей.
        actual = {'accuracy': accuracy_score(truth, labels),
            'precision': precision_score(truth, labels), 'recall': recall_score(truth, labels),
            'f1': f1_score(truth, labels), 'roc_auc': roc_auc_score(truth, probability),
            'average_precision': average_precision_score(truth, probability),
            'brier_score': brier_score_loss(truth, probability), 'log_loss': log_loss(truth, probability)}
        assert all(np.isclose(value, metrics[name][key], atol=1e-12) for key, value in actual.items()), name
        model = joblib.load(ROOT / 'models' / filename)
        actual_probability = model.predict_proba(X)[:, 1]
        # Pandas записывает float32 XGBoost с меньшим числом значащих цифр.
        tolerance = 1e-6 if actual_probability.dtype == np.float32 else 1e-10
        assert np.allclose(actual_probability, probability, rtol=tolerance, atol=1e-8), name
        # На каждом сохранённом кандидате статистики должны совпадать с train.
        if name != 'calibrated_bonus':
            preprocessor = model.named_steps['preprocess']
            expected = train[NUMERIC].apply(pd.to_numeric, errors='coerce').median()
            assert all(np.isclose(preprocessor.numeric_medians_[col], expected[col]) for col in NUMERIC)
            assert 'customerID' not in preprocessor.feature_names_in_ and 'Churn' not in preprocessor.feature_names_in_
            assert len(preprocessor.get_feature_names_out()) == 23
            assert len(metrics[name]['fold_roc_auc']) == 5
            assert np.isclose(np.mean(metrics[name]['fold_roc_auc']), metrics[name]['cv_roc_auc'])
            # Перестановка колонок и неизвестная категория не ломают инференс.
            assert np.allclose(model.predict_proba(X.head(3)), model.predict_proba(X.head(3)[X.columns[::-1]]))
            unknown = X.iloc[:1].copy()
            unknown['Contract'] = 'Previously unseen contract'
            assert np.isfinite(model.predict_proba(unknown)).all()

    best = max(metadata['selection_candidates'], key=lambda name: metrics[name]['cv_roc_auc'])
    assert best == metadata['best_model']
    model = joblib.load(ROOT / 'models/best_model.pkl')
    assert np.allclose(model.predict_proba(X)[:, 1], predictions[best], rtol=1e-10)
    assert load('models/feature_cols.json') == FEATURES and not metadata['scaler_applied']
    # Отдельно проверяем, что медианы не подменяются статистиками нового входа.
    training_example = train.iloc[:2].copy()
    training_example['TotalCharges'] = [100.0, 300.0]
    preprocessor = TelcoPreprocessor().fit(training_example)
    new = X.iloc[:1].copy()
    new['TotalCharges'] = ' '
    new['Contract'] = 'Novel category'
    transformed = preprocessor.transform(new)
    assert transformed.TotalCharges.iloc[0] == 200.0 and transformed.Contract.iloc[0] == -1
    assert transformed.TenureYears.iloc[0] == transformed.tenure.iloc[0] / 12
    assert transformed.AvgMonthlyCharge.iloc[0] == 200.0 / (transformed.tenure.iloc[0] + 1)

    demo = load('examples/demo_clients.json')
    demo_predictions = pd.read_csv(ROOT / 'examples/demo_predictions.csv')
    assert np.allclose(predict_churn(demo, model).churn_probability, demo_predictions.churn_probability)
    try:
        predict_churn(demo, model, threshold=2)
    except ValueError:
        pass
    else:
        raise AssertionError('Не отклонён некорректный порог')

    trials = pd.read_csv(ROOT / 'examples/optuna_trials.csv')
    study = load('models/optuna_study.json')
    assert len(trials) == study['n_trials'] == metadata['optuna_trials'] == 30
    assert trials.state.eq('COMPLETE').all() and np.isclose(trials.value.max(), study['best_value'])
    assert np.isclose(study['best_value'], metrics['lightgbm_optuna']['cv_roc_auc'])
    interpretation = load('examples/shap_interpretation.json')
    arrays = np.load(ROOT / 'examples/shap_values.npz')
    additive = arrays['base_value'] + arrays['values'].sum(axis=1)
    recovered = expit(additive) if interpretation['unit'] == 'log_odds' else additive
    assert interpretation['additivity_verified']
    assert np.allclose(recovered, arrays['probability'], atol=1e-5)
    assert np.allclose(arrays['probability'], predictions[best].to_numpy()[arrays['positions']], atol=1e-10)

    notebook = nbformat.read(ROOT / 'notebook.ipynb', as_version=4)
    nbformat.validate(notebook)
    # Полный код в приложении не добавляет отдельных основных разделов.
    headings = [line for cell in notebook.cells if cell.cell_type == 'markdown'
                for line in cell.source.splitlines() if line.startswith('## ')]
    assert len(headings) == 14
    code = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
    assert not any(out.output_type == 'error' for cell in code for out in cell.outputs)
    assert not any(token in cell.source for token in ['{{EDA}}', '{{SHAP}}', '{{CONCLUSIONS}}', '{{COMPARISON}}'] for cell in notebook.cells)
    appendix = notebook.cells[-1].source
    for filename in ['churn_pipeline.py', 'lab_workflow.py']:
        assert (ROOT / filename).read_text(encoding='utf-8') in appendix
    figures = ['eda_plots.png', 'missing_values.png', 'roc_curves.png', 'confusion_matrix.png',
        'feature_importance.png', 'optuna_history.png', 'calibration.png',
        'shap_summary.png', 'shap_bar.png', 'shap_waterfall.png', 'shap_dependence.png', 'metrics_table.png']
    assert all((ROOT / 'examples' / name).stat().st_size > 1000 for name in figures)
    report = {'status': 'passed', 'source_hash_verified': True, 'train_only_preprocessing': True,
        'metrics_recomputed': True, 'saved_models_match': True, 'selection_uses_cv': True,
        'unknown_categories': True, 'optuna_trials': len(trials), 'shap_additivity': True,
        'notebook_sections': len(headings), 'executed_cells': len(code), 'holdout_clients': len(test),
        'figures': len(figures), 'best_model': best,
        'confusion_matrix': confusion_matrix(truth, predictions[best] >= 0.5).tolist(),
        'classification_report': classification_report(truth, predictions[best] >= 0.5, output_dict=True)}
    (ROOT / 'models/verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'classification_report'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
