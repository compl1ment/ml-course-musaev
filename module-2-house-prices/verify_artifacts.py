"""Проверяет сохранённую лабораторную без повторного обучения моделей."""

import hashlib
import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from house_price_pipeline import predict_price


def main():
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / 'models/metadata.json').read_text(encoding='utf-8'))
    metrics = json.loads((root / 'models/metrics.json').read_text(encoding='utf-8'))
    train = pd.read_csv(root / 'data/train.csv')
    test = pd.read_csv(root / 'data/test.csv')
    predictions = pd.read_csv(root / 'examples/validation_predictions.csv')

    # Размеры, хеши и Id связывают результаты с неизменёнными исходными файлами.
    assert train.shape == (1460, 81) and test.shape == (1459, 80)
    for name, expected in metadata['source_sha256'].items():
        assert hashlib.sha256((root / 'data' / name).read_bytes()).hexdigest() == expected
    assert predictions['Id'].is_unique and len(predictions) == 292
    assert set(predictions['Id']) == set(metadata['validation_ids'])
    assert set(metadata['excluded_train_ids']).isdisjoint(predictions['Id'])
    assert set(predictions['Id']).issubset(train['Id'])

    # Независимо пересчитываем метрики из опубликованной таблицы прогнозов.
    truth = train.set_index('Id').loc[predictions['Id'], 'SalePrice'].to_numpy()
    assert np.array_equal(truth, predictions['SalePrice'].to_numpy())
    for name, saved in metrics.items():
        estimate = predictions[name].to_numpy()
        actual = {
            'rmse_usd': root_mean_squared_error(truth, estimate),
            'mae_usd': mean_absolute_error(truth, estimate),
            'r2_usd': r2_score(truth, estimate),
            'rmse_log': root_mean_squared_error(np.log1p(truth), np.log1p(estimate)),
            'mae_log': mean_absolute_error(np.log1p(truth), np.log1p(estimate)),
            'r2_log': r2_score(np.log1p(truth), np.log1p(estimate)),
        }
        for metric, value in actual.items():
            assert np.isclose(value, saved[metric], rtol=1e-9, atol=1e-9), (name, metric)

    # Выбор основной модели должен зависеть от train-CV, а не от holdout.
    regular = [name for name in metrics if name != 'stacking_bonus']
    selected = min(regular, key=lambda name: metrics[name]['cv_rmse_log'])
    assert selected == metadata['best_model']
    model = joblib.load(root / 'models/best_model.pkl')
    holdout = train.set_index('Id').loc[predictions['Id']].reset_index().drop(columns='SalePrice')
    assert np.allclose(np.expm1(model.predict(holdout)), predictions[selected], rtol=1e-9)

    # Проверяем, что сохранённые статистики получены именно из обучающей части.
    excluded = set(metadata['validation_ids']) | set(metadata['excluded_train_ids'])
    fit_rows = train.loc[~train['Id'].isin(excluded)]
    assert len(fit_rows) == metadata['n_train'] == 1166
    expected_frontage = fit_rows.groupby('Neighborhood')['LotFrontage'].median().to_dict()
    builder = model.named_steps['features']
    assert expected_frontage.keys() == builder.frontage_by_neighborhood_.keys()
    for group, value in expected_frontage.items():
        assert np.isclose(value, builder.frontage_by_neighborhood_[group], equal_nan=True)
    assert 'Id' not in builder.feature_names_in_ and 'SalePrice' not in builder.feature_names_in_

    # Новый порядок колонок, неизвестный район и частичный ввод не ломают прогноз.
    sample = holdout.head(5)
    assert np.allclose(model.predict(sample), model.predict(sample[sample.columns[::-1]]), rtol=1e-12)
    unknown = sample.iloc[0].to_dict()
    unknown['Neighborhood'] = 'PreviouslyUnseenNeighborhood'
    assert np.isfinite(predict_price(unknown, model))
    minimum = {'OverallQual': 6, 'GrLivArea': 1500, 'YearBuilt': 2000, 'YrSold': 2010}
    assert predict_price(minimum, model) > 0
    for bad in [dict(minimum, OverallQual=11), dict(minimum, GrLivArea=-1),
                dict(minimum, YearBuilt=2020)]:
        try:
            predict_price(bad, model)
        except ValueError:
            pass
        else:
            raise AssertionError('Некорректный ввод не отклонён')

    # Все пять демонстраций проверяем по полным исходным характеристикам.
    demo = json.loads((root / 'examples/demo_houses.json').read_text(encoding='utf-8'))
    demo_table = pd.read_csv(root / 'examples/demo_predictions.csv').set_index('Id')
    filenames = {'linear_regression': 'lr_model.pkl', 'ridge': 'ridge_model.pkl',
                 'lasso': 'lasso_model.pkl', 'elasticnet': 'enet_model.pkl'}
    for name, filename in filenames.items():
        saved_model = joblib.load(root / 'models' / filename)
        for house in demo:
            estimate = predict_price(house, saved_model)
            assert np.isclose(estimate, demo_table.loc[house['Id'], name], rtol=1e-9)

    # Kaggle-файл содержит по одному положительному прогнозу для каждого test Id.
    submission = pd.read_csv(root / 'submission.csv')
    assert list(submission.columns) == ['Id', 'SalePrice']
    assert submission['Id'].tolist() == test['Id'].tolist()
    assert np.isfinite(submission['SalePrice']).all() and submission['SalePrice'].gt(0).all()
    assert np.allclose(submission['SalePrice'], np.expm1(model.predict(test)), rtol=1e-9)

    # Отчёт имеет 14 основных разделов и выполненные кодовые ячейки без ошибок.
    notebook = nbformat.read(root / 'notebook.ipynb', as_version=4)
    nbformat.validate(notebook)
    headings = [line for cell in notebook.cells if cell.cell_type == 'markdown'
                for line in cell.source.splitlines() if line.startswith('## ')]
    assert len(headings) == 14, headings
    code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(out.output_type == 'error' for cell in code_cells for out in cell.outputs)
    assert not any('{{COMPARISON}}' in cell.source or '{{CONCLUSIONS}}' in cell.source
                   for cell in notebook.cells)
    assert all((root / 'examples' / name).is_file() for name in
               ['eda_plots.png', 'residuals.png', 'predictions_scatter.png', 'feature_importance.png'])

    report = {'status': 'passed', 'notebook_sections': len(headings),
              'executed_code_cells': len(code_cells), 'validation_predictions': len(predictions),
              'demo_houses': len(demo), 'kaggle_predictions': len(submission),
              'metrics_recomputed': True, 'saved_models_match': True,
              'training_only_frontage_statistics': True, 'unknown_category_and_partial_input': True}
    (root / 'models/verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
