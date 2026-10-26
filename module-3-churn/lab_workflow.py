"""Вычисления лабораторной 3. Функции вызываются по порядку из notebook.ipynb."""
from io import BytesIO
from pathlib import Path
import hashlib
import importlib
import json
import os
import platform
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import requests
import seaborn as sns
import shap
from scipy.special import expit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from threadpoolctl import threadpool_limits
from lightgbm import LGBMClassifier

from churn_pipeline import (CATEGORICAL, FEATURES, NUMERIC, baseline_models,
                            model_pipeline, predict_churn)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
EXAMPLES = ROOT / 'examples'
NAMES = {'random_forest': 'Random Forest', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
         'catboost': 'CatBoost', 'lightgbm_optuna': 'LightGBM (Optuna)',
         'calibrated_bonus': 'Калибровка (бонус)'}
FILES = {'random_forest': 'rf_model.pkl', 'xgboost': 'xgb_model.pkl',
         'lightgbm': 'lgbm_model.pkl', 'catboost': 'cat_model.pkl',
         'lightgbm_optuna': 'lgbm_optuna_model.pkl'}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_figure(name):
    # Один файл соответствует текущему графику, который виден и в ноутбуке.
    plt.tight_layout()
    plt.savefig(EXAMPLES / name, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def load_data():
    for folder in [MODELS, EXAMPLES]:
        folder.mkdir(exist_ok=True)
    sns.set_theme(style='whitegrid', context='notebook')
    frame = pd.read_csv(DATA / 'telco_churn.csv')
    assert frame.shape == (7043, 21) and frame.customerID.is_unique
    assert set(frame.Churn) == {'Yes', 'No'} and not frame.duplicated().any()
    # Сначала фиксируем holdout, затем обучаем любую предобработку на train.
    y = frame.Churn.eq('Yes').astype(int)
    train_index, test_index = train_test_split(frame.index, test_size=0.2, stratify=y, random_state=42)
    X = frame.drop(columns=['customerID', 'Churn'])
    state = {'df': frame, 'X_train': X.loc[train_index], 'X_test': X.loc[test_index],
             'y_train': y.loc[train_index], 'y_test': y.loc[test_index],
             'train_index': train_index, 'test_index': test_index,
             'cv': StratifiedKFold(5, shuffle=True, random_state=42),
             'models': {}, 'metrics': {}}
    print('Размер:', frame.shape, '| Train:', len(train_index), '| Test:', len(test_index))
    print('Классы:', frame.Churn.value_counts().to_dict())
    print('Пустой TotalCharges:', pd.to_numeric(frame.TotalCharges, errors='coerce').isna().sum())
    print(frame.head().to_string(index=False))
    frame.info()
    print(frame.describe(include='all').T.to_string())
    return state


def create_eda(state):
    # Выводы EDA строим по train; закрытая часть не определяет обработку данных.
    frame = state['df'].loc[state['train_index']].copy()
    frame['TotalCharges'] = pd.to_numeric(frame.TotalCharges, errors='coerce')
    frame['churn_flag'] = frame.Churn.eq('Yes').astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    sns.countplot(data=frame, x='Churn', order=['No', 'Yes'], ax=axes[0, 0])
    axes[0, 0].set(title='Оставшиеся и ушедшие клиенты', xlabel='Churn', ylabel='Клиентов')
    sns.histplot(data=frame, x='tenure', hue='Churn', bins=24, ax=axes[0, 1])
    axes[0, 1].set(title='Стаж клиента и отток', xlabel='Месяцы', ylabel='Клиентов')
    sns.boxplot(data=frame, x='Churn', y='MonthlyCharges', order=['No', 'Yes'], ax=axes[0, 2])
    axes[0, 2].set(title='Ежемесячные платежи', xlabel='Churn', ylabel='Платёж, USD')
    sns.heatmap(frame[NUMERIC + ['churn_flag']].corr(), annot=True, fmt='.2f',
                cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Корреляции числовых признаков (train)')
    for ax, column in [(axes[1, 1], 'Contract'), (axes[1, 2], 'InternetService')]:
        rates = frame.groupby(column)['churn_flag'].mean().sort_values(ascending=False) * 100
        rates.plot.bar(ax=ax, rot=12, color='#3677a8')
        ax.set(title=f'Доля ушедших по {column}', xlabel=column, ylabel='Ушедшие, %')
        for bar, rate in zip(ax.patches, rates):
            ax.annotate(f'{rate:.1f}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', xytext=(0, 3), textcoords='offset points')
        ax.set_ylim(0, max(rates) * 1.2)
    save_figure('eda_plots.png')
    # Пропуски после преобразования строк в числа также показываем отдельно.
    missing = frame[NUMERIC].isna().sum()
    missing.plot.bar(figsize=(7, 3), rot=0, title='Пропуски в числовых признаках train')
    plt.ylabel('Значений')
    save_figure('missing_values.png')
    state['business'] = {col: frame.groupby(col)['churn_flag'].agg(['mean', 'size']).reset_index().to_dict('records')
                         for col in ['Contract', 'InternetService', 'OnlineSecurity', 'TechSupport']}
    write_json(EXAMPLES / 'business_insights.json', state['business'])
    print('Пропуски train:', missing.to_dict())
    print('Отток по контракту:', state['business']['Contract'])


def train_candidate(state, name, estimator, cv_scores=None, search_seconds=0):
    # CV оценивает весь Pipeline, включая медианы и категории каждого фолда.
    started = time.perf_counter()
    if cv_scores is None:
        with threadpool_limits(limits=2):
            cv_scores = cross_val_score(estimator, state['X_train'], state['y_train'],
                scoring='roc_auc', cv=state['cv'], n_jobs=1, error_score='raise')
    cv_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        estimator.fit(state['X_train'], state['y_train'])
    fit_seconds = time.perf_counter() - started
    state['models'][name] = estimator
    state['metrics'][name] = {'cv_roc_auc': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)), 'fold_roc_auc': list(map(float, cv_scores)),
        'fit_time_sec': fit_seconds, 'cv_time_sec': cv_seconds, 'search_time_sec': search_seconds}
    print(NAMES[name], '| CV AUC:', round(np.mean(cv_scores), 5), '| fit:', round(fit_seconds, 3), 'с', flush=True)


def train_baselines(state):
    for name, estimator in baseline_models().items():
        train_candidate(state, name, estimator)
    state['baseline_best'] = max(state['metrics'], key=lambda name: state['metrics'][name]['cv_roc_auc'])
    print('Лучшая базовая модель по train-CV:', NAMES[state['baseline_best']])
    return pd.DataFrame(state['metrics']).T


def probability_metrics(truth, probability):
    # Порог 0.5 фиксирован заранее; AUC и AP используют вероятности, а не классы.
    prediction = (np.asarray(probability) >= 0.5).astype(int)
    return {'accuracy': float(accuracy_score(truth, prediction)),
        'precision': float(precision_score(truth, prediction, zero_division=0)),
        'recall': float(recall_score(truth, prediction, zero_division=0)),
        'f1': float(f1_score(truth, prediction, zero_division=0)),
        'roc_auc': float(roc_auc_score(truth, probability)),
        'average_precision': float(average_precision_score(truth, probability)),
        'brier_score': float(brier_score_loss(truth, probability)),
        'log_loss': float(log_loss(truth, probability))}


def evaluate_models(state):
    for name, model in state['models'].items():
        state['metrics'][name].update(probability_metrics(state['y_test'], model.predict_proba(state['X_test'])[:, 1]))
        state['metrics'][name]['train_roc_auc'] = float(roc_auc_score(
            state['y_train'], model.predict_proba(state['X_train'])[:, 1]))
    best = state.get('best_name', state['baseline_best'])
    plt.figure(figsize=(9, 6))
    for name, model in state['models'].items():
        prob = model.predict_proba(state['X_test'])[:, 1]
        fpr, tpr, _ = roc_curve(state['y_test'], prob)
        plt.plot(fpr, tpr, label=f'{NAMES[name]}: {state["metrics"][name]["roc_auc"]:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC-кривые на отложенных клиентах')
    plt.legend(loc='lower right')
    save_figure('roc_curves.png')
    probability = state['models'][best].predict_proba(state['X_test'])[:, 1]
    matrix = confusion_matrix(state['y_test'], probability >= 0.5)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Остался', 'Ушёл'], yticklabels=['Остался', 'Ушёл'])
    plt.xlabel('Предсказано')
    plt.ylabel('Истинно')
    plt.title(f'Матрица ошибок: {NAMES[best]}')
    save_figure('confusion_matrix.png')
    print(classification_report(state['y_test'], probability >= 0.5,
                                target_names=['Остался', 'Ушёл'], digits=4))
    columns = ['cv_roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fit_time_sec']
    table = pd.DataFrame(state['metrics']).T[columns].round(5)
    # Отдельное изображение таблицы удобно приложить при сдаче лабораторной.
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis('off')
    rows = [[NAMES[name]] + [f'{table.loc[name, key]:.4f}' for key in columns] for name in table.index]
    picture = ax.table(cellText=rows, colLabels=['Модель', 'CV AUC', 'Accuracy', 'Precision',
        'Recall', 'F1', 'ROC-AUC', 'Fit, c'], cellLoc='center', loc='center',
        colWidths=[0.24] + [0.108] * 7)
    picture.auto_set_font_size(False)
    picture.set_fontsize(10)
    picture.scale(1, 1.7)
    for (row, col), cell in picture.get_celld().items():
        if row == 0:
            cell.set_facecolor('#dfeaf3')
            cell.set_text_props(weight='bold')
    ax.set_title('Customer Churn: 1409 отложенных клиентов, CV на train', pad=12)
    save_figure('metrics_table.png')
    return table


def feature_importance(state):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, name in zip(axes, ['random_forest', 'lightgbm']):
        # RF показывает impurity importance, LightGBM — gain; шкалы не сравниваем.
        values = pd.Series(state['models'][name].named_steps['model'].feature_importances_, index=FEATURES)
        values.sort_values().tail(15).plot.barh(ax=ax)
        ax.set(title=f'ТОП-15: {NAMES[name]}', xlabel='Важность внутри модели')
        values.sort_values(ascending=False).rename('importance').to_csv(EXAMPLES / f'{name}_importance.csv', index_label='feature')
    save_figure('feature_importance.png')
    return pd.read_csv(EXAMPLES / 'lightgbm_importance.csv').head(10)


def optimize_lightgbm(state, n_trials=30):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        # Пространство поиска задаётся до обращения к holdout.
        depth = trial.suggest_int('max_depth', 3, 8)
        params = {'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
            'max_depth': depth, 'num_leaves': trial.suggest_int('num_leaves', 5, min(63, 2 ** depth)),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.2, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'subsample': trial.suggest_float('subsample', 0.65, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])}
        model = model_pipeline(LGBMClassifier(**params, subsample_freq=1,
            random_state=42, n_jobs=2, verbosity=-1, importance_type='gain'))
        with threadpool_limits(limits=2):
            scores = cross_val_score(model, state['X_train'], state['y_train'], scoring='roc_auc',
                                     cv=state['cv'], n_jobs=1, error_score='raise')
        trial.set_user_attr('fold_roc_auc', list(map(float, scores)))
        print(f'Trial {trial.number + 1}/{n_trials}: CV AUC={scores.mean():.5f}', flush=True)
        return float(scores.mean())
    # Последовательный TPE с seed обеспечивает воспроизводимость последовательности.
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    started = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    elapsed = time.perf_counter() - started
    tuned = model_pipeline(LGBMClassifier(**study.best_params, subsample_freq=1,
        random_state=42, n_jobs=2, verbosity=-1, importance_type='gain'))
    train_candidate(state, 'lightgbm_optuna', tuned,
        cv_scores=study.best_trial.user_attrs['fold_roc_auc'], search_seconds=elapsed)
    state['metrics']['lightgbm_optuna']['cv_time_sec'] = study.best_trial.duration.total_seconds()
    state['study'] = study
    # Из пяти кандидатов выбираем по CV; holdout не входит в условие выбора.
    state['best_name'] = max(state['metrics'], key=lambda name: state['metrics'][name]['cv_roc_auc'])
    state['best_model'] = state['models'][state['best_name']]
    study.trials_dataframe().to_csv(EXAMPLES / 'optuna_trials.csv', index=False)
    write_json(MODELS / 'optuna_study.json', {'n_trials': len(study.trials), 'seed': 42,
        'best_value': study.best_value, 'best_params': study.best_params,
        'best_trial': study.best_trial.number, 'search_time_sec': elapsed})
    values = np.asarray([trial.value for trial in study.trials])
    plt.figure(figsize=(9, 4))
    plt.scatter(np.arange(1, len(values) + 1), values, label='Испытание')
    plt.plot(np.arange(1, len(values) + 1), np.maximum.accumulate(values), label='Лучший CV ROC-AUC')
    plt.xlabel('Испытание Optuna')
    plt.ylabel('Средний ROC-AUC, 5 фолдов')
    plt.legend()
    save_figure('optuna_history.png')
    print('Победитель по train-CV:', NAMES[state['best_name']])
    table = evaluate_models(state)
    # Бонус: калибровка обучается только на train и не меняет основной выбор.
    calibrated = CalibratedClassifierCV(clone(state['best_model']), method='sigmoid', cv=3, ensemble=False, n_jobs=1)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        calibrated.fit(state['X_train'], state['y_train'])
    fit_seconds = time.perf_counter() - started
    calibrated_probability = calibrated.predict_proba(state['X_test'])[:, 1]
    state['calibrated_model'] = calibrated
    state['metrics']['calibrated_bonus'] = {**probability_metrics(state['y_test'], calibrated_probability),
        'fit_time_sec': fit_seconds, 'cv_roc_auc': None, 'threshold': 0.5}
    plt.figure(figsize=(7, 5))
    for label, model in [('Исходная модель', state['best_model']), ('Sigmoid-калибровка', calibrated)]:
        fraction, mean = calibration_curve(state['y_test'], model.predict_proba(state['X_test'])[:, 1], n_bins=8, strategy='quantile')
        plt.plot(mean, fraction, 'o-', label=label)
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('Средняя предсказанная вероятность')
    plt.ylabel('Фактическая доля ушедших')
    plt.legend()
    save_figure('calibration.png')
    print('Brier исходной / калиброванной:', state['metrics'][state['best_name']]['brier_score'],
          state['metrics']['calibrated_bonus']['brier_score'])
    return table


def shap_analysis(state):
    pipeline = state['best_model']
    estimator = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocess']
    probability = pipeline.predict_proba(state['X_test'])[:, 1]
    worst = int(np.argmax(probability))
    # Добавляем клиента с наибольшим риском к фиксированной случайной подвыборке.
    positions = np.unique(np.r_[worst, np.random.default_rng(42).choice(len(probability), 400, replace=False)])
    original = state['X_test'].iloc[positions]
    processed = preprocessor.transform(original)
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(processed)
    expected = explainer.expected_value
    if isinstance(values, list):
        values, expected = values[1], np.asarray(expected)[1]
    elif np.asarray(values).ndim == 3:
        values, expected = np.asarray(values)[:, :, 1], np.asarray(expected)[1]
    values = np.asarray(values)
    base = float(np.asarray(expected).ravel()[0])
    assert values.shape == (len(positions), len(FEATURES))
    # У RF raw-выход — вероятность; у трёх бустингов — логарифм шансов.
    unit = 'probability' if state['best_name'] == 'random_forest' else 'log_odds'
    additive = base + values.sum(axis=1)
    explained_probability = additive if unit == 'probability' else expit(additive)
    assert np.allclose(explained_probability, probability[positions], rtol=1e-4, atol=1e-5)
    display_frame = preprocessor.display_frame(original)
    np.random.seed(42)  # Фиксируем случайное разнесение точек на summary plot.
    shap.summary_plot(values, display_frame, show=False, plot_size=(10, 7))
    plt.title(f'SHAP: {NAMES[state["best_name"]]} ({unit})')
    save_figure('shap_summary.png')
    shap.summary_plot(values, display_frame, plot_type='bar', show=False, plot_size=(10, 7))
    save_figure('shap_bar.png')
    local_position = int(np.flatnonzero(positions == worst)[0])
    explanation = shap.Explanation(values=values[local_position], base_values=base,
        data=processed.iloc[local_position].to_numpy(), feature_names=FEATURES)
    shap.plots.waterfall(explanation, max_display=12, show=False)
    save_figure('shap_waterfall.png')
    ranking = pd.Series(np.abs(values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    shap.dependence_plot(ranking.index[0], values, display_frame, interaction_index=None, show=False)
    save_figure('shap_dependence.png')
    ranking.rename('mean_abs_shap').to_csv(EXAMPLES / 'shap_importance.csv', index_label='feature')
    client_id = state['df'].loc[state['X_test'].index[worst], 'customerID']
    summary = {'model': state['best_name'], 'unit': unit, 'sample_size': len(positions),
        'base_value': base, 'top_features': ranking.head(10).to_dict(),
        'client_id': client_id, 'client_probability': float(probability[worst]),
        'client_features': state['X_test'].iloc[worst].to_dict(),
        'client_contributions': dict(zip(FEATURES, map(float, values[local_position]))),
        'additivity_verified': True}
    write_json(EXAMPLES / 'shap_interpretation.json', summary)
    np.savez_compressed(EXAMPLES / 'shap_values.npz', values=values, positions=positions,
                        base_value=base, probability=probability[positions])
    state['shap'] = summary
    print('SHAP проверен:', unit, '| клиент', client_id, '| риск', round(probability[worst], 4))
    return ranking.head(10)


def save_artifacts(state):
    # Файлы содержат полные Pipeline: повторять предобработку вручную не нужно.
    for name, pipeline in state['models'].items():
        joblib.dump(pipeline, MODELS / FILES[name], compress=3)
    joblib.dump(state['best_model'], MODELS / 'best_model.pkl', compress=3)
    joblib.dump(state['calibrated_model'], MODELS / 'calibrated_model.pkl', compress=3)
    preprocessor = state['best_model'].named_steps['preprocess']
    joblib.dump(preprocessor.label_encoders_, MODELS / 'label_encoders.pkl', compress=3)
    joblib.dump(preprocessor.scaler_, MODELS / 'scaler.pkl', compress=3)
    write_json(MODELS / 'feature_cols.json', FEATURES)
    packages = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'scikit-learn': 'sklearn',
        'xgboost': 'xgboost', 'lightgbm': 'lightgbm', 'catboost': 'catboost', 'optuna': 'optuna',
        'shap': 'shap', 'matplotlib': 'matplotlib', 'seaborn': 'seaborn', 'joblib': 'joblib',
        'requests': 'requests', 'nbformat': 'nbformat', 'nbclient': 'nbclient', 'nbconvert': 'nbconvert',
        'ipykernel': 'ipykernel', 'threadpoolctl': 'threadpoolctl'}
    versions = {name: importlib.import_module(module).__version__ for name, module in packages.items()}
    (ROOT / 'requirements.txt').write_text('\n'.join(f'{name}=={version}' for name, version in versions.items()) + '\n', encoding='utf-8')
    metadata = {'date': '2026-10-26', 'python': platform.python_version(), 'versions': versions,
        'best_model': state['best_name'], 'selection_metric': 'mean_train_cv_roc_auc',
        'selection_candidates': list(state['models']), 'seed': 42, 'n_train': len(state['X_train']),
        'n_test': len(state['X_test']), 'cv_folds': 5, 'optuna_trials': len(state['study'].trials),
        'threshold': 0.5, 'positive_class': 'Yes (ушёл)', 'scaler_applied': False,
        'native_catboost_categories': True, 'source_sha256': hashlib.sha256((DATA / 'telco_churn.csv').read_bytes()).hexdigest(),
        'train_ids': state['df'].loc[state['train_index'], 'customerID'].tolist(),
        'test_ids': state['df'].loc[state['test_index'], 'customerID'].tolist(),
        'preprocessing': {'numeric_medians': preprocessor.numeric_medians_,
            'label_classes': {name: encoder.classes_.tolist() for name, encoder in preprocessor.label_encoders_.items()},
            'service_count_rule': '8 полей Yes + InternetService в {DSL, Fiber optic}'}}
    write_json(MODELS / 'metadata.json', metadata)
    write_json(MODELS / 'metrics.json', state['metrics'])
    # Таблица вероятностей позволяет независимо пересчитать все метрики.
    predictions = pd.DataFrame({'customerID': metadata['test_ids'], 'Churn': state['y_test'].to_numpy()})
    for name, pipeline in {**state['models'], 'calibrated_bonus': state['calibrated_model']}.items():
        predictions[name] = pipeline.predict_proba(state['X_test'])[:, 1]
    predictions.to_csv(EXAMPLES / 'holdout_predictions.csv', index=False)
    demo = state['df'].loc[state['test_index'][:3]].drop(columns='Churn')
    write_json(EXAMPLES / 'demo_clients.json', json.loads(demo.to_json(orient='records')))
    loaded = joblib.load(MODELS / 'best_model.pkl')
    assert np.allclose(loaded.predict_proba(state['X_test']), state['best_model'].predict_proba(state['X_test']), rtol=1e-12)
    result = predict_churn(demo, loaded)
    result.insert(0, 'customerID', demo.customerID.to_numpy())
    result.to_csv(EXAMPLES / 'demo_predictions.csv', index=False)
    print('Сохранены 5 кандидатов, лучшая и калиброванная модели. Локальные вероятности совпали.')
    return result


def check_github(state):
    # При первом запуске модуль ещё не опубликован; этот факт выводим честно.
    ref = os.environ.get('HOUSE_CHURN_GITHUB_REF', 'main')
    base = f'https://raw.githubusercontent.com/compl1ment/ml-course-musaev/{ref}/module-3-churn'
    status = {'github_verified': False, 'remote_ref': ref, 'source': 'local'}
    if os.environ.get('HOUSE_CHURN_CHECK_GITHUB', '1') == '1':
        try:
            response = requests.get(f'{base}/models/best_model.pkl', timeout=30)
            response.raise_for_status()
            loaded = joblib.load(BytesIO(response.content))
            status.update(source='GitHub raw', github_verified=True)
        except requests.RequestException as exc:
            status['reason'] = str(exc)
            loaded = joblib.load(MODELS / 'best_model.pkl')
    else:
        loaded = joblib.load(MODELS / 'best_model.pkl')
        status['reason'] = 'network_check_disabled'
    clients = state['X_test'].iloc[:3]
    before = state['best_model'].predict_proba(clients)[:, 1]
    after = loaded.predict_proba(clients)[:, 1]
    assert np.allclose(before, after, rtol=1e-10, atol=1e-12)
    status.update(predictions_matched=3, probabilities=list(map(float, after)))
    write_json(MODELS / 'repository_verification.json', status)
    print('Источник:', status['source'], '| совпало 3 из 3:', after)
    if not status['github_verified']:
        print('Сетевая проверка ещё не подтверждена:', status.get('reason'))
    return status
