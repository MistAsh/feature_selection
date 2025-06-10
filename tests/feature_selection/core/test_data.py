import numpy as np
from sklearn.datasets import make_regression, make_classification
from feature_selection.core import (
    AdaptiveMonteCarloFeatureSelector,
    FullSearch,
    BeamSearchFeatureSelector,
    DFSFeatureSelector,
    GeneticFeatureSelector,
    ForwardBackwardFeatureSelector,
    BackwardFeatureSelector,
    ForwardFeatureSelector
) # Предполагается, что ваш модуль feature_selection.core доступен
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

def make_synthetic_regression_data(n_samples=100, n_features=10, n_informative=3, random_state=42):
    """Generates synthetic regression data with a specified number of informative features."""
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=10.0,
        coef=True,
        random_state=random_state
    )
    # Return X, y, and the mask of informative features
    return X, y

def make_synthetic_classification_data(n_samples=100, n_features=10, n_informative=3, n_classes=2, random_state=42):
    """Generates synthetic classification data with a specified number of informative features."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.01,
        random_state=random_state
    )
    # Note: make_classification doesn't easily give us the *exact* informative feature mask
    # We'll assume the first n_informative features are the most relevant for testing purposes
    # For more robust tests, one might need a custom data generator.
    # For now, we'll just return X and y.
    return X, y


# Базовые параметры для кросс-валидации и оценки модели
BASE_PARAMS = [
    {
        'description': 'Regression with LinearRegression',
        'params': {
        'estimator': LinearRegression(),
        'scoring': 'neg_mean_squared_error',
        'n_splits': 3,
        },
        'data_generator': make_synthetic_regression_data,
        'data_type': 'regression'
    },
    {
        'description': 'Classification with DecisionTreeClassifier',
        'params': {
        'estimator': DecisionTreeClassifier(random_state=42),
        'scoring': 'accuracy',
        'n_splits': 3,
        },
        'data_generator': make_synthetic_classification_data,
        'data_type': 'classification'
    }
]

# Специфичные параметры для каждого алгоритма выбора признаков
ALGORITHM_PARAMS = [
    {
        'name': 'AdaptiveSearch',
        'class': AdaptiveMonteCarloFeatureSelector,
        'params': {'max_steps': 100},
    },
    {
        'name': 'BeamSearch',
        'class': BeamSearchFeatureSelector,
        'params': {'beam_width': 3, 'max_feature_size': 5},
    },
    {
        'name': 'DFS',
        'class': DFSFeatureSelector,
        'params': {'max_depth': 3},
    },
    {
        'name': 'FullSearch',
        'class': FullSearch,
        'params': {},
    },
    {
        'name': 'Genetic',
        'class': GeneticFeatureSelector,
        'params': {
            'population_size': 10,
            'max_iterations': 5,
            'mutation_probability': 0.1,
            'crossover_method': 'random',
            'selection_method': 'elitism',
            'fitness_function': lambda x, _: x
        },
    },
    {
        'name': 'ForwardStepwise',
        'class': ForwardFeatureSelector,
        'params': {},
    },
    {
        'name': 'BackwardStepwise',
        'class': BackwardFeatureSelector,
        'params': {},
    },
    {
        'name': 'ForwardBackwardStepwise',
        'class': ForwardBackwardFeatureSelector,
        'params': {},
    },
]

# Собираем все тестовые случаи
TOTAL_TEST_CASES = []

for base_case in BASE_PARAMS:

    X, y = base_case['data_generator']()
    for algo_case in ALGORITHM_PARAMS:
        selectors_params = {**base_case['params'], **algo_case['params']}
        test_case = {
            'test_name': f"{base_case['description']} - {algo_case['name']}",
            'selector_class': algo_case['class'],
            'selector_params': selectors_params,
            'fit_data': {'X': X, 'y': y}, # Данные для fit
            'data_type': base_case['data_type']
        }
        TOTAL_TEST_CASES.append(test_case)