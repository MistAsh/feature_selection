# from feature_selection.core import (
#     FullSearch, BeamSearchFeatureSelector,
#     ForwardFeatureSelector, BackwardFeatureSelector,
#     ForwardBackwardFeatureSelector, GeneticFeatureSelector,
#     AdaptiveMonteCarloFeatureSelector, DFSFeatureSelector
# )
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.svm import SVR, SVC
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import mean_squared_error, accuracy_score, make_scorer
# import numpy as np
#
# # Импортируем функции для генерации тестовых данных
# from .test_data import make_synthetic_regression_data, make_synthetic_classification_data
#
# # Список тестовых случаев для параметризации
# # Каждый случай - это кортеж: (КлассСелектора, параметры_инициализации, данные_X, данные_y, ожидаемые_свойства)
# # Ожидаемые свойства могут включать 'expected_features_count' или 'expected_informative_features_selected'
# test_cases = [
#     # FullSearch (регрессия)
#     (
#         FullSearch,
#         {'estimator': LinearRegression(), 'scoring': 'neg_mean_squared_error', 'n_splits': 3},
#         *make_synthetic_regression_data(n_samples=50, n_features=5, n_informative=2, random_state=42)[:2], # Берем только X, y
#         {'expected_features_count': 2} # FullSearch находит оптимальное подмножество
#     ),
#     # FullSearch (классификация)
#     (
#         FullSearch,
#         {'estimator': LogisticRegression(solver='liblinear'), 'scoring': 'accuracy', 'n_splits': 3},
#         *make_synthetic_classification_data(n_samples=50, n_features=5, n_informative=2, random_state=42),
#         {} # Для классификации с make_classification сложно предсказать точное кол-во
#     ),
#     # ForwardFeatureSelector (регрессия)
#     (
#         ForwardFeatureSelector,
#         {'estimator': LinearRegression(), 'scoring': 'neg_mean_squared_error', 'n_splits': 3, 'max_iterations_without_improvement': 2},
#         *make_synthetic_regression_data(n_samples=100, n_features=10, n_informative=3, random_state=42)[:2],
#         {}
#     ),
#     # BackwardFeatureSelector (классификация)
#     (
#         BackwardFeatureSelector,
#         {'estimator': LogisticRegression(solver='liblinear'), 'scoring': 'accuracy', 'n_splits': 3, 'max_iterations_without_improvement': 2},
#         *make_synthetic_classification_data(n_samples=100, n_features=10, n_informative=3, random_state=42),
#         {}
#     ),
#     # BeamSearchFeatureSelector (регрессия)
#     (
#         BeamSearchFeatureSelector,
#         {'estimator': SVR(), 'scoring': 'neg_mean_squared_error', 'n_splits': 3, 'beam_width': 2, 'max_feature_size': 5},
#         *make_synthetic_regression_data(n_samples=100, n_features=10, n_informative=3, random_state=42)[:2],
#         {}
#     ),
#     # GeneticFeatureSelector (классификация)
#     # Примечание: Для GeneticFeatureSelector требуется fitness_function.
#     # Мы можем использовать make_scorer из sklearn.metrics
#     (
#         GeneticFeatureSelector,
#         {'estimator': SVC(gamma='auto'), 'scoring': 'accuracy', 'n_splits': 3, 'fitness_function': make_scorer(accuracy_score), 'population_size': 10, 'max_iterations': 5},
#         *make_synthetic_classification_data(n_samples=100, n_features=10, n_informative=3, random_state=42),
#         {}
#     ),
#     # AdaptiveMonteCarloFeatureSelector (регрессия)
#     (
#         AdaptiveMonteCarloFeatureSelector,
#         {'estimator': RandomForestRegressor(n_estimators=10), 'scoring': 'neg_mean_squared_error', 'n_splits': 3, 'max_steps': 10, 'learning_rate': 0.1},
#         *make_synthetic_regression_data(n_samples=100, n_features=10, n_informative=3, random_state=42)[:2],
#         {}
#     ),
#      # DFSFeatureSelector (классификация)
#     (
#         DFSFeatureSelector,
#         {'estimator': RandomForestClassifier(n_estimators=10), 'scoring': 'accuracy', 'n_splits': 3, 'max_depth': 3, 'max_steps': 20},
#         *make_synthetic_classification_data(n_samples=100, n_features=10, n_informative=3, random_state=42),
#         {}
#     ),
# ]
