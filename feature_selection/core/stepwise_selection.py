from abc import ABC
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator

from feature_selection.core.base import BaseFeatureSelector
from feature_selection.schemas.history import FeatureSet, History, Iteration


class StepwiseFeatureSelector(BaseFeatureSelector, ABC):

    def __init__(
            self,
            estimator: BaseEstimator,
            scoring: str | Callable,
            cv: int,
            max_iterations_without_improvement: int = 3,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv
        )
        self.max_iterations_without_improvement = (
            max_iterations_without_improvement)

    def _add_step(self, current_features, X, y, cache):

        dim = X.shape[1]
        features = set(range(dim))

        best_subset = None
        best_score = -np.inf

        for feature in features - current_features:
            new_subset = current_features | {feature}

            mask = self._set_to_mask(new_subset, dim)
            current_score = self._estimate_algorithm_on_subset(
                mask,
                X,
                y,
                cache
            )
            if current_score > best_score:
                best_score = current_score
                best_subset = new_subset

        return best_subset, best_score, cache

    def _del_step(self, current_features, X, y, cache):

        dim = X.shape[1]

        best_subset = None
        best_score = -np.inf

        for feature in current_features:
            new_subset = current_features - {feature}
            mask = self._set_to_mask(new_subset, dim)
            current_score = self._estimate_algorithm_on_subset(
                mask,
                X,
                y,
                cache
            )
            if current_score > best_score:
                best_score = current_score
                best_subset = new_subset

        return best_subset, best_score, cache


class ForwardFeatureSelector(StepwiseFeatureSelector):

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.history = History()

        dim = X.shape[1]
        features = set(range(dim))
        current_features = set()

        best_score = -np.inf
        iterations_without_improvement = 0
        iterations = 1
        cache = {}
        while (
                len(current_features) < len(features) and
                iterations_without_improvement <
                self.max_iterations_without_improvement
        ):
            current_features, score, cache = self._add_step(
                current_features,
                X,
                y,
                cache
            )
            current_feature_set = FeatureSet(
                features=current_features,
                score=score,
                original_feature_num=dim
            )

            if score > best_score:
                best_score = score
                self.selected_features = current_feature_set
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            iteration_data = Iteration(
                iteration=iterations,
                candidate_features=[current_feature_set],
                selected_features=self.selected_features,
            )
            self.history.add_iteration(iteration_data)

            iterations += 1

        return self


class BackwardFeatureSelector(StepwiseFeatureSelector):

    def fit(self, X: np.ndarray, y: np.ndarray):

        dim = X.shape[1]
        features = set(range(dim))
        current_features = features.copy()

        self.history = History()
        cache = {}

        best_score = self._estimate_algorithm_on_subset(
            self._set_to_mask(current_features, dim),
            X,
            y,
            cache
        )

        initial_iteration = FeatureSet(
            features=current_features,
            score=best_score,
            original_feature_num=dim
        )
        self.selected_features = initial_iteration

        self.history.add_iteration(Iteration(
            candidate_features=[initial_iteration],
            iteration=1,
            selected_features=self.selected_features,
        ))

        iterations_without_improvement = 0
        iterations = 2

        while (
                len(current_features) > 1 and
                iterations_without_improvement <
                self.max_iterations_without_improvement
        ):
            current_features, score, cache = self._del_step(
                current_features,
                X,
                y,
                cache
            )
            current_feature_set = FeatureSet(
                features=current_features,
                score=score,
                original_feature_num=dim
            )

            if score > best_score:
                best_score = score
                self.selected_features = current_feature_set
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            iteration_data = Iteration(
                candidate_features=[current_feature_set],
                selected_features=self.selected_features,
                iteration=iterations,
            )
            self.history.add_iteration(iteration_data)

            iterations += 1

        return self


class ForwardBackwardFeatureSelector(StepwiseFeatureSelector):
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.history = History()
        cache = {}
        dim = X.shape[1]
        features = set(range(dim))
        current_features = set()

        best_score = -np.inf
        iterations_without_improvement = 0
        iterations = 1

        while iterations_without_improvement < self.max_iterations_without_improvement:
            add_score = -np.inf
            del_score = -np.inf
            features_after_add = None
            feature_after_del = None
            feature_set_del = None
            feature_set_add = None

            if len(current_features) < len(features):
                features_after_add, add_score, cache = self._add_step(
                    current_features.copy(),
                    X,
                    y,
                    cache
                )
                feature_set_add = FeatureSet(
                    features=features_after_add,
                    score=add_score,
                    original_feature_num=dim
                )



            if len(current_features) > 1:
                feature_after_del, del_score, cache = self._del_step(
                    current_features.copy(),
                    X,
                    y,
                    cache
                )
                feature_set_del = FeatureSet(
                    features=feature_after_del,
                    score=del_score,
                    original_feature_num=dim
                )

            if add_score > del_score:
                current_features = features_after_add
                score = add_score
            else:
                current_features = feature_after_del
                score = del_score


            if score > best_score:
                best_score = score
                self.selected_features = FeatureSet(
                    features=current_features,
                    score=score,
                    original_feature_num=dim
                )
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            filtered_candidates = [
                i for i in [feature_set_add, feature_set_del]
                if i is not None
            ]

            iteration_data = Iteration(
                candidate_features=filtered_candidates,
                iteration=iterations,
                selected_features=self.selected_features,
            )
            self.history.add_iteration(iteration_data)

            iterations += 1

        return self
