from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator

from feature_selection.core.base import BaseFeatureSelector
from feature_selection.schemas.history import FeatureSet, History, Iteration


class DFSFeatureSelector(BaseFeatureSelector):

    def __init__(
            self,
            estimator: BaseEstimator,
            scoring: str | Callable,
            cv: int,
            max_depth: int,
            max_steps: int = 100,
            pruning_threshold: float = 0.8,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv
        )
        self.max_depth = max_depth
        self.pruning_threshold = pruning_threshold
        self.max_steps = max_steps
        self.history = History()

    def fit(self, X: np.ndarray, y: np.ndarray):

        dim = X.shape[1]
        self.history = History()
        self.selected_features = FeatureSet(
            features=frozenset(),
            score=-np.inf,
            original_feature_num=dim
        )
        best_score = -np.inf
        stack = [(frozenset(), 0)]

        step = 0
        cache = {}

        while stack:
            current_set, current_depth = stack.pop()

            mask = np.zeros(dim, dtype=bool)
            mask[list(current_set)] = True
            score = self._estimate_algorithm_on_subset(
                mask,
                X,
                y,
                cache
            )
            if (score != float('-inf') and
                    (current_depth > self.max_depth or
                     score < best_score * self.pruning_threshold or
                     step > self.max_steps)
            ):
                continue


            feature_subset = FeatureSet(
                features=current_set,
                score=score,
                original_feature_num=dim
            )

            if score > best_score:
                best_score = score
                self.selected_features = feature_subset

            if step != 0:
                # skip first step with empty set
                iteration_data = Iteration(
                    candidate_features=[feature_subset],
                    selected_features=self.selected_features,
                    iteration=step,
                )
                self.history.add_iteration(iteration_data)

            for feature_index in range(dim):
                if not mask[feature_index]:
                    new_subset = current_set | {feature_index}
                    stack.append((new_subset, current_depth + 1))

            step += 1

        return self
