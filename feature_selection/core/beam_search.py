from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator

from feature_selection.core.base import BaseFeatureSelector
from feature_selection.schemas.history import FeatureSet, History, Iteration


class BeamSearchFeatureSelector(BaseFeatureSelector):

    def __init__(
            self,
            estimator: BaseEstimator,
            scoring: str | Callable,
            cv: int,
            beam_width: int,
            max_feature_size: int,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv
        )
        self.beam_width = beam_width
        self.max_feature_size = max_feature_size
        self.history = History()

    def fit(self, X: np.ndarray, y: np.ndarray):

        dim = X.shape[1]

        self.history = History()

        features = set([i for i in range(dim)])
        best_score = -np.inf
        current_sets = [frozenset()]
        cache = {}

        for step in range(1, self.max_feature_size + 1):
            candidates = []
            for subset in current_sets:
                for feature in features - subset:
                    new_subset = subset | {feature}
                    mask = self._set_to_mask(new_subset, dim)
                    score = self._estimate_algorithm_on_subset(
                        mask,
                        X,
                        y,
                        cache
                    )
                    feature_set = FeatureSet(
                        features=new_subset,
                        score=score,
                        original_feature_num=dim
                    )
                    candidates.append(feature_set)

            candidates = sorted(
                candidates,
                key=lambda c: c.score,
                reverse=True
            )[:self.beam_width]

            if candidates[0].score > best_score:
                best_score = candidates[0].score
                self.selected_features = candidates[0]

            iteration_data = Iteration(
                candidate_features=candidates,
                selected_features=self.selected_features,
                iteration=step,
            )

            current_sets = [c.features for c in candidates]

            self.history.add_iteration(iteration_data)


        return self
