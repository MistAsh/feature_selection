from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator

from feature_selection.core.base import BaseFeatureSelector
from feature_selection.schemas.history import FeatureSet, History, Iteration


class AdaptiveMonteCarloFeatureSelector(BaseFeatureSelector):

    def __init__(self,
                 estimator: BaseEstimator,
                 scoring: str | Callable,
                 cv: int,
                 max_steps: int,
                 learning_rate: float = 0.05,
                 min_features: int = 1
                 ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            cv=cv
        )
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.min_features = min_features
        self.probs = None


    def fit(self, X: np.ndarray, y: np.ndarray):
        dim = X.shape[1]
        self.probs = np.ones(dim) * 0.5
        self.history = History()

        feature_scores = np.zeros(dim)
        n_evaluations = np.zeros(dim)
        total_score = 0.0
        n_subsets = 0
        cache = {}
        best_score = -np.inf

        for step in range(self.max_steps):
            subset_mask = self._generate_feature_mask()
            n_features = np.sum(subset_mask)

            if n_features < self.min_features:
                continue

            score = self._estimate_algorithm_on_subset(
                features_subset_mask=subset_mask,
                X=X,
                y=y,
                cache=cache
            )

            feature_scores += subset_mask * score
            n_evaluations += subset_mask
            total_score += score
            n_subsets += 1

            avg_total_score = total_score / n_subsets
            avg_feature_score = np.divide(
                feature_scores, n_evaluations,
                out=np.zeros_like(feature_scores),
                where=n_evaluations != 0
            )

            delta = avg_feature_score - avg_total_score
            self.probs += self.learning_rate * delta
            self.probs = np.clip(self.probs, 0.05, 0.95)

            current_feature_set = FeatureSet(
                features=set(np.flatnonzero(subset_mask).tolist()),
                score=score,
                original_feature_num=dim
            )

            if score > best_score:
                best_score = score
                self.selected_features = current_feature_set

            self.history.add_iteration(Iteration(
                iteration=step,
                candidate_features=[current_feature_set],
                selected_features=self.selected_features,
                metadata={
                    'feature_scores': feature_scores.copy(),
                    'n_evaluations': n_evaluations.copy(),
                    'avg_total_score': avg_total_score,
                    'probs': self.probs.copy(),
                    'delta': delta.copy()
                }
            ))

        return self

    def _generate_feature_mask(self) -> np.ndarray:
        if np.sum(self.probs) == 0:
            raise Exception('Probs array is invalid')

        mask = np.random.random(len(self.probs)) < self.probs
        while np.sum(mask) == 0:
            mask = np.random.random(len(self.probs)) < self.probs

        return mask