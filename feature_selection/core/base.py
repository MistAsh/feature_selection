from abc import ABC, abstractmethod
from typing import Callable, Tuple

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, KFold

from feature_selection.schemas.history import *


class BaseFeatureSelector(ABC):

    def __init__(self,
                 estimator: BaseEstimator,
                 scoring: str | Callable,
                 cv: int | KFold=None,
                 ):

        self.selected_features: FeatureSet | None = None
        self.history: History = History()
        self.estimator = estimator
        self.scoring = scoring
        if cv is None:
            cv = 5
        self.cv = cv


    @staticmethod
    def _set_to_mask(
            feature_set: set[int] | frozenset[int],
            dim: int
    ) -> np.ndarray:
        mask = np.zeros(dim, dtype=bool)
        mask[list(feature_set)] = True
        return mask

    @abstractmethod
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        pass

    def transform(
            self,
            X: np.ndarray,
            y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.selected_features is None:
            raise ValueError("Must call fit first")
        return X[:, self.selected_features.to_list()], y

    def _estimate_algorithm_on_subset(
            self,
            features_subset_mask: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            cache: dict
    ):
        key = tuple(np.where(features_subset_mask)[0])

        if key in cache:
            return cache[key]

        if np.sum(features_subset_mask) == 0:
            return -float('inf')

        scores = cross_val_score(
            estimator=self.estimator,
            X=X[:, features_subset_mask],
            y=y,
            scoring=self.scoring,
            cv=self.cv,
        )
        score = scores.mean()
        cache[key] = score

        return score
