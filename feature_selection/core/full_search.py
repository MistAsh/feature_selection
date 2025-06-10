from feature_selection.core.base import BaseFeatureSelector
from feature_selection.schemas.history import *


class FullSearch(BaseFeatureSelector):

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.history = History()
        self.selected_features = None
        cache = {}
        features_num = X.shape[1]
        best_score = -np.inf


        for i in range(1, 1 << features_num):
            feature_mask = np.zeros(features_num, dtype=bool)
            for j in range(features_num):
                if i & (1 << j):
                    feature_mask[j] = True

            score = self._estimate_algorithm_on_subset(
                feature_mask,
                X,
                y,
                cache
            )
            features_set = set(
                np.where(feature_mask)[0].tolist(),
            )
            features_set = FeatureSet(
                features=features_set,
                score=score,
                original_feature_num=features_num,
            )
            if score > best_score:
                best_score = score
                self.selected_features = features_set

            iteration_data = Iteration(
                iteration=i,
                candidate_features=[features_set],
                selected_features=self.selected_features
            )

            self.history.add_iteration(iteration_data)

        return self
