from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class FeatureSet:
    features: set[int] | frozenset[int]
    score: float
    original_feature_num: int

    def to_list(self) -> list[int]:
        return list(self.features)

    def to_mask(self) -> np.ndarray:
        mask = np.zeros(self.original_feature_num, dtype=bool)
        mask[self.to_list()] = True
        return mask

@dataclass
class Iteration:
    """
    Представляет одну итерацию в процессе оптимизации/поиска.
    """
    iteration: int
    candidate_features: list[FeatureSet]
    selected_features: FeatureSet | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class History:
    iterations: list[Iteration] = field(default_factory=list)

    def add_iteration(self, iteration_data: Iteration) -> None:
        self.iterations.append(iteration_data)

    def get_latest_iteration(self) -> Iteration | None:
        if not self.iterations:
            return None
        return self.iterations[-1]

    def get_best_overall_iteration(self) -> Iteration | None:
        if not self.iterations:
            return None
        return max(
            self.iterations,
            key=lambda i: i.selected_features.score
        )

    def __len__(self) -> int:
        return len(self.iterations)

    def __getitem__(self, index: int) -> Iteration:
        return self.iterations[index]

    def get_all_candidate_features(self) -> list[FeatureSet]:
        candidate_features = []
        for iteration in self.iterations:
            candidate_features.extend(iteration.candidate_features)
        return candidate_features