from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, precision_score


class SentimentEvaluator:

    CLASS_NAMES = ("Negative", "Neutral", "Positive")

    def __init__(
        self,
        reference_scores: Sequence[float],
        reference_label: str,
        comparisons: Mapping[str, Sequence[float]] | None = None,
        *,
        round_classes: bool = False,
    ) -> None:
        self.reference_label = self._validate_label(reference_label)
        self.reference_scores = self._as_scores(reference_scores, "reference_scores")
        if not isinstance(round_classes, bool):
            raise TypeError("round_classes must be a boolean")
        self.round_classes = round_classes
        self.comparisons: dict[str, np.ndarray] = {}

        if comparisons is not None:
            if not isinstance(comparisons, Mapping):
                raise TypeError("comparisons must map labels to score lists")
            for label, scores in comparisons.items():
                self.add_comparison(scores, label)

    @staticmethod
    def _validate_label(label: str) -> str:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Each list must have a nonempty string label")
        return label

    @staticmethod
    def _as_scores(scores: Sequence[float], name: str) -> np.ndarray:
        try:
            values = np.asarray(scores, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric scores") from exc
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} must be a nonempty, one-dimensional list")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains a NaN or infinite score")
        return values.copy()

    def add_comparison(self, scores: Sequence[float], label: str) -> None:
        label = self._validate_label(label)
        if label == self.reference_label:
            raise ValueError("Comparison label must differ from reference label")
        if label in self.comparisons:
            raise ValueError(f"Comparison label {label!r} is already registered")
        values = self._as_scores(scores, label)
        if len(values) != len(self.reference_scores):
            raise ValueError(
                f"{label!r} has {len(values)} scores, but the reference has "
                f"{len(self.reference_scores)}. Lists must be aligned."
            )
        self.comparisons[label] = values

    @staticmethod
    def accuracy_within_one(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(np.abs(y_true - y_pred) <= 1))

    @staticmethod
    def precision_within_one(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """macro tolerance precision over predicted values.
        For continuous predictions this treats each distinct float as a value;
        it is not conventional nominal-class precision.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        class_precisions = []
        for k in np.unique(y_pred):
            predicted_as_k = y_pred == k
            precision_k = np.mean(np.abs(y_true[predicted_as_k] - k) <= 1)
            class_precisions.append(precision_k)
        return float(np.mean(class_precisions))

    @staticmethod
    def sentiment_classes(scores: Sequence[float] | np.ndarray) -> np.ndarray:
        """Convert 1-10 continuous scores to negative(0), neutral(1), positive(2).

        First round half UP to the nearest integer, because mean annotator
        scores may be fractional. Then map 1-3 -> 0, 4-6 -> 1, 7-10 -> 2.
        Unlike np.rint, 6.5 is rounded to 7, not 6.
        """
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all() or np.any((scores < 1) | (scores > 10)):
            raise ValueError("Three-class sentiment metrics require scores in [1, 10]")
        rounded = np.floor(scores + 0.5).astype(int)
        return np.where(rounded <= 3, 0, np.where(rounded <= 6, 1, 2))

    @staticmethod
    def _safe_correlation(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
        """Return NaN when correlation is undefined (n<2 or constant vector)."""
        if (len(y_true) < 2 or np.all(y_true == y_true[0]) or
                np.all(y_pred == y_pred[0])):
            return float("nan")
        if method == "kendall":
            value = kendalltau(y_true, y_pred).statistic
        elif method == "spearman":
            value = spearmanr(y_true, y_pred).statistic
        elif method == "pearson":
            value = pearsonr(y_true, y_pred).statistic
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        return float(value)

    @classmethod
    def _three_class_metrics(
        cls, y_true: np.ndarray, y_pred: np.ndarray, suffix: str = ""
    ) -> dict[str, float]:
        """Macro and per-class precision plus exact 3-class accuracy.

        All three labels (0/1/2) are included in macro precision; classes
        with no predicted observations receive precision 0 (zero_division=0).
        """
        if len(y_true) == 0:
            return {
                f"Sentiment precision (macro){suffix}": float("nan"),
                **{
                    f"Precision {name}{suffix}": float("nan")
                    for name in cls.CLASS_NAMES
                },
                f"Sentiment accuracy (3-class){suffix}": float("nan"),
            }
        true_classes = cls.sentiment_classes(y_true)
        pred_classes = cls.sentiment_classes(y_pred)
        per_class = precision_score(
            true_classes, pred_classes, labels=[0, 1, 2],
            average=None, zero_division=0,
        )
        return {
            f"Sentiment precision (macro){suffix}": float(np.mean(per_class)),
            **{
                f"Precision {name}{suffix}": float(value)
                for name, value in zip(cls.CLASS_NAMES, per_class)
            },
            f"Sentiment accuracy (3-class){suffix}": float(
                np.mean(true_classes == pred_classes)
            ),
        }

    @staticmethod
    def linear_scaling_relative(
        input_scores: Sequence[float], new_range: tuple[float, float] = (1, 10)
    ) -> list[float]:
        """Optional min-max scaling helper. Scaling is never automatic."""
        values = SentimentEvaluator._as_scores(input_scores, "input_scores")
        old_min, old_max = values.min(), values.max()
        if old_min == old_max:
            raise ValueError("Cannot scale because all input values are the same.")
        return np.interp(values, (old_min, old_max), new_range).tolist()

    def compare(self) -> pd.DataFrame:
        """Return one row per comparison, with full and non-neutral metrics.

        The non-neutral subset retains the ORIGINAL definition: reference
        scores <4 or >6, irrespective of the three-class rounding convention.
        """
        if not self.comparisons:
            raise ValueError("No comparison lists registered; call add_comparison()")

        reference_for_tolerance = (
            np.rint(self.reference_scores)
            if self.round_classes else self.reference_scores
        )
        non_neutral_mask = (
            (self.reference_scores < 4) | (self.reference_scores > 6)
        )
        n_non_neutral = int(non_neutral_mask.sum())
        y_non = self.reference_scores[non_neutral_mask]
        y_non_tolerance = reference_for_tolerance[non_neutral_mask]
        rows: list[dict] = []

        for label, predictions in self.comparisons.items():
            predictions_for_tolerance = (
                np.rint(predictions) if self.round_classes else predictions
            )
            row = {
                "Reference": self.reference_label,
                "Model": label,
                "N": len(self.reference_scores),
                "N (non-neutral)": n_non_neutral,
                "MAE": float(mean_absolute_error(self.reference_scores, predictions)),
                "Kendall's tau": self._safe_correlation(
                    self.reference_scores, predictions, "kendall"
                ),
                "Spearman's rho": self._safe_correlation(
                    self.reference_scores, predictions, "spearman"
                ),
                "Pearson's r": self._safe_correlation(
                    self.reference_scores, predictions, "pearson"
                ),
                "Precision (±1)": self.precision_within_one(
                    reference_for_tolerance, predictions_for_tolerance
                ),
                "Accuracy (±1)": self.accuracy_within_one(
                    reference_for_tolerance, predictions_for_tolerance
                ),
            }
            row.update(self._three_class_metrics(self.reference_scores, predictions))

            if n_non_neutral:
                pred_non = predictions[non_neutral_mask]
                pred_non_tolerance = predictions_for_tolerance[non_neutral_mask]
                row.update({
                    "MAE (non-neutral)": float(mean_absolute_error(y_non, pred_non)),
                    "Spearman's rho (non-neutral)": self._safe_correlation(
                        y_non, pred_non, "spearman"
                    ),
                    "Pearson's r (non-neutral)": self._safe_correlation(
                        y_non, pred_non, "pearson"
                    ),
                    "Precision (±1, non-neutral)": self.precision_within_one(
                        y_non_tolerance, pred_non_tolerance
                    ),
                    "Accuracy (±1, non-neutral)": self.accuracy_within_one(
                        y_non_tolerance, pred_non_tolerance
                    ),
                })
                row.update(self._three_class_metrics(y_non, pred_non, " (non-neutral)"))
            else:
                row.update({
                    "MAE (non-neutral)": float("nan"),
                    "Spearman's rho (non-neutral)": float("nan"),
                    "Pearson's r (non-neutral)": float("nan"),
                    "Precision (±1, non-neutral)": float("nan"),
                    "Accuracy (±1, non-neutral)": float("nan"),
                })
                row.update(self._three_class_metrics(y_non, y_non, " (non-neutral)"))
            rows.append(row)

        return pd.DataFrame(rows)

    def save_table(self, path: str | Path) -> Path:
        """Write the comparison table to a .csv or .json file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = self.compare()
        if path.suffix.lower() == ".csv":
            table.to_csv(path, index=False)
        elif path.suffix.lower() == ".json":
            table.to_json(path, orient="records", indent=2, force_ascii=False)
        else:
            raise ValueError("Use a .csv or .json filename")
        return path
