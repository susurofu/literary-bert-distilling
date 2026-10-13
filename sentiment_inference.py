from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from transformers import AutoTokenizer

# Reuse the original ordinal head: loading with a generic BERT classifier
# would replace the trained thresholds and produce incorrect predictions.
from model_trainer import BertForOrdinalRegression


class OrdinalBertInference:
    """Predict 1-based ordinal sentiment labels from a saved BERT model.

    Parameters
    ----------
    model_path : str | Path
        Local directory containing a saved model and tokenizer.
    max_len : int, default 256
        Maximum number of tokens, matching the original training default.
    batch_size : int, default 32
        Number of sentences processed at once.
    device : str | torch.device | None
        Torch device ("cuda", "mps", or "cpu"). Auto-select if omitted.

    Examples
    --------
    >>> predictor = OrdinalBertInference("outputs/bert_ordinal/best_model")
    >>> labels = predictor.predict(
    ...     ["A wonderful day!", "A terrible experience."],
    ...     output_file="results/sentiment_predictions.json",
    ... )
    """

    def __init__(
        self,
        model_path: str | Path,
        max_len: int = 256,
        batch_size: int = 32,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        if max_len < 1:
            raise ValueError("max_len must be a positive integer")
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        self.max_len = max_len
        self.batch_size = batch_size
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = BertForOrdinalRegression.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        self.num_classes = self.model.config.num_labels

    def predict(
        self,
        sentences: list[str],
        output_file: str | Path | None = None,
    ) -> list[int]:
        """Return labels in input order; optionally write sentence/label JSON.
        """
        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list of strings")
        if any(not isinstance(s, str) for s in sentences):
            raise TypeError("Every sentence must be a string")
        if any(not s.strip() for s in sentences):
            raise ValueError("Sentences must not be empty or whitespace-only")

        labels: list[int] = []
        for start in range(0, len(sentences), self.batch_size):
            batch = sentences[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.inference_mode():
                logits = self.model(**encoded).logits.detach().cpu().numpy()

            # Preserve exactly the original evaluation/inference conversion.
            expected_scores = 1.0 + expit(logits).sum(axis=1)
            integer_predictions = np.rint(expected_scores)
            integer_predictions = np.clip(
                integer_predictions, 1, self.num_classes
            ).astype(int)
            labels.extend(integer_predictions.tolist())

        if output_file is not None:
            self.save_json(sentences, labels, output_file)
        return labels

    @staticmethod
    def save_json(
        sentences: list[str],
        labels: list[int],
        output_file: str | Path,
    ) -> Path:
        """Save aligned sentences and integer sentiment labels to JSON."""
        if len(sentences) != len(labels):
            raise ValueError("Number of sentences and labels must match")
        path = Path(output_file).expanduser()
        if path.suffix.lower() != ".json":
            path = Path(str(path) + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {"sentence": sentence, "sentiment_label": int(label)}
            for sentence, label in zip(sentences, labels)
        ]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        return path
