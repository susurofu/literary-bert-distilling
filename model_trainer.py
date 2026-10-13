# class for model training 

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import Dataset
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BertForOrdinalRegression(BertPreTrainedModel):
    """
    BERT with a cumulative ordinal regression head.

    For 9-ordered classes, the model learns 8 probabilities:

        P(y > 1)
        P(y > 2)
        .
        P(y > 8)

    """

    def __init__(self, config):
        super().__init__(config)

        self.num_classes = config.num_labels

        if self.num_classes < 2:
            raise ValueError("num_labels must be at least 2.")

        self.bert = BertModel(config)

        dropout_probability = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(dropout_probability)

        #0ne latent sentiment score for every sentence.
        self.sentiment_score = nn.Linear(
            config.hidden_size,
            1,
        )

        #k - 1 thresholds for K ordinal classes.
        self.first_threshold = nn.Parameter(torch.tensor(-2.0))

        self.threshold_deltas = nn.Parameter(torch.zeros(self.num_classes - 2))
        self.post_init()

    def get_ordered_thresholds(self) -> torch.Tensor:
        """
        construct monotonically increasing thresholds.

        note: softplus(delta) is always positive, so every subsequent threshold is greater than the preceding threshold.
        """

        if self.num_classes == 2:
            return self.first_threshold.reshape(1)

        positive_deltas = F.softplus(self.threshold_deltas)

        remaining_thresholds = (self.first_threshold + torch.cumsum(positive_deltas, dim=0))

        return torch.cat([self.first_threshold.reshape(1),remaining_thresholds,])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # shape: [batch_size, 1]
        latent_score = self.sentiment_score(pooled_output)

        # shape: [8] in the current case
        thresholds = self.get_ordered_thresholds()

        # shape: [batch_size, 8]
        # higher latent sentiment means that more thresholds
        # are likely to be crossed.
        logits = latent_score - thresholds.unsqueeze(0)

        loss = None

        if labels is not None:
            labels = labels.long()

            # [1, 2, ..., 8] 1 of 10 is guaranteed anywhere
            levels = torch.arange(
                1,
                self.num_classes,
                device=labels.device,
            ).unsqueeze(0)


            # e.g. label 7 -> [1, 1, 1, 1, 1, 1, 0, 0]
            ordinal_targets = (labels.unsqueeze(1) > levels).float()

            loss = F.binary_cross_entropy_with_logits(logits,ordinal_targets)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModelTrainer:
    """

    Param

    base_model : str
        HF model identifier or local BERT checkpoint directory.
    dataset : str or Path or pd.DataFrame
    num_classes : int def 10
    max_len : int def 256
    output_dir : str or Path
    seed : int def 1111

    run: read_validate_data -> set_training_parameters ->
    tokenize_dataset -> train -> evaluate -> save_best_models
    """

    def __init__(
        self,
        base_model: str,
        dataset: str | Path | pd.DataFrame,
        num_classes: int = 10,
        max_len: int = 256,
        output_dir: str | Path = "bert_sentiment_ordinal",
        seed: int = 1111,
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        if max_len < 1:
            raise ValueError("max_len must be a positive integer.")

        self.base_model = base_model
        self.dataset = dataset
        self.num_classes = num_classes
        self.max_len = max_len
        self.output_dir = str(output_dir)
        self.seed = seed

        self.df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.validation_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self.tokenizer = None
        self.train_dataset: Dataset | None = None
        self.validation_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.data_collator = None

        self.model: BertForOrdinalRegression | None = None
        self.training_arguments: TrainingArguments | None = None
        self.trainer: Trainer | None = None
        self.test_results: dict | None = None
        self.checkpoint_results: pd.DataFrame | None = None

        self.early_stopping_patience = 4
        self._training_settings: dict = {}
        set_seed(self.seed)

    def read_validate_data(self) -> pd.DataFrame:
        if isinstance(self.dataset, pd.DataFrame):
            df = self.dataset.copy()
        else:
            df = pd.read_csv(self.dataset)

        df.rename(columns={"Sentence": "text", "Sentiment_score": "label"}, inplace=True)

        required_columns = {"text", "label"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        df = df[["text", "label"]].copy()
        df = df.dropna(subset=["text", "label"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""]
        df["label"] = pd.to_numeric(df["label"], errors="raise")

        if not np.all(df["label"] == np.floor(df["label"])):
            raise ValueError(f"Ordinal model expects integer labels from 1 to {self.num_classes}.")
        df["label"] = df["label"].astype(int)

        if not df["label"].between(1, self.num_classes).all():
            invalid = sorted(df.loc[
                ~df["label"].between(1, self.num_classes), "label"
            ].unique())
            raise ValueError(
                f"Labels must be between 1 and {self.num_classes}. "
                f"Invalid labels: {invalid}"
            )

        df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
        self.df = df
        # clear downstream representations if data are read again.
        self.train_df = self.validation_df = self.test_df = None
        self.train_dataset = self.validation_dataset = self.test_dataset = None
        self.trainer = None
        self.test_results = None
        self.checkpoint_results = None

        print("Number of observations:", len(df))
        print("\nLabel distribution:")
        print(df["label"].value_counts().sort_index())
        return df

    def split_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """80/10/10 stratified split, with the same random seed."""
        if self.df is None:
            self.read_validate_data()

        train_df, temporary_df = train_test_split(
            self.df,
            test_size=0.20,
            random_state=self.seed,
            stratify=self.df["label"],
        )
        validation_df, test_df = train_test_split(
            temporary_df,
            test_size=0.50,
            random_state=self.seed,
            stratify=temporary_df["label"],
        )

        self.train_df = train_df
        self.validation_df = validation_df
        self.test_df = test_df
        self.train_dataset = self.validation_dataset = self.test_dataset = None

        print("\nSplit sizes:")
        print("Train:", len(train_df))
        print("Validation:", len(validation_df))
        print("Test:", len(test_df))
        return train_df, validation_df, test_df

    def set_training_parameters(
        self,
        *,
        early_stopping_patience: int | None = None,
        **training_overrides,
    ) -> TrainingArguments:

        if self.train_df is None:
            self.split_dataset()

        if early_stopping_patience is not None:
            if early_stopping_patience < 1:
                raise ValueError("early_stopping_patience must be >= 1")
            self.early_stopping_patience = early_stopping_patience

        defaults = dict(
            output_dir=self.output_dir,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.10,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=1,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            eval_steps = 100,
            save_steps = 100,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="mae",
            greater_is_better=False,
            save_total_limit=6,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=self.seed,
            data_seed=self.seed,
        )
        defaults.update(self._training_settings)
        defaults.update(training_overrides)
        self.training_arguments = TrainingArguments(**defaults)
        self._training_settings = defaults
        return self.training_arguments

    def tokenize_dataset(self) -> tuple[Dataset, Dataset, Dataset]:
        if self.train_df is None:
            self.split_dataset()

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        def tokenize_batch(batch: dict) -> dict:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.max_len,
            )

        def make_dataset(frame: pd.DataFrame) -> Dataset:
            dataset = Dataset.from_pandas(
                frame.reset_index(drop=True),
                preserve_index=False,
            )
            dataset = dataset.rename_column("label", "labels")
            dataset = dataset.map(
                tokenize_batch,
                batched=True,
                remove_columns=["text"],
            )
            return dataset

        self.train_dataset = make_dataset(self.train_df)
        self.validation_dataset = make_dataset(self.validation_df)
        self.test_dataset = make_dataset(self.test_df)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return self.train_dataset, self.validation_dataset, self.test_dataset


    def logits_to_predictions(
        self, logits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Original cumulative probability -> expected score -> integer score."""
        cumulative_probabilities = expit(logits)
        expected_scores = 1.0 + cumulative_probabilities.sum(axis=1)
        integer_predictions = np.rint(expected_scores)
        integer_predictions = np.clip(
            integer_predictions, 1, self.num_classes
        ).astype(int)
        return expected_scores, integer_predictions

    def compute_metrics(self, eval_prediction) -> dict[str, float]:
        logits, labels = eval_prediction
        labels = np.asarray(labels).astype(int)
        expected_scores, integer_predictions = self.logits_to_predictions(logits)

        mae = mean_absolute_error(labels, expected_scores)
        rmse = np.sqrt(mean_squared_error(labels, expected_scores))
        exact_accuracy = accuracy_score(labels, integer_predictions)
        within_one = np.mean(np.abs(integer_predictions - labels) <= 1)
        quadratic_kappa = cohen_kappa_score(
            labels, integer_predictions, weights="quadratic"
        )
        spearman_result = spearmanr(labels, expected_scores)
        spearman_correlation = float(spearman_result.statistic)
        if np.isnan(spearman_correlation):
            spearman_correlation = 0.0

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "accuracy": float(exact_accuracy),
            "within_1": float(within_one),
            "quadratic_kappa": float(quadratic_kappa),
            "spearman": spearman_correlation,
        }

    def train(
        self,
        *,
        eval_steps: int | None = None,
        save_steps: int | None = None,
        eval_strategy: str | None = None,
        save_strategy: str | None = None,
        resume_from_checkpoint: str | bool | None = None,
        evaluate_test: bool = True,
    ):

        # do not replace the hyperparameters previously set by the caller.
        overrides = {}
        if eval_steps is not None:
            overrides["eval_steps"] = eval_steps
        if save_steps is not None:
            overrides["save_steps"] = save_steps
        if eval_strategy is not None:
            overrides["eval_strategy"] = eval_strategy
        if save_strategy is not None:
            overrides["save_strategy"] = save_strategy

        if self.training_arguments is None or overrides:
            self.set_training_parameters(**overrides)
        if self.train_dataset is None:
            self.tokenize_dataset()

        self.model = BertForOrdinalRegression.from_pretrained(
            self.base_model,
            num_labels=self.num_classes,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            processing_class=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience
                )
            ],
        )
        train_result = self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )

        if evaluate_test:
            self.evaluate(split="test")
        return train_result

    def evaluate(self, split: str = "test") -> dict[str, float]:
        """Evaluate the best loaded model on 'test' or 'validation'."""
        if self.trainer is None:
            raise RuntimeError("Train the model before calling evaluate().")
        if split not in ("test", "validation"):
            raise ValueError("split must be 'test' or 'validation'.")
        if self.test_dataset is None:
            self.tokenize_dataset()

        dataset = self.test_dataset if split == "test" else self.validation_dataset
        results = self.trainer.evaluate(dataset, metric_key_prefix=split)
        if split == "test":
            self.test_results = results
        print(f"\n{split.capitalize()} results:")
        for metric_name, metric_value in results.items():
            print(f"{metric_name}: {metric_value}")
        return results

    def evaluate_interim_models(
        self, split: str = "validation"
    ) -> pd.DataFrame:
        if self.training_arguments is None:
            self.set_training_parameters()
        if self.validation_dataset is None:
            self.tokenize_dataset()
        if split not in ("validation", "test"):
            raise ValueError("split must be 'validation' or 'test'.")

        checkpoints = sorted(
            (p for p in Path(self.output_dir).glob("checkpoint-*")
             if p.is_dir() and p.name.rsplit("-", 1)[-1].isdigit()),
            key=lambda p: int(p.name.rsplit("-", 1)[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint-* directories found in {self.output_dir!r}."
            )

        dataset = (self.validation_dataset if split == "validation" else self.test_dataset)
        rows = []
        best_checkpoint = (self.trainer.state.best_model_checkpoint if self.trainer is not None else None)

        for checkpoint in checkpoints:
            checkpoint_model = BertForOrdinalRegression.from_pretrained(str(checkpoint))
            checkpoint_trainer = Trainer(
                model=checkpoint_model,
                args=self.training_arguments,
                eval_dataset=dataset,
                processing_class=self.tokenizer,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
            )
            metrics = checkpoint_trainer.evaluate(dataset, metric_key_prefix=split)
            rows.append({
                "checkpoint": str(checkpoint),
                "step": int(checkpoint.name.rsplit("-", 1)[-1]),
                "is_best": (
                    Path(best_checkpoint).resolve() == checkpoint.resolve()
                    if best_checkpoint is not None else False
                ),
                **metrics,
            })
            del checkpoint_trainer, checkpoint_model

        self.checkpoint_results = pd.DataFrame(rows)
        return self.checkpoint_results

    def save_best_models(self, best_model_dir: str | Path | None = None) -> Path:
        if self.trainer is None or self.tokenizer is None:
            raise RuntimeError("Train the model before saving it.")
        destination = Path(
            best_model_dir if best_model_dir is not None
            else Path(self.output_dir) / "best_model"
        )
        self.trainer.save_model(str(destination))
        self.tokenizer.save_pretrained(str(destination))
        print(f"\nModel saved to: {destination}")
        return destination

    save_best_model = save_best_models
