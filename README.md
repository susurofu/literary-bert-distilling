# Distilling Literary Sentiment from LLM Annotations into an Ordinal BERT Model

This is a repository for a preprint "Distilling Literary Sentiment from LLM Annotations into an Ordinal BERT Model."

This repo contains:
1. training_models.ipynb : trains 6 BERT models, main training code with Cumulative ordinal regression acrchitecture is located at model_trainer.py;
2. preparation_of_bert-sentiments.ipynb : prepares sentences for evaluation by inferencing trained models. This inference is slightly different from models uploaded to HF. Use HF pages for updated inference instructions;
3. evaluation.ipynb : evaluates trained models, RoBERTa models, teacher model (Gemma 4: 31b) against human annotation;
4. human_average_arc.pkl : conntains average between two sets of sentiment scores by two human annotators;
5. machine_arcs.pkl : contains different sentiment data from a paper by Bizzoni and Feldkamp. Current research uses only roberta_twt and roberta_twt_xlm;
6. 9 - Hemingway, Ernest - The Old Man and the Sea.csv : contains scores by a teacher model.
7. bert_sentiments : contains sentiment scores inferenced by trained models。
8. results : contains accumulated data from evaluation.ipynb


## Trained models.
Models can be accessed via the links below:

- **BERT-large**: [`susurofu/literary-bert-en-large`](https://huggingface.co/susurofu/literary-bert-en-large) (340M)
- **BERT-base**: [`susurofu/literary-bert-en-base`](https://huggingface.co/susurofu/literary-bert-en-base) (110M)
- **BERT-medium**: [`susurofu/literary-bert-en-medium`](https://huggingface.co/susurofu/literary-bert-en-medium) (41.7M)
- **BERT-small**: [`susurofu/literary-bert-en-small`](https://huggingface.co/susurofu/literary-bert-en-small) (29.1M)
- **BERT-mini**: [`susurofu/literary-bert-en-mini`](https://huggingface.co/susurofu/literary-bert-en-mini) (11.3M)

BERT-tiny was trained during development but is **not released** because it converged to constant neutral predictions. The constant-neutral baseline remains in the evaluation tables for context.









## Credentials

The human-annotated sentences of E. Hemingway's *The Old Man and the Sea" and sentiment scores for roberta_twt and roberta_twt_xlm are taken from Yuri Bizzoni and Pascale Feldkamp. 2024. Sentiment
analysis for literary texts: Hemingway as a case-
study. *Journal of Data Mining & Digital Humanities*,
NLP4DH:1–20.


