# Albert Zhang - Fetch ML Apprentice Take Home

## Overview

This project implements a sentence transformer using HuggingFace's Transformers library and extends it into a multi-task learning (MTL) framework. The goal is to build a modular, extensible architecture that supports multiple NLP tasks in parallel.

This README provides an in-depth explanation of all design decisions and implementation choices made throughout the exercise.

---

## Task 1: Sentence Transformer Implementation

The base model is a `SentenceTransformer` that uses the `distilbert-base-uncased` pretrained model from HuggingFace. It encodes a batch of sentences into fixed-length embeddings using the `[CLS]` token output.

### Key Design Decisions:

- Framework: PyTorch with HuggingFace Transformers for ease of implementation, reproducibility, and fine-tuning support.
- Model Choice: `distilbert-base-uncased` was chosen for its small size, fast inference, and effectiveness on sentence-level tasks.
- Embedding Strategy: We use the output of the `[CLS]` token from the last hidden layer. This is a common and effective approach for sentence-level classification.
- Tokenizer: HuggingFace's `AutoTokenizer` handles tokenization, padding, and truncation to a consistent max sequence length.
- Batch Support: The `forward()` function accepts a list of sentences and outputs a corresponding tensor of embeddings.

---

## Task 2: Multi-Task Learning Expansion

To support multi-task learning, we extend the `SentenceTransformer` to create a `MultiTaskModel`. This architecture supports two NLP tasks simultaneously:

- Task A: Sentence Classification (3 classes)
- Task B: Sentiment Analysis (2 classes)

Each task has a separate classification head, and both share the same transformer backbone.

### Architectural Modifications:

- Reused the base `SentenceTransformer` for encoding sentences.
- Added two independent linear layers:
  - `classifier_a`: Outputs logits for Task A (3-way classification).
  - `classifier_b`: Outputs logits for Task B (binary classification).
- Returned both outputs as a dictionary for use in training and evaluation.

---

## Task 3: Training Considerations

This section explores different freezing strategies and transfer learning configurations, including justifications.

### 1. Entire Network Frozen

Use Case: Feature extraction

- The model can be used purely for embedding extraction, e.g., for clustering or nearest neighbor search.
- Drawback: No task-specific tuning; general embeddings may not be optimal for classification.

### 2. Only Transformer Backbone Frozen

Use Case: Fast adaptation to new tasks with limited data

- The classifier heads can be trained from scratch on new tasks.
- This allows for quick adaptation without overfitting the massive transformer.
- Justification: Ideal when reusing embeddings for similar tasks or with domain-specific classifiers.

### 3. Only One Task Head Frozen

Use Case: Asymmetric transfer learning

- Suppose Task B is well-trained on a large dataset. If Task A is new and Task B is relevant, we freeze B and train A.
- Benefit: Maintains prior knowledge while adapting to a new task.
- Drawback: Might require regularization if tasks interfere too much.

---

### Transfer Learning Strategy

Chosen Pretrained Model: `distilbert-base-uncased`

- Light-weight and widely used for sentence-level tasks
- Performs well even with frozen weights

Transfer Steps:
1. Load pre-trained transformer weights.
2. Freeze transformer if data is limited or task is unrelated.
3. Train task-specific heads.
4. Optionally fine-tune the entire model if performance stagnates.

Why Freeze vs. Unfreeze:
- Freeze to avoid overfitting and for faster training.
- Unfreeze when embeddings must be adapted to task-specific nuances.

---

## Task 4: Training Loop Implementation (BONUS)

See `train/train_loop.py`.

This basic training loop handles batches of:
- Sentences
- Task A labels (3 classes)
- Task B labels (2 classes)

### Design Assumptions:
- DataLoader yields batches of `(sentences, label_a, label_b)`
- Cross-entropy is used for both tasks (classification)
- Losses from both heads are summed for optimization
- Optimizer updates both the transformer (if unfrozen) and heads

### MTL Considerations:

- A single forward pass produces both task outputs, reducing redundant computation.
- Tasks are assumed to be balanced in importance; losses are weighted equally.
- In practice, loss weights may need tuning based on task difficulty or label noise.
- Can easily be extended to include evaluation metrics and logging.
