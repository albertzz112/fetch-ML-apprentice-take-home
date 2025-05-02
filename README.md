# Albert Zhang - Fetch ML Apprentice Take Home

## Overview

This project implements a sentence transformer using HuggingFace's Transformers library and extends it into a multi-task learning (MTL) framework. The goal is to build a modular, extensible architecture that supports multiple NLP tasks in parallel.

This README provides an in-depth explanation of all design decisions and implementation choices made throughout the exercise.

---

## Task 1: Sentence Transformer Implementation

The base model is a transformer that uses the distilbert-base-uncased pretrained model from HuggingFace. It encodes a batch of sentences into fixed-length embeddings with CLS token.

Model Design:

- Framework: PyTorch with HuggingFace Transformers for ease of implementation and fast training. 
- Model Choice: distilbert-base-uncased was used for its small size and fast inference. 
- Embedding Strategy: We use the output of the CLS token from the last hidden layer. This helps for simple feature extraction for sentence classification and sentiment analysis. 
- Tokenizer: HuggingFace's AutoTokenizer handles tokenization, padding, and truncation to a consistent max sequence length.

---

## Task 2: Multi-Task Learning Expansion

To support multi-task learning, we extend the transformer to create a multi-task model. This supports two NLP tasks simultaneously:

- Task A: Sentence Classification (3 classes)
- Task B: Sentiment Analysis (2 classes)

Model Design: 

- Both tasks share a transformer backbone (distilbert-base-uncased).
- Task A uses an MLP head with ReLU and Dropout to capture more classes. 
- Task B uses a simple linear head for binary classification.

---

## Task 3: Training Considerations
Discuss the implications and advantages of each scenario and explain your rationale as to how the model should be trained given the following:

1. If the entire network should be frozen: This is helpful if the model is already trained and you need to use the model to extract features from text. You are unable to train the model as it won't update any parameters. Another approach when the entire network is frozen is parameter efficient fine-tuning. This is a process where the existing parameters are frozen and the model is augmented with additional, trainable parameters. This can be done via LoRA to fine tune the model for more specific purposes. 

2. If only the transformer backbone should be frozen: This is used when you have a trained backbone that generalizes well. By freezing it you can train task-specific heads without training the backbone to the level of overfitting. This also allows for faster training as you don't need to update the entire backbone. Some disadvantages is that each task-specific head is heavily reliant on how the backbone is trained, and if the backbone is not producing enough information or the information is to generalized, it may cap the learning of the task-specific heads. When training, only pass the heads into the optimizer. 

3. If only one of the task-specific heads (either for Task A or Task B) should be frozen: This is helpful when you want to focus on training one task without it affecting and potentially degrade the other head's performance. The backbone can be either frozen or unfrozen, depending on whether or not you want the backbone to train more heavily on the specific unfrozen task. This is useful when doing multi-task continual learning or transfer learning, where you want to adapt the model to learn a new task without forgetting what it already learned for a previous task. When training, only pass the unfrozen trainable components into the optimizer. 

Consider a scenario where transfer learning can be beneficial. Explain how you would approach the transfer learning process including:

1. The choice of a pre-trained model: When transfer learning, using a generalized pretrained model is helpful for extracting base meaning from text. For this, I would use a pretrained BERT model like distilbert-base-uncased since it's a relatively small model that gives generalized embeddings of sentences. So, given its small size and good generalization, I could quickly train for specific tasks using it as the backbone to my model. 

2. The layers you would freeze/unfreeze: When transfer learning, I would first freeze the entire backbone to train the task-specific head. This allows the model to quickly learn how the task-specific head should respond to generalized information from the backbone. Then, I would unfreeze some number of layers from the top of the backbone with a low learning rate, allowing for finetuning of the backbone to my specific task without the backbone forgetting what it has already learned. 

3. The rationale behind these choices: Freezing the entire transformer backbone initially allows the model to quickly learn the task-specific mapping without updating millions of backbone parameters. This approach is efficient and reduces the risk of overfitting. Once the task-specific head has stabilized, gradually unfreezing upper layers of the backbone allows the model to refine its language representations based on the new task, improving performance while still retaining the general knowledge that the pretrained model came with. 

## Task 4: Training Loop Implementation

Design Decisions:

- Batches are Python dictionaries with keys, allowing for easy expansion of a hypothetical dataset. 
- Cross-entropy loss is used for both tasks, to help model learn the correct probability distribution for classification. 
- Losses are summed and used to update model weights. Both tasks are untrained and equally important, so no weighting is used when calculating total loss. 
- The optimizer updates only the task specific heads to keep the generalization of the transformer and quickly train for multi task learning. 
