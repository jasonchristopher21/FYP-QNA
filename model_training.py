from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import evaluate

def train(all_dataset, test_dataset, seed, num_epochs=10, cuda="cuda:0", printout=True, lr=1e-5, num_layer_to_unfreeze=1):

    # Tokenize the datasets
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Split train dataset into train and validation
    split_datasets = all_dataset.train_test_split(test_size=0.1, seed=seed)
    train_dataset = split_datasets["train"]
    validation_dataset = split_datasets["test"]
    
    train_tokenized = train_dataset.map(preprocess_function(tokenizer), batched=True)
    val_tokenized = validation_dataset.map(preprocess_function(tokenizer), batched=True)
    test_tokenized = test_dataset.map(preprocess_function(tokenizer), batched=True)

    # Initialise Model
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    # Train the model
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory for checkpoints
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        learning_rate=2e-5,  # Learning rate for fine-tuning
        per_device_train_batch_size=16,  # Batch size per device for training
        per_device_eval_batch_size=64,  # Batch size per device for evaluation
        num_train_epochs=3,  # Number of epochs to train
        weight_decay=0.01,  # Weight decay for optimization
        logging_dir="./logs",  # Directory for logs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # Pass the metric function
    )

    # Start training
    trainer.train() 

    # Evaluate the model
    task_evaluator = evaluate.evaluator("question-answering")
    results = task_evaluator.compute(
        model_or_pipeline=model,
        data=test_tokenized,
        metric="squad",
    )

    print("Evaluation results:", results)

    return results, model

def compute_metrics(tokenizer, metric, p):
    predictions, labels = p

    # Convert the predicted token IDs to text
    preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute metrics using the SQuAD evaluation
    result = metric.compute(predictions=preds, references=labels)

    # Return the metrics in a dictionary
    return result

def preprocess_function(tokenizer):
    def preprocess_helper(examples):
        # Tokenize context and question together
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation=True,
            padding="max_length",  # Or use dynamic padding depending on your needs
            max_length=512  # Adjust depending on the model size and dataset
        )
        
        # Find the start and end position of the answer (you may need to process this)
        start_positions = []
        end_positions = []
        for answer in examples["answers"]:
            start_positions.append(answer["answer_start"][0])
            end_positions.append(answer["answer_start"][0] + len(answer["text"]) - 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    return preprocess_helper