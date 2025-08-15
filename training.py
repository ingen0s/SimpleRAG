from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets

# Load datasets
original_dataset = load_dataset("csv", data_files={"train": "original_train.csv"})
new_dataset = load_dataset("csv", data_files={"train": "new_data.csv"})
combined_dataset = concatenate_datasets([original_dataset["train"], new_dataset["train"]])

# Load fine-tuned model and tokenizer
model_path = "./fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Tokenize combined dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./retrained_results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train and save
trainer.train()
model.save_pretrained("./retrained_model")
tokenizer.save_pretrained("./retrained_model")