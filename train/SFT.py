from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset

# Step 1: Load Dataset
dataset_path = "path_to_your_dataset.json"  # Replace with your dataset path
dataset = load_dataset("json", data_files=dataset_path)

# Step 2: Prepare the Model and Tokenizer
model_name = "llama-3.2-model-name"  # Replace with the correct LLaMA 3.2 model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 3: Preprocess Dataset for Instruction Tuning
def preprocess_function(examples):
    instruction = examples["instruction"]
    input_text = examples["input"]
    output_text = examples["output"]
    prompt = f"{instruction}\n\n{input_text}" if input_text.strip() else instruction
    return tokenizer(prompt, text_target=output_text, max_length=512, truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Step 4: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./llama-fine-tuned",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
)

# Step 5: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation", None),
    tokenizer=tokenizer,
)

# Step 6: Start Training
trainer.train()

# Save the final model
trainer.save_model("./llama-fine-tuned")
