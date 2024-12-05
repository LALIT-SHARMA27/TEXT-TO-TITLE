import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from title import get_metadata, process_papers, split_data, get_filtered_categories
# Step 1: Get metadata
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

metadata = get_metadata()

# Step 2: Define the categories to filter
paper_categories = get_filtered_categories()

# Step 3: Process papers and create a DataFrame
papers = process_papers(metadata, paper_categories)

# Step 4: Split the data into training and evaluation sets
train_df, eval_df = split_data(papers)

print(f"Training data size: {len(train_df)}")
print(f"Evaluation data size: {len(eval_df)}")

model_name = "bartowski/QwQ-32B-Preview-GGUF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

def preprocess_data(df):
    inputs = df['input_text'].apply(lambda x: tokenizer(x, padding="max_length", truncation=True, max_length=512))
    outputs = df['target_text'].apply(lambda x: tokenizer(x, padding="max_length", truncation=True, max_length=128))
    return inputs, outputs

train_inputs, train_outputs = preprocess_data(train_df)
eval_inputs, eval_outputs = preprocess_data(eval_df)

# Step 2: Create a Hugging Face Dataset from DataFrame
train_dataset = Dataset.from_pandas(pd.DataFrame({"input_ids": train_inputs['input_ids'], "labels": train_outputs['input_ids']}))
eval_dataset = Dataset.from_pandas(pd.DataFrame({"input_ids": eval_inputs['input_ids'], "labels": eval_outputs['input_ids']}))

# Step 3: Define training arguments
training_args = TrainingArguments(
    output_dir="/home/cvlab/laliot",  # output directory for model checkpoints
    evaluation_strategy="epoch",  # evaluate at the end of each epoch
    learning_rate=2e-5,  # learning rate for fine-tuning
    per_device_train_batch_size=4,  # batch size for training
    per_device_eval_batch_size=8,  # batch size for evaluation
    num_train_epochs=3,  # number of epochs to train the model
    weight_decay=0.01,  # weight decay for regularization
    logging_dir="/home/cvlab/laliot/logs",  # directory for logs
    logging_steps=10,  # log every 10 steps
    save_steps=10_000,  # save the model every 10,000 steps
    save_total_limit=2,  # limit the total number of saved models
)

# Step 4: Initialize the Trainer
trainer = Trainer(
    model=model,  # the model to be fine-tuned
    args=training_args,  # training arguments
    train_dataset=train_dataset,  # the training dataset
    eval_dataset=eval_dataset,  # the evaluation dataset
    tokenizer=tokenizer,  # the tokenizer
)

# Step 5: Fine-tune the model
trainer.train()

# Step 6: Save the fine-tuned model and tokenizer
model.save_pretrained("/home/cvlab/laliot/fine_tuned_model")
tokenizer.save_pretrained("/home/cvlab/laliot//fine_tuned_model")

eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

input_text = "We demonstrate the capability of accurate time transfer using optical fibers over long distances utilizing a dark fiber and hardware which is usually employed in two-way satellite time and frequency transfer (TWSTFT)."
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Generated Summary:", summary)
