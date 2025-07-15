from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from evaluate import load as load_metric
import os
os.environ["WANDB_DISABLED"] = "true"

def train_model(direction="en-id"):
    # Load data
    df = pd.read_csv('identic_en_id.csv').dropna()

    # Model configuration
    if direction == "en-id":
        model_name = "Helsinki-NLP/opus-mt-en-id"
        source_col, target_col = "en_sentence", "id_sentence"
    else:
        model_name = "Helsinki-NLP/opus-mt-id-en"
        source_col, target_col = "id_sentence", "en_sentence"

    # Load model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Prepare dataset
    dataset = Dataset.from_pandas(df)

    def preprocess(examples):
        inputs = tokenizer(examples[source_col], padding="max_length", truncation=True, max_length=128)
        targets = tokenizer(examples[target_col], padding="max_length", truncation=True, max_length=128)
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(preprocess, batched=True)
    split = tokenized_dataset.train_test_split(test_size=0.1)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./model_{direction}",
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_total_limit=1,
        predict_with_generate=True,
        logging_steps=50,
        fp16=True
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()
    trainer.save_model(f"./model_{direction}/final")
    print(f"✅ {direction} model trained!")

    return f"./model_{direction}/final"