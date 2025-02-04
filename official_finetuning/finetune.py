from typing import Optional
import inspect

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from process_data import get_dataset

model_name = "seeklhy/codes-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(samples):
    tokenized_sample = tokenizer(samples["text"],truncation=True,padding="max_length",
        max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(samples["labels"],truncation=True,padding="max_length",
        max_length=512)
    tokenized_sample["labels"] = labels["input_ids"]
    #print(tokenized_sample["labels"])
    return tokenized_sample


def main(model_name: Optional[str] = None):
    if model_name is None:
        model_name = "seeklhy/codes-7b"

    # device = "cuda" if torch.cuda.is_available() else "cpu"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        use_cache=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        #target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    dataset = get_dataset()
    dataset = dataset.map(
    tokenize_dataset,
    batched=True,
    remove_columns=["text", "labels"]
)
    print(dataset)

    
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=TrainingArguments(
            per_device_train_batch_size=4,
            num_train_epochs=4,
            logging_steps=20,
            learning_rate=5e-6,
            output_dir="outputs",
	    fp16 = True,
        ),
    )

    trainer.train()
    trainer.save_model("outputs")  # Save the model
    tokenizer.save_pretrained("outputs")  # Save the tokenizer


if __name__ == "__main__":
    main()
