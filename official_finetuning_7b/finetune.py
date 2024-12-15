from typing import Optional
import inspect

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from process_data import get_dataset, get_eval_dataset

model_name = "seeklhy/codes-7b-spider"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(samples, tokenizer, max_length=512):
    # Ensure the tokenizer uses a valid padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use <|endoftext|> as padding token

    tokenized_batch = {"input_ids": [], "attention_mask": [], "labels": []}

    for text, label in zip(samples["text"], samples["labels"]):
        # Prepare inputs and labels using the provided logic
        prefix_ids = [tokenizer.bos_token_id] + tokenizer(text, truncation=False)["input_ids"]
        target_ids = tokenizer(label, truncation=False)["input_ids"] + [tokenizer.eos_token_id]

        seq_length = len(prefix_ids) + len(target_ids)

        if seq_length <= max_length:  # Pad inputs with pad_token_id
            pad_length = max_length - seq_length
            input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
            attention_mask = [1] * seq_length + [0] * pad_length  # Mask for non-padding tokens
            labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length  # Loss only for target_ids
        else:  # Truncate to max_length
            print("The current input sequence exceeds max_length; truncating.")
            input_ids = prefix_ids + target_ids
            input_ids = [tokenizer.bos_token_id] + input_ids[-(max_length - 1):]  # Ensure <bos> starts the sequence
            attention_mask = [1] * max_length
            labels = [-100] * len(prefix_ids) + target_ids
            labels = labels[-max_length:]  # Truncate labels to match max_length

        # Append processed inputs to batch
        tokenized_batch["input_ids"].append(input_ids)
        tokenized_batch["attention_mask"].append(attention_mask)
        tokenized_batch["labels"].append(labels)

    # Convert to tensors
    tokenized_batch["input_ids"] = torch.tensor(tokenized_batch["input_ids"], dtype=torch.int64)
    tokenized_batch["attention_mask"] = torch.tensor(tokenized_batch["attention_mask"], dtype=torch.int64)
    tokenized_batch["labels"] = torch.tensor(tokenized_batch["labels"], dtype=torch.int64)

    return tokenized_batch

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,  # Disable shuffling here
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

def main(model_name: Optional[str] = None):
    if model_name is None:
        model_name = "seeklhy/codes-7b-spider"

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
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_proj", "c_attn", "q_attn"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    max_length = 4096
    dataset = get_dataset()
    dataset = dataset.map(
        lambda examples: tokenize_dataset(examples, tokenizer, max_length=max_length),
        batched=True,
        remove_columns=["text", "labels"],
    )

    # eval_dataset = get_eval_dataset()
    # eval_dataset = eval_dataset.map(
    # lambda examples: tokenize_dataset(examples, tokenizer, max_length=max_length),
    # batched=True,
    # remove_columns=["text", "labels"]
    # )


    
    trainer = CustomTrainer(
        model=model,
        train_dataset=dataset,
        # eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=TrainingArguments(
            per_device_train_batch_size=1,
            num_train_epochs=4,
            logging_steps=1,
            # evaluation_strategy="steps",  
            # eval_steps=20, #adjust evaluation loss logging here
            learning_rate=5e-6,
            output_dir="a40_outputs_zero_shot_4_epochs_target_modules_small",
	    fp16 = True,
        ),
    )

    trainer.train()
    trainer.save_model("a40_outputs_zero_shot_4_epochs_target_modules_small")  # Save the model
    tokenizer.save_pretrained("a40_outputs_zero_shot_4_epochs_target_modules_small")  # Save the tokenizer


if __name__ == "__main__":
    main()
