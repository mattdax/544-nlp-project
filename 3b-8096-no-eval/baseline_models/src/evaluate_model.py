from argparse import ArgumentParser
from typing import Optional

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from spider_dataset import get_spider_devset

# Some models to try
# model = "defog/sqlcoder-7b-2"
# model = "seeklhy/codes-1b"
# model = "seeklhy/codes-7b"
# model = "seeklhy/codes-7b-merged"


def main(model_name: Optional[str] = None, batch_size: int = 1, quantize: bool = False):

    if model_name is None:
        model_name = "seeklhy/codes-7b-merged"

    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # Use torch.float16 for GPUs older than Ampere (RTX 3000 series)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device,
        use_cache=True,
    )
    device = model.device

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    dataset = get_spider_devset(tokenizer)

    dataset.encodings.to(device)

    gold = open("./gold_query.txt", "w")
    generated = open("./generated.txt", "w")

    dataloader = DataLoader(dataset, batch_size=batch_size)

    # generation params
    num_beams = 4

    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, labels = batch

            # Generation parameters
            generated_ids = model.generate(
                **inputs,
                num_return_sequences=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=256,
                do_sample=False,
                num_beams=num_beams,
            )
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # empty cache to generate more results w/o memory crashing
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Write outputs
            for output in outputs[::num_beams]:
                generated_sql = (
                    output.split("[SQL]")[-1].split("[")[0].replace("\n", " ").strip()
                )
                generated.write(generated_sql + "\n")

            # Write gold queries
            for sql, db in zip(labels["sql"], labels["database"]):
                gold.write(f"{sql}\t{db}\n")

    gold.close()
    generated.close()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("-B", "--batch_size", type=int, required=False)
    parser.add_argument("-Q", "--quantize", action="store_true")

    args = parser.parse_args()

    main(**vars(args))
