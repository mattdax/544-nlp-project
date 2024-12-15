from transformers import AutoTokenizer, AutoModelForCausalLM
import re

from typing import List

import pandas as pd

from datasets import Dataset, DatasetDict

from peft import PeftModel
import torch
import json
import numpy as np


template = template = """\
You are an SQL expert. The task you are going to perform is to generate reasoning chains and SQLite query given instructions, user questions, and relevant schema. Always follow the instructions given to you as it will help you generate reasoning chains and SQLite query in a structured manner. Give your output in a txt code block. Do not use asterisk to highlight SQLite keywords. Encapsulate chains within <chains> and </chains>. Encapsulate SQLite query within <SQL> and </SQL>.

# Instructions 
Follow the below instructions step by step while generating reasoning chains:
1. Sequential Structure (Determine the order of SQL clauses: SELECT, FROM, JOIN, GROUP BY, ORDER BY, etc.)
2. Condition Structure (Apply filtering using WHERE or HAVING clauses to define specific conditions)
3. Join Structure (Use JOIN clauses to combine tables based on shared keys or relationships)
4. Aggregation Structure (Use aggregate functions like COUNT, SUM, AVG, etc., to summarize data)

# 
Here is one example for you to understand the task better:
##
Example User question:
Find the total budgets of the Marketing or Finance department.
 
##
Example Relevant schema:
Schema_links:
['department.budget', 'department.dept_name', 'Marketing', 'Finance']
 
##
Output for Example 1
<chains>
1. Sequential Structure: Begin with the SELECT clause to specify the required field. Since the question asks for "total budgets," an aggregation function (SUM) will be used on the "department.budget" field.
2. Condition Structure: Apply a WHERE clause to filter for the specific departments mentioned, i.e., "Marketing" or "Finance."
3. Join Structure: No JOIN is needed here, as the query only involves the "department" table.
4. Aggregation Structure: Use SUM to aggregate the budget values for the specified departments, providing the total budget for each.
</chains>
<SQL>
SELECT SUM(department.budget)
FROM department
WHERE department.dept_name = 'Marketing' OR department.dept_name = 'Finance';
</SQL>
Example 1 ends here

Below you can find user question and relevant schema
# User question:
{question}

# Relevant schema:
{schema}

"""

model_name = "seeklhy/codes-7b-spider"  # Base model used
adapter_path = "weights"  # Path to the adapter_model.safetensors
tokenizer_path = "weights"  # Path to tokenizer files
eval_file = "test_no_fields.csv" 
output_file = "predictions.json"

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def convert_int64_to_int(obj):
    if isinstance(obj, dict):
        return {key: convert_int64_to_int(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return obj.apply(lambda x: int(x) if not pd.isna(x) else None).to_dict()  # Handle NaN values
    elif isinstance(obj, pd.DataFrame):
        # Handle NaN values in DataFrame
        return obj.applymap(lambda x: int(x) if not pd.isna(x) and isinstance(x, (np.int64, np.Int64)) else x).to_dict(orient='records')
    elif isinstance(obj, (np.int64, np.float64)):
        # Check for NaN and handle it
        if pd.isna(obj):
            return None  # or return 0 or some default value if needed
        return int(obj)
    else:
        return obj


def process_dataset(df):
    print("Start processing dataset", flush=True)
    dataset = []
    for index, row in df.iterrows():
        try:
            print(f"Processing row {index}", flush=True)
            sample = template.format(
                question=row.get("question", "No question provided"),
                schema=row.get("schema_links", "No schema provided"),
            )
            
            # print(sample)
            dataset.append({"sample": sample, "index": index})
        except Exception as e:
            print(f"Error processing row {index}: {e}", flush=True)
    return dataset

def get_dataset() -> DatasetDict:
    eval_path = "eval.json"
    eval_df = pd.read_json(eval_path)

    return DatasetDict(
        {
            "eval": Dataset.from_dict({"text_to_sql": process_dataset(eval_df)}),
        }
    )

def get_answer_sql_block(answer: str):
    matches = re.findall(r"<SQL>(.*?)</SQL>", answer, re.DOTALL)

    if len(matches) >= 3:
        return matches[2].strip()
    else:
        return None

def evaluate_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Move model to CUDA if available
    model = model.to(device)
    print(f"Model is loaded on device: {device}", flush=True)

    # Open output file
    eval_df = pd.read_csv(eval_file)
    # eval_df = eval_df[eval_df["score"] == 1]
    # eval_df = eval_df[eval_df["classification"] == "EASY"]
    eval_df['model_predicted_sql'] = None
    # eval_df = eval_df.head(20)

    eval = process_dataset(eval_df)
    
    batch_size = 4  # Adjust this number based on your GPU memory capacity

    for i in range(0, len(eval), batch_size):
        batch = eval[i:i + batch_size]  # Take a smaller chunk of the data

        print(f"Processing batch {i//batch_size + 1}", flush=True)

    # Tokenize and move the batch to the GPU
        inputs = tokenizer(
        [row["sample"] for row in batch],
        return_tensors="pt",
        truncation=True,
        padding=True
    )
        inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate outputs for the batch
        outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode and process each output
        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        print(decoded_outputs)
        for row, decoded_output in zip(batch, decoded_outputs):
            second_sql = get_answer_sql_block(decoded_output)
            print(second_sql)
            eval_df.at[row["index"], 'model_predicted_sql'] = second_sql

    result_dict = {}

    for column in eval_df.columns:
        result_dict[column] = {str(idx): eval_df.at[idx, column] for idx in eval_df.index}
    
    result_dict = convert_int64_to_int(result_dict)

    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    print("Start", flush=True)
    evaluate_model()

