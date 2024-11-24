from transformers import AutoTokenizer, AutoModelForCausalLM
import re

from typing import List

import pandas as pd

from datasets import Dataset, DatasetDict

from peft import PeftModel

template = """\
You are an SQL expert. The task you are going to perform is to generate reasoning chains and SQL query given user questions, relevant schema, and a thought process. Always follow the instructions given to you as it will help you generate reasoning chains and SQL in a structured manner. Give your output in a txt code block. Do not use asterisk to highlight SQL keywords. Encapsulate chains within <chains> and </chains>. Encapsulate SQL within <SQL> and </SQL>:


# Instructions 
Follow the below instructions step by step
1. Sequential Structure (Determine the order of SQL clauses: SELECT, FROM, JOIN, GROUP BY, ORDER BY, etc.)
2. Condition Structure (Apply filtering using WHERE or HAVING clauses to define specific conditions)
3. Join Structure (Use JOIN clauses to combine tables based on shared keys or relationships)
4. Aggregation Structure (Use aggregate functions like COUNT, SUM, AVG, etc., to summarize data)

#
Here is one example for you to understand the task better:
##
User question:
Find the total budgets of the Marketing or Finance department.
 
##
Relevant schema:
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

# Relevant schema:
{schema}

# User question:
{question}

"""

model_name = "seeklhy/codes-7b"  # Base model used
adapter_path = "outputs"  # Path to the adapter_model.safetensors
tokenizer_path = "outputs"  # Path to tokenizer files
eval_file = "eval.json"
output_file = "evaluation_outputs.txt"


def process_dataset(df):
    dataset = []
    for index, row in df.iterrows():
        sample = template.format(
            question=row["question"],
            schema=row["filtered_schema"],
        )
        dataset.append(sample)
    return dataset

def get_dataset() -> DatasetDict:
    eval_path = "eval.json"
    eval_df = pd.read_json(eval_path)

    return DatasetDict(
        {
            "eval": Dataset.from_dict({"text_to_sql": process_dataset(eval_df)}),
        }
    )

def evaluate_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    # Open output file
    eval_df = pd.read_json(eval_file)
    eval_df = eval_df[eval_df["score"] == 1]
    eval_df = eval_df[eval_df["classification"] == "EASY"]
    eval = process_dataset(eval_df)
    print("here")
    with open(output_file, "w") as out_f:
        for idx, sample in enumerate(eval):
            # Tokenize the input
            inputs = tokenizer(sample, return_tensors="pt", truncation=True)
            #print("here 2")
            # Generate output
            outputs = model.generate(**inputs, max_length=512)
            #print("here 3")
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            #print(answer)
            # Write to the output file
            #out_f.write(f"Sample {idx + 1}:\n{sample}\n")
            out_f.write(f"{answer}\n")

if __name__ == "__main__":
    evaluate_model()

