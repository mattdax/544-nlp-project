import re

from typing import List

import pandas as pd

from datasets import Dataset, DatasetDict

template = """\
You are an SQL expert. The task you are going to perform is to generate reasoning chains and SQL query given user questions, relevant schema, and a thought process. Always follow the instructions given to you as it will help you generate reasoning chains and SQL in a structured manner. Give your output in a txt code block. Do not use asterisk to highlight SQL keywords. Encapsulate chains within <chains> and </chains>. Encapsulate SQL within <SQL> and </SQL>:


# Instructions 
Follow the below instructions step by step
1. Sequential Structure (Determine the order of SQL clauses: SELECT, FROM, JOIN, GROUP BY, ORDER BY, etc.)
2. Condition Structure (Apply filtering using WHERE or HAVING clauses to define specific conditions)
3. Join Structure (Use JOIN clauses to combine tables based on shared keys or relationships)
4. Aggregation Structure (Use aggregate functions like COUNT, SUM, AVG, etc., to summarize data)

# User question:
{question}

# Relevant schema:
{schema}

<chains>
{reasoning}
</chains>
<SQL>
{sql}
</SQL>\
"""


def process_reasoning(reasoning: str) -> str:
    return re.sub(r"(\d\.)", r"\n\1", reasoning).strip()


def process_schema(fields: str, schema_links: str) -> str:
    return f"Fields:\n{fields}\nSchema_links:{schema_links}"


def process_dataset(df: pd.DataFrame) -> List[str]:
    # Remove entries with invalid reasoning
    df = df[df["score"] == 1]

    dataset = []
    for index, row in df.iterrows():
        sample = template.format(
            question=row["question"],
            schema=row["fields"],
            reasoning=process_reasoning(row["reasoning"]),
            sql=row["predicted_sql"],
        )
        dataset.append(sample)
    return dataset


def get_dataset(load_easy: bool = True) -> DatasetDict:
    train_path = "./dataset/full_train_gpt4o_mini_query_gen_gpt4o_query_correction.json"
    eval_path = "./dataset/full_val_gpt4o_mini_query_gen_gpt4o_query_correction.json"

    train_df = pd.read_json(train_path)
    eval_df = pd.read_json(eval_path)

    if load_easy:
        train_df = train_df[train_df["classification"] == "EASY"]
        eval_df = eval_df[eval_df["classification"] == "EASY"]

    return DatasetDict(
        {
            "train": Dataset.from_dict({"text_to_sql": process_dataset(train_df)}),
            "eval": Dataset.from_dict({"text_to_sql": process_dataset(eval_df)}),
        }
    )


if __name__ == "__main__":
    df = pd.read_json(
        "./dataset/full_train_gpt4o_mini_query_gen_gpt4o_query_correction.json"
    )

    print(df.columns)

    # Remove entries with invalid reasoning
    df = df[df["score"] == 1]
    print(f"Dataset size: {len(df)}")

    # Get a sample from the dataset
    print(process_dataset(df[:1])[0])
