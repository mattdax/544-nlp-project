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
            schema=process_schema(row["fields"],row["schema_links"]),
            reasoning=process_reasoning(row["reasoning"]),
            sql=row["predicted_sql"],
        )
        sql = row["gold_sql"]
        dataset.append({"text": sample, "labels": sql})
    return dataset


def get_dataset(load_easy: bool = True) -> DatasetDict:
    train_path = "train.json"
    #eval_path = "val.json"

    train_df = pd.read_json(train_path)
    #eval_df = pd.read_json(eval_path)

    if load_easy:
        train_df = process_dataset(train_df[train_df["classification"] == "EASY"])
        #eval_df = eval_df[eval_df["classification"] == "EASY"]
        
    return Dataset.from_dict(
    {
        "text": [sample["text"] for sample in train_df],
        "labels": [sample["labels"] for sample in train_df],
    }
)


if __name__ == "__main__":
    df = pd.read_json(
        "train.json"
    )

    print(df.columns)

    # Remove entries with invalid reasoning
    df = df[df["score"] == 1]
    print(f"Dataset size: {len(df)}")

    # Get a sample from the dataset
    print(process_dataset(df[:1])[0])
