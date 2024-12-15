import re
from typing import List
import pandas as pd
from datasets import Dataset, DatasetDict

template = """\
You are an SQLite expert. The task you are going to perform is to generate step-by-step reasoning chains and SQLite query given instructions, user questions, and relevant schema. Always follow the instructions given to you as it will help you generate reasoning chains and SQLite query in a step-by-step manner. Give your output in a txt code block. Do not use asterisk to highlight SQLite keywords. Encapsulate chains within <chains> and </chains>. Encapsulate SQLite query within <SQL> and </SQL>.
Only answer from relevant schema given to you.

# Instructions 
Follow the below instructions while generating the step-by-step reasoning chains:
1. Sequential Structure (Determine the order of SQL clauses: SELECT, FROM, JOIN, GROUP BY, ORDER BY, etc.)
2. Condition Structure (Apply filtering using WHERE or HAVING clauses to define specific conditions)
3. Join Structure (Use JOIN clauses to combine tables based on shared keys or relationships)
4. Aggregation Structure (Use aggregate functions like COUNT, SUM, AVG, etc., to summarize data)

Below you can find user question and relevant schema
# User question:
{question}

# Relevant schema:
{schema}
"""

intermediate = """
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


"""
y_format = """

<chains>
{reasoning}
</chains>
<SQL>
{sql}
</SQL>\

"""


def process_reasoning(reasoning: str) -> str:
    return re.sub(r"(\d\.)", r"\n\1", reasoning).strip()


def process_schema(schema_links: str) -> str:
    return f"\nSchema_links: {schema_links}"


def process_dataset(df: pd.DataFrame) -> List[str]:
    # Remove entries with invalid reasoning
    df = df[df["score"] == 1]

    dataset = []
    for index, row in df.iterrows():
        sample = template.format(
            question=row["question"],
            schema=process_schema(row["schema_links"]),
        )
        label = y_format.format(
            reasoning = row["reasoning"],
            sql = row["predicted_sql"]
        )
        
        dataset.append({"text": sample, "labels": label})
    return dataset


def get_dataset() -> DatasetDict:
    train_path = "best_train.json"

    train_df = pd.read_json(train_path)


    train_df["classification"] = train_df["classification"].astype("category")
    train_df["classification"] = train_df["classification"].cat.set_categories(["EASY", "NON-NESTED", "NESTED"], ordered=True)
    train_df = train_df.sort_values("classification")
    

    print(train_df.head(10))

    print(train_df["classification"].unique())
    train_df = process_dataset(train_df)
    return Dataset.from_dict(
    {
        "text": [sample["text"] for sample in train_df],
        "labels": [sample["labels"] for sample in train_df],
    }
)

def get_eval_dataset(difficulty: str = "easy") -> DatasetDict:
    eval_path = "val.json"

    eval_df = pd.read_json(eval_path)

    if difficulty=="easy":
        eval_df = process_dataset(eval_df[eval_df["classification"] == "EASY"])

    if difficulty=="medium":
        eval_df = process_dataset(eval_df[eval_df["classification"] == "NON-NESTED"])

    return Dataset.from_dict(
    {
        "text": [sample["text"] for sample in eval_df],
        "labels": [sample["labels"] for sample in eval_df],
    }
)


if __name__ == "__main__":
    df = pd.read_json(
        "best_train.json"
    )

    print(df.columns)

    # Remove entries with invalid reasoning
    df = df[df["score"] == 1]
    print(f"Dataset size: {len(df)}")

    # Get a sample from the dataset
    print(process_dataset(df[:1])[0])

    get_dataset()

    #df1 = pd.read_json(
    #    "val.json"
    #)
    #print(df1.columns)

    #df1 = df1[df1["score"] == 1]
    #print(f"Validation dataset size: {len(df1)}")

    #print(process_dataset(df1[:1])[0])
