import json

from typing import Dict, List

from torch.utils.data import Dataset

prompt = """\
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

### Database Schema
This query will run on a database whose schema is represented in this string:
{schema}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""


def build_prompt(query, schemas) -> str:
    """
    Generate Text2SQL prompt containing database schema and natural language query

    TODO: make this prompt customizable
    """
    question = query["question"]
    schema = schemas[query["database"]]
    question_prompt = prompt.format(question=question, schema=schema)
    return question_prompt


class SpiderDataset(Dataset):
    def __init__(self, tokenizer, sql_queries, schemas) -> None:
        super().__init__()
        prompts = [build_prompt(query, schemas) for query in sql_queries]
        self.encodings = tokenizer(
            prompts, truncation=True, padding=True, max_length=4096, return_tensors="pt"
        )
        self.labels = sql_queries

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item, self.labels[idx]

    def __len__(self):
        return len(self.labels)


def parse_spider_schemas(schema_path) -> Dict[str, str]:
    databases = {}

    with open(schema_path) as f:
        data = json.loads(f.read())

        for db in data:
            db_name = db["db_id"]

            # Schema strings formatted in the style of codes/utils/db_utils.py
            schema = ""
            for i, table_name in enumerate(db["table_names_original"]):
                column_info_list = []
                for j, column in enumerate(db["column_names_original"]):
                    if column[0] == i:
                        column_name = column[1]
                        additional_column_info = []
                        additional_column_info.append(db["column_types"][j])

                        if j in db["primary_keys"]:
                            additional_column_info.append("primary key")

                        column_info_list.append(
                            table_name
                            + "."
                            + column_name
                            + " ( "
                            + " | ".join(additional_column_info)
                            + " )"
                        )

                schema += (
                    "table "
                    + table_name
                    + " , columns = [ "
                    + " , ".join(column_info_list)
                    + " ]\n"
                )

            if len(db["foreign_keys"]) != 0:
                schema += "foreign keys :\n"
                for foreign_key in db["foreign_keys"]:
                    col1 = db["column_names_original"][foreign_key[0]]
                    col2 = db["column_names_original"][foreign_key[1]]

                    schema += f"{db['table_names_original'][col1[0]]}.{col1[1]} = {db['table_names_original'][col2[0]]}.{col2[1]}\n"
            else:
                schema += "foreign keys : None\n"

            # print(schema)
            databases[db_name] = schema.strip()

    return databases


def parse_spider_queries(query_path) -> List[Dict[str, str]]:
    queries = []

    with open(query_path) as f:
        data = json.loads(f.read())

        for sample in data:
            queries.append(
                {
                    "database": sample["db_id"],
                    "question": sample["question"],
                    "sql": sample["query"],
                }
            )

    return queries


def get_spider_dataset(tokenizer, json_path, schema_path) -> SpiderDataset:

    # Parse database schema
    db_schema_str = parse_spider_schemas(schema_path)

    # Parse spider queries
    queries = parse_spider_queries(json_path)

    dataset = SpiderDataset(tokenizer, queries, db_schema_str)

    return dataset


def get_spider_devset(tokenizer):
    return get_spider_dataset(
        tokenizer,
        json_path="./benchmarks/spider_data/dev.json",
        schema_path="./benchmarks/spider_data/tables.json",
    )


def get_spider_testset(tokenizer):
    return get_spider_dataset(
        tokenizer,
        json_path="./benchmarks/spider_data/test.json",
        schema_path="./benchmarks/spider_data/test_tables.json",
    )
