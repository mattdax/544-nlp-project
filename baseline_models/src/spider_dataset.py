import csv
import json

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
        self.labels = [query["sql"] for query in sql_queries]

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item, self.labels[idx]

    def __len__(self):
        return len(self.labels)


def get_spider_devset(tokenizer, json_path, schema_path) -> SpiderDataset:

    # Parse database schema
    # Schema parsing adapted from text2sql-data/tools/spider_schema_to_sqlite.py

    databases = {}

    with open(schema_path) as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader, None)  # read the header

        for line in reader:
            row = dict(zip(header, line))

            db = row["Database name"].lower()
            table = row["Table Name"].lower()
            column = row["Field Name"].lower()
            column_type = row["Type"]
            column_primray = row["Is Primary Key"]
            if db not in databases:
                databases[db] = {"tables": {}}
            if table not in databases[db]["tables"]:
                databases[db]["tables"][table] = {"columns": {}, "primary": []}
            if column not in databases[db]["tables"][table]["columns"]:
                databases[db]["tables"][table]["columns"][column] = {
                    "name": column,
                    "type": column_type,
                }
                if column_primray == "True":
                    databases[db]["tables"][table]["primary"].append(column)

    db_schema_str = {}
    for db in databases:
        schema_str = ""
        for table in databases[db]["tables"]:
            if "sqlite_sequence" == table:
                continue
            tablesql = "CREATE TABLE " + table + "("
            coldelim = " "
            for col in databases[db]["tables"][table]["columns"]:
                col = databases[db]["tables"][table]["columns"][col]
                tablesql += coldelim + '"' + col["name"] + '" ' + col["type"]
                coldelim = ", "
            if len(databases[db]["tables"][table]["primary"]):
                tablesql += (
                    ", PRIMARY KEY ("
                    + ",".join(databases[db]["tables"][table]["primary"])
                    + ")"
                )
            tablesql += " );"

            schema_str += tablesql + "\n"

        db_schema_str[db] = schema_str.rstrip()

    # Parse Spider devset
    # JSON parsing adapted from text2sql-data/tools/json_to_flat.py
    dev_queries = []

    with open(json_path) as f:
        data = json.loads(f.read())

        for sample in data:
            var_sql = sample["sql"][0]
            for sentence in sample["sentences"]:
                # Only include dev questions in the dataset
                if sentence["question-split"] == "dev":
                    text = sentence["original"]
                    sql = var_sql  # Needed to do variable replacement correctly

                    # Variable replacement
                    for name in sentence["variables"]:
                        value = sentence["variables"][name]
                        if len(value) == 0:
                            for variable in sample["variables"]:
                                if variable["name"] == name:
                                    value = variable["example"]
                        #     text = value.join(text.split(name))
                        sql = value.join(sql.split(name))

                    dev_queries.append(
                        {
                            "database": sentence["database"].lower(),
                            "question": text,
                            "sql": sql,
                        }
                    )

    dev_set = SpiderDataset(tokenizer, dev_queries, db_schema_str)

    return dev_set


# if __name__ == "__main__":

#     model = "seeklhy/codes-1b"
#     # model = 'distilbert-base-uncased'

#     tokenizer = AutoTokenizer.from_pretrained(model)

#     dataset = get_spider_devset(
#         tokenizer,
#         json_path="./benchmarks/text2sql-data/data/spider.json",
#         schema_path="./benchmarks/text2sql-data/data/spider-schema.csv",
#     )
