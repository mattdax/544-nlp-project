import os
import sys
import re

from typing import List, Dict

import pandas as pd

module_path = os.path.abspath(os.path.join("./text2sql-schema-filter"))
sys.path.append(module_path)

from schema_filter import filter_func, SchemaItemClassifierInference


# Parameters used by CodeS
num_top_k_tables = 6
num_top_k_columns = 10


def parse_table(table_str: str) -> List[Dict]:
    tables = []
    for t in table_str.strip().split("\n"):
        str_match = re.search(r"Table ([a-zA-Z_]+), columns = \[\*,(.+)\]", t)
        table_name = str_match.group(1)
        column_names = str_match.group(2).split(",")
        table = {
            "table_name": table_name,
            "table_comment": "",
            "column_names": column_names,
            "column_comments": ["" for _ in range(len(column_names))],
        }
        tables.append(table)

    return tables


def parse_foreign_keys(foreign_key_str: str) -> List[List]:
    foreign_keys = re.search(r"\[(.*)\]", foreign_key_str).group(1).split(",")
    return [re.split(r" = |\.", foreign_key) for foreign_key in foreign_keys]


def process_row(row: pd.Series):
    # print(row["foriegn keys"])
    data = {
        "text": row["question"],
        "sql": "",
        "schema": {"schema_items": parse_table(row["fields"])},
    }
    return data


def get_db_schema_sequence(schema, foreign_keys):
    schema_sequence = ""
    for table in schema["schema_items"]:
        table_name, table_comment = table["table_name"], table["table_comment"]

        if table_comment != "":
            table_name += " ( comment : " + table_comment + " )"

        column_info_list = [
            table_name + "." + column_name for column_name in table["column_names"]
        ]

        schema_sequence += (
            "table "
            + table_name
            + " , columns = [ "
            + " , ".join(column_info_list)
            + " ]\n"
        )

    foreign_keys = re.search(r"\[(.*)\]", foreign_keys).group(1)
    if len(foreign_keys) != 0:
        schema_sequence += "foreign keys :\n"
        foreign_keys = foreign_keys.split(",")
        for foreign_key in foreign_keys:
            schema_sequence += foreign_key
    else:
        schema_sequence += "foreign keys : None\n"

    return schema_sequence.strip()


def main(path: str):
    df = pd.read_json(path)
    # df = df[:2]

    dataset = [process_row(row) for _, row in df.iterrows()]
    # print(dataset)

    # values used by CodeS model
    num_top_k_tables = 6
    num_top_k_columns = 10

    # load fine-tuned schema filter
    sic = SchemaItemClassifierInference("sic_merged")
    dataset = filter_func(
        dataset=dataset,
        dataset_type="eval",
        sic=sic,
        num_top_k_tables=num_top_k_tables,
        num_top_k_columns=num_top_k_columns,
    )

    # print(dataset)
    filtered_schema = [
        get_db_schema_sequence(schema["schema"], df.iloc[i]["foriegn keys"])
        for i, schema in enumerate(dataset)
    ]
    df["filtered_schema"] = filtered_schema
    return df


if __name__ == "__main__":
    train_path = "./dataset/full_train_gpt4o_mini_query_gen_gpt4o_query_correction.json"
    train_df = main(train_path)
    train_df.to_json("train.json")
    del train_df

    val_path = "./dataset/full_val_gpt4o_mini_query_gen_gpt4o_query_correction.json"
    val_df = main(val_path)
    val_df.to_json("val.json")
