import os
from argparse import ArgumentParser
from contextlib import closing

import psycopg2
from lxml import etree


def export_dictionary(dictfile: str, table_name: str):
    dictionary = list()

    with open(dictfile) as file:
        xml = etree.parse(file)
        root = xml.getroot()

        for (word,) in root.findall("ar"):
            ru_word = word.text.strip()
            en_word = word.tail.split('"')[-1].strip().lower()
            dictionary.append((ru_word, en_word))

    DATABASE_URL = os.environ("DATABASE_URL")
    CREATE_TABLE = f"""
            CREATE TABLE {table_name} (
                translation_id SERIAL PRIMARY KEY,
                word VARCHAR(30) NOT NULL,
                translation VARCHAR(30) NOT NULL
            )"""
    INSERT = f"INSERT INTO {table_name}(word, translation) VALUES(%s, %s)"

    with closing(psycopg2.connect(DATABASE_URL, sslmode="require")) as conn:
        with conn.cursor() as cursor:
            cursor.execute(CREATE_TABLE)
            cursor.executemany(INSERT, dictionary)
            conn.commit()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--df", type=str, required=True, help=".xdxf dictionary file")
    parser.add_argument("--table", type=str, required=True, help="database table name")
    args = parser.parse_args()

    export_dictionary(args.df, args.table)
