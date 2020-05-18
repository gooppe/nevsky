import os
from contextlib import closing
from typing import List, Tuple

import psycopg2


def select_translation(word: str, table="ru_en") -> List[str]:
    """Select translation for word.

    Args:
        word (str): desired word.
        table (str, optional): translation table. Defaults to "ru_en".

    Returns:
        List[str]: possible translations.
    """
    DATABASE_URL = os.environ["DATABASE_URL"]

    with closing(psycopg2.connect(DATABASE_URL, sslmode="require")) as conn:
        with conn.cursor() as cursor:
            sql = f"SELECT translation FROM {table} WHERE word=%s"
            cursor.execute(sql, (word,))
            return [row[0] for row in cursor]


def take_random_words(n: int, table="ru_en") -> List[Tuple[str, str]]:
    """Take n random words from table.
    Args:
        n (int): number of words.
    Returns:
        List[Tuple[str, str]]: random words.
    """

    DATABASE_URL = os.environ["DATABASE_URL"]

    with closing(psycopg2.connect(DATABASE_URL, sslmode="require")) as conn:
        with conn.cursor() as cursor:
            sql = f"SELECT word, translation FROM {table} ORDER BY random() LIMIT {n}"
            cursor.execute(sql)
            return [row[0:2] for row in cursor]
