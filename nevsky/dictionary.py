import os
from contextlib import closing
from typing import List

import psycopg2

DATABASE_URL = os.environ["DATABASE_URL"]


def select_translation(word: str, table="ru_en") -> List[str]:
    """Select translation for word.
    Args:
        word (str): desired word.
        table (str, optional): translation table. Defaults to "ru_en".
    Returns:
        List[str]: possible translations.
    """
    with closing(psycopg2.connect(DATABASE_URL, sslmode="require")) as conn:
        with conn.cursor() as cursor:
            sql = f"SELECT translation FROM {table} WHERE word=%s"
            cursor.execute(sql, (word,))
            return [row[0] for row in cursor]