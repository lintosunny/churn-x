from urllib.parse import quote_plus
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv(override=True)


def get_postgres_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_NAME"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD")
    )