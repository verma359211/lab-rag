"""Apply the small, idempotent SQL migrations in the migrations folder.

Run this module from ``backend-python`` with:

    python -m scripts.apply_migrations

Each migration uses operations such as ``IF NOT EXISTS``, so running this
command again is safe for this demonstration project.
"""

from pathlib import Path

import psycopg

from app.config import get_psycopg_database_url
from app.vector_store import get_vector_store


MIGRATIONS_FOLDER = Path(__file__).resolve().parents[1] / "migrations"


def apply_migrations() -> None:
    """Create LangChain's tables if needed, then apply each SQL file."""

    # PGVector creates its collection and tables when this object is first
    # initialized. The keyword index can only be created after that table exists.
    get_vector_store()

    migration_files = sorted(MIGRATIONS_FOLDER.glob("*.sql"))

    with psycopg.connect(get_psycopg_database_url()) as connection:
        for migration_file in migration_files:
            sql = migration_file.read_text(encoding="utf-8")
            connection.execute(sql)
            print(f"Applied {migration_file.name}")


if __name__ == "__main__":
    apply_migrations()
