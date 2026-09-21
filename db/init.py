"""Database initialization and connection management."""

import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/fraud_detection")

def get_engine():
    """Get SQLAlchemy engine."""
    return create_engine(DATABASE_URL)

def init_db(schema_path: str = None):
    """Run schema.sql to initialize tables."""
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.sql"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    engine = get_engine()
    with engine.connect() as conn:
        for statement in schema_sql.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
        conn.commit()
    print("Database tables successfully initialized.")

if __name__ == "__main__":
    init_db()
