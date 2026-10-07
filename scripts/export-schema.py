import sys
from pathlib import Path
from sqlalchemy.schema import CreateTable, CreateIndex
from sqlalchemy.dialects import postgresql

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.db import Base

print("CREATE EXTENSION IF NOT EXISTS vector;\n")
for table in Base.metadata.sorted_tables:
    print(
        "\n".join(
            line.rstrip()
            for line in str(
                CreateTable(table).compile(dialect=postgresql.dialect())
            ).splitlines()
        )
        + ";"
    )
    for index in table.indexes:
        print(str(CreateIndex(index).compile(dialect=postgresql.dialect())) + ";")
