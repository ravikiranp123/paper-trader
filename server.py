import os
from paper_trader_mcp import main

db_type = os.getenv("DB_TYPE", "sqlite")
db_name = os.getenv("DB_NAME", "default_trader")
transport = os.getenv("TRANSPORT", "streamable-http")

main(db_type, db_name, transport)