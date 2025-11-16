# Save this file as: database.py
# This version is updated to use PostgreSQL.
import logging
import os
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    CheckConstraint,
    func,
    select,
    insert,
    text,
    update,
    delete,
    desc,
)
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Setup ---
logger = logging.getLogger(__name__)

metadata = MetaData()


def get_engine(db_name, db_type="postgres"):
    """Creates a new SQLAlchemy engine to ensure a fresh connection."""
    if db_type == "sqlite":
        db_path = os.path.join("db", f"{db_name}.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return create_engine(f"sqlite:///{db_path}")
    else:  # postgres
        base_url = os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost:5432/"
        )
        try:
            engine = create_engine(base_url + db_name)
            # Test connection to the database
            with engine.connect():
                pass
        except Exception:
            # If the database doesn't exist, create it
            postgres_engine = create_engine(
                base_url + "postgres", isolation_level="AUTOCOMMIT"
            )
            with postgres_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE {db_name}"))
            # Reconnect to the newly created database
            engine = create_engine(base_url + db_name)
        return engine


# --- Table Definitions (No changes needed) ---
portfolio_table = Table(
    "portfolio",
    metadata,
    Column("id", Integer, primary_key=True, default=1),
    Column("cash", Float, nullable=False),
)
holdings_table = Table(
    "holdings",
    metadata,
    Column("ticker", String, primary_key=True),
    Column("quantity", Integer, nullable=False),
    Column("avg_price", Float, nullable=False),
)
transactions_table = Table(
    "transactions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), server_default=func.now()),
    Column("ticker", String, nullable=False),
    Column("action", String, nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("price_per_share", Float),
    Column("total_amount", Float),
    Column("status", String),
    Column("notes", String),
    Column("reason", String),
)


history_table = Table(
    "history",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), server_default=func.now()),
    Column("cash_value", Float, nullable=False),
    Column("stocks_value", Float, nullable=False),
    Column("combined_value", Float, nullable=False),
    Column("run_id", String),
    Column("notes", String),
    CheckConstraint("cash_value >= 0", name="chk_cash_value_non_negative"),
    CheckConstraint("stocks_value >= 0", name="chk_stocks_value_non_negative"),
    CheckConstraint("combined_value >= 0", name="chk_combined_value_non_negative"),
    CheckConstraint(
        "combined_value = cash_value + stocks_value",
        name="chk_combined_value_equals_sum",
    ),
)


# --- Helper (No changes needed) ---
def _row_to_dict(row):
    if row is None:
        return None
    return {key: value for key, value in row._mapping.items()}


# --- Public Database Functions (Updated to use get_engine) ---
def initialize_database(
    database_name: str, db_type: str = "postgres", initial_cash=100000.0
):
    engine = get_engine(database_name, db_type)
    metadata.create_all(engine)
    with engine.connect() as conn:
        with conn.begin():
            if conn.execute(select(portfolio_table)).first() is None:
                conn.execute(insert(portfolio_table).values(id=1, cash=initial_cash))
    logger.info("Database initialized successfully.")


def get_portfolio(database_name: str, db_type: str = "postgres"):
    engine = get_engine(database_name, db_type)
    with engine.connect() as conn:
        cash_row = conn.execute(
            select(portfolio_table.c.cash).where(portfolio_table.c.id == 1)
        ).scalar()
        holdings_rows = conn.execute(select(holdings_table)).fetchall()
        return {
            "cash": cash_row if cash_row is not None else 0,
            "holdings": [_row_to_dict(h) for h in holdings_rows],
        }


def log_transaction(database_name: str, db_type: str = "postgres", **kwargs):
    engine = get_engine(database_name, db_type)
    with engine.connect() as conn:
        kwargs["total_amount"] = kwargs.get("price_per_share", 0) * kwargs.get(
            "quantity", 0
        )
        conn.execute(insert(transactions_table).values(**kwargs))
        conn.commit()


def buy_stock(
    database_name: str, db_type: str, ticker: str, quantity: int, price: float
):
    cost = price * quantity
    engine = get_engine(database_name, db_type)
    with engine.connect() as conn:
        with conn.begin():
            cash_balance = conn.execute(
                select(portfolio_table.c.cash).where(portfolio_table.c.id == 1)
            ).scalar_one()
            if cash_balance < cost:
                return {
                    "error": f"Insufficient funds. Need {cost:.2f}, but only have {cash_balance:.2f}."
                }

            conn.execute(
                update(portfolio_table)
                .where(portfolio_table.c.id == 1)
                .values(cash=portfolio_table.c.cash - cost)
            )
            existing = conn.execute(
                select(holdings_table).where(holdings_table.c.ticker == ticker)
            ).first()
            if existing:
                new_qty = existing.quantity + quantity
                new_avg = ((existing.avg_price * existing.quantity) + cost) / new_qty
                conn.execute(
                    update(holdings_table)
                    .where(holdings_table.c.ticker == ticker)
                    .values(quantity=new_qty, avg_price=new_avg)
                )
            else:
                conn.execute(
                    insert(holdings_table).values(
                        ticker=ticker, quantity=quantity, avg_price=price
                    )
                )
            return {
                "status": "success",
                "message": f"Bought {quantity} of {ticker} at {price:.2f} each.",
            }


def sell_stock(
    database_name: str, db_type: str, ticker: str, quantity: int, price: float
):
    engine = get_engine(database_name, db_type)
    with engine.connect() as conn:
        with conn.begin():
            holding = conn.execute(
                select(holdings_table).where(holdings_table.c.ticker == ticker)
            ).first()
            if not holding or holding.quantity < quantity:
                held = holding.quantity if holding else 0
                notes = f"Insufficient holdings. Tried to sell {quantity}, but only hold {held}."
                return {"error": notes}

            proceeds = price * quantity
            conn.execute(
                update(portfolio_table)
                .where(portfolio_table.c.id == 1)
                .values(cash=portfolio_table.c.cash + proceeds)
            )
            new_qty = holding.quantity - quantity
            if new_qty == 0:
                conn.execute(
                    delete(holdings_table).where(holdings_table.c.ticker == ticker)
                )
            else:
                conn.execute(
                    update(holdings_table)
                    .where(holdings_table.c.ticker == ticker)
                    .values(quantity=new_qty)
                )
            return {
                "status": "success",
                "message": f"Sold {quantity} of {ticker} at {price:.2f} each.",
            }


def get_transactions(database_name: str, db_type: str = "postgres", limit: int = 20):
    """Retrieves the most recent transactions from the database."""
    engine = get_engine(database_name, db_type)
    with engine.connect() as conn:
        query = (
            select(transactions_table)
            .order_by(desc(transactions_table.c.timestamp))
            .limit(limit)
        )
        results = conn.execute(query).fetchall()
        return [
            {**_row_to_dict(row), "timestamp": row.timestamp.isoformat()}
            for row in results
        ]


def insert_history_record(database_name: str, db_type: str = "postgres", **kwargs):
    """Inserts a new record into the history table."""
    engine = get_engine(database_name, db_type)
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(insert(history_table).values(**kwargs))
    logger.info("History record inserted successfully.")


def get_history_records(database_name: str, db_type: str = "postgres", limit: int = 50):
    """Retrieves history records from the database."""
    engine = get_engine(database_name, db_type)
    with engine.connect() as conn:
        query = (
            select(history_table).order_by(desc(history_table.c.timestamp)).limit(limit)
        )
        results = conn.execute(query).fetchall()
        return [
            {**_row_to_dict(row), "timestamp": row.timestamp.isoformat()}
            for row in results
        ]


def reset_database(database_name: str, db_type: str = "postgres"):
    """Drops all tables and recreates them."""
    engine = get_engine(database_name, db_type)
    with engine.connect() as conn:
        metadata.drop_all(engine)
        metadata.create_all(engine)
        initialize_database(database_name, db_type)
    logger.info("Database has been reset.")


def migrate_database(database_name: str, db_type: str = "postgres"):
    engine = get_engine(database_name, db_type)
    with engine.connect() as conn:
        metadata.create_all(engine)
    logger.info("Database has been migrated.")
