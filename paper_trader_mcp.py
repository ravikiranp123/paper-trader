import asyncio
import logging
import yfinance as yf
from typing import List, Dict, Any, Optional
from enum import Enum
from fastmcp import Context, FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
import talib
import numpy as np
from talib.abstract import Function
import argparse
from datetime import datetime, time as datetime_time
import pytz

# --- Setup ---
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

import database

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)
mcp = FastMCP(name="PaperTrader")

# --- Market Context Configuration ---
MARKET_CONTEXT = {
    "IN": {
        "name": "Indian Stock Market (NSE/BSE)",
        "currency": "INR",
        "indices": "NIFTY 50 and SENSEX",
        "market_close": "15:30",
        "timezone": "Asia/Kolkata",
    },
    "US": {
        "name": "US Stock Market (NASDAQ/NYSE)",
        "currency": "USD",
        "indices": "S&P 500 and Nasdaq Composite",
        "market_close": "16:00",
        "timezone": "America/New_York",
    },
    "SE": {
        "name": "Shanghai Stock Exchange",
        "currency": "CNY",
        "indices": "SSE Composite Index",
        "market_close": "15:00",
        "timezone": "Asia/Shanghai",
    },
    "SZSE": {
        "name": "Shenzhen Stock Exchange",
        "currency": "CNY",
        "indices": "SZSE Composite Index",
        "market_close": "15:00",
        "timezone": "Asia/Shanghai",
    },
    "EURONEXT": {
        "name": "Euronext",
        "currency": "EUR",
        "indices": "Euronext 100",
        "market_close": "17:30",
        "timezone": "Europe/Paris",
    },
    "JPX": {
        "name": "Japan Exchange Group",
        "currency": "JPY",
        "indices": "Nikkei 225",
        "market_close": "15:00",
        "timezone": "Asia/Tokyo",
    },
    "LSE": {
        "name": "London Stock Exchange",
        "currency": "GBP",
        "indices": "FTSE 100",
        "market_close": "16:30",
        "timezone": "Europe/London",
    },
    "HKEX": {
        "name": "Hong Kong Exchanges and Clearing",
        "currency": "HKD",
        "indices": "Hang Seng Index",
        "market_close": "16:00",
        "timezone": "Asia/Hong_Kong",
    },
}


# --- Internal Helper ---
def _fetch_market_data_from_yfinance(ticker: str) -> dict:
    """Internal helper to fetch a curated summary of market data from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or info.get("symbol") is None:
            logger.error(
                f"No data found for ticker '{ticker}'. It may be an invalid symbol."
            )
            return {
                "error": f"No data found for ticker '{ticker}'. It may be an invalid symbol."
            }
        market_data = {
            "symbol": info.get("symbol"),
            "longName": info.get("longName"),
            "currency": info.get("currency"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
            "dayHigh": info.get("dayHigh"),
            "dayLow": info.get("dayLow"),
            "volume": info.get("volume"),
            "marketCap": info.get("marketCap"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "marketState": info.get("marketState"),
        }
        if market_data["currentPrice"] is None:
            logger.error(
                f"A valid price for {ticker} could not be found. Market may be closed."
            )
            return {
                "error": f"A valid price for {ticker} could not be found. Market may be closed."
            }
        logger.info(f"Successfully fetched market data for {ticker}.")
        return market_data
    except Exception as e:
        logger.error(f"yfinance error for ticker '{ticker}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while fetching data: {str(e)}"}


# --- Core Trading & Portfolio Tools ---
@mcp.tool()
async def get_market_data(ticker: str) -> dict:
    """Gets a summary of market data for a stock (price, volume, daily/yearly range, etc.)."""
    return await asyncio.to_thread(_fetch_market_data_from_yfinance, ticker)


@mcp.tool()
async def get_market_status(ticker: str, region: str) -> dict:
    """Gets the market status for a given ticker and calculates time to market close."""
    market_data = await asyncio.to_thread(_fetch_market_data_from_yfinance, ticker)
    logger.info(market_data)
    market_state = market_data.get("marketState")

    time_to_close = "N/A"
    if market_state == "REGULAR":
        try:
            market_info = MARKET_CONTEXT[region.upper()]
            tz = pytz.timezone(market_info["timezone"])
            now = datetime.now(tz)
            close_time = datetime.strptime(market_info["market_close"], "%H:%M").time()
            close_datetime = now.replace(
                hour=close_time.hour, minute=close_time.minute, second=0, microsecond=0
            )

            if now < close_datetime:
                time_diff = close_datetime - now
                hours, remainder = divmod(time_diff.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                time_to_close = f"{hours} hours and {minutes} minutes"
            else:
                time_to_close = "Market is closing soon or already closed."
        except Exception as e:
            logger.error(f"Error calculating time to market close: {e}", exc_info=True)
            time_to_close = "Error calculating time to close."

    return {"market_state": market_state, "time_to_close": time_to_close}


@mcp.tool()
async def get_portfolio(ctx: Context) -> dict:
    """Retrieves the current portfolio, showing cash and all stock holdings."""
    logger.info("Fetching portfolio data...")
    logger.info(f"Using database name: {ctx.fastmcp.database_name}")
    return await asyncio.to_thread(
        database.get_portfolio, ctx.fastmcp.database_name, ctx.fastmcp.db_type
    )


# @mcp.tool()
async def get_portfolio_summary(database_name, db_type) -> dict:
    """Returns a complete summary of the portfolio's value (holdings + cash)."""
    logger.info("Generating portfolio summary...")
    portfolio = await asyncio.to_thread(database.get_portfolio, database_name, db_type)
    cash_balance = portfolio.get("cash", 0)
    holdings_list = portfolio.get("holdings", [])
    total_holdings_value = 0.0

    for holding in holdings_list:
        ticker = holding.get("ticker")
        quantity = holding.get("quantity")
        if ticker and quantity:
            logger.info(f"Fetching market data for {ticker} for portfolio summary.")
            market_data = await asyncio.to_thread(
                _fetch_market_data_from_yfinance, ticker
            )
            if "error" not in market_data and market_data.get("currentPrice"):
                total_holdings_value += market_data["currentPrice"] * quantity
            else:
                logger.warning(
                    f"Could not get current price for {ticker}: {market_data.get('error', 'Unknown error')}"
                )

    total_portfolio_value = total_holdings_value + cash_balance

    summary = {
        "total_holdings_value": total_holdings_value,
        "cash_balance": cash_balance,
        "total_portfolio_value": total_portfolio_value,
    }
    logger.info(f"Portfolio summary generated: {summary}")
    return summary


@mcp.tool()
async def get_financial_statements(ticker: str, statement_type: str) -> dict:
    """Fetches financial statements (income_stmt, balance_sheet, or cash_flow) for a given ticker."""
    logger.info(f"Fetching {statement_type} for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        if statement_type == "income_stmt":
            data = stock.income_stmt
        elif statement_type == "balance_sheet":
            data = stock.balance_sheet
        elif statement_type == "cash_flow":
            data = stock.cash_flow
        else:
            return {
                "error": "Invalid statement_type. Choose from 'income_stmt', 'balance_sheet', 'cash_flow'."
            }

        if data.empty:
            logger.warning(f"No {statement_type} found for {ticker}.")
            return {"error": f"No {statement_type} found for {ticker}."}

        # Convert DataFrame to dictionary for JSON serialization
        result = data.to_dict()
        logger.info(f"Successfully fetched {statement_type} for {ticker}.")
        return result
    except Exception as e:
        logger.error(
            f"Error fetching {statement_type} for {ticker}: {e}", exc_info=True
        )
        return {"error": f"Failed to fetch {statement_type} for {ticker}: {str(e)}"}


@mcp.tool()
async def get_key_statistics(ticker: str) -> dict:
    """Retrieves key financial statistics for a given ticker."""
    logger.info(f"Fetching key statistics for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info:
            logger.warning(f"No key statistics found for {ticker}.")
            return {"error": f"No key statistics found for {ticker}."}

        # Filter for common key statistics
        key_stats = {
            "symbol": info.get("symbol"),
            "longName": info.get("longName"),
            "marketCap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "dividendYield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "volume": info.get("volume"),
            "averageVolume": info.get("averageVolume"),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "heldPercentInstitutions": info.get("heldPercentInstitutions"),
            "heldPercentInsiders": info.get("heldPercentInsiders"),
        }
        logger.info(f"Successfully fetched key statistics for {ticker}.")
        return {
            k: v for k, v in key_stats.items() if v is not None
        }  # Remove None values
    except Exception as e:
        logger.error(f"Error fetching key statistics for {ticker}: {e}", exc_info=True)
        return {"error": f"Failed to fetch key statistics for {ticker}: {str(e)}"}


@mcp.tool()
async def get_analyst_recommendations(ticker: str) -> list:
    """Fetches analyst recommendations for a given ticker."""
    logger.info(f"Fetching analyst recommendations for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        if recommendations.empty:
            logger.warning(f"No analyst recommendations found for {ticker}.")
            return []

        # Convert DataFrame to list of dictionaries
        result = recommendations.to_dict(orient="records")
        logger.info(f"Successfully fetched analyst recommendations for {ticker}.")
        return result
    except Exception as e:
        logger.error(
            f"Error fetching analyst recommendations for {ticker}: {e}", exc_info=True
        )
        return {
            "error": f"Failed to fetch analyst recommendations for {ticker}: {str(e)}"
        }


@mcp.tool()
async def get_company_news(ticker: str) -> list:
    """Retrieves recent news articles for a given company."""
    logger.info(f"Fetching company news for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            logger.warning(f"No news found for {ticker}.")
            return []
        logger.info(f"Successfully fetched company news for {ticker}.")
        return news
    except Exception as e:
        logger.error(f"Error fetching company news for {ticker}: {e}", exc_info=True)
        return {"error": f"Failed to fetch company news for {ticker}: {str(e)}"}


@mcp.tool()
async def get_dividends_splits(ticker: str) -> dict:
    """Fetches dividend payments and stock splits history for a given ticker."""
    logger.info(f"Fetching dividends and splits for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        dividends = stock.dividends
        splits = stock.splits

        result = {
            "dividends": (
                dividends.to_dict(orient="records") if not dividends.empty else []
            ),
            "splits": splits.to_dict(orient="records") if not splits.empty else [],
        }
        logger.info(f"Successfully fetched dividends and splits for {ticker}.")
        return result
    except Exception as e:
        logger.error(
            f"Error fetching dividends and splits for {ticker}: {e}", exc_info=True
        )
        return {"error": f"Failed to fetch dividends and splits for {ticker}: {str(e)}"}


@mcp.tool()
async def buy(
    ticker: str, quantity: int, reason: str = None, ctx: Context = None
) -> dict:
    """Executes a buy order for a specified quantity of a stock."""
    logger.info(f"[BUY] Attempting to BUY {quantity} of {ticker}.")
    logger.info(f"[BUY] Fetching market data for {ticker}...")
    market_data = await asyncio.to_thread(_fetch_market_data_from_yfinance, ticker)
    if "error" in market_data:
        logger.error(
            f"[BUY] Market data fetch failed for {ticker}: {market_data['error']}"
        )
        await asyncio.to_thread(
            database.log_transaction,
            ctx.fastmcp.database_name,
            ctx.fastmcp.db_type,
            ticker=ticker,
            action="BUY",
            quantity=quantity,
            status="FAILED",
            notes=market_data["error"],
        )
        return market_data
    price = market_data["currentPrice"]
    logger.info(f"[BUY] Market data fetched for {ticker}. Current price: {price:.2f}.")
    logger.info(f"[BUY] Attempting to log transaction and execute buy for {ticker}...")
    result = await asyncio.to_thread(
        database.buy_stock,
        ctx.fastmcp.database_name,
        ctx.fastmcp.db_type,
        ticker,
        quantity,
        price,
    )
    if result.get("status") == "success":
        logger.info(
            f"[BUY] Successfully executed BUY order for {quantity} of {ticker} at {price:.2f}."
        )
    else:
        logger.error(f"[BUY] BUY order failed for {ticker}: {result.get('message')}")
    await asyncio.to_thread(
        database.log_transaction,
        ctx.fastmcp.database_name,
        ctx.fastmcp.db_type,
        ticker=ticker,
        action="BUY",
        quantity=quantity,
        price_per_share=price,
        status=result.get("status", "UNKNOWN"),
        notes=result.get("message"),
        reason=reason,
    )
    logger.info(
        f"[BUY] Transaction logged for {ticker}. Result: {result.get('status')}"
    )
    return result


@mcp.tool()
async def sell(
    ticker: str, quantity: int, reason: str = None, ctx: Context = None
) -> dict:
    """Executes a sell order for a specified quantity of a stock."""
    logger.info(f"[SELL] Attempting to SELL {quantity} of {ticker}.")
    logger.info(f"[SELL] Fetching market data for {ticker}...")
    market_data = await asyncio.to_thread(_fetch_market_data_from_yfinance, ticker)
    if "error" in market_data:
        logger.error(
            f"[SELL] Market data fetch failed for {ticker}: {market_data['error']}"
        )
        await asyncio.to_thread(
            database.log_transaction,
            ctx.fastmcp.database_name,
            ctx.fastmcp.db_type,
            ticker=ticker,
            action="SELL",
            quantity=quantity,
            status="FAILED",
            notes=market_data["error"],
        )
        return market_data
    price = market_data["currentPrice"]
    logger.info(f"[SELL] Market data fetched for {ticker}. Current price: {price:.2f}.")
    logger.info(
        f"[SELL] Attempting to log transaction and execute sell for {ticker}..."
    )
    result = await asyncio.to_thread(
        database.sell_stock,
        ctx.fastmcp.database_name,
        ctx.fastmcp.db_type,
        ticker,
        quantity,
        price,
    )
    if result.get("status") == "success":
        logger.info(
            f"[SELL] Successfully executed SELL order for {quantity} of {ticker} at {price:.2f}."
        )
    else:
        logger.error(f"[SELL] SELL order failed for {ticker}: {result.get('message')}")
    await asyncio.to_thread(
        database.log_transaction,
        ctx.fastmcp.database_name,
        ctx.fastmcp.db_type,
        ticker=ticker,
        action="SELL",
        quantity=quantity,
        price_per_share=price,
        status=result.get("status", "UNKNOWN"),
        notes=result.get("message"),
        reason=reason,
    )
    logger.info(
        f"[SELL] Transaction logged for {ticker}. Result: {result.get('status')}"
    )
    return result


# --- Technical Analysis Tools ---
@mcp.tool()
async def get_all_indicator_details() -> Dict[str, Any]:
    """Provides a complete 'recipe book' of all available TA-Lib indicators."""
    logger.info("Tool 'get_all_indicator_details' called")
    groups = talib.get_function_groups()
    detailed_indicators = {}
    for group, indicator_list in groups.items():
        detailed_indicators[group] = []
        for indicator_name in indicator_list:
            try:
                func = Function(indicator_name)
                details = {
                    "name": indicator_name,
                    "inputs": func.input_names,
                    "parameters": func.parameters,
                    "outputs": func.output_names,
                }
                detailed_indicators[group].append(details)
            except Exception:
                continue
    return detailed_indicators


class GetIndicatorInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g., 'AAPL').")
    indicator_name: str = Field(
        description="The name of the TA-Lib indicator to calculate.",
        json_schema_extra={"enum": talib.get_functions()},
    )
    historical_period: str = Field(
        default="6mo",
        description="The historical data period to use (e.g., '1d', '5d', '1mo', '6mo', '1y', 'ytd', 'max').",
    )
    indicator_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="A dictionary of parameters for the indicator (e.g., {'timeperiod': 14}).",
    )


@mcp.tool()
async def get_indicator(
    ticker: str,
    indicator_name: str,
    historical_period: str,
    indicator_params: Optional[Dict[str, Any]],
) -> Dict:
    """Calculates a TA-Lib indicator using specific inputs and parameters."""
    params = indicator_params or {}
    logger.info(
        f"[INDICATOR] Tool 'get_indicator' called for {ticker} with '{indicator_name}' and params {params}"
    )
    try:
        logger.info(f"[INDICATOR] Fetching historical data for {ticker}...")
        history_df = yf.Ticker(ticker).history(
            period=historical_period, auto_adjust=False
        )
        if history_df.empty:
            logger.error(f"[INDICATOR] Could not fetch historical data for {ticker}.")
            return {"error": f"Could not fetch historical data for {ticker}."}
        logger.info(
            f"[INDICATOR] Historical data fetched for {ticker}. Calculating indicator..."
        )

        inputs = {
            "open": history_df["Open"].to_numpy(dtype=np.double),
            "high": history_df["High"].to_numpy(dtype=np.double),
            "low": history_df["Low"].to_numpy(dtype=np.double),
            "close": history_df["Close"].to_numpy(dtype=np.double),
            "volume": history_df["Volume"].to_numpy(dtype=np.double),
        }

        indicator_function = Function(indicator_name)
        results = indicator_function(inputs, **params)

        if isinstance(results, list):
            output = {
                f"{indicator_function.output_names[i]}": arr[~np.isnan(arr)][
                    -5:
                ].tolist()
                for i, arr in enumerate(results)
            }
        else:
            output = {
                f"{indicator_function.output_names[0]}": results[~np.isnan(results)][
                    -5:
                ].tolist()
            }
        logger.info(
            f"[INDICATOR] Successfully calculated {indicator_name} for {ticker}. Output: {output}"
        )
        return {
            "success": True,
            "ticker": ticker,
            "indicator": indicator_name,
            "note": "Results are from most recent to oldest (last 5 valid values).",
            "output": output,
        }
    except Exception as e:
        logger.error(
            f"[INDICATOR] TA-Lib calculation error for {ticker}: {e}", exc_info=True
        )
        return {
            "error": f"Failed to calculate '{indicator_name}'. Check function name and parameters. Error: {str(e)}"
        }


class GetTransactionHistoryInput(BaseModel):
    limit: int = Field(
        default=20,
        description="The maximum number of recent transactions to retrieve.",
        json_schema_extra={"anyOf": [{"type": "integer"}]},
    )


@mcp.tool()
async def get_transaction_history(limit: int, ctx: Context) -> List[Dict[str, Any]]:
    """Retrieves a log of the most recent transactions (buys and sells)."""
    logger.info(f"Tool 'get_transaction_history' called with limit={limit}")
    return await asyncio.to_thread(
        database.get_transactions, ctx.fastmcp.database_name, ctx.fastmcp.db_type, limit
    )


# --- Custom Routes & Main Entry Point ---
@mcp.custom_route("/", methods=["GET"])
async def root_health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "server": "PaperTraderMCP"})


@mcp.custom_route("/portfolio_summary", methods=["GET"])
async def portfolio_summary_route(request: Request, ctx: Context) -> JSONResponse:
    database_name = ctx.fastmcp.database_name
    summary = await get_portfolio_summary(database_name)
    return JSONResponse(summary)


def main(db_type, db_name, transport):

    mcp.database_name = db_name
    mcp.db_type = db_type

    try:
        database.initialize_database(db_name, db_type)
        logger.info("Starting PaperTrader MCP server...")
        
        if transport == "stdio":
            mcp.run(transport="stdio")
        elif transport == "http":
            mcp.run(
            transport="http",
            host="0.0.0.0",
            port=8000,
            path="/",
            log_level="info",
            )
        elif transport == "streamable-http":
            mcp.run(
            transport="streamable-http",
            host="0.0.0.0",
            port=8000,
            path="/",
            log_level="info",
            )
        else:
            logger.error(f"Invalid transport method: {transport}")
            raise ValueError(f"Invalid transport method: {transport}")
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper Trader MCP Server")
    parser.add_argument(
        "--database-name", required=True, help="Database connection name"
    )
    parser.add_argument(
        "--db-type", default="postgres", help="Database type (postgres or sqlite)"
    )
    parser.add_argument(
        "--transport",
        default="streamable-http",
        choices=["http", "stdio", "streamable-http"],
        help="Transport method for the MCP server (http, stdio, or streamable-http).",
    )
    args = parser.parse_args()

    main(args.db_type, args.database_name, args.transport)
