# Paper Trader MCP

A Model Context Protocol (MCP) server for paper trading with real-time market data and portfolio management. Built with FastMCP and SQLAlchemy, this service allows you to simulate stock trading without real money while tracking performance metrics.

## Features

### Current Features

- **Real-time Market Data**: Fetch current prices, market status, and time-to-market-close for any ticker
- **Portfolio Management**: Track cash balance and stock holdings with average cost basis
- **Trading Operations**: Buy and sell stocks with automatic order logging and validation
- **Financial Analysis**:
  - Fetch financial statements (income statement, balance sheet, cash flow)
  - Get key statistics (P/E ratio, dividend yield, market cap, etc.)
  - Retrieve analyst recommendations and company news
  - Access dividend and stock split history
- **Technical Indicators**: Calculate TA-Lib technical indicators (RSI, MACD, Bollinger Bands, etc.) on historical data
- **Transaction History**: Complete audit trail of all buy/sell transactions
- **Multi-Market Support**: Configured for markets in India, US, China, Japan, Europe, and more
- **Database Flexibility**: SQLite for development, PostgreSQL for production

### Supported Markets

- 🇮🇳 Indian Stock Market (NSE/BSE)
- 🇺🇸 US Stock Market (NASDAQ/NYSE)
- 🇨🇳 Shanghai Stock Exchange (SSE)
- 🇨🇳 Shenzhen Stock Exchange (SZSE)
- 🇪🇺 Euronext
- 🇯🇵 Japan Exchange Group (JPX)
- 🇬🇧 London Stock Exchange (LSE)
- 🇭🇰 Hong Kong Exchanges and Clearing (HKEX)

## Getting Started

### Prerequisites

- Python 3.12+
- Docker (optional)
- `uv` package manager

### Installation

#### Local Development

```bash
# Install dependencies
uv sync

# Initialize database
uv run server.py --database-name trader_db --db-type sqlite

# Or with environment variables
export DB_TYPE=sqlite
export DB_NAME=default_trader
export TRANSPORT=streamable-http
uv run server.py
```

#### Docker

```bash
# Build the image
docker build -t paper-trader .

# Run the container
docker run -p 8000:8000 paper-trader
```

### Configuration

The server can be configured via environment variables:

- `DB_TYPE`: Database type (`sqlite` or `postgres`, default: `sqlite`)
- `DB_NAME`: Database name (default: `default_trader`)
- `TRANSPORT`: MCP transport method (`http`, `stdio`, or `streamable-http`, default: `streamable-http`)
- `DATABASE_URL`: PostgreSQL connection string (if using postgres)

### Available Tools

#### Market Data
- `get_market_data(ticker)` - Get current price and market info
- `get_market_status(ticker, region)` - Get market state and time to close
- `get_financial_statements(ticker, statement_type)` - Fetch income statement, balance sheet, or cash flow
- `get_key_statistics(ticker)` - Get P/E ratio, dividend yield, market cap, etc.
- `get_analyst_recommendations(ticker)` - Retrieve analyst recommendations
- `get_company_news(ticker)` - Get recent company news
- `get_dividends_splits(ticker)` - Get dividend and split history

#### Trading
- `buy(ticker, quantity, reason)` - Execute a buy order
- `sell(ticker, quantity, reason)` - Execute a sell order
- `get_portfolio()` - Get current cash and holdings
- `get_transaction_history(limit)` - Get recent transactions

#### Technical Analysis
- `get_all_indicator_details()` - List all available TA-Lib indicators with parameters
- `get_indicator(ticker, indicator_name, historical_period, indicator_params)` - Calculate technical indicators

## Project Structure

```
paper-trader/
├── paper_trader_mcp.py    # Main MCP server implementation
├── database.py            # Database models and operations
├── server.py              # Server entry point
├── pyproject.toml         # Project configuration
├── Dockerfile             # Docker configuration
├── README.md              # This file
└── db/                    # SQLite databases (local development)
```

## Database

### Supported Databases

- **SQLite**: Default for local development. Databases stored in `db/` directory.
- **PostgreSQL**: For production deployments.

### Tables

- `portfolio`: Cash balance
- `holdings`: Current stock positions with average cost basis
- `transactions`: Complete transaction history
- `history`: Portfolio value snapshots over time

## Future Plans

### 1. Real-time Data Streaming from yfinance

**Goal**: Stream live market data instead of polling at request time.

- Implement WebSocket connections to fetch intraday price updates
- Build a data cache layer to track price movements
- Provide live price feeds to connected clients
- Enable real-time portfolio valuation as prices change

**Benefits**: 
- Reduced API calls to yfinance
- More responsive trading signals
- Better simulation of live trading conditions

### 2. Automated Entry and Exit Conditions

**Goal**: Enable automatic buy/sell execution based on user-defined conditions.

**Features to implement**:
- **Entry Conditions**: Buy when conditions are met (e.g., price crosses below MA, RSI < 30)
- **Exit Conditions**: Sell when conditions are met:
  - **Stop Loss**: Sell if price drops X% or to specific price
  - **Take Profit**: Sell if price rises X% or reaches target price
  - **Trailing Stop**: Dynamically adjust stop based on price peaks
  - **Technical Indicators**: Sell on signal (RSI > 70, MACD crossover, etc.)

**Implementation**:
- Store watched tickers with their entry/exit conditions
- Background service to continuously monitor market conditions
- Automatic order execution when conditions trigger
- Detailed logging of condition triggers and executions

**Use Cases**:
```
Example 1: Simple buy-and-hold with stop loss
- Buy AAPL at market
- Set stop loss at -5%
- Automatically sells if price drops 5%

Example 2: Technical indicator trading
- Buy when RSI < 30 (oversold)
- Sell when RSI > 70 (overbought)
- Track multiple tickers simultaneously
```

### 3. Additional Planned Features

- **Backtesting Engine**: Test strategies on historical data
- **Performance Analytics**: Calculate Sharpe ratio, max drawdown, win rate, etc.
- **Strategy Templates**: Pre-built common trading strategies
- **Risk Management**: Position sizing, portfolio allocation limits
- **Multi-timeframe Analysis**: Analyze indicators across different time periods
- **Alert System**: Notifications for price milestones, news events, etc.
- **Export Capabilities**: CSV/JSON export of trades and performance
- **Dashboard UI**: Web interface for portfolio management and monitoring

## Development

### Running Tests

```bash
# (Tests to be added)
uv run pytest
```

### Code Structure

- **MCP Tools**: Each trading function is exposed as an MCP tool for LLM integration
- **Async Operations**: Supports concurrent requests with asyncio
- **Transaction Logging**: Every trade is logged with reason and status
- **Error Handling**: Comprehensive error messages for failed operations

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT

## Support

For issues or questions, please open an issue on the project repository.
