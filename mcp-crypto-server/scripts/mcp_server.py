"""
This script demonstrates how to create a simple MCP server that fetches
the current price of a cryptocurrency using the CoinGecko API.
It uses the FastMCP library to create the server and handle requests.
"""

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# Create our MCP server with a descriptive name
mcp = FastMCP("crypto_price_tracker")


# Now let's define our first tool - getting the current price of a cryptocurrency
@mcp.tool()
async def get_crypto_price(crypto_id: str, currency: str = "usd") -> str:
    """
    Get the current price of a cryptocurrency in a specified currency.

    Parameters:
    - crypto_id: The ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
    - currency: The currency to display the price in (default: 'usd')

    Returns:
    - Current price information as a formatted string
    """
    # Construct the API URL
    url = f"{COINGECKO_BASE_URL}/simple/price"

    # Set up the query parameters
    params = {"ids": crypto_id, "vs_currencies": currency}

    try:
        # Make the API call
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the response
            data = response.json()

            # Check if we got data for the requested crypto
            if crypto_id not in data:
                return f"Cryptocurrency '{crypto_id}' not found. Please check the ID and try again."

            # Format and return the price information
            price = data[crypto_id][currency]
            return f"The current price of {crypto_id} is {price} {currency.upper()}"

    except httpx.HTTPStatusError as e:
        return f"API Error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error fetching price data: {str(e)}"


# You can add more tools here, following the same pattern as above


@mcp.tool()
async def get_crypto_market_info(crypto_ids: str, currency: str = "usd") -> str:
    """
    Get market information for one or more cryptocurrencies.

    parameters:
    - crypto_ids: Comma-separated list of cryptocurrency IDs (e.g., 'bitcoin, ethereum')
    - currency: The Currency to display values in (default: 'usd')

    Returns:
    - Market information including price, market cap, volume, and price changes
    """

    url = f"{COINGECKO_BASE_URL}/coins/markets"

    params = {
        "vs_currency": currency,
        "ids": crypto_ids,
        "order": "market_cap_desc",
        "page": 1,
        "sparkline": "false",
    }

    try:
        # Make the API Call
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

            # Parse the response
            data = response.json()

            # Check if we got any data
            if not data:
                return f"No data found for cryptocurrency: '{crypto_ids}'. Please check the IDs and try again."

            # Format the response
            result = ""
            for crypto in data:
                name = crypto.get("name", "Unknown")
                symbol = crypto.get("symbol", "???").upper()
                price = crypto.get("current_price", "Unknown")
                market_cap = crypto.get("market_cap", "Unknown")
                volume = crypto.get("total_volume", "Unknown")
                price_change = crypto.get("price_change_percentage_24h", "Unknown")

                result += f"{name} ({symbol}):\n"
                result += f"Current price: {price} {currency.upper()}\n"
                result += f"Market cap: {market_cap} {currency.upper()}\n"
                result += f"24h trading volume: {volume} {currency.upper()}\n"
                result += f"24h price change: {price_change}%\n\n"

            return result
    except Exception as e:
        return f"Error fetching market data: {str(e)}"


# Run the MCP server
# This will start the server and listen for incoming requests
if __name__ == "__main__":
    mcp.run(transport="stdio")
