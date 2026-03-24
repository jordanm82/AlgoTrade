# Coinbase tradeable pairs
SPOT_PAIRS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
]

PERP_PAIRS = [
    "BTC-PERP-INTX",
    "ETH-PERP-INTX",
    "SOL-PERP-INTX",
]

# Binance symbols for broader data (mapped to Coinbase equivalents)
BINANCE_SPOT = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "SOL/USDT": "SOL-USD",
}

BINANCE_FUTURES = {
    "BTC/USDT:USDT": "BTC-PERP-INTX",
    "ETH/USDT:USDT": "ETH-PERP-INTX",
    "SOL/USDT:USDT": "SOL-PERP-INTX",
}
