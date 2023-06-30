import yfinance as yf
import streamlit as st
import pandas as pd

st.write("""
# Simple Stock Price App

Shown are the stock closing price and volume of Google!

""")

tickerSymbol = 'GOOGL'

tickerData = yf.Ticker(tickerSymbol)

tickerDf = tickerData.history(period='1d', start='2013-06-30', end='2023-06-30')

st.line_chart(tickerDf.Close)



st.write("""

Shown are the stock closing price and volume of Microsoft!

""")

tickerSymbol = 'MSFT'

tickerData = yf.Ticker(tickerSymbol)

tickerDf = tickerData.history(period='1d', start='2013-06-30', end='2023-06-30')

st.line_chart(tickerDf.Close)


st.write("""

Shown are the stock closing price and volume of Amazon!

""")

tickerSymbol = 'AMZN'

tickerData = yf.Ticker(tickerSymbol)

tickerDf = tickerData.history(period='1d', start='2013-06-30', end='2023-06-30')

st.line_chart(tickerDf.Close)