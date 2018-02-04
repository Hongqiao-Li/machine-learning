import talib
import pandas as pd


def generateIndicator(inputs, price_name):
    indicator_price = inputs[price_name].as_matrix()
    low = inputs['Low'].as_matrix()
    high = inputs['High'].as_matrix()
    # close = inputs['Close'].as_matrix()
    close = indicator_price
    volume = inputs['Volume'].as_matrix()

    # MA moving average
    ma = talib.MA(close, timeperiod=30, matype=0)
    inputs['MA'] = pd.Series(ma, index=inputs.index)

    # MACD - Moving Average Convergence/Divergence
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    inputs['MACD'] = pd.Series(macd, index=inputs.index)

    # ADX - Average Directional Movement Index
    adx = talib.ADX(high, low, close, timeperiod=14)
    inputs['ADX'] = pd.Series(adx, index=inputs.index)

    # CCI - Commodity Channel Index
    cci = talib.CCI(high, low, close, timeperiod=14)
    inputs['CCI'] = pd.Series(cci, index=inputs.index)
    # RSI - Relative Strength Index
    rsi = talib.RSI(close, timeperiod=14)
    inputs['CCI'] = pd.Series(rsi, index=inputs.index)

    # WILLR - Williams' %R
    willr = talib.WILLR(high, low, close, timeperiod=14)
    inputs['WILLR'] = pd.Series(willr, index=inputs.index)

    return inputs