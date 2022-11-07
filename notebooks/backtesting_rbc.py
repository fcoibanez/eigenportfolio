"""
This file contains two python classes that can be used in a backtesting
system:
    1. Portfolio:
        Handles the day-to-day portfolio operations, including rebalancing,
        cash management, and nav/positions mark-to-market, which can be used
        to calculate daily p&l. The API receives target weights and executed
        prices when rebalacing, and closing prices when marking the book.
    2. Data:
        Connects the code to WRDS/CRSP and gets historical S&P 500 constituents
        and their historical prices and traded volume. These are then used to
        run the backtesting.

Notes
-----
About the data vendor
    - WRDS and CRSP are commonly used in academic empirical equity research
    - Even though my code does not support option analysis, the databases also
        contain option data that can be used to expand this work with
    - The class can also be adapted to connect to the python Bloomberg API in
        case we want to take this system to live production

About the use cases
    - At the end of the file, the equal-dollar stock strategy is implemented
    - I am interpreting "top 30 US Stocks" as the most liquid ones. For this
        reason, on each day, I am using dollar volume and selecting the 30
        stocks with the highest value.
    - I am omitting the rolling option strategy due to time constraints

Potential issues of this program
    - It does not handle delisting. In other words, if we want to control for
        survivorship bias, we will need to include stocks that are not tradable
        anymore. When these stocks are taken out of the market, due to either
        exchange delisting, or other reason, the portfolio could be left
        underinvested until next rebalance (due to having a portion of the
        portfolio invested on a security that does not produce returns). This
        is a bigger deal when backtesting with less frequent rebalances.

Converting this program into a live-trading system
    - This will require changing the data class from WRDS/CRSP to another,
        higher frequency, data provider such as Bloomberg or Reuters
    - Additionally, the system only supports fractional shares. A proper
        productionalized version will require the program to be able to
        handle integer shares, which might require integer programming.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import wrds

class Portfolio:
    def __init__(self, starting_nav, inception_date):
        self.securities = []
        self._dates = pd.DatetimeIndex([inception_date])
        self.nav = pd.Series(starting_nav, index=self._dates, dtype=float)
        self.cash = pd.Series(starting_nav, index=self._dates, dtype=float)
        self.shares = pd.DataFrame(index=self._dates, dtype=float)
        self.notional = pd.DataFrame(index=self._dates, dtype=float)
        self.model_weights = pd.DataFrame(index=self._dates, dtype=float)
        self.target_weights = pd.DataFrame(index=self._dates, dtype=float)
        self.prices = pd.DataFrame(index=self._dates, dtype=float)
        self._yesterday = inception_date
        self._today = inception_date
        self._active_positions = []
        self._live = False

    def rebalance(self, target_weights, trade_prices, tcost=None):
        assert isinstance(target_weights, pd.Series)

        # Preamble
        new_securities = [x for x in target_weights.index if x not in self.securities]
        if len(new_securities) > 0:
            new_index = self.securities + new_securities
            self.expand(expanded_index=new_index, axis=1)
            self.securities += new_securities

        # Calculate number of stocks to trade
        w_star = target_weights.reindex(self.securities, fill_value=0)
        self.target_weights.loc[self._today] = w_star
        target_shares = self.nav[self._yesterday] * w_star / trade_prices.reindex(self.securities)
        current_shares = self.shares.loc[self._yesterday].fillna(0)
        shares_to_trade = target_shares - current_shares
        notional_to_trade = shares_to_trade * trade_prices.reindex(self.securities)
        names_to_trade = list(shares_to_trade.replace(0, np.nan).dropna().index)

        assert all([x in trade_prices.dropna().index for x in names_to_trade]), \
            'Some securities are being trades, but prices are not being provided.'

        # Trade execution
        self.shares.loc[self._today] = self.shares.loc[self._yesterday].fillna(0) + shares_to_trade
        self.cash[self._today] = self.cash[self._yesterday] - notional_to_trade.sum()

        # Transaction costs impact
        if tcost is None:
            tcost = pd.Series(0, index=self.securities)
        tcost_paid = notional_to_trade.mul(tcost).sum()
        self.nav[self._today] -= tcost_paid

        self._live = True
        self._active_positions = list(self.notional.loc[self._today].dropna().index)

    def mark2market(self, closing_prices):
        assert isinstance(closing_prices, pd.Series), 'trade_prices has to be a pandas Series object'

        # Mark-to-market
        self.prices.loc[self._today] = closing_prices.reindex(self.securities)  # Today's closing prices
        if self._live:
            assert all([x in closing_prices.index for x in self._active_positions]), \
                'Prices for every active position are required.'
            self.notional.loc[self._today] = self.shares.loc[self._today].fillna(0) * closing_prices
            self.nav[self._today] = self.notional.loc[self._today].sum() + self.cash.loc[self._today].sum()
            self.model_weights.loc[self._today] = self.notional.loc[self._today] / self.nav[self._today]

        # End of day cleaning
        self._yesterday = self._today

    def new_day(self, date):
        if date > self._dates[-1]:
            new_index = self._dates.union([date])
            self.expand(expanded_index=new_index, axis=0)
            self.shares.loc[date] = self.shares.loc[self._yesterday]
            self.notional.loc[date] = self.notional.loc[self._yesterday]
            self.cash[date] = self.cash[self._yesterday]
            self.nav[date] = self.nav[self._yesterday]
            self._today = date
            self._dates = new_index

    def expand(self, expanded_index, axis):
        self.prices = self.prices.reindex(expanded_index, axis=axis)
        self.model_weights = self.model_weights.reindex(expanded_index, axis=axis)
        self.target_weights = self.target_weights.reindex(expanded_index, axis=axis)
        self.notional = self.notional.reindex(expanded_index, axis=axis)
        self.shares = self.shares.reindex(expanded_index, axis=axis)
        self.prices = self.prices.reindex(expanded_index, axis=axis)


class GetData:
    def __init__(self, wrds_username, from_dt, to_dt):
        self.start = from_dt
        self.end = to_dt
        self.conn = wrds.Connection(wrds_username=wrds_username)
        self.meta = None
        self.prices = None
        self.returns = None
        self.volume = None

    def get_constituents(self):
        query = \
            f'''
            SELECT sp.permno, meta.permco, meta.comnam, meta.namedt, meta.cusip, meta.ticker, sp.start, sp.ending
            FROM crsp_a_indexes.dsp500list AS sp
            LEFT JOIN crsp.stocknames AS meta
            ON sp.permno = meta.permno
            WHERE sp.start >= '{self.start.strftime('%Y-%m-%d')}'
            AND sp.ending < '{self.end.strftime('%Y-%m-%d')}'
            '''
        meta_sp = self.conn.raw_sql(query)
        self.meta = meta_sp

    def get_data(self):
        id_str = ', '.join([f"'{str(int(x))}'" for x in self.meta['permno'].unique()])

        query = \
            f'''
            SELECT date, permno, vol, prc
            FROM crsp_a_stock.dsf
            WHERE permno IN ({id_str})
            AND date >= '{self.start.strftime('%Y-%m-%d')}'
            AND date < '{self.end.strftime('%Y-%m-%d')}'
            '''
        long = self.conn.raw_sql(query)
        self.prices = pd.pivot_table(long, index='date', columns='permno', values='prc')
        self.prices.index = pd.DatetimeIndex(data.prices.index)
        self.volume = pd.pivot_table(long, index='date', columns='permno', values='vol')
        self.volume.index = pd.DatetimeIndex(data.volume.index)


if __name__ == '__main__':
    start_dt = datetime(1999, 12, 31)
    end_dt = datetime(2021, 12, 31)

    # Pulling data for S&P500 constituents from WRDS
    data = GetData(
        wrds_username='fcoibanez',
        from_dt=start_dt - pd.DateOffset(years=1),
        to_dt=end_dt + pd.DateOffset(years=1)
    )
    data.get_constituents()
    data.get_data()

    # Holding equal-dollar ammount on the top 30 US stocks and rebalancing daily
    idx = pd.date_range(start=start_dt, end=end_dt, freq='B')
    port = Portfolio(starting_nav=100, inception_date=start_dt - pd.DateOffset(days=1))
    top_n = 30
    for dt in idx:
        print(dt)
        port.new_day(dt)
        ticker_tape = data.prices.resample('B').ffill().loc[dt].dropna()  # Today's closing prices

        # Top 30 US stocks (highest market cap)
        volume = data.volume.resample('B').ffill().loc[dt].dropna()
        dollar_volume = volume * ticker_tape
        top_30_names = dollar_volume.sort_values(ascending=False).iloc[:top_n].index
        tgt_wts = pd.Series(1 / top_n, index=top_30_names).reindex(ticker_tape.index).fillna(0)

        port.rebalance(target_weights=tgt_wts, trade_prices=ticker_tape)
        port.mark2market(closing_prices=ticker_tape)

    print(port.nav)
