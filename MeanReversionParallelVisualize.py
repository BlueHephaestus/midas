"""
Improved from original static GA model to try a rolling model, that will predict at a given timestep t
    by taking in a window of data up to t-1.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
#random.seed(42)

from multiprocessing import Pool


def act(action, price, shares, wallet, trade_fee):
    if action == "SELL":
        # SELL
        if shares > 0:
            # Sell shares at current price and put it in the wallet.
            wallet = (shares * price) - (trade_fee*wallet)
            shares = 0

    elif action == "BUY":
        # BUY
        if shares == 0:
            # Buy shares at current price, emptying our wallet.
            shares = (wallet - (wallet*trade_fee)) / price
            wallet = 0
    else:
        # DO NOTHING
        pass

    # shares, wallet
    return shares, wallet, (shares * price) - (trade_fee*wallet) + wallet

# def get_net_worth(price, shares, wallet, trade_fee):
#     return (shares * price) - trade_fee + wallet


# Shape = (stocks, days, values)
# Values = high_price, open_price, close_price, low_price, volume
data = np.load(f"jan_15_2022/5year.npy").astype(np.float32)


# there's a random all 0 value at the end, we remove this
# Find the first all-zero row in the second axis
mask = np.all(data == 0, axis=2)

# Find the index of the first all-zero row for each first-axis element
idx = np.argmax(mask)

# Remove these zeroes
data = data[:,:idx]

# Convert to "typical price", high, low, close mean
data = np.mean(data[:,:,(0,2,3)],axis=-1)

stock_n, n = data.shape

# Get moving day averages
periods = [60]
# stock_idxs = [1338]
stock_idxs = list(range(1000,1300))
#stock_idxs = [1326]
stock_n = len(stock_idxs)

INITIAL_SHARES = 0
INITIAL_WALLET = 100

# stock_n, n = stock.shape
train_n = int(n * .66)

#for period in periods:
# period = 90
# cooldown_type = "both"
# cooldown = 1
# std_multiplier = 2
# early_cashout = False

period = 60
cooldown_type = "buy"
cooldown = 4
std_multiplier = 0
early_cashout = False

train = range(train_n)
test = range(train_n,n)

buys = {} # i: price
sells = {} # i: price
net_worths = []


for dataset_i, dataset in enumerate([train,test]):
    scores = np.zeros((stock_n))
    for stock_i in tqdm(stock_idxs, disable=True):
        shares = INITIAL_SHARES
        wallet = INITIAL_WALLET
        COOLDOWN = cooldown
        cooldown_count = COOLDOWN
        trade_fee = 0.01


        for i in dataset:
            if i >= period:
                price = data[stock_i, i]
                mean = np.mean(data[stock_i,i-period:i])
                std = np.std(data[stock_i,i-period:i])

                hi, lo = mean+std*std_multiplier, mean-std*std_multiplier

                # We can sell if it's off cooldown and we need a cooldown, or if we don't need a cooldown
                if (cooldown_count == COOLDOWN and cooldown_type in ["both", "sell"]) or cooldown_type == "buy":  # off cooldown, can buy/sell
                    # Now have mean, std
                    if price > hi and shares != 0:
                        # sell
                        shares, wallet, net_worth = act("SELL", price, shares, wallet, trade_fee)

                        sells[i] = price

                        cooldown_count = 0 # reset cooldown

                        if early_cashout and wallet > INITIAL_WALLET:
                            continue # go to next stock, cash out


                # We can buy if it's off cooldown and we need a cooldown, or if we don't need a cooldown
                if (cooldown_count == COOLDOWN and cooldown_type in ["both","buy"]) or cooldown_type == "sell": # off cooldown, can buy/sell
                    if price != 0 and price < lo and shares == 0:
                        # buy
                        shares, wallet, net_worth = act("BUY", price, shares, wallet, trade_fee)
                        cooldown_count = 0 # reset cooldown
                        buys[i] = price

                if cooldown_count != COOLDOWN:
                    cooldown_count += 1 # otherwise we are on cooldown, increment

                shares, wallet, net_worth = act("", price, shares, wallet, trade_fee)
                #print(i, net_worth)
        if net_worth > 10 * INITIAL_WALLET:
            net_worth = INITIAL_WALLET
        net_worths.append(net_worth-INITIAL_WALLET)
        print(net_worth-INITIAL_WALLET)
    # # print()
    # times = np.arange(len(dataset))
    # if dataset_i == 1: times = np.arange(len(dataset))+train_n
    # prices = [data[stock_i, i] for i in dataset]
    # buy_times = list(buys.keys())
    # buy_prices = list(buys.values())
    # sell_times = list(sells.keys())
    # sell_prices = list(sells.values())
    #
    # # Plot using plotly
    # trace1 = go.Scatter(x=times, y=prices, mode='lines', name='Prices')
    # trace2 = go.Scatter(x=buy_times, y=buy_prices, mode='markers', name='Buy Points',
    #                     marker=dict(color='green', size=10, symbol='circle-open'), text=buy_prices)
    # trace3 = go.Scatter(x=sell_times, y=sell_prices, mode='markers', name='Sell Points',
    #                     marker=dict(color='red', size=10, symbol='x'), text=sell_prices)
    # #trace4 = go.Scatter(x=times, y=net_worths, mode='lines', name='Net Worth')
    #
    # layout = go.Layout(
    #     title='Stock Prices and Trade Points',
    #     xaxis=dict(title='Time Step'),
    #     yaxis=dict(title='Price'),
    #     hovermode='closest'
    # )
    #
    # fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    ## fig.show()
    #print(net_worth)
    print(np.sum(net_worths), INITIAL_WALLET*len(net_worths))
    # net_worths = []
    # continue
    # break

