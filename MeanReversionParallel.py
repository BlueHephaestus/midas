"""
Improved from original static GA model to try a rolling model, that will predict at a given timestep t
    by taking in a window of data up to t-1.
"""
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


# random.seed(42)


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
# periods = [90]
stock_n = 1000
# stock_n = 5000

INITIAL_SHARES = 0
INITIAL_WALLET = 100

# stock_n, n = stock.shape
train_n = int(n * .66)

import itertools

periods = 30,60,90,120,150,180,210,240
# periods = 60,120
periods = 60,
cooldown_types = ["buy", "sell", "both"]
# cooldown_types = ["buy"]
cooldowns = [1,2,4,8]
# cooldowns = [0,1,2,3,4]
# cooldowns = [4,5,6]
# cooldowns = [4,]
# std_multipliers = [0.5,1,1.5,2,2.5,3,3.5,4]
std_multipliers = [0,1,2,3,4]
# std_multipliers = [0,0.25,0.5,0.75,1.0]
# std_multipliers = [0,0.25,0.5,0.75]
# std_multipliers = [0,]
early_cashouts = [False, True]
# early_cashouts = [False]

# Precompute all the averages and stds
avgs = np.zeros((len(periods), stock_n, n))
stds = np.zeros((len(periods), stock_n, n))

print("Precomputing Period Averages and Stds")
for period_i, period in enumerate(tqdm(periods)):
    for i in range(period, n):
        avgs[period_i, :, i] = np.mean(data[:stock_n, i-period:i], axis=1)
        stds[period_i, :, i] = np.std(data[:stock_n, i-period:i], axis=1)
print("Precomputing Complete")

#for period in periods:
# period = 90
# cooldown_type = "both"
# cooldown = 1
# std_multiplier = 2
# early_cashout = False

grid_search = list(itertools.product(periods, cooldown_types, cooldowns, std_multipliers, early_cashouts))
#grid_scores = np.zeros((len(grid_search), 3), dtype=np.object)

print(f"Grid Searching through {len(grid_search)} combinations of hyperparameters")

def grid_search_function(args):
    grid_i, params = args
    period, cooldown_type, cooldown, std_multiplier, early_cashout = params

    grid_scores = np.zeros((3,), dtype=np.object)
    grid_scores[0] = f"Period: {period}, Cooldown Type: {cooldown_type}, Cooldown: {cooldown}, Std Multiplier: {std_multiplier}, Early Cashout: {early_cashout}"
    train = range(train_n)
    test = range(train_n,n)
    period_i = periods.index(period)
    for dataset_i, dataset in enumerate([train,test]):
        scores = np.zeros((stock_n))
        for stock_i in tqdm(range(stock_n), disable=True):
            shares = INITIAL_SHARES
            wallet = INITIAL_WALLET
            COOLDOWN = cooldown
            cooldown_count = COOLDOWN
            trade_fee = 0.01

            for i in dataset:
                if i >= period:
                    price = data[stock_i, i]
                    mean = avgs[period_i, stock_i,i]
                    std = stds[period_i, stock_i,i]
                    # mean = np.mean(data[stock_i,i-period:i])
                    # std = np.std(data[stock_i,i-period:i])

                    hi, lo = mean+std*std_multiplier, mean-std*std_multiplier
                    #if (cooldown_type == "both" or cooldown_type == "sell") and cooldown_count == COOLDOWN: # off cooldown, can buy/sell
                    #elif (cooldown_type == "both" or cooldown_type == "buy") and cooldown_count == COOLDOWN: # off cooldown, can buy/sell
                    #if cooldown_count == COOLDOWN: # off cooldown, can buy/sell

                    # We can sell if it's off cooldown and we need a cooldown, or if we don't need a cooldown
                    if (cooldown_count == COOLDOWN and cooldown_type in ["both", "sell"]) or cooldown_type == "buy":  # off cooldown, can buy/sell
                        # Now have mean, std
                        if price > hi and shares != 0:
                            # sell
                            shares, wallet, net_worth = act("SELL", price, shares, wallet, trade_fee)

                            cooldown_count = 0 # reset cooldown

                            if early_cashout and wallet > INITIAL_WALLET:
                                continue # go to next stock, cash out


                    # We can buy if it's off cooldown and we need a cooldown, or if we don't need a cooldown
                    if (cooldown_count == COOLDOWN and cooldown_type in ["both","buy"]) or cooldown_type == "sell": # off cooldown, can buy/sell
                        if price != 0 and price < lo and shares == 0:
                            # buy
                            shares, wallet, net_worth = act("BUY", price, shares, wallet, trade_fee)
                            cooldown_count = 0 # reset cooldown

                    if cooldown_count != COOLDOWN:
                        cooldown_count += 1 # otherwise we are on cooldown, increment

                    shares, wallet, net_worth = act("", price, shares, wallet, trade_fee)

            # get net worth now that sim is complete
            #if price != data[stock_i,-1]: import sys; print("ASDFASDFAASDF");sys.exit()
            #scores[stock_i] = get_net_worth(shares, wallet, data[stock_i,-1], trade_fee)
            if net_worth > 10 * INITIAL_WALLET:
                net_worth = INITIAL_WALLET
            scores[stock_i] = net_worth - INITIAL_WALLET
            #scores[stock_i] = get_net_worth(shares, wallet, price, trade_fee)

        grid_scores[dataset_i+1] = np.sum(scores)

    return grid_i, grid_scores


# Pre-compute the parameter grid to divide among processes
param_grid = [(i, params) for i, params in enumerate(grid_search)]

# Create a pool of worker processes
with Pool() as pool:
    # Use the pool's map method to assign the grid search tasks to the worker processes
    #results = pool.map(grid_search_function, param_grid)
    results = list(tqdm(pool.imap(grid_search_function, param_grid), total=len(param_grid)))


# Collect the results from the worker processes and populate grid_scores
grid_scores = [result[1] for result in sorted(results, key=lambda x: x[0])]


np.save("grid_scores.npy", grid_scores)
for score in grid_scores:
    print(score)
# Idea for a model
# if below 2stds, buy
# if above and have stock, sell
#
# NOTHING ELSE AGHH
#
# """
# print(data.shape)
# for i in range(10):
#     plt.plot(np.arange(data.shape[1]), data[i,:])
#     for p_i,p in enumerate(periods):
#         plt.plot(np.arange(data.shape[1]), avgs[i,:,p_i], label=str(p))
#         plt.plot(np.arange(data.shape[1]), avgs[i, :, p_i]-stds[i,:,p_i], label=str(p))
#         plt.plot(np.arange(data.shape[1]), avgs[i, :, p_i]+stds[i,:,p_i], label=str(p))
#     plt.legend()
#     plt.show()
#
