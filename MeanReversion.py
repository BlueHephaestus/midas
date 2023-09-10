"""
Improved from original static GA model to try a rolling model, that will predict at a given timestep t
    by taking in a window of data up to t-1.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#random.seed(42)

class StaticStockEnvironment:
    def __init__(self, stock, initial_shares=0, initial_wallet=100, trade_fee=0.00):
        """
        A reinforcement learning environment, made from weekly stock data for ONE stock.
            to use all stock data, run this concurrently with others.

        More details in individual class functions.

        :param stock: Data for this stock. Will be in shape (# days, # timesteps / day, # features),
            e.g. shape of (5, 78, 5).
            This will by default only use the first 3 days as training data, and reserve the last two for
            validation and test data, respectively.
        :param initial_net_worth: Starting wallet for this environment for the agent to use.
            Used to determine the reward at each timestep, since this is one and the same.
        :param trade_fee: % taken from every trade, i.e. commission
        """
        # DATA
        stock_n, n = stock.shape
        train_n = int(n*.66)
        self.train = stock[:train_n]
        self.test = stock[train_n:]

        # CONSTANTS
        self.INITIAL_SHARES = initial_shares
        self.INITIAL_WALLET = initial_wallet
        self.TRADE_FEE = trade_fee

        # SIMULATION VARIABLES
        # Data currently being simulated, this is what will be referenced on each call to start() and act()
        # This can be changed when running validation and testing.
        self.env = self.train
        self.shares = self.INITIAL_SHARES # Used for computing net worth at each timestep, and indicating if we are invested.
        self.wallet = self.INITIAL_WALLET # Used for computing net worth
        self.prices = self.get_prices() # Get all prices beforehand

    def get_prices(self):
        return self.env

    def get_net_worth(self, price):
        """
        Using current simulation variables, return the current net worth of the bot.
        This is used as the reward at each timestep.

        Net worth = (shares * price/share) * (1 - self.TRADE_FEE) + self.wallet

        :param price: Price at timestep to compute net worth
        :return: net worth
        """
        return (self.shares * price) * (1.0 - self.TRADE_FEE) + self.wallet

    def update_env(self, data):
        """
        Reset this environment with new data, and recompute relevant data.
        """
        self.env = data
        self.shares = self.INITIAL_SHARES
        self.wallet = self.INITIAL_WALLET
        self.prices = self.get_prices()

    def reset(self):
        """
        Reset agent-specific attributes used to test this agent in this environment.

        DO NOT CHANGE THE PRICES
        :return:
        """
        self.shares = self.INITIAL_SHARES
        self.wallet = self.INITIAL_WALLET

    def act_single(self, action, i):
        """
        Same as below, but only on one timestep.
        :param action:
        :return:
        """
        price = self.prices[i]
        if action < -1:
            # SELL
            if self.shares > 0:
                # Sell shares at current price and put it in the wallet.
                self.wallet = (self.shares * price) * (1.0 - self.TRADE_FEE)
                self.shares = 0

        elif action > 1:
            # BUY
            if self.shares == 0:
                # Buy shares at current price, emptying our wallet.
                self.shares = (self.wallet * (1.0 - self.TRADE_FEE)) / price
                self.wallet = 0
        else:
            # DO NOTHING
            pass

        return self.get_net_worth(price=price)

    def act(self, actions):
        """
        Iterate through all actions and get final reward after all data.

        For each state value at timestep t:
            price is the actual buy price at that timestep
            action is the actual deicsion made by the agent at that timestep using that state

            by applying the action with the given price, we get our resulting reward.
            recall that the next state value is not determined by our actions in the stock market,
                since our own investments causing change would only occur with very very large investments.

            s_t+1, r_t+1 = environment(a_t*) # state does not depend on action, reward does.
            a_t = agent(s_t, r_t*) # agent does not use reward at the moment
            By combining them together we get the resulting reward for the next timestep

        :param actions: Vector of continuous values from an RL agent.
            action < -1 = sell (can only do this if shares > 0)
            action > 1 = buy (can only do this if shares == 0)
            -1 < action < 1 = do nothing

            For now it is an all-or-nothing, it can not invest portions of its net worth.

        :return:
            Iterates through all states, and computes final net worth
        """
        # Get all actions
        for action,price in zip(actions, self.prices):
            if action < -1:
                # SELL
                if self.shares > 0:
                    # Sell shares at current price and put it in the wallet.
                    self.wallet = (self.shares * price) * (1.0 - self.TRADE_FEE)
                    self.shares = 0

            elif action > 1:
                # BUY
                if self.shares == 0:
                    # Buy shares at current price, emptying our wallet.
                    self.shares = (self.wallet * (1.0 - self.TRADE_FEE)) / price
                    self.wallet = 0
            else:
                # DO NOTHING
                pass

        # Now everything is finalized, reset and return final reward after the final action
        reward = self.get_net_worth(price=self.prices[-1])
        return reward


# # Shape = (stocks, days, timesteps, values)
# # Values = high_price, open_price, close_price, low_price, volume
# data = np.load(f"jan_15_2022/week.npy").astype(np.float32)
#
# # Convert to "typical price", high, low, close mean
# data = np.mean(data[:,:,:,(0,2,3)],axis=-1)
#
# # Combine the 5 days to get a week
# data = np.reshape(data,(data.shape[0],-1))
# print(data.shape)
# plt.plot(np.arange(data.shape[1]), data[0,:])
# plt.show()
# plt.plot(np.arange(data.shape[1]), data[2,:])
# plt.show()
# plt.plot(np.arange(data.shape[1]), data[8,:])
# plt.show()
def get_net_worth(price, shares, wallet, trade_fee):
    """
    Using current simulation variables, return the current net worth of the bot.
    This is used as the reward at each timestep.

    Net worth = (shares * price/share) * (1 - self.TRADE_FEE) + self.wallet

    :param price: Price at timestep to compute net worth
    :return: net worth
    """
    return (shares * price) * (1.0 - trade_fee) + wallet

# def act(action, price, shares, wallet, trade_fee):
# used proportional trading fee
#     if action == "SELL":
#         # SELL
#         if shares > 0:
#             # Sell shares at current price and put it in the wallet.
#             wallet = (shares * price) * (1.0 - trade_fee)
#             shares = 0
#
#     elif action == "BUY":
#         # BUY
#         if shares == 0:
#             # Buy shares at current price, emptying our wallet.
#             shares = (wallet * (1.0 - trade_fee)) / price
#             wallet = 0
#     else:
#         # DO NOTHING
#         pass
#
#     # shares, wallet, net worth
#     return shares, wallet, (shares * price) * (1.0 - trade_fee) + wallet

def act(action, price, shares, wallet, trade_fee):
    if action == "SELL":
        # SELL
        if shares > 0:
            # Sell shares at current price and put it in the wallet.
            wallet = (shares * price) - trade_fee
            shares = 0

    elif action == "BUY":
        # BUY
        if shares == 0:
            # Buy shares at current price, emptying our wallet.
            shares = (wallet - trade_fee) / price
            wallet = 0
    else:
        # DO NOTHING
        pass

    # shares, wallet, net worth
    return shares, wallet, (shares * price) - trade_fee + wallet

    # shares, wallet
    # return shares, wallet

def get_net_worth(price, shares, wallet, trade_fee):
    return (shares * price) - trade_fee + wallet


# Shape = (stocks, days, values)
# Values = high_price, open_price, close_price, low_price, volume
data = np.load(f"jan_15_2022/5year.npy").astype(np.float32)

# avgs = np.load("avgs.npy")
# stds = np.load("stds.npy")
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
#stock_n = 1000
stock_n = 10

INITIAL_SHARES = 0
INITIAL_WALLET = 100

# stock_n, n = stock.shape
train_n = int(n * .66)

import itertools

periods = 30,60,90,120,150,180,210,240
cooldown_types = ["buy", "sell", "both"]
cooldowns = [0,1,2,4,8,16,32]
std_multipliers = [0.5,1,1.5,2,2.5,3,3.5,4]
early_cashouts = [False, True]

# Precompute all the averages and stds
avgs = np.zeros((len(periods), stock_n, n))
stds = np.zeros((len(periods), stock_n, n))

print("Precomputing Period Averages and Stds")
for period_i, period in enumerate(tqdm(periods)):
    for stock_i in range(stock_n):
        for i in range(period, n):
            avgs[period_i, stock_i, i] = np.mean(data[stock_i, i - period:i])
            stds[period_i, stock_i, i] = np.std(data[stock_i, i - period:i])

print("Precomputing Complete")

#for period in periods:
# period = 90
# cooldown_type = "both"
# cooldown = 1
# std_multiplier = 2
# early_cashout = False

grid_search = list(itertools.product(periods, cooldown_types, cooldowns, std_multipliers, early_cashouts))
grid_scores = np.zeros((len(grid_search), 3), dtype=np.object)

print(f"Grid Searching through {len(grid_search)} combinations of hyperparameters")

for grid_i, (period, cooldown_type, cooldown, std_multiplier, early_cashout) in enumerate(tqdm(grid_search)):
    grid_scores[grid_i][0] = f"Period: {period}, Cooldown Type: {cooldown_type}, Cooldown: {cooldown}, Std Multiplier: {std_multiplier}, Early Cashout: {early_cashout}"
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
            net_worth = wallet
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
            #scores[stock_i] = get_net_worth(shares, wallet, data[stock_i,-1], trade_fee)
            scores[stock_i] = net_worth

        grid_scores[grid_i][dataset_i+1] = np.mean(scores)
    #print("\t", np.mean(scores))
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
