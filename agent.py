from collections import defaultdict
import datetime
import time
from dateutil.relativedelta import relativedelta
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import random
import os
import pickle
import yfinance as yf
import talib
import dill as pickle
import csv
 
STATES_DIM = 100 # number of discrete bins for each feature
ALPHA = 0.5 # learning rate
GAMMA = 0.9 # discount factor
EPSILON = 1.00 # exploration rate
EPSILON_DECAY = 0.9995 # decay rate

def get_stock_data(stock_ticker, years_ago):
    start = datetime.datetime.now() - relativedelta(years=years_ago)
    end = datetime.datetime.now()
    data = yf.download(stock_ticker, start, end)

    # Keep only relevant columns
    data = data[['Adj Close', 'Volume']]

    # calculate features
    data.loc[:, '5-day-ma'] = data['Adj Close'].rolling(window=5).mean()
    data.loc[:, '10-day-ma'] = data['Adj Close'].rolling(window=10).mean()
    data.loc[:, '20-day-ma'] = data['Adj Close'].rolling(window=20).mean()
    data.loc[:, 'RSI'] = talib.RSI(data['Adj Close'])
    
    # discretize into bins for q-table
    data.loc[:, 'Adj-Close-bin'] = data['Adj Close'].astype(int)
    data.loc[:, 'Volume-bin'] = data['Volume'].astype(int)
    data.loc[:, 'RSI-bin'] = pd.cut(data['RSI'], bins=STATES_DIM, labels=False)
    data.loc[:, '5-day-ma-bin'] = pd.cut(data['5-day-ma'], bins=STATES_DIM, labels=False)
    data.loc[:, '10-day-ma-bin'] = pd.cut(data['10-day-ma'], bins=STATES_DIM, labels=False)
    data.loc[:, '20-day-ma-bin'] = pd.cut(data['20-day-ma'], bins=STATES_DIM, labels=False)

    data = data.dropna()

    return data


# Manages account details for buying/selling stock
# Small brokerage fee applies to trades
class Account:
    def __init__(self, cash=100000, brokerage_fee=0.001):
        self.cash = cash
        self.brokerage_fee = brokerage_fee
        self.stocks = 0
        self.stock_ticker = None
        self.stock_owned = False

    def total_value(self, row):
        if self.stock_owned:
            return self.cash + row['Adj Close'] * self.stocks
        else:
            return self.cash

    def buy_stock(self, stock_ticker, row):
        if self.stock_owned:
            return
        self.stock_ticker = stock_ticker
        self.stocks = int(self.cash // (row['Adj Close'] * (1.0 + self.brokerage_fee)))
        self.cash -= self.stocks * row['Adj Close'] * 1.001
        self.stock_owned = True
        self.show_info(row, "Buy")

    def sell_stock(self, row):
        if not self.stock_owned:
            return
        self.show_info(row, "Sell")
        self.cash += self.stocks * (row['Adj Close'] * (1.0 - self.brokerage_fee))
        self.stock_ticker = None
        self.stocks = 0
        self.stock_owned = False

    def show_info(self, row, label="FINAL"):
        if self.stock_owned:
            print(label, self.stock_ticker, "TOTAL:", round(self.cash, 2) + self.stocks * round(float(row['Adj Close']),2))
            print(" - ", row.name, "price", round(row['Adj Close'], 2))
        else:
            print(label, "TOTAL", round(self.cash, 2))


# Manages Q-Table and Q-updates
class QAgent:
    def __init__(self, alpha, gamma, epsilon, epsilon_decay):
        self.learning_rate  = alpha
        self.discount_rate  = gamma
        self.exploration_rate  = epsilon
        self.decay_rate  = epsilon_decay

        self.bins_per_feature  = STATES_DIM
        self.dim = 6
        self.states = (self.bins_per_feature ** self.dim) * 2
        self.actions = 2 # buy/sell

        self.q_table_file_name = "q_table.pickle"
        self.q_table = defaultdict(lambda: np.zeros(self.actions))
        if os.path.isfile(self.q_table_file_name):
                print("Loading pickle")
                with open(self.q_table_file_name, "rb") as f:
                    self.q_table = defaultdict(lambda: np.zeros(self.actions), pickle.load(f))

    def save_q_table(self):
        with open(self.q_table_file_name, "wb") as f:
            pickle.dump(self.q_table, f)

    def get_state(self, row, stock_owned):
        dim = [
            int(row['Adj-Close-bin']),
            int(row['Volume-bin']),
            int(row['RSI-bin']),
            int(row['5-day-ma-bin']),
            int(row['10-day-ma-bin']),
            int(row['20-day-ma-bin'])
        ]

        # if any of the feature bin values are None, set them to 0
        dim = [0 if d is None else d for d in dim]

        # calculate the state dimension by combining the feature bin values
        dimension = sum(d * (self.bins_per_feature ** i) for i, d in enumerate(dim))

        if stock_owned:
            dimension += self.bins_per_feature ** self.dim

        return dimension

    def get_action(self, row, stock_owned):
        # get the current state based on the row and whether or not we own stock
        state = self.get_state(row, stock_owned)

        # based on exploration choose random action or take the action with the highest Q-value in the current state
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randrange(0, self.actions)
        else:
            action = np.argmax(self.q_table[state])

        return action, state

    def update_reward(self, row, stock_owned, last_action, last_state, reward):
        # get the next state based on the row and whether or not we own stock
        next_state = self.get_state(row, stock_owned)

        # update the Q-value of the last state-action pair based on the
        # reward received and the maximum Q-value in the next state
        old_value = self.q_table[last_state, last_action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_rate * next_max)
        self.q_table[last_state, last_action] = new_value

# Some helper functions

def split_data(data):
    train_data = data[:'2019']
    test_data = data['2020':]
    return train_data, test_data


def save_results_to_csv(training_total, testing_total, PL, filename='results.csv'):
    with open(filename, mode='a', newline='') as csvfile:
        fieldnames = ['training_total', 'testing_total', 'P/L']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if the file is new
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({'training_total': training_total, 'testing_total': testing_total, 'P/L': PL})


def main():
    # Define the stock ticker and years to look back on
    stock_ticker = 'ENPH'
    years_ago = 6
    data = get_stock_data(stock_ticker, years_ago)

    # Split data into training and testing sets
    train_data, test_data = split_data(data)

    q_learning = QAgent(ALPHA, GAMMA, EPSILON, EPSILON_DECAY)
    account = Account()

    state = None
    reward = 0.0
    action = 0
    last_value = 0.0

    runs = 20
    result_filename = 'results_ENPH.csv'
    while runs > 0:

        # ---- TRAINING ----

        # Loop through the stock data
        for index, row in train_data.iterrows():
            if state is not None:
                reward = account.total_value(row) - last_value
                q_learning.update_reward(row, account.stock_owned, action, state, reward)

            # Determine the next action
            action, state = q_learning.get_action(row, account.stock_owned)
            if action == 0:
                pass
            elif action == 1:
                if account.stock_owned:
                    account.sell_stock(row)
                else:
                    account.buy_stock(stock_ticker, row)

            # Update the last value (total value of cash and stocks) for the next iteration
            last_value = account.total_value(row)

            # Update the exploration rate (epsilon) using the decay rate
            q_learning.exploration_rate *= q_learning.decay_rate

        training_total = last_value
        q_learning.save_q_table()
        print("END OF TRAINING")
        print("START OF TESTING")

        # ---- TESTING ----
        time.sleep(1.5) # wait 1.5 seconds to ensure q-table save

        # Re-initialize QAgent (this will load the q-table made from testing data)
        q_learning = QAgent(ALPHA, GAMMA, EPSILON, EPSILON_DECAY)

        # Evaluate on the testing set with saved Q-table
        for index, row in test_data.iterrows():
            action, state = q_learning.get_action(row, account.stock_owned)
            if action == 0:
                pass
            elif action == 1:
                if account.stock_owned:
                    account.sell_stock(row)
                else:
                    account.buy_stock(stock_ticker, row)

            last_value = account.total_value(row)
            q_learning.exploration_rate *= q_learning.decay_rate

        testing_total = last_value

        # Delete the Q-table file after testing
        if os.path.exists(q_learning.q_table_file_name):
            os.remove(q_learning.q_table_file_name)

        time.sleep(1.5) # wait 1.5 seconds to ensure q-table remove

        print(" --------------------- ")
        print(f"TRAINING RESULT: ", round(training_total,2))
        print(f"TESTING RESULT: ", round(testing_total,2))

        pl = "PROFIT" if testing_total > training_total else "LOSS"
        save_results_to_csv(training_total, testing_total, pl, result_filename)

        runs -= 1


if __name__ == "__main__":
    main()