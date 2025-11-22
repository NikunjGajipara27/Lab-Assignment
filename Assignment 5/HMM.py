import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# -------- PART 1: DATA COLLECTION --------
ticker = "AAPL"
data = yf.download(ticker, start="2014-01-01", end="2024-12-31", auto_adjust=False)
prices = data['Close']

# Compute log returns
returns = np.log(prices / prices.shift(1)).dropna()

# Align prices with returns
prices = prices.iloc[1:]

# Prepare data for HMM
X = returns.values.reshape(-1, 1)

# -------- PART 2: FIT GAUSSIAN HMM --------
model = GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=300,
    random_state=42
)
model.fit(X)
states = model.predict(X)

# -------- PART 4: VISUALIZATION --------
plt.figure(figsize=(12,5))

for s in range(model.n_components):
    idx = (states == s)
    plt.plot(prices.index[idx], prices.values[idx],
             '.', label=f"State {s}")

plt.legend(title="Market Regimes")
plt.title("Gaussian HMM Hidden Regimes for Apple (AAPL) Price Series")
plt.xlabel("Date")
plt.ylabel("Price ($)")       # <-- UPDATED
plt.tight_layout()
plt.savefig("assignment5_hmm_regimes.png", dpi=300)
plt.show()

print("State Means:\n", model.means_.ravel())
print("\nState Variances:\n",
      [np.diag(cov)[0] for cov in model.covars_])
print("\nTransition Matrix:\n", model.transmat_)
