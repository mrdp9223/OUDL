import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

# --- Load CSV ---
df = pd.read_csv("RELIANCE_1min_3months.csv")

# --- Feature Engineering ---
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['price_change'] = df['close'].diff()
df['hour'] = pd.to_datetime(df['date']).dt.hour
df['minute'] = pd.to_datetime(df['date']).dt.minute
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
df.dropna(inplace=True)

# --- Features and Normalization ---
feature_cols = ['log_return', 'price_change', 'volume', 'hour_sin', 'minute_sin']
features = df[feature_cols].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# --- Inputs and Targets (predict next log return) ---
X = torch.tensor(features_scaled[:-1], dtype=torch.float32)
y_target = torch.tensor(features_scaled[1:, 0], dtype=torch.float32).unsqueeze(1)

# --- Train / Validation Split ---
N = len(X)
N_train = int(0.8 * N)
train_X, train_y = X[:N_train], y_target[:N_train]
val_X, val_y = X[N_train:], y_target[N_train:]

train_ds = TensorDataset(train_X, train_y)
val_ds = TensorDataset(val_X, val_y)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

# --- Neural Network Definition with Learnable Lambda ---
class OURegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.log_lambda = nn.Parameter(torch.tensor(-2.3))  # Initial lambda approx 0.1

    @property
    def lambda_(self):
        return F.softplus(self.log_lambda)

    def forward(self, x):
        return self.net(x)

model = OURegressor(input_dim=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- OU Loss with Learnable Lambda ---
def ou_loss(y_pred, lambda_):
    y_t = y_pred[:-1]
    y_next = y_pred[1:]
    return ((y_next - y_t + lambda_ * y_t) ** 2).mean()

# --- Training Loop ---
train_losses, val_losses = [], []
for epoch in range(20):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = ou_loss(pred, model.lambda_)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = ou_loss(pred, model.lambda_)
            val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}, lambda: {model.lambda_.item():.6f}")

# --- Plot Losses ---
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("OU Loss")
plt.title("OU Training Loss (Learnable lambda)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Predict on Full Data ---
model.eval()
with torch.no_grad():
    full_X = torch.tensor(features_scaled[:-1], dtype=torch.float32)
    y_pred_all = model(full_X).squeeze().numpy()

# --- ADF Test ---
result = adfuller(y_pred_all)
print(f"\nADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
if result[1] < 0.05:
    print("The learned signal is likely mean-reverting")
else:
    print("Signal may not be stationary")

# --- Time Index for Plotting ---
time_index = pd.to_datetime(df['date'].iloc[1:].values)[:len(y_pred_all)]

# --- Plot Learned Signal ---
plt.figure(figsize=(15, 4))
plt.plot(time_index, y_pred_all, label="Learned OU Signal", linewidth=1)
plt.axhline(0, color='gray', linestyle='--')
plt.title("Learned Signal Over Time")
plt.xlabel("Time")
plt.ylabel("Signal Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Correlation Analysis ---
future_log_return = df['log_return'].shift(-1).iloc[1:len(y_pred_all)+1].values
future_price_change = df['price_change'].shift(-1).iloc[1:len(y_pred_all)+1].values
future_close = df['close'].shift(-1).iloc[1:len(y_pred_all)+1].values

valid_idx = ~np.isnan(future_log_return) & ~np.isnan(future_price_change) & ~np.isnan(future_close)
signal = y_pred_all[valid_idx]
log_return = future_log_return[valid_idx]
price_change = future_price_change[valid_idx]
close_next = future_close[valid_idx]

corr_log = np.corrcoef(signal, log_return)[0, 1]
corr_change = np.corrcoef(signal, price_change)[0, 1]
corr_close = np.corrcoef(signal, close_next)[0, 1]

print(f"\nCorr with next log return: {corr_log:.4f}")
print(f"Corr with next price change: {corr_change:.4f}")
print(f"Corr with next close: {corr_close:.4f}")

# --- Z-score the signal ---
mean = np.mean(signal)
std = np.std(signal)
z_score = (signal - mean) / std

# --- Long-only Position Logic ---
position = np.zeros_like(z_score)
position[z_score < -0.1] = 1
position[np.abs(z_score) < 0.2] = 0

# Hold until neutral
for i in range(1, len(position)):
    if position[i] == 0:
        position[i] = position[i - 1]
position[np.abs(z_score) < 0.2] = 0

# --- Backtest ---
returns = log_return
strategy_returns = position * returns
cumulative_returns = np.cumsum(strategy_returns)

# --- Plot PnL ---
plt.figure(figsize=(14, 5))
plt.plot(cumulative_returns, label="Cumulative PnL (Long Only)", linewidth=1.5)
plt.title("Backtest of OU Signal (Long Only)")
plt.xlabel("Time (1-min steps)")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Summary Stats ---
total_return = cumulative_returns[-1]
sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252 * 390)
print(f"\nTotal Return: {total_return:.4f}")
print(f"Approx. Sharpe Ratio: {sharpe:.2f}")