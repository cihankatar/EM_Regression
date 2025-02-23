import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

wd = os.getcwd()
file_path = wd+ "/data/DATA_01.xlsx" 
df_raw_data = pd.read_excel(file_path,sheet_name="Sayfa1",skiprows=3)

columns = ["Thickness", "Frequency (GHz)", "S11_real", "S11_imag", "S22_real", "S22_imag", "epsR_real", "epsR_imag"]

df_clean = df_raw_data.iloc[2:,[2, 3, 4, 5, 6, 7, 10, 11]]
df_clean.columns = columns

# 📌 3. ÖZELLİKLER VE ÇIKTILARI AYIRMA
X = df_clean.iloc[:, :6].values  # Thickness, Frequency, S11_real, S11_imag, S22_real, S22_imag
y = df_clean.iloc[:, 6:].values  # epsR_real, epsR_imag

# 📌 4. TRAIN-VALIDATION-TEST AYIRMA (%70 Train - %15 Val - %15 Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 📌 5. VERİYİ NORMALİZE ETME
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)
y_test = scaler_y.transform(y_test)

# 📌 6. TORCH TENSORLERİNE DÖNÜŞTÜRME
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 📌 7. MLP MODELİ
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # 2 çıkış: epsR_real, epsR_imag
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 📌 8. MODELİ BAŞLATMA
input_dim = X_train.shape[1]
model = MLPRegressor(input_size=input_dim)

# 📌 9. KAYIP FONKSİYONU VE OPTİMİZASYON
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📌 10. MODELİ EĞİTME
epochs = 200
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward Pass
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())

    # Validation Loss Hesaplama
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        val_loss = criterion(y_val_pred, y_val_tensor).item()
        val_losses.append(val_loss)

    # Her 10 epoch'ta bir ekrana yazdır
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

# 📌 11. TEST METRİKLERİ HESAPLAMA
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    test_loss = criterion(y_test_pred, y_test_tensor).item()
    y_test_pred = y_test_pred.numpy()
    y_test_true = y_test_tensor.numpy()

# 📌 12. TÜM TEST VERİSİ İÇİN MSE HESAPLAMA
mse_all = mean_squared_error(y_test_true, y_test_pred)
rmse_all = np.sqrt(mse_all)
mae_all = mean_absolute_error(y_test_true, y_test_pred)
r2_all = r2_score(y_test_true, y_test_pred)

# 📌 13. TEST SONUÇLARINI YAZDIR
print("\n📌 Test Set Metrics:")
print(f"✅ MSE (Mean Squared Error): {mse_all:.4f}")
print(f"✅ RMSE (Root Mean Squared Error): {rmse_all:.4f}")
print(f"✅ MAE (Mean Absolute Error): {mae_all:.4f}")
print(f"✅ R² Score (R-Squared): {r2_all:.4f}")
print(f"✅ Test Loss (MSE): {test_loss:.4f}")

# 📌 14. LOSS EĞRİSİNİ ÇİZME
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", color='blue')
plt.plot(val_losses, label="Validation Loss", color='red')
plt.axhline(y=test_loss, color='green', linestyle='--', label=f"Test Loss: {test_loss:.4f}")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid()
plt.savefig("loss_curve_mlp.png")
plt.show()
