import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 📌 1. VERİYİ OKUMA
file_path = "DATA_01.xlsx"
df_raw = pd.read_excel(file_path, sheet_name="Sayfa1", skiprows=3)

# 📌 2. VERİYİ TEMİZLEME
columns = ["Thickness", "Frequency (GHz)", "S11_real", "S11_imag", "S22_real", "S22_imag", "epsR_real", "epsR_imag"]
df_clean = df_raw.iloc[2:, [2, 3, 4, 5, 6, 7, 10, 11]].copy()
df_clean.columns = columns
df_clean = df_clean.astype(float)

# 📌 3. ÖZELLİKLER VE ÇIKTILARI AYIRMA
X = df_clean.iloc[:, :6].values  # Thickness, Frequency, S11_real, S11_imag, S22_real, S22_imag
y = df_clean.iloc[:, 6:].values  # epsR_real, epsR_imag

# 📌 4. TRAIN-TEST AYIRMA (%80 Train - %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 5. VERİYİ NORMALİZE ETME
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# 📌 6. TORCH TENSORLERİNE DÖNÜŞTÜRME
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 📌 7. MLP MODELİ TANIMLAMA
class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, 2)  # 2 çıkış: epsR_real, epsR_imag
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Son katmanda aktivasyon yok (regresyon)
        return x

# 📌 8. MODELİ BAŞLATMA
input_dim = X_train.shape[1]
model = MLPRegressor(input_size=input_dim)

# 📌 9. KAYIP FONKSİYONU VE OPTİMİZASYON
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📌 10. MODELİ EĞİTME
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward Pass
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    
    # Backpropagation
    loss.backward()
    optimizer.step()

    # Her 10 epoch'ta bir ekrana yazdır
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {loss.item():.4f}")

# 📌 11. TEST METRİKLERİ HESAPLAMA (TÜM TEST SETİ İÇİN)
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    test_loss = criterion(y_test_pred, y_test_tensor).item()
    y_test_pred = y_test_pred.numpy()
    y_test_true = y_test_tensor.numpy()

# 📌 12. REGRESYON METRİKLERİ (TÜM TEST VERİLERİ İÇİN)
mse = mean_squared_error(y_test_true, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_true, y_test_pred)
r2 = r2_score(y_test_true, y_test_pred)

# 📌 13. TEST SONUÇLARINI YAZDIR (TEST PLOT YOK)
print("\n📌 Test Set Metrics:")
print(f"✅ MSE (Mean Squared Error): {mse:.4f}")
print(f"✅ RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"✅ MAE (Mean Absolute Error): {mae:.4f}")
print(f"✅ R² Score (R-Squared): {r2:.4f}")
print(f"✅ Test Loss (MSE): {test_loss:.4f}")
