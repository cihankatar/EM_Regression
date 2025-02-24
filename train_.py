import pandas as pd 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os 
from models.MLP_regressor import MLP_Regressor
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


wd = os.getcwd()
file_path = wd+ "/data/DATA_01.xlsx" 
df_raw_data = pd.read_excel(file_path,sheet_name="Sayfa1",skiprows=3)

columns = ["Thickness", "Frequency (GHz)", "S11_real", "S11_imag", "S22_real", "S22_imag", "epsR_real", "epsR_imag"]

df_clean = df_raw_data.iloc[2:,[2, 3, 4, 5, 6, 7, 10, 11]]
df_clean.columns = columns

X = df_clean.iloc[:, :6].astype(float).values  # Thickness, Frequency, S11_real, S11_imag, S22_real, S22_imag
y = df_clean.iloc[:, 6:].astype(float).values  # epsR_real, epsR_imag

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

# 📌 7. DATALOADER (BATCH TRAINING)
batch_size = 30000
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 📌 8. MODELİ OLUŞTURMA
input_dim = X_train.shape[1]
model = MLP_Regressor(input_dim)

# 📌 9. KAYIP FONKSİYONU VE OPTİMİZASYON
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📌 10. MODELİ EĞİTME
epochs = 200
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred,y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    train_losses.append(epoch_train_loss / len(train_loader))

        # Validation Loss Hesaplama
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_val_pred = model(X_batch)
            val_loss = criterion(y_val_pred, y_batch)
            epoch_val_loss += val_loss.item()

    val_losses.append(epoch_val_loss / len(val_loader))

    # Her 10 epoch'ta bir ekrana yazdır
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")


# 📌 11. TEST METRİKLERİ HESAPLAMA
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    test_loss = criterion(y_test_pred, y_test_tensor).item()
    y_test_pred = y_test_pred.numpy()
    y_test_true = y_test_tensor.numpy()

# 📌 12. REGRESYON METRİKLERİ
mse = mean_squared_error(y_test_true, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_true, y_test_pred)
r2 = r2_score(y_test_true, y_test_pred)

# 📌 13. TEST SONUÇLARINI YAZDIR
print("\n📌 Test Set Metrics:")
print(f"✅ MSE (Mean Squared Error): {mse:.4f}")
print(f"✅ RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"✅ MAE (Mean Absolute Error): {mae:.4f}")
print(f"✅ R² Score (R-Squared): {r2:.4f}")
print(f"✅ Test Loss (MSE): {test_loss:.4f}")

# 📌 14. MODELİ KAYDETME
folder_path = file_path = wd + "/output" 
model_path = "/mlp_regressor.pth"
save_path = folder_path + model_path
torch.save(model.state_dict(), save_path)
print(f"\n✅ Model saved: {model_path}")

# 📌 15. TEST SONUÇLARINI KAYDETME
test_results = {
    "Epoch": epochs,
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "R2": r2,
    "Test Loss": test_loss
}

df_results = pd.DataFrame([test_results])
df_results.to_csv("test_results.csv", index=False)
print("\n✅ Test sonuçları başarıyla kaydedildi: test_results.csv")


# 📌 16. LOSS GRAFİĞİNİ ÇİZDİRME
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="blue")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Eğitim ve Validation Loss Grafiği")
plt.legend()
plt.grid()
plt.savefig("/output/loss_curve_mlp.png")
plt.show()
