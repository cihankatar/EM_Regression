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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)
y_test = scaler_y.transform(y_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

input_dim = X_train.shape[1]
model = MLP_Regressor(input_dim)  # Modeli tekrar oluştur
criterion = nn.MSELoss()
model.load_state_dict(torch.load("/output/mlp_regressor.pth"))  # Eğitilmiş ağırlıkları yükle
model.eval()  # Modeli eval moduna al

epochs = 200
model.eval()

with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    test_loss = criterion(y_test_pred, y_test_tensor).item()
    y_test_pred = y_test_pred.numpy()
    y_test_true = y_test_tensor.numpy()

mse = mean_squared_error(y_test_true, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_true, y_test_pred)
r2 = r2_score(y_test_true, y_test_pred)

print("\n📌 Test Set Metrics:")
print(f"✅ MSE (Mean Squared Error): {mse:.4f}")
print(f"✅ RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"✅ MAE (Mean Absolute Error): {mae:.4f}")
print(f"✅ R² Score (R-Squared): {r2:.4f}")
print(f"✅ Test Loss (MSE): {test_loss:.4f}")


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





sample_inputs,sample_label = X_test_tensor[1],y_test_tensor[1]

sample_inputs_scaled = scaler_X.transform(sample_inputs)  # Daha önce fit edilen scaler'ı kullan

sample_tensor = torch.tensor(sample_inputs_scaled, dtype=torch.float32)

with torch.no_grad():
    predictions = model(sample_tensor)

predictions = scaler_y.inverse_transform(predictions.numpy())

for i, pred in enumerate(predictions):
    print(f"✅ Örnek {i+1} - epsR_real: {pred[0]:.4f}, epsR_imag: {pred[1]:.4f}")
