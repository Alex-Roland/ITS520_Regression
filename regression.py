import numpy as np
import torch
import pandas as pd
import sklearn
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import xgboost as xgb
import onnxruntime as rt
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType

# Parameters
batch_size = 32
learning_rate = 0.0003
N_epochs = 1000
epsilon = 0.0001

# Read Data
path_data = 'all_data_merged.csv'
temp_raw_data_df = pd.read_csv(path_data, delimiter=",")

headers_list = temp_raw_data_df.columns.values.tolist()

# Data Analysis
cm = np.corrcoef(temp_raw_data_df[headers_list].values.T)
hm = heatmap(cm, row_names=headers_list, column_names=headers_list, figsize=(20,10))
plt.show()

# Process Data
temp_raw_data_np = temp_raw_data_df.to_numpy()

X = temp_raw_data_np[:, :-2]
y = temp_raw_data_np[:, 2:4]

random_seed = int(random.random() * 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Fix in case float64 error
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

X_train_tr = torch.from_numpy(X_train)
X_test_tr = torch.from_numpy(X_test)
y_train_tr = torch.from_numpy(y_train)
y_test_tr = torch.from_numpy(y_test)

# Normalization
x_means = X_train_tr.mean(0, keepdim=True)
x_deviations = X_train_tr.std(0, keepdim=True) + epsilon

# Create the DataLoader
train_ds = TensorDataset(X_train_tr, y_train_tr)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Neural Network Architectures

## Linear Regression
class LinRegNet(nn.Module):
    ## init the class
    def __init__(self, x_means, x_deviations):
        super().__init__()
        
        self.x_means      = x_means
        self.x_deviations = x_deviations
        
        self.linear1 = nn.Linear(2, 2)
        
    ## perform inference
    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        
        y_pred = self.linear1(x)
        return y_pred

## MLP
class MLP_Net(nn.Module):
    ## init the class
    def __init__(self, x_means, x_deviations):
        super().__init__()
        
        self.x_means      = x_means
        self.x_deviations = x_deviations
        
        self.linear1 = nn.Linear(2, 8)
        self.act1    = nn.Sigmoid()
        self.linear2 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(0.25)
        
    ## perform inference
    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        
        x = self.linear1(x)
        x = self.act1(x)
        ## x = self.dropout(x)

        y_pred = self.linear2(x)
        return y_pred

## Deep Learning with hidden layers
class DL_Net(nn.Module):
    ## init the class
    def __init__(self, x_means, x_deviations):
        super().__init__()
        
        self.x_means      = x_means
        self.x_deviations = x_deviations
        
        self.linear1 = nn.Linear(2, 16)
        self.act1    = nn.ReLU()
        self.linear2 = nn.Linear(16, 8)
        self.act2    = nn.ReLU()
        self.linear3 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(0.25)
        
    ## perform inference
    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        ## x = self.dropout(x)

        y_pred = self.linear3(x)
        return y_pred

# Training Loop
def training_loop(N_Epochs, model, loss_fn, opt):
    for epoch in range(N_Epochs):
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss   = loss_fn(y_pred, yb)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        if epoch % 20 == 0:
            print(epoch, "loss=", loss)

model = LinRegNet(x_means, x_deviations)
# model = MLP_Net(x_means, x_deviations)
# model = DL_Net(x_means, x_deviations)

opt     = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = F.mse_loss

training_loop(N_epochs, model, loss_fn, opt)

# Evaluate Model
y_pred_test = model(X_test_tr)

print("Testing R**2: ", r2_score(y_test_tr.numpy(), y_pred_test.detach().numpy()))

list_preds = []
list_reals = []

for i in range(len(X_test_tr)):
    print("************************************")
    print("pred, real")
    np_real = y_test_tr[i].detach().numpy()
    np_pred = y_pred_test[i].detach().numpy()
    print((np_pred, np_real))
    list_preds.append(np_pred[0])
    list_reals.append(np_real[0])

# Deploy PyTorch Model
model.eval()

dummy_input = torch.randn(1, 2)

input_names  = ["input1"]
output_names = ["output1"]

torch.onnx.export(
        model,
        dummy_input,
        "temperature_humidity_data.onnx",
        input_names = input_names,
        output_names = output_names,
        opset_version=15,
        do_constant_folding=True,
        dynamic_axes={
            "input1": {0: "batch"},
            "output1": {0: "batch"}
        }
)
print("ONNX model saved")