import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import numpy as np
from sklearn.metrics import mean_squared_error

class TorchLinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def train_torch(opt_name, xtr, ytr, xte, yte, epochs, batch_size, lr_dict):
    device = torch.device("cpu")  # или "cuda" если доступно

    model = TorchLinearModel(xtr.shape[1]).to(device)
    criterion = nn.MSELoss()

    # Выбор оптимизатора
    lr = lr_dict[opt_name]
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "SGD+Momentum":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "SGD+Nesterov":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    # Обучение
    start_time = time.time()
    process = psutil.Process()

    dataset = torch.utils.data.TensorDataset(xtr, ytr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb.squeeze())
            loss.backward()
            optimizer.step()

    duration = time.time() - start_time
    memory = process.memory_info().rss / 1024**2  # В мегабайтах

    # Оценка на тесте
    # Оценка на тесте
    model.eval()
    with torch.no_grad():
        preds = model(xte).squeeze().numpy()
        y_true = yte.squeeze().numpy()
        mse = mean_squared_error(np.atleast_1d(y_true), np.atleast_1d(preds))

        # Вывод предсказаний
        # print(f"[{opt_name}] True y (scaled): {y_true}")
        # print(f"[{opt_name}] Pred y (scaled): {preds}")

    return {
        "optimizer": opt_name,
        "framework": "PyTorch",
        "mse": mse,
        "time": duration,
        "memory": memory,
    }
