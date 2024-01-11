import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# Data set construction
class CustomTimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, num_steps):
        self.data = data
        self.seq_length = seq_length
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) - self.seq_length - self.num_steps + 1

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_length]
        target_seq = self.data[idx + self.seq_length:idx + self.seq_length + self.num_steps]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_steps):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        outputs = []

        hn, cn = h0, c0
        step_input = x

        # Iterate for multi-step prediction
        for _ in range(self.num_steps):
            step_output, (hn, cn) = self.lstm(step_input, (hn, cn))
            step_output = self.fc1(step_output[:, -1, :])
            outputs.append(step_output)
            next_input = torch.cat((x[:, 1:, :], step_output.unsqueeze(1)), dim=1)
            step_input = next_input
        return torch.cat(outputs, dim=1).unsqueeze(-1)


def train(model, data_loader, criterion, optimizer, scheduler, num_epochs, save_path):
    model.train()
    epoch_losses = []  # Used to record losses for each epoch

    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(data_loader)
        epoch_losses.append(average_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
        scheduler.step()

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    return epoch_losses


def predict(model, test_data_loader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_data_loader):
            # Discard insufficient batch
            if inputs.size(0) != batch_size:
                continue
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().cpu().numpy())
            actuals.extend(targets.squeeze().cpu().numpy())

            # Draw images of some of the results
            if i % 20 == 0:
                truth = targets.cpu().numpy()[0, :, -1]
                forecasting = outputs.cpu().numpy()[0, :, -1]
                plot_predictions(truth, forecasting, i)

    return predictions, actuals


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')


def calculate_metrics(actuals, predictions):
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mae, mse, mape


def plot_predictions(actuals, predictions, name):
    plt.figure()
    plt.plot(actuals, label='GroundTruth', linewidth=2)
    plt.plot(predictions, label='Prediction', linewidth=2)
    plt.title('Result in test set')
    plt.xlabel('Time Step')
    plt.ylabel('power consumption')
    plt.legend()
    plt.show()
    plt.savefig(str(name), bbox_inches='tight')


def plot_loss(epoch_losses):
    plt.figure()
    plt.plot(epoch_losses, label='Training Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def load_model(model_path, model_class, input_size, hidden_size, output_size, num_layers, num_steps):
    model = model_class(input_size, hidden_size, output_size, num_layers, num_steps)
    model_state = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state)
    return model


# Defining hyperparameters
input_size = 1  # Input dimensions
hidden_size = 100  # Model dimensions
output_size = 1  # Output dimensions
num_layers = 2  # Lstm Number of layers
seq_length = 96  # Length of the input sequence
num_steps = 48  # The number of future time steps to be predicted
learning_rate = 0.001
num_epochs = 100
batch_size = 32

# Data set reading and cutting
file_path = '/home/lzy/Desktop/dataset/power/2018-2020.csv'
data = pd.read_csv(file_path)
data = data["OT"]
scaler = StandardScaler()
scaler.fit(data.values.reshape(-1, 1))
data = scaler.transform(data.values.reshape(-1, 1))
num_train = int(len(data) * 0.7)
num_test = int(len(data) * 0.2)
num_vali = len(data) - num_train - num_test
train_data = data[0:num_train]
test_data = data[len(data) - num_test - seq_length: len(data)]
vail_data = data[num_train - seq_length: num_train + num_vali]

# Translation into supervised learning
train_dataset = CustomTimeSeriesDataset(train_data, seq_length, num_steps)
test_dataset = CustomTimeSeriesDataset(test_data, seq_length, num_steps)
vail_dataset = CustomTimeSeriesDataset(vail_data, seq_length, num_steps)
print("train:", len(train_dataset))
print("vail:", len(vail_dataset))
print("test:", len(test_dataset))
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = '/home/lzy/Desktop/lstm48.pth'

Istrain = 1  # Whether to read the model
if Istrain == 1:
    # Train a new model
    model = LSTMModel(input_size, hidden_size, output_size, num_layers, num_steps)
    model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    epoch_losses = train(model, train_data_loader, criterion, optimizer, scheduler, num_epochs, save_path)
    plot_loss(epoch_losses)
else:
    # load existing model
    model = load_model(save_path, LSTMModel, input_size, hidden_size, output_size, num_layers, num_steps)
    model.to(device)
    print("successful load model")

# Predict in test set
predictions, actuals = predict(model, test_data_loader)
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

mae, mse, mape = calculate_metrics(actuals, predictions)
print(f"MAE: {mae}, MSE: {mse}, MAPE: {mape}%")

# Draw
num_sequences = 5
sequence_length = 48

predictions_array = np.array(predictions).reshape(-1, sequence_length)[:num_sequences]
actuals_array = np.array(actuals).reshape(-1, sequence_length)[:num_sequences]

plt.figure(figsize=(15, 10))
for i in range(num_sequences):
    plt.subplot(num_sequences, 1, i + 1)
    plt.plot(predictions_array[i], label='Predictions', marker='o')
    plt.plot(actuals_array[i], label='Actuals', marker='x')
    plt.title(f'LSTM Sequence {i + 1}')
    plt.xlabel('Time Step')
    plt.ylabel('Power')
    if i == 0:
        plt.legend()
plt.tight_layout()
plt.show()
