from data_processing import *

filepath = 'lstm_on_bouy.pth'
input_size = 3 # number of features
hidden_size = 64 # number of features in hidden state
num_layers = 3 # number of stacked lstm layers
num_classes = 2
lstm = LSTM(num_classes, input_size, hidden_size, num_layers).to(device)
lstm.load_state_dict(torch.load(filepath, map_location=device))


dataX, do, future_predicts, train_size, val_size, mean, std = Predict(1,6,lstm)
print(future_predicts)