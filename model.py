import torch
import torch.nn as nn

class WeatherForecast(nn.Module):
    def __init__(self, input_features, hidden_dim, num_layers, output_dim, dropout_rate=0.0):
        super(WeatherForecast, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_features, hidden_dim, num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout1(out)
        out = self.fc(out[:, -1, :])
        return out