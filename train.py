#not updated yet



from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader

from model import *
from common_functions import *

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(data.size(0)-seq_length*2):
        x = data[i:(i+seq_length)]
        y = data[(i+seq_length):(i+seq_length*2)]
        xs.append(x)
        ys.append(y)

    return torch.stack(xs), torch.stack(ys)

file1 = "all_training_data_sorted.csv"
north = 'data_north_p.csv'
east = 'data_east_p.csv'
south = 'data_south_p.csv'
west = 'data_west_p.csv'
device = 'cuda'

imported_data_raw = list(group_column(file1, north, east, south, west))
imported_data = torch.tensor(imported_data_raw)

k = 16  # Dimension of transformer key/query/value vectors 10
heads = 16  # Number of attention heads 2
depth = 12 # Number of transformer blocks 2
num_features = 16  # Number of input features (temperature, pressure)
seq_length = 72  # Length of the sequence

X, y = create_sequences(imported_data, seq_length)

# Calculate split sizes
num_samples = len(X)
num_train = round(num_samples * 0.8)
num_val = num_samples - num_train

# Create data sets
train_data, val_data = random_split(TensorDataset(X, y), [num_train, num_val])

batch_size = 32

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

model = Predictor(k, heads, depth, seq_length, num_features, max_seq_len=80).to(device)

#load model:
#model.load_state_dict(torch.load('weather_pfullingen.pth'))

# Training Loop
num_epochs = 10
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        # Move data to GPU if available
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Get model predictions for the inputs

        # Compute the loss
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights

    print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_losses = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            val_losses.append(val_loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, num_epochs, avg_val_loss))