# %% [code] {"id":"IHio-jVxNEWU","execution":{"iopub.status.busy":"2023-04-22T01:55:35.925045Z","iopub.execute_input":"2023-04-22T01:55:35.925307Z","iopub.status.idle":"2023-04-22T01:55:36.785490Z","shell.execute_reply.started":"2023-04-22T01:55:35.925279Z","shell.execute_reply":"2023-04-22T01:55:36.783967Z"}}
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# %% [code] {"id":"O26aH-r4SKtM","execution":{"iopub.status.busy":"2023-04-22T01:55:36.787180Z","iopub.execute_input":"2023-04-22T01:55:36.787874Z","iopub.status.idle":"2023-04-22T01:55:36.800971Z","shell.execute_reply.started":"2023-04-22T01:55:36.787828Z","shell.execute_reply":"2023-04-22T01:55:36.799514Z"}}

# Create a list to store the column data
xdata = []
ydata = []

# Open the CSV file in read mode
with open('/kaggle/input/heart-disease-cleveland/Heart_disease_cleveland_new.csv', 'r') as file:
    # Create a CSV reader object
    reader = csv.reader(file)

    # Skip the header row if present
    next(reader)

    # Iterate over each row in the CSV file
    for row in reader:
        # Convert the values in each column to float
        row = [float(val) for val in row]

        # Append the first n-1 values to xdata (inputs)
        xdata.append(row[:len(row)-1])

        # Append the last value to ydata (output)
        ydata.append(row[len(row)-1])

# Convert xdata and ydata to numpy arrays for further processing
xdata = np.array(xdata)
ydata = np.array(ydata)


# %% [code] {"id":"thKGd9zRVDzY","execution":{"iopub.status.busy":"2023-04-22T01:55:36.803026Z","iopub.execute_input":"2023-04-22T01:55:36.803389Z","iopub.status.idle":"2023-04-22T01:55:36.808868Z","shell.execute_reply.started":"2023-04-22T01:55:36.803352Z","shell.execute_reply":"2023-04-22T01:55:36.807830Z"}}
def listToTensor(L):
  return torch.tensor(L)

# %% [code] {"id":"1DnlVnoGWnov","outputId":"1831be29-63ec-4a9d-d5ec-1f5434257ba6","execution":{"iopub.status.busy":"2023-04-22T01:55:36.812167Z","iopub.execute_input":"2023-04-22T01:55:36.812512Z","iopub.status.idle":"2023-04-22T01:55:36.820580Z","shell.execute_reply.started":"2023-04-22T01:55:36.812430Z","shell.execute_reply":"2023-04-22T01:55:36.819353Z"}}
print('xdata - ', type(xdata), len(xdata))
print('ydata - ', type(ydata), len(ydata))

# %% [code] {"id":"4XNeYSUmOUMS","execution":{"iopub.status.busy":"2023-04-22T01:55:36.821591Z","iopub.execute_input":"2023-04-22T01:55:36.822906Z","iopub.status.idle":"2023-04-22T01:55:36.831314Z","shell.execute_reply.started":"2023-04-22T01:55:36.822878Z","shell.execute_reply":"2023-04-22T01:55:36.829398Z"}}
xdata = listToTensor(xdata)
ydata = listToTensor(ydata)

# %% [code] {"id":"JH5qD-J4OuIO","outputId":"6ef09726-ef1f-44e9-8d47-79e920454768","execution":{"iopub.status.busy":"2023-04-22T01:55:36.832863Z","iopub.execute_input":"2023-04-22T01:55:36.833178Z","iopub.status.idle":"2023-04-22T01:55:36.840681Z","shell.execute_reply.started":"2023-04-22T01:55:36.833097Z","shell.execute_reply":"2023-04-22T01:55:36.839632Z"}}
print('xdata - ', type(xdata), len(xdata))
print('ydata - ', type(ydata), len(ydata))

# %% [code] {"id":"argME9FhSSL6","execution":{"iopub.status.busy":"2023-04-22T01:55:36.842191Z","iopub.execute_input":"2023-04-22T01:55:36.842450Z","iopub.status.idle":"2023-04-22T01:55:36.849153Z","shell.execute_reply.started":"2023-04-22T01:55:36.842424Z","shell.execute_reply":"2023-04-22T01:55:36.847711Z"}}
trainPerc = 0.9
x_train = xdata[:int(len(xdata)*trainPerc)]
x_test = xdata[int(len(xdata)*trainPerc):]

y_train = ydata[:int(len(ydata)*trainPerc)]
y_test = ydata[int(len(ydata)*trainPerc):]

# %% [code] {"id":"LBQdFzDJP5td","outputId":"006a271f-5917-49f2-9d3f-fd4f80480d78","execution":{"iopub.status.busy":"2023-04-22T01:55:36.850710Z","iopub.execute_input":"2023-04-22T01:55:36.850970Z","iopub.status.idle":"2023-04-22T01:55:36.862845Z","shell.execute_reply.started":"2023-04-22T01:55:36.850945Z","shell.execute_reply":"2023-04-22T01:55:36.861967Z"}}
x_train = x_train.float()
y_train = y_train.float()

# %% [code] {"id":"IUBZATTa_9li","execution":{"iopub.status.busy":"2023-04-22T01:55:36.864031Z","iopub.execute_input":"2023-04-22T01:55:36.864312Z","iopub.status.idle":"2023-04-22T01:55:36.877030Z","shell.execute_reply.started":"2023-04-22T01:55:36.864283Z","shell.execute_reply":"2023-04-22T01:55:36.875893Z"}}

# Make sure that x_train and y_train have the correct shapes
# Assuming x_train has shape (num_samples, num_features) and y_train has shape (num_samples,)

# Create a TensorDataset to wrap the training data
train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).unsqueeze(1))
# Use unsqueeze(1) to add a singleton dimension to y_train to match the expected shape for labels in MSE loss

# Create a DataLoader to handle batching and shuffling of the training data
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)

# Define the neural network model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Define the layers using nn.ModuleList
        self.layers = nn.Sequential(
            nn.Linear(13, 100),
            nn.ReLU(),  # Use ReLU activation function for improved training
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Linear(60, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Create an instance of the model and define the loss function and optimizer
model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# %% [code] {"id":"Qrf5ydd6ckAy","outputId":"009e6ac9-5643-4ccc-ead1-efa95d676f55","execution":{"iopub.status.busy":"2023-04-22T01:55:36.878591Z","iopub.execute_input":"2023-04-22T01:55:36.879413Z","iopub.status.idle":"2023-04-22T01:56:28.884051Z","shell.execute_reply.started":"2023-04-22T01:55:36.879366Z","shell.execute_reply":"2023-04-22T01:56:28.882639Z"}}
#Train
for epoch in range(10000):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # Zero the gradients before each batch
        optimizer.zero_grad()

        # Forward pass through the network
        outputs = model(inputs)

        # Compute the loss between the predicted and actual outputs
        loss = criterion(outputs, labels)

        # Backward pass through the network to compute the gradients
        loss.backward()

        # Update the weights using the gradients and optimizer
        optimizer.step()

        # Accumulate the running loss
        running_loss += loss.item()

    # Print the average loss over the epoch
    if(epoch%100==1):
      print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / (i + 1)))


# %% [code] {"id":"2cgKKC4MDg5p","outputId":"cb488611-caf4-4ec6-f31a-9926055e8c8c","execution":{"iopub.status.busy":"2023-04-22T01:57:27.845090Z","iopub.execute_input":"2023-04-22T01:57:27.845482Z","iopub.status.idle":"2023-04-22T01:57:27.901014Z","shell.execute_reply.started":"2023-04-22T01:57:27.845450Z","shell.execute_reply":"2023-04-22T01:57:27.900179Z"}}
def accuracy(xvalues,yvalues):
  correct=0
  for i in range(len(xvalues)):
    prediction = float(model.forward(torch.tensor(xvalues[i],dtype=torch.float32)))
    if(int(prediction+0.5)==yvalues[i]):
      correct+=1
  print('Accuracy:',correct/len(xvalues)*100,'%')
print('Test data:')
accuracy(x_test,y_test)
print('Train data')
accuracy(x_train,y_train)