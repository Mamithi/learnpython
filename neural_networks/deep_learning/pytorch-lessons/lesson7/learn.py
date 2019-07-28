import torch
from torch import nn
import numpy as np  
import matplotlib.pyplot as plt 

plt.figure(figsize=(8, 5))

# How many time steps/data are in one batch of data
seq_length = 20
# generate evenly spaced data pts
time_steps = np.linspace(0, np.pi, seq_length+1)
data = np.sin(time_steps)
data.resize(seq_length+1, 1)

x = data[:-1] # all but the last piece of data
y = data[1:] # all but the first

# display the data
plt.plot(time_steps[1:], x, 'r.', label='input, x')
plt.plot(time_steps[1:], y, 'b.', label='target, y')

plt.legend(loc='best')
plt.show()

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        # Define an RNN with a specified patameters
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # last, fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*sequence_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)
        # get final output
        output = self.fc(r_out)

        return output, hidden

# test that dimension are as expected
test_run = RNN(input_size=1, output_size=1, hidden_dim=10, n_layers=2)
# generate evenly spaced, test data pts
time_steps = np.linspace(0, np.pi, seq_length)
data = np.sin(time_steps)
data.resize((seq_length, 1))

test_input = torch.Tensor(data).unsqueeze(0)
# give it a batch_size of 1 as first dimension
print('Input size: ', test_input.size())

# test out rnn sizes
test_out, test_h = test_run(test_input,None)
print('Output size: ', test_out.size())
print('Hidden state size: ', test_h.size())

# Traiing teh RNN
# decide on hyperparameters
input_size  = 1
output_size = 1
hidden_dim = 32
n_layers = 1

# instantiate an RNN
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)

# loss and optimization
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# Defining the training function
def train(rnn, n_steps, print_every):
    # initialize teh hidden state
    hidden = None
    for batch_i, step in enumerate(range(n_steps)):
        # defining the training data
        time_steps = np.linspace(step * np.pi, (step+1)*np.pi, seq_length+1)
        data = np.sin(time_steps)
        data.resize((seq_length+1, 1))
        x = data[:-1]
        y = data[1:]

        # convert data into Tensors
        x_tensor = torch.Tensor(x).unsqueeze(0)
        y_tensor = torch.Tensor(y)

        # outpus fro the rnn
        prediction, hidden = rnn(x_tensor, hidden)

        hidden= hidden.data

        # calculate the loss
        loss = criterion(prediction, y_tensor)

        #zero gradients
        optimizer.zero_grad()

        # perform backdrop and update weight
        loss.backward()
        optimizer.step()

        # disply loss and prediction
        if batch_i%print_every == 0:
            print('Loss: ', loss.item())
            plt.plot(time_steps[1:], x, 'r.') # input
            plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.') # prediction
            plt.show()
    return rnn

# train the rnn and monitor results
n_steps = 75
print_every = 15

trained_rnn = train(rnn, n_steps, print_every)