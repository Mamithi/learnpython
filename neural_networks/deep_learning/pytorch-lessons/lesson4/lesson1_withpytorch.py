from torch import nn 

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        #imputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)

        # Output layer 10 units - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

# Create the network and look at its text representation
model = Network()
model
