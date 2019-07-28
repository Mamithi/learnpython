import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# open text file and read in data as 'text'
with open('data/anna.txt', 'r') as f:
    text = f.read()

# encode the text and map each character to an integer and vice versa
###################################################
## We create two dictionaries                 #####
## 1. int2char, which maps integers to        #####
##    characters.                             #####
## 2. char2int, which maps characters to      #####
## unique integers                            #####
###################################################
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# encode the text
encoded = np.array([char2int[ch] for ch in text])

# Pre-processing the data


def one_hot_encode(arr, n_labels):
    # initialize the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    # fill teh appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1

    # finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


test_seq = np.array([[3, 5, 1]])
one_hot = one_hot_encode(test_seq, 8)


def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size * seq_length
    # total number pf batches we can make
    n_batches = len(arr)//batch_size_total
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


# Tesy the implementation
batches = get_batches(encoded, 8, 50)
x, y = next(batches)

# printing out teh first 10 items in a sequence
# print('x\n', x[:10, :10])
# print('\ny\n', y[:10, :10])

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print("Training on GPU!")
else:
    print('No GPU available, training on CPU; consider making epochs very small.')


class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden,
                            n_layers, dropout=drop_prob, batch_first=True)
        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # Define the final and fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        # get the outputs and the new hidden state from the ltsm
        r_output, hidden = self.lstm(x, hidden)
        # Pass through a dropout layer
        out = self.dropout(r_output)
        # stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)
        # put x through teh fully connected
        out = self.fc(out)
        # return teh final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

# Training the network


def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if(train_on_gpu):
        net.cuda()

    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            # one hot encode our data and make tehm torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # creating new variables for the hidden state and otherwise
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate teh loss and perform backprop
            loss = criterion(output, targets.view(
                batch_size*seq_length).long())
            loss.backward()

            # clip gradnorm helps prevent the exploding gradient problem in RNNs/LSTMs
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss starts
            if counter % print_every == 0:
                # get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # creating new variables for teh hidden state and otherwise
                    val_h = tuple([each.data for each in val_h])
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(
                        batch_size*seq_length).long())
                    val_losses.append(val_loss.item())

                net.train()  # Reset to train model after iterating through validation data

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


# Instantiating the model
# define and print the net
n_hidden = 512
n_layers = 2

net = CharRNN(chars, n_hidden, n_layers)
# print(net)

batch_size = 128
seq_length = 100
n_epochs = 20
# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size,
      seq_length=seq_length, lr=0.001, print_every=10)

# change the name, for saving multiple files
model_name = 'rnn_20_epoch.net'
checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)

def predict(net, char, h=None, top_k=None):
    # Given a character, predict the next character
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))

    inputs = torch.from_numpy(x)
    if(train_on_gpu):
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])
    out, h = net(inputs, h)

    # get teh character probabilities
    p = F.softmax(out, dim=1).data
    if(train_on_gpu):
        p = p.cpu() # move to cpu


    # get top charachters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select teh likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    #return teh encoded value of teh predicted and teh hideen state
    return net.int2char[char], h
# Priming and generating text
def sample(net, size, prime='The', top_k=None):
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval()

    # First off, run through teh prime charachters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)
    chars.append(char)
    # now pass in the previous charachter and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

print(sample(net, 1000, prime='Anna', top_k=5))

# Loading a checkpoint
with open('rnn_20_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)

loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])

# sample using a loaded model
print(sample(loaded, 2000, top_k=5, prime='And Levin said'))
