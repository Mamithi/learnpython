import cv2 
import matplotlib.pyplot as plt 
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()
        # Initilize the weights of te convolution layer to be the weights of the defined filters
        k_height, k_width = weight.shape[2:]

        # Assumes there are 4 gray scale filters
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight  = torch.nn.Parameter(weight)

    
    def forward(self, x):
        # Calculates the output of a convolutional layer
        conv_x  =self.conv(x)
        activated_x = F.relu(conv_x)

        # returns both layers
        return conv_x, activated_x


img_path = 'data/udacity_sdc.png'

# load color image
bgr_img = cv2.imread(img_path)

# Covert to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie in [0,1]
gray_img =gray_img.astype("float32")/255

# plot image
plt.imshow(gray_img, cmap='gray')
plt.show()

# Define and visualize the filters
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

# define four filtersorward behavior of a network that ap
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])


# Visualize all four filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
            horizontalalignment='center',
            verticalalignment='center',
            color='white' if filters[i][x][y] < 0 else 'black')

plt.show()

# Instantiat the model and set teh weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)


# Convert the image into an input Tensor
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

# get the convolutional layer( pre and post activation)
conv_layer, activated_layer = model(gray_img_tensor)


def viz_layer(layer, n_filters=4):
        fig = plt.figure(figsize=(20, 20))

        for i in range(n_filters):
            ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
            ax.set_title('Output %s' % str(i+1))

        plt.show()


# Visualize the output of the conv layer
viz_layer(conv_layer)
viz_layer(activated_layer)