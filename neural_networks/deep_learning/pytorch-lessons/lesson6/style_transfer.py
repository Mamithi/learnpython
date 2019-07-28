from PIL import Image
from io import BytesIO 
import matplotlib.pyplot as plt 
import numpy as np 

import torch 
import torch.optim as optim 
import requests 
from torchvision import transforms, models 

# Get the "features" portion of VGG19
vgg = models.vgg19(pretrained=True).features

# Freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

def load_image(img_path, max_size=400, shape=None):
    # Load in and transform an image
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")

    else:
        image = Image.open(img_path).convert("RGB")

    # Large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225))
    ])

    # Discard the transparent, alpha channel (that's teh :3) and add teh batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image

# load in content and style image
content = load_image('images/octopus.jpg').to(device)
# Resize style to match content, makes code easier
style = load_image('images/hockney.jpg', shape=content.shape[-2:]).to(device)

# Function to un-normalize an image and converting it from tensor image to a numpy image for display
def im_convert(tensor):
    # Display a tensor as an aimage
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# Content and style ims side by side
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))

plt.show()

# print out vgg19 structure so you can see the names of various layers
# print(vgg)

def get_features(image, model, layers=None):
    # Run an image forward through a model and get features for a set of layers.
    # Default layers are for VGGNet matching Gatys
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '15': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1',
        }
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

def gram_matrix(tensor):
    # Calculate teh Gram Matrix of a given tensor
    # get teh batch_size, depth, height and width of the Tensor
    _, d, h, w = tensor.size()

    # reshape so we're multiplying teh features for each comment
    tensor = tensor.view(d, h*w)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram

# get content and style features only once before tarining
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Calculate teh gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create a third target image and prep it for change
# it is a good idea to start off with teh target as a copy of our *content* image
#then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# weights for each style layer
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
stye_weights = {
    'conv1_1' : 1.,
    'conv2_1' : 0.75,
    'conv3_1' : 0.2,
    'conv4_1' : 0.2,
    'conv5_1' : 0.2,
}

content_weight = 1 # alpha
style_weight = 1e6 # beta

# for displaying the target image, intermittently
show_every = 400

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 2000

for ii in range(1, steps+1):
    # get the features from your target image
    target_features = get_features(target, vgg)

    # Content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    # the style loss
    # intialize teh style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in stye_weights:
        # get teh "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)

        _, d, h, w = target_feature.shape
        # get teh "style" style representation
        style_gram = style_grams[layer]
        # teh style loss for one layer, weighted appropriately
        layer_style_loss = stye_weights[layer]*torch.mean(target_gram - style_gram)**2

        # add to the style loss
        style_loss += layer_style_loss / (d * h *w)

    # calculate teh *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # display intermediate images and print the loss
    if ii % show_every == 0:
        print("Total loss: ", total_loss.item())

        # plt.imshow(im_convert(target))
        # plt.show()

# display content and final, target image
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))
plt.show()