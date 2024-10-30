"""
Compute style and content loss by a VGG-16 model pretrained on ImageNet
& Employ perceptual loss to evaluate style and content differences.
"""
import torch
import torchvision
from collections import namedtuple

# define a named tuple to store intermediate activation values
LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])

def gram_matrix(x):
    """
    Generate the Gram matrix for the input tensor, for capturing style information.
    """
    b, c, h, w = x.shape
    phi = x.view(b, c, h * w)
    return phi.bmm(phi.transpose(1, 2)) / (c * h * w)

class PerceptualLossNet(torch.nn.Module):
    """
    Neural network module for computing perceptual losses used in style transfer.
    A pretrained VGG-16 model for extracting intermediate layer activations.
    """

    def __init__(self, content_weight, style_weights, regularization_weight):
        """
        Initialize the network with specified weights and setup VGG-16.

        Args:
            content_weight (float): Weight for content loss.
            style_weights (list of float): Weights for style loss at each layer.
            regularization_weight (float): Weight for total variation regularization.
        """
        super(PerceptualLossNet, self).__init__()
        # load the pretrained VGG-16 model and use its features for perceptual loss
        vgg_model = torchvision.models.vgg16(pretrained=True)
        self.vgg_layers = vgg_model.features
        # mapping from layer index to activation layer
        self.layer_name_mapping = {"3": "relu1_2", "8": "relu2_2", "15": "relu3_3", "22": "relu4_3"}
        # disable gradients for all the VGG layers
        for param in self.parameters():
            param.requires_grad = False
        # parameter for precomputed Gram matrix of style image
        self.ys_grams = None
        # save given weights for different loss components
        self.content_weight = content_weight
        self.style_weights = style_weights
        self.regularization_weight = regularization_weight
        # mean squared error loss function for computing perceptual losses
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x):
        """
        Perform a forward pass through the VGG-16 network to compute perceptual loss.

        Args:
            x (Tensor): Input tensor representing an image or a batch of images.
        
        Returns:
            LossOutput: Named tuple containing activations from selected layers.
        """
        output = {}
        # pass the input through each layer and save activations of interest
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

    @staticmethod
    def normalize_batch(batch):
        """
        Normalize a batch of images using ImageNet mean and standard deviation.

        Args:
            batch (Tensor): Batch of images to normalize.
        
        Returns:
            Tensor: Normalized batch of images.
        """
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        batch = batch / 255.0
        return (batch - mean) / std

    def compute_perceptual_loss(self, y, yc, ys):
        """
        Compute perceptual loss between stylized and target images.

        Args:
            y (Tensor): Stylized image tensor.
            yc (Tensor): Content image tensor.
            ys (Tensor): Style image tensor.
        
        Returns:
            Tensor: Total perceptual loss.
        """
        # precompute Gram matrices of style image features if not done
        if self.ys_grams == None:
            ys_features = self(self.normalize_batch(ys))
            self.ys_grams = [gram_matrix(feature) for feature in ys_features]

        # normalize and extract features from content image
        yc = self.normalize_batch(yc)
        yc_features = self(yc)
        # normalize and extract features from stylized image
        y = self.normalize_batch(y)
        y_features = self(y)

        # compute content loss as MSE between content and stylized image features
        content_loss = self.content_weight * self.mse_loss(y_features.relu2_2, yc_features.relu2_2)

        # compute style loss using the Gram matrices and the given weights
        style_loss = 0.0
        for y_feature, ys_gram, style_weight in zip(
                y_features, self.ys_grams, self.style_weights):
            style_loss += style_weight * self.mse_loss(
                gram_matrix(y_feature), ys_gram[:yc.shape[0]])

        # compute total variation regularization to encourage smoothness in the stylized image
        total_variation = self.regularization_weight * (
            torch.sum(torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])) +
            torch.sum(torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])))

        # total loss = sum of content, style, and total variation losses
        return content_loss + style_loss + total_variation
