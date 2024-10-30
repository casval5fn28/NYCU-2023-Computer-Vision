"""Image Transformation Network for style transfer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class ResidualBlock(nn.Module):
    """Defines a residual block used in the style transfer network."""

    def __init__(self, nchannels):
        """initialize layers of the residual block."""
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nchannels, nchannels, kernel_size=3)
        self.conv2 = nn.Conv2d(nchannels, nchannels, kernel_size=3)
        self.norm_conv1 = nn.InstanceNorm2d(nchannels, affine=True)
        self.norm_conv2 = nn.InstanceNorm2d(nchannels, affine=True)
        self.nonlinearity = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        """Pass input through the residual block."""
        residual = x[:, :, 2:-2, 2:-2]  # crop input to match output size
        out = self.nonlinearity(self.norm_conv1(self.conv1(x)))
        out = self.norm_conv2(self.conv2(out))
        return out + residual  # add residual

class UpsampleConv2d(nn.Module):
    """Performs upsampling followed by convolution to avoid checkerboard artifacts."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upsample):
        """Initialize the upsampling and convolution layers."""
        super(UpsampleConv2d, self).__init__()
        self.upsample = upsample
        self.padding = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        """Upsample the input and then apply convolution."""
        x = F.interpolate(x, mode="nearest", scale_factor=self.upsample)
        return self.conv(self.padding(x))

class ImageTransformNet(nn.Module):
    """
    Image Transformation Network architecture for style transfer.
    Architecture from: https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf
    """

    def __init__(self):
        """Construct the image transformation network."""
        super(ImageTransformNet, self).__init__()
        self.reflection_padding = nn.ReflectionPad2d(40)  # to preserve image dimensions

        # downsampling layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.pad_conv1 = nn.ReflectionPad2d(padding=4)
        self.pad_conv2 = nn.ReflectionPad2d(padding=1)
        self.pad_conv3 = nn.ReflectionPad2d(padding=1)
        self.norm_conv1 = nn.InstanceNorm2d(32, affine=True)
        self.norm_conv2 = nn.InstanceNorm2d(64, affine=True)
        self.norm_conv3 = nn.InstanceNorm2d(128, affine=True)

        # residual blocks
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(128)
        self.res_block3 = ResidualBlock(128)
        self.res_block4 = ResidualBlock(128)
        self.res_block5 = ResidualBlock(128)

        # upsampling layers
        self.upsample_conv1 = UpsampleConv2d(128, 64, kernel_size=3, stride=1, padding=1, upsample=2)
        self.upsample_conv2 = UpsampleConv2d(64, 32, kernel_size=3, stride=1, padding=1, upsample=2)
        self.upsample_conv3 = UpsampleConv2d(32, 3, kernel_size=9, stride=1, padding=4, upsample=1)
        self.norm_upsample_conv1 = nn.InstanceNorm2d(64, affine=True)
        self.norm_upsample_conv2 = nn.InstanceNorm2d(32, affine=True)
        self.norm_upsample_conv3 = nn.InstanceNorm2d(3, affine=True)

        self.nonlinearity = nn.ReLU()  # ReLU activation function
        self.tanh = nn.Tanh()  # Tanh activation for output normalization
        self.output_nonlinearity = lambda x: (self.tanh(x) + 1) / 2 * 255  # normalize to [0, 255]

    def forward(self, x):
        """process input image through network layers."""
        x = self.reflection_padding(x)
        x = self.nonlinearity(self.norm_conv1(self.conv1(self.pad_conv1(x))))
        x = self.nonlinearity(self.norm_conv2(self.conv2(self.pad_conv2(x))))
        x = self.nonlinearity(self.norm_conv3(self.conv3(self.pad_conv3(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.nonlinearity(self.norm_upsample_conv1(self.upsample_conv1(x)))
        x = self.nonlinearity(self.norm_upsample_conv2(self.upsample_conv2(x)))
        x = self.norm_upsample_conv3(self.upsample_conv3(x))
        return self.output_nonlinearity(x)

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    net = ImageTransformNet()
    x = net(x)
    print(x.shape)
    print(numpy.max(x.detach().numpy()))