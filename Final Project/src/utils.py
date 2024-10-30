"""Utilities for style transfer."""
import pathlib
import numpy
import torch
from torchvision import transforms, datasets
from PIL import Image

image_net_mean = [0.485, 0.456, 0.406]
image_net_std = [0.229, 0.224, 0.225]


def load_checkpoint(filename, model, optimizer, lr):
    """
    Load model and optimizer state from a checkpoint file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        lr (float): The learning rate to set in the optimizer.

    Returns:
        int: The epoch to start training from.
    """
    print("Loading checkpoint {}".format(filename))
    checkpoint = torch.load(str(filename), map_location="cpu")
    start_epoch = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("Continuing training from checkpoint "
              "{} at epoch {:d}\n".format(filename, start_epoch))
    return start_epoch


def save_checkpoint(filename, epoch, model, optimizer, device):
    """
    Save the current state of the model and optimizer to a checkpoint file.

    Args:
        filename (str): Path to save the checkpoint file.
        epoch (int): The current epoch of training.
        model (torch.nn.Module): The model whose state is to be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state is to be saved.
    """
    model.eval().cpu()
    checkpoint = {"epoch": epoch, "model": model.state_dict(),
                  "optimizer": optimizer.state_dict()}
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(filename))
    print(("Saved checkpoint to {0}. You can run "
           "`python train.py --checkpoint {0}` to continue training from "
           "this state.").format(filename))
    model.to(device).train()


def load_model(filename, model):
    """
    Load model parameters from a file.

    Args:
        filename (str): Path to the file containing model parameters.
        model (torch.nn.Module): The model to load parameters into.

    Returns:
        torch.nn.Module: The model with loaded parameters.
    """
    model_params = torch.load(str(filename))
    model.load_state_dict(model_params)
    return model


def save_model(filename, model, device):
    """
    Save model parameters to a file.
    """
    model.eval().cpu()
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(filename))
    print(("Saved model to {0}. You can run "
           "`python stylize.py --model {0}` to stylize an image").format(filename))
    model.to(device).train()


def load_content_dataset(content_path, content_size, batch_size):
    """
    Load content images from a directory for training.

    Args:
        content_path (str): Path to the directory containing content images.
        content_size (int): The size to resize and crop the images to.
        batch_size (int): The batch size for loading the data.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the content images.
    """
    print("Creating dataset for content images in {}".format(content_path))
    content_transform = transforms.Compose([
                            transforms.Resize(content_size),
                            transforms.CenterCrop(content_size),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: x * 255)])
    content_data = datasets.ImageFolder(str(content_path), content_transform)
    return torch.utils.data.DataLoader(content_data, batch_size=batch_size)


def load_image_tensor(filename, batch_size, image_shape=None):
    """
    Load an image file into a tensor.

    Args:
        filename (str): Path to the image file.
        batch_size (int): The batch size to repeat the image tensor.
        image_shape (tuple, optional): The shape to resize the image to (width, height).

    Returns:
        torch.Tensor: The loaded image tensor.
    """
    image = Image.open(filename)
    # downsample the image
    if image_shape:  
        image = image.resize(image_shape, Image.ANTIALIAS)
    image_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x * 255)])
    # repeat the image so it matches the batch size for loss computations
    return image_transform(image)[:3].repeat(batch_size, 1, 1, 1)


def save_image_tensor(filename, image_tensor):
    """
    Save a tensor as an image file.
    """
    image_array = image_tensor.clone().squeeze(0).numpy().clip(0, 255)
    image = Image.fromarray(image_array.transpose(1, 2, 0).astype("uint8"))
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    image.save(filename, subsampling=0, quality=100)


# try importing OpenCV for video writer
try:
    import cv2 as cv
except ImportError as e:
    print("OpenCV not installed or not found.")
else:
    class VideoReaderWriter:
        """
        Class to read/write a video file with the same properties.
        """

        def __init__(self, in_file, out_file, batch_size=16):
            """
            Initialize the video reader and writer.

            Args:
                in_file (str): Path to the input video file.
                out_file (str): Path to the output video file.
                batch_size (int, optional): Number of frames to process in a batch.
            """
            self.in_video = cv.VideoCapture(str(in_file))
            self.frame_count = int(self.in_video.get(cv.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.in_video.get(cv.CAP_PROP_FPS))
            self.frame_width = int(self.in_video.get(cv.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.in_video.get(cv.CAP_PROP_FRAME_HEIGHT))

            self.batch_size = batch_size
            self.buf = numpy.empty((self.batch_size, self.frame_height, self.frame_width, 3),dtype='uint8')

            fourcc = cv.VideoWriter_fourcc(*'MPEG')
            pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            self.out_video = cv.VideoWriter(str(out_file), fourcc, self.fps, (self.frame_width, self.frame_height))

        def frames(self):
            """Generator to yield batches of frames as a tensor, from the input video."""
            fc = 0
            ret = True

            while fc < self.frame_count and ret:
                nFrames = min(self.batch_size, self.frame_count - fc)
                for i in range(nFrames):
                    ret, buffer_i = self.in_video.read()
                    if buffer_i is not None:
                        self.buf[i] = buffer_i
                    else:
                        nFrames = i
                        ret = False
                        break
                    fc += 1
                yield torch.tensor(self.buf[:nFrames].transpose(0, 3, 1, 2)).float()

            self.close()

        def write(self, frames):
            """
            Write a batch of frames to the output video.

            Args:
                frames (torch.Tensor): A batch of frames as a tensor.
            """
            for frame in frames:
                self.out_video.write(frame.numpy().transpose(1, 2, 0).astype("uint8"))

        def close(self):
            """Close the input and output video files."""
            self.in_video.release()
            self.out_video.release()