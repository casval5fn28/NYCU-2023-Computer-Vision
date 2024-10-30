import pathlib
import torch
# from torch.autograd import Variable
from torchvision import transforms, datasets
from PIL import Image

from . import net

containing_dir = pathlib.Path(__file__).resolve().parent
face_patch_dir = containing_dir.parents[1] / "images/faces/patches"

model = net.model
model.load_state_dict(torch.load(containing_dir / "net.pth"))
model.eval()


def load_image_tensor(filename, batch_size, image_shape=None):
    """Load an image for torch."""
    image = Image.open(filename)
    if image_shape:  # Downsample the image
        image = image.resize(image_shape, Image.ANTIALIAS)
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)])
    # Repeat the image so it matches the batch size for loss computations
    return image_transform(image)[:3].repeat(batch_size, 1, 1, 1)


original = load_image_tensor(face_patch_dir / "original.png", 1)
bw = load_image_tensor(face_patch_dir / "bw.png", 1)
stylized_face_false = load_image_tensor(
    face_patch_dir / "stylized-face=false.png", 1)
stylized_face_true_1 = load_image_tensor(
    face_patch_dir / "stylized-face=true-1.png", 1)
stylized_face_true_3 = load_image_tensor(
    face_patch_dir / "stylized-face=true-3.png", 1)
other = load_image_tensor(face_patch_dir / "other.png", 1)

# CPU
f1 = model(original)
f2 = model(bw)
f3 = model(stylized_face_false)
f4 = model(stylized_face_true_1)
f5 = model(stylized_face_true_3)
f6 = model(other)

print("MSE(original, original)                                = {:.4f}".format(
    torch.nn.functional.mse_loss(f1, f1).data.item()))
print("MSE(original, black and white)                         = {:.4f}".format(
    torch.nn.functional.mse_loss(f1, f2).data.item()))
print("MSE(original, stylized without facial preservation)    = {:.4f}".format(
    torch.nn.functional.mse_loss(f1, f3).data.item()))
print("MSE(original, stylized with heavy facial preservation) = {:.4f}".format(
    torch.nn.functional.mse_loss(f1, f4).data.item()))
print("MSE(original, stylized with light facial preservation) = {:.4f}".format(
    torch.nn.functional.mse_loss(f1, f5).data.item()))
print("MSE(original, other)                                   = {:.4f}".format(
    torch.nn.functional.mse_loss(f1, f6).data.item()))


breakpoint()
# print('CPU done')

# GPU
# model.cuda()
# input = input.cuda()
# output = model(Variable(input))
# print('GPU done')
