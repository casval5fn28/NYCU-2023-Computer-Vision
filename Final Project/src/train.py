"""Train the image transformation network model."""
import pathlib
import argparse

import torch
from torchvision import transforms, datasets

from .image_transform_net import ImageTransformNet
from .perceptual_loss_net import PerceptualLossNet
from . import utils


def train(args):
    """Train the image transformation network model."""
    # determine whether to utilize GPU for computations, if available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 10, "pin_memory": True} if use_cuda else {}

    # load content images dataset
    content_loader = utils.load_content_dataset(args.content_images, args.content_size, args.batch_size)

    # load style image for transformation
    print(f"Loading style image {args.style_image}\n")
    style_images = utils.load_image_tensor(args.style_image, args.batch_size, args.style_size).to(device)

    # initialize the image transformation network
    print("Initializing image transformation network... ", end="")
    image_transform_net = ImageTransformNet().to(device)
    optimizer = torch.optim.Adam(image_transform_net.parameters(), lr=args.lr)
    print("done")

    # check if there are faces in image, add face loss if there are
    print("Initializing loss network... ", end="")
    if args.face:
        from .face_perceptual_loss_net import FacePerceptualLossNet
        loss_net = FacePerceptualLossNet(
            args.content_weight, args.style_weights,
            args.regularization_weight, args.face_weight).to(device)
    else:
        loss_net = PerceptualLossNet(args.content_weight, args.style_weights,args.regularization_weight).to(device)
    print("done\n")

    # load checkpoint if provided
    start_epoch = utils.load_checkpoint(
        args.checkpoint, image_transform_net, optimizer, args.lr) if args.checkpoint else 1

    # start training the image transformation network
    for epoch in range(start_epoch, args.epochs + 1):
        for batch_idx, (content_images, _) in enumerate(content_loader):
            optimizer.zero_grad()

            # apply transformation to content images
            content_images = content_images.to(device)
            transformed_images = image_transform_net(content_images)

            # compute perceptual loss
            loss = loss_net.compute_perceptual_loss(transformed_images, content_images, style_images)

            # backpropagate then optimize
            loss.backward()
            optimizer.step()

            # log training progress
            if batch_idx % args.log_interval == 0:
                n_examples_seen = (batch_idx + 1) * args.batch_size
                progress = 100. * n_examples_seen / len(content_loader.dataset)
                print(f"Train Epoch: {epoch:02d} [{n_examples_seen:06d}/{len(content_loader.dataset):06d} ({progress:.0f}%)]\tLoss: {loss.data.item():12.2f}")

            # save checkpoint periodically, in case something interrupt the process
            if batch_idx % args.checkpoint_interval == 0:
                epoch_batch_str = f"_{epoch:02d}_{batch_idx:06d}.pth"
                utils.save_checkpoint(
                    args.output_dir / f"checkpoint{epoch_batch_str}",
                    epoch, image_transform_net, optimizer, device)
                utils.save_model(args.output_dir / f"model{epoch_batch_str}",image_transform_net, device)

    # save final model and checkpoint for later evaluation
    utils.save_model(args.output_dir / "final_model.pth", image_transform_net, device)
    utils.save_checkpoint(args.output_dir / "final_checkpoint.pth", epoch,image_transform_net, optimizer, device)


if __name__ == "__main__":
    def main():
        """Parse training parameters."""
        parser = argparse.ArgumentParser(
            description="Train a CNN for Neural Style Transfer.")
        parser.add_argument("--content-images", type=pathlib.Path,
                            required=True, metavar="path/to/content/",
                            help="Directory containing the training data")
        parser.add_argument("--content-size", type=int, default=256,
                            metavar="N",
                            help="Target size for rescaling content images (default: 256 x 256)")
        parser.add_argument("--content-weight", type=float, default=1e5,
                            help="Weights the content loss function (default: 1e5)")
        parser.add_argument("--style-image", type=pathlib.Path,
                            required=True, metavar="path/to/style.image",
                            help="Path to the style image for training")
        parser.add_argument("--style-size", type=int, default=None,
                            metavar="N", nargs=2,
                            help="Target size for rescaling the style image (default: no scaling)")
        parser.add_argument("--style-weights", type=float, default=[1e10, 1e10, 1e10, 1e10], nargs=4,
                            help="Weights for style loss function (default: [1e10, 1e10, 1e10, 1e10])")
        parser.add_argument("--regularization-weight", type=float, default=1e-6,
                            help="Weights for total variation regularization (default: 1e-6)")
        parser.add_argument("--output-dir", default=pathlib.Path("."),
                            metavar="path/to/output/", type=pathlib.Path,
                            help="Directory to save model and checkpoint files")
        parser.add_argument("--batch-size", type=int, default=4, metavar="N",
                            help="Batch size for training (default: 4)")
        parser.add_argument("--epochs", type=int, default=2, metavar="N",
                            help="Number of epochs to train (default: 2)")
        parser.add_argument("--lr", type=float, default=1e-3, metavar="LR",
                            help="Learning rate (default: 1e-3)")
        parser.add_argument("--log-interval", type=int, default=1,
                            metavar="N", help="Number of batches between log outputs")
        parser.add_argument("--checkpoint", default=None, type=pathlib.Path,
                            metavar="path/to/checkpoint.pth",
                            help="Checkpoint file to resume training from")
        parser.add_argument("--checkpoint-interval", default=5000, type=int,
                            metavar="N", help="Frequency of checkpoint saves (in batches)")
        parser.add_argument("--no-cuda", action="store_true",
                            help="Disable CUDA training")
        parser.add_argument("--face", action="store_true",
                            help="Apply facial preservation ")
        parser.add_argument("--face-weight", type=float, default=1e7,
                            help="Weights for face loss (default: 1e7)")
        args = parser.parse_args()
        print(f"{args}\n")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        train(args)
    main()
