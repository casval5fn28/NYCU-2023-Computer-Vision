"""Apply style transfer to an image using a pre-trained image transformation network."""
import pathlib
import argparse
import sys
import torch

from .image_transform_net import ImageTransformNet
from . import utils


def stylize(args):
    """apply style transfer to an image or video using the pre-trained ImageTransformNet."""
    
    # determine if the computation should be performed on GPU, if available.
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load content image/video frames specified by the user.
    if args.video:
        args.output = args.output.with_suffix(".mkv")#mkv
        video = utils.VideoReaderWriter(args.content_file, args.output, 4)
    else:
        content_image = utils.load_image_tensor(
            args.content_file, 1, args.content_shape).to(device)
    print("Loaded content {} ({})".format("video" if args.video else "image", args.content_file))

    with torch.no_grad():
        # load the style transfer model specified by the user.
        print("Loading the style transfer model ({}) ... ".format(args.style_model), end="")
        img_transform = utils.load_model(args.style_model, ImageTransformNet()).to(device)
        print("done")

        # apply style transfer to content image/video frames
        if args.video:
            for i, frames in enumerate(video.frames()):
                stylized_frames = img_transform(frames.to(device)).cpu()
                video.write(stylized_frames)
                if i % 1 == 0:
                    current_frame = max((i + 1) * 4, video.frame_count)
                    print(("Saved frame: {:04d}/{:04d} ({:.0f}%)").format(
                            current_frame, video.frame_count,100. * current_frame / video.frame_count))
        else:
            print("Stylizing image ... ", end="")
            sys.stdout.flush()
            stylized_img = img_transform(content_image).cpu()
            print("done")
            utils.save_image_tensor(args.output, stylized_img)

    print("Saved stylized {} to {}".format("video" if args.video else "image", args.output))


if __name__ == "__main__":
    def main():
        """Handle command line argument parsing."""
        parser = argparse.ArgumentParser(
            description="Apply style transfer to an image using a trained ImageTransformNet")
        parser.add_argument("--content-image", type=pathlib.Path,
                            dest="content_file",
                            required=True, metavar="path/to/content/",
                            help="Path to the content image or video file")
        parser.add_argument("--content-shape", type=int, default=None,
                            metavar="N", nargs=2,
                            help=("Dimensions to resize content image(s) to"))
        parser.add_argument("--model", "--style-model", type=pathlib.Path,
                            required=True, metavar="path/to/model.pth",
                            dest="style_model",
                            help="Path to the pre-trained ImageTransformNet model")
        parser.add_argument("--output", default=pathlib.Path("out.png"),
                            metavar="path/to/output.ext", type=pathlib.Path,
                            help="Filename for the output image or video")
        parser.add_argument("--no-cuda", action="store_true",
                            help="Disable CUDA computation")
        parser.add_argument("--video", action="store_true",
                            help="Process and stylize a video file")
        args = parser.parse_args()
        args.video &= "cv2" in sys.modules  # to ensure OpenCV is available if video processing requested
        stylize(args)
    main()
