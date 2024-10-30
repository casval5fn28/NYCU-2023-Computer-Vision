"""
Self-defined face Loss Network for style transfer.
& Incorporate facial recognition loss using OpenFace and MTCNN for face detection.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import mtcnn
from . import openface
from . import perceptual_loss_net


class FacePerceptualLossNet(perceptual_loss_net.PerceptualLossNet):
    """Calculate perceptual loss, including an additional loss term for facial recognition."""

    def __init__(self, content_weight, style_weights, regularization_weight, face_weight):
        """Initialize the face perceptual loss network.

        Args:
            content_weight (float): Weight for content loss.
            style_weights (dict): Weights for style layers.
            regularization_weight (float): Weight for total variation loss.
            face_weight (float): Weight for face loss.
        """
        super(FacePerceptualLossNet, self).__init__(content_weight, style_weights, regularization_weight)
        self.face_weight = face_weight  # weight for face loss
        self.face_recog_model = openface.net.model
        self.face_recog_model.load_state_dict(torch.load(openface.openface_model_path))
        self.face_recog_model.eval()

    def compute_face_perceptual_loss(self, y, yc):
        """Compute the face loss using MTCNN for face detection and OpenFace for face recognition.

        Args:
            y (torch.Tensor): Stylized image batch.
            yc (torch.Tensor): Content image batch.

        Returns:
            torch.Tensor: Weighted face loss.
        """
        face_loss = 0.0  # initialize face loss to zero

        for i, image in enumerate(yc):
            # convert the content image from a tensor to a PIL Image
            image_array = image.cpu().numpy().clip(0, 255)
            pil_image = Image.fromarray(image_array.transpose(1, 2, 0).astype("uint8"))

            # detect faces by MTCNN
            bounding_boxes, landmarks = mtcnn.detector.detect_faces(pil_image)

            for face_bb in bounding_boxes:
                if face_bb[-1] > 0.9:  # Only consider faces with high confidence
                    b = face_bb[:-1].round().astype("int")
                    b[::2] = np.clip(b[::2], 0, yc.shape[2])
                    b[1::2] = np.clip(b[1::2], 0, yc.shape[3])

                    # extract face patches from content and stylized images
                    yc_face = yc[i, :, b[1]:b[3], b[0]:b[2]]
                    if yc_face.nelement() == 0:  # skip if no pixels are selected
                        continue
                    yc_face = yc_face.unsqueeze(0)  # add batch dimension
                    y_face = y[i, :, b[1]:b[3], b[0]:b[2]].unsqueeze(0)

                    # resize face patches to 96x96
                    yc_face = F.interpolate(yc_face, size=(96, 96), mode="bilinear", align_corners=False)
                    y_face = F.interpolate(y_face, size=(96, 96), mode="bilinear", align_corners=False)

                    # compute mean squared error of the face descriptors
                    face_loss += F.mse_loss(self.face_recog_model(yc_face), self.face_recog_model(y_face))

        return self.face_weight * face_loss  # return weighted face loss

    def compute_perceptual_loss(self, y, yc, ys):
        """Compute the total perceptual loss, including content, style, and face loss.

        Args:
            y (torch.Tensor): Stylized image batch.
            yc (torch.Tensor): Content image batch.
            ys (torch.Tensor): Style image batch.

        Returns:
            torch.Tensor: Total perceptual loss.
        """
        return super(FacePerceptualLossNet, self).compute_perceptual_loss(y, yc, ys) + self.compute_face_perceptual_loss(y, yc)