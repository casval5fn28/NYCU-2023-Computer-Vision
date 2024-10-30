from PIL import Image
from src import detector, box_utils, visualization_utils
import torch
from torchvision import transforms

image = Image.open("../../images/results/mosaic-face/bridge-face=True-3.png")
bounding_boxes, landmarks = detector.detect_faces(image)
print(bounding_boxes)

image_transform = transforms.ToTensor()
image_tensor = image_transform(image).unsqueeze(0)
for i, b in enumerate(bounding_boxes[:, :-1].round().astype("int")):
    x = image_tensor[:, :, b[1]:b[3], b[0]:b[2]].clone()
    x = torch.nn.functional.interpolate(
        x, size=(96, 96), mode="bilinear", align_corners=True)
    image_array = x.numpy().squeeze(0) * 255
    image = Image.fromarray(image_array.transpose(1, 2, 0).astype("uint8"))
    image.save(f"../../images/faces/test-face-{i}-p={bounding_boxes[i, 4]}.png", subsampling=0, quality=100)

# img_boxes = box_utils.get_image_boxes(bounding_boxes, image, size=96)
# img_copy = visualization_utils.show_bboxes(image, bounding_boxes, landmarks)
# img_copy.save("tmp.png")
