from collections import namedtuple
import cv2
from torch import nn
from torch.utils import model_zoo
from segmentation_models_pytorch import Unet
import albumentations as albu
import torch
from pathlib import Path
from u2net import U2NET
from skimage import io
from typing import Any, Dict, Tuple, Union
from utils import *
import numpy as np
from PIL import Image, ImageChops

model = namedtuple("model", ["path", "model"])
models = {
    "Unet_cloth": model(
        path=Path('./pretrained_models/unet_cloth_seg.pth'),
        model=Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None),
    ),
    "Unet_human": model(
        path=Path('./pretrained_models/u2net_human_seg.pth'),
        model=U2NET(3, 1),
    )
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    model_dir = models[model_name].path

    if model_name == "Unet_cloth":
        state_dict = torch.load(model_dir, map_location="cpu")["state_dict"]
        state_dict = rename_layers(state_dict, {"model.": ""})
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    return model


def human_binarization(image_file):
    net = create_model("Unet_human")
    net.eval()

    image = io.imread(image_file)
    resized_image = temp_resize(384, image)
    tmp = (resized_image*255).astype(np.uint8)
    resized_orig_img = Image.fromarray(tmp)

    image_tensor = image_to_tensor(resized_image)

    d1, _, _, _, _, _, _ = net(image_tensor)
    prediction = d1[:, 0, :, :]
    max_p = torch.max(prediction)
    min_p = torch.min(prediction)
    normalized_pred = (prediction - min_p) / (max_p - prediction)
    pred = normalized_pred.squeeze()
    pred_np = pred.cpu().data.numpy()

    masked_image = Image.fromarray(pred_np*255).convert('RGB')
    del d1

    return resized_orig_img, masked_image


def trim(mask_im):
    background = Image.new(mask_im.mode, mask_im.size, mask_im.getpixel((0, 0)))
    diff = ImageChops.difference(mask_im, background)
    diff = ImageChops.add(diff, diff, 12, -17)
    bbox = diff.getbbox()
    if bbox:
        ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
        if ratio > 0.75:
            x1 = bbox[0] - 20
            x2 = bbox[2] + 20
            w = x2 - x1
            c = (bbox[3] - bbox[1]) / 2 + bbox[1]
            h = 4 / 3 * w
            y1 = c - (h / 2)
            y2 = c + (h / 2)
            bbox = (x1, y1, x2, y2)

        elif ratio < 0.75:
            y1 = bbox[1] - 20
            y2 = bbox[3] + 20
            h = y2 - y1
            c = (bbox[2] - bbox[0]) / 2 + bbox[0]
            w = 3 / 4 * h
            x1 = c - (w / 2)
            x2 = c + (w / 2)
            bbox = (x1, y1, x2, y2)

        return bbox

    else:
        print('Failure!')
        return None


def crop_and_resize(images, bbox, target_size):
    crop_images = [image.crop(bbox) for image in images]
    return (img.resize(target_size) for img in crop_images)


def resizing(image, masked_image):
    # target_size 설정
    target_size = (192, 256)

    bbox = trim(masked_image)
    resized_img, resized_mask = crop_and_resize(
        [image, masked_image], bbox, target_size
    )

    return resized_img, resized_mask


