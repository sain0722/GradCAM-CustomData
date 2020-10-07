import torch
from torchvision import transforms
import numpy as np
import cv2


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    return cam


def make_transforms():

    ## Define data augmentation and transforms
    chosen_transforms = {'data/train': transforms.Compose([
        transforms.RandomResizedCrop(size=227),
        # transforms.RandomRotation(degrees=10),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        # transforms.Normalize(mean_nums, std_nums)
    ]), 'data/val': transforms.Compose([
        transforms.Resize(227),
        # transforms.CenterCrop(227),
        transforms.ToTensor(),
        # transforms.Normalize(mean_nums, std_nums)
    ]), 'data/test': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean_nums, std_nums)
    ])
    }

    return chosen_transforms