
import cv2
import numpy as np
import torch
import torch.nn.functional

print(torch.__version__)


def cross_entropy(y_pred, y_labe):
    y_labe = torch.nn.functional.one_hot(y_labe, num_classes=10)
    y_pred = torch.clamp(y_pred, min = 1e-9, max = 1.)

    return torch.mean(-torch.sum(y_labe * torch.log(y_pred), dim = 0), dim=0)