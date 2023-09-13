import random

import math
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

thickness_augment_types = ['identity', 'rotation', 'shift',
                           'scale',
                           'all',
                           'trivialaugment']

fundus_augment_types = ['identity', 'rotation', 'shift',
                        'scale',
                        'brightness_contrast_saturation',
                        'all',
                        'trivialaugment']


def identity(x):
    return x


def rotation(x, degree=(-15, 15)):
    p0, p1 = degree
    p = torch.randint(low=p0, high=p1 + 1, size=(1,)).item()

    x = TF.rotate(x, angle=p)
    return x


def shift(x, param=0.1):
    # RandomCrop
    w, h = TF.get_image_size(x)

    h_offset = int(h * param)
    w_offset = int(w * param)

    i = torch.randint(low=-h_offset, high=h_offset + 1, size=(1,)).item()
    j = torch.randint(low=-w_offset, high=w_offset + 1, size=(1,)).item()

    x = TF.crop(x, i, j, h, w)

    return x


# ratio=(0.9, 1.1)
# ratio=(1.0, 1.0)
# 3.0 / 4.0, 4.0 / 3.0
def scale(x, scale=(0.9, 1.1), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    # RandomResizedCrop
    width, height = TF.get_image_size(x)
    area = width * height

    i, j, h, w = None, None, None, None
    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        # print(_, h, w)

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()

            # print(_, i, j, h, w)

            # return i, j, h, w

    if i is None or j is None or h is None or w is None:

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2

    x = TF.resized_crop(x, i, j, h, w, [height, width])

    # print(i, j, h, w)
    # print(TF.to_tensor(x).shape)

    # print(type(x))

    return x


def brightness_contrast_saturation(x, param=(0.75, 1.25)):
    # ColorJitter
    l, h = param
    b = torch.empty((1,)).uniform_(l, h).item()
    c = torch.empty((1,)).uniform_(l, h).item()
    s = torch.empty((1,)).uniform_(l, h).item()

    x = TF.adjust_brightness(x, b)
    x = TF.adjust_contrast(x, c)
    x = TF.adjust_saturation(x, s)

    return x


# def noise(x, var=0.07):
#     # GaussianBlur
#     if isinstance(x, Image.Image):
#         x = TF.to_tensor(x)
#
#     assert isinstance(x, torch.Tensor)
#
#     print(x.var(dim=0).sqrt().mean())
#
#     x += torch.randn(x.size()) * math.sqrt(var)
#
#     x = TF.to_pil_image(x)
#
#     return x

FUNDUS_TRANSFORMS = [identity,
                     rotation,
                     shift,
                     scale,
                     brightness_contrast_saturation]

THICKNESS_TRANSFORMS = [identity,
                        rotation,
                        shift,
                        scale]


def trivial_augment(x, ops):
    # op = random.choices(ALL_TRANSFORMS, k=1)
    op = random.choice(ops)
    # print(op)
    x = op(x)
    return x


def fundus_trivial_augment(x):
    return trivial_augment(x, FUNDUS_TRANSFORMS)


def thickness_trivial_augment(x):
    return trivial_augment(x, THICKNESS_TRANSFORMS)


def load_fundus_transforms(fundus_augment, input_resolution):
    if fundus_augment == 'identity':
        fundus_train_transform = transforms.Compose([transforms.Resize(size=input_resolution),
                                                     transforms.CenterCrop(size=input_resolution),
                                                     transforms.ToTensor()])
    elif fundus_augment == 'rotation':
        fundus_train_transform = transforms.Compose([transforms.Resize(size=input_resolution),
                                                     transforms.CenterCrop(size=input_resolution),
                                                     rotation,
                                                     transforms.ToTensor()])
    elif fundus_augment == 'shift':
        fundus_train_transform = transforms.Compose([transforms.Resize(size=input_resolution),
                                                     transforms.CenterCrop(size=input_resolution),
                                                     shift,
                                                     transforms.ToTensor()])
    elif fundus_augment == 'scale':
        fundus_train_transform = transforms.Compose([transforms.Resize(size=input_resolution),
                                                     transforms.CenterCrop(size=input_resolution),
                                                     scale,
                                                     transforms.ToTensor()])
    elif fundus_augment == 'brightness_contrast_saturation':
        fundus_train_transform = transforms.Compose([transforms.Resize(size=input_resolution),
                                                     transforms.CenterCrop(size=input_resolution),
                                                     brightness_contrast_saturation,
                                                     transforms.ToTensor()])
    elif fundus_augment == 'all':
        fundus_train_transform = transforms.Compose([transforms.Resize(size=input_resolution),
                                                     transforms.CenterCrop(size=input_resolution),
                                                     transforms.Compose(FUNDUS_TRANSFORMS),
                                                     transforms.ToTensor()])
    elif fundus_augment == 'trivialaugment':
        fundus_train_transform = transforms.Compose([transforms.Resize(size=input_resolution),
                                                     transforms.CenterCrop(size=input_resolution),
                                                     fundus_trivial_augment,
                                                     transforms.ToTensor()])
    else:
        raise NotImplementedError

    fundus_test_transform = transforms.Compose([transforms.Resize(size=input_resolution),
                                                transforms.CenterCrop(size=input_resolution),
                                                transforms.ToTensor()])

    return fundus_train_transform, fundus_test_transform


def load_thickness_transforms(thickness_augment, input_resolution):
    # for thickness
    if thickness_augment == 'identity':
        thickness_train_transform = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size=input_resolution)])
    elif thickness_augment == 'rotation':
        thickness_train_transform = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size=input_resolution),
                                                        rotation])
    elif thickness_augment == 'shift':
        thickness_train_transform = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size=input_resolution),
                                                        shift])
    elif thickness_augment == 'scale':
        thickness_train_transform = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size=input_resolution),
                                                        scale])
    elif thickness_augment == 'all':
        thickness_train_transform = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size=input_resolution),
                                                        transforms.Compose(THICKNESS_TRANSFORMS)])
    elif thickness_augment == 'trivialaugment':
        thickness_train_transform = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize(size=input_resolution),
                                                        thickness_trivial_augment])
    else:
        raise NotImplementedError

    thickness_test_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=input_resolution)])

    return thickness_train_transform, thickness_test_transform
