from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and name.startswith("resnet")
                     and callable(models.__dict__[name]))


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self):
        raise NotImplementedError

    def loss(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError


def load_resnet_backbone(arch: str, in_channels: int, pretrained: bool) -> nn.Module:
    assert arch in model_names
    assert arch.startswith('resnet'), 'Only support resnet familiy'

    backbone = models.__dict__[arch](pretrained=pretrained)

    # for in_channels inconsistent
    if in_channels != 3:
        backbone.conv1 = nn.Conv2d(in_channels,
                                   backbone.conv1.out_channels,
                                   kernel_size=backbone.conv1.kernel_size,
                                   stride=backbone.conv1.stride,
                                   padding=backbone.conv1.padding,
                                   bias=backbone.conv1.bias)

    backbone = nn.Sequential(OrderedDict([
        ('conv1', backbone.conv1),
        ('bn1', backbone.bn1),
        ('relu', backbone.relu),
        ('maxpool', backbone.maxpool),

        ('layer1', backbone.layer1),
        ('layer2', backbone.layer2),
        ('layer3', backbone.layer3),
        ('layer4', backbone.layer4),
    ]))

    return backbone


class Regressor(BaseModel):
    def __init__(self, arch: str, in_channels: int, pretrained: bool, num_outputs: int):
        super(Regressor, self).__init__()

        self.backbone = load_resnet_backbone(arch, in_channels, pretrained)
        self.regressor = nn.Linear(in_features=512, out_features=num_outputs)

        self.arch = arch
        self.in_channels = in_channels
        self.num_outputs = num_outputs

    def forward(self, x):
        # if x is fundus, x.shape: [m, 3, 384, 384]
        x = self.backbone(x)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [m, 512, 1, 1]
        x = torch.flatten(x, 1)  # [m, 512]
        x = self.regressor(x)
        return x

    def loss(self, pred, target):
        return F.mse_loss(pred, target)

    def score(self, pred, target):
        rmse = F.mse_loss(pred, target).sqrt()
        mae = (pred - target).abs().mean()
        return rmse, mae


def get_heads(dims: list[int], pools: list[bool], num_classes: int):
    heads = []
    for dim, pool in zip(dims, pools):
        if pool:
            heads.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(12),
                nn.Conv2d(in_channels=dim, out_channels=num_classes, kernel_size=3),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
            ))
        else:
            heads.append(nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=num_classes, kernel_size=3),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
            ))

    assert len(heads) == len(dims)
    return heads


def load_resnet_layers(arch, in_channels, pretrained):
    backbone = models.__dict__[arch](pretrained=pretrained)
    # for in_channels inconsistent
    if in_channels != 3:
        backbone.conv1 = nn.Conv2d(in_channels,
                                   backbone.conv1.out_channels,
                                   kernel_size=backbone.conv1.kernel_size,
                                   stride=backbone.conv1.stride,
                                   padding=backbone.conv1.padding,
                                   bias=backbone.conv1.bias)

    layer0 = nn.Sequential(OrderedDict([
        ('conv1', backbone.conv1),
        ('bn1', backbone.bn1),
        ('relu', backbone.relu),
        ('maxpool', backbone.maxpool)]))

    layer1 = backbone.layer1
    layer2 = backbone.layer2
    layer3 = backbone.layer3
    layer4 = backbone.layer4

    return layer0, layer1, layer2, layer3, layer4


class Model(nn.Module):
    def __init__(self, arch, in_channels, num_classes, mask, pretrained):
        super(Model, self).__init__()

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = load_resnet_layers(arch, in_channels,
                                                                                             pretrained)

        self.head0, self.head1, self.head2, self.head3, self.head4 = get_heads(dims=[64, 64, 128, 256, 512],
                                                                               pools=[True, True, True, True, False],
                                                                               num_classes=num_classes - 1)
        # shared OR bias
        self.bias = nn.Parameter(torch.zeros([num_classes - 1, 10, 10]).float())

        self.arch = arch
        self.in_channels = in_channels

        self.num_classes = num_classes
        self.mask = mask

    def forward(self, x):
        # x: [m, 3, 384, 384]
        x0 = self.layer0(x)  # [m, 64, 96, 96]
        x1 = self.layer1(x0)  # [m, 64, 96, 96]
        x2 = self.layer2(x1)  # [m, 128, 48, 48]
        x3 = self.layer3(x2)  # [m, 256, 24, 24]
        x4 = self.layer4(x3)  # [m, 512, 12, 12]

        x = self.head0(x0) + self.head1(x1) + self.head2(x2) + self.head3(x3) + self.head4(x4)

        # add OR bias
        x = x + self.bias  # [m, num_classes-1, 10, 10]

        return x

    def features(self, x):
        x0 = self.layer0(x)  # [m, 64, 96, 96]
        x1 = self.layer1(x0)  # [m, 64, 96, 96]
        x2 = self.layer2(x1)  # [m, 128, 48, 48]
        x3 = self.layer3(x2)  # [m, 256, 24, 24]
        x4 = self.layer4(x3)  # [m, 512, 12, 12]
        return x4

    def loss(self, output, target):
        """

        :param output: logits: [m, num_classes, 10, 10]
        :param target: target: [m, num_classes, 10, 10]
        :return:
        """
        output = output[:, :, self.mask]  # [m, num_classes-1, 52]
        log_p = F.logsigmoid(output)
        target = target[:, :, self.mask]  # [m, num_classes-1, 52]
        loss = (- (log_p * target + (log_p - output) * (1 - target))).sum(dim=1).mean()
        return loss

    def score(self, output, target, return_mse=False):
        p = torch.sigmoid(output)  # [m, num_classes-1, 10, 10]
        pred = p > 0.5  # [m, num_classes-1, 10, 10]

        pred = pred.sum(dim=1)  # [m, 10, 10]
        target = target.sum(dim=1)  # [m, 10, 10]

        mse = F.mse_loss(pred[:, self.mask].float(), target[:, self.mask].float())
        rmse = mse.sqrt()
        mae = (pred[:, self.mask].float() - target[:, self.mask].float()).abs().mean()
        if return_mse:
            return mse, rmse, mae
        else:
            return rmse, mae


class ModelAux(Model):
    def __init__(self, arch, in_channels, num_classes, mask, pretrained, num_aux_classes, lam):
        super(ModelAux, self).__init__(arch, in_channels, num_classes, mask, pretrained)

        self.aux_head = nn.Linear(in_features=512, out_features=num_aux_classes - 1)
        self.aux_bias = nn.Parameter(torch.zeros([num_aux_classes - 1]).float())
        self.num_aux_classes = num_aux_classes
        self.lam = lam

    def forward(self, x, return_aux=False):
        # x: [m, 3, 384, 384]
        x0 = self.layer0(x)  # [m, 64, 96, 96]
        x1 = self.layer1(x0)  # [m, 64, 96, 96]
        x2 = self.layer2(x1)  # [m, 128, 48, 48]
        x3 = self.layer3(x2)  # [m, 256, 24, 24]
        x4 = self.layer4(x3)  # [m, 512, 12, 12]

        x = self.head0(x0) + self.head1(x1) + self.head2(x2) + self.head3(x3) + self.head4(x4)

        # add or bias
        x = x + self.bias  # [m, num_classes-1, 10, 10]

        if return_aux:
            aux_x = F.adaptive_avg_pool2d(x4, (1, 1))  # [m, 512, 1, 1]
            aux_x = torch.flatten(aux_x, 1)  # [m, 512]
            aux_x = self.aux_head(aux_x)  # [m, num_aux_classes]
            # add or bias
            aux_x = aux_x + self.aux_bias  # [m, num_aux_classes]
            return x, aux_x
        else:
            return x

    def loss(self, output, target, aux_output=None, aux_target=None):
        """

        :param output: [m, num_classes-1, 10, 10]
        :param target: [m, num_classes-1, 10, 10]
        :param aux_output: [m, num_aux_classes-1]
        :param aux_target: [m, num_aux_classes-1]
        :return:
        """
        output = output[:, :, self.mask]
        log_p = F.logsigmoid(output)
        # mask
        target = target[:, :, self.mask]  # [m, num_classes-1, 52]

        # BCE for primary
        main_loss = (- (log_p * target + (log_p - output) * (1 - target))).sum(dim=1).mean()

        if aux_output is not None and aux_target is not None:
            aux_log_p = F.logsigmoid(aux_output)  # [m, num_classes-1]

            # BCE for auxiliary
            # aux_loss = F.cross_entropy(aux_output, aux_target)
            aux_loss = (- (aux_log_p * aux_target + (aux_log_p - aux_output) * (1 - aux_target))).sum(dim=1).mean()

            total_loss = main_loss + self.lam * aux_loss
            return total_loss, main_loss, aux_loss
        else:
            return main_loss

    def get_shared_params(self, include_bn=True):
        if include_bn:
            params = list(self.layer0.parameters()) + \
                     list(self.layer1.parameters()) + \
                     list(self.layer2.parameters()) + \
                     list(self.layer3.parameters()) + \
                     list(self.layer4.parameters())
        else:
            params = []
            for m in self.layer0.modules():
                if not isinstance(m, nn.BatchNorm2d):
                    params += list(m.parameters())

            for m in self.layer1.modules():
                if not isinstance(m, nn.BatchNorm2d):
                    params += list(m.parameters())

            for m in self.layer2.modules():
                if not isinstance(m, nn.BatchNorm2d):
                    params += list(m.parameters())

            for m in self.layer3.modules():
                if not isinstance(m, nn.BatchNorm2d):
                    params += list(m.parameters())

            for m in self.layer4.modules():
                if not isinstance(m, nn.BatchNorm2d):
                    params += list(m.parameters())

        return params

    @staticmethod
    def get_grad_cos_sim(grad0, grad1):
        grad0 = torch.concat([g.reshape(-1) for g in grad0])
        grad1 = torch.concat([g.reshape(-1) for g in grad1])

        # print(grad0.shape, grad1.shape)

        sim = F.cosine_similarity(grad0, grad1, dim=0)

        return sim.item()


def filter_aux_grad(main_grad: torch.Tensor, aux_grad: torch.Tensor, mode: str) -> torch.Tensor:
    """
    :param main_grad:
    :param aux_grad:
    :param mode:
    :return: filtered aux_grad
    """
    """To check whether aux_grad is related to main_grad or not
    https://github.com/vivien000/auxiliary-learning/blob/master/auxiliary_real.ipynb

      - Unweighted cosine: cf. https://arxiv.org/abs/1812.02224
      - Weighted cosine: cf. https://arxiv.org/abs/1812.02224
      - Projection: cf. https://github.com/vivien000/auxiliary-learning
      - Parameter-wise: same as projection but at the level of each parameter
    """

    assert main_grad.shape == aux_grad.shape

    # if len(main_grad.shape) != 1:
    #     main_g = main_grad.reshape(-1)
    #     main_g_norm = torch.norm(main_g, p='fro')
    # else:
    #     main_g_norm = torch.norm(main_grad, p='fro')
    #
    # if len(aux_grad.shape) != 1:
    #     aux_g = aux_grad.reshape(-1)

    if len(main_grad.shape) != 1:
        main_g = main_grad.reshape(-1)
        aux_g = aux_grad.reshape(-1)
    else:
        main_g = main_grad
        aux_g = aux_grad
        # main_g_norm = torch.norm(main_grad, p='fro')
        # cos_sim = F.cosine_similarity(main_grad, aux_grad, dim=0)

    main_g_norm = torch.norm(main_g, p='fro')
    cos_sim = F.cosine_similarity(main_g, aux_g, dim=0)

    if mode == 'unweighted_cosine':
        # return aux_grad if dot_sum > 0 else torch.zeros_like(aux_grad, device=aux_grad.device)
        return aux_grad if cos_sim > 0 else torch.zeros_like(aux_grad, device=aux_grad.device)
    elif mode == 'weighted_cosine':
        # return tf.maximum(u_dot_v, 0) * u / l_u / l_v
        # return torch.maximum(dot_sum, 0) * aux_grad / aux_g_norm / main_g_norm
        return torch.maximum(torch.zeros_like(cos_sim), cos_sim) * aux_grad
    elif mode == 'projection':
        # return u - tf.minimum(u_dot_v, 0) * v / l_v / l_v
        # print((aux_g * main_g / main_g_norm).shape, aux_g.shape)
        # print(torch.minimum(torch.zeros_like(aux_g), aux_g * main_g / main_g_norm).shape)
        # print((main_g / main_g_norm).shape)
        # print(aux_grad.shape)
        return aux_grad - (torch.minimum(torch.zeros_like(aux_g), aux_g * main_g / main_g_norm) * (
                main_g / main_g_norm)).view_as(aux_grad)
    elif mode == 'parameter_wise':
        # return u * ((tf.math.sign(u * v) + 1) / 2)
        return aux_grad * ((torch.sign(aux_grad * main_grad) + 1) / 2)
    else:
        raise NotImplementedError


mitigation_modes = ['unweighted_cosine',
                    'weighted_cosine',
                    'projection',
                    'parameter_wise']


def mitigate_negative_transfer(params, main_grad: torch.Tensor, aux_grad: torch.Tensor, lam: float,
                               mode: str) -> torch.Tensor:
    assert mode in mitigation_modes

    for i, p in enumerate(params):
        p.grad = main_grad[i] + lam * filter_aux_grad(main_grad[i], aux_grad[i], mode)
