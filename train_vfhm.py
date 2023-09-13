import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboard_logger import configure, log_value

from augment import load_fundus_transforms
from dataset import eye_types, VFDatasetVFHM, L_mask, R_mask
from model import model_names, ModelAux, mitigation_modes, mitigate_negative_transfer
from utils import save_checkpoint, set_seed, save_yaml_file, print_info, AverageMeter, ProgressMeter, \
    adjust_learning_rate

parser = argparse.ArgumentParser(description='VF-HM for VF estimation using fundus')

# dataset
parser.add_argument('--train_csv_file', type=str,
                    help='train csv file')

parser.add_argument('--train_fundus_dir', type=str,
                    help='train fundus dir')

parser.add_argument('--train_vf_dir', type=str,
                    help='train vf dir')

parser.add_argument('--test_csv_file', type=str,
                    help='test csv file')

parser.add_argument('--test_fundus_dir', type=str,
                    help='test fundus dir')

parser.add_argument('--test_vf_dir', type=str,
                    help='test vf dir')

parser.add_argument('--eye_type', type=str, default='L', choices=eye_types,
                    help='eye_type (default: L)')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')

parser.add_argument('--pretrained', action='store_true',
                    help='use ImageNet-1k pretrained weights')

parser.add_argument('--mode', default='weighted_cosine',
                    choices=mitigation_modes,
                    help='mitigation mode: ' +
                         ' | '.join(mitigation_modes) +
                         ' (default: weighted_cosine)')

parser.add_argument('--lam', type=float, default='0.1',
                    help='aux task coefficient, default: 0.1')

# training config
parser.add_argument('--log_dir', default='./runs', type=str, metavar='PATH',
                    help='where checkpoint and logs to be saved (default: ./runs)')
parser.add_argument('--num_workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=80, metavar='N',
                    help='number of total epochs to run (default: 80)')
parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 32)')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='wd',
                    help='weight decay, default: 1e-4')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')

augment_types = ['identity', 'rotation', 'shift',
                 'scale',
                 'brightness_contrast_saturation',
                 'all',
                 'trivialaugment']

parser.add_argument('--augment',
                    choices=augment_types,
                    default='trivialaugment',
                    help='fundus augment types: ' +
                         ' | '.join(augment_types))

parser.add_argument('--seed', type=int, default=42)


def train(train_loader, model, optimizer, epoch, print_freq, mitigation_mode):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3e')
    rmse_losses = AverageMeter('RMSE', ':.3e')
    mae_losses = AverageMeter('MAE', ':.3e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, rmse_losses, mae_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (fundus, mm, vf) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        fundus = fundus.cuda()
        vf = vf.long().cuda()
        mm = mm.long().cuda()

        # compute output
        output, aux_output = model(fundus, True)
        loss, main_loss, aux_loss = model.loss(output, vf, aux_output, mm)

        # grad
        params = model.get_shared_params()
        main_grad = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
        aux_grad = torch.autograd.grad(aux_loss, params, retain_graph=True, allow_unused=True)

        with torch.no_grad():
            rmse, mae = model.score(output, vf)

        losses.update(loss.item(), fundus.size(0))
        rmse_losses.update(rmse.item(), fundus.size(0))
        mae_losses.update(mae.item(), fundus.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # mitigate negative transfer
        mitigate_negative_transfer(params, main_grad, aux_grad, model.lam, mitigation_mode)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    log_value('train/loss', losses.avg, epoch)
    log_value('train/rmse', rmse_losses.avg, epoch)
    log_value('train/mae', mae_losses.avg, epoch)
    log_value('train/data_time', data_time.avg, epoch)
    log_value('train/batch_time', batch_time.avg, epoch)


def test(test_loader, model, epoch, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3e')
    rmse_losses = AverageMeter('RMSE', ':.3e')
    mae_losses = AverageMeter('MAE', ':.3e')

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, losses, rmse_losses, mae_losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (fundus, vf) in enumerate(test_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            fundus = fundus.cuda()
            vf = vf.long().cuda()

            # compute output
            output = model(fundus)

            loss = model.loss(output, vf)
            rmse, mae = model.score(output, vf)

            losses.update(loss.item(), fundus.size(0))
            rmse_losses.update(rmse.item(), fundus.size(0))
            mae_losses.update(mae.item(), fundus.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        print('* Loss: {losses.avg: .3f}, RMSE: {rmse_losses.avg: .3f}, MAE: {mae_losses.avg: .3f}'
              .format(losses=losses, rmse_losses=rmse_losses, mae_losses=mae_losses))

    # log
    log_value('test/loss', losses.avg, epoch)
    log_value('test/rmse', rmse_losses.avg, epoch)
    log_value('test/mae', mae_losses.avg, epoch)
    log_value('test/data_time', data_time.avg, epoch)
    log_value('test/batch_time', batch_time.avg, epoch)

    return rmse_losses.avg, mae_losses.avg


def main():
    args = parser.parse_args()

    set_seed(args.seed)

    args.datetime = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    # args.num_outputs = 52
    args.num_classes = 41
    args.num_aux_classes = 5
    args.vf_type = 'array'
    args.input_resolution = 384

    print_info(args)

    # dataset
    train_transform, test_transform = load_fundus_transforms(args.augment,
                                                             args.input_resolution)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    configure(args.log_dir)
    save_yaml_file(args.log_dir, vars(args))

    vf_transform = None

    train_dataset = VFDatasetVFHM(csv_file=args.train_csv_file,
                                  fundus_dir=args.train_fundus_dir,
                                  vf_dir=args.train_vf_dir,
                                  vf_type=args.vf_type,
                                  include_mm=True,
                                  fundus_transform=train_transform,
                                  vf_transform=vf_transform,
                                  eye_type=args.eye_type)

    test_dataset = VFDatasetVFHM(csv_file=args.test_csv_file,
                                 fundus_dir=args.test_fundus_dir,
                                 vf_dir=args.test_vf_dir,
                                 vf_type=args.vf_type,
                                 include_mm=False,
                                 fundus_transform=test_transform,
                                 vf_transform=vf_transform,
                                 eye_type=args.eye_type)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.eye_type == 'L':
        mask = L_mask
    elif args.eye_type == 'R':
        mask = R_mask
    else:
        raise NotImplementedError

    mask = torch.from_numpy(mask).cuda()

    model = ModelAux(arch=args.arch,
                     in_channels=args.in_channels,
                     num_classes=args.num_classes,
                     mask=mask,
                     pretrained=args.pretrained,
                     num_aux_classes=args.num_aux_classes,
                     lam=args.lam)

    model = model.cuda()

    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.wd,
                                nesterov=True)

    for epoch in range(args.epochs):
        print('epoch: {}, lr: {:.5e}'.format(epoch, optimizer.param_groups[0]['lr']))

        train(train_loader, model, optimizer, epoch, args.print_freq, args.mitigation_mode)

        # lr_scheduler.step()
        adjust_learning_rate(optimizer, args.lr, epoch, args.epochs)

        rmse, mae = test(test_loader, model, epoch, args.print_freq)

    # save at the end
    save_checkpoint(model.state_dict(), False, args.log_dir)

    result_dict = {'rmse': rmse, 'mae': mae}

    print(result_dict)

    save_yaml_file(args.log_dir, result_dict, 'result.yml')


if __name__ == '__main__':
    main()
