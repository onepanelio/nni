"""
A general purpose classification script using PyTorch.
"""

import argparse
import logging
import json
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import math
writer = SummaryWriter("/mnt/output/fixed_param_tb")
logger = logging.getLogger('pytorch_classifier')


# mean = 0.0
# for images, _ in loader:
#     batch_samples = images.size(0) 
#     images = images.view(batch_samples, images.size(1), -1)
#     mean += images.mean(2).sum(0)
# mean = mean / len(loader.dataset)

# var = 0.0
# for images, _ in loader:
#     batch_samples = images.size(0)
#     images = images.view(batch_samples, images.size(1), -1)
#     var += ((images - mean.unsqueeze(1))**2).sum([0,2])
# std = torch.sqrt(var / (len(loader.dataset)*224*224))


def build_model(model_type, num_classes):
    if model_type == "googlenet":
        model = models.googlenet(pretrained=True)
        in_features = 1024
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        in_features = 2048
    elif model_type == "alexnet":
        model = models.alexnet(pretrained=True)
        in_features = 4096
    elif model_type == "vgg19":
        model = models.alexnet(pretrained=True)
        in_features = 4096
    if model_type in ['alexnet', 'vgg19']:
        model.classifier._modules['6'] = nn.Sequential(nn.Linear(in_features, num_classes),
                                        nn.LogSoftmax(dim=1))
    else:
        model.fc = nn.Sequential(nn.Linear(in_features, num_classes),
                                 nn.LogSoftmax(dim=1))
    return model


def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if (args['batch_num'] is not None) and batch_idx >= args['batch_num']:
            break
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(train_loader.dataset)
        writer.add_scalar("Loss/train", loss, batch_idx + (epoch * 10))
        writer.add_scalar("Accuracy/train", accuracy, batch_idx + (epoch * 10))
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy, test_loss


def train(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    if args['model_type'] == 'alexnet':
        w, h = 256, 256
    else:
        w, h = 224, 224

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
            ImageFolder(root=args['train_dir'], transform=transforms.Compose([
                transforms.Resize((w, h)),transforms.ToTensor(),
              # add Normlize with mean and std
        ])),
        batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            ImageFolder(root=args['test_dir'], transform=transforms.Compose([
                transforms.Resize((w, h)),transforms.ToTensor(),
              # add Normlize with mean and std
        ])),
        batch_size=args['batch_size'], shuffle=True, **kwargs)


    model = build_model(args['model_type'], args['num_classes']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
                          momentum=args['momentum'])
    
    if not os.path.exists('/mnt/output/fixed-params'):
        os.makedirs('/mnt/output/fixed-params')

    for epoch in range(1, args['epochs'] + 1):
        train_one_epoch(args, model, device, train_loader, optimizer, epoch)
        test_acc, test_loss = test(args, model, device, test_loader)
        writer.add_scalar("Loss/test", test_loss, epoch )
        writer.add_scalar("Accuracy/test", test_acc, epoch )
        torch.save(model, '/mnt/output/fixed-params/fixed-params-model-epochs-{}-acc-{}'.format(epoch, round(test_acc, 2)))
        # report intermediate result
        print('test accuracy: {} test loss: {}'.format(test_acc, test_loss))

    # report final result
    print('Final result is ', test_acc)
    return test_acc, test_loss


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Classification Example')
    parser.add_argument("--train_dir", type=str,
                        default='/home/savan/Documents/train_data', help="train data directory")
    parser.add_argument("--test_dir", type=str,
                        default='/home/savan/Documents/test_data', help="test data directory")
    parser.add_argument("--model_type", type=str,
                        default='alexnet', help="model to train")
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=2, metavar='N',
                        help='number of classes in the dataset')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        params = vars(get_params())
        print("Current Parameters:\n")
        print(params)
        acc, loss = train(params)
        if loss is None or math.isnan(loss):
            loss = 0
        metrics = [
          {'name': 'accuracy', 'value': acc},
          {'name': 'loss', 'value': loss},
        ]
        
        # Write metrics to `/tmp/sys-metrics.json`
        with open('/tmp/sys-metrics.json', 'w') as f:
            json.dump(metrics, f)
    except Exception as exception:
        logger.exception(exception)
        raise
