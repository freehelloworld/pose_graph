from __future__ import print_function, division

import argparse
from torch.utils.data import DataLoader
import torch
from pose_dataset import PoseDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from scipy.spatial.distance import cdist
from pose_dataset import PoseNomalise

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/pd_pose')


class PoseGraphNet(nn.Module):

    def __init__(self, img_size=10):
        super(PoseGraphNet, self).__init__()

        N = img_size ** 2
        self.fc = nn.Linear(N*2, 2, bias=False)

        col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
        coord = np.stack((col, row), axis=2).reshape(-1, 2)
        coord = (coord - np.mean(coord, axis=0)) / (np.std(coord, axis=0) + 1e-5)

        coord = torch.from_numpy(coord).float()

        coord = torch.cat((coord.unsqueeze(0).repeat(N, 1, 1),
                           coord.unsqueeze(1).repeat(1, N, 1)), dim=2)

        self.pred_edge_fc = nn.Sequential(nn.Linear(4, 64),
                                          nn.ReLU(),
                                          nn.Linear(64, 1),
                                          nn.Tanh())
        self.register_buffer('coord', coord)

    def forward(self, x):
        B = x.size(0)
        self.A = self.pred_edge_fc(self.coord).squeeze()

        layer_1 = torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1), x.view(B, -1, 2).float())
        layer_2 = torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1), layer_1)

        result = layer_1.add(layer_2)/2

        return self.fc(result.view(B, -1))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    running_loss = 0.0
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched['pose']
        target = sample_batched['label']
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        output = torch.sigmoid(output)

        loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar('training loss',
                              running_loss / 1000,
                              epoch * len(train_loader) + batch_idx)

            running_loss = 0.0


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(test_loader):
            data = sample_batched['pose']
            target = sample_batched['label']
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.sigmoid(output)

            postive_results = output[:, 1]

            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)

            # for calc ROC
            result = torch.stack([postive_results, target], dim=-1)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def plot_roc(y_true, y_score):
    import sklearn.metrics as metrics
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
    print(threshold)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PD pose graph')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    use_cuda = False

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    pose_data = PoseDataset('video_data.csv', transform=transforms.Compose([
                                               PoseNomalise(100)
                                           ]))

    train_data, test_data = torch.utils.data.random_split(pose_data, [12000, 4074])

    train_loader = DataLoader(train_data, batch_size=500,
                            shuffle=True, num_workers=0)

    test_loader = DataLoader(test_data, batch_size=4074,
                              shuffle=True, num_workers=0)

    model = PoseGraphNet()
    #writer.add_graph(model)
    model.to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print('number of trainable parameters: %d' %
          np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()]))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)

    writer.close()
    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
