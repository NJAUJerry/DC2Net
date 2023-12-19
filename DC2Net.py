from torch.autograd import Variable, Function
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchsummary import summary
import scipy.io as sio
import numpy as np
import pandas as pd
import time


class DeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformConv3d, self).__init__()
        self.kernel_size = kernel_size
        N = kernel_size ** 3
        self.stride = stride
        self.padding = padding
        self.zero_padding = nn.ConstantPad3d(padding, 0)
        self.conv_kernel = nn.Conv3d(in_channels * N, out_channels, kernel_size=1, bias=bias)
        self.offset_conv_kernel = nn.Conv3d(in_channels, N * 3, kernel_size=kernel_size, padding=padding, bias=bias)

        self.mode = "deformable"

    def deformable_mode(self, on=True):  #
        if on:
            self.mode = "deformable"
        else:
            self.mode = "regular"

    def forward(self, x):
        if self.mode == "deformable":
            offset = self.offset_conv_kernel(x)
        else:
            b, c, h, w, d = x.size()
            offset = torch.zeros(b, 3 * self.kernel_size ** 3, h, w, d).to(x)

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 3

        if self.padding:
            x = self.zero_padding(x)

        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)
        p = p[:, :, ::self.stride, ::self.stride, ::self.stride]

        # (b, h, w, d, 3N), N == ks ** 3, 3N - 3 coords for each point on the activation map
        p = p.contiguous().permute(0, 2, 3, 4, 1)  # 5D array

        q_sss = Variable(p.data, requires_grad=False).floor()  # point with all smaller coords
        #         q_sss = p.data.floor() - same? / torch.Tensor(p.data).floor()
        q_lll = q_sss + 1  # all larger coords

        # 8 neighbor points with integer coords
        q_sss = torch.cat([
            torch.clamp(q_sss[..., :N], 0, x.size(2) - 1),  # h_coord
            torch.clamp(q_sss[..., N:2 * N], 0, x.size(3) - 1),  # w_coord
            torch.clamp(q_sss[..., 2 * N:], 0, x.size(4) - 1)  # d_coord
        ], dim=-1).long()
        q_lll = torch.cat([
            torch.clamp(q_lll[..., :N], 0, x.size(2) - 1),  # h_coord
            torch.clamp(q_lll[..., N:2 * N], 0, x.size(3) - 1),  # w_coord
            torch.clamp(q_lll[..., 2 * N:], 0, x.size(4) - 1)  # d_coord
        ], dim=-1).long()
        q_ssl = torch.cat([q_sss[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_sls = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_sll = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lss = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_lsl = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lls = torch.cat([q_lll[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)

        # (b, h, w, d, N)
        mask = torch.cat([
            p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
            p[..., N:2 * N].lt(self.padding) + p[..., N:2 * N].gt(x.size(3) - 1 - self.padding),
            p[..., 2 * N:].lt(self.padding) + p[..., 2 * N:].gt(x.size(4) - 1 - self.padding),
        ], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))  # все еще непонятно, что тут происходит за wtf
        p = p * (1 - mask) + floor_p * mask

        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:2 * N], 0, x.size(3) - 1),
            torch.clamp(p[..., 2 * N:], 0, x.size(4) - 1),
        ], dim=-1)

        # trilinear kernel (b, h, w, d, N)
        g_sss = (1 + (q_sss[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_sss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_sss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lll = (1 - (q_lll[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_lll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_lll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_ssl = (1 + (q_ssl[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_ssl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_ssl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sls = (1 + (q_sls[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_sls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_sls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sll = (1 + (q_sll[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_sll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_sll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lss = (1 - (q_lss[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_lss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_lss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lsl = (1 - (q_lsl[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_lsl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_lsl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lls = (1 - (q_lls[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_lls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_lls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        # get values in all 8 neighbor points
        # (b, c, h, w, d, N) - 6D-array
        x_q_sss = self._get_x_q(x, q_sss, N)
        x_q_lll = self._get_x_q(x, q_lll, N)
        x_q_ssl = self._get_x_q(x, q_ssl, N)
        x_q_sls = self._get_x_q(x, q_sls, N)
        x_q_sll = self._get_x_q(x, q_sll, N)
        x_q_lss = self._get_x_q(x, q_lss, N)
        x_q_lsl = self._get_x_q(x, q_lsl, N)
        x_q_lls = self._get_x_q(x, q_lls, N)

        # (b, c, h, w, d, N)
        x_offset = g_sss.unsqueeze(dim=1) * x_q_sss + \
                   g_lll.unsqueeze(dim=1) * x_q_lll + \
                   g_ssl.unsqueeze(dim=1) * x_q_ssl + \
                   g_sls.unsqueeze(dim=1) * x_q_sls + \
                   g_sll.unsqueeze(dim=1) * x_q_sll + \
                   g_lss.unsqueeze(dim=1) * x_q_lss + \
                   g_lsl.unsqueeze(dim=1) * x_q_lsl + \
                   g_lls.unsqueeze(dim=1) * x_q_lls

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            indexing='ij')

        # (3N, 1) - 3 coords for each of N offsets
        # (x1, ... xN, y1, ... yN, z1, ... zN)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3 * N, 1, 1, 1))
        p_n = torch.from_numpy(p_n).type(dtype)

        return p_n

    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = torch.from_numpy(p_0).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype).to(offset)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype).to(offset)
        p = p_0 + p_n + offset

        return p

    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()

        #           (0, 1, 2, 3, 4)
        # x.size == (b, c, h, w, d)
        padded_w = x.size(3)
        padded_d = x.size(4)
        c = x.size(1)
        # (b, c, h*w*d)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        # offset_x * w * d + offset_y * d + offset_z
        index = q[..., :N] * padded_w * padded_d + q[..., N:2 * N] * padded_d + q[..., 2 * N:]
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, d, N = x_offset.size()
        x_offset = x_offset.permute(0, 1, 5, 2, 3, 4)
        x_offset = x_offset.contiguous().view(b, c * N, h, w, d)

        return x_offset

def deform_3d(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return DeformConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# Residual module
class ResidualBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1, self).__init__()

        #
        self.redis = nn.Sequential(
            nn.Conv3d(1, 8, (1, 1, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm3d(8),
            nn.Conv3d(8, 16, (11, 3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, (1, 1, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm3d(32),
        )

        self.alive = nn.Sequential(
            nn.Conv3d(1, 32, (1, 1, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm3d(32),
        )

    def forward(self, x):
        residual = self.redis(x)
        alive = self.alive(x)
        out = residual + alive

        return out



class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock2, self).__init__()

        #
        self.redis = nn.Sequential(
            nn.Conv3d(32, 32, (1, 1, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, (3, 3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, (1, 1, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm3d(128),
        )

        self.alive = nn.Sequential(
            nn.Conv3d(32, 128, (1, 1, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm3d(128),
        )

    def forward(self, x):
        residual = self.redis(x)
        alive = self.alive(x)
        out = residual + alive

        return out


class D3DCNN(nn.Module):
    def __init__(self):
        super(D3DCNN, self).__init__()

        self.sdc = nn.Sequential(
            nn.Conv3d(1, 16, (3, 1, 1), dilation=(3, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 1, (3, 1, 1), dilation=(3, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(1),
        )

        self.deform3d = deform_3d(1, 1)

        self.res1 = ResidualBlock1(1, 32)
        self.pooling1 = nn.AdaptiveAvgPool3d((9, 12, 12))
        self.res2 = ResidualBlock2(32, 128)
        self.pooling2 = nn.AdaptiveAvgPool3d((4, 6, 6))

        self.linear = nn.Sequential(
            nn.Linear(18432, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.sdc(x)
        x = self.deform3d(x)
        x = self.res1(x)
        x = self.pooling1(x)
        x = self.res2(x)
        x = self.pooling2(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])
        x = self.linear(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d3d_net = D3DCNN().to(device)
summary(d3d_net, input_size=(1, 70, 13, 13))

def test_net():


    x = torch.randn(1, 1, 50, 9, 9)
    net = D3DCNN()
    y = net(x)
    print(y.shape)



def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX



def createImageCubes(X, y, windowSize=25, removeZeroLabels=False):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros(
        (X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0


    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1


    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def split_train_test_set(x, y, test_ratio, random_state=957):
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_ratio,
                                                        random_state=random_state,
                                                        stratify=y)

    return x_train, x_test, y_train, y_test


def create_data_loader():
    y = np.load('dataset/soy3d_y.npy')

    X = pd.read_csv('dataset/soybean306.csv')
    X = X.iloc[:, 50:120]
    x = X.to_numpy()

    x = np.resize(x, (110, 60, 70))
    y = np.resize(y, (110, 60))

    test_ratio = 0.8
    patch_size = 13
    pca_components = 70

    print('Hyperspectral data shape: ', x.shape)
    print('Label shape: ', y.shape)

    print('\n... ... create data cubes ... ...')
    x, y = createImageCubes(x, y, windowSize=patch_size)
    print('Data cube X shape: ', x.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    x_train, x_test, y_train, y_test = split_train_test_set(x, y, test_ratio)
    print('Xtrain shape: ', x_train.shape)
    print('Xtest  shape: ', x_test.shape)

    x_train = x_train.reshape(-1, patch_size, patch_size, pca_components, 1)
    x_test = x_test.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', x_train.shape)
    print('before transpose: Xtest  shape: ', x_test.shape)

    x_train = x_train.transpose(0, 4, 3, 1, 2)
    x_test = x_test.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', x_train.shape)
    print('after transpose: Xtest  shape: ', x_test.shape)

    train_set = TrainDS(x_train, y_train)
    test_set = TestDS(x_test, y_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=64,
                                              shuffle=False,
                                              num_workers=2)

    return train_loader, test_loader, y_test


""" Training dataset"""


class TrainDS(torch.utils.data.Dataset):

    def __init__(self, x_train, y_train):
        self.len = x_train.shape[0]
        self.x_data = torch.FloatTensor(x_train)
        self.y_data = torch.LongTensor(y_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


""" Testing dataset"""


class TestDS(torch.utils.data.Dataset):

    def __init__(self, x_test, y_test):
        self.len = x_test.shape[0]
        self.x_data = torch.FloatTensor(x_test)
        self.y_data = torch.LongTensor(y_test)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



lr, num_epochs = 0.0001, 30


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = D3DCNN().to(device)


optimizer = torch.optim.Adam(net.parameters(), lr=lr)

trainacc = []
testacc = []


def train(train_iter, test_iter, net, optimizer, device, num_epochs):

    net = net.to(device)
    print("training on", device)

    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:

            X = X.to(device)

            y = y.to(device)

            y_hat = net(X)

            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1


        train_acc = train_l_sum / batch_count
        trainacc.append(train_acc)

        test_acc = evaluate_accuracy(test_iter, net)
        testacc.append(test_acc)
        print('epoch {}, loss {:.6f}, train acc {:.6f}, test acc{:.6f}, time {:.2f} sec'
              .format(epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):

        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):

                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()

                net.train()
            else:
                if ('is_training' in net.__code__.co_varnames):

                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

# 306
train(train_iter=train_loader, test_iter=test_loader, net=net, optimizer=optimizer,
      device=device,num_epochs=num_epochs)


train_acc = np.array(trainacc)
print(train_acc.shape)
np.save("ans/D3D_all_train_acc.npy",train_acc)

test_acc = np.array(testacc)
print(test_acc.shape)
np.save("ans/D3D_all_test_acc.npy",test_acc)

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(device, net, test_loader, y_test):
    count = 0

    # Model Testing
    y_pred_test = 0
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))

    target_names = ['Healthy', 'Asymptomatic', 'Symptomatic']
    classification = classification_report(y_test, y_pred_test, target_names=target_names, digits=6)

    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100

classification, confusion, oa, each_acc, aa, kappa = reports (device, net, test_loader, y_test)


# DC2Net-all
print(classification)
print(confusion)
print("oa:",oa)
print("each_acc:",each_acc)
print("aa:",aa)
print("kappa:",kappa)


