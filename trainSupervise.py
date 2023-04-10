import torch
from torch import nn
from tqdm import tqdm
from utils.dataLoader import LoadSuperviseDataset
from utils.models import select_model
from utils.plotShow import plot_history
from torchvision import transforms
from utils.losses import Focal_loss

def accuracy(y_hat, y):
    # 预测精度
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / len(y)


def train(model_in_use, model_kargs, train_iter, test_iter, num_epochs, lr, device, threshold, save_checkpoint=False, save_steps=50):
    # 训练模型
    net = select_model(model_in_use, model_kargs)

    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print("device in : ", device)
    net = net.to(device)

    loss = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        data_num = 0

        with tqdm(range(len(train_iter)), ncols=100, colour='red',
                  desc="{} train epoch {}/{}".format(model_in_use, epoch + 1, num_epochs)) as pbar:
            for i, (X, y) in enumerate(train_iter):
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                train_loss += l.detach()
                train_acc += accuracy(y_hat.detach(), y.detach())
                data_num += 1
                pbar.set_postfix({'loss': "{:.4f}".format(train_loss / data_num), 'acc': "{:.4f}".format(train_acc / data_num)})
                pbar.update(1)

        history['train_loss'].append(float(train_loss / data_num))
        history['train_acc'].append(float(train_acc / data_num))

        net.eval()
        test_loss = 0.0
        test_acc = 0.0
        data_num = 0
        with tqdm(range(len(test_iter)), ncols=100, colour='blue',
                  desc="{} test epoch {}/{}".format(model_in_use, epoch + 1, num_epochs)) as pbar:
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                with torch.no_grad():
                    l = loss(y_hat, y)
                    test_loss += l.detach()
                    test_acc += accuracy(y_hat.detach(), y.detach())

                    data_num += 1
                    pbar.set_postfix({'loss': "{:.4f}".format(test_loss / data_num), 'acc': "{:.4f}".format(test_acc / data_num)})
                    pbar.update(1)

        history['test_loss'].append(float(test_loss / data_num))
        history['test_acc'].append(float(test_acc / data_num))
        if history['test_acc'][-1] > threshold:
            print("early stop")
            break
        if save_checkpoint and (epoch+1) % save_steps == 0:
            torch.save(net.state_dict(), "./model_weights/supervise-ep{}-{}-acc-{:.4f}-loss-{:.4f}.pth".format(epoch+1, model_in_use, history['test_acc'][-1], history['test_loss'][-1]))

    torch.save(net.state_dict(), "./model_weights/{}-supervise.pth".format(model_in_use))
    return history


if __name__ == '__main__':
    batch_size = 4086  # 批量大小
    in_channels = 3  # 输入通道数
    num_classes = 10  # 预测类别
    num_epochs = 200  # 训练轮次
    lr = 2e-2
    threshold = 0.95  # 提前停止的阈值，即测试精度超过这个阈值就停止训练
    model_in_use = 'resNet18'  # 模型选用，可选项有：leNet, alexNet， vgg11, NiN, GoogLeNet, resNet18, denseNet
    model_kargs = {'in_channels': in_channels, "num_classes": num_classes}  # 模型参数，即输入通道数和总类别
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 测试cuda并使用
    trans = [transforms.ToTensor(), transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])]
    if model_in_use == 'leNet':
        trans.insert(0, transforms.Resize((28, 28)))
    elif model_in_use in ['alexNet', 'vgg11']:
        trans.insert(0, transforms.Resize((224, 224)))

    trans = transforms.Compose(trans)

    train_iter, test_iter = LoadSuperviseDataset(batch_size=batch_size, trans=trans)

    history = train(model_in_use, model_kargs, train_iter, test_iter, num_epochs, lr, device, threshold, save_checkpoint=True)  # 训练

    plot_history(model_in_use + "_supervise", history)  # 画出损失和精度的图
