import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from FashionDataloder import FashionDataset
from resnet import build_ResNet18
import torch.optim as optim


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')  #数据路径

parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')                #迭代回合数

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')     #

parser.add_argument('--b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')  #batchsize

parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')          #学习率

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')                                     #

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')   #打印频率

parser.add_argument('--resume', default='', type=str, metavar='PATH', #./model_save/resnet/checkpoint.pth.tar
                    help='path to latest checkpoint (default: none)')   #载入模型的路径  ./checkpoint/resnet50.pth

parser.add_argument('--evaluate', default=True, type=bool,
                    help='evaluate mode')     #


# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')                       # 使用预训练模型

# best_prec1 = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    global args, best_prec1
    args = parser.parse_args()

    model = build_ResNet18().to(device)  #创建resnet18 模型




    cudnn.benchmark = True

    transforms_train_data = transforms.Compose([
                                           transforms.Resize((32,32)),
                                           transforms.ToTensor(),
                                                 ])

    transforms_test_data = transforms.Compose([ transforms.Resize((32,32)),
                                           transforms.ToTensor(),
                                           ])


    FashionMinst_train = FashionDataset(r"H:\Dataset\FashionMNIST\raw",transform =transforms_train_data,Train=True)
    train_loader = torch.utils.data.DataLoader(FashionMinst_train,batch_size=args.b, shuffle=True,num_workers=0, pin_memory=True)
    FashionMinst_test = FashionDataset(r"H:\Dataset\FashionMNIST\raw",transform =transforms_test_data,Train=False)
    val_loader = torch.utils.data.DataLoader(FashionMinst_test,batch_size=args.b, shuffle=True,num_workers=0, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

    # optionally resume from a checkpoint
    if args.resume:  #载入预训练的模型
        model.load_state_dict(torch.load("checkpoint/discriminator_0004000.pt"))
        optimizer.load_state_dict(torch.load("checkpoint/dis_optimizer_0004000.pt"))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        if epoch%5==0:
            validate(val_loader, model, criterion)

        if epoch%10==0:
            torch.save(model.state_dict(), f'checkpoint/generator_{epoch+args.start_epoch}.pt')
            torch.save(optimizer.state_dict(), f'checkpoint/gen_optimizer_{epoch+args.start_epoch}.pt')

        if epoch%30==0:  #每过30个epoch,调整学习率
            scheduler_lr.step()




def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    # model.train()
    corrent = 0
    print("iter_num:",len(train_loader))
    time1 = time.time()
    for i, (input, target) in enumerate(train_loader):#train_loader
        # measure data loading time
        input_var = input.to(device)
        target_var = target.long().to(device)

        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i%100==0:
            time2 = time.time()
            print(time2-time1)
            time1 = time2

        pre_index = output.argmax(dim=1)  #获取每个样本的预测下表
        for index in range(len(pre_index)):
            if pre_index[index] == target_var[index]:
                corrent += 1

    corrent_rate = corrent / (args.b*len(train_loader))
    print("训练准确率 %d : %f",epoch,corrent_rate)


def validate(val_loader, model, criterion):

    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    corrent = 0

    print("iter_num:",len(val_loader))

    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():

            input_var = input.to(device)
            target_var = target.long().to(device)

            # compute output
            output = model(input_var)

            pre_index = output.argmax(dim=1)  # 获取每个样本的预测下表
            for index in range(len(pre_index)):
                if pre_index[index] == target_var[index]:
                    corrent += 1

    correct = corrent / (args.b*len(val_loader))
    print("测试集准确率：",correct)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()