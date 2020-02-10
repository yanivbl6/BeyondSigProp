"""
Based on code by xternalz: https://github.com/xternalz/WideResNet-pytorch
Wide ResNet by Sergey Zagoruyko and Nikos Komodakis
"""

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import torchvision

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.autograd import Variable

##from model import WideResNet
from remodel import LRNet as WideResNet

##from model import ResNet5
##from remodel import ConvNet as WideResNet
##from remodel import NarrowNet as WideResNet

from utils.cutout import Cutout
from utils.radam import RAdam, AdamW
from utils.imgnet import IMGNET


from tensorboard_logger import configure, log_value

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchviz import make_dot

import prunhild
import correlation


import time

parser = argparse.ArgumentParser(description="PyTorch WideResNet Training")
parser.add_argument("--print-freq", "-p", default=10, type=int, help="default: 10")
parser.add_argument("--layers", default=28, type=int, help="default: 28")
parser.add_argument("--widen-factor", default=10, type=int, help="default: 10")
parser.add_argument("--batchnorm", default=False, help="apply BatchNorm",  action='store_true')
parser.add_argument("--fixup", default=True, help="apply Fixup", action='store_false'  )
parser.add_argument("--droprate", default=0.0, type=float, help="default: 0.0")
parser.add_argument("--cutout", default=False, type=bool, help="apply cutout")
parser.add_argument("--length", default=16, type=int, help="length of the holes")
parser.add_argument("--n_holes", default=1, type=int, help="number of holes to cut out")
parser.add_argument(
    "--dataset", default="cifar10", type=str, help="cifar10 [default], cifar100, cinic10, imgnet10 or imgnet100"
)

parser.add_argument("--epochs", default=200, type=int, help="default: 200")
parser.add_argument("--start-epoch", default=0, type=int, help="epoch for restart")
parser.add_argument("-b", "--batch-size", default=128, type=int, help="default: 128")

parser.add_argument('--optimizer', default='sgd', type=str, choices=['radam', 'sgd'])

parser.add_argument('--beta1', default=0.9, type=float,
                    help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='beta2 for adam')

parser.add_argument(
    "--lr", "--learning-rate", default=0.1, type=float, help="default: 0.1"
)
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--nesterov", default = False, action = "store_true" , help="nesterov momentum")
parser.add_argument(
    "--weight-decay", "--wd", default=5e-4, type=float, help="default: 5e-4"
)
parser.add_argument('--alpha', default=0.0, type=float,
                    help='mixup interpolation coefficient (default: 0.0)')


parser.add_argument('--forceW', default=0.0, type=float,
                    help='mixup interpolation coefficient (default: 0.0)')



parser.add_argument(
    "--resume", default="", type=str, help="path to latest checkpoint (default: '')"
)
parser.add_argument(
    "-d","--device", default="0", type=str, help="GPU Device"
)
parser.add_argument(
    "--name", default="WideResNet-28-10", type=str, help="name of experiment"
)

parser.add_argument(
    "--init", default="None", type=str, help="rlnet init"
)

parser.add_argument(
    "--dir", default="/home/NAME/Datasets", type=str, help="dataset directory"
)

parser.add_argument(
    "--tensorboard", help="Log progress to TensorBoard", action="store_true"
)
parser.add_argument(
    "--no-augment",
    dest="augment",
    action="store_false",
    help="whether to use standard augmentation (default: True)",
)

parser.add_argument(
    "--no-saves",
    dest="savenet",
    action="store_false",
    help="whether to save networks weights (default: True)",
)

parser.add_argument(
    "--prune", default="", type=str, help="path to checkpoint to prone"
)

parser.add_argument(
    "--prune_epoch", default=0, type=int, help="base epoch to use weights from"
)

parser.add_argument(
    "--prune_classes", default="0", type=str, help="Number of classes in the source pruned network (0- same as current dataset)"
)

parser.add_argument("--randomize_mask", default=False, action='store_true' , help="Use random mask for pruning")


parser.add_argument(
    "--cutoff", default=0.15, type=float, help="ratio of weights to keep"
)

parser.add_argument("--eval", default=False, action='store_true' , help="Evaluation only")

parser.add_argument("--varnet", default=False, action='store_true' , help="Use diversed initialization")

parser.add_argument("--symmetry_break", default=False, action='store_true' , help="Quit if the accuracy is over 50% for some time")

parser.add_argument(
    "--noise", default=0.0, type=float, help="noise. 0.0: constnet, 1.0: varnet"
)

parser.add_argument(
    "--lrelu", default=0.0, type=float, help="leaky relu variable"
)


parser.add_argument(
    "-W", "--sigmaW", default=-1.0, type=float, help="Varnet parameter for initializing weights"
)

parser.add_argument(
    "--freeze", default=0, type=int, help="Number of top layers to freeze"
)

parser.add_argument(
    "--freeze_start", default=0, type=int, help="Number of top layers not to freeze"
)

parser.add_argument(
    "--res_freeze", default=0, type=int, help="Number of conv layers to freeze"
)

parser.add_argument(
    "--res_freeze_start", default=0, type=int, help="Number of conv layers not to freeze"
)

parser.add_argument(
    "--cudaNoise", default=True, action='store_false' , help="Turn cudann to deterministic"
)

parser.set_defaults(augment=True)


def getQ(T):
    t = T.detach()
    s = t.shape
    a = t - torch.mean(t,[1]).view(s[0],1,s[2],s[3]).expand_as(t)
    return torch.mean(a*a)



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

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




def draw(args,model):
    misc = torch.zeros([args.batch_size,3,32,32])
    dot = make_dot(model(misc), params= dict(model.named_parameters()))
    dot.format = 'pdf'
    

    if args.batchnorm:
        dot.render("batchnorm-graph") 
    elif args.fixup:
        dot.render("fixup-graph")
    else:
        dot.render("no-fixup-no-bn-graph")
    
def justParse(txt=None):
    if not txt:
        args = parser.parse_args()
    else:
        args = parser.parse_args(txt.split())

    if args.sigmaW == -1.0:
        if args.varnet:
            args.sigmaW = 1.0
        else:
            args.sigmaW = 0.0

    return args

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

def prune_on(args,str):
    if intersection(['bn','scale','conv_res','fc'],str.split('.')):
        return False
    return True

def get_params_for_pruning(args,model):
    return [param for param,name in zip(list(model.parameters()),list(model.named_parameters())) if prune_on(args,name[0])]

def count_pruned_weights(parameters, cutoff = 1.0):
    return int(np.sum([param.nelement() for  param in parameters]) * (cutoff))

def randomize_mask(mask,cutoff):
    st = mask['state']
    for k in st.keys():
        m = st[k]['prune_mask']
        mshape = (m.shape)
        m.data = torch.bernoulli(cutoff * torch.ones(mshape))
    return mask

def getPruneMask(args):
    baseTar =  "runs/%s-net/checkpoint.pth.tar" % args.prune
    if os.path.isfile(baseTar):
        

        classes = onlydigits(args.prune_classes)
        if classes == 0:
            classes = args.classes

        fullModel = WideResNet(
            args.layers,
            classes,
            args.widen_factor,
            droprate=args.droprate,
            use_bn=args.batchnorm,
            use_fixup=args.fixup,
            varnet = args.varnet,
            noise = args.noise,
            lrelu = args.lrelu,
            sigmaW = args.sigmaW,
        )


        if torch.cuda.device_count() > 1:
            
            start = int(args.device[0])
            end  = int(args.device[2])+1
            torch.cuda.set_device(start)
            dev_list=[]
            for i in range(start,end):
                dev_list.append("cuda:%d" % i)
            fullModel = torch.nn.DataParallel(fullModel, device_ids=dev_list)

        fullModel = fullModel.cuda()


        print(f"=> loading checkpoint {baseTar}")

        checkpoint = torch.load(baseTar)
        fullModel.load_state_dict(checkpoint["state_dict"])


        # --------------------------- #
        # --- Pruning Setup Start --- #

        cutoff = prunhild.cutoff.LocalRatioCutoff(args.cutoff)
        # don't prune the final bias weights
        params = get_params_for_pruning(args,fullModel)

        print(params)

        pruner = prunhild.pruner.CutoffPruner(params, cutoff, prune_online=True)
        pruner.prune()
       
        print(f"=> loaded checkpoint '{baseTar}' (epoch {checkpoint['epoch']})")

        if torch.cuda.device_count() > 1:
            start = int(args.device[0])
            end  = int(args.device[2])+1
            for i in range(start,end):
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()

        mask = pruner.state_dict()
        if args.randomize_mask:
            mask = randomize_mask(mask,args.cutoff)

        return mask
    else:
        print(f"=> no checkpoint found at {baseTar}")
        return None
        
def main(txt=None):
    args = justParse(txt)

    return main2(args)

def nondigits(txt):
    return ''.join([i for i in txt if not i.isdigit()])

def onlydigits(txt):
    return int(''.join([i for i in txt if i.isdigit()]))





def main2(args):
    best_prec1 = 0.0

    torch.backends.cudnn.deterministic = not args.cudaNoise

    torch.manual_seed(time.time())

    if args.init != "None":
        args.name = "lrnet_%s" % args.init

    if args.tensorboard:
        configure(f"runs/{args.name}")

    dstype = nondigits(args.dataset)
    if dstype ==  "cifar":
        means = [125.3, 123.0, 113.9]
        stds =  [63.0, 62.1, 66.7]
    elif dstype ==  "imgnet":
        means = [123.3, 118.1, 108.0]
        stds =  [54.1, 52.6, 53.2]

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in means],
        std=[x / 255.0 for x in stds],
    )


    writer = SummaryWriter(log_dir="runs/%s" % args.name  ,comment=str(args))
    args.classes =  onlydigits(args.dataset)


    if args.augment:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: F.pad(
                        x.unsqueeze(0), (4, 4, 4, 4), mode="reflect"
                    ).squeeze()
                ),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform_train = transforms.Compose([transforms.ToTensor(), normalize])

    if args.cutout:
        transform_train.transforms.append(
            Cutout(n_holes=args.n_holes, length=args.length)
        )

    transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    kwargs = {"num_workers": 1, "pin_memory": True}


    assert dstype in ["cifar","cinic","imgnet"]

    if dstype == "cifar": 
        train_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()](
                "../data", train=True, download=True, transform=transform_train
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs,
        )
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[args.dataset.upper()](
                "../data", train=False, transform=transform_test
            ),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs,
        )
    elif  dstype == "cinic":
        cinic_directory = "%s/cinic10" % args.dir
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(cinic_directory + '/train',
                transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
            batch_size=args.batch_size, shuffle=True, **kwargs,)
        print("Using CINIC10 dataset")
        val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(cinic_directory + '/valid',
                transform=transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
            batch_size=args.batch_size, shuffle=True, **kwargs,)
    elif dstype == "imgnet":
        print("Using converted imagenet")
        train_loader = torch.utils.data.DataLoader(
            IMGNET("%s" % args.dir, train=True, transform=transform_train, target_transform=None, classes = args.classes),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs,
        )
        val_loader = torch.utils.data.DataLoader(
            IMGNET("%s" % args.dir, train=False, transform=transform_test, target_transform=None, classes = args.classes),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs,
        )
    else:
        print("Error matching dataset %s" % dstype)

    ##print("main bn:")
    ##print(args.batchnorm)
    ##print("main fixup:")
    ##print(args.fixup)

    if args.prune:
        pruner_state = getPruneMask(args)
        if pruner_state is None:
            print("Failed to prune network, aborting")
            return None


    model = WideResNet(
        args.layers,
        args.classes,
        args.widen_factor,
        droprate=args.droprate,
        use_bn=args.batchnorm,
        use_fixup=args.fixup,
        varnet = args.varnet,
        noise = args.noise,
        lrelu = args.lrelu,
        sigmaW = args.sigmaW,
        writer = writer,
        init = args.init,
        device = ("cuda:%d" % int(args.device[0]))
    )
    
    draw(args,model)
    
    param_num = sum([p.data.nelement() for p in model.parameters()])
    
    print(f"Number of model parameters: {param_num}")

    if torch.cuda.device_count() > 1:
        
        start = int(args.device[0])
        end  = int(args.device[2])+1
        torch.cuda.set_device(start)
        dev_list=[]
        for i in range(start,end):
            dev_list.append("cuda:%d" % i)
        model = torch.nn.DataParallel(model, device_ids=dev_list)

    model = model.cuda()

    if args.freeze>0:
        cnt = 0
        for name,param in model.named_parameters():
            if intersection(['scale'],name.split('.')):
                cnt=cnt+1
                if cnt == args.freeze:
                    break

            if cnt >= args.freeze_start:
##                if intersection(['conv','conv1'],name.split('.')):
##                    print("Freezing Block: %s" % name.split('.')[1:3]  )
                if not intersection(['conv_res','fc'],name.split('.')):
                    param.requires_grad = False
                    print("Freezing Block: %s" % name)


    elif args.freeze < 0:
        cnt = 0
        for name,param in model.named_parameters():
            if intersection(['scale'],name.split('.')):
                cnt=cnt+1

            if cnt >  args.layers - 3 + args.freeze - 1:
##                if intersection(['conv','conv1'],name.split('.')):
##                    print("Freezing Block: %s" % name  )

                if not intersection(['conv_res','fc'],name.split('.')):
                    param.requires_grad = False
                    print("Freezing Block: %s" % name  )


    if args.res_freeze > 0:
        cnt = 0
        for name,param in model.named_parameters():
            if intersection(['conv_res'],name.split('.')):
                cnt=cnt+1
                if cnt > args.res_freeze_start:
                    param.requires_grad = False
                    print("Freezing Block: %s" % name)
                if cnt >= args.res_freeze:
                    break
    elif args.res_freeze < 0:
        cnt = 0
        for name,param in model.named_parameters():
            if intersection(['conv_res'],name.split('.')):
                cnt=cnt+1
                if cnt > 3 + args.res_freeze:
                    param.requires_grad = False
                    print("Freezing Block: %s" % name)


    if args.prune: 
        if  args.prune_epoch >= 100:
            weightsFile =  "runs/%s-net/checkpoint.pth.tar" % args.prune
        else:
            weightsFile =  "runs/%s-net/model_epoch_%d.pth.tar" % (args.prune, args.prune_epoch)

        if os.path.isfile(weightsFile):
            print(f"=> loading checkpoint {weightsFile}")
            checkpoint = torch.load(weightsFile)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"=> loaded checkpoint '{weightsFile}' (epoch {checkpoint['epoch']})")
        else:
            if args.prune_epoch == 0:
                print(f"=> No source data, Restarting network from scratch")
            else:
                print(f"=> no checkpoint found at {weightsFile}, aborting...")
                return None


    else:
        if args.resume:
            tarfile = "runs/%s-net/checkpoint.pth.tar" % args.resume 
            if os.path.isfile(tarfile):
                print(f"=> loading checkpoint {args.resume}")
                checkpoint = torch.load(tarfile)
                args.start_epoch = checkpoint["epoch"]
                best_prec1 = checkpoint["best_prec1"]
                model.load_state_dict(checkpoint["state_dict"])
                print(f"=> loaded checkpoint '{tarfile}' (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at {tarfile}, aborting...")
                return None

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer.lower() == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    if args.prune and pruner_state is not None:
        cutoff_retrain = prunhild.cutoff.LocalRatioCutoff(args.cutoff)
        params_retrain = get_params_for_pruning(args,model)
        pruner_retrain = prunhild.pruner.CutoffPruner(params_retrain, cutoff_retrain)
        pruner_retrain.load_state_dict(pruner_state) 
        pruner_retrain.prune(update_state=False)
        pruned_weights_count = count_pruned_weights(params_retrain, args.cutoff)
        params_left = param_num - pruned_weights_count
        print("Pruned %d weights, New model size:  %d/%d (%d%%)"  % (pruned_weights_count, params_left ,param_num, int(100*params_left/param_num) ))
        
    else:
        pruner_retrain = None

    if args.eval:
        best_prec1 = validate(args,val_loader, model, criterion, 0,None)
    else:

        if args.varnet:
            save_checkpoint(args,
                {
                    "epoch": 0,
                    "state_dict": model.state_dict(),
                    "best_prec1": 0.0,
                },
                True,
            )
            best_prec1 = 0.0


        turns_above_50=0

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(args,optimizer, epoch + 1)
            train(args,train_loader, model, criterion, optimizer, epoch, pruner_retrain, writer)

            prec1 = validate(args,val_loader, model, criterion, epoch,writer)
            correlation.measure_correlation(model, epoch, writer=writer)    

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if args.savenet:
                save_checkpoint(args,
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_prec1": best_prec1,
                    },
                    is_best,
                )

            if args.symmetry_break:
                if prec1 > 50.0:
                    turns_above_50+=1
                    if turns_above_50>3:
                        return epoch

    writer.close()

    print("Best accuracy: ", best_prec1)
    return best_prec1


def train(args,train_loader, model, criterion, optimizer, epoch, pruner, writer):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    

    model.train()
    total =0 
    correct = 0
    reg_loss = 0.0
    train_loss = 0.0
    end = time.time()

    for i, (inputs, target) in enumerate(train_loader):

        target = target.cuda()
        inputs = inputs.cuda()
        
        inputs, targets_a, targets_b, lam = mixup_data(inputs, target, args.alpha, True)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

        ##input_var = torch.autograd.Variable(input)
        ##target_var = torch.autograd.Variable(target)


        outputs = model(inputs)
        ##outputs, Qs, Ys = model(inputs)
        ##loss = criterion(output, target_var)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
##        print("loss:")
##        print(loss)
##        print(loss.item())
##        train_loss += loss.data[0]
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

##        prec1 = accuracy(output.data, target, topk=(1,))[0]
##        losses.update(loss.data.item(), input.size(0))
##        top1.update(prec1.item(), input.size(0))

        optimizer.zero_grad()



##        for y in Ys:
##            y.retain_grad()



        loss.backward()


        optimizer.step()




        if pruner is not None:
            pruner.prune(update_state=False)

        batch_time.update(time.time() - end)
        end = time.time()


        if 0:
            kwalt = epoch*len(train_loader)+i
            if writer is not None:
                for j,q in enumerate(Qs):
                    writer.add_scalar("variances %d" % j, q.cpu().numpy(), kwalt)

                for l,y in enumerate(Ys):
                    if y.grad is not None:
                        writer.add_scalar("grad %d" % (l-j), getQ(y.grad).cpu().numpy(), kwalt)

##            writer.add_scalars("variancess", { "%d"% j :  q.cpu().numpy() for j,q in enumerate(Qs)}, i)



        if 0:
            if i % args.print_freq == 0:
                print(
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})"
                )
            niter = epoch*len(train_loader)+i

    batch_idx = i
    if writer is not None:
        writer.add_scalar('Train/Loss', train_loss/batch_idx, epoch)
        writer.add_scalar('Train/Prec@1', 100.*correct/total, epoch)
##        writer.add_scalar('Train/RegLoss', reg_loss/batch_idx, niter)




def validate(args,val_loader, model, criterion, epoch, writer, quiet=False):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            ##output,Qs, Ys = model(input_var)
            output = model(input_var)
        loss = criterion(output, target_var)

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if 0:
                print(
                    f"Test: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})"
                )
            niter = epoch*len(val_loader)+i
            ##writer.add_scalar('Test/Loss', losses.val, niter)
            ##writer.add_scalar('Test/Prec@1', top1.val, niter)

    if writer is not None:
        writer.add_scalar('AvgTest/Prec@1', top1.avg, epoch)
        writer.add_scalar('AvgTest/Loss', losses.avg, epoch)
        if args.tensorboard:
            log_value("val_loss", losses.avg, epoch)
            log_value("val_acc", top1.avg, epoch)


    if not quiet:
        print(f" * Prec@1 {top1.avg:.3f}")

    return top1.avg


def save_checkpoint(args,state, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint to disk"""
    directory = "runs/%s-net/" % (args.name)

    if not os.path.exists(directory):
        os.makedirs(directory)


    epoch = state['epoch']

    filename = directory + filename
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, "runs/%s-net/" % (args.name) + "model_best.pth.tar")

    if epoch==0 or epoch==2:
        shutil.copyfile(filename, "runs/%s-net/" % (args.name) + "model_epoch_%d.pth.tar" % epoch )


def adjust_learning_rate(args,optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = args.lr * (
        (0.2 ** int(epoch >= 60))
        * (0.2 ** int(epoch >= 120))
        * (0.2 ** int(epoch >= 160))
    )

    ##lr = args.lr  ##DELETE ME!

    if args.tensorboard:
        log_value("learning_rate", lr, epoch)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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


if __name__ == "__main__":
    main()
