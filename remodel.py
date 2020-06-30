"""
Wide ResNet by Sergey Zagoruyko and Nikos Komodakis
Fixup initialization by Hongyi Zhang, Yann N. Dauphin, Tengyu Ma
Based on code by xternalz and Andy Brock:
https://github.com/xternalz/WideResNet-pytorch
https://github.com/ajbrock/BoilerPlate
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def getQ(T):
    t = T.detach()
    s = t.shape
    a = t - torch.mean(t,[1]).view(s[0],1,s[2],s[3]).expand_as(t)
    return torch.mean(a*a), torch.mean(t*t)


class BasicBlock(nn.Module):
    droprate = 0.0
    use_bn = True
    use_fixup = False
    fixup_l = 12
    const = 1.0
    lrelu = 0.0
    init = []
    def __init__(self, in_planes, out_planes, stride,i):

        super(BasicBlock, self).__init__()
        if i % 2:
            phase = -1
        else:
            phase = 1

        self.i = i
        ##print("Use fixup:")
        ##print(self.use_fixup)
        
        self.bn = nn.BatchNorm2d(out_planes)

        if self.lrelu ==0.0:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(self.lrelu)

        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.equalInOut = in_planes == out_planes
        assert (
            self.use_fixup or self.use_bn
        ), "Need to use at least one thing: Fixup or BatchNorm"

        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(2)]
        )

        k = (
            self.conv.kernel_size[0]
            * self.conv.kernel_size[1]
            * self.conv.out_channels
        )


        ConstAny(self.conv.weight, self.conv.bias, self.init[int(i)],  phase, self.lrelu)

        self.phase = phase

    def forward(self, x):

        out = self.relu(self.conv(x + self.biases[0]))
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        if self.use_bn:
            if self.phase < 0:
                out = self.bn(out)

        ##self.step = self.step + 1
        ##print("%d %d "%  (self.step, self.i))
        ##self.writer.add_scalar('Variance_%d' %  self.i , getQ(out) , self.step) 
        return out

        ##y = torch.autograd.Variable(out, requires_grad=True)
        ##y.retain_grad()
        ##y.requires_grad_()
        ##return y, getQ(out) 


class NetworkBlock2(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride,pos,dev):
        super(NetworkBlock, self).__init__()
        ##self.layer = 
        self._make_layer(block, in_planes, out_planes, nb_layers, stride, pos,dev)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,pos,dev):
        layers = []
        print("-------")
        for i in range(int(nb_layers)):
            _in_planes = i == 0 and in_planes or out_planes
            _stride = i == 0 and stride or 1
            layers.append(block(_in_planes, out_planes, _stride,i+pos).cuda(torch.device(dev)))

        ##return nn.Sequential(*layers)

        self.layers = layers

    def forward(self, x):
        out = x
        outs = []
        ys = []
        for op in self.layers:
            ##print(x)
            out, q = op(out)
            ##print(q)
            outs.append(q)
            ys.append(out)
        return out, outs, ys


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride,pos,dev):
        super(NetworkBlock, self).__init__()
        self.layer =  self._make_layer(block, in_planes, out_planes, nb_layers, stride, pos, dev)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride,pos,dev):
        layers = []
        print("-------------")
        for i in range(int(nb_layers)):
            _in_planes = i == 0 and in_planes or out_planes
            _stride = i == 0 and stride or 1
            layers.append(block(_in_planes, out_planes, _stride,i+pos))

        return nn.Sequential(*layers)

        ##self.layers = layers

    def forward(self, x):
        return self.layer(x)


class LRNet(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        widen_factor=1,
        droprate=0.0,
        use_fixup=False,
        use_bn=True,
        varnet=False,
        noise=0.0,
        lrelu=0.0,
        sigmaW=1.0,
        writer = None,
        init = "1_iiii_iiii_iiii",
        device = "cpu",
    ):
        super(LRNet, self).__init__()
        if varnet:
            noise=1.0

        print("Using LRNet")

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

##        nChannels = [widen_factor, widen_factor , widen_factor , widen_factor]


        assert (depth - 4) % 3 == 0, "You need to change the number of layers"
        n = (depth - 4) / 3



        initial = []
        init = init.strip('_')
        init = init.replace("_","")
        for k in range(1,len(init)):
            initial.append(init[k])
        print(initial)
        BasicBlock.droprate = droprate
        BasicBlock.use_bn = use_bn
        BasicBlock.fixup_l = n * 3
        BasicBlock.use_fixup = use_fixup
        BasicBlock.const = 1.0-noise
        BasicBlock.init = initial
        BasicBlock.lrelu = lrelu

        ##print("Use fixup WideResnet:")
        ##print(use_fixup)
        ##print("Use BN WideResnet:")
        ##print(use_bn)
        block = BasicBlock

        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        k = (
            self.conv1.kernel_size[0]
            * self.conv1.kernel_size[1]
            * self.conv1.out_channels
        )
        
##        ConstDeltaOrthogonal(self.conv1.weight, self.conv1.bias, torch.nn.init.calculate_gain('relu'), 1.0 - noise)


        ConstAny(self.conv1.weight, self.conv1.bias, init[0],  1, lrelu)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1,0,device)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,n,device)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,2*n,device)

        if lrelu ==0.0:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = nn.LeakyReLU(lrelu)


        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        if noise > 0.0:
            self.fc.weight.data.normal_(0, math.sqrt(noise / k))
        else:
            self.fc.weight.data.zero_()
        
        self.fc.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        return self.fc(out)




    def forward2(self, x):

        ys = []
        
        out = self.conv1(x)
        
        l = [getQ(out)]

        out, Qs0 ,y  = self.block1(out)

        ys  = ys + y
        l = l + Qs0 
              
        out, Qs1 ,y = self.block2(out)
        
        ys  = ys + y
        l = l + Qs1       
        out, Qs2 ,y = self.block3(out)
        
        ys  = ys + y
        l = l + Qs2      
        out = self.relu(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        return self.fc(out) , l, ys




    def misc(self,x):
        h1 = self.conv1(x)
        h2 = self.block1(h1)
        h3 = self.block2(h2)
        h4 = self.block3(h3)
        h45 = self.relu(h4)
        h5 = F.avg_pool2d(h45, 8)
        h6 = h5.view(-1, self.nChannels)
        return self.fc(h6), h1, h2, h3, h4, h5, h6


def ConstAny(weight ,bias, init,  phase, lrelu):
    gain =  torch.nn.init.calculate_gain('relu')*(1.0+lrelu)
    if  init != "i" and init !="k":
        if init == "1":
            const1 =  1.0 
            print("Average")
        elif init[0] == 'h':
            const1 =  0.0
            print("Random")
        else:
            const1 =  0.99
            print("Noisy Average")

        ConstAvg(weight, bias, gain, const1, phase , lrelu)
    else:
        if init =="i":
            const1 =  1.0 
            print("Identity")
        else:
            const1 = 0.99
            print("Noisy Identity")
        ConstIdentity(weight, bias, gain , const1, phase, lrelu)

def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q

def makeLambdaDeltaOrthogonal(weights, bias, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if bias is not None:
        nn.init.constant_(bias, 0)
    dim = max(rows, cols)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    nn.init.constant_(weights, 0)
    weights.data[:, :, mid1, mid2] = gain ##torch.ones([rows, cols]) * gain

def ConstIdentity(weights, bias, gain, const = 1.0, phase = 1, lrelu = 0.0):

    rows = weights.size(0)
    cols = weights.size(1)

    assert(rows >= cols)
    ##assert(rows % cols == 0)


 

    if const < 1.0:
        k = cols * weights.size(2) * weights.size(3)
        weights.data.normal_(0, math.sqrt(gain*(1.0-const) / k))
    else: 
        nn.init.constant_(weights, 0)

    if bias is not None:
        nn.init.constant_(bias, 0)

    if const > 0.0:

        mid1 = weights.size(2) // 2
        mid2 = weights.size(3) // 2
        factor = phase*(1.0)*const

        if phase == -1 and lrelu > 0.0:
            factor = factor/lrelu

        for d0 in range(rows):
            weights.data[d0, d0 % cols, mid1, mid2] += factor 







def ConstAvg(weights, bias, gain, const = 1.0, phase = 1, lrelu = 0.0):
    rows = weights.size(0)
    cols = weights.size(1)

    if const < 1.0:
        k = cols * weights.size(2) * weights.size(3)
        weights.data.normal_(0, math.sqrt(gain*(1.0-const) / k))
    else: 
        nn.init.constant_(weights, 0)

    if bias is not None:
        nn.init.constant_(bias, 0)

    if const > 0.0:

        mid1 = weights.size(2) // 2
        mid2 = weights.size(3) // 2
        factor = phase*(1.0)*const/cols

        if phase == -1 and lrelu > 0.0:
            factor = factor/lrelu

        weights.data[:, :, mid1, mid2] += factor 


def ConstDeltaOrthogonal(weights, bias, gain, const = 1.0):

    rows = weights.size(0)
    cols = weights.size(1)
 
    if const < 1.0:
        k = cols * weights.size(2) * weights.size(3)
        weights.data.normal_(0, math.sqrt(gain*(1.0-const) / k))

    else: 
        nn.init.constant_(weights, 0)


    if bias is not None:
        nn.init.constant_(bias, 0)

    if const > 0.0:
        mid1 = weights.size(2) // 2
        mid2 = weights.size(3) // 2
        weights.data[:, :, mid1, mid2] = gain/cols ##torch.ones([rows, cols]) * gain      
