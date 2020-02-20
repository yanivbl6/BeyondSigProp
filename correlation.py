import torch
import numpy as np


def make_list(model):
    list=[model.module.conv1, model.module.fc, None, None]
    for k in range(len(model.module.block1.layer)):
        list.append(model.module.block1.layer[k].conv)

    for k in range(len(model.module.block2.layer)):
        list.append(model.module.block2.layer[k].conv)

    for k in range(len(model.module.block3.layer)):
        list.append(model.module.block3.layer[k].conv)
        
    try:
        list.extend([model.module.block1.layer[0].conv_res,model.module.block2.layer[0].conv_res,model.module.block3.layer[0].conv_res, None])
    except:
        misc = 0
        
    return list


def pair_correllation(m, fanin = True, fanout = True, maxN = 5000):
    N= maxN
    A = m.numpy()

    tot = 1
    for k in range(len(A.shape)):
        tot = tot * A.shape[k]

    if len(A.shape)>2:
        if fanin and fanout:
            n = A.shape[0]*A.shape[1]
        else:
            if not fanin:
                A = np.swapaxes(A, 0,1)
            n = A.shape[0]
        A = A.reshape([n,tot//n])
    else:
        if fanin and fanout:
            n = tot
        else:
            if not fanin:
                A = np.swapaxes(A, 0,1)
            n = A.shape[0]
        A = A.reshape([n,tot//n])

        
    n2 = tot//n
    if n > N:
        A = A[np.random.permutation(n)[0:N],:]
        n = N
    B = np.corrcoef(A)
    B[np.isnan(B)]=1.0
    corr = np.sum(np.tril(B,-1))/(((n-1)*n)/2)

    return corr

def logc(x):
    return -np.log(np.abs(x))

def sinc(x):
    return np.sqrt(1-x**2)

def measure_correlation(model, epoch, N = 5000, writer = None):
    list = make_list(model)
 
    res = {} 

    diversity = 0.0
    for k,m in enumerate(list):
        if m is None:
            continue
        
        mm = m.weight.data
        
        if len(m.weight.data.shape)>=2:

##            corrs =  pair_correllation(mm.cpu(),fanin=True,fanout=True, maxN = N)
            forward = pair_correllation(mm.cpu(),fanin=False,fanout=True, maxN = N)
            backward = pair_correllation(mm.cpu(),fanin=True,fanout=False, maxN = N)

            name = "%d) %s" % (k,m)
            name = name.replace(" ",'')
            name=  name[0:31] + ')'
         
            if writer is not None:
##                writer.add_scalar('%s/correlations' % name, logc(corrs), epoch)
                writer.add_scalar('%s/forward' % name, logc(forward), epoch)
                writer.add_scalar('%s/backward' % name, logc(backward), epoch)
      


            ##print(np.log( sinc(backward))) 
            diversity = diversity + np.log(sinc(backward))
            

            res["%s" % m] =  (corrs, forward, backward)
    ##print(diversity)

    writer.add_scalar('Diversity' , diversity, epoch)

    return res

def cross_correllation(m1, m2 , forward = True):
    A = m1.numpy()
    B = m2.numpy()
    Atot = 1
    Btot = 1
    for k in range(len(A.shape)):
        Atot = Atot * A.shape[k]
        
    for k in range(len(B.shape)):
        Btot = Btot * B.shape[k]
        
    if not forward:
        A = np.swapaxes(A, 0,1)
        B = np.swapaxes(B, 0,1)
        
    n1 = A.shape[0]
    n2 = B.shape[0]

    A = A.reshape([n1,Atot//n1])
    B = B.reshape([n2,Btot//n2])

    if A.shape[1] != B.shape[1]:
        return np.nan
    
    Bt = np.transpose(B)

    AA = np.linalg.norm(A,axis =1)
    BB = np.linalg.norm(B,axis =1)

    AA = (np.asmatrix(AA))
    BB = (np.asmatrix(BB))
    
    norm = (np.transpose(AA)*BB)
    ##print(norm)
    
    AB = np.matmul(A,Bt)
    ##print(AB)
    C = AB/norm
    C[np.isnan(C)]=1.0
    
    return np.mean(C)

def measure_cross_correlation(model, results, forward = True,list= None):
    
    if list is None:
        ##list=[model.module.block1.layer[0].conv_res]
        list=[]
        for k in range(len(model.module.block1.layer)):
            list.append(model.module.block1.layer[k].conv)
        
    for k,m1 in enumerate(list):
        for L,m2 in enumerate(list):
            ##if m1 is None or m2 is None:
            ##    continue

            mm1 = m1.weight.data
            mm2 = m2.weight.data
            
            if (k,L) not in results:
                results[(k,L)] = []
            res= results[(k,L)]
            
            res.append(cross_correllation(mm1.cpu(), mm2.cpu(),forward = forward))
        
    return results




