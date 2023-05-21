from mpi4py import MPI
import numpy as np
from scipy.io import loadmat
import sklearn.datasets
import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# passing MPI datatypes explicitly


class Data:
    def __init__(self, filename, filetype, size):
        if filetype==1:
            m=loadmat(filename)
            self.X= np.array(m['A']).astype("float")
            self.Y= np.array(m['b']).astype("float")
            self.d = self.X.shape[1]
            n=self.X.shape[0]
            self.p=n//size
            self.n=self.p*size
            self.X=self.X[:self.n,:]
            self.Y=self.Y[:self.n,:]
        
        else:
            Sparse=sklearn.datasets.load_svmlight_file(filename)[0]
            target=sklearn.datasets.load_svmlight_file(filename)[1]
            self.X=Sparse.toarray().astype("float")
            self.Y=target.reshape([np.shape(target)[0],1]).astype("float")
            self.d =self.X.shape[1]
            n=self.X.shape[0]
            self.p=n//size
            self.n=self.p*size
            self.X=self.X[:self.n,:]
            self.Y=self.Y[:self.n,:]

def grad(dataset,idx, w, reg):
    X=dataset.X[idx,:]
    Y=dataset.Y[idx,:]
    N=X.shape[0]
    pred = Y * (X @ w)
    p = 0.5 * (1 + np.tanh(-0.5 * pred))
    return -X.T @ (Y * p)/N + reg * w


def Hessian(dataset, idx, w, reg):
    X=dataset.X[idx,:]
    Y=dataset.Y[idx,:]
    N=X.shape[0]
    d=X.shape[1]
    pred = Y * (X @ w)
    p = 0.5 * (1 + np.tanh(-0.5 * pred))
    return X.T @ (X * p * (1-p))/N + reg * np.eye(d)

def Hes_Vector(dataset, idx, w, reg, u):
    X=dataset.X[idx,:]
    Y=dataset.Y[idx,:]
    N=X.shape[0]
    d=X.shape[1]      
    pred = Y * (X @ w)
    p = 0.5 * (1 + np.tanh(-0.5 * pred))
    return X.T @((X @ u * p * (1-p))/N)+ reg * u


def fvalue(dataset,idx,w,reg):
    X=dataset.X[idx,:]
    Y=dataset.Y[idx,:]
    N=X.shape[0]
    pred = Y * (X @ w)
    pos = np.sum(np.log(1+np.exp(-pred[pred>=0])))/N
    neg = np.sum(np.log(1+np.exp(pred[pred<0]))-pred[pred<0])/N
    return pos + neg + 0.5 * reg* (w.T @ w)[0, 0]

filename="splice.txt"
filetype=0
m=size # local client number 
data=Data(filename,filetype,m)
epochs=2000
reg=1e-6
eta=0.01
eta_agd=0.1
eta_giant=1e-3
gamma=0.9

if rank==0:
    f_cedin = []
    g_cedin = []
    t_cedin = []

    f_agd = []
    g_agd = []
    t_agd = []

    f_giant = []
    g_giant = []
    t_giant = []

#AGD
x=np.zeros([data.d,1])
y=np.zeros([data.d,1])
x_=x.copy()
if rank==0:
    ts=time.time()
    y=x+gamma*(x-x_)
y=comm.bcast(y,root=0)
for i in range(4*epochs):
    idx= np.arange(data.p)+data.p*rank
    gy=grad(data,idx,y,reg)
    g=comm.gather(gy,root=0)
    fx = fvalue(data,idx,y,reg)
    f=comm.gather(fx,root=0)
    if rank==0:
        gy=np.mean(g,0)
        x_=x.copy()
        x=y-eta*gy
        g_agd.append(np.linalg.norm(gy))
        f_agd.append(np.mean(f,0))
        t_agd.append(time.time()-ts)
        y=x+gamma*(x-x_)
    y=comm.bcast(y,root=0)
    

# GIANT
x=np.zeros([data.d,1])
if rank== 0:
    ts=time.time()
for i in range(epochs):
    idx= np.arange(data.p)+data.p*rank
    gx=grad(data,idx,x,reg)
    g=comm.gather(gx,root=0)
    fx = fvalue(data,idx,x,reg)
    f=comm.gather(fx,root=0)
    if rank==0:
        gx=np.mean(g,0)
        g_giant.append(np.linalg.norm(gx))
        f_giant.append(np.mean(f,0))
        t_giant.append(time.time()-ts)
    gx=comm.bcast(gx,root=0)
    Hlocal=Hessian(data,idx,x,reg)
    d=-np.linalg.inv(Hlocal)@gx
    d=comm.gather(d,root=0)
    if rank==0:
        d= np.mean(d,0)
        x=x+eta_giant*d
    x = comm.bcast(x,root=0)

#CEDIN
x=np.zeros([data.d,1])
if rank== 0:
    ts=time.time()
u=x.copy()
if rank==0:
    H0=np.zeros([data.d,data.d])
    H1=np.zeros([data.d,data.d])
    H0inv=np.zeros([data.d,data.d])
for i in range(epochs):
    idx= np.arange(data.p)+data.p*rank
    gx=grad(data,idx,x,reg) 
    e= np.zeros([data.d,1])
    e[i%data.d]=1
    Hv=Hes_Vector(data,idx,u,reg,e)
    g=comm.gather(gx,root=0)
    Hv=comm.gather(Hv,root=0)
    fx = fvalue(data,idx,x,reg)
    f=comm.gather(fx,root=0)
    if i<data.d:
        if rank==0:
            gx=np.mean(g,0)
            Hv=np.mean(Hv,0)
            H1[i,:]=Hv.reshape(-1)
            x=x-eta*gx
            g_cedin.append(np.linalg.norm(gx))
            f_cedin.append(np.mean(f,0))
            t_cedin.append(time.time()-ts)
        x = comm.bcast(x,root=0)
    elif i>=data.d:
        if i%data.d==0:  
            u=x.copy()
            if rank==0:
                H0=H1.copy()
                H0inv=np.linalg.inv(H0)
                gx=np.mean(g,0)
                Hv=np.mean(Hv,0)
                H1[i%data.d,:]=Hv.reshape(-1)
                x=x-H0inv@gx
                g_cedin.append(np.linalg.norm(gx))
                f_cedin.append(np.mean(f,0))
                t_cedin.append(time.time()-ts)
            x=comm.bcast(x,root=0)
        else: 
            if rank==0:
                gx=np.mean(g,0)
                Hv=np.mean(Hv,0)
                H1[i%data.d,:]=Hv.reshape(-1)
                x=x-H0inv@gx
                g_cedin.append(np.linalg.norm(gx))
                f_cedin.append(np.mean(f,0))
                t_cedin.append(time.time()-ts)
            x=comm.bcast(x,root=0)
    


if rank == 0:


    for i in range(len(t_agd)):
        t_agd[i] = t_agd[i] - t_agd[0]
    for i in range(len(t_giant)):
        t_giant[i] = t_giant[i] - t_giant[0]
    for i in range(len(t_cedin)):
        t_cedin[i] = t_cedin[i] - t_cedin[0]

    f_star = min(f_agd[-1], min(f_giant[-1], f_cedin[-1] ) )

    for i in range(len(f_agd)):
        f_agd[i] = f_agd[i] - f_star
    for i in range(len(f_giant)):
        f_giant[i] = f_giant[i] - f_star
    min_fcedin = f_cedin[0]
    for i in range(len(f_cedin)):
        if f_cedin[i] < min_fcedin:
            min_fcedin = f_cedin[i]
        f_cedin[i] = min_fcedin - f_star
    end = 500

    plt.rc('font', size=21)
    plt.figure()
    plt.grid()
    plt.yscale('log')
    plt.plot(g_agd, '-.b', label = 'AGD', linewidth = 3)
    plt.plot(g_giant, ':r', label = 'GIANT', linewidth = 3)
    plt.plot(g_cedin[:end], '-k', label = 'C2EDEN',  linewidth=3)
    plt.xlim((0, len(g_cedin)))
    plt.ylim(bottom=1e-14)
    plt.legend(fontsize=23,loc='lower right')
    plt.tick_params('x',labelsize=21)
    plt.tick_params('y',labelsize=21)
    plt.tight_layout()
    plt.savefig('img/splice.epoch_grad.sc16.png')
    plt.savefig('img/splice.epoch_grad.sc16.svg', format = 'svg', transparent=True)

    plt.rc('font', size=21)
    plt.figure()
    plt.grid()
    plt.yscale('log')
    plt.plot(f_agd , '-.b', label = 'AGD', linewidth = 3)
    plt.plot(f_giant , ':r', label = 'GIANT', linewidth = 3)
    plt.plot(f_cedin[:end] , '-k', label = 'C2EDEN',  linewidth=3)
    plt.xlim((0, len(f_cedin)))
    plt.ylim(bottom=1e-14)
    plt.legend(fontsize=23,loc='lower right')
    plt.tick_params('x',labelsize=21)
    plt.tick_params('y',labelsize=21)
    plt.tight_layout()
    plt.savefig('img/splice.epoch_func.sc16.png')
    plt.savefig('img/splice.epoch_func.sc16.svg', format = 'svg', transparent=True)

    plt.figure()
    plt.grid()
    plt.yscale('log')
    plt.plot(t_agd, g_agd, '-.b', label = 'AGD', linewidth = 3)
    plt.plot(t_giant, g_giant, ':r', label = 'GIANT', linewidth = 3)
    plt.plot(t_cedin[:end], g_cedin[:end], '-k', label = 'C2EDEN',  linewidth=3)
    plt.xlim((0, 0.9))
    plt.ylim(bottom=1e-14)
    plt.legend(fontsize=23,loc='lower right')
    plt.tick_params('x',labelsize=21)
    plt.tick_params('y',labelsize=21)
    plt.tight_layout()
    plt.savefig('img/splice.time_grad.sc16.png')
    plt.savefig('img/splice.time_grad.sc16.svg', format = 'svg', transparent=True)

    plt.rc('font', size=21)
    plt.figure()
    plt.grid()
    plt.yscale('log')
    plt.plot(t_agd, f_agd , '-.b', label = 'AGD', linewidth = 3)
    plt.plot(t_giant, f_giant , ':r', label = 'GIANT', linewidth = 3)
    plt.plot(t_cedin[:end], f_cedin[:end] , '-k', label = 'C2EDEN',  linewidth=3)
    plt.xlim((0, 0.9))
    plt.ylim(bottom=1e-14)
    plt.legend(fontsize=23,loc='lower right')
    plt.tick_params('x',labelsize=21)
    plt.tick_params('y',labelsize=21)
    plt.tight_layout()
    plt.savefig('img/splice.time_func.sc16.png')
    plt.savefig('img/splice.time_func.sc16.svg', format = 'svg', transparent=True)

# run the code with "mpiexec -n 16 python xxx.py"