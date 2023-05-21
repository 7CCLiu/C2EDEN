# nc-experiment
from mpi4py import MPI
import numpy as np
from scipy.io import loadmat
import sklearn.datasets
import matplotlib.pyplot as plt
import time
import scipy.linalg
import scipy.special
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, linalg

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

def grad(dataset,idx, w, reg, gamma=1):
    t0=time.time()
    X=dataset.X[idx,:]
    Y=dataset.Y[idx,:]
    N=X.shape[0]
    pred = Y * (X @ w)
    p = 0.5 * (1 + np.tanh(-0.5 * pred))
    tg=time.time()-t0
    return -X.T @ (Y * p)/N + reg * w + reg *(2*w/gamma**2)/((1/gamma**2+w**2)**2)

def fvalue(dataset,idx,w,reg,gamma=1):
    X=dataset.X[idx,:]
    Y=dataset.Y[idx,:]
    N=X.shape[0]
    pred = Y * (X @ w)
    pos = np.sum(np.log(1+np.exp(-pred[pred>=0])))/N
    neg = np.sum(np.log(1+np.exp(pred[pred<0]))-pred[pred<0])/N
    return pos + neg + reg* np.sum(w**2/(1/gamma**2+w**2))

def Hessian(dataset, idx, w, reg, gamma=1):
    X=dataset.X[idx,:]
    Y=dataset.Y[idx,:]
    N=X.shape[0]
    d=X.shape[1]
    pred = Y * (X @ w)
    p = 0.5 * (1 + np.tanh(-0.5 * pred))
    return X.T @ (X * p * (1-p))/N + reg * (np.diag(np.reshape((-6*w**4/gamma**2-4*w**2/gamma*4+2/gamma**6)/((1/gamma**2+w**2)**4),-1)))


def Hes_Vector(dataset, idx, w, reg, u, gamma=1):
    X=dataset.X[idx,:]
    Y=dataset.Y[idx,:]
    N=X.shape[0]
    d=X.shape[1]      
    pred = Y * (X @ w)
    p = 0.5 * (1 + np.tanh(-0.5 * pred))
    return X.T @((X @ u * p * (1-p))/N)+ reg *( (-6*w**4/gamma**2-4*w**2/gamma*4+2/gamma**6)/((1/gamma**2+w**2)**4))*u
    

def fast_cubic_newton_step(r_ini, g, invU, Gamma, H, eps=1e-7):
    n = g.shape[0]
  
    def h(r, der=False):
        xxx=(1/(Gamma+H*r)).reshape(n,1)
        T= -invU.T@(xxx*(invU@g))
        T_norm = np.linalg.norm(T)
        h_r = r - T_norm
        if der:
            T1=invU.T@(xxx*(invU@T))
            h_r_prime = 1 + (H / T_norm) * (T1.reshape(-1)).dot(T.reshape(-1))
        else:    
            h_r_prime = None
        return h_r, T_norm, T, h_r_prime

        # Find max_r such that h(max_r) is nonnegative
    
    max_r = r_ini
    max_iters = 20
        # Find max_r such that h(max_r) is nonnegative
    for i in range(max_iters):
        h_r, T_norm, T, _ = h(max_r)
        if h_r < -eps:
            max_r *= 2
        elif -eps <= h_r <= eps:
            return T, h_r, max_r, "success"
        else:
            break
        
        # Univariate Newton's
    r = max_r
    for i in range(max_iters):
        h_r, T_norm, T, h_r_prime = h(r, der=True)
        if -eps <= h_r <= eps:
            return T, h_r, r, "success" 
        r -= h_r / h_r_prime
    return np.zeros([n,1]), 0.0, 0.0, "iterations_exceeded"

def cubic_newton_step(g, A, H,  B=None, eps=1e-8):
    n = g.shape[0]
    if B is None:
        B = np.eye(n)
        l2_norm_sqr = lambda x: x.dot(x)
    else:
        l2_norm_sqr = lambda x: B.dot(x).dot(x)

    def f(T, T_norm):
        return g.dot(T) + 0.5 * A.dot(T).dot(T) + H * T_norm ** 3 / 3.0
    
    def h(r, der=False):
        ArB_cho_factor = scipy.linalg.cho_factor(A + H * r * B, lower=False)
        T = scipy.linalg.cho_solve(ArB_cho_factor, -g)
        T_norm = l2_norm_sqr(T) ** 0.5
        h_r = r - T_norm
        if der:
            BT = B.dot(T)
            h_r_prime = 1 + H / T_norm * \
                        scipy.linalg.cho_solve(ArB_cho_factor, BT).dot(BT)
        else:
            h_r_prime = None
        return h_r, T_norm, T, h_r_prime

    try:
        max_r = 1.0
        max_iters = 50
        # Find max_r such that h(max_r) is nonnegative
        for i in range(max_iters):
            h_r, T_norm, T, _ = h(max_r)
            if h_r < -eps:
                max_r *= 2
            elif -eps <= h_r <= eps:
                return T, h_r, r, "success"
            else:
                break
        
        # Univariate Newton's
        r = max_r
        for i in range(max_iters):
            h_r, T_norm, T, h_r_prime = h(r, der=True)
            if -eps <= h_r <= eps:
                return T, h_r, r, "success" 
            r -= h_r / h_r_prime
    except (np.linalg.LinAlgError, ValueError) as e:
            return np.zeros([n,1]), 0.0, 0.0, "linalg_error"

    return np.zeros([n,1]), 0.0, 0.0, "iterations_exceeded"


filename="w8a.mat"
filetype=1
m=size # local client number 
data=Data(filename,filetype,m)
epochs=8000
reg=1e-6
eta=0.1
gamma=1
Hlip=1
Hcubic=1

if rank==0:
    # graddient 
    g_gd=[]
    g_cubic=[]
    g_cedin=[]
    
    # time 
    t_gd=[]
    t_cubic=[]
    t_cedin=[]
    
    # function value
    f_gd=[]
    f_cubic=[]
    f_cedin=[]


#GD
x=np.zeros([data.d,1])
if rank==0:
    ts=time.time()
for i in range(2*epochs):
    idx= np.arange(data.p)+data.p*rank
    gx=grad(data,idx,x,reg,gamma)
    g=comm.gather(gx,root=0)
    fx = fvalue(data,idx,x,reg,gamma)
    f=comm.gather(fx,root=0)
    if rank==0:
        gx=np.mean(g,0)
        x=x-eta*gx
        g_gd.append(np.linalg.norm(gx))
        f_gd.append(np.mean(f,0))
        t_gd.append(time.time()-ts)
    x=comm.bcast(x,root=0)

# Local-CUBIC
x=np.zeros([data.d,1])
if rank==0:
    ts=time.time()

for i in range(epochs):
    idx=np.arange(data.p)+data.p*rank
    gx=grad(data,idx,x,reg) 
    Hess=Hessian(data,idx,x,reg,gamma)
    T,_,_,_=cubic_newton_step(gx.reshape(-1),Hess,Hcubic)
    T=T.reshape(data.d,1)
    T=comm.gather(T,root=0)
    g=comm.gather(gx,root=0)
    fx = fvalue(data,idx,x,reg,gamma)
    f=comm.gather(fx,root=0)
    if rank==0:
        T=np.mean(T,0)
        gx=np.mean(g,0)
        g_cubic.append(np.linalg.norm(gx))
        f_cubic.append(np.mean(f,0))
        t_cubic.append(time.time()-ts)
        x=x+T
    x=comm.bcast(x,root=0)

#CEDIN
x=np.zeros([data.d,1])
if rank== 0:
    ts=time.time()
    r_ini=1
u=x.copy()
if rank==0:
    H0=np.zeros([data.d,data.d])
    H1=np.zeros([data.d,data.d])
    H0inv=np.zeros([data.d,data.d])
for i in range(epochs):
    idx= np.arange(data.p)+data.p*rank
    gx=grad(data,idx,x,reg,gamma) 
    e= np.zeros([data.d,1])
    e[i%data.d]=1
    Hv=Hes_Vector(data,idx,u,reg,e,gamma)
    g=comm.gather(gx,root=0)
    Hv=comm.gather(Hv,root=0)
    fx = fvalue(data,idx,x,reg,gamma)
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
        if rank==0:
            if i%data.d == 0:
                H0=H1.copy()
             #   u=x.copy()
                Gamma, U= np.linalg.eigh(H0)
                invU=U.T        
            gx=np.mean(g,0)
            Hv=np.mean(Hv,0)
            H1[i%data.d,:]=Hv.reshape(-1)
            T, _, r_ini,_ =fast_cubic_newton_step(r_ini,gx,invU,Gamma,Hlip)
            g_cedin.append(np.linalg.norm(gx))
            f_cedin.append(np.mean(f,0))
            t_cedin.append(time.time()-ts)   
            x=x+T  
        x=comm.bcast(x,root=0)      
        if i%data.d==0:
            u=x.copy()

if rank == 0:


    f_star = f_cedin[7500]

    for i in range(len(f_gd)):
        f_gd[i] = f_gd[i] - f_star
    for i in range(len(f_cubic)):
        f_cubic[i] = f_cubic[i] - f_star
    for i in range(len(f_cedin)):
        f_cedin[i] = f_cedin[i] - f_star

    for i in range(len(t_gd)):
        t_gd[i] = t_gd[i] - t_gd[0]
    for i in range(len(t_cubic)):
        t_cubic[i] = t_cubic[i] - t_cubic[0]
    for i in range(len(t_cedin)):
        t_cedin[i] = t_cedin[i] - t_cedin[0]
        
    end = 7900

    plt.rc('font', size=21)
    plt.figure()
    plt.grid()
    plt.yscale('log')
    plt.plot(g_gd, '-.b', label = 'GD', linewidth = 3)
    plt.plot(g_cubic, ':r', label = 'LCRN', linewidth = 3)
    plt.plot(g_cedin[:end], '-k', label = 'C2EDEN',  linewidth=3)
    plt.xlim((0, len(g_cedin)))
    plt.ylim(bottom=1e-11)
    plt.legend(fontsize=23,loc='lower left')
    plt.tick_params('x',labelsize=21)
    plt.tick_params('y',labelsize=21)
    plt.tight_layout()
    plt.savefig('img/w8a.epoch_grad.nc16.png')
    plt.savefig('img/w8a.epoch_grad.nc16.svg', format = 'svg', transparent=True)

    plt.rc('font', size=21)
    plt.figure()
    plt.grid()
    plt.yscale('log')
    plt.plot(f_gd , '-.b', label = 'GD', linewidth = 3)
    plt.plot(f_cubic , ':r', label = 'LCRN', linewidth = 3)
    plt.plot(f_cedin[:end] , '-k', label = 'C2EDEN',  linewidth=3)
    plt.xlim((0, len(f_cedin)))
    plt.ylim(bottom=1e-11)
    plt.legend(fontsize=23,loc='lower left')
    plt.tick_params('x',labelsize=21)
    plt.tick_params('y',labelsize=21)
    plt.tight_layout()
    plt.savefig('img/w8a.epoch_func.nc16.png')
    plt.savefig('img/w8a.epoch_func.nc16.svg', format = 'svg', transparent=True)

    plt.rc('font', size=21)
    plt.figure()
    plt.grid()
    plt.yscale('log')
    plt.plot(t_gd, g_gd, '-.b', label = 'GD', linewidth = 3)
    plt.plot(t_cubic, g_cubic, ':r', label = 'LCRN', linewidth = 3)
    plt.plot(t_cedin[:end], g_cedin[:end], '-k', label = 'C2EDEN',  linewidth=3)
    plt.xlim((0, t_cedin[-1]))
    plt.ylim(bottom=1e-11)
    plt.legend(fontsize=23,loc='lower left')
    plt.tick_params('x',labelsize=21)
    plt.tick_params('y',labelsize=21)
    plt.tight_layout()
    plt.savefig('img/w8a.time_grad.nc16.png')
    plt.savefig('img/w8a.time_grad.nc16.svg', format = 'svg', transparent=True)

    plt.rc('font', size=21)
    plt.figure()
    plt.grid()
    plt.yscale('log')
    plt.plot(t_gd, f_gd , '-.b', label = 'GD', linewidth = 3)
    plt.plot(t_cubic, f_cubic , ':r', label = 'LCRN', linewidth = 3)
    plt.plot(t_cedin[:end], f_cedin[:end] , '-k', label = 'C2EDEN',  linewidth=3)
    plt.xlim((0, t_cedin[-1]))
    plt.ylim(bottom=1e-11)
    plt.legend(fontsize=23,loc='lower left')
    plt.tick_params('x',labelsize=21)
    plt.tick_params('y',labelsize=21)
    plt.tight_layout()
    plt.savefig('img/w8a.time_func.nc16.png')
    plt.savefig('img/w8a.time_func.nc16.svg', format = 'svg', transparent=True)




