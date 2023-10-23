import os
import numpy as np
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.datasets import make_spd_matrix
import tarfile
from scipy.optimize import fmin_bfgs as bfgs
from scipy.stats import ortho_group
import time
import pandas as pd









class SyntheticBilevelMultiobjProblem:
    """
    Class used to define the following synthetic quadratic bilevel problem:
        min_{x} f_u(x,y) =  a'x + b'y + 0.5*x'H1 y + 0.5 x'H2 x
        s.t. y = argmin_{y} F_l(x,y)
        
        where F_l(x,y) can be:
            - SP1
            - JOS1
            - GKV1

    Attributes
        x_dim:                        Dimension of the upper-level problem
        y_dim:                        Dimension of the lower-level problem 
        num_obj:                      Number of objective functions
        std_dev:                      Standard deviation of the stochastic gradient and Hessian estimates 
        ul_prob:                      Upper-level problem (Options: Quadratic1)
        ll_prob:                      Lower-level problem (Options: SP1)
        seed (int, optional):         The seed used for the experiments (default 42)
        Q:                            Orthogonal matrix of dimension m \times m
    """
    
    def __init__(self, x_dim, y_dim, num_obj, std_dev, hess_std_dev, seed=42):
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_obj = num_obj
        self.std_dev = std_dev
        self.hess_std_dev = hess_std_dev
        self.seed = seed
        
        self.set_seed(self.seed)
        
        self.Q = np.eye(self.y_dim,self.y_dim) 
        # self.Q = ortho_group.rvs(dim=self.y_dim)
        
        self.ul_prob = Quadratic1(self.x_dim, self.y_dim, self.std_dev)
        # self.ll_prob = SP1(self.x_dim, self.y_dim, self.std_dev, self.Q)
        # self.ll_prob = JOS1(self.x_dim, self.y_dim, self.std_dev, self.Q)
        self.ll_prob = GKV1(self.x_dim, self.y_dim, self.std_dev, self.hess_std_dev)
        self.y_opt_func = self.ll_prob.y_opt_func
        
        self.projection = True
        self.ub = np.infty
        self.lb = 0
        
    def set_seed(self, seed):
        """
        Sets the seed
    	"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)
        

    def f(self, x):
        """
        The true objective function of the optimistic formulation
     	"""
        lam = x[:self.num_obj]
        xx = x[self.num_obj:]
        # y_opt = ((lam[0]+lam[1])/(lam[0]+2*lam[1]))*xx + ((3*lam[1])/(lam[0]+2*lam[1]))
        y_opt = self.ll_prob.y_opt_func(lam, xx)
        return self.f_u(x, y_opt)
    
    
    def f_risk_neutral(self, x, simplex_set_discr):
        """
        The true objective function of the risk-neutral formulation
        
        Args:
           x                      Upper-level variables
           simplex_set_discr      Matrix with columns representing vectors of weights of dim self.num_obj in the simplex set 
    	"""       
        out = 0
        # Iterate through the columns
        for lam in simplex_set_discr.T:
            lam_x = np.concatenate((lam.reshape(-1,1), x), axis=0)
            out = out + self.f(lam_x)
        out = out/simplex_set_discr.shape[1]    
        return np.squeeze(out)
    
    
    def f_risk_averse(self, x, y):
        """
        The objective function of the inner problem in the risk-averse formulation.
        
        Args:
            x                        Upper-level variables
            y   y[:self.num_obj] --> Weights in the simplex set for lower-level objective functions
                y[self.num_obj:] --> Lower-level variables
     	""" 
        lam = y[:self.num_obj]
        y_opt = self.ll_prob.y_opt_func(lam, x)
        out = self.ul_prob.f_u_(x, y_opt)    
        return np.squeeze(out)
        
    
    # def f_opt(self):
    #     """
    #     The optimal value of the bilevel problem
    #  	"""
    #     invH3 = np.linalg.inv(self.H3)
    #     aux = np.dot(self.H1,np.dot(invH3,self.H4)) + 2*self.H2 + np.dot(self.H4,np.dot(invH3.T,self.H1.T))
    #     x_opt = -2*np.dot(np.linalg.inv(aux),self.a + np.dot(self.H4,np.dot(invH3.T,self.b)))
    #     return self.f(x_opt)
   
    
    def f_u(self, x, y):
        """
        The upper-level objective function in the optimistic formulation
        
        Args:
            x   x[:,self.num_obj] --> Weights in the simplex set for lower-level objective functions
                x[self.num_obj,:] --> Upper-level variables
            y                         Lower-level variables
    	"""
        lam = x[:self.num_obj]
        xx = x[self.num_obj:]
        out = self.ul_prob.f_u_(xx, y)
        return np.squeeze(out)
    

    def f_u_risk_neutral(self, x, simplex_set_discr, y_array):
        """
        The objective function of the risk-neutral formulation
        
        Args:
           simplex_set_discr      Matrix with columns representing vectors of weights of dim self.num_obj in the simplex set 
           y_array                Matrix with columns representing vectors of lower-level variables, each associated with the 
                                  corresponding column of weights in the matrix simplex_set_discr
    	"""       
        out = 0
        # Iterate through the columns
        for count, y in enumerate(y_array.T):
            lam_x = np.concatenate((simplex_set_discr[:,count].reshape(-1,1), x), axis=0)
            out = out + self.f_u(lam_x, y.T)
        out = out/y_array.shape[1]    
        return np.squeeze(out)
    
    
    def f_u_risk_averse(self, x, y):
        """
        The objective function of the inner problem in the risk-averse formulation.
        
        Args:
            x                         Upper-level variables
            y   y[:,self.num_obj] --> Weights in the simplex set for lower-level objective functions
                y[self.num_obj,:] --> Lower-level variables
    	""" 
        lam = y[:self.num_obj]
        yy = y[self.num_obj:]
        out = self.ul_prob.f_u_(x, yy)    
        return np.squeeze(out)
    

    def grad_fu_ul_vars(self, x, y):
        """
        The gradient of the upper-level objective function wrt the upper-level variables
        
        Args:
            x   x[:,self.num_obj] --> Weights in the simplex set for lower-level objective functions
                x[self.num_obj,:] --> Upper-level variables
     	"""
        lam = x[:self.num_obj]
        xx = x[self.num_obj:]
        out_lam = np.zeros((self.num_obj,1))
        out_x = self.ul_prob.grad_fu_ul_vars_(xx, y) 
        out = np.concatenate((out_lam,out_x), axis=0)
        # out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1]) 
        # print('grad_fu_ul_vars',out)
        return out


    def grad_fu_ll_vars(self, x, y):
        """
        The gradient of the upper-level objective function wrt the lower-level variables
        
        Args:
            x   x[:,self.num_obj] --> Weights in the simplex set for lower-level objective functions
                x[self.num_obj,:] --> Upper-level variables
    	"""
        lam = x[:self.num_obj]
        xx = x[self.num_obj:]
        out = self.ul_prob.grad_fu_ll_vars_(xx, y)
        # print('grad_fu_ll_vars',out)
        return out
    

    def f_l(self, x, y):
        """
        The weigthed lower-level objective function: f_l = \sum_{j=1}^{q} f_{\ell}^j 
    	Such a function is used in the optimistic and risk-neutral formulations.
        
        Args:
            x   x[:,self.num_obj] --> Weights in the simplex set for lower-level objective functions
                x[self.num_obj,:] --> Upper-level variables
        """
        lam = x[:self.num_obj]
        xx = x[self.num_obj:]
        return (lam[0]*self.ll_prob.f_l_1(xx, y) + lam[1]*self.ll_prob.f_l_2(xx, y)).squeeze()


    def grad_fl_ul_vars(self, x, y):
        """
        The gradient of the lower-level objective function wrt the upper-level variables
        
        Args:
            x   x[:,self.num_obj] --> Weights in the simplex set for lower-level objective functions
                x[self.num_obj,:] --> Upper-level variables
    	"""
        lam = x[:self.num_obj]
        xx = x[self.num_obj:]
        out_lam = np.array([[self.ll_prob.f_l_1(xx, y)], [self.ll_prob.f_l_2(xx, y)]])
        out_x = lam[0]*self.ll_prob.grad_fl1_ul_vars(xx, y) + lam[1]*self.ll_prob.grad_fl2_ul_vars(xx, y)
        out = np.concatenate((out_lam,out_x), axis=0)
        # print('grad_fl_ul_vars',out)
        return out


    def grad_fl_ll_vars(self, x, y):
        """
        The gradient of the lower-level objective function wrt the lower-level variables
        
        Args:
            x   x[:,self.num_obj] --> Weights in the simplex set for lower-level objective functions
                x[self.num_obj,:] --> Upper-level variables
    	"""
        lam = x[:self.num_obj]
        xx = x[self.num_obj:]
        out = lam[0]*self.ll_prob.grad_fl1_ll_vars(xx, y) + lam[1]*self.ll_prob.grad_fl2_ll_vars(xx, y)
        # print('grad_fl_ll_vars',out)
        return out
    
    
    def hess_fl_ul_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function wrt the lower-level variables
        
        Args:
            x   x[:,self.num_obj] --> Weights in the simplex set for lower-level objective functions
                x[self.num_obj,:] --> Upper-level variables
    	"""
        lam = x[:self.num_obj]
        xx = x[self.num_obj:]
        out_1 = self.ll_prob.grad_fl1_ll_vars(xx, y).T
        out_2 = self.ll_prob.grad_fl2_ll_vars(xx, y).T
        out_3 = lam[0]*self.ll_prob.hess_fl1_ul_vars_ll_vars(xx, y) + lam[1]*self.ll_prob.hess_fl2_ul_vars_ll_vars(xx, y)
        out = np.concatenate((out_1, out_2, out_3), axis=0)
        # print('hess_fl_ul_vars_ll_vars',out)
        return out


    def hess_fl_ll_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function wrt the upper and lower level variables
        
        Args:
            x   x[:,self.num_obj] --> Weights in the simplex set for lower-level objective functions
                x[self.num_obj,:] --> Upper-level variables
    	"""
        lam = x[:self.num_obj]
        xx = x[self.num_obj:]
        out = lam[0]*self.ll_prob.hess_fl1_ll_vars_ll_vars(xx, y) + lam[1]*self.ll_prob.hess_fl2_ll_vars_ll_vars(xx, y)
        # print('hess_fl_ll_vars_ll_vars',out)
        return out
    
        
    def grad_lagrangian_ul_vars_risk_averse(self, x, y, lagr_mul, hess):
        """
        The gradient of the Lagrangian used in the risk-averse approach with respect to the upper-level variables
        
        Args:
            x                         Upper-level variables
            y   y[:,self.num_obj] --> Weights in the simplex set for lower-level objective functions
                y[self.num_obj,:] --> Lower-level variables
            lagr_mul                  Vectors of Lagrange multipliers
            hess                      Flag to use true Hessians
    	"""
        lam = y[:self.num_obj]
        yy = y[self.num_obj:]
        if hess:
            out = self.ul_prob.grad_fu_ul_vars_(x, yy) + lam[0]*np.dot(self.ll_prob.hess_fl1_ul_vars_ll_vars(x, yy),lagr_mul) + lam[1]*np.dot(self.ll_prob.hess_fl2_ul_vars_ll_vars(x, yy),lagr_mul)            
        else:    
            out = self.ul_prob.grad_fu_ul_vars_(x, yy) + lam[0]*np.dot(self.ll_prob.grad_fl1_ul_vars(x, yy),np.dot(self.ll_prob.grad_fl1_ll_vars(x, yy).T,lagr_mul)) + lam[1]*np.dot(self.ll_prob.grad_fl2_ul_vars(x, yy),np.dot(self.ll_prob.grad_fl2_ll_vars(x, yy).T,lagr_mul))
        # print('grad_lagrangian_ul_vars_risk_averse',out)
        return out    
    
      
class Quadratic1:
    
    def __init__(self, x_dim, y_dim, std_dev):
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.std_dev = std_dev
    
        self.a = np.random.uniform(0, -5,(self.x_dim,1))
        self.b = np.random.uniform(0, -3,(self.y_dim,1))
        
        # self.a = np.array([[-3],[-3]])
        # self.b = np.array([[-1],[-1]])
        
        # self.a = np.ones((self.x_dim,1))*3#*3
        # self.b = np.ones((self.y_dim,1))*2    
        
        # self.H1 = np.eye(self.x_dim,self.y_dim) 
        # self.H2 = make_spd_matrix(self.x_dim, random_state=0) 
        
        # self.H1 = np.array([[1]])
        # self.H2 = np.array([[1]])
        
        self.H1 = np.eye(self.x_dim,self.y_dim) 
        self.H2 = np.eye(self.x_dim,self.y_dim) 
    
    
    def f_u_(self, x, y):
        """
        The original upper-level objective function
    	"""
        # y = np.dot(self.Q, y)
        out = np.dot(self.a.T,x) + np.dot(self.b.T,y) + 0.5*np.dot(x.T,np.dot(self.H1,y)) + 0.5*np.dot(x.T,np.dot(self.H2,x))
        return np.squeeze(out)   


    def grad_fu_ul_vars_(self, x, y):
        """
        The gradient of the upper-level objective function wrt the upper-level variables
     	"""
        # y = np.dot(self.Q, y)
        out = self.a + 0.5*np.dot(self.H1,y) + np.dot(self.H2,x) 
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1]) 
        return out
    
    
    def grad_fu_ll_vars_(self, x, y):
        """
        The gradient of the upper-level objective function wrt the lower-level variables
    	"""
        # y = np.dot(self.Q, y)
        out = self.b + 0.5*np.dot(self.H1.T,x)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def hess_fu_ll_vars_(self, x, y):
        """
        The Hessian of the upper-level objective function wrt the lower-level variables
    	"""
        # y = np.dot(self.Q, y)
        out = np.zeros((self.y_dim,self.y_dim))
        return out
    
      
class SP1:
    
    def __init__(self, x_dim, y_dim, std_dev, Q):
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.std_dev = std_dev
        self.Q = Q
        
    
    def y_opt_func(self, lam, x):
        """
        The lower-level optimal solution as a function of lam and x
        
        Args:
           lam                    Vector of weights in the simplex set
           x                      Upper-level variables
    	""" 
        y_opt = ((lam[0]+lam[1])/(lam[0]+2*lam[1]))*x + ((3*lam[1])/(lam[0]+2*lam[1]))
        # y_opt = np.dot(self.Q, y_opt)
        return y_opt


    def f_l_1(self, x, y):
        """
        The lower-level objective function 1 - Problem SP1
    	"""
        y = np.dot(self.Q, y)
        return ((1/self.x_dim)*np.sum((x-1)**2 + (x-y)**2, axis=0)).squeeze()


    def f_l_2(self, x, y):
        """
        The lower-level objective function 2 - Problem SP1
    	"""
        y = np.dot(self.Q, y)
        return ((1/self.x_dim)*np.sum((y-3)**2 + (x-y)**2)).squeeze()
    

    def grad_fl1_ul_vars(self, x, y):
        """
        The gradient of the lower-level objective function 1 wrt the upper-level variables - Problem SP1
    	"""
        y = np.dot(self.Q, y)
        out = (1/self.x_dim)*(2*(x-1) + 2*(x-y))
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def grad_fl2_ul_vars(self, x, y):
        """
        The gradient of the lower-level objective function 2 wrt the upper-level variables - Problem SP1
    	"""
        y = np.dot(self.Q, y)
        out = (1/self.x_dim)*(2*(x-y)) 
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def grad_fl1_ll_vars(self, x, y):
        """
        The gradient of the lower-level objective function 1 wrt the lower-level variables - Problem SP1
    	"""
        y = np.dot(self.Q, y)
        out = (1/self.x_dim)*(-2*(x-y))
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def grad_fl2_ll_vars(self, x, y):
        """
        The gradient of the lower-level objective function 2 wrt the lower-level variables - Problem SP1
    	"""
        y = np.dot(self.Q, y)
        out = (1/self.x_dim)*(2*(y-3) - 2*(x-y))
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out

    
    def hess_fl1_ul_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 1 wrt the upper and lower level variables - Problem SP1
    	"""
        y = np.dot(self.Q, y)
        out = (1/self.x_dim)*np.eye(self.x_dim,self.y_dim)*(-2)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def hess_fl1_ll_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 1 wrt the lower-level variables - Problem SP1
    	"""
        y = np.dot(self.Q, y)
        out = (1/self.x_dim)*np.eye(self.y_dim,self.y_dim)*(2)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def hess_fl2_ul_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 2 wrt the upper and lower level variables - Problem SP1
    	"""
        y = np.dot(self.Q, y)
        out = (1/self.x_dim)*np.eye(self.x_dim,self.y_dim)*(-2)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def hess_fl2_ll_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 2 wrt the lower-level variables - Problem SP1
    	"""
        y = np.dot(self.Q, y)
        out = (1/self.x_dim)*np.eye(self.y_dim,self.y_dim)*(4)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
class JOS1:
    
    def __init__(self, x_dim, y_dim, std_dev, Q):
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.std_dev = std_dev
        self.Q = Q
        
    
    def y_opt_func(self, lam, x):
        """
        The lower-level optimal solution as a function of lam and x
        
        Args:
           lam                    Vector of weights in the simplex set
           x                      Upper-level variables
    	"""  
        y_opt = np.zeros((self.y_dim,1))
        # for i in range(self.y_dim):
        #     y_opt[i] = (2*lam[1]*x[i]-4*lam[1])/((lam[0]+lam[1])*x[i] - 2*lam[1]) 
        for i in range(self.y_dim):
            y_opt[i] = (2*lam[1]*(x[i]-2)**2)/(lam[0]*x[i]**2 + lam[1]*(x[i]-2)**2)
        # y_opt = np.dot(self.Q, y_opt)
        return y_opt

    def f_l_1(self, x, y):
        """
        The lower-level objective function 1 
    	"""
        y = np.dot(self.Q, y)
        return ((1/self.x_dim)*np.sum(x**2*y**2, axis=0)).squeeze()


    def f_l_2(self, x, y):
        """
        The lower-level objective function 2 
    	"""
        y = np.dot(self.Q, y)
        return ((1/self.x_dim)*np.sum((x-2)**2*(y-2)**2, axis=0)).squeeze()
    

    def grad_fl1_ul_vars(self, x, y):
        """
        The gradient of the lower-level objective function 1 wrt the upper-level variables 
    	"""
        y = np.dot(self.Q, y)
        out = (2/self.x_dim)*x*y**2
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def grad_fl2_ul_vars(self, x, y):
        """
        The gradient of the lower-level objective function 2 wrt the upper-level variables 
    	"""
        y = np.dot(self.Q, y)
        out = (2/self.x_dim)*(x-2)*(y-2)**2
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def grad_fl1_ll_vars(self, x, y):
        """
        The gradient of the lower-level objective function 1 wrt the lower-level variables 
    	"""
        y = np.dot(self.Q, y)
        out = (2/self.x_dim)*x**2*y
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def grad_fl2_ll_vars(self, x, y):
        """
        The gradient of the lower-level objective function 2 wrt the lower-level variables 
    	"""
        y = np.dot(self.Q, y)
        out = (2/self.x_dim)*(x-2)**2*(y-2)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out

    
    def hess_fl1_ul_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 1 wrt the upper and lower level variables 
    	"""
        y = np.dot(self.Q, y)
        out = np.eye(self.y_dim)
        np.fill_diagonal(out,x*y)
        out = (4/self.x_dim)*out
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def hess_fl1_ll_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 1 wrt the lower-level variables 
    	"""
        y = np.dot(self.Q, y)
        out = np.eye(self.y_dim)
        np.fill_diagonal(out,x**2)
        out = (2/self.x_dim)*out
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def hess_fl2_ul_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 2 wrt the upper and lower level variables 
    	"""
        y = np.dot(self.Q, y)
        out = np.eye(self.y_dim)
        np.fill_diagonal(out,(x-2)*(y-2))
        out = (4/self.x_dim)*out
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def hess_fl2_ll_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 2 wrt the lower-level variables 
    	"""
        y = np.dot(self.Q, y)
        out = np.eye(self.y_dim)
        np.fill_diagonal(out,(x-2)**2)
        out = (2/self.x_dim)*out
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
class GKV1:
    
    def __init__(self, x_dim, y_dim, std_dev, hess_std_dev):
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.std_dev = std_dev
        self.hess_std_dev = hess_std_dev
        
        # self.H3 = np.eye(self.y_dim,self.y_dim) 
        self.H3 = make_spd_matrix(self.y_dim, random_state=0)*20
        # self.H3 = np.array([[2,-1],[-1,2]])
        self.H4 = np.eye(self.y_dim,self.x_dim)
        self.H5 = np.eye(self.y_dim,self.x_dim) 
        # self.H6 = np.eye(self.y_dim,self.y_dim)
        self.H6 = make_spd_matrix(self.y_dim, random_state=0)*20 
        # self.H6 = np.array([[7,2],[2,1]])
        # self.H6 = np.array([[4,-2],[-2,2]])
    
    def y_opt_func(self, lam, x):
        """
        The lower-level optimal solution as a function of lam and x
        
        Args:
           lam                    Vector of weights in the simplex set
           x                      Upper-level variables
    	"""  
        y_opt = np.zeros((self.y_dim,1))
        y_opt = np.dot(np.linalg.inv(lam[0]*self.H3 + lam[1]*self.H6), np.dot((lam[0]/2)*self.H4 - (lam[1]/2)*self.H5,x)) 
        return y_opt


    def f_l_1(self, x, y):
        """
        The lower-level objective function 1 
    	"""
        return (0.5*np.dot(y.T,np.dot(self.H3,y)) - 0.5*np.dot(y.T,np.dot(self.H4,x))).squeeze()


    def f_l_2(self, x, y):
        """
        The lower-level objective function 2 
    	"""
        return (0.5*np.dot(y.T,np.dot(self.H5,x)) + 0.5*np.dot(y.T,np.dot(self.H6,y))).squeeze()
    

    def grad_fl1_ul_vars(self, x, y):
        """
        The gradient of the lower-level objective function 1 wrt the upper-level variables 
    	"""
        out = -0.5*np.dot(self.H4.T,y)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def grad_fl2_ul_vars(self, x, y):
        """
        The gradient of the lower-level objective function 2 wrt the upper-level variables 
    	"""
        out = 0.5*np.dot(self.H5.T,y)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def grad_fl1_ll_vars(self, x, y):
        """
        The gradient of the lower-level objective function 1 wrt the lower-level variables 
    	"""
        out = np.dot(self.H3.T,y) - 0.5*np.dot(self.H4,x)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def grad_fl2_ll_vars(self, x, y):
        """
        The gradient of the lower-level objective function 2 wrt the lower-level variables 
    	"""
        out = 0.5*np.dot(self.H5,x) + np.dot(self.H6.T,y)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out

    
    def hess_fl1_ul_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 1 wrt the upper and lower level variables 
    	"""
        out = -0.5*self.H4.T
        out = out + self.hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def hess_fl1_ll_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 1 wrt the lower-level variables 
    	"""
        out = self.H3.T
        out = out + self.hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def hess_fl2_ul_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 2 wrt the upper and lower level variables 
    	"""
        out = 0.5*self.H5.T
        out = out + self.hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def hess_fl2_ll_vars_ll_vars(self, x, y):
        """
        The Hessian of the lower-level objective function 2 wrt the lower-level variables 
    	"""
        out = self.H6.T
        out = out + self.hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out    
    
    
    
 

    



class FairBilevelMachineLearning:
    """
    Class used to define a fair bilevel machine learning problem

    Attributes
        device (str):               Name of the device used, whether GPU or CPU (if GPU is not available)
        file_path (str):            Path to the dataset file 
        download_celebA_bool (bool): A flag to download the dataset CelebA
        seed (int, optional):       The seed used for the experiments (default 42)
        val_pct (real):             Percentage of data used for validation (default 0.2)
        train_batch_size (real):    Minibatch size for training (default 0.01)
        batch_size (real):          Minibatch size for validation (default 0.05)
        training_loaders:           List of 5 increasing training datasets (01, 0123, 012345, etc.)
        testing_loaders:            List of 5 increasing testing datasets (01, 0123, 012345, etc.) 
        training_task_loaders:      List of 5 training datasets, each associated with a task
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, file_path, download_celebA_bool, seed=42, val_pct = 0.2 , train_batch_size = 0.01, batch_size = 0.05):       
        self.file_path = file_path
        self.download_celebA_bool = download_celebA_bool
        self.seed = seed
        self.val_pct = val_pct
        self.train_batch_size = train_batch_size
        self.batch_size = batch_size
        
        data_out = self.load_CIFAR()
        self.training_loaders = data_out[0]
        self.testing_loaders = data_out[1]
        self.training_task_loaders = data_out[2]
        
    def set_seed(self, seed):
        """
        Sets the seed  
    	"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)
                 
    def download_CelebA(self):    
        """
        Downloads CelebA and create folder 'celebA' within file_path
    	"""
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        download_url(dataset_url, root='.')
		#Extract from archive into the 'data' directory
        with tarfile.open('./celebA.tgz', 'r:gz') as tar:
            tar.extractall(path=self.file_path)        
         
    def load_CelebA(self):
    	"""
        Loads the CelebA dataset and generates the data loaders    
    	"""
    	self.set_seed(self.seed)

        # Download CelebA
    	if self.download_celebA_bool:
            self.download_celebA()          

     	# Load the CelebA dataset
    	data_dir = self.file_path + '/celebA' 
    	classes = os.listdir(data_dir + "/train")
    	# Load the data as PyTorch tensors
    	dataset = ImageFolder(data_dir + '/train', transform=ToTensor())
    	
    	# Generate training and validation indices
    	def split_indices(n, val_pct, seed):
    			# Determine the size of the validation set
    			n_val = int(val_pct*n)
    	        # Set the random seed
    			np.random.seed(seed)
    	        # Create random permutation of 0 to n-1
    			idxs = np.random.permutation(n)
    	        # Pick first n_val indices for validation
    			return idxs[n_val:], idxs[:n_val] 

    	# Generate the set of indices to sample from each set
    	train_indices, val_indices = split_indices(len(dataset), self.val_pct, self.seed) 
    	   	
    	# Load the full train and validation datasets
    	train_sampler = SubsetRandomSampler(train_indices)
    	train_full_loader = DataLoader(dataset, sampler=train_sampler) #50,000 images
    	test_sampler = SubsetRandomSampler(val_indices)
    	test_full_loader = DataLoader(dataset, sampler=test_sampler) #9,999 images  	
    	
    	# Set up the training data loader
    	train_loader_01 = [] #Will add 7,958 images. Total 7,958. (airplanes, automobiles)

    	for X,y in train_full_loader:
            if y == 0 or y == 1:
                train_loader_01.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3:
                train_loader_0123.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5:
                train_loader_012345.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5 or y == 6 or y == 7:
                train_loader_01234567.append([X,y])
            train_loader_0123456789.append([X,y])

    	for X,y in train_full_loader:
            if y == 0 or y == 1:
                train_task_loader_01.append([X,y]) 
            if y == 2 or y == 3:
                train_task_loader_23.append([X,y]) 
            if y == 4 or y == 5:
                train_task_loader_45.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5 or y == 6 or y == 7:
                train_task_loader_67.append([X,y])
            train_task_loader_89.append([X,y])
    
        # Batch the training loaders
    	loaders = [train_loader_01, train_loader_0123, train_loader_012345, train_loader_01234567, train_loader_0123456789]
    	train_loader_1 = []; train_loader_2 = []; train_loader_3 = []; train_loader_4 = []; train_loader_5 = []
    	for i in range(len(loaders)):
            X = []; y = []
            counter = 0
            for X_y in loaders[i]:
                X.append(X_y[0][0].numpy())
                y.append(X_y[1].numpy()) 
                if len(X) == math.trunc(len(loaders[i]) * self.train_batch_size):
                    if i == 0:
                        train_loader_1.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 1:
                        train_loader_2.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 2:
                        train_loader_3.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 3:
                        train_loader_4.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 4:
                        train_loader_5.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    counter = 0
                    X = []; y = []
                else:
                    counter += 1
    	
       	# Set up the five testing data loaders
    	test_loader_01 = [] #Will add a total of 2,042 images. Total 2,042.
    	test_loader_0123 = [] #Will add a total of 1,993 images. Total 4,035.
    	test_loader_012345 = [] #Will add a total of 2,016 images. Total 6,051.
    	test_loader_01234567 = [] #Will add a total of 1,982 images. Total 8,033.
    	test_loader_0123456789 = [] #Will add a total of 1,966 images. Total 9,999.
        
    	for X,y in test_full_loader:
            if y == 0 or y == 1:
                test_loader_01.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3:
                test_loader_0123.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5:
                test_loader_012345.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5 or y == 6 or y == 7:
                test_loader_01234567.append([X,y])
            test_loader_0123456789.append([X,y])
        
        # Batch the validation data
    	test_loaders = [test_loader_01, test_loader_0123, test_loader_012345, test_loader_01234567, test_loader_0123456789]
    	test_loader_1 = []; test_loader_2 = []; test_loader_3 = []; test_loader_4 = []; test_loader_5 = []
    	for i in range(len(test_loaders)):
            X = []; y = []
            counter = 0
            for X_y in test_loaders[i]:
                X.append(X_y[0][0].numpy())
                y.append(X_y[1].numpy())
                if len(X) == math.trunc(len(test_loaders[i]) * self.batch_size):
                    if i == 0:
                        test_loader_1.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 1:
                        test_loader_2.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 2:
                        test_loader_3.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 3:
                        test_loader_4.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 4:
                        test_loader_5.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    counter = 0
                    X = []; y = []
                else:
                    counter += 1

        # Set up a training data loader for each task  
    	train_task_loader_01 = [] 
    	train_task_loader_23 = [] 
    	train_task_loader_45 = [] 
    	train_task_loader_67 = [] 
    	train_task_loader_89 = []
    
    	for X,y in train_full_loader:
            if y == 0 or y == 1:
                train_task_loader_01.append([X,y]) 
            if y == 2 or y == 3:
                train_task_loader_23.append([X,y]) 
            if y == 4 or y == 5:
                train_task_loader_45.append([X,y])
            if y == 0 or y == 1 or y == 2 or y == 3 or y == 4 or y == 5 or y == 6 or y == 7:
                train_task_loader_67.append([X,y])
            train_task_loader_89.append([X,y])

        # Batch the training data for each task               
    	task_loaders = [train_task_loader_01, train_task_loader_23, train_task_loader_45, train_task_loader_67, train_task_loader_89]
    	train_task_loader_1 = []; train_task_loader_2 = []; train_task_loader_3 = []; train_task_loader_4 = []; train_task_loader_5 = []
    	for i in range(len(task_loaders)):
            X = []; y = []
            counter = 0
            for X_y in task_loaders[i]:
                X.append(X_y[0][0].numpy())
                y.append(X_y[1].numpy()) 
                if len(X) == math.trunc(len(task_loaders[i]) * self.train_batch_size):
                    if i == 0:
                        train_task_loader_1.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 1:
                        train_task_loader_2.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 2:
                        train_task_loader_3.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 3:
                        train_task_loader_4.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    if i == 4:
                        train_task_loader_5.append([torch.Tensor(X),torch.from_numpy(np.asarray(y))])
                    counter = 0
                    X = []; y = []
                else:
                    counter += 1 

    	training_loaders = [train_loader_1, train_loader_2, train_loader_3, train_loader_4, train_loader_5]
    	testing_loaders = [test_loader_1, test_loader_2, test_loader_3, test_loader_4, test_loader_5]
    	training_task_loaders = [train_task_loader_1, train_task_loader_2, train_task_loader_3, train_task_loader_4, train_task_loader_5]
    	
    	return training_loaders, testing_loaders, training_task_loaders

    class flatten(nn.Module):
        """
        Flattens a Convolutional Deep Neural Network
    	"""
        def forward(self, x):
            return x.view(x.shape[0], -1)

    def generate_model(self):
        """
        Creates a Convolutional Deep Neural Network
    	"""
   	    # Convolutional Deep Neural Network for CIFAR
        model_cnn_robust = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), #This turns the [3,32,32] image into a [32,32,32] image
    	                                     nn.ReLU(),
    	                                     nn.Conv2d(32, 64, kernel_size=3, padding=1), #This turns the [32,32,32] image into a [64,32,32] image
    	                                     nn.ReLU(),
    	                                     nn.MaxPool2d(2, stride=2), #Turns the [64,32,32] image into a [64,16,16] image
    	                                     self.flatten(),
    	                                     nn.Linear(64*16*16, 10)).to(self.device)
        return model_cnn_robust

    def loss_func(self, in_1, in_2):
     	"""
        Loss function
    	"""
     	func = nn.CrossEntropyLoss().to(self.device)
     	return func(in_1, in_2)    



 
    








########################## OLD ####################################


# class LogisticRegression:
#     """
#     Class used to define a logistic regression problem

#     Attributes
#         device (str):                 Name of the device used
#         train_size (int):             Dimension of the training set 
#         seed (int, optional):         The seed used for the experiments (default 42)
#         file_path (str, optional):    Path to the dataset file (default "./data")
#         batch_size (int, optional):   Dimension of minibatches (default 512)
#     """
    
#     device = torch.device('cpu')
#     train_size = 22500 #22500 #37500
    
#     def __init__(self, seed=42, file_path='Adult_income_race&gender_reduced_smg.csv', batch_size=512, lamb=1):
        
#         self.seed = seed
#         self.file_path = file_path
#         self.batch_size = batch_size
#         self.lamb = lamb
#         self.load_data()  

#         data_out = self.load_data()
#         self.training_loader = data_out[0]
#         self.validation_loader = data_out[1]
#         self.fulltrain_X = data_out[2]
#         self.fulltrain_y = data_out[3]
#         self.fullvalid_X = data_out[4]
#         self.fullvalid_y = data_out[5]
#         self.num_classes = data_out[6]        
    
#     def set_seed(self, seed):
#         """
#         Sets the seed
#     	"""
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.cuda.manual_seed(seed)
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#         os.environ['PYTHONHASHSEED'] = str(seed)
    
#     def load_data(self):
#         """
#         Loads the dataset
#     	"""
#         self.set_seed(self.seed)
        
#         # Load the dataset
#         income_data = pd.read_csv(self.file_path, header=None)
#         # Create the training set     
#         training_data = income_data[:self.train_size] #here we select rows        
#         training_labels = training_data[0] #here we select columns
#         training_data.drop(training_data.columns[[0]], axis=1, inplace=True)
#         training_data = training_data.to_numpy()
#         # Create the validation set     
#         validation_data = income_data[self.train_size:]        
#         validation_labels = validation_data[0]
#         validation_data.drop(validation_data.columns[[0]], axis=1, inplace=True)
#         validation_data = validation_data.to_numpy()        
#         num_classes = training_data.shape[1]
                
#         # Batch the training data
#         X = []; y = []
#         batched_training = []
#         counter = 0
#         for i in range(len(training_data)):
#             X.append(training_data[i])
#             y.append(training_labels[i])
#             if len(X) == self.batch_size:
#                 batched_training.append([np.asarray(X), np.asarray(y)])
#                 counter = 0
#                 X = []; y = []
#             else:
#                 counter += 1
        
#         # Batch the validation data
#         X = []; y = []
#         batched_validation = []
#         counter = 0
#         for i in range(len(validation_data)):
#             X.append(validation_data[i])
#             y.append(validation_labels[i+self.train_size])
#             # y.append(validation_labels[i]) #Tommaso Matlab
#             if len(X) == self.batch_size:
#                 batched_validation.append([np.asarray(X), np.asarray(y)])
#                 counter = 0
#                 X = []; y = []
#             else:
#                 counter += 1
                
#         # Training Full Batch        
#         Xfull = []; yfull = []
#         fullbatched_training = []
#         for i in range(len(training_data)):
#             Xfull.append(training_data[i])
#             yfull.append(training_labels[i])
#         fullbatched_training.append([np.asarray(Xfull), np.asarray(yfull)])
#         Xfull = []; yfull = []
#         fulltrain_X = fullbatched_training[0][0]
#         fulltrain_y = fullbatched_training[0][1]
        
#         # Validation Full Batch
#         Xfull = []; yfull = []
#         fullbatched_validation = []
#         for i in range(len(validation_data)):
#             Xfull.append(validation_data[i])
#             yfull.append(validation_labels[i+self.train_size])
#         fullbatched_validation.append([np.asarray(Xfull), np.asarray(yfull)])
#         Xfull = []; yfull = []
#         fullvalid_X = fullbatched_validation[0][0]
#         fullvalid_y = fullbatched_validation[0][1]
        
#         return batched_training, batched_validation, fulltrain_X, fulltrain_y, fullvalid_X, fullvalid_y, num_classes

#     def log_loss(self, X, y, h_b):
#         """
#         The logistic loss function
#     	"""
#         Z = np.multiply(y, np.inner(h_b, X))
#         Z = np.log(1 + np.exp(-Z))
#         return np.mean(Z)
    
#     def grad_log_loss(self, X, y, h_b):
#         """
#         The gradient of the logistic loss function
#     	"""
#         numerator = np.exp(-np.multiply(y, np.inner(h_b, X)))
#         denominator = 1 + numerator
#         prod = -np.multiply(y,X.T).T
#         result = np.multiply(numerator/denominator, prod.T).T
#         return np.mean(result, axis=0)
    
#     def hess_log_loss(self, xi, yi, h_b):    
#         """
#         The Hessian of the logistic loss function
#     	"""
#         numerator = np.exp(-np.multiply(yi, np.dot(h_b, xi)))
#         denominator = (1 + numerator)**2
#         prod = np.outer(xi,xi)
#         result = np.multiply(numerator/denominator, prod)
#         return result
    
#     def f_u(self, train_X, train_y, valid_X, valid_y, ul_vars, ll_vars):
#         """
#         The UL objective function
#     	"""
#         return 0.5*(self.log_loss(train_X, train_y, ul_vars) + self.log_loss(valid_X, valid_y, ll_vars)) + (self.lamb/2) * np.linalg.norm(ul_vars - ll_vars)**2

#     def f_l(self, valid_X, valid_y, ul_vars, ll_vars):
#         """
#         The LL objective function
#     	"""
#         return self.log_loss(valid_X, valid_y, ll_vars) + (self.lamb/2) * np.linalg.norm(ul_vars - ll_vars)**2 #+ 0.01*np.linalg.norm(ll_vars)**2

#     def grad_fu_ul_vars(self, train_X, train_y, valid_X, valid_y, ul_vars, ll_vars):
#         """
#         The gradient of the UL objective function wrt the UL variables
#     	"""
#         return 0.5*self.grad_log_loss(train_X, train_y, ul_vars) + self.lamb*(ul_vars - ll_vars)#+ self.lamb*(ll_vars)

#     def grad_fu_ll_vars(self, train_X, train_y, valid_X, valid_y, ul_vars, ll_vars):
#         """
#         The gradient of the UL objective function wrt the LL variables
#     	"""
#         return 0.5*self.grad_log_loss(valid_X, valid_y, ll_vars) - self.lamb*(ul_vars - ll_vars)#+ self.lamb*(ll_vars)

#     def grad_fl_ul_vars(self, valid_X, valid_y, ul_vars, ll_vars):
#         """
#         The gradient of the UL objective function wrt the UL variables
#     	"""
#         return  self.lamb*(ul_vars - ll_vars)

#     def grad_fl_ll_vars(self, valid_X, valid_y, ul_vars, ll_vars):
#         """
#         The gradient of the LL objective function wrt the LL variables
#     	"""
#         return self.grad_log_loss(valid_X, valid_y, ll_vars) - self.lamb*(ul_vars - ll_vars) #+ 0.01*2*ll_vars

#     def hess_fl_ll_vars_ll_vars(self, valid_X, valid_y, ul_vars, ll_vars):
#         """
#         The Hessian of the LL objective function wrt the LL variables
#     	"""
#         hess = np.zeros((ll_vars.shape[0],ll_vars.shape[0]))
#         for i in range(valid_X.shape[0]):
#             xi = valid_X[i]
#             yi = valid_y[i]
#             hess = hess + self.hess_log_loss(xi, yi, ll_vars)
#         return (1/valid_X.shape[0])*hess + self.lamb*np.eye(ll_vars.shape[0]) #+ np.eye(ll_vars.shape[0])*0.01*2

#     def hess_fl_ul_vars_ll_vars(self, valid_X, valid_y, ul_vars, ll_vars):
#         """
#         The Hessian of the LL objective function wrt the UL and LL variables
#     	"""
#         return -self.lamb*np.eye(ll_vars.shape[0])



























 

               
    
    