import os
import numpy as np
import random
import time

import torch
import torch.nn as nn
import copy

from scipy.optimize import minimize_scalar 
from scipy.sparse.linalg import cg
from scipy.stats import t

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
from scipy.optimize import SR1
from scipy.optimize import minimize
import scipy.special as sc





class BilevelMultiobjSolverSyntheticProb:
    """
    Class used to implement bilevel mutilobjective stochastic algorithms for synthetic quadratic bilevel problems

    Attributes
        ul_lr (real):                         Current upper-level stepsize (default 1)
        ll_lr (real):                         Current lowel-level stepsize (default 1)
        llp_iters (int):                      Current number of lower-level steps (default 1)
        func (obj):                           Object used to define an optimization problem to solve 
        algo (str):                           Name of the algorithm to run ('bsg' or 'darts')
        exact_pareto_point (bool, optional)   A flag to exactly compute y as a function of x and lower-level weights (default False)                  
        num_pareto_points_risk_neutral (int)  Number of weights in the discretization of the simplex set       
        minibatch_size_rn (int)               Number of points considered in the discretization of the simplex set (default num_pareto_points_risk_neutral)
        ul_stepsize_scheme (int, optional):   A flag to choose the upper-level stepsize scheme (default 0): 0 --> decaying; 1 --> fixed; 2 --> Bilevel Armijo backtracking line search 
        ll_stepsize_scheme (int, optional):   A flag to choose the lower-level stepsize scheme (default 0): 0 --> decaying; 1 --> fixed; 2 --> Armijo backtracking line search
        seed (int, optional):                 The seed used for the experiments (default 0)
        ul_lr_init (real, optional):          Initial upper-level stepsize (default 5); if ul_stepsize_scheme == 2, the initial stepsize is 1
        ll_lr_init (real, optional):          Initial lower-level stepsize (default 0.1); if ll_stepsize_scheme == 2, the initial stepsize is 1
        max_iter (int, optional):             Maximum number of iterations (default 500)
        normalize (bool, optional):           A flag to normalize the direction used for the upper-level update (default False)
        hess (bool, optional):                A flag to compute stochastic Hessians instead of rank-1 approximations when using BSG (default False)
        llp_iters_init (real, optional):      Initial number of lower-level steps (default 1)
        inc_acc (bool, optional):             A flag to use an increasing accuracy strategy for the lower-level problem (default False)
        true_func (bool, optional):           A flag to compute the true function (default True)
        opt_stepsize (bool, optional):        A flag to compute the optimal stepsize when computing the true objective function of the bilevel problem (default True)
        use_stopping_iter (bool, optional):   A flag to use the total number of iterations as a stopping criterion (default True)
        stopping_time (real, optional):       Maximum running time (in sec) used when use_stopping_iter is False (default 0.5)
        iprint (int):                         Used to print info on the optimization (default 1): 0 --> no printing; 1 --> at the end of the optimization; 2 --> at each iteration 
    """    
    
    ul_lr = 1
    ll_lr = 1
    llp_iters = 1
    
    
    def __init__(self, func, algo, risk_level, exact_pareto_point=False, num_pareto_points_risk_neutral=2, \
                 minibatch_size_rn=2, ul_stepsize_scheme=2, ll_stepsize_scheme=2, \
                 seed=0, ul_lr=5, ll_lr=0.1, max_iter=2000, normalize=False,\
                 hess=False, llp_iters=1, inc_acc=False, backtracking=True, true_func=True, \
                 opt_stepsize = False, use_stopping_iter=True, stopping_time=0.5, iprint = 1):

        self.func = func
        self.algo = algo
        self.risk_level = risk_level #0 1 2
        
        self.exact_pareto_point = exact_pareto_point
        self.num_pareto_points_risk_neutral = num_pareto_points_risk_neutral-1
        self.minibatch_size_rn = minibatch_size_rn # num_pareto_points_risk_neutral # minibatch size risk neutral approach
        self.ul_stepsize_scheme = ul_stepsize_scheme
        self.ll_stepsize_scheme = ll_stepsize_scheme
        
        self.seed = seed
        self.ul_lr_init = ul_lr
        self.ll_lr_init = ll_lr
        self.max_iter = max_iter
        self.normalize = normalize
        self.hess = hess
        self.llp_iters_init = llp_iters
        self.inc_acc = inc_acc
        self.true_func = true_func
        self.opt_stepsize = opt_stepsize        
        self.use_stopping_iter = use_stopping_iter
        self.stopping_time = stopping_time
        self.iprint = iprint
        
        
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
    
    
    def generate_ll_vars(self):
        """
        Creates a vector of LL variables 
        """
        ll_vars = np.random.uniform(0, 0.1, (self.func.y_dim, 1))
        return ll_vars
  
    
    # def create_stream_idxs(self):
    #     """
    #     Creates a stream of indexes
    #     """
    #     stream_idxs = np.random.permutation(self.simplex_set_discr.shape[1])
    #     return stream_idxs
    
    
    # def rand_samp_no_replace(self, stream_idxs):
    #     """
    #     Randomly samples (without replacement) from a stream of mini-batches 
    #     """
    #     idxs = stream_idxs[len(stream_idxs)-1]
    #     stream_idxs = np.delete(stream_idxs, len(stream_idxs)-1, axis=1)
    #     return idxs
    
    
    # def minibatch(self, stream_idxs):
    #     """
    #     Returns a mini-batch of indexes and the updated stream of mini-batches
    #     """
    #     if len(stream_idxs) == 0:
    #         stream_idxs = self.create_stream_idxs()
    #     minibatch_idxs = self.rand_samp_no_replace(stream_idxs)
    #     return minibatch_idxs, stream_idxs
    
    
    def create_loader_rn(self):
        idxs = np.random.permutation(self.simplex_set_discr.shape[1])
        loader = [idxs[item:item+self.minibatch_size_rn] for item in range(0, len(idxs), self.minibatch_size_rn)]
        return loader
    
    
    def create_stream_rn(self, loader):
        """
        Creates a stream of mini-batches of data
        """
        stream = random.sample(loader,len(loader)) 
        random.shuffle(stream) 
        return stream
    
    
    def rand_samp_no_replace_rn(self, stream):
        """
        Randomly samples (without replacement) from a stream of mini-batches data
        """
        aux = [item for list_aux in stream[-1:] for item in list_aux]
        del stream[-1:]
        return aux
    
    
    def minibatch_rn(self, loader, stream):
        """
        Returns a mini-batch of data (features and labels) and the updated stream of mini-batches
        """
        if len(stream) == 0:
            stream = self.create_stream_rn(loader)
        minibatch_idxs = self.rand_samp_no_replace_rn(stream)
        return minibatch_idxs, stream
    
    
    def update_llp(self, ul_vars, ll_vars): 
        """
        Updates the LL variables by taking a gradient descent step for the LL problem
        """        
        # Exact Pareto point y(x) 
        if self.exact_pareto_point:
            lam = ul_vars[:self.func.num_obj]
            xx = ul_vars[self.func.num_obj:]
            # ll_vars = ((lam[0]+lam[1])/(lam[0]+2*lam[1]))*xx + (3*lam[1]/(lam[0]+2*lam[1])) 
            ll_vars = self.func.y_opt_func(lam, xx)
        else:
            for i in range(self.llp_iters):
                # Obtain gradient of the LLP wrt LL variables
                grad_llp_ll_vars = self.func.grad_fl_ll_vars(ul_vars, ll_vars)
                # Update the variables
                if self.ll_stepsize_scheme == 0:
                    ll_vars = ll_vars - self.ll_lr/(i+1) * grad_llp_ll_vars
                elif self.ll_stepsize_scheme == 1:
                    ll_vars = ll_vars - self.ll_lr * grad_llp_ll_vars
                elif self.ll_stepsize_scheme == 2: 
                    alpha = self.backtracking_armijo_ll(ul_vars, ll_vars) 
                    self.ll_lr = alpha
                    ll_vars = ll_vars - alpha * grad_llp_ll_vars
        
        return ll_vars
    
    
    def update_llp_risk_averse(self, ul_vars, ll_vars): 
        """
        Updates the LL variables after solving the subproblem in the risk-averse formulation
        """        
        
        lb = np.concatenate((np.zeros(self.func.num_obj),-np.inf*np.ones(self.func.y_dim)))
        ub = np.inf*np.ones(self.func.num_obj+self.func.y_dim)
        bounds = Bounds(lb, ub) 
        
        matrix_constr_lam = np.concatenate((np.ones((1,self.func.num_obj)),np.zeros((1,self.func.y_dim))),axis=1)
        linear_constraint = LinearConstraint(matrix_constr_lam, lb=1, ub=1)
        
        # \sum_{j=1}^{q} \lambda_j \nabla_y f^j_\ell(x,y)=0
        def constr_pareto(y):
            y = y.reshape(-1,1)
            lam = y[:self.func.num_obj]
            yy = y[self.func.num_obj:]
            out = lam[0]*self.func.ll_prob.grad_fl1_ll_vars(ul_vars, yy) + lam[1]*self.func.ll_prob.grad_fl2_ll_vars(ul_vars, yy)
            # print('constr_pareto',out.T)
            return out.flatten()

        def jac_constr(y):
           y = y.reshape(-1,1)
           lam = y[:self.func.num_obj]
           yy = y[self.func.num_obj:]
           jac_lam = np.concatenate((self.func.ll_prob.grad_fl1_ll_vars(ul_vars, yy), self.func.ll_prob.grad_fl2_ll_vars(ul_vars, yy)), axis=1) 
           # if self.hess == True:
           jac_yy = lam[0]*self.func.ll_prob.hess_fl1_ll_vars_ll_vars(ul_vars, yy) + lam[1]*self.func.ll_prob.hess_fl2_ll_vars_ll_vars(ul_vars, yy)
           # else:
           #     jac_yy = lam[0]*np.matmul(self.func.ll_prob.grad_fl1_ll_vars(ul_vars, yy),self.func.ll_prob.grad_fl1_ll_vars(ul_vars, yy).T) + lam[1]*np.matmul(self.func.ll_prob.grad_fl2_ll_vars(ul_vars, yy),self.func.ll_prob.grad_fl2_ll_vars(ul_vars, yy).T)
           # print('jac_constr',np.concatenate((jac_lam, jac_yy), axis=1))
           return np.concatenate((jac_lam, jac_yy), axis=1)             

        def zero_Hess_constr(y, v):
           return np.zeros((len(ll_vars), len(ll_vars))) 
        
        if not(self.hess):
            # nonlinear_constraint = NonlinearConstraint(constr_pareto, lb=0, ub=0, jac='2-point', hess=BFGS())
            nonlinear_constraint = NonlinearConstraint(constr_pareto, lb=0, ub=0, jac='2-point', hess = zero_Hess_constr)
        else:
            # nonlinear_constraint = NonlinearConstraint(constr_pareto, lb=0, ub=0, jac=jac_constr, hess = BFGS())
            nonlinear_constraint = NonlinearConstraint(constr_pareto, lb=0, ub=0, jac=jac_constr, hess = zero_Hess_constr)
        
        # Hot start
        if True: 
            y_init = ll_vars
        else:
            y_init = np.zeros((self.func.num_obj+self.func.y_dim,1))
        
        def obj_funct(y):
            y = y.reshape(-1,1)
            lam = y[:self.func.num_obj]
            yy = y[self.func.num_obj:]
            return -np.squeeze(self.func.ul_prob.f_u_(ul_vars, yy))
        
        def jac_obj_funct(y):
            y = y.reshape(-1,1)
            lam = y[:self.func.num_obj]
            yy = y[self.func.num_obj:]
            out = -np.concatenate((np.zeros((self.func.num_obj,1)),self.func.ul_prob.grad_fu_ll_vars_(ul_vars, yy)), axis=0)
            # print('jac_obj_funct',out.flatten())
            return out.flatten()
        
        def hess_obj_funct(y, p):
            y = y.reshape(-1,1)
            lam = y[:self.func.num_obj]
            yy = y[self.func.num_obj:]
            # print('hess_obj_funct',np.concatenate((np.zeros((self.func.num_obj,1)),np.dot(self.func.ul_prob.hess_fu_ll_vars_(ul_vars, yy),p[self.func.num_obj:].reshape(-1,1))), axis=0))
            return np.concatenate((np.zeros((self.func.num_obj,1)),np.dot(self.func.ul_prob.hess_fu_ll_vars_(ul_vars, yy),p[self.func.num_obj:].reshape(-1,1))), axis=0)
        
        xtol_param = 1e-08
        gtol_param = 1e-08
        barrier_tol_param = 1e-08
        
        if not(self.hess):
            
            def zero_hess_obj_funct(y):
               return np.zeros((len(ll_vars), len(ll_vars))) 
            
            res = minimize(obj_funct, y_init, method='trust-constr', jac=jac_obj_funct, hess=zero_hess_obj_funct, #hess=SR1(),
                    constraints=[linear_constraint, nonlinear_constraint],
                    options={'xtol': xtol_param, 'gtol': gtol_param, 'barrier_tol': barrier_tol_param, 'verbose': 0}, bounds=bounds)
        else:
            # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html
            res = minimize(obj_funct, y_init, method='trust-constr', jac=jac_obj_funct, hessp=hess_obj_funct,
                   constraints=[linear_constraint, nonlinear_constraint],
                   options={'xtol': xtol_param, 'gtol': gtol_param, 'barrier_tol': barrier_tol_param, 'verbose': 0}, bounds=bounds)
               
        # print('res',res) 
        ll_vars = res.x.reshape(-1,1)
        lagr_mul = res.v
        lagr_mul = -lagr_mul[1].reshape(-1,1)
        # print('lagr_mul',lagr_mul)
        
        return ll_vars, lagr_mul
    
    
    def darts_ulp_grad(self, ul_vars, ll_vars, ll_orig_vars): 
        """
        Updates the UL variables based on the DARTS step for the UL problem
        """      
        # Gradient of the UL objective function wrt LL variables
        grad_ulp_ll_vars = self.func.grad_fu_ll_vars(ul_vars, ll_vars)
        grad_norm = np.linalg.norm(grad_ulp_ll_vars)
        # Gradient of the UL objective function wrt UL variables
        grad_ulp_ul_vars = self.func.grad_fu_ul_vars(ul_vars, ll_vars)
        
        ep = 0.01 / grad_norm
        # Define y+ and y-
        ll_plus = ll_orig_vars + ep * grad_ulp_ll_vars
        ll_minus = ll_orig_vars - ep * grad_ulp_ll_vars
        
        # Gradient of the LL objective function wrt UL variables at the point (ul_vars, ll_plus)
        grad_plus = self.func.grad_fl_ul_vars(ul_vars, ll_plus)
        
        # Gradient of the LL objective function wrt UL variables at the point (ul_vars, ll_minus)
        grad_minus = self.func.grad_fl_ul_vars(ul_vars, ll_minus)
    
        grad_approx = grad_ulp_ul_vars - self.ll_lr * ((grad_plus - grad_minus) / (2 * ep))
        
        # Normalize the direction
        if self.normalize:
            grad_approx = grad_approx / np.linalg.norm(grad_approx, np.inf)
        
        # Update the UL variables
        if self.ul_stepsize_scheme == 2:
            alpha, ll_vars = self.backtracking_armijo_ul(ul_vars, ll_vars, grad_approx)
            self.ul_lr = alpha
            ul_vars = ul_vars - alpha * grad_approx
        elif self.ul_stepsize_scheme == 0 or self.ul_stepsize_scheme == 1:    
            ul_vars = ul_vars - self.ul_lr * grad_approx
        
        # Project the upper-level variables onto the simplex set
        ul_vars = self.simplex_projection(ul_vars)
        
        ## project if a simple bound exists
        if self.func.projection:
            ul_vars = np.where(ul_vars <= self.func.ub, ul_vars, self.func.ub)
            ul_vars = np.where(ul_vars >= self.func.lb, ul_vars, self.func.lb) 
        
        return ul_vars, ll_vars 

    
    def darts_ulp_grad_risk_neutral(self, ul_vars, ll_vars_array, ll_orig_vars_array, minibatch_idxs): 
        """
        Updates the UL variables based on the DARTS step for the UL problem
        """   
        
        grad_approx = np.zeros((self.func.x_dim,self.simplex_set_discr.shape[1]))
            
        for count, lam in enumerate(self.simplex_set_discr.T):
            if count in minibatch_idxs:
                lam_ul_vars = np.concatenate((lam.reshape(-1,1), ul_vars), axis=0)
                
                # Gradient of the UL objective function wrt LL variables
                grad_ulp_ll_vars = self.func.grad_fu_ll_vars(lam_ul_vars, ll_vars_array[:,count].reshape(-1,1))
                grad_norm = np.linalg.norm(grad_ulp_ll_vars)
                # Gradient of the UL objective function wrt UL variables
                grad_ulp_ul_vars = self.func.grad_fu_ul_vars(lam_ul_vars, ll_vars_array[:,count].reshape(-1,1))
                
                ep = 0.01 / grad_norm
                # Define y+ and y-
                ll_plus = ll_orig_vars_array[:,count].reshape(-1,1) + ep * grad_ulp_ll_vars
                ll_minus = ll_orig_vars_array[:,count].reshape(-1,1) - ep * grad_ulp_ll_vars
                
                # Gradient of the LL objective function wrt UL variables at the point (ul_vars, ll_plus)
                grad_plus = self.func.grad_fl_ul_vars(lam_ul_vars, ll_plus)
                
                # Gradient of the LL objective function wrt UL variables at the point (ul_vars, ll_minus)
                grad_minus = self.func.grad_fl_ul_vars(lam_ul_vars, ll_minus)
            
                grad_approx = grad_ulp_ul_vars - self.ll_lr * ((grad_plus - grad_minus) / (2 * ep))
                
                # Normalize the direction
                if self.normalize:
                    grad_approx = grad_approx / np.linalg.norm(grad_approx, np.inf)
        
        grad_approx_tot = grad_approx.mean(axis=1)
        
        # Update the UL variables
        if self.ul_stepsize_scheme == 2:
            alpha, ll_vars_array = self.backtracking_armijo_ul_risk_neutral(ul_vars, ll_vars_array, grad_approx_tot, minibatch_idxs)
            self.ul_lr = alpha
            ul_vars = ul_vars - alpha * grad_approx_tot
        elif self.ul_stepsize_scheme == 0 or self.ul_stepsize_scheme == 1:    
            ul_vars = ul_vars - self.ul_lr * grad_approx_tot
            
        ## project if a simple bound exists
        if self.func.projection:
            ul_vars = np.where(ul_vars <= self.func.ub, ul_vars, self.func.ub)
            ul_vars = np.where(ul_vars >= self.func.lb, ul_vars, self.func.lb) 
        
        return ul_vars, ll_vars_array
    
    
    def calc_dx_ulp(self, ul_vars, ll_vars): 
        """
        Updates the UL variables based on the BSG step for the UL problem 
        """   
        
        # Gradient of the UL objective function wrt LL variables
        grad_ulp_ll_vars = self.func.grad_fu_ll_vars(ul_vars, ll_vars)
        # Gradient of the UL problem wrt UL variables
        grad_ulp_ul_vars = self.func.grad_fu_ul_vars(ul_vars, ll_vars)
        
        # Gradient of the LL objective function wrt LL variables
        grad_llp_ll_vars = self.func.grad_fl_ll_vars(ul_vars, ll_vars)
        # Grad of the LL problem wrt UL vars
        grad_LLP_ul_vars = self.func.grad_fl_ul_vars(ul_vars, ll_vars)
        
        # Compute the bsg direction
        if self.hess == True:
            # Hessian of the LL problem wrt UL and LL variables
            hess_LLP_ul_vars_ll_vars = self.func.hess_fl_ul_vars_ll_vars(ul_vars, ll_vars)
            # Hessian of the LL problem wrt LL variables
            hess_LLP_ll_vars_ll_vars = self.func.hess_fl_ll_vars_ll_vars(ul_vars, ll_vars)
            
            lambda_adj, exit_code = cg(hess_LLP_ll_vars_ll_vars,grad_ulp_ll_vars,x0=None, tol=1e-4, maxiter=100)
            lambda_adj = np.reshape(lambda_adj, (-1,1))
            
            # ### Use exact inverse Hessian in the adjoint gradient
            # lambda_adj = np.matmul(np.linalg.inv(hess_LLP_ll_vars_ll_vars),grad_ulp_ll_vars)
            
            bsg = grad_ulp_ul_vars - np.matmul(hess_LLP_ul_vars_ll_vars,lambda_adj)
        else:
            bsg = grad_ulp_ul_vars - ((np.dot(grad_llp_ll_vars.T, grad_ulp_ll_vars)) / np.linalg.norm(grad_llp_ll_vars)**2) * grad_LLP_ul_vars
            
            # print('-----------------')
            # print('1',grad_ulp_ll_vars)           
            # print('2',np.linalg.inv(hess_LLP_ll_vars_ll_vars))
            # print('3', hess_LLP_ul_vars_ll_vars)
            # print('4', grad_ulp_ul_vars)
            # print('5',grad_llp_ll_vars)
            # print('6',grad_LLP_ul_vars)
            # print('true bsg',bsg_t.T,' bsg ',bsg.T)
            # print('-----------------')
            
        # Normalize the direction
        if self.normalize:
            bsg = bsg / np.linalg.norm(bsg, np.inf)
        
        # Update the UL variables
        if self.ul_stepsize_scheme == 2:
            alpha, ll_vars = self.backtracking_armijo_ul(ul_vars, ll_vars, bsg)
            self.ul_lr = alpha
            ul_vars = ul_vars - alpha * bsg
        elif self.ul_stepsize_scheme == 0 or self.ul_stepsize_scheme == 1:   
            ul_vars = ul_vars - self.ul_lr * bsg
            
        # Project the upper-level variables (weights) onto the simplex set
        ul_vars = self.simplex_projection(ul_vars)
        
        return ul_vars, ll_vars
        

    def calc_dx_ulp_risk_neutral(self, ul_vars, ll_vars_array, minibatch_idxs): 
        """
        Updates the UL variables based on the BSG step for the UL problem 
        """ 

        bsg = np.zeros((self.func.x_dim,self.simplex_set_discr.shape[1]))
            
        for count, lam in enumerate(self.simplex_set_discr.T):
            if count in minibatch_idxs:
                lam_ul_vars = np.concatenate((lam.reshape(-1,1), ul_vars), axis=0)
                
                # Gradient of the UL objective function wrt LL variables
                grad_ulp_ll_vars = self.func.grad_fu_ll_vars(lam_ul_vars, ll_vars_array[:,count].reshape(-1,1))
                # Gradient of the UL problem wrt UL variables
                grad_ulp_ul_vars = self.func.grad_fu_ul_vars(lam_ul_vars, ll_vars_array[:,count].reshape(-1,1))
                
                # Gradient of the LL objective function wrt LL variables
                grad_llp_ll_vars = self.func.grad_fl_ll_vars(lam_ul_vars, ll_vars_array[:,count].reshape(-1,1))
                # Grad of the LL problem wrt UL vars
                grad_LLP_ul_vars = self.func.grad_fl_ul_vars(lam_ul_vars, ll_vars_array[:,count].reshape(-1,1))
                
                # Compute the bsg direction
                if self.hess == True:
                    # Hessian of the LL problem wrt UL and LL variables
                    hess_LLP_ul_vars_ll_vars = self.func.hess_fl_ul_vars_ll_vars(lam_ul_vars, ll_vars_array[:,count].reshape(-1,1))
                    # Hessian of the LL problem wrt LL variables
                    hess_LLP_ll_vars_ll_vars = self.func.hess_fl_ll_vars_ll_vars(lam_ul_vars, ll_vars_array[:,count].reshape(-1,1))
                    
                    # lambda_adj, exit_code = cg(hess_LLP_ll_vars_ll_vars,grad_ulp_ll_vars,x0=None, tol=1e-4, maxiter=100)
                    # lambda_adj = np.reshape(lambda_adj, (-1,1))
                    
                    ### Use exact inverse Hessian in the adjoint gradient
                    lambda_adj = np.matmul(np.linalg.inv(hess_LLP_ll_vars_ll_vars),grad_ulp_ll_vars)
                    
                    bsg_aux = grad_ulp_ul_vars - np.matmul(hess_LLP_ul_vars_ll_vars,lambda_adj)
                    bsg[:,count] = bsg_aux[self.func.num_obj:].reshape(1,-1)
                else:
                    bsg_aux = grad_ulp_ul_vars - ((np.dot(grad_llp_ll_vars.T, grad_ulp_ll_vars)) / np.linalg.norm(grad_llp_ll_vars)**2) * grad_LLP_ul_vars
                    bsg[:,count] = bsg_aux[self.func.num_obj:].reshape(1,-1)
                
                # Normalize the direction
                if self.normalize:
                    bsg[:,count] = bsg[:,count] / np.linalg.norm(bsg[:,count], np.inf)
        
        bsg_tot = bsg.mean(axis=1).reshape(-1,1)
        
        # Update the UL variables
        if self.ul_stepsize_scheme == 2: 
            alpha, ll_vars_array = self.backtracking_armijo_ul_risk_neutral(ul_vars, ll_vars_array, bsg_tot, minibatch_idxs)
            self.ul_lr = alpha
            ul_vars = ul_vars - alpha * bsg_tot
        elif self.ul_stepsize_scheme == 0 or self.ul_stepsize_scheme == 1:    
            ul_vars = ul_vars - self.ul_lr * bsg_tot

        # Project if a simple bound exists
        if self.func.projection:
            ul_vars = self.bound_projection(ul_vars)               
        
        return ul_vars, ll_vars_array


    def calc_dx_ulp_risk_averse(self, ul_vars, ll_vars, lagr_mul): 
        """
        Updates the UL variables based on the BSG step for the UL problem 
        """   
        
        # Compute the bsg direction
        if self.hess == True:
            grad_lagrangian = self.func.grad_lagrangian_ul_vars_risk_averse(ul_vars, ll_vars, lagr_mul, hess=True)
        else:
            grad_lagrangian = self.func.grad_lagrangian_ul_vars_risk_averse(ul_vars, ll_vars, lagr_mul, hess=False)
        
        # Normalize the direction
        if self.normalize:
            grad_lagrangian = grad_lagrangian / np.linalg.norm(grad_lagrangian, np.inf)
        
        # Update the UL variables
        if self.ul_stepsize_scheme == 2: 
            alpha, ll_vars = self.backtracking_armijo_ul_risk_averse(ul_vars, ll_vars, grad_lagrangian)
            self.ul_lr = alpha
            ul_vars = ul_vars - alpha * grad_lagrangian
        elif self.ul_stepsize_scheme == 0 or self.ul_stepsize_scheme == 1:    
            ul_vars = ul_vars - self.ul_lr * grad_lagrangian
           
        # Project if a simple bound exists
        if self.func.projection:
            ul_vars = self.bound_projection(ul_vars)    
           
        return ul_vars, ll_vars
    
    
    def simplex_projection(self, ul_vars):
        """
        Computes the projection of the upper-level variables (weights) onto the simplex set
        
        Args:
            ul_vars  ul_vars[:self.func.num_obj]   weights
                     ul_vars[self.func.num_obj:]   upper-level variables
        """ 
        lam_proj = np.maximum(ul_vars[0:self.func.num_obj], np.ones_like(ul_vars[0:self.func.num_obj])*10**-8)
        norm_1 = np.linalg.norm(lam_proj, 1)
        ul_vars[0:self.func.num_obj] = lam_proj/norm_1
        
        # Project if a simple bound exists
        if self.func.projection:
            ul_vars[self.func.num_obj:] = self.bound_projection(ul_vars[self.func.num_obj:])
        return ul_vars
    
    
    def bound_projection(self, ul_vars):
        """
        Computes the projection of the upper-level variables x onto X
        """ 
        ul_vars = np.where(ul_vars <= self.func.ub, ul_vars, self.func.ub)
        ul_vars = np.where(ul_vars >= self.func.lb, ul_vars, self.func.lb)
        return ul_vars
    
    
    def update_llp_true_funct(self, ul_vars, ll_vars, LL_iters_true_funct): 
        """
        Updates the LL problem when considering the true objective function of the bilevel problem
        """        
        # Exact Pareto point y(x) 
        if self.exact_pareto_point:
            lam = ul_vars[:self.func.num_obj]
            xx = ul_vars[self.func.num_obj:]
            # ll_vars = ((lam[0]+lam[1])/(lam[0]+2*lam[1]))*xx + (3*lam[1]/(lam[0]+2*lam[1])) 
            ll_vars = self.func.y_opt_func(lam, xx)
        else:
            for i in range(LL_iters_true_funct):
                # Obtain gradient of the LL objective function wrt LL variables
                grad_llp_ll_vars = self.func.grad_fl_ll_vars(ul_vars, ll_vars) 
                
                # Update the variables
                if self.opt_stepsize: 
                    def obj_funct_2(stepsize):
                        aux_var = ll_vars - stepsize * grad_llp_ll_vars
                        return self.func.f_l(ul_vars, aux_var)
                    res = minimize_scalar(obj_funct_2, bounds=(0, 1), method='bounded')
                    ll_lr = res.x
                else:
                    # Update the variables
                    if self.ll_stepsize_scheme == 0:
                        ll_lr = self.ll_lr_init/(i+1)
                    elif self.ll_stepsize_scheme == 1:
                        ll_lr = self.ll_lr_init
                        # ll_lr = 1
                    elif self.ll_stepsize_scheme == 2:
                        ll_lr = self.backtracking_armijo_ll(ul_vars, ll_vars)   
                ll_vars = ll_vars - ll_lr * grad_llp_ll_vars
                # End if gradient is small enough
                if np.linalg.norm(grad_llp_ll_vars) <= 1e-4:
                    return ll_vars
        
        return ll_vars

    # def true_function_value(self, ul_vars, ll_vars, true_func_list): 
    #     """
    #     Computes the true function value
    #     """
    #     rand_init_ll_vars = ll_vars = np.random.uniform(0, 10, (self.func.y_dim, 1))
    #     ll_vars = self.update_llp_true_funct(ul_vars, rand_init_ll_vars, LL_iters_true_funct=100) 
            
    #     true_func_list.append(self.func.f_u(ul_vars, ll_vars)) 
    #     return true_func_list    
    
    
    def true_function_value(self, ul_vars, ll_vars, true_func_list): 
        """
        Computes the true function value
        """ 
        true_func_list.append(self.func.f(ul_vars)) 
        return true_func_list
    
    
    def true_function_value_risk_neutral(self, ul_vars, ll_vars, true_func_list): 
        """
        Computes the true function value
        """ 
        true_func_list.append(self.func.f_risk_neutral(ul_vars, self.simplex_set_discr)) 
        return true_func_list


    def true_function_value_risk_averse(self, ul_vars, ll_vars, true_func_list): 
        """
        Computes the true function value 
        """ 
        # true_func_list.append(self.func.f_u_risk_averse(ul_vars, ll_vars)) 
        true_func_list.append(self.func.f_risk_averse(ul_vars, ll_vars))
        return true_func_list
    
    
    def backtracking_armijo_ul(self, x, y, d, eta = 10**-4, tau = 0.5):
        initial_rate = 1
        
        alpha_ul = initial_rate
        iterLS = 0
        y_tilde = self.update_llp_true_funct(self.simplex_projection(x - alpha_ul * d), y, LL_iters_true_funct=10) 
        
        sub_obj = np.linalg.norm(d)**2
        while (self.func.f_u(self.simplex_projection(x - alpha_ul * d), y_tilde) > self.func.f_u(x, y) - eta * alpha_ul * sub_obj):                                    
            iterLS = iterLS + 1
            alpha_ul = alpha_ul * tau
            if alpha_ul <= 10**-8:
                alpha_ul = 0 
                y_tilde = y;
                break  
            
            y_tilde = self.update_llp_true_funct(self.simplex_projection(x - alpha_ul * d), y, LL_iters_true_funct=10) 
   
        return alpha_ul, y_tilde
    
    
    def backtracking_armijo_ul_risk_neutral(self, x, y_array, d, minibatch_idxs, eta = 10**-4, tau = 0.5):
        initial_rate = 1
        
        alpha_ul = initial_rate
        iterLS = 0
        
        y_tilde_array = np.copy(y_array)
        
        for count, lam in enumerate(self.simplex_set_discr.T): 
            if count in minibatch_idxs:
                lam_ul_vars = np.concatenate((lam.reshape(-1,1), self.bound_projection(x - alpha_ul * d)), axis=0)
                y_tilde_array[:,count] = self.update_llp_true_funct(lam_ul_vars, y_array[:,count].reshape(-1,1), LL_iters_true_funct=10).reshape(1,-1) # LL_iters_true_funct=10
        
        sub_obj = np.linalg.norm(d)**2

        while (self.func.f_u_risk_neutral(self.bound_projection(x - alpha_ul * d), self.simplex_set_discr, y_tilde_array) > self.func.f_u_risk_neutral(x, self.simplex_set_discr, y_array) - eta * alpha_ul * sub_obj):                                    
            iterLS = iterLS + 1
            alpha_ul = alpha_ul * tau
            if alpha_ul <= 10**-8:
                alpha_ul = 0 
                y_tilde_array = y_array;
                break  
            
            for count, lam in enumerate(self.simplex_set_discr.T):
                if count in minibatch_idxs:
                    lam_ul_vars = np.concatenate((lam.reshape(-1,1), self.bound_projection(x - alpha_ul * d)), axis=0)            
                    y_tilde_array[:,count] = self.update_llp_true_funct(lam_ul_vars, y_array[:,count].reshape(-1,1), LL_iters_true_funct=10).reshape(1,-1)  # LL_iters_true_funct=10
   
        return alpha_ul, y_tilde_array


    def backtracking_armijo_ul_risk_averse(self, x, y, d, eta = 10**-4, tau = 0.5):
        initial_rate = 1
        
        alpha_ul = initial_rate
        iterLS = 0
        
        y_tilde, _ = self.update_llp_risk_averse(self.bound_projection(x - alpha_ul * d), y) 
        
        sub_obj = np.linalg.norm(d)**2
        
        while (self.func.f_u_risk_averse(self.bound_projection(x - alpha_ul * d), y_tilde) > self.func.f_u_risk_averse(x, y) - eta * alpha_ul * sub_obj):                                                
            # print('aa ',self.func.f_u_risk_averse(x - alpha_ul * d, y_tilde),' bb ',self.func.f_u_risk_averse(x, y) - eta * alpha_ul * sub_obj, 'cc', self.func.f_u_risk_averse(x, y), 'dd', eta * alpha_ul * sub_obj)
            iterLS = iterLS + 1
            
            # print('LS ',self.func.f_u_risk_averse(x - alpha_ul * d, y_tilde),self.func.f_u_risk_averse(x, y))
            
            alpha_ul = alpha_ul * tau
            if alpha_ul <= 10**-8:
                alpha_ul = 0 
                y_tilde = y;
                break  
            
            y_tilde, _ = self.update_llp_risk_averse(self.bound_projection(x - alpha_ul * d), y) 
           
        return alpha_ul, y_tilde
    
    
    def backtracking_armijo_ll(self, x, y, eta = 10**-4, tau = 0.5):
        initial_rate = 1
        
        alpha = initial_rate
        iterLS = 0
        
        grad_f_l_y = self.func.grad_fl_ll_vars(x, y) 
        
        while (self.func.f_l(x, y - alpha * grad_f_l_y) > self.func.f_l(x,y) - eta * alpha * np.dot(grad_f_l_y.T, grad_f_l_y)):                                    
            iterLS = iterLS + 1
            alpha = alpha * tau
            if alpha <= 10**-8:
                alpha = 0 
                break 
          
        return alpha     
    
    
    def main_algorithm(self): 
        """
        Bilevel stochastic gradient algorithm for the optimistic formulation of a bilevel multiobjective problem
        """
        cur_time = time.time()    
        
        # Initialize lists
        func_val_list = []
        true_func_list = []
        time_list = [] 
        
        # Initialize the variables
        ul_vars_x = np.random.uniform(0, 2, (self.func.x_dim, 1))
        ll_vars = np.random.uniform(0, 1, (self.func.y_dim, 1))
        ul_vars_lam = np.random.uniform(0, 2, (self.func.num_obj, 1))
        ul_vars = np.concatenate((ul_vars_lam, ul_vars_x), axis=0)
        ul_vars = self.simplex_projection(ul_vars)
        
        # ul_vars = np.array([[0],[1],[1]])
        # ll_vars = np.array([[1]])
        
        # ul_vars = np.array([[0],[1],[1],[1]])
        # ll_vars = np.array([[1],[1]])
        
        # Compute the true function (only for plotting purposes)                
        if self.true_func == True: 
            true_func_list = self.true_function_value(ul_vars, ll_vars, true_func_list) 
        
        end_time =  time.time() - cur_time
        time_list.append(end_time)
        
        self.ul_lr = self.ul_lr_init
        self.ll_lr = self.ll_lr_init
        self.llp_iters = self.llp_iters_init
        
        print('INFO OPT: lambda=',ul_vars[0],ul_vars[1],' x=',ul_vars[2],' y=',ll_vars[0])
        
        j = 1            
        for it in range(self.max_iter): 
            # Check if we stop the algorithm based on time
            if not(self.use_stopping_iter) and (time.time() - cur_time >= self.stopping_time): 
                break
            
            pre_obj_val = self.func.f_u(ul_vars, ll_vars) 
            
            # Update the LL variables with a single step
            ll_orig_vars = ll_vars
            
            ll_vars = self.update_llp(ul_vars, ll_vars)
            
            # Estimate the gradient of the UL objective function
            if self.algo == 'darts':
                ul_vars = self.darts_ulp_grad(ul_vars, ll_vars, ll_orig_vars) 
            elif self.algo == 'bsg':
                ul_vars, ll_vars = self.calc_dx_ulp(ul_vars, ll_vars)
                
            print('INFO OPT: lambda=',ul_vars[0],ul_vars[1],' x=',ul_vars[2],' y=',ll_vars[0])
            
            # print(' fl1 = ',self.func.f_l_1(ul_vars[self.func.num_obj:], ll_vars))
            # print(' fl2 = ',self.func.f_l_2(ul_vars[self.func.num_obj:], ll_vars))
            
            j += 1  
            func_val_list.append(self.func.f_u(ul_vars, ll_vars))
            
            end_time =  time.time() - cur_time
            time_list.append(end_time)

            # Compute the true function (only for plotting purposes)                
            if self.true_func == True: 
                true_func_list = self.true_function_value(ul_vars, ll_vars, true_func_list) 

            # Increasing accuracy strategy            
            if self.inc_acc == True:
             	if self.llp_iters > 30:
              		self.llp_iters = 30
             	else:
                    post_obj_val = self.func.f_u(ul_vars, ll_vars) 
                    obj_val_diff = abs(post_obj_val - pre_obj_val)
                    if obj_val_diff <= 1e-1:  #1e-4              
                        self.llp_iters += 1
            
            # Update the learning rates
            if self.ul_stepsize_scheme == 0:
                self.ul_lr = self.ul_lr_init/j 
            elif self.ul_stepsize_scheme == 1:
                self.ul_lr = self.ul_lr_init 
            if self.ll_stepsize_scheme == 0 or self.ll_stepsize_scheme == 1:
                self.ll_lr = self.ll_lr_init 

            if self.iprint >= 2:
                if self.true_func:
                    print("Algorithm: ",self.algo,' iter: ',it,' f_u: ',func_val_list[len(func_val_list)-1],' f: ',true_func_list[len(true_func_list)-1],' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
                else:
                    print("Algorithm: ",self.algo,' iter: ',it,' f_u: ',func_val_list[len(func_val_list)-1],' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)   
            
        if self.iprint >= 1:
            if self.true_func:
                print("Algorithm: ",self.algo,' f_u: ',func_val_list[len(func_val_list)-1],' f: ',true_func_list[len(true_func_list)-1],' time: ',time.time() - cur_time,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
            else:
                print("Algorithm: ",self.algo,' f_u: ',func_val_list[len(func_val_list)-1],' time: ',time.time() - cur_time,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
        
        return [func_val_list, true_func_list, time_list, ul_vars, ll_vars]
    
    
    def main_algorithm_risk_neutral(self): 
        """
        Bilevel stochastic gradient algorithm for the risk-neutral formulation of a bilevel multiobjective problem
        """
        cur_time = time.time()    
        
        # Initialize lists
        func_val_list = []
        true_func_list = []
        time_list = [] 
        
        lam_1 = np.arange(0, 1.1, 1/self.num_pareto_points_risk_neutral).reshape(1,-1)
        # lam_1 = np.arange(0, 1.1, 0.01).reshape(1,-1)
        # lam_1 = np.array([1, 0]).reshape(1,-1)
        lam_2 = 1-lam_1
        self.simplex_set_discr = np.concatenate((lam_1, lam_2),axis=0) # ex.: [[1,0.8,0.6,...],[0,0.2,0.4,...]]
        
        # Initialize the variables
        ul_vars = np.random.uniform(0, 2, (self.func.x_dim, 1))
        ll_vars_array = np.random.uniform(0, 2, (self.func.y_dim, self.simplex_set_discr.shape[1]))
        
        # ul_vars = np.array([[1], [1]])
        # ll_vars_array = np.ones((self.func.y_dim, self.simplex_set_discr.shape[1]))
        
        loader = self.create_loader_rn()
        stream = self.create_stream_rn(loader)
        
        # Compute the true function (only for plotting purposes)                
        if self.true_func == True: 
            true_func_list =self.true_function_value_risk_neutral(ul_vars, ll_vars_array, true_func_list)
        
        end_time =  time.time() - cur_time
        time_list.append(end_time)
        
        self.ul_lr = self.ul_lr_init
        self.ll_lr = self.ll_lr_init
        self.llp_iters = self.llp_iters_init
        
        ## print('INFO: x=',ul_vars[0],' y=',ll_vars_array[:,0],ll_vars_array[:,1],ll_vars_array[:,2],ll_vars_array[:,3],ll_vars_array[:,4])
        print('INFO RN: x=',ul_vars[0],' y=',ll_vars_array[:,0],ll_vars_array[:,1])      
        
        j = 1            
        for it in range(self.max_iter): 
            # Check if we stop the algorithm based on time
            if not(self.use_stopping_iter) and (time.time() - cur_time >= self.stopping_time): 
                break
            
            minibatch_idxs, stream = self.minibatch_rn(loader, stream)
            
            pre_obj_val = self.func.f_u_risk_neutral(ul_vars, self.simplex_set_discr, ll_vars_array) 
            
            # Update the LL variables with a single step
            ll_orig_vars_array = ll_vars_array
            
            for count, lam in enumerate(self.simplex_set_discr.T):
                if count in minibatch_idxs:
                    lam_ul_vars = np.concatenate((lam.reshape(-1,1), ul_vars), axis=0)
                    ll_vars_array[:,count] = self.update_llp(lam_ul_vars, ll_vars_array[:,count].reshape(-1,1)).reshape(1,-1)
            
            # Estimate the gradient of the UL objective function
            if self.algo == 'darts':
                ul_vars, ll_vars_array = self.darts_ulp_grad_risk_neutral(ul_vars, ll_vars_array, ll_orig_vars_array, minibatch_idxs) 
            elif self.algo == 'bsg':
                ul_vars, ll_vars_array = self.calc_dx_ulp_risk_neutral(ul_vars, ll_vars_array, minibatch_idxs) 
            
            ## print('INFO: x=',ul_vars[0],' y=',ll_vars_array[:,0],ll_vars_array[:,1],ll_vars_array[:,2],ll_vars_array[:,3],ll_vars_array[:,4])
            print('INFO RN: x=',ul_vars[0],' y=',ll_vars_array[:,0],ll_vars_array[:,1])
            
            j += 1 
            func_val_list.append(self.func.f_u_risk_neutral(ul_vars, self.simplex_set_discr, ll_vars_array))
            
            end_time =  time.time() - cur_time
            time_list.append(end_time)

            # Compute the true function (only for plotting purposes)                
            if self.true_func == True: 
                true_func_list = self.true_function_value_risk_neutral(ul_vars, ll_vars_array, true_func_list) 
            
            # Increasing accuracy strategy            
            if self.inc_acc == True:
             	if self.llp_iters > 30:
              		self.llp_iters = 30
             	else:
                    post_obj_val = self.func.f_u_risk_neutral(ul_vars, self.simplex_set_discr, ll_vars_array) 
                    obj_val_diff = abs(post_obj_val - pre_obj_val)
                    if obj_val_diff <= 9*1e-1:  #1e-4              
                        self.llp_iters += 1
            
            # Update the learning rates
            if self.ul_stepsize_scheme == 0:
                self.ul_lr = self.ul_lr_init/j 
            elif self.ul_stepsize_scheme == 1:
                self.ul_lr = self.ul_lr_init 
            if self.ll_stepsize_scheme == 0 or self.ll_stepsize_scheme == 1:
                self.ll_lr = self.ll_lr_init  

            if self.iprint >= 2:
                if self.true_func:
                    print("Algorithm: ",self.algo,' iter: ',it,' f_u: ',func_val_list[len(func_val_list)-1],' f: ',true_func_list[len(true_func_list)-1],' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
                else:
                    print("Algorithm: ",self.algo,' iter: ',it,' f_u: ',func_val_list[len(func_val_list)-1],' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)   
            
        if self.iprint >= 1:
            if self.true_func:
                print("Algorithm: ",self.algo,' f_u: ',func_val_list[len(func_val_list)-1],' f: ',true_func_list[len(true_func_list)-1],' time: ',time.time() - cur_time,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
            else:
                print("Algorithm: ",self.algo,' f_u: ',func_val_list[len(func_val_list)-1],' time: ',time.time() - cur_time,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
        
        return [func_val_list, true_func_list, time_list, ul_vars, ll_vars_array]
    
    
    def main_algorithm_risk_averse(self): 
        """
        Bilevel stochastic gradient algorithm for the risk-averse formulation of a bilevel multiobjective problem
        """
        cur_time = time.time()    
        
        # Initialize lists
        func_val_list = []
        true_func_list = []
        time_list = [] 
        
        # Initialize the variables
        ul_vars = np.random.uniform(0, 2, (self.func.x_dim, 1))
        y_init = np.random.uniform(0, 1, (self.func.y_dim, 1))
        lam_init = np.random.uniform(0, 2, (self.func.num_obj, 1))
        norm_1 = np.linalg.norm(lam_init, 1)
        lam_init = lam_init/norm_1
        ll_vars = np.concatenate((lam_init,y_init), axis=0)
        
        # ul_vars = np.array([[1], [1]])
        # ll_vars = np.array([[0],[1],[1],[1]])
        
        # Compute the true function (only for plotting purposes)                
        if self.true_func == True: 
            true_func_list = self.true_function_value_risk_averse(ul_vars, ll_vars, true_func_list) 
        
        end_time =  time.time() - cur_time
        time_list.append(end_time)
        
        self.ul_lr = self.ul_lr_init
        self.ll_lr = self.ll_lr_init
        self.llp_iters = self.llp_iters_init
        
        print('INFO RA: x=',ul_vars[0],' lambda=',ll_vars[0],ll_vars[1],' y=',ll_vars[2])
        
        j = 1            
        for it in range(self.max_iter): 
            # Check if we stop the algorithm based on time
            if not(self.use_stopping_iter) and (time.time() - cur_time >= self.stopping_time): 
                break
            
            pre_obj_val = self.func.f_u_risk_averse(ul_vars, ll_vars) 
            
            ### # Update the LL variables with a single step
            ### ll_orig_vars = ll_vars
            
            ll_vars, lagr_mul = self.update_llp_risk_averse(ul_vars, ll_vars) 
            
            ### Estimate the gradient of the UL objective function
            ### if self.algo == 'darts':
            ###     ul_vars = self.darts_ulp_grad_risk_averse(ul_vars, ll_vars, ll_orig_vars) 
            ### elif self.algo == 'bsg':
                
            if self.algo == 'bsg':
                ul_vars, ll_vars = self.calc_dx_ulp_risk_averse(ul_vars, ll_vars, lagr_mul) 
                
            print('INFO RA: x=',ul_vars[0],' lambda=',ll_vars[0],ll_vars[1],' y=',ll_vars[2])#,' lagr_mul=',lagr_mul)
            
            j += 1  
            func_val_list.append(self.func.f_u_risk_averse(ul_vars, ll_vars))
            
            end_time =  time.time() - cur_time
            time_list.append(end_time)

            # Compute the true function (only for plotting purposes)                
            if self.true_func == True: 
                true_func_list = self.true_function_value_risk_averse(ul_vars, ll_vars, true_func_list) 

            # Increasing accuracy strategy            
            if self.inc_acc == True:
             	if self.llp_iters > 30:
              		self.llp_iters = 30
             	else:
                    post_obj_val = self.func.f_u_risk_averse(ul_vars, ll_vars) 
                    obj_val_diff = abs(post_obj_val - pre_obj_val)
                    if obj_val_diff <= 1e-1:  #1e-4              
                        self.llp_iters += 1
            
            # Update the learning rates
            if self.ul_stepsize_scheme == 0:
                self.ul_lr = self.ul_lr_init/j 
            elif self.ul_stepsize_scheme == 1:
                self.ul_lr = self.ul_lr_init 
            if self.ll_stepsize_scheme == 0 or self.ll_stepsize_scheme == 1:
                self.ll_lr = self.ll_lr_init 

            if self.iprint >= 2:
                if self.true_func:
                    print("Algorithm: ",self.algo,' iter: ',it,' f_u: ',func_val_list[len(func_val_list)-1],' f: ',true_func_list[len(true_func_list)-1],' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
                else:
                    print("Algorithm: ",self.algo,' iter: ',it,' f_u: ',func_val_list[len(func_val_list)-1],' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)   
            
        if self.iprint >= 1:
            if self.true_func:
                print("Algorithm: ",self.algo,' f_u: ',func_val_list[len(func_val_list)-1],' f: ',true_func_list[len(true_func_list)-1],' time: ',time.time() - cur_time,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
            else:
                print("Algorithm: ",self.algo,' f_u: ',func_val_list[len(func_val_list)-1],' time: ',time.time() - cur_time,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    
        
        return [func_val_list, true_func_list, time_list, ul_vars, ll_vars]
    
    
    def piecewise_func(self, x, boundaries, func_val):
        """
        Computes the value of a piecewise constant function defined by boundaries and func_val at x
        """
        for i in range(len(boundaries)):
            if x <= boundaries[i]:
              return func_val[i]
        return func_val[len(boundaries)-1]
    
    
    def main_algorithm_avg_ci(self, num_rep=1):
        """
        Returns arrays with averages and 95% confidence interval half-widths for function values or true function values at each iteration obtained over multiple runs
        """
        self.set_seed(self.seed) 
        
        # Solve the problem for the first time
        if self.risk_level == 0:
            sol = self.main_algorithm()  
        elif self.risk_level == 1:
            sol = self.main_algorithm_risk_neutral()
        elif self.risk_level == 2:
            sol = self.main_algorithm_risk_averse()
            
        values = sol[0]
        true_func_values = sol[1]
        times = sol[2]
        values_rep = np.zeros((len(values),num_rep))
        values_rep[:,0] = np.asarray(values) 
        if self.true_func:
            true_func_values_rep = np.zeros((len(true_func_values),num_rep))
            true_func_values_rep[:,0] = np.asarray(true_func_values)
            
        # Solve the problem num_rep-1 times
        for i in range(num_rep-1):
          self.set_seed(self.seed+1+i) 
          if self.risk_level == 0:
              sol = self.main_algorithm()  
          elif self.risk_level == 1:
              sol = self.main_algorithm_risk_neutral()
          elif self.risk_level == 2:
              sol = self.main_algorithm_risk_averse() 
              
          if self.use_stopping_iter:
              values_rep[:,i+1] = np.asarray(sol[0])
              if self.true_func:
                  true_func_values_rep[:,i+1] = np.asarray(sol[1])
          else:
              values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[0]),times))
              if self.true_func:
                  true_func_values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[1]),times))
        values_avg = np.mean(values_rep, axis=1)
        values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
        
        if self.true_func:
            true_func_values_avg = np.mean(true_func_values_rep, axis=1)
            true_func_values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(true_func_values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
        else:
            true_func_values_avg = []
            true_func_values_ci = []
          
            
        def compute_integral_risk_neutral(fl1_pareto_value_list, ll_vars_pareto_list, fu_pareto_value_list):
            '''
            Compute the true objective function value of the risk neutral approach by computing the integral with the trapezoidal rule
            '''
            integral = 0
            for ii in range(len(fl1_pareto_value_list)-1):
                interval_length = ll_vars_pareto_list[ii]-ll_vars_pareto_list[ii+1]
                integral = integral + 0.5*(fu_pareto_value_list[ii]+fu_pareto_value_list[ii+1])*interval_length
            return integral    
          
        
        # Obtain the values to compute the lower-level Pareto front
        fl1_pareto_value = []
        fl2_pareto_value = []
        
        if self.risk_level == 0:
            ul_vars = sol[3]
            ll_vars = sol[4]
            fl1_pareto_value_list, fl2_pareto_value_list, fu_pareto_value_list, ul_vars_pareto_list, ll_vars_pareto_list = self.main_wsm(ul_vars[self.func.num_obj:], ll_vars)
            fl1_pareto_value = self.func.ll_prob.f_l_1(ul_vars[self.func.num_obj:], ll_vars)
            fl2_pareto_value = self.func.ll_prob.f_l_2(ul_vars[self.func.num_obj:], ll_vars)
            ul_vars = ul_vars[self.func.num_obj:]
            
            # print('OPT integral',compute_integral_risk_neutral(fl1_pareto_value_list, ll_vars_pareto_list, fu_pareto_value_list))
            
        elif self.risk_level == 1:   
            ul_vars = sol[3]
            ll_vars_array = sol[4]
            fl1_pareto_value_list, fl2_pareto_value_list, fu_pareto_value_list, ul_vars_pareto_list, ll_vars_pareto_list = self.main_wsm(ul_vars, ll_vars_array[:,0])
            ll_vars = []
            
            # print('RN integral',compute_integral_risk_neutral(fl1_pareto_value_list, ll_vars_pareto_list, fu_pareto_value_list))
            
        elif self.risk_level == 2:
            ul_vars = sol[3]
            ll_vars = sol[4]
            fl1_pareto_value_list, fl2_pareto_value_list, fu_pareto_value_list, ul_vars_pareto_list, ll_vars_pareto_list = self.main_wsm(ul_vars, ll_vars[self.func.num_obj:])                
            fl1_pareto_value = self.func.ll_prob.f_l_1(ul_vars, ll_vars[self.func.num_obj:])
            fl2_pareto_value = self.func.ll_prob.f_l_2(ul_vars, ll_vars[self.func.num_obj:])
            ll_vars = ll_vars[self.func.num_obj:]
            
            # print('RA integral',compute_integral_risk_neutral(fl1_pareto_value_list, ll_vars_pareto_list, fu_pareto_value_list))
            
           
        return values_avg, values_ci, true_func_values_avg, true_func_values_ci, times, fl1_pareto_value_list, fl2_pareto_value_list, fl1_pareto_value, fl2_pareto_value, fu_pareto_value_list, ul_vars, ll_vars, ul_vars_pareto_list, ll_vars_pareto_list


    def main_wsm(self, x_init, y_init):
        """
        Returns two lists containing the lower-level objective function values for each point 
        obtained by applying the weighted-sum method to the lower-level problem from the initial point (x_init, y_init)
        """
        num_steps = 10        
        
        f1_value_list = []
        f2_value_list = []
        fu_value_list = []
        x_list = []
        y_list = []
        
        # Define the weights in a simplex set 
        lam_1 = np.arange(0, 1, 0.010).reshape(1,-1)
        lam_2 = 1-lam_1
        simplex_set_discr = np.concatenate((lam_1, lam_2),axis=0) # ex.: [[1,0.8,0.6,...],[0,0.2,0.4,...]]
        
        ## Initialize the variables
        # x_init = np.random.uniform(0, 10, (self.func.x_dim, 1))
        # y_init = np.random.uniform(0, 10, (self.num_starting_pts,self.func.y_dim))       
        
        x_init = x_init.reshape(-1,1)
        y_init = y_init.reshape(-1,1)
              
        # f1_value_list.append(self.func.ll_prob.f_l_1(x_init,y_init))
        # f2_value_list.append(self.func.ll_prob.f_l_2(x_init,y_init))
        # fu_value_list.append(self.func.ul_prob.f_u_(x_init,y_init)) 
        
        for lam in simplex_set_discr.T:
           f1, f2, fu, y = self.weighted_sum_method(x_init, y_init, num_steps, lam)
            
           f1_value_list.append(f1)
           f2_value_list.append(f2)
           fu_value_list.append(fu)
           x_list.append(x_init)
           y_list.append(y)
            
        return f1_value_list, f2_value_list, fu_value_list, x_list, y_list       
  
    
    def weighted_sum_method(self, x_k, y_k, num_steps, lam):
        """
        Returns the lower-level objective function values for the point obtained 
        by applying the weighted sum method from (x_k, y_k) with weights lam
        """
        
        lam_x_k = np.concatenate((lam.reshape(-1,1), x_k), axis=0)
        
        for it in range(num_steps):
            y_k = self.update_llp(lam_x_k, y_k)
        return self.func.ll_prob.f_l_1(x_k, y_k.reshape(-1,1)), self.func.ll_prob.f_l_2(x_k, y_k.reshape(-1,1)), self.func.ul_prob.f_u_(x_k, y_k.reshape(-1,1)), y_k

  
    
  

































########################## OLD ####################################




# class BilevelSolverCL:
#     """
#     Class used to implement bilevel stochastic algorithms for continual learning

#     Attributes
#         ul_lr (real):                       Current upper-level stepsize (default 1)
#         ll_lr (real):                       Current lowel-level stepsize (default 1)
#         llp_iters (int):                    Current number of lower-level steps (default 1)
#         func (obj):                         Object used to define the bilevel problem to solve 
#         algo (str):                         Name of the algorithm to run ('bsg' or 'darts')
#         seed (int, optional):               The seed used for the experiments (default 0)
#         ul_lr_init (real, optional):        Initial upper-level stepsize (default 5)
#         ll_lr_init (real, optional):        Initial lower-level stepsize (default 0.1)
#         max_epochs (int, optional):         Maximum number of epochs (default 1)
#         normalize (bool, optional):         A flag used to normalize the direction used in the update of the upper-level variables (default False)
#         llp_iters_init (real, optional):    Initial number of lower-level steps (default 1)
#         inc_acc (bool, optional):           A flag to use an increasing accuracy strategy for the lower-level problem (default False)
#         true_func (bool, optional):         A flag to compute the true objective function of the bilevel problem (default False)
#         constrained (bool, optional):       A flag to consider lower-level constraints in the bilevel problem 
#         pen_param (bool, optional):         Penalty parameter used in the lower-level penalty function when constrained is True (default 10**(-1))
#         use_stopping_iter (bool, optional): A flag to use the maximum number of epochs as a stopping criterion; if False, the running time is used as a stopping criterion (default True)
#         stopping_times (real, optional):    List of times used when use_stopping_iter is False to determine when a new task must be added to the problem (default [20, 40, 60, 80, 100])
#         iprint (int):                       Used to print info on the optimization (default 1): 0 --> no printing; 1 --> at the end of every task; 2 --> at each iteration 
#         training_loaders:                   List of 5 increasing training datasets (01, 0123, 012345, etc.)
#         testing_loaders:                    List of 5 increasing testing datasets (01, 0123, 012345, etc.) 
#         training_task_loaders:              List of 5 training datasets, each associated with a task
#     """    
    
#     ul_lr = 1
#     ll_lr = 1
#     llp_iters = 1
    
    
#     def __init__(self, func, algo, seed=0, ul_lr=5, ll_lr=0.1, max_epochs=1, normalize=False,\
#                  llp_iters=1, inc_acc=False, true_func=False, constrained=False, \
#                  pen_param = 10**(-1), use_stopping_iter=True, stopping_times=[20, 40, 60, 80, 100],\
#                  iprint = 1):
#         self.func = func
#         self.algo = algo

#         self.seed = seed
#         self.ul_lr_init = ul_lr
#         self.ll_lr_init = ll_lr
#         self.max_epochs = max_epochs
#         self.normalize = normalize
#         self.llp_iters_init = llp_iters
#         self.inc_acc = inc_acc
#         self.true_func = true_func
#         self.constrained = constrained
#         self.pen_param = pen_param        
#         self.use_stopping_iter = use_stopping_iter
#         self.stopping_times = stopping_times
#         self.iprint = iprint
        
#         self.set_seed(self.seed)
#         self.model = self.func.generate_model() 
#         self.training_loaders = self.func.training_loaders
#         self.testing_loaders = self.func.testing_loaders
#         self.training_task_loaders = self.func.training_task_loaders


#     def set_seed(self, seed):
#         """
#         Sets the seed
#         """
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.cuda.manual_seed(seed)
#         torch.backends.cudnn.benchmark = False
#         torch.backends.cudnn.deterministic = True
#         os.environ['PYTHONHASHSEED'] = str(seed) 


#     def minibatch(self, loader, stream):
#         """
#         Returns a mini-batch of data (features and labels) and the updated stream of mini-batches
#         """
#         if len(stream) == 0:
#             stream = self.create_stream(loader)
#         minibatch_feat, minibatch_label = self.rand_samp_no_replace(stream)
#         return minibatch_feat, minibatch_label, stream


#     class flatten(nn.Module):
#         """ 
#         Defines a flattened layer of the DNN 
#         """
#         def forward(self, x):
#             return x.view(x.shape[0], -1)
    
    
#     def yield_2d_tensor(self, model, grad, layer, used):
#         """
#         Repacks a 2-d tensor for a layer in the network
#         """
#         tensor = []
#         for i in range(model[layer].weight.shape[0]):
#             new_used = used + model[layer].weight.shape[1]
#             sub_list = grad[used : new_used]
#             used = new_used
#             tensor.append(sub_list)
#         tensor = torch.from_numpy(np.asarray(tensor)).requires_grad_(True)
#         return tensor, used
    
    
#     def yield_4d_tensor(self, model, grad, layer, used):
#         """
#         Repacks a 4-d tensor for a layer in the network
#         """        
#         tensor = []
#         for i in range(model[layer].weight.shape[0]):
#             sub_tensor = []
#             for j in range(model[layer].weight.shape[1]):
#                 new_used = used + model[layer].weight.shape[2] * model[layer].weight.shape[3]
#                 sub_list = grad[used : new_used]
#                 used = new_used
#                 sub_tensor.append(sub_list.reshape(model[layer].weight.shape[2], model[layer].weight.shape[3]))
#             tensor.append(sub_tensor)
#         tensor = torch.from_numpy(np.asarray(tensor)).requires_grad_(True)
#         return tensor, used
    
    
#     def repack_weights(self, model, grad):
#         """
#         Takes a single vector of weights and restructures it into a tensor of correct dimensions for the network
#         """
#         repacked_list = []
#         used = 0
#         i = 0
#         for param in model.parameters():
#             try:
#                 model[i].weight
#                 if len(model[i].weight.shape) == 4:
#                     tensor, used = self.yield_4d_tensor(model, grad, i, used)
#                     repacked_list.append(tensor.requires_grad_(False).float())
#                 elif len(model[i].weight.shape) == 2:
#                     tensor, used = self.yield_2d_tensor(model, grad, i, used)
#                     repacked_list.append(tensor.requires_grad_(False).float())
#                 elif len(model[i].weight.shape) == 1:
#                     new_used = used + param.shape[0]
#                     tensor = torch.from_numpy(np.asarray(grad[used : new_used]))
#                     repacked_list.append(tensor.requires_grad_(False).float())
#                     used = new_used
#             except:
#                 if len(param.shape) == 2:
#                     tensor = []
#                     for i in range(param.shape[0]):
#                         new_used = used + param.shape[1]
#                         sub_list = grad[used : new_used]
#                         used = new_used
#                         tensor.append(sub_list)
#                     tensor = torch.from_numpy(np.asarray(tensor)).requires_grad_(True)
#                     repacked_list.append(tensor.requires_grad_(False).float())
#                 elif len(param.shape) == 1:
#                     new_used = used + param.shape[0]
#                     tensor = torch.from_numpy(np.asarray(grad[used : new_used]))
#                     repacked_list.append(tensor.requires_grad_(False).float())
#                     used = new_used
#             i += 1
#         # Returns a list of tensors
#         return repacked_list 
    
    
#     def unpack_weights(self, model, grad):
#         """
#         Unpacks a gradient wrt the UL variables, thus yielding a very large single vector of weights
#         """
#         unpacked_list = []
#         i=0
#         for i in range(len(grad)):
#             unpacked_list.append(grad[i].flatten().detach().cpu().numpy())
#         weights_list = np.hstack(np.array(unpacked_list))
#         # Returns a numpy array
#         return weights_list 
    
    
#     def create_stream(self, loader):
#         """
#         Creates a stream of mini-batches of data
#         """
#         stream = random.sample(loader,len(loader)) 
#         random.shuffle(stream) 
#         return stream
    
    
#     def rand_samp_no_replace(self, stream):
#         """
#         Randomly samples (without replacement) from a stream of mini-batches data
#         """
#         aux = [item for list_aux in stream[-1:] for item in list_aux]
#         C, u = aux[0], aux[1]
#         del stream[-1:]
#         return [C, u]
    
    
#     def repack_y_weights(self, y_weights):
#         """
#         Repacks the delta weight vector into a tensor of correct dimensions
#         """
#         repacked_list = []
#         used = 0
#         value = 64*16*16 #This needs to match the number of nodes in the last hidden layer of the DNN
#         for i in range(int(len(y_weights)/value)):
#             new_used = used + value
#             sub_list = y_weights[used : new_used]
#             used = new_used
#             repacked_list.append(sub_list)
#         tensor = torch.stack(repacked_list)
#         return tensor
 
    
#     def update_model_wrt_x(self, model, ulp_grad):
#         """
#         Updates the hidden layer weights and biases
#         """
#         g=0
#         for param in model.parameters():
#             if g == 4:
#                 break
#             param.data = param.data - self.ul_lr * ulp_grad[g].to(self.func.device) 
#             g+=1
#         return model.to(self.func.device)   
 
    
#     def update_model_wrt_y(self, model, y_weights, y_biases):
#         """
#         Updates the output layer weights and biases
#         """
#         f = 0
#         for param in model.parameters():
#             if f == 4:
#                 param.data = self.repack_y_weights(y_weights).float()
#             elif f == 5:
#                 param.data = y_biases.float()
#             f+=1
#         return model.to(self.func.device)
    
    
#     def generate_y(self, model, randomize):
#         """
#         Creates a vector of LL variables 
#         """
#         # Initial model parameters
#         params = [param.data for param in model.parameters()] 
#         y_weights = params[4].flatten()
#         y_biases = params[5]
#         if randomize:
#             y_init_weights = np.random.uniform(-0.02, 0.02, y_weights.shape[0])
#             y_init_biases = np.zeros(y_biases.shape[0]) 
#         else:
#             y_init_weights = np.zeros(y_weights.shape[0])
#             y_init_biases = np.zeros(y_biases.shape[0])
#         # Update the model before returning
#         model = self.update_model_wrt_y(model, torch.from_numpy(y_init_weights), torch.from_numpy(y_init_biases))
#         return [y_init_weights, y_init_biases], model.to(self.func.device)
    
    
#     def inequality_constraint_task(self, id_task, train_task_loader, model, oldmodel, stream_task_training_list):
#         """
#         Computes the value of the inequality constraint associated with a specific task
#         """
#         if len(stream_task_training_list[id_task]) == 0:
#           stream_task_training_list[id_task] = self.create_stream(train_task_loader[id_task])
#         task_train_X, task_train_y = self.rand_samp_no_replace(stream_task_training_list[id_task])
#         C, u = task_train_X.to(self.func.device), task_train_y.to(self.func.device)
#         u = torch.flatten(u)
#         return self.func.loss_func(oldmodel(C), u) - self.func.loss_func(model(C), u)
    
    
#     def inequality_constraint(self, num_constr, train_task_loader, model, oldmodel, stream_task_training_list):
#         """
#         Computes the array of the inequality constraints associated with each task
#         """
#         vec = np.zeros((num_constr,1))
#         for j in range(num_constr):
#           inconstr_value = self.inequality_constraint_task(j, train_task_loader, model, oldmodel, stream_task_training_list)
#           vec[j][0] = inconstr_value
#         return vec
    
    
#     def inequality_constraint_task_grad(self, id_task, train_task_loader, model, oldmodel, stream_task_training_list):
#         """
#         Computes the gradient of the inequality constraint associated with a specific task
#         """
#         output = self.inequality_constraint_task(id_task, train_task_loader, model, oldmodel, stream_task_training_list)
#         # print('constraint value',output)
#         model.zero_grad()
#         #Calculate the gradient
#         output.backward() 
#         grad_model = [param.grad for param in model.parameters()]
#         # Pull out the y variables that we need for the LL problem
#         y_weights_grad = grad_model[4].to(self.func.device)
#         y_biases_grad = grad_model[5].to(self.func.device)
#         grad_y_weights = y_weights_grad.flatten()
#         grad_y_biases = y_biases_grad
#         grad_y = np.concatenate((grad_y_weights.cpu().numpy(), grad_y_biases.cpu().numpy()))
#         # Gradient of the LL objective function wrt x
#         grad_model.pop(5); grad_model.pop(4)
#         grad_x = self.unpack_weights(model, grad_model)
#         return grad_x, grad_y
    
    
#     def inequality_constraint_jacob(self, num_constr, train_task_loader, model, oldmodel, stream_task_training_list):
#         """
#         Computes the Jacobian matrices of the inequality constraints
#         """
#         for j in range(num_constr):
#             grad_x, grad_y = self.inequality_constraint_task_grad(j, train_task_loader, model, oldmodel, stream_task_training_list)
#             grad_x = np.reshape(grad_x, (1,grad_x.shape[0]))
#             grad_y = np.reshape(grad_y, (1,grad_y.shape[0]))
#             if j == 0:
#               jacob_x = grad_x
#               jacob_y = grad_y
#             else:
#               jacob_x = np.concatenate((jacob_x, grad_x), axis=0)
#               jacob_y = np.concatenate((jacob_y, grad_y), axis=0)
    
#         return jacob_x, jacob_y
    
    
#     def update_llp_constr(self, train_loader, train_task_loader, model, oldmodel, num_constr, y, LL_iters, stream_training, stream_task_training_list): 
#         """
#         Updates the LL variables when the LL problem is constrained
#         """        
#         def add_penalty_term(output,model,oldmodel):
#           pen_term = 0
#           for j in range(num_constr):
#             inconstr_value = self.inequality_constraint_task(j, train_task_loader, model, oldmodel, stream_task_training_list)
#             pen_term = pen_term + torch.max(torch.Tensor([0]).to(self.func.device),-(inconstr_value))      
#           output = output + (1/self.pen_param)*pen_term  
#           return output
        
#         y_weights = torch.from_numpy(y[0]).to(self.func.device)
#         y_biases = torch.from_numpy(y[1]).to(self.func.device)
#         for i in range(LL_iters):
#             # Sample a new training batch
#             if len(stream_training) == 0:
#               stream_training = self.create_stream(train_loader)
#             train_X, train_y = self.rand_samp_no_replace(stream_training)
#             C, u = train_X.to(self.func.device), train_y.to(self.func.device)            
#             # Obtain gradient of the LL objective function wrt y at point (x,y)
#             u = torch.flatten(u) 
#             output = self.func.loss_func(model(C), u)
#             output = add_penalty_term(output,model,oldmodel) 
#             model.zero_grad()
#             # Calculate the gradient
#             output.backward() 
#             grad_model = [param.grad for param in model.parameters()]
#             # Pull out the y variables that we need for the LLP
#             y_weights_grad = grad_model[4].to(self.func.device)
#             y_biases_grad = grad_model[5].to(self.func.device)
#             # Update the variables
#             y_weights = y_weights - self.ll_lr * y_weights_grad.flatten()
#             y_biases = y_biases - self.ll_lr * y_biases_grad.flatten()
#             # Update the model before returning
#             model = self.update_model_wrt_y(model, y_weights, y_biases)
#         return [y_weights, y_biases], model.to(self.func.device)
    
    
#     def update_llp(self, train_loader, model, y, LL_iters, stream_training):
#         """
#         Updates the LL variables by taking a gradient descent step for the LL problem
#         """   
#         y_weights = torch.from_numpy(y[0]).to(self.func.device)
#         y_biases = torch.from_numpy(y[1]).to(self.func.device)
#         for i in range(LL_iters):
#             train_X, train_y, stream_training = self.minibatch(train_loader,stream_training)
#             C, u = train_X.to(self.func.device), train_y.to(self.func.device)
#             # Obtain gradient of the LL objective function wrt y at point (x,y)
#             u = torch.flatten(u) 
#             output = self.func.loss_func(model(C), u)
#             model.zero_grad()
#             # Calculate the gradient
#             output.backward() 
#             grad_model = [param.grad for param in model.parameters()]
#             # Pull out the y variables that we need for the LLP
#             y_weights_grad = grad_model[4].to(self.func.device)
#             y_biases_grad = grad_model[5].to(self.func.device)
#             y_weights = y_weights - self.ll_lr * y_weights_grad.flatten()
#             y_biases = y_biases - self.ll_lr * y_biases_grad.flatten()
#             # Update the model before returning
#             model = self.update_model_wrt_y(model, y_weights, y_biases)
#         return [y_weights, y_biases], model.to(self.func.device)
    
    
#     def darts_ulp_grad(self, train_loader, test_loader, model, y, y_orig, stream_training, stream_validation):
#         """
#         Updates the UL variables based on the DARTS step for the UL problem
#         """
#         y_orig = np.concatenate((torch.from_numpy(y_orig[0]).to(self.func.device).cpu(), torch.from_numpy(y_orig[1]).to(self.func.device).cpu()))
#         train_X, train_y, stream_training = self.minibatch(train_loader,stream_training)
#         valid_X, valid_y, stream_validation = self.minibatch(test_loader,stream_validation)
#         C, u = train_X.to(self.func.device), train_y.to(self.func.device)
#         c_valid, u_valid = valid_X.to(self.func.device), valid_y.to(self.func.device)
        
#         # Gradient of the UL objective function wrt y
#         u_valid = torch.flatten(u_valid) 
#         output = self.func.loss_func(model(c_valid), u_valid)
#         model.zero_grad()
#         output.backward()
#         ulp_grad_model = [param.grad for param in model.parameters()]
#         grad_y_prime_weights = ulp_grad_model[4].flatten()
#         grad_y_prime_biases = ulp_grad_model[5]
#         grad_y_prime = np.concatenate((grad_y_prime_weights.cpu().numpy(), grad_y_prime_biases.cpu().numpy()))
#         grad_norm = np.linalg.norm(grad_y_prime)
#         # Remove the delta variables
#         ulp_grad_model.pop(5); ulp_grad_model.pop(4) 
#         grad_x_prime = self.unpack_weights(model, ulp_grad_model)
        
#         ep = 0.01 / grad_norm
#         # Define y+ and y-
#         y_plus = y_orig + ep * grad_y_prime
#         y_plus_biases = y_plus[len(y_plus)-10:]; y_plus_weights = y_plus[:len(y_plus)-10]
#         y_minus = y_orig - ep * grad_y_prime
#         y_minus_biases = y_minus[len(y_minus)-10:]; y_minus_weights = y_minus[:len(y_minus)-10]
        
#         # Gradient of the LL objective function wrt x at the point (x,y+)
#         model = self.update_model_wrt_y(model, torch.from_numpy(y_plus_weights), torch.from_numpy(y_plus_biases)) ######
#         u = torch.flatten(u) 
#         out_plus = self.func.loss_func(model(C), u)
#         model.zero_grad()
#         out_plus.backward()
#         grad_x_plus = [plus_grad.grad for plus_grad in model.parameters()]
#         # Remove the y variables
#         grad_x_plus.pop(5); grad_x_plus.pop(4) 
#         grad_x_plus = self.unpack_weights(model, grad_x_plus)
        
#         # Gradient of the LL objective function wrt x at the point (x,y-)
#         model = self.update_model_wrt_y(model, torch.from_numpy(y_minus_weights), torch.from_numpy(y_minus_biases))
#         u = torch.flatten(u) 
#         out_minus = self.func.loss_func(model(C), u)
#         model.zero_grad()
#         out_minus.backward()
#         grad_x_minus = [minus_grad.grad for minus_grad in model.parameters()]
#         # Remove the y variables
#         grad_x_minus.pop(5); grad_x_minus.pop(4) 
#         grad_x_minus = self.unpack_weights(model, grad_x_minus)
    
#         grad_approx = grad_x_prime - self.ll_lr * ((grad_x_plus - grad_x_minus) / (2 * ep))
#         # Normalize the direction
#         if self.normalize:
#             grad_approx = grad_approx / np.linalg.norm(grad_approx, np.inf)
#         # Repack the weights
#         ulp_grad = self.repack_weights(model, grad_approx) 
#         ulp_grad.pop(5); ulp_grad.pop(4)
        
#         # Update the UL variables
#         model = self.update_model_wrt_x(model, ulp_grad)
#         model = self.update_model_wrt_y(model, y[0], y[1])
#         return model.to(self.func.device)
    
    
#     def calc_dx_ulp(self, train_loader, test_loader, y, model, stream_training, stream_validation):
#         """
#         Updates the UL variables based on the BSG step for the UL problem
#         """        
#         train_X, train_y, stream_training = self.minibatch(train_loader,stream_training)
#         valid_X, valid_y, stream_validation = self.minibatch(test_loader,stream_validation)
#         C, u = train_X.to(self.func.device), train_y.to(self.func.device)
#         c_valid, u_valid = valid_X.to(self.func.device), valid_y.to(self.func.device)
        
#         # Gradient of the UL objective function wrt y
#         u_valid = torch.flatten(u_valid) 
#         output = self.func.loss_func(model(c_valid), u_valid)
#         model.zero_grad()
#         output.backward()
#         ulp_grad_model = [param.grad for param in model.parameters()]
#         grad_y_weights = ulp_grad_model[4].flatten()
#         grad_y_biases = ulp_grad_model[5]
#         ulp_grad_y = np.concatenate((grad_y_weights.cpu().numpy(), grad_y_biases.cpu().numpy()))
#         # Gradient of the UL problem wrt x
#         ulp_grad_model.pop(5); ulp_grad_model.pop(4)
#         ulp_grad_x = self.unpack_weights(model, ulp_grad_model)
        
#         # Gradient of the LL objective function wrt y
#         u = torch.flatten(u)
#         out_prime = self.func.loss_func(model(C), u)
#         model.zero_grad()
#         out_prime.backward()
#         llp_grad_model = [param.grad for param in model.parameters()]
#         LL_grad_y_weights = llp_grad_model[4].flatten()
#         ll_grad_y_biases = llp_grad_model[5]
#         llp_grad_y = np.concatenate((LL_grad_y_weights.cpu().numpy(), ll_grad_y_biases.cpu().numpy()))
#         # Gradient of the LL problem wrt x
#         llp_grad_model.pop(5); llp_grad_model.pop(4)
#         llp_grad_x = self.unpack_weights(model, llp_grad_model)
        
#         # Compute the bsg direction
#         bsg = ulp_grad_x - ((np.dot(llp_grad_y, ulp_grad_y)) / np.linalg.norm(llp_grad_y)**2) * llp_grad_x
        
#         # Normalize the direction
#         if self.normalize:
#             bsg = bsg / np.linalg.norm(bsg, np.inf)
            
#         # Repack the weights
#         ulp_grad = self.repack_weights(model, bsg) 
#         ulp_grad.pop(5); ulp_grad.pop(4)
        
#         # Update the UL variables
#         model = self.update_model_wrt_x(model, ulp_grad)
#         return model.to(self.func.device)
       
    
#     def calc_dx_ulp_constr(self, train_loader, train_task_loader, test_loader, jacob_x, jacob_y, inconstr_vec, lam0, num_constr, lagr_mul,  y, model, oldmodel, batch_num, stream_training, stream_task_training_list, stream_validation):
#         """
#         Updates the UL variables based on the BSG step for the UL problem in the LL constrained case
#         """         
#         def add_Lagrangian_term(output,model,oldmodel,lagr_mul):
#           lagr_term = 0
#           for j in range(num_constr):
#             inconstr_value = self.inequality_constraint_task(j, train_task_loader, model, oldmodel, stream_task_training_list)
#             lagr_mul = torch.tensor(lagr_mul).to(self.func.device)
#             lagr_term = lagr_term + lagr_mul[j]*inconstr_value      
#           return lagr_term
    
#         # Sample new batches
#         if len(stream_training) == 0:
#           stream_training = self.create_stream(train_loader)
#         if len(stream_validation) == 0:
#           stream_validation = self.create_stream(test_loader)
#         train_X, train_y = self.rand_samp_no_replace(stream_training)
#         valid_X, valid_y = self.rand_samp_no_replace(stream_validation)
#         C, u = train_X.to(self.func.device), train_y.to(self.func.device)
#         c_valid, u_valid = valid_X.to(self.func.device), valid_y.to(self.func.device)
        
#         # Gradient of the UL objective function wrt y
#         u_valid = torch.flatten(u_valid) 
#         output = self.func.loss_func(model(c_valid), u_valid)    
#         model.zero_grad()
#         output.backward()
#         ulp_grad_model = [param.grad for param in model.parameters()]
#         grad_y_weights = ulp_grad_model[4].flatten()
#         grad_y_biases = ulp_grad_model[5]
#         ulp_grad_y = np.concatenate((grad_y_weights.cpu().numpy(), grad_y_biases.cpu().numpy()))
#         # Gradient of the UL objective function wrt x
#         ulp_grad_model.pop(5); ulp_grad_model.pop(4)
#         ulp_grad_x = self.unpack_weights(model, ulp_grad_model)
    
#         ulp_grad_x = np.reshape(ulp_grad_x, (ulp_grad_x.shape[0],1)) 
#         ulp_grad_y = np.reshape(ulp_grad_y, (ulp_grad_y.shape[0],1))
    
#         # Gradient of the Lagrangian wrt y
#         u = torch.flatten(u) 
#         out_prime = self.func.loss_func(model(C), u)
#         out_prime = add_Lagrangian_term(out_prime,model,oldmodel,lagr_mul)  
#         model.zero_grad()
#         out_prime.backward()
#         llp_lagr_grad_model = [param.grad for param in model.parameters()]
#         ll_lagr_grad_y_weights = llp_lagr_grad_model[4].flatten()
#         ll_lagr_grad_y_biases = llp_lagr_grad_model[5]
#         llp_lagr_grad_y = np.concatenate((ll_lagr_grad_y_weights.cpu().numpy(), ll_lagr_grad_y_biases.cpu().numpy()))
#         # Gradient of the LL objective function wrt x
#         llp_lagr_grad_model.pop(5); llp_lagr_grad_model.pop(4)
#         llp_lagr_grad_x = self.unpack_weights(model, llp_lagr_grad_model)
    
#         llp_lagr_grad_x = np.reshape(llp_lagr_grad_x, (llp_lagr_grad_x.shape[0],1)) 
#         llp_lagr_grad_y = np.reshape(llp_lagr_grad_y, (llp_lagr_grad_y.shape[0],1))     
    
#         rhs = np.concatenate((ulp_grad_y,np.zeros((lagr_mul.shape[0],1))), axis=0)
#         if batch_num == 0:
#           lam0 = np.random.rand(llp_lagr_grad_y.shape[0] + lagr_mul.shape[0],1)
#         lam = self.gmres(llp_lagr_grad_y, jacob_y, inconstr_vec, rhs, lagr_mul, lam0, max_it=50)
        
#         # Compute the bsg direction
#         bsg = ulp_grad_x - np.matmul(llp_lagr_grad_x,np.matmul(llp_lagr_grad_y.T,lam[0:llp_lagr_grad_y.shape[0]])) - np.matmul(np.multiply(lagr_mul,jacob_x).T,lam[llp_lagr_grad_y.shape[0]:])
       
#         bsg = bsg.flatten()
#         # Normalize the direction
#         if self.normalize:
#             bsg = bsg / np.linalg.norm(bsg, np.inf)
            
#         #Repack the weights
#         ulp_grad = self.repack_weights(model, bsg) 
#         ulp_grad.pop(5); ulp_grad.pop(4)
        
#         # Update the UL variables
#         model = self.update_model_wrt_x(model, ulp_grad)
#         return lam, model.to(self.func.device)
    
    
#     def true_function_value_constr(self, train_loader, training_task_loaders, test_loader, y, model, oldmodel, num_constr, batch, LLP_steps, true_func_list, stream_training, stream_task_training_list, stream_validation): 
#         """
#         Computes the true objective function of the bilevel problem in the LL constrained case
#         """   
#         y_weights_orig = torch.from_numpy(y[0]); y_biases_orig = torch.from_numpy(y[1])
#         # Reinitialize the y variables
#         y, model = self.generate_y(model, randomize = True) 
#         y, model = self.update_llp_constr(train_loader, training_task_loaders, model, oldmodel, num_constr, y, LLP_steps, stream_training, stream_task_training_list) 
        
#         if len(stream_validation) == 0:
#             stream_validation = self.create_stream(test_loader)
#         valid_X, valid_y = self.rand_samp_no_replace(stream_validation)
#         c_valid = valid_X.to(self.func.device); u_valid = valid_y.to(self.func.device)
        
#         u_valid = torch.flatten(u_valid) 
#         true_func_list.append(float(self.func.loss_func(model(c_valid), u_valid)))
#         model = self.update_model_wrt_y(model, y_weights_orig, y_biases_orig)
#         return true_func_list
    
    
#     def linear_cg_kkt(self, vec_1, vec_2, b, max_it, tolcg, lam):
#       """
#       Applies the Linear CG method to solve the KKT system to obtain an array of approximate Lagrange multipliers in the LL constrained case
#       """        
#       def m_times_vec_matrixFree_kkt(vec_1, vec_2, p):            
#         vec = np.matmul(vec_1,np.matmul(vec_1.T,p)) + np.multiply(vec_2**2,p)
#         return vec
    
#       if len(lam) == 0:
#         lambda_vec = b
#         # lambda_vec = np.zeros_like(b)
#       else:
#         lambda_vec = lam
#       tol = 10**(-5)
#       r = b - m_times_vec_matrixFree_kkt(vec_1, vec_2, lambda_vec)
#       beta = []
#       d = []
#       rho_1 = []
#       iflag = 1
#       for iter in range(max_it):
#         z = r
#         rho = np.matmul(r.T,z)
#         if (rho < tol):
#           iflag = 0
#           break
#         if (np.sqrt(rho) <= tolcg):
#           iflag = 0
#           break
#         if (iter > 0):
#           beta = rho / rho_1
#           d = z + beta*d
#         else:
#           d = z
#         q = m_times_vec_matrixFree_kkt(vec_1, vec_2, d)
#         alpha = rho / np.matmul(d.T,q)
#         lambda_vec = lambda_vec + alpha * d
#         r = r - alpha*q
#         rho_1 = rho
#       return lambda_vec
    
    
#     def gmres(self, vec_y_1, vec_y_2, vec_3, b, lagr_mul, x0, max_it):
#         """
#         GMRES method to compute the second term of the BSG direction in the LL constrained case
#         """            
#         def matrix_vector_prod(vec_y_1, vec_y_2, vec_3, lagr_mul, p):
#           pp = copy.deepcopy(p)
#           pp = np.reshape(pp, (-1,1)) 
          
#           aa = pp[0:vec_y_1.shape[0]]
#           bb = pp[vec_y_1.shape[0]:]
    
#           aux_1 = np.array(np.matmul(vec_y_1, np.matmul(vec_y_1.T,aa)) + np.matmul(np.multiply(lagr_mul,vec_y_2).T,bb)) 
#           aux_2 = np.array(np.matmul(vec_y_2,aa) + np.multiply(vec_3,bb)) 
    
#           vec = np.concatenate((aux_1,aux_2),axis=0)
          
#           return vec.flatten()
    
#         r = b.flatten() - np.asarray(matrix_vector_prod(vec_y_1, vec_y_2, vec_3, lagr_mul, x0)).reshape(-1)
#         x = []
#         q = [0] * (max_it)
#         x.append(r)     
#         q[0] = r / np.linalg.norm(r)    
#         h = np.zeros((max_it + 1, max_it))
    
#         for k in range(min(max_it, vec_y_1.shape[0] + lagr_mul.shape[0])):
#             y = np.asarray(matrix_vector_prod(vec_y_1, vec_y_2, vec_3, lagr_mul, q[k])).reshape(-1)
#             for j in range(k+1):
#                 h[j, k] = np.dot(q[j], y)
#                 y = y - h[j, k] * q[j]
#             h[k + 1, k] = np.linalg.norm(y)
#             if (h[k + 1, k] != 0 and k != max_it - 1):
#                 q[k + 1] = y / h[k + 1, k]
    
#             bb = np.zeros(max_it + 1)
#             bb[0] = np.linalg.norm(r)
#             result = np.linalg.lstsq(h, bb)[0]
#             x.append(np.dot(np.asarray(q).transpose(), result) + x0.flatten())
            
#         return np.reshape(x[len(x)-1], (-1,1))
    
    
#     def true_function_value(self, train_loader, test_loader, y, model, llp_steps_true_funct, true_func_list, stream_training, stream_validation):
#         """
#         Computes the true objective function of the bilevel problem 
#         """
#         y_weights_orig = torch.from_numpy(y[0]); y_biases_orig = torch.from_numpy(y[1])
#         # Reinitialize the y variables
#         y, model = self.generate_y(model, randomize = True) 
#         y, model = self.update_llp(train_loader, model, y, 0.0007, llp_steps_true_funct, stream_training)
        
#         valid_X, valid_y, stream_validation = self.minibatch(test_loader,stream_validation)
#         c_valid = valid_X.to(self.func.device); u_valid = valid_y.to(self.func.device)
        
#         u_valid = torch.flatten(u_valid) 
#         true_func_list.append(float(self.func.loss_func(model(c_valid), u_valid)))
#         model = self.update_model_wrt_y(model, y_weights_orig, y_biases_orig)
#         return true_func_list


#     def main_algorithm(self):
#     	"""
#     	Main body of a bilevel stochastic algorithm
#     	"""     
#     	# Initialize lists
#     	func_val_list = []
#     	true_func_list = []
#     	time_list = []
    	
#     	# Timing data
#     	cur_time = time.time()
#     	time_tru_func_tot = 0   
    
#     	# LL constrained case
#     	if self.constrained:
#     		stream_task_training_list = [] 
    
#     	self.ul_lr = self.ul_lr_init
#     	self.ll_lr = self.ll_lr_init
#     	self.llp_iters = self.llp_iters_init
    	
#     	model = self.model #self.func.generate_model() #self.model
    
#     	# Loop through the five tasks
#     	for i in range(len(self.training_loaders)):
#     		istop = 0 
    		
#     		orig_ul_lr = self.ul_lr
#     		orig_ll_lr = self.ll_lr
    		
#     		stream_training = self.create_stream(self.training_loaders[i])
#     		stream_validation = self.create_stream(self.testing_loaders[i])
#     		# First batch
#     		valid_X, valid_y, stream_validation = self.minibatch(self.testing_loaders[i],stream_validation)
#     		C = valid_X.to(self.func.device); u = valid_y.to(self.func.device)
    
#     		# LL constrained case
#     		if self.constrained and i > 0:
#     		  stream_task_training = self.create_stream(self.training_task_loaders[i])
#     		  stream_task_training_list.append(stream_task_training)
#     		  oldmodel = copy.deepcopy(model) 
    		
#     		# Initialize the y variables (output weights of the network)
#     		y, model = self.generate_y(model, randomize = True)
    		
#     		for cur_epoch in range(self.max_epochs):
#     			if istop:
#     			  break
#     			if  cur_epoch == 0:
#     				u = torch.flatten(u) 
#     				func_val_list.append(float(self.func.loss_func(model(C), u)))
#     				time_list.append(time.time() - cur_time - time_tru_func_tot) 
#     				if self.true_func == True: 
#     						cur_time_aux = time.time() 
    
#     						if (self.constrained and i > 0):
#     							num_constr = i
#     							true_func_list = self.true_function_value_constr(self.training_loaders[i], self.training_task_loaders, self.testing_loaders[i], y, model, oldmodel, num_constr, [], 50000, true_func_list, stream_training, stream_task_training_list, stream_validation)
#     						else:
#     							true_func_list = self.true_function_value(self.training_loaders[i], self.testing_loaders[i], y, model, 50000, true_func_list, stream_training, stream_validation)
    						
#     						time_tru_func = time.time() - cur_time_aux 
#     						time_tru_func_tot = time_tru_func_tot + time_tru_func 
    												
#     			j = 1
#     			total_err = 0
#     			for batch in range(len(self.training_loaders[i])):
#     				# Check if a new task must be added
#     				if not(self.use_stopping_iter) and (time.time() - cur_time - time_tru_func_tot) >= self.stopping_times[i]: 
#     					istop = 1
#     					break
    				
#     				valid_X, valid_y, stream_validation = self.minibatch(self.testing_loaders[i],stream_validation)
#     				c_valid = valid_X.to(self.func.device); u_valid = valid_y.to(self.func.device)
#     				u_valid = torch.flatten(u_valid) 
#     				pre_obj_val = float(self.func.loss_func(model(c_valid), u_valid).detach().cpu().numpy())
    				
#     				# Update the LL variables (delta) with a single step
#     				y_orig = y
#     				if (self.constrained and i > 0):
#     				  num_constr = i
#     				  y, model = self.update_llp_constr(self.training_loaders[i], self.training_task_loaders, model, oldmodel, num_constr, y, self.llp_iters, stream_training, stream_task_training_list) 
#     				else:
#     				  y, model = self.update_llp(self.training_loaders[i], model, y, self.llp_iters, stream_training)
    
#     				# LL constrained case
#     				if (self.constrained and i > 0):
#     				  jacob_x, jacob_y = self.inequality_constraint_jacob(i, self.training_task_loaders, model, oldmodel, stream_task_training_list)
#     				  inconstr_vec = self.inequality_constraint(i, self.training_task_loaders, model, oldmodel, stream_task_training_list)
    	
#     				  if len(stream_training) == 0:
#     				    stream_training = self.create_stream(self.training_loaders[i])
#     				  train_X, train_y = self.rand_samp_no_replace(stream_training)
#     				  C, u = train_X.to(self.func.device), train_y.to(self.func.device)
    	
#     				  # Gradient of LL objective function wrt y without penalty
#     				  u = torch.flatten(u) 
#     				  out = self.func.loss_func(model(C), u)  
#     				  model.zero_grad()
#     				  out.backward()
#     				  llp_grad_model = [param.grad for param in model.parameters()]
#     				  LL_grad_y_weights = llp_grad_model[4].flatten()
#     				  ll_grad_y_biases = llp_grad_model[5]
#     				  llp_grad_y = np.concatenate((LL_grad_y_weights.cpu().numpy(), ll_grad_y_biases.cpu().numpy()))
#     				  llp_grad_y = np.reshape(llp_grad_y, (llp_grad_y.shape[0],1))
    	
#     				  if (batch == 0):
#     				  	lagr_mul = []
    	
#     				  lagr_mul = self.linear_cg_kkt(jacob_y, inconstr_vec, -np.matmul(jacob_y,llp_grad_y), 100, 10**-4, lagr_mul)
      				
#     				# Compute the gradient of the UL objective function wrt x (weights of the hidden layers except the last)
#     				if self.algo == 'darts':
#     					model = self.darts_ulp_grad(self.training_loaders[i], self.testing_loaders[i], model, y, y_orig, stream_training, stream_validation)
#     				elif self.algo == 'bsg':
#     				  if self.constrained and i > 0:
#         				if batch == 0:
#         				  lam = []
#         				lam, model = self.calc_dx_ulp_constr(self.training_loaders[i], self.training_task_loaders, self.testing_loaders[i], jacob_x, jacob_y, inconstr_vec, lam, i, lagr_mul, y, model, oldmodel, batch, stream_training, stream_task_training_list, stream_validation) 
#     				  else:
#         				model = self.calc_dx_ulp(self.training_loaders[i], self.testing_loaders[i], y, model, stream_training, stream_validation)	              
    					
#     				j += 1
#     				# Convert y back to numpy arrays
#     				y = [y[0].cpu().numpy(), y[1].cpu().numpy()]
#     				valid_X, valid_y, stream_validation = self.minibatch(self.testing_loaders[i],stream_validation)
#     				c_valid = valid_X.to(self.func.device); u_valid = valid_y.to(self.func.device)
    				
#     				u_pred = model(c_valid).to(self.func.device)
#     				total_err += (u_pred.max(dim=1)[1] != u_valid).sum().item()
#     				u_valid = torch.flatten(u_valid) 
#     				func_val_list.append(float(self.func.loss_func(u_pred, u_valid).detach().cpu().numpy()))
#     				time_list.append(time.time() - cur_time) 
    								
#     				if self.true_func == True: 
#     					time_list.append(time.time() - cur_time  - time_tru_func_tot)
    
#     					cur_time_aux = time.time() 
    					
#     					if (self.constrained and i > 0):
#     					  num_constr = i
#     					  true_func_list = self.true_function_value_constr(self.training_loaders[i], self.training_task_loaders, self.testing_loaders[i], y, model, oldmodel, num_constr, batch, 50000, true_func_list, stream_training, stream_task_training_list, stream_validation)
#     					else:
#     					  true_func_list = self.true_function_value(self.training_loaders[i], self.testing_loaders[i], y, model, 50000, true_func_list, stream_training, stream_validation) 
    
#     					time_tru_func = time.time() - cur_time_aux 
#     					time_tru_func_tot = time_tru_func_tot + time_tru_func                        
    						
#     				if self.inc_acc == True:
#     					if self.llp_iters > 30:
#     						self.llp_iters = 30
#     					else:
#     						u_valid = torch.flatten(u_valid) 
#     						post_obj_val = float(self.func.loss_func(u_pred, u_valid).detach().cpu().numpy())
#     						obj_val_diff = abs(post_obj_val - pre_obj_val)
#     						if obj_val_diff <= 1e-2:
#     							self.llp_iters += 1
    				
#     				# Update the learning rates
#     				if self.algo == 'bsg':
#     					self.ul_lr = orig_ul_lr/j
#     					self.ll_lr = orig_ll_lr
#     				elif self.algo == 'darts':
#     					self.ul_lr = orig_ul_lr/j
#     					self.ll_lr = orig_ll_lr
                    
#     				if self.iprint >= 2:
#     				        print("Algo: ",self.algo," task: ",i,' batch: ',batch,' f_u: ',func_val_list[len(func_val_list)-1],' time: ',time.time() - cur_time - time_tru_func_tot,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)    				
    		
#     		if self.iprint >= 1:
#     			print("Algo: ",self.algo," task: ",i,' f_u: ',func_val_list[len(func_val_list)-1],' time: ',time.time() - cur_time - time_tru_func_tot,' LL iter: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ll_lr: ',self.ll_lr)
    
#     		# string_name = Algo + '_task' + str(i) + '_values.csv'
#     		# pd.DataFrame(func_val_list).to_csv(string_name, index=False)
#     		# string_name = Algo + '_task' + str(i) + '_true_func_vals.csv'
#     		# pd.DataFrame(true_func_list).to_csv(string_name, index=False)
    
#     	return [func_val_list, true_func_list, time_list]
    
    
#     def piecewise_func(self,x,boundaries,func_val):
#       """
#       Computes the value of a piecewise constant function at x
#       """
#       for i in range(len(boundaries)):
#         if x <= boundaries[i]:
#           return func_val[i]
#       return func_val[len(boundaries)-1]
    
    
#     def main_algorithm_avg_ci(self, num_rep=1):
#         """
#         Returns arrays with averages and 95% confidence interval half-widths for function values or true function values at each iteration obtained over multiple runs
#         """ 
#         self.set_seed(self.seed)
#         # Solve the problem for the first time
#         sol = self.main_algorithm() 
#         values = sol[0]
#         true_func_values = sol[1]
#         times = sol[2]
#         values_rep = np.zeros((len(values),num_rep))
#         values_rep[:,0] = np.asarray(values)
#         if self.true_func:
#             true_func_values_rep = np.zeros((len(true_func_values),num_rep))
#             true_func_values_rep[:,0] = np.asarray(true_func_values)
#         # Solve the problem num_rep-1 times
#         for i in range(num_rep-1):
#           self.set_seed(self.seed+1+i)
#           sol = self.main_algorithm() 
#           if self.use_stopping_iter:
#               values_rep[:,i+1] = np.asarray(sol[0])
#               if self.true_func:
#                   true_func_values_rep[:,i+1] = np.asarray(sol[1])
#           else:
#               values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[0]),times))
#               if self.true_func:
#                   true_func_values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[1]),times))
#         values_avg = np.mean(values_rep, axis=1)
#         values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
#         if self.true_func:
#             true_func_values_avg = np.mean(true_func_values_rep, axis=1)
#             true_func_values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(true_func_values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
#         else:
#             true_func_values_avg = []
#             true_func_values_ci = []
             
#         return values_avg, values_ci, true_func_values_avg, true_func_values_ci, times









  
    
    
    
    
    
