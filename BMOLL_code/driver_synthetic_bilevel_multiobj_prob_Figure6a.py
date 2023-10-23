import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import functions_bilevel_multiobj as func
import bilevel_multiobj_solver_Figure6 as bls
import time
import pickle 






#--------------------------------------------------#
#-------------- Auxiliary Functions  --------------#
#--------------------------------------------------#

def run_experiment(exp_param_dict, num_rep_value=10):
    """
    Auxiliary function to run the experiments
    
    Args:
        exp_param_dict (dictionary):   Dictionary having some of the attributes of the class SyntheticProblem in functions.py as keys  
        num_rep_value (int, optional): Number of runs for each algorithm (default 10)    
    """
    # if the algorithm is not BSG-RN, we have fewer parameters to consider
    if exp_param_dict['risk_level'] != 1:
        run = bls.BilevelMultiobjSolverSyntheticProb(prob, algo=exp_param_dict['algo'], risk_level=exp_param_dict['risk_level'], \
                                      ul_stepsize_scheme=exp_param_dict['ul_stepsize_scheme'], ul_lr=exp_param_dict['ul_lr'], \
                                      ll_lr=exp_param_dict['ll_lr'], use_stopping_iter=exp_param_dict['use_stopping_iter'], \
                                      max_iter=exp_param_dict['max_iter'], stopping_time=exp_param_dict['stopping_time'], \
                                      inc_acc=exp_param_dict['inc_acc'], hess=exp_param_dict['hess'], normalize=exp_param_dict['normalize'], \
                                      iprint=exp_param_dict['iprint'])
    else:
        run = bls.BilevelMultiobjSolverSyntheticProb(prob, algo=exp_param_dict['algo'], risk_level=exp_param_dict['risk_level'], \
                                      num_pareto_points_risk_neutral=exp_param_dict['num_pareto_points_risk_neutral'], minibatch_size_rn=exp_param_dict['minibatch_size_rn'], \
                                      ul_stepsize_scheme=exp_param_dict['ul_stepsize_scheme'], ul_lr=exp_param_dict['ul_lr'], \
                                      ll_lr=exp_param_dict['ll_lr'], use_stopping_iter=exp_param_dict['use_stopping_iter'], \
                                      max_iter=exp_param_dict['max_iter'], stopping_time=exp_param_dict['stopping_time'], \
                                      inc_acc=exp_param_dict['inc_acc'], hess=exp_param_dict['hess'], normalize=exp_param_dict['normalize'], \
                                      iprint=exp_param_dict['iprint'])

    run_out = run.main_algorithm_avg_ci(num_rep=num_rep_value)
    values_avg = run_out[0]
    values_ci = run_out[1]
    true_func_values_avg = run_out[2]
    true_func_values_ci = run_out[3]
    times = run_out[4]
    
    fl1_pareto_value_list = run_out[5]
    fl2_pareto_value_list = run_out[6]
    fl1_pareto_value = run_out[7]
    fl2_pareto_value = run_out[8]
    fu_pareto_value_list = run_out[9]
    ul_vars_pareto = run_out[10]
    ll_vars_pareto = run_out[11]
    ul_vars_pareto_list = run_out[12]
    ll_vars_pareto_list = run_out[13]

    # pd.DataFrame(values_avg).to_csv(exp_param_dict['algo_full_name'] + '_values_avg.csv', index=False)
    # pd.DataFrame(values_ci).to_csv(exp_param_dict['algo_full_name'] + '_values_ci.csv', index=False)
    # pd.DataFrame(true_func_values_avg).to_csv(exp_param_dict['algo_full_name'] + '_true_func_values_avg.csv', index=False)
    # pd.DataFrame(true_func_values_ci).to_csv(exp_param_dict['algo_full_name'] + '_true_func_values_ci.csv', index=False)
    # pd.DataFrame(times).to_csv(exp_param_dict['algo_full_name'] + '_times.csv', index=False)       
    
    return run, values_avg, values_ci, true_func_values_avg, true_func_values_ci, \
        times, fl1_pareto_value_list, fl2_pareto_value_list, fl1_pareto_value, \
        fl2_pareto_value, fu_pareto_value_list, ul_vars_pareto, ll_vars_pareto, ul_vars_pareto_list, ll_vars_pareto_list


def get_nparray(file_name):
    """
    Auxiliary function to obtain numpy arrays from csv files
    """
    values_avg = pd.read_csv(file_name)
    values_avg = [item for item_2 in values_avg.values.tolist() for item in item_2]
    values_avg = np.array(values_avg)
    return values_avg








#------------------------------------------------#
#-------------- Define the problem --------------#
#------------------------------------------------#

# Dimension of the upper-level problem without lambda
x_dim_val = 50
# Dimension of the lower-level problem 
y_dim_val = 50
# Standard deviation of the stochastic gradient and Hessian estimates
std_dev_val = 0   
hess_std_dev_val = 0              
# Number of objective functions
num_obj_val = 2 # only 2 is supported

prob = func.SyntheticBilevelMultiobjProblem(x_dim=x_dim_val, y_dim=y_dim_val, num_obj=num_obj_val, std_dev=std_dev_val, hess_std_dev=hess_std_dev_val)






#----------------------------------------------------------------------#
#-------------- Parameters common to all the algorithms  --------------#
#----------------------------------------------------------------------#

# A flag to use the total number of iterations as a stopping criterion
use_stopping_iter = True
# Maximum number of iterations
max_iter = 100 #100 #100000
# Maximum running time (in sec) used when use_stopping_iter is False
stopping_time = 0.5 #0.5 #1800
# Number of runs for each algorithm
num_rep_value = 10
# Used to print info on the optimization (default 1): 0 --> no printing; 1 --> at the end of the optimization; 2 --> at each iteration
iprint = 2
# List of colors for the algorithms in the plots
# plot_color_list = ['orange','red','#2ca02c','magenta']
# plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#1f77b4','#2ca02c','#bcbd22']
plot_color_list = ['#1f77b4','#2ca02c','#bcbd22']
# List of line styles for the algorithms in the plots
# plot_linestyle_list = ['solid','dashdot','dashed','dotted']
# plot_linestyle_list = ['solid','solid','solid','dashed','dashed','dashed']
plot_linestyle_list = ['solid','solid','solid']
# List of names for the algorithms in the legends of the plots
plot_legend_list = ['BSG-OPT-H FS','BSG-RN-H FS','BSG-RA-H FS']  
# plot_legend_list = ['BSG-OPT-1 FS','BSG-RN-1 FS','BSG-RA-1 FS','BSG-OPT-H FS','BSG-RN-H FS','BSG-RA-H FS']  
# plot_legend_list = ['BSG-OPT-1 LS','BSG-RN-1 LS','BSG-RA-1 LS','BSG-OPT-H LS','BSG-RN-H LS','BSG-RA-H LS']  
# plot_legend_list = ['BSG-RN-H LS $N=500$ $Q=500$','BSG-RN-H LS $N=500$ $Q=10$', 'BSG-RN-H LS $N=500$ $Q=20$', 'BSG-RN-H LS $N=500$ $Q=40$']  




#--------------------------------------------------------------------#
#-------------- Parameters specific for each algorithm --------------#
#--------------------------------------------------------------------#

# Create a dictionary with parameters for each experiment
exp_param_dict = {}    

### # BSG-1 (1 step)
### exp_param_dict[3] = {'algo': 'bsg', 'algo_full_name': 'bsg', 'ul_lr': 0.1, 'll_lr': 0.1, 'use_stopping_iter': use_stopping_iter, \
###                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': False, 'hess': False, 'normalize': False, 'iprint': iprint}

# # BSG-OPT-1 (inc. acc.)
# exp_param_dict[0] = {'algo': 'bsg', 'risk_level': 0, 'algo_full_name': 'bsgopt1_incacc_FS_std2', \
#                      'ul_stepsize_scheme': 1, 'ul_lr': 1e-1, 'll_lr': 1e-3, 'use_stopping_iter': use_stopping_iter, \
#                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': False, 'normalize': False, 'iprint': iprint, \
#                      'run_exp': True, 'save_output': False}                                                                                                                                                     
                    
# # BSG-RN-1 (inc. acc.)
# exp_param_dict[1] = {'algo': 'bsg', 'risk_level': 1, 'algo_full_name': 'bsgrn1_incacc_FS_std2', 'num_pareto_points_risk_neutral': 500, 'minibatch_size_rn': 20,\
#                      'ul_stepsize_scheme': 1, 'ul_lr': 1e-0, 'll_lr': 1e-3, 'use_stopping_iter': use_stopping_iter, \
#                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': False, 'normalize': False, 'iprint': iprint, \
#                      'run_exp': True, 'save_output': False}
                    
# # BSG-RA-1 (inc. acc.)   'ul_stepsize_scheme': 0, 'ul_lr': 1e-0, 'ul_stepsize_scheme': 1, 'ul_lr': 1e-2
# exp_param_dict[2] = {'algo': 'bsg', 'risk_level': 2, 'algo_full_name': 'bsgra1_incacc_FS_std05', \
#                      'ul_stepsize_scheme': 1, 'ul_lr': 1e-1, 'll_lr': 1e-3, 'use_stopping_iter': use_stopping_iter, \
#                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': False, 'normalize': False, 'iprint': iprint, \
#                      'run_exp': True, 'save_output': False}                                                                                                                                                 

### # BSG-H (1 step)
### exp_param_dict[3] = {'algo': 'bsg', 'algo_full_name': 'bsgh', 'ul_lr': 0.002, 'll_lr': 0.03, 'use_stopping_iter': use_stopping_iter, \
###                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': False, 'hess': True, 'normalize': False, 'iprint': iprint}
                    
# BSG-OPT-H (inc. acc.)
exp_param_dict[0] = {'algo': 'bsg', 'risk_level': 0, 'algo_full_name': 'bsgopth_incacc_FS_std05', \
                     'ul_stepsize_scheme': 1, 'ul_lr': 1e-1, 'll_lr': 1e-3, 'use_stopping_iter': use_stopping_iter, \
                     'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'iprint': iprint, \
                     'run_exp': True, 'save_output': False}
                    
# BSG-RN-H (inc. acc.)
exp_param_dict[1] = {'algo': 'bsg', 'risk_level': 1, 'algo_full_name': 'bsgrnh_incacc_LS_500_mb500_std0_time1800', 'num_pareto_points_risk_neutral': 500, 'minibatch_size_rn': 20,\
                      'ul_stepsize_scheme': 1, 'ul_lr': 1e-0, 'll_lr': 1e-3, 'use_stopping_iter': use_stopping_iter, \
                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'iprint': iprint, \
                      'run_exp': True, 'save_output': False} #bsgrnh_incacc_LS_500_mb500 #bsgrnh_incacc_LS_500_mb500_std0_time1800

# # BSG-RN-H (inc. acc.)
# exp_param_dict[1] = {'algo': 'bsg', 'risk_level': 1, 'algo_full_name': 'bsgrnh_incacc_LS_500_mb10_std0_time1800', 'num_pareto_points_risk_neutral': 500, 'minibatch_size_rn': 10,\
                      # 'ul_stepsize_scheme': 2, 'ul_lr': 1e-1, 'll_lr': 1e-3, 'use_stopping_iter': use_stopping_iter, \
                      # 'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'iprint': iprint, \
                      # 'run_exp': True, 'save_output': True}
    
# # BSG-RN-H (inc. acc.)
# exp_param_dict[2] = {'algo': 'bsg', 'risk_level': 1, 'algo_full_name': 'bsgrnh_incacc_LS_500_mb20_std0_time1800', 'num_pareto_points_risk_neutral': 500, 'minibatch_size_rn': 20,\
                      # 'ul_stepsize_scheme': 2, 'ul_lr': 1e-1, 'll_lr': 1e-3, 'use_stopping_iter': use_stopping_iter, \
                      # 'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'iprint': iprint, \
                      # 'run_exp': True, 'save_output': True}   
 
# # BSG-RN-H (inc. acc.)
# exp_param_dict[3] = {'algo': 'bsg', 'risk_level': 1, 'algo_full_name': 'bsgrnh_incacc_LS_500_mb40_std0_time1800', 'num_pareto_points_risk_neutral': 500, 'minibatch_size_rn': 40,\
                      # 'ul_stepsize_scheme': 2, 'ul_lr': 1e-1, 'll_lr': 1e-3, 'use_stopping_iter': use_stopping_iter, \
                      # 'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'iprint': iprint, \
                      # 'run_exp': True, 'save_output': True}    
    
# BSG-RA-H (inc. acc.)
exp_param_dict[2] = {'algo': 'bsg', 'risk_level': 2, 'algo_full_name': 'bsgrah_incacc_FS_std05', \
                     'ul_stepsize_scheme': 1, 'ul_lr': 1e-1, 'll_lr': 1e-3, 'use_stopping_iter': use_stopping_iter, \
                     'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': True, 'hess': True, 'normalize': False, 'iprint': iprint, \
                     'run_exp': True, 'save_output': False}
                    
### # DARTS
### exp_param_dict[3] = {'algo': 'darts', 'algo_full_name': 'darts', 'ul_lr': 0.2, 'll_lr': 0.3, 'use_stopping_iter': use_stopping_iter, \
###                      'max_iter': max_iter, 'stopping_time': stopping_time, 'inc_acc': False, 'hess': False, 'normalize': False, 'iprint': iprint} 
    
    
    


#--------------------------------------------------------------------#
#-------------- Run the experiments and make the plots --------------#
#--------------------------------------------------------------------#

# Create a dictionary collecting the output for each experiment
exp_out_dict = {}

for i in range(len(exp_param_dict)):
    if i <= 2:
        # if True, run the experiments to obtain exp_out_dict[i]. Otherwise, read exp_out_dict[i]
        if exp_param_dict[i]['run_exp']:
            run, values_avg, values_ci, true_func_values_avg, true_func_values_ci, times, fl1_pareto_value_list, fl2_pareto_value_list, fl1_pareto_value, fl2_pareto_value, fu_pareto_value_list,  \
            ul_vars_pareto, ll_vars_pareto, ul_vars_pareto_list, ll_vars_pareto_list     = run_experiment(exp_param_dict[i], num_rep_value=num_rep_value)
            exp_out_dict[i] = {'run': run, 'values_avg': values_avg, 'values_ci': values_ci,\
                               'true_func_values_avg': true_func_values_avg, 'true_func_values_ci': true_func_values_ci,\
                               'times': times, 'fl1_pareto_value_list': fl1_pareto_value_list, 'fl2_pareto_value_list': fl2_pareto_value_list,\
                               'fl1_pareto_value': fl1_pareto_value, 'fl2_pareto_value': fl2_pareto_value, 'fu_pareto_value_list': fu_pareto_value_list,\
                               'ul_vars_pareto': ul_vars_pareto, 'll_vars_pareto': ll_vars_pareto, 'ul_vars_pareto_list': ul_vars_pareto_list, 'll_vars_pareto_list': ll_vars_pareto_list, \
                               'exp_param_dict_elem': exp_param_dict[i]}
            
            # Save exp_out_dict[i]    
            if exp_param_dict[i]['save_output']: 
                string_dict = r'Dict2/' + exp_param_dict[i]['algo_full_name'] + '_dict.pkl'
                with open(string_dict, 'wb') as f:
                    pickle.dump(exp_out_dict[i], f)

        else:
            string_dict = r'Dict2/' + exp_param_dict[i]['algo_full_name'] + '_dict.pkl'
            with open(string_dict, 'rb') as f:
                exp_out_dict[i] = loaded_dict = pickle.load(f)
          





# # Make the plots

plt.figure()

for i in range(len(exp_out_dict)):
    if exp_out_dict[i]['run'].use_stopping_iter:
        if exp_out_dict[i]['run'].true_func:
            val_x_axis = [i for i in range(len(exp_out_dict[i]['true_func_values_avg']))]
        else:
            val_x_axis = [i for i in range(len(exp_out_dict[i]['values_avg']))]
    else:
        val_x_axis = exp_out_dict[i]['times']
        val_x_axis = [item*10**3 for item in val_x_axis]
    if exp_out_dict[i]['run'].true_func:
        val_y_axis_avg = exp_out_dict[i]['true_func_values_avg'] 
        val_y_axis_ci = exp_out_dict[i]['true_func_values_ci'] 
    else:
        val_y_axis_avg = exp_out_dict[i]['values_avg'] 
        val_y_axis_ci = exp_out_dict[i]['values_ci']        
    # string_legend = r'{0} $\alpha^u = {1}$, $\alpha^\ell = {2}$'.format(plot_legend_list[i],exp_param_dict[i]['ul_lr'],exp_param_dict[i]['ll_lr'])
    string_legend = r'{0}'.format(plot_legend_list[i],exp_param_dict[i]['ul_lr'],exp_param_dict[i]['ll_lr'])
    sns.lineplot(x=val_x_axis, y=val_y_axis_avg, linewidth = 2, label = string_legend, color = plot_color_list[i], linestyle = plot_linestyle_list[i])
    plt.fill_between(val_x_axis, (val_y_axis_avg-val_y_axis_ci), (val_y_axis_avg+val_y_axis_ci), alpha=.4, linewidth = 0.5, color = plot_color_list[i])   

plt.gca().set_ylim([-1.5,2]) 
# plt.gca().set_xlim([0,1.25e6])
plt.gca().set_ylim([-1.8e2,-1.5e2]) 
# The optimal value of the bilevel problem
# plt.hlines(exp_out_dict[0]['run'].func.f_opt(), 0, val_x_axis[len(val_x_axis)-1], color='red', linestyle='dotted') 

if exp_out_dict[0]['run'].use_stopping_iter:
    plt.xlabel("Iterations", fontsize = 20)
else:
    plt.xlabel("Time (ms)", fontsize = 20)
# plt.ylabel("f", fontsize = 20)
plt.tick_params(axis='both', labelsize=11)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

plt.legend(frameon=True) # no borders in the legend
# string = ' standard deviation = ' + str(exp_out_dict[0]['run'].func.std_dev)
string = ' grad std dev = ' + str(exp_out_dict[0]['run'].func.std_dev) + ', Hess std dev = ' + str(exp_out_dict[0]['run'].func.hess_std_dev)
plt.title(string)

fig = plt.gcf()
# fig.set_size_inches(7, 5.5)  
fig.tight_layout()

# Uncomment the next line to save the plot
string = r'D:\Lehigh University Classes\Research\Code\Code for Multiobjective Paper\New Code for Revision 2023\Plots\Figure 6\fig_solution_GKV1_nonsep_stoch_0.pdf'
fig.savefig(string, dpi = 100, format='pdf')








# # Plot the Pareto fronts

# fl1_value_list_dict = {}
# fl2_value_list_dict = {}

# plt.figure()

# i_optimistic = 3
# fl1_value_list_dict[i_optimistic] = exp_out_dict[i_optimistic]['fl1_pareto_value_list']
# fl2_value_list_dict[i_optimistic] = exp_out_dict[i_optimistic]['fl2_pareto_value_list']
# plt.scatter(fl1_value_list_dict[i_optimistic], fl2_value_list_dict[i_optimistic], color='#1f77b4', label='Optimistic Pareto Front', s=10)
# fl1_value = exp_out_dict[i_optimistic]['fl1_pareto_value']
# fl2_value = exp_out_dict[i_optimistic]['fl2_pareto_value']
# plt.scatter(fl1_value, fl2_value, color='red', label='Optimistic Pareto Point', s=10)
# ## plt.xlabel("$f_{\ell}^1$", fontsize = 20)
# ## plt.ylabel("$f_{\ell}^2$", fontsize = 20)
# ## plt.legend()
# ## plt.tight_layout()
# ## plt.show()

# i_risk_neutral = 4
# ## plt.figure()
# fl1_value_list_dict[i_risk_neutral] = exp_out_dict[i_risk_neutral]['fl1_pareto_value_list']
# fl2_value_list_dict[i_risk_neutral] = exp_out_dict[i_risk_neutral]['fl2_pareto_value_list']
# plt.scatter(fl1_value_list_dict[i_risk_neutral], fl2_value_list_dict[i_risk_neutral], color='#2ca02c', label='Risk-Neutral Pareto Front', s=10)
# ## plt.xlabel("$f_{\ell}^1$", fontsize = 20)
# ## plt.ylabel("$f_{\ell}^2$", fontsize = 20)
# ## plt.legend()
# ## plt.tight_layout()
# ## plt.show()

# i_risk_averse = 5
# # plt.figure()
# fl1_value_list_dict[i_risk_averse] = exp_out_dict[i_risk_averse]['fl1_pareto_value_list']
# fl2_value_list_dict[i_risk_averse] = exp_out_dict[i_risk_averse]['fl2_pareto_value_list']
# plt.scatter(fl1_value_list_dict[i_risk_averse], fl2_value_list_dict[i_risk_averse], color='#bcbd22', label='Risk-Averse Pareto Front', s=10)
# fl1_value = exp_out_dict[i_risk_averse]['fl1_pareto_value']
# fl2_value = exp_out_dict[i_risk_averse]['fl2_pareto_value']
# plt.scatter(fl1_value, fl2_value, color='magenta', label='Risk-Averse Pareto Point', s=10)
# # for ii in range(len(exp_out_dict[i_risk_averse]['fl1_pareto_value_list'])):
# #     # print('{0}, {1}, {2}'.format(exp_out_dict[i_risk_averse]['fl1_pareto_value_list'][ii],exp_out_dict[i_risk_averse]['fl2_pareto_value_list'][ii],exp_out_dict[i_risk_averse]['fu_pareto_value_list'][ii]))

# plt.xlabel("$f_{\ell}^1$", fontsize = 20)
# plt.ylabel("$f_{\ell}^2$", fontsize = 20)
# plt.legend()
# plt.tight_layout()
# plt.show()

# # fig = plt.gcf()

# # # Uncomment the next line to save the plot
# # string = 'fig_solution_pareto_fronts.png'
# # fig.savefig(string)







# # Plot the Pareto points

# ul_vars_pareto_list_dict = {}
# ll_vars_pareto_list_dict = {}

# plt.figure()

# i_optimistic = 0
# ul_vars_pareto_list_dict[i_optimistic] = exp_out_dict[i_optimistic]['ul_vars_pareto_list']
# ll_vars_pareto_list_dict[i_optimistic] = exp_out_dict[i_optimistic]['ll_vars_pareto_list']
# plt.scatter(ul_vars_pareto_list_dict[i_optimistic], ll_vars_pareto_list_dict[i_optimistic], color='#1f77b4', label='$\{(x_{OPT},y): y \in P(x_{OPT})\}$', s=10)
# fl1_value = exp_out_dict[i_optimistic]['ul_vars_pareto']
# fl2_value = exp_out_dict[i_optimistic]['ll_vars_pareto']
# plt.scatter(fl1_value, fl2_value, color='red', s=25) #label='$(x_{OPT},y_{OPT})$'
# plt.text(fl1_value, fl2_value+0.4, "$(x_{OPT},y_{OPT})$", horizontalalignment='left', size='large', color='black', weight='extra bold')

# # ## plt.legend()
# # ## plt.tight_layout()
# # ## plt.show()

# i_risk_neutral = 1
# # plt.figure()
# ul_vars_pareto_list_dict[i_risk_neutral] = exp_out_dict[i_risk_neutral]['ul_vars_pareto_list']
# ll_vars_pareto_list_dict[i_risk_neutral] = exp_out_dict[i_risk_neutral]['ll_vars_pareto_list']
# plt.scatter(ul_vars_pareto_list_dict[i_risk_neutral], ll_vars_pareto_list_dict[i_risk_neutral], color='#2ca02c', label='$\{(x_{RN},y): y \in P(x_{RN})\}$', s=10)

# # ## plt.legend()
# # ## plt.tight_layout()
# # ## plt.show()

# i_risk_averse = 2
# # plt.figure()
# ul_vars_pareto_list_dict[i_risk_averse] = exp_out_dict[i_risk_averse]['ul_vars_pareto_list']
# ll_vars_pareto_list_dict[i_risk_averse] = exp_out_dict[i_risk_averse]['ll_vars_pareto_list']
# plt.scatter(ul_vars_pareto_list_dict[i_risk_averse], ll_vars_pareto_list_dict[i_risk_averse], color='#bcbd22', label='$\{(x_{RA},y): y \in P(x_{RA})\}$', s=10)
# fl1_value = exp_out_dict[i_risk_averse]['ul_vars_pareto']
# fl2_value = exp_out_dict[i_risk_averse]['ll_vars_pareto']
# plt.scatter(fl1_value, fl2_value, color='magenta', s=25) #label='$(x_{RA},y_{RA})$'
# plt.text(fl1_value, fl2_value+0.3, "$(x_{RA},y_{RA})$", horizontalalignment='center', size='large', color='black', weight='extra bold')
# ## for ii in range(len(exp_out_dict[i_risk_averse]['fl1_pareto_value_list'])):
# ##     print('{0}, {1}, {2}'.format(exp_out_dict[i_risk_averse]['fl1_pareto_value_list'][ii],exp_out_dict[i_risk_averse]['fl2_pareto_value_list'][ii],exp_out_dict[i_risk_averse]['fu_pareto_value_list'][ii]))

# # Contour plot
# # x_vals = np.arange(-2.0,3.0,0.05) #SP1
# # x_vals = np.arange(-3.0,3.0,0.05) #JOS1
# x_vals = np.arange(-5.5,-1.0,0.05) #GKV1
# y_vals = np.arange(-5.5,3.5,0.05)
# X, Y = np.meshgrid(x_vals, y_vals)
# Z = 3*X + Y + 0.5*X*Y + 0.5*X**2
# cp = plt.contour(X, Y, Z, levels=100, linewidths=0.5)

# # Lines delimiting P(x)
# # plt.plot(x_vals, x_vals, color='black', linestyle='dashed') #SP1
# # plt.plot(x_vals, (lambda x: 0.5*x + 1.5)(x_vals), color='black', linestyle='dashed') #SP1
# # plt.plot(x_vals, (lambda x: 0*x)(x_vals), color='black', linestyle='dashed') #JOS1
# # plt.plot(x_vals, (lambda x: 2 + 0*x)(x_vals), color='black', linestyle='dashed') #JOS1
# plt.plot(x_vals, (lambda x: -0.5*x)(x_vals), color='black', linestyle='dashed') #GKV1
# plt.plot(x_vals, (lambda x: 0.5*x)(x_vals), color='black', linestyle='dashed') #GKV1

# plt.xlabel("$x$", fontsize = 20)
# plt.ylabel("$y$", fontsize = 20)

# plt.legend(frameon=True, fontsize = 12)
# plt.tight_layout()
# plt.show()

# # fig = plt.gcf()

# # # Uncomment the next line to save the plot
# # string = 'fig_solution_2D.png'
# # fig.savefig(string)