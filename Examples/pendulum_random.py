#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/CPDP')
sys.path.append(os.getcwd()+'/JinEnv')
sys.path.append(os.getcwd()+'/lib')
import CPDP
import JinEnv
from casadi import *


# ---------------------------------------load environment---------------------------------------
env = JinEnv.SinglePendulum()
env.initDyn(l=1, m=1, damping_ratio=0.1)
env.initCost(wu=.01)

# --------------------------- create optimal control object ----------------------------------------
oc = CPDP.COCSys()
beta=SX.sym('beta')
dyn=beta*env.f
oc.setAuxvarVariable(vertcat(beta,env.cost_auxvar))
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
oc.setDyn(dyn)
path_cost=beta*env.path_cost
oc.setPathCost(path_cost)
oc.setFinalCost(env.final_cost)
# set initial condition
ini_state = [0.0, 0.0]


# ---------------------- define the loss function and interface function ------------------=
interface_fn = Function('interface', [oc.state], [oc.state[0]])
diff_interface_fn = Function('diff_interface', [oc.state], [jacobian(oc.state[0], oc.state)])
def getloss_corrections(time_grid, waypoints, opt_sol, auxsys_sol):
    loss = 0
    diff_loss = numpy.zeros(oc.n_auxvar)
    for k,t in enumerate(time_grid):
        # solve loss
        waypoint = waypoints[k,:]
        measure = interface_fn(opt_sol(t)[0:oc.n_state]).full().flatten()
        loss += numpy.linalg.norm(waypoint - measure) ** 2
        # solve gradient by chain rule
        dl_dy=measure-waypoint
        dy_dx=diff_interface_fn(opt_sol(t)[0:oc.n_state]).full()
        dx_dp=auxsys_sol(t)[0:oc.n_state * oc.n_auxvar].reshape((oc.n_state, oc.n_auxvar))
        dl_dp=np.matmul(numpy.matmul(dl_dy,dy_dx),dx_dp)
        diff_loss+=dl_dp
    return loss, diff_loss


# --------------------------- create way points randomly ----------------------------------------
T = 1
time_tau=np.array([0.1, 0.3, 0.6, 0.7, 0.9])
waypoints = np.array([[0.5],
                      [1.8],
                      [2.0],
                      [2.9],
                      [3.1], ])


# --------------------------- start the learning process --------------------------------
lr = 1e-2
loss_trace, parameter_trace = [], []
current_parameter = np.array([1,0.5,1.5])
parameter_trace+=[current_parameter.tolist()]
for j in range(int(100)):
    # initial guess of trajectory based on initial parameters
    time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
    # # Establish the auxiliary control system
    auxsys_sol = oc.auxSysSolver(time_grid, opt_sol, current_parameter)
    # Use the chain rule
    loss, diff_loss = getloss_corrections(time_tau, waypoints, opt_sol, auxsys_sol)
    # update
    current_parameter -= lr * diff_loss
    current_parameter[0] = fmax(current_parameter[0], 0.00000001)  # projection
    loss_trace += [loss]
    parameter_trace += [current_parameter.tolist()]
    # print
    print('iter:', j, 'loss:', loss_trace[-1].tolist(), 'parameter:', current_parameter)


# save the results
save_data = {'parameter_trace': parameter_trace,
             'loss_trace': loss_trace,
             'learning_rate': lr,
             'waypoints':waypoints,
             'time_grid':time_tau,
             'T':T}

# sio.savemat('../data/pendulum_results_random_2.mat', {'results': save_data})