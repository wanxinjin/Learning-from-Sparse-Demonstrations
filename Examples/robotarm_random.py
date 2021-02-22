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


# ---------------------------------------load environment-------------------------------
env = JinEnv.RobotArm()
env.initDyn(l1=1,m1=1,l2=1,m2=1,g=0)
env.initCost_Polynomial(wu=.5)

# --------------------------- create optimal control object --------------------------
oc = CPDP.COCSys()
beta = SX.sym('beta')
dyn = beta * env.f
oc.setAuxvarVariable(vertcat(beta, env.cost_auxvar))
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
oc.setDyn(dyn)
path_cost = beta * env.path_cost
oc.setPathCost(path_cost)
oc.setFinalCost(env.final_cost)
oc.setIntegrator(n_grid=30)

# --------------------------- set initial condition and horizon ------------------------------
ini_state = [-pi/2, 0, 0, 0]
T = 1

# ---------------------------- define loss function ------------------------------------
interface_pos_fn = Function('interface', [oc.state], [oc.state[0:2]])
diff_interface_pos_fn = Function('diff_interface', [oc.state], [jacobian(oc.state[0:2], oc.state)])
def getloss_pos_corrections(time_grid, target_waypoints, opt_sol, auxsys_sol):
    loss = 0
    diff_loss = numpy.zeros(oc.n_auxvar)
    for k, t in enumerate(time_grid):
        # solve loss
        target_waypoint = target_waypoints[k, :]
        target_position = target_waypoint[0:2]
        current_position = interface_pos_fn(opt_sol(t)[0:oc.n_state]).full().flatten()

        loss += numpy.linalg.norm(target_position - current_position) ** 2
        # solve gradient by chain rule
        dl_dpos = current_position - target_position
        dpos_dx = diff_interface_pos_fn(opt_sol(t)[0:oc.n_state]).full()
        dxpos_dp = auxsys_sol(t)[0:oc.n_state * oc.n_auxvar].reshape((oc.n_state, oc.n_auxvar))

        dl_dp = np.matmul(numpy.matmul(dl_dpos, dpos_dx), dxpos_dp)
        diff_loss += dl_dp
    return loss, diff_loss

# --------------------------- give the sparse demonstration ----------------------------------------
taus = np.array([0.3 ])
waypoints = np.array([[-pi/4, 2*pi/3]])

# --------------------------- the learning process ----------------------------------------
lr = 1e-1   # learning rate
initial_parameter=np.array([5., 1, 1, 1, 1]) # initial parameter
loss_trace, parameter_trace = [], []
current_parameter = initial_parameter
parameter_trace += [current_parameter.tolist()]
for j in range(int(100)):
    time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
    auxsys_sol = oc.auxSysSolver(time_grid, opt_sol, current_parameter)
    loss, diff_loss = getloss_pos_corrections(taus, waypoints, opt_sol, auxsys_sol)
    current_parameter -= lr * diff_loss
    # do the projection step
    current_parameter[0] = fmax(current_parameter[0],0.00000001)

    loss_trace += [loss]
    parameter_trace += [current_parameter.tolist()]
    print('iter:', j, 'loss:', loss_trace[-1].tolist(), 'parameter:', current_parameter)

# save the results
save_data = {'parameter_trace': parameter_trace,
             'loss_trace': loss_trace,
             'learning_rate': lr,
             'waypoints': waypoints,
             'time_grid': taus,
             'T': T}

# sio.savemat('data/arm_results.mat', {'results': save_data})