#!/usr/bin/env python3
import os
import sys
import time
sys.path.append(os.getcwd()+'/CPDP')
sys.path.append(os.getcwd()+'/JinEnv')
import CPDP
import JinEnv
from casadi import *
import scipy.io as sio


# ---------------------------------------load environment---------------------------------------
env = JinEnv.Quadrotor()
env.initDyn(Jx=.5, Jy=.5, Jz=1, mass=1, l=1, c=0.02)
env.initCost_Polynomial(w_thrust=0.1)

# --------------------------- create optimal control object ----------------------------------------
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
oc.setIntegrator(n_grid=10)

# --------------------------- set initial condition and horizon ------------------------------
# set initial condition
ini_r_I = [-8, -8, 5.]
ini_v_I = [15.5, 5.50, -10.50]
ini_q = JinEnv.toQuaternion(0, [0, 0, 1])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
T = 1

# ---------------------- define the loss function and interface function ------------------
interface_pos_fn = Function('interface', [oc.state], [oc.state[0:3]])
interface_ori_fn = Function('interface', [oc.state], [oc.state[6:10]])
diff_interface_pos_fn = Function('diff_interface', [oc.state], [jacobian(oc.state[0:3], oc.state)])
diff_interface_ori_fn = Function('diff_interface', [oc.state], [jacobian(oc.state[6:10], oc.state)])
def getloss_corrections(time_grid, target_waypoints, opt_sol, auxsys_sol):
    loss = 0
    diff_loss = numpy.zeros(oc.n_auxvar)
    for k, t in enumerate(time_grid):
        # solve loss
        target_waypoint = target_waypoints[k, :]
        target_position = target_waypoint[0:3]
        target_orientation = target_waypoint[3:]
        current_position = interface_pos_fn(opt_sol(t)[0:oc.n_state]).full().flatten()
        current_orientation = interface_ori_fn(opt_sol(t)[0:oc.n_state]).full().flatten()

        loss += numpy.linalg.norm(target_position - current_position) ** 2 + \
                numpy.linalg.norm(target_orientation - current_orientation) ** 2
        # solve gradient by chain rule
        dl_dpos = current_position - target_position
        dpos_dx = diff_interface_pos_fn(opt_sol(t)[0:oc.n_state]).full()
        dxpos_dp = auxsys_sol(t)[0:oc.n_state * oc.n_auxvar].reshape((oc.n_state, oc.n_auxvar))

        dl_dori = current_orientation - target_orientation
        dori_dx = diff_interface_ori_fn(opt_sol(t)[0:oc.n_state]).full()
        dxori_dp = auxsys_sol(t)[0:oc.n_state * oc.n_auxvar].reshape((oc.n_state, oc.n_auxvar))

        dl_dp = np.matmul(numpy.matmul(dl_dpos, dpos_dx), dxpos_dp) + np.matmul(numpy.matmul(dl_dori, dori_dx),
                                                                                dxori_dp)
        diff_loss += dl_dp
    return loss, diff_loss
def getloss_pos_corrections(time_grid, target_waypoints, opt_sol, auxsys_sol):
    loss = 0
    diff_loss = numpy.zeros(oc.n_auxvar)
    for k, t in enumerate(time_grid):
        # solve loss
        target_waypoint = target_waypoints[k, :]
        target_position = target_waypoint[0:3]
        current_position = interface_pos_fn(opt_sol(t)[0:oc.n_state]).full().flatten()

        loss += numpy.linalg.norm(target_position - current_position) ** 2
        # solve gradient by chain rule
        dl_dpos = current_position - target_position
        dpos_dx = diff_interface_pos_fn(opt_sol(t)[0:oc.n_state]).full()
        dxpos_dp = auxsys_sol(t)[0:oc.n_state * oc.n_auxvar].reshape((oc.n_state, oc.n_auxvar))

        dl_dp = np.matmul(numpy.matmul(dl_dpos, dpos_dx), dxpos_dp)
        diff_loss += dl_dp
    return loss, diff_loss


# --------------------------- create different sparse waypionts （just uncomment it) ----------------------------------------
# taus = np.array([0.06666667, 0.2, 0.4, 0.46666667, 0.66666667])
# waypoints = np.array([[-4, -6., 4],
#                       [1, -5., 3],
#                       [1, -1., 4],
#                       [-1.0, 1., 4],
#                       [2.0, 3.0, 4.0]])

taus = np.array([ 0.2, 0.46666667, ])
waypoints = np.array([
                    [1, -5., 3],
                    [-1.0, 1., 4], ])

# --------------------------- start the learning process --------------------------------
# lr = 5e-3
lr = 1e-3
loss_trace, parameter_trace = [], []
current_parameter = np.array([1, .1, .1, .1, .1, .1, -1])
parameter_trace += [current_parameter.tolist()]
for j in range(int(1000)):
    time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
    auxsys_sol = oc.auxSysSolver(time_grid, opt_sol, current_parameter)
    loss, diff_loss = getloss_pos_corrections(taus, waypoints, opt_sol, auxsys_sol)
    current_parameter -= lr * diff_loss
    # do the projection step
    current_parameter[0] = fmax(current_parameter[0], 0.00000001)
    loss_trace += [loss]
    parameter_trace += [current_parameter.tolist()]
    print('iter:', j, 'loss:', loss_trace[-1].tolist())


# Below is to obtain the final uav trajectory based on the learned objective function (under un-warping settings)
horizon = current_parameter[0]*T # note this is the uav actual horizon after warping (T is before warping)
current_parameter[0] = 1  # the learned cost function, but set the time-warping function as unit (un-warping)
_, opt_sol = oc.cocSolver(ini_state, horizon, current_parameter)
time_steps = np.linspace(0, horizon, num=100) # generate the time inquiry grid with N is the point number
opt_traj = opt_sol(time_steps)
opt_state_traj = opt_traj[:, :oc.n_state]  # state trajectory ----- N*[r,v,q,w]
opt_control_traj = opt_traj[:, oc.n_state:oc.n_state+oc.n_control] #control trajectory ---- N*[t1,t2,t3,t4]

# save the results
save_data = {'parameter_trace': parameter_trace,
             'loss_trace': loss_trace,
             'learning_rate': lr,
             'waypoints': waypoints,
             'time_grid': taus,
             'time_steps':time_steps,
             'opt_state_traj': opt_state_traj,
             'opt_control_traj':opt_control_traj,
             'horizon':horizon,
             'T': T}

sio.savemat(os.getcwd() + '/data/uav_results_random_' + time.strftime("%Y%m%d%H%M%S") + '.mat', {'results': save_data})