#!/usr/bin/env python3
import os
import sys
sys.path.append(os.getcwd()+'/CPDP')
sys.path.append(os.getcwd()+'/JinEnv')
import CPDP
import JinEnv
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio

# ---------------------------------------load environment---------------------------------------
env = JinEnv.Quadrotor()
env.initDyn(Jx=1, Jy=1, Jz=1, mass=1, l=1, c=0.02)
env.initCost2(wthrust=.1)

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
oc.setIntegrator(n_grid=15)

# set initial condition
ini_r_I=[-8, -0, 10.]
ini_v_I = [-3.0, -9.0, 0.0]
ini_q = JinEnv.toQuaternion(0.5, [0, -1, 0])
ini_w = [0.0, 0.0, 0.0]
ini_state=ini_r_I+ini_v_I + ini_q + ini_w

# ---------------------- define the loss function and interface function ------------------
interface_pos_fn = Function('interface', [oc.state], [oc.state[0:3]])
interface_ori_fn = Function('interface', [oc.state], [oc.state[6:10]])
diff_interface_pos_fn = Function('diff_interface', [oc.state], [jacobian(oc.state[0:3], oc.state)])
diff_interface_ori_fn=Function('diff_interface',[oc.state],[jacobian(oc.state[6:10],oc.state)])
def getloss_corrections(time_grid, target_waypoints, opt_sol, auxsys_sol):
    loss = 0
    diff_loss = numpy.zeros(oc.n_auxvar)
    for k, t in enumerate(time_grid):
        # solve loss
        target_waypoint = target_waypoints[k, :]
        target_position=target_waypoint[0:3]
        target_orientation=target_waypoint[3:]
        current_position = interface_pos_fn(opt_sol(t)[0:oc.n_state]).full().flatten()
        current_orientation=interface_ori_fn(opt_sol(t)[0:oc.n_state]).full().flatten()

        loss += numpy.linalg.norm(target_position - current_position) ** 2+\
                numpy.linalg.norm(target_orientation-current_orientation) **2
        # solve gradient by chain rule
        dl_dpos = current_position - target_position
        dpos_dx = diff_interface_pos_fn(opt_sol(t)[0:oc.n_state]).full()
        dxpos_dp = auxsys_sol(t)[0:oc.n_state * oc.n_auxvar].reshape((oc.n_state, oc.n_auxvar))

        dl_dori = current_orientation - target_orientation
        dori_dx = diff_interface_ori_fn(opt_sol(t)[0:oc.n_state]).full()
        dxori_dp = auxsys_sol(t)[0:oc.n_state * oc.n_auxvar].reshape((oc.n_state, oc.n_auxvar))

        dl_dp = np.matmul(numpy.matmul(dl_dpos, dpos_dx), dxpos_dp)+np.matmul(numpy.matmul(dl_dori, dori_dx), dxori_dp)
        diff_loss += dl_dp
    return loss, diff_loss

# --------------------------- create the sparse demonstration from ground-truth----------------------------------------
T = 2
true_parameter = [2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5, 0.5, 0.5,0.5]
true_time_grid, true_opt_sol = oc.cocSolver(ini_state, T, true_parameter)
env.play_animation(wing_len=1,state_traj=true_opt_sol(true_time_grid)[:,:oc.n_state])
taus = true_time_grid[[1, 3, 6, 10, 13]]
waypoints = np.zeros((taus.size, interface_pos_fn.numel_out()+interface_ori_fn.numel_out()))
for k, t in enumerate(taus):
    waypoint_position=interface_pos_fn(true_opt_sol(t)[0:oc.n_state]).full().flatten()
    waypoint_orientation = interface_ori_fn(true_opt_sol(t)[0:oc.n_state]).full().flatten()
    waypoints[k, :] = np.hstack((waypoint_position,waypoint_orientation))


# --------------------------- start the learning process --------------------------------
lr = 1e-2
loss_trace, parameter_trace = [], []
current_parameter = np.array([1.0, 1, 1, 1, 1, 1, 1,1, 1, 1,1])
parameter_trace += [current_parameter.tolist()]
for j in range(int(3000)):
    time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
    auxsys_sol = oc.auxSysSolver(time_grid, opt_sol, current_parameter)
    loss, diff_loss = getloss_corrections(taus, waypoints, opt_sol, auxsys_sol)
    current_parameter -= lr * diff_loss
    # do the projection step
    current_parameter[0] = fmax(current_parameter[0], 0.00000001)
    loss_trace += [loss]
    parameter_trace += [current_parameter.tolist()]
    print('iter:', j, 'loss:', loss_trace[-1].tolist())

# save the results
save_data = {'parameter_trace': parameter_trace,
             'loss_trace': loss_trace,
             'learning_rate': lr,
             'true_parameter':true_parameter,
             'waypoints':waypoints,
             'time_grid':taus,
             'T':T}

# sio.savemat('../data/uav_results.mat', {'results': save_data})