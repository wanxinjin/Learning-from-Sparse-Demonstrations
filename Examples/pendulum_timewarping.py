from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio

# ---------------------------------------load environment---------------------------------------
env = JinEnv.SinglePendulum()
env.initDyn(l=1, m=1, damping_ratio=0.1)
env.initCost(wu=.01)

# --------------------------- create optimal control object ----------------------------------------
oc = CPDP.COCSys_TimeVarying()
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
t = SX.sym('t')
oc.setTimeVariable(t)

beta1 = SX.sym('beta1')
beta2 = SX.sym('beta2')
beta3 = SX.sym('beta3')
beta4 = SX.sym('beta4')

# first-order poly time-warping (uncomment the corresponding projection operator below)
oc.setAuxvarVariable(vcat([beta1, env.cost_auxvar]))
v = beta1
current_parameter =  np.array([1.0,1,1])   # initial guess for all learnable parameters

# second-order poly time-warping (uncomment the corresponding projection operator below)
# oc.setAuxvarVariable(vcat([beta1, beta2, env.cost_auxvar]))
# v = beta1 + 2*beta2 *t
# current_parameter =  np.array([1., 1.,1,1])    # initial guess for all learnable parameters

# thrid-order poly time-warping (uncomment the corresponding projection operator below)
# oc.setAuxvarVariable(vcat([beta1, beta2, beta3, env.cost_auxvar]))
# v = beta1 + 2 * beta2 * t + 3 * beta3 * t**2
# current_parameter =  np.array([1., 1., 1, 1, 1])    # initial guess for all learnable parameters

# fourth-order poly time-warping (uncomment the corresponding projection operator below)
# oc.setAuxvarVariable(vcat([beta1, beta2, beta3, beta4, env.cost_auxvar]))
# v = beta1 + 2 * beta2 * t + 3 * beta3 * t**2 + 4*beta4 *t**3
# current_parameter =  np.array([1., 1., 1, 1, 1, 1])   # initial guess for all learnable parameters

dyn = v * env.f
oc.setDyn(dyn)
path_cost = v * env.path_cost
oc.setPathCost(path_cost)
oc.setFinalCost(env.final_cost)
ini_state = [0.0, 0.0]

# --------------------------- create way points ----------------------------------------
T = 0.2
waypoints = np.array([[0.5],
                      [1.8],
                      [2.0],
                      [2.9],
                      [3.1], ])
time_tau=np.array([0.1, 0.3, 0.6, 0.7, 0.9])/1*T


# ---------------------- define the loss function and interface function ------------------
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

# --------------------------- start the learning process --------------------------------
lr = 5e-3
loss_trace, parameter_trace = [], []
parameter_trace += [current_parameter.tolist()]
for j in range(int(5000)):
    time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
    auxsys_sol = oc.auxSysSolver(time_grid, opt_sol, current_parameter)
    loss, diff_loss = getloss_corrections(time_tau, waypoints, opt_sol, auxsys_sol)
    current_parameter -= lr * diff_loss
    # projection when using first-order poly time-warping function
    current_parameter[0] = fmax(current_parameter[0], 0.00000001)
    # projection when using second-order poly time-warping function
    # current_parameter[0:2] = fmax(current_parameter[0:2], 0.00000001).full().flatten()
    # projection projection when using  third-order poly time-warping function
    # current_parameter[0:3] = fmax(current_parameter[0:3], 0.00000001).full().flatten()
    # projection when using fourth-order poly time-warping function
    # current_parameter[0:4] = fmax(current_parameter[0:4], 0.00000001).full().flatten()
    loss_trace += [loss]
    parameter_trace += [current_parameter.tolist()]
    print('iter:', j, 'loss:', loss_trace[-1].tolist(), 'parameter:', current_parameter)

# save the results
save_data = {'parameter_learned': parameter_trace[-1],
             'loss_learned': loss_trace[-1],
             'learning_rate': lr,
             'waypoints':waypoints,
             'time_grid':time_tau,
             'T':T}
# sio.savemat('../data/pendulum_timepoly1.mat', {'results': save_data})