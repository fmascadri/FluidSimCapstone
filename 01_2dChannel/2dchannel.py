"""
2D channel flow case

Adapted from Tau Method documentation from Dedalus 
(https://dedalus-project.readthedocs.io/en/latest/pages/tau_method.html) and the 
Periodic Shear Flow example script 
(https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_2d_shear_flow.html)

"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Lz = 4, 1           # [m] fluid domain
Nx, Nz = 256, 64
Schmidt = 1
Re_tau = 180                                # selected value for friction Reynolds
rho = 1e3                                   # density of water
nu = 1.8e-6                                 # kinematic viscosity
delta = Lz/2                                # channel half-height
u_tau = Re_tau * nu / delta                 # friction velocity from selected friction Re number 
print('U_tau = ', u_tau)
tau_w = rho * u_tau**2                      # wall shear stress
pressure_drop_along_channel = (2*tau_w)/(2*delta)
F = (1/rho) * pressure_drop_along_channel   # forcing term for momentum equation

dealias = 3/2
stop_sim_time = 1500                        # flow time  
timestepper = d3.RK222
max_timestep = 10
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))                 # pressure
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))   # velocity
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
tau_p = dist.Field(name='tau_p')

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)             # Chebyshev U basis
lift = lambda A: d3.Lift(A, lift_basis, -1)         # Shortcut for multiplying by U_{N-1}(y)
G = d3.grad(u) - ez*lift(tau_u1)                    # Operator representing G

# Problem
problem = d3.IVP([u, p, tau_u1, tau_u2, tau_p], namespace=locals())
problem.add_equation("trace(G) + tau_p = 0")                                                # continuity
problem.add_equation("dt(u) - nu*div(G) + grad(p) + lift(tau_u2) = -u@grad(u) + F*ex")      # momentum
problem.add_equation("u(z=0) = 0")                                                          # no-slip at bottom wall
problem.add_equation("u(z=Lz) = 0")                                                         # no-slip at top wall
problem.add_equation("integ(p) = 0")                                                        # pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
umax = 0.05                     
u0 = -(4*umax)*(z - Lz/2)**2 + umax         # parabolic (TODO: convert to x^6 exponential initial velocity profile)
for i in range(0,np.shape(u0)[1]):
    u0[0][i] += np.random.random()*0.001
u0[0][0] = 0                                # fix wall velocities to 0 as per no-slip condition
u0[0][-1] = 0
u['g'][0] = u0                              # m/s in x-direction (zeroth index)

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.3, max_writes=50)
snapshots.add_task(u, name='velocity')
snapshots.add_task(p, name='pressure')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)*delta/nu, name='Re')


prev_max_Re = 0

# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            deltaRe = max_Re - prev_max_Re
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f, delta(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re, deltaRe))
            prev_max_Re = max_Re
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()