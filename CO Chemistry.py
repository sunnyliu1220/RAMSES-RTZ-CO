import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Functions

# The ODE
def NL97(y, k0, nC, nH2, beta, gammaCO):
    '''
    Compute dy/dt, the derivative, in the NL97 equation.
    Inputs:
    * y: The number density of CO, nCO.
    * k0: The rate coefficient for the formation of CHx.
    * nC: The number density of C+ ions.
    * nH2: The number density of hydrogen molecules.
    * beta: The proportion of CHx that successfully forms CO.
    * gammaCO: The photodissociation rate of CO
    
    Outputs:
    * dydt: The time derivative of nCO.
    '''
    
    # Calculate the derivative
    dydt = k0*nC*nH2*beta - gammaCO*y
    
    # Return
    return dydt

# One Step of ODE Solver
def ode_solve_step(f, y0, t, dt):
    '''
    One step in solving the ODE problem, i.e. dy/dt = f(y,t).
    Input:
    * f: a function that receives the current state, y, and the current position/time, t,
    and returns the derivative value of the state, dy/dt.
    * y0: The current value.
    * t: The current time.
    * dt: The time interval to be stepped through.
    
    Output:
    * yf: The final value at t+dt.
    '''
    
    # Step through the most basic algorithm
    yf = y0 + f(y0)*dt
    
    # Return
    return yf
    
# ODE Solver
def ode_solve(f, y0, t):
    '''
    Solve the ODE problem, i.e. dy/dt = f(y,t).
    Input:
    * f: a function that receives the current state, y, and the current position/time, t,
    and returns the derivative value of the state, dy/dt.
    * y0: The initial value.
    * t: numpy array of time steps with length N where the values of y will be returned.
    
    Output:
    * y: (1 x N) numpy array that contains the values of y at every position/time step. Columns correspond to time.
    '''
    
    num_times = t.shape[0] # The dimension of vector t
    y = np.zeros((1, num_times))
    y[:, 0] = y0 # Set y at time 0 as y0
    
    # Make the calculation
    for i in np.arange(num_times - 1):
        y[:, i+1] = ode_solve_step(f, y[:, i], t[i], t[i+1] - t[i])
        
    # Return
    return y

# Convert from seconds to Myr
def sec_to_Myr(t):
    '''
    Convert from seconds to Myr.
    Input:
    * t: Numpy array of times in seconds.
    
    Output:
    * t_Myr: Numpy array of times in Myr.
    '''
    
    return t / 1e6 / 365 / 86400

# Inputs
k0 = 5e-16 # cm^3 s^-1
n0 = 100 # Initial number density of hydrogen nuclei
AV = 3.3 # Mean extinction, 3.3 for n0=100
xC = 1.41e-4 # Relative abundance to hydrogen of carbon
xO = 3.16e-4 # Relative abundance to hydrogen of oxygen
xH2 = 0.5 # Relative abundance to hydrogen nuclei of olecular hydrogen
k1 = 5e-10 # Rate coefficient for the formation of CO from O + CHx
G0 = 1.7 # The strength of the ultraviolet radiation field in units of the Habing field

# Derived quantities
n = n0 # Assume the number density of hydrogen doesn't change
gammaCHx = 5e-10 * G0 * np.exp(-2.5*AV)
gammaCO = 1e-10 * G0 * np.exp(-2.5*AV)
beta = k1*xO / (k1*xO + gammaCHx / n)
nC = n0 * xC
nH2 = n0 * xH2

# Solver parameters
n = 1000 # Number of steps
dt_Myr = 0.01 # Step size in Myr
y0 = 0 # Initial nCO

# Run the solver
dt = dt_Myr * 1e6 * 365 * 86400 # Time in seconds
tf = n * dt # Final time
t = np.linspace(0, tf, n+1) # Generate the times to report y values
dydt = lambda y : NL97(y, k0, nC, nH2, beta, gammaCO)
y = ode_solve(dydt, y0, t)
t_Myr = sec_to_Myr(t)

# Plot nCO vs. t
fig, ax = plt.subplots()
y_line = ax.plot(t_Myr, y[0,:])
ax.set_xlabel('Time (Myr)')
ax.set_ylabel('nCO (cm^-3)')

# Save the graphs as pdfs
pp = PdfPages('CO Chemistry.pdf')
pp.savefig(fig)
pp.close()

print ('saved')