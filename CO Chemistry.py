import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import solve_ivp

# Conventions

# y is a vector containing the abundance (or relative) of all the gases we want to track.
# y_0 = nCO
# y_1 = xH2 (nuclei ratio, =2*nH2/nH_nuclei)

# Functions

# The ODE
def NL97(y, t, k0, nC, n0, beta, gammaCO, k_coll_e, k_coll_HI, k_coll_He, k_coll_H2, xe, xHe, k_UV, k_LW):
    '''
    Compute dy/dt, the derivative, in the NL97 equation.
    Inputs:
    * t: Time, doesn't matter here, just to make it compatible with scipy solve_ivp.
    * y: The number densities of CO and H2.
    * k0: The rate coefficient for the formation of CHx.
    * nC: The number density of C+ ions.
    * n0: The number density of everything., mostly hydrogen atoms.
    * beta: The proportion of CHx that successfully forms CO.
    * gammaCO: The photodissociation rate of CO
    * k_coll_e: The collisional dissociation rate of e-, as given in Table A1 8-11 of Gloevr & Abel 2008.
    * k_coll_HI: The collisional dissociation rate of H.
    * k_coll_He: The collisional dissociation rate of He.
    * k_coll_H2: The collisional dissociation rate of H2.
    * xe: Relative abundance of e-
    * xHe: Relative abundance of He
    * k_UV: UV photoionization and photodissociation rate
    * k_LW: Lyman-Werner photoionization and photodissociation rate
    * R_d: Formation rate of H2 on dust.
    
    Outputs:
    * dydt: The time derivative of the vector y.
    '''
    
    # Extract the abundances from the array y
    nCO = y[0]
    xH2 = y[1]
    
    # Calculate the derivative
    nH2 = n0 * xH2
    xHI = 1 - xH2
    dy0dt = k0*nC*nH2*beta - gammaCO*nCO
    
    ne = n0 * xe
    nHI = n0 * xHI
    nHe = n0 * xHe
    C_coll = k_coll_e*ne + k_coll_HI*nHI + k_coll_He*nHe + k_coll_H2*nH2 
    dy1dt = -C_coll*xH2 - (k_UV+k_LW)*xH2 + R_d*nHI
    
    # Return
    dydt = np.array([dy0dt, dy1dt])
    return dydt

# The construction part of the ODE
def NL97_c(y, t, k0, nC, n0, beta, gammaCO, k_coll_e, k_coll_HI, k_coll_He, k_coll_H2, xe, xHe, k_UV, k_LW):
    '''
    Compute the construction rate in the NL97 equation. See the docstring for NL97.
    '''
    
    # Extract the abundances from the array y
    nCO = y[0]
    xH2 = y[1]
    
    # Calculate the derivative
    nH2 = n0 * xH2
    xHI = 1 - xH2
    dy0dt = k0*nC*nH2*beta
    
    ne = n0 * xe
    nHI = n0 * xHI
    nHe = n0 * xHe
    C_coll = k_coll_e*ne + k_coll_HI*nHI + k_coll_He*nHe + k_coll_H2*nH2 
    dy1dt = R_d*nHI
    
    # Return
    dydt_c = np.array([dy0dt, dy1dt])
    return dydt_c

# The destruction part of the ODE, DIVIDED BY Y
def NL97_d(y, t, k0, nC, n0, beta, gammaCO, k_coll_e, k_coll_HI, k_coll_He, k_coll_H2, xe, xHe, k_UV, k_LW):
    '''
    Compute the destruction rate in the NL97 equation, DIVIDED BY Y. See the docstring for NL97.
    '''
    
    # Extract the abundances from the array y
    nCO = y[0]
    xH2 = y[1]
    
    # Calculate the derivative
    nH2 = n0 * xH2
    xHI = 1 - xH2
    dy0dt = gammaCO # Absolute value, same below
    
    ne = n0 * xe
    nHI = n0 * xHI
    nHe = n0 * xHe
    C_coll = k_coll_e*ne + k_coll_HI*nHI + k_coll_He*nHe + k_coll_H2*nH2 
    dy1dt = C_coll + (k_UV+k_LW)
    
    # Return
    dydt_d = np.array([dy0dt, dy1dt])
    return dydt_d

# One Step of explicit ODE Solver
def ode_solve_step_explicit(f, y0, t, dt):
    '''
    One step in solving the ODE problem, i.e. dy/dt = f(y,t), explicitly.
    Input:
    * f: a function that receives the current state, y, and the current position/time, t,
    and returns the derivative value of the state, dy/dt.
    * y0: The current value as an array.
    * t: The current time.
    * dt: The time interval to be stepped through.
    
    Output:
    * yf: The final value at t+dt.
    '''
    
    # Step through the most basic algorithm
    yf = y0 + f(y0,t)*dt
    
    # Return
    return yf

# One Step of semi-implicit ODE Solver
def ode_solve_step_semi_implicit(fc, fd, y0, t, dt):
    '''
    One step in solving the ODE problem, i.e. dy/dt = f(y,t), semi-implicitly, using the algorithm in eq. (3) in RAMSES-RTZ.
    Input:
    * fc: The creation part of dydt.
    * fd: The destruction part of dydt.
    * y0: The current value as an array.
    * t: The current time.
    * dt: The time interval to be stepped through.
    
    Output:
    * yf: The final value at t+dt.
    '''
    
    # Step through the semi-implicit algorithm
    yf = (y0 + fc(y0,t)*dt) / (np.full_like(y0, 1) + fd(y0,t)*dt)
    
    # Return
    return yf
    
# ODE Solver
def ode_solve(f, fc, fd, y0, t, method):
    '''
    Solve the ODE problem, i.e. dy/dt = f(y,t).
    Input:
    * f: a function that receives the current state, y, and the current position/time, t,
    and returns the derivative value of the state, dy/dt.
    * fc: The creation part of dydt. None if not applicable.
    * fd: The destruction part of dydt. None if not applicable.
    * y0: The initial value.
    * t: numpy array of time steps with length N where the values of y will be returned.
    * method: The method of integration. Choices:
        'explicit': The most basic algorithm.
        'semi-implicit': The semi-implicit method in RAMSES-RTZ.
    
    Output:
    * y: (M x N) numpy array that contains the values of y at every position/time step. Different columns are different times.
    '''
    
    num_times = t.shape[0] # The dimension of vector t
    num_vars = y0.shape[0] # The number of variables in question
    y = np.zeros((num_vars, num_times))
    y[:, 0] = y0 # Set y at time 0 as y0
    
    # Make the calculation
    if (method == 'explicit'):   
        for i in np.arange(num_times - 1):
            y[:, i+1] = ode_solve_step_explicit(f, y[:, i], t[i], t[i+1] - t[i])
    elif (method == 'semi-implicit'):   
        for i in np.arange(num_times - 1):
            y[:, i+1] = ode_solve_step_semi_implicit(fc, fd, y[:, i], t[i], t[i+1] - t[i])
            
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
# xH2 = 1 # Relative abundance to hydrogen nuclei of molecular hydrogen nuclei
k1 = 5e-10 # Rate coefficient for the formation of CO from O + CHx
G0 = 1.7 # The strength of the ultraviolet radiation field in units of the Habing field
T = 300 # Temperature
xe = 1e-4 # Relative abundance of e-
# xHI = 1 # Relative abundance of H
xHe = 1e-4 # Relative abundance of He
k_UV = 1e-14 # UV photoionization and photodissociation rate
k_LW = 1e-14 # Lyman-Werner photoionization and photodissociation rate
Z = 1 # Metallicity of the gas
C_f = 10 # Clumping factor

# Derived quantities
n = n0 # Assume the number density of hydrogen doesn't change
gammaCHx = 5e-10 * G0 * np.exp(-2.5*AV)
gammaCO = 1e-10 * G0 * np.exp(-2.5*AV)
beta = k1*xO / (k1*xO + gammaCHx / n)
nC = n0 * xC
# nH2 = n0 * xH2
k_coll_e = 4.49e-9 * T**0.11 * np.exp(-101858/T) # k_coll_e-: The collisional dissociation rate of e-, as given in Table A1 8-11 of Gloevr & Abel 2008.
k_coll_HI = 6.67e-12 * T**0.5 * np.exp(-(1+63593/T))# k_coll_HI: The collisional dissociation rate of H.
k_coll_He = 10**(-27.029 + 3.801*np.log10(T) - 29487/T) # k_coll_He: The collisional dissociation rate of He.
k_coll_H2 = 5.996e-30 * T**4.1881 / (1+6.761e-6*T)**5.6881 * np.exp(-54657.4/T) # k_coll_H2: The collisional dissociation rate of H2.
R_d = 3.5e-17 * Z * C_f # Formation of H2 on dust

# Solver parameters
n = 1000 # Number of steps
dt_Myr = 0.01 # Step size in Myr

# Run the explicit solver
dt = dt_Myr * 1e6 * 365 * 86400 # Time in seconds
t0 = 0 # Initial time
tf = t0 + n * dt # Final time
y0 = np.array([0,0]) # Initial abundances (or relative ~)
t = np.linspace(t0, tf, n+1) # Generate the times to report y values
dydt = lambda y, t : NL97(y, t, k0, nC, n0, beta, gammaCO, k_coll_e, k_coll_HI, k_coll_He, k_coll_H2, xe, xHe, k_UV, k_LW)
y = ode_solve(dydt, None, None, y0, t, 'explicit')
t_Myr = sec_to_Myr(t)

# Run the scipy ODE solver using rk45
dydt_scipy = lambda t, y: dydt(y, t) # Swap variables just to make it compatible with scipy's ODE solver
y_scipy = solve_ivp(dydt_scipy, [t0, tf], y0, t_eval=t).y

# Run the semi-implicit solver
dydt_c = lambda y, t : NL97_c(y, t, k0, nC, n0, beta, gammaCO, k_coll_e, k_coll_HI, k_coll_He, k_coll_H2, xe, xHe, k_UV, k_LW)
dydt_d = lambda y, t : NL97_d(y, t, k0, nC, n0, beta, gammaCO, k_coll_e, k_coll_HI, k_coll_He, k_coll_H2, xe, xHe, k_UV, k_LW)
y_semi = ode_solve(dydt, dydt_c, dydt_d, y0, t, 'semi-implicit')
t_Myr = sec_to_Myr(t)

# Plot nCO vs. t using the explicit method
fig_nCO, ax_nCO = plt.subplots()
y_line, = ax_nCO.plot(t_Myr, y[0,:])
ax_nCO.set_xlabel('Time (Myr)')
ax_nCO.set_ylabel('nCO (cm^-3)')

# Plot nCO vs. t using the semi-implicit method
y_semi_line, = ax_nCO.plot(t_Myr, y_semi[0,:])

# Plot nCO vs. t again using scipy
y_scipy_line, = ax_nCO.plot(t_Myr, y_scipy[0,:])

# Add legend
ax_nCO.legend([y_line, y_semi_line, y_scipy_line], ['simple explicit', 'semi-implicit', 'Scipy rk45'])

# Plot xH2 vs. t using the explicit method
fig_xH2, ax_xH2 = plt.subplots()
y_line, = ax_xH2.plot(t_Myr, y[1,:])
ax_xH2.set_xlabel('Time (Myr)')
ax_xH2.set_ylabel('xH2')

# Plot xH2 vs. t using the semi-implicit method
y_semi_line, = ax_xH2.plot(t_Myr, y_semi[1,:])

# Plot xH2 vs. t again using scipy
y_scipy_line, = ax_xH2.plot(t_Myr, y_scipy[1,:])

# Add legend
ax_xH2.legend([y_line, y_semi_line, y_scipy_line], ['simple explicit', 'semi-implicit', 'Scipy rk45'])

# Save the graphs as pdfs
pp = PdfPages('CO Chemistry.pdf')
pp.savefig(fig_nCO)
pp.savefig(fig_xH2)
pp.close()

print ('saved')