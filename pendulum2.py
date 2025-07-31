import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# Pendulum parameters
m = 1.0
l = 1.0
g = 9.81
u_max = 15.0
b = 0.1

def lyapunov(theta, theta_dot):
    kinetic = 0.5 * m * (l * theta_dot)**2
    potential = m * g * l * (1 - np.cos(theta))
    V = np.abs(kinetic + potential - 2 * m * g * l)
    return V

def lyapunov_bang_bang(theta, theta_dot, u_sat=None):
    if u_sat is None:
        u_sat = u_max
    den = m * l * (g * np.cos(theta) + g - 0.5 * l * theta_dot**2)
    c = 0.75
    u_e = c * np.sqrt((np.pi - np.abs(theta % (2*np.pi) - np.pi))**2 + theta_dot**2)
    u_e = np.clip(u_e, 0, u_sat)
    if den < 0:
        u = -u_e
    else:
        u = u_e
    return np.clip(u, -u_sat, u_sat)

def pendulum_ode(t, y):
    theta, theta_dot = y
    theta_norm = ((theta + np.pi) % (2 * np.pi)) - np.pi
    upright_thresh = 0.15
    vel_thresh = 0.5
    if abs(theta_norm - np.pi) < upright_thresh and abs(theta_dot) < vel_thresh:
        Kp = 20.0
        Kd = 5.0
        u = Kp * (np.pi - theta_norm) - Kd * theta_dot
        u = np.clip(u, -u_max, u_max)
    else:
        u = lyapunov_bang_bang(theta, theta_dot)
    theta_ddot = (u - m * g * l * np.sin(theta) - b * theta_dot) / (m * l**2)
    return [theta_dot, theta_ddot]

def time_to_stability(theta0, theta_dot0, V_thresh=0.01, T=40, num_points=4000):
    t_eval = np.linspace(0, T, num_points)
    sol = solve_ivp(pendulum_ode, [0, T], [theta0, theta_dot0], t_eval=t_eval, rtol=1e-8)
    V = np.array([lyapunov(th, thd) for th, thd in zip(sol.y[0], sol.y[1])])
    below = np.where(V < V_thresh)[0]
    if len(below) == 0:
        return None, None, None, sol, V
    for idx in below:
        if np.all(V[idx:] < V_thresh):
            return sol.t[idx], sol.y[0][idx], sol.y[1][idx], sol, V
    return None, None, None, sol, V

# Initial values
init_v0 = 2.0
theta0 = 0.0
V_thresh = 0.01
T = 40

# Multiple initial velocities for batch simulation
initial_velocities = np.linspace(0, 10, 5)  # 5 initial velocities from 0 to 10

# Run initial simulation
t_stab, theta_stab, theta_dot_stab, sol, V = time_to_stability(theta0, init_v0, V_thresh=V_thresh, T=T)

# Set up plot
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.5)

# Plot Lyapunov function decay for main initial condition
line1, = axs[0].plot(sol.t, V, label=f'Initial velocity = {init_v0:.2f}\nTime to stability: {t_stab:.2f}s' if t_stab else 'Did not stabilize')
axs[0].axhline(V_thresh, color='k', linestyle='--', label='Stability threshold')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Lyapunov function V')
axs[0].set_title('Lyapunov Function Decay')
axs[0].legend()

# Plot state variables
line2_theta, = axs[1].plot(sol.t, sol.y[0], label=r'$\theta$ (rad)')
line2_theta_dot, = axs[1].plot(sol.t, sol.y[1], label=r'$\dot{\theta}$ (rad/s)')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('State')
axs[1].set_title('State Variables Over Time')
axs[1].legend()

# Plot Lyapunov derivative (point 1)
V_dot = np.gradient(V, sol.t)
line3, = axs[2].plot(sol.t, V_dot, label=r'Lyapunov function derivative $\dot{V}$')
axs[2].axhline(0, color='k', linestyle='--')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel(r'$\dot{V}$')
axs[2].set_title('Time Derivative of Lyapunov Function')
axs[2].legend()

# Plot multiple initial conditions (point 3)
colors = plt.cm.viridis(np.linspace(0, 1, len(initial_velocities)))
for i, v0 in enumerate(initial_velocities):
    _, _, _, sol_i, V_i = time_to_stability(theta0, v0, V_thresh=V_thresh, T=T)
    axs[0].plot(sol_i.t, V_i, color=colors[i], alpha=0.6, label=f'Init vel {v0:.1f}')

axs[0].legend()

# Slider setup for interactive exploration
axcolor = 'lightgoldenrodyellow'
ax_v0 = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
slider_v0 = Slider(ax_v0, 'Initial Kick', 0.0, 15.0, valinit=init_v0, valstep=0.1)

def update(val):
    v0 = slider_v0.val
    t_stab, theta_stab, theta_dot_stab, sol, V = time_to_stability(theta0, v0, V_thresh=V_thresh, T=T)
    V_dot = np.gradient(V, sol.t)
    line1.set_xdata(sol.t)
    line1.set_ydata(V)
    line2_theta.set_xdata(sol.t)
    line2_theta.set_ydata(sol.y[0])
    line2_theta_dot.set_xdata(sol.t)
    line2_theta_dot.set_ydata(sol.y[1])
    line3.set_xdata(sol.t)
    line3.set_ydata(V_dot)
    axs[0].relim()
    axs[0].autoscale_view()
    axs[1].relim()
    axs[1].autoscale_view()
    axs[2].relim()
    axs[2].autoscale_view()
    fig.canvas.draw_idle()

slider_v0.on_changed(update)

plt.show()
