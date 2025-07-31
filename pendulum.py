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

def lyapunov_bang_bang(theta, theta_dot, u_sat=u_max):
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

# --- Matplotlib interactive plot with slider ---

# Initial values
init_v0 = 2.0
theta0 = 0.0
V_thresh = 0.01
T = 40

# Run initial simulation
t_stab, theta_stab, theta_dot_stab, sol, V = time_to_stability(theta0, init_v0, V_thresh=V_thresh, T=T)

# Set up plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.4)

[line1] = ax1.plot(sol.t, V, label=f'Initial velocity = {init_v0:.2f}\nTime to stability: {t_stab:.2f}s' if t_stab else 'Did not stabilize')
ax1.axhline(V_thresh, color='k', linestyle='--', label='Stability threshold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Lyapunov function V')
ax1.set_title('Lyapunov Function Decay')
ax1.legend()

line2_theta, = ax2.plot(sol.t, sol.y[0], label=r'$\theta$ (rad)')
line2_theta_dot, = ax2.plot(sol.t, sol.y[1], label=r'$\dot{\theta}$ (rad/s)')
theta_marker = ax2.scatter([], [], color='red', marker='o', label='Theta at stabilization')
theta_dot_marker = ax2.scatter([], [], color='green', marker='o', label='Theta dot at stabilization')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('State')
ax2.set_title('State Variables Over Time')
ax2.legend()

# Markers for stabilization point in Lyapunov plot
stab_marker = ax1.scatter([], [], color='red', label='Stabilization point')

# Slider setup
axcolor = 'lightgoldenrodyellow'
ax_v0 = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)
slider_v0 = Slider(ax_v0, 'Initial Kick', 0.0, 15.0, valinit=init_v0, valstep=0.1)

def update(val):
    v0 = slider_v0.val
    t_stab, theta_stab, theta_dot_stab, sol, V = time_to_stability(theta0, v0, V_thresh=V_thresh, T=T)
    line1.set_xdata(sol.t)
    line1.set_ydata(V)
    line2_theta.set_xdata(sol.t)
    line2_theta.set_ydata(sol.y[0])
    line2_theta_dot.set_xdata(sol.t)
    line2_theta_dot.set_ydata(sol.y[1])
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    # Clear previous markers
    for coll in ax1.collections:
        coll.remove()
    for coll in ax2.collections:
        coll.remove()
    # Add new markers if stabilized
    if t_stab is not None:
        print(f"Stabilized at time: {t_stab:.4f} s")
        print(f"Theta at stabilization: {theta_stab:.4f} rad")
        print(f"Theta dot at stabilization: {theta_dot_stab:.4f} rad/s")
        ax1.scatter(t_stab, lyapunov(theta_stab, theta_dot_stab), color='red', label='Stabilization point')
        ax2.scatter(t_stab, theta_stab, color='red', marker='o', label='Theta at stabilization')
        ax2.scatter(t_stab, theta_dot_stab, color='green', marker='o', label='Theta dot at stabilization')
        line1.set_label(f'Initial velocity = {v0:.2f}\nTime to stability: {t_stab:.2f}s')
    else:
        print("The system did not stabilize within the simulation time.")
        line1.set_label('Did not stabilize')
    ax1.legend()
    ax2.legend()
    fig.canvas.draw_idle()

slider_v0.on_changed(update)

plt.show()
