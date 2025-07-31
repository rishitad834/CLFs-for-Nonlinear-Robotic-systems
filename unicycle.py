import numpy as np
import matplotlib.pyplot as plt

# System and Simulation Parameters 
v_max = 2.0  # Maximum linear velocity (m/s)
w_max = 2.0  # Maximum angular velocity (rad/s)
time_step = 0.1
max_sim_time = 20.0

class UnicycleRobot:
    
    def __init__(self, initial_state, goal_state, gains, stop_dist):
        self.state = np.array(initial_state, dtype=float)  # [x, y, theta]
        self.goal_state = np.array(goal_state, dtype=float)
        
        # Controller gains and parameters
        self.k_v = gains['v']  # Proportional gain for linear velocity
        self.k_w = gains['w']  # Proportional gain for angular velocity
        self.stop_dist = stop_dist # Distance threshold to switch to rotation

        # History lists for plotting
        self.history = [self.state.copy()]
        self.lyapunov_history = []

    def calculate_lyapunov_value(self, current_state):
        """
        Calculates a simple Lyapunov-like function V = error_dist^2.
        This function should decay to zero as the robot reaches the goal.
        """
        ex = self.goal_state[0] - current_state[0]
        ey = self.goal_state[1] - current_state[1]
        return ex**2 + ey**2 # V = rho^2

    def analytical_controller(self, current_state):
        """
        Calculates control input using a direct, two-stage formula.
        """
        #  Calculate Errors 
        ex = self.goal_state[0] - current_state[0]
        ey = self.goal_state[1] - current_state[1]
        dist_error = np.sqrt(ex**2 + ey**2)
        angle_to_goal = np.arctan2(ey, ex)
        angle_error = angle_to_goal - current_state[2]
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        #  Implement Two-Stage Control Logic 
        if dist_error > self.stop_dist:
            v = self.k_v * dist_error
            w = self.k_w * angle_error
        else:
            final_angle_error = self.goal_state[2] - current_state[2]
            final_angle_error = (final_angle_error + np.pi) % (2 * np.pi) - np.pi
            v = 0
            w = self.k_w * final_angle_error

        #  Saturate Controls 
        v = np.clip(v, -v_max, v_max)
        w = np.clip(w, -w_max, w_max)
        
        return np.array([v, w])

    def update_state(self, control_input):
        """Updates the robot's state using Euler integration."""
        v, w = control_input
        _, _, theta = self.state
        
        derivatives = np.array([v * np.cos(theta), v * np.sin(theta), w])
        self.state += derivatives * time_step
        
        # Record history for plotting
        self.history.append(self.state.copy())
        self.lyapunov_history.append(self.calculate_lyapunov_value(self.state))

def plot_trajectory(robot):
    """Plots the robot's path, start, and goal positions."""
    trajectory = np.array(robot.history)
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label="Robot's Path")
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    plt.plot(robot.goal_state[0], robot.goal_state[1], 'r*', markersize=12, label='Goal')
    final_state = trajectory[-1]
    plt.arrow(final_state[0], final_state[1], 0.5 * np.cos(final_state[2]), 0.5 * np.sin(final_state[2]),
              head_width=0.15, head_length=0.2, fc='c', ec='k', label='Final Orientation')
    plt.xlabel('X-position (m)')
    plt.ylabel('Y-position (m)')
    plt.title('Unicycle Robot Navigation Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

def plot_state_variables(robot, time_array):
    """Plots the state variables x, y, and theta over time."""
    state_history = np.array(robot.history)
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, state_history[:, 0], 'r-', label='x (m)')
    plt.plot(time_array, state_history[:, 1], 'g-', label='y (m)')
    plt.plot(time_array, state_history[:, 2], 'b-', label='θ (rad)')
    plt.xlabel('Time (s)')
    plt.ylabel('State Value')
    plt.title('State Variables Evolution')
    plt.grid(True)
    plt.legend()

def plot_lyapunov_function(robot, time_array):
    """Plots the Lyapunov function value over time."""
    lyapunov_history = np.array(robot.lyapunov_history)
    plt.figure(figsize=(10, 6))
    plt.plot(time_array[1:], lyapunov_history, 'm-', label='V(t) = distance_error²')
    plt.xlabel('Time (s)')
    plt.ylabel('Lyapunov Function Value')
    plt.title('Lyapunov Function Decay')
    plt.grid(True)
    plt.legend()

if __name__ == '__main__':
    # Configuration
    initial_position = [0.0, 0.0, np.pi / 2]
    goal_position = [4.0, 5.0, 0.0]
    controller_gains = {'v': 0.5, 'w': 1.5}
    stop_distance = 0.1

    # Simulation 
    robot = UnicycleRobot(initial_position, goal_position, controller_gains, stop_distance)
    time_data = [0]
    
    for t in np.arange(0, max_sim_time, time_step):
        final_dist_err = np.linalg.norm(robot.state[:2] - robot.goal_state[:2])
        final_angle_err = np.abs(robot.state[2] - robot.goal_state[2])
        if final_dist_err < stop_distance and final_angle_err < 0.05:
            print(f"✅ Goal successfully reached in {t:.2f} seconds.")
            break
        
        control = robot.analytical_controller(robot.state)
        robot.update_state(control)
        time_data.append(t + time_step)
    else:
        print("Robot did not reach the goal within the maximum simulation time.")

    # Plotting 
    time_array = np.array(time_data)
    plot_trajectory(robot)
    plot_state_variables(robot, time_array)
    plot_lyapunov_function(robot, time_array)
    
    # Show all plots
    plt.show()