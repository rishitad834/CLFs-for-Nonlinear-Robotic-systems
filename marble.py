import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class LyapunovNet(nn.Module):
    """Neural network to approximate Lyapunov function"""
    def __init__(self, input_dim=2, hidden_dim=64):
        super(LyapunovNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures positive output
        )
    
    def forward(self, x):
        return self.network(x)
class SelfSupervisedSampler:
    """Intelligent sampling based on Lyapunov violation regions"""
    def __init__(self, system, lyapunov_net):
        self.system = system
        self.lyapunov_net = lyapunov_net
        self.violation_memory = []
        
    def adaptive_sampling(self, n_samples=1000, violation_ratio=0.3):
        """Sample more points where Lyapunov conditions are violated"""
        # Regular uniform sampling
        uniform_samples = int(n_samples * (1 - violation_ratio))
        x_uniform = self.generate_uniform_samples(uniform_samples)
        
        # Violation-based sampling
        violation_samples = n_samples - uniform_samples
        x_violation = self.sample_violation_regions(violation_samples)
        
        return torch.cat([x_uniform, x_violation], dim=0)
    
    def sample_violation_regions(self, n_samples):
        """Sample around regions with high Lyapunov violations"""
        if len(self.violation_memory) < 10:
            return self.generate_uniform_samples(n_samples)
        
        # Use Gaussian mixture around violation points
        violation_centers = torch.stack(self.violation_memory[-50:])  # Last 50 violations
        samples = []
        
        for _ in range(n_samples):
            center_idx = torch.randint(0, len(violation_centers), (1,))
            center = violation_centers[center_idx]
            noise = torch.randn_like(center) * 0.5
            samples.append(center + noise)
        
        return torch.stack(samples).squeeze(1)


class NonlinearSystem:
    """Define the 2D nonlinear dynamical system"""
    def __init__(self, system_type="van_der_pol"):
        self.system_type = system_type
    
    def dynamics(self, x):
        """Compute dx/dt = f(x) for different system types"""
        if self.system_type == "van_der_pol":
            # Van der Pol oscillator: dx1/dt = x2, dx2/dt = mu*(1-x1^2)*x2 - x1
            mu = 0.5
            x1, x2 = x[:, 0:1], x[:, 1:2]
            dx1_dt = x2
            dx2_dt = mu * (1 - x1**2) * x2 - x1
            return torch.cat([dx1_dt, dx2_dt], dim=1)
        
        elif self.system_type == "simple_nonlinear":
            # Simple nonlinear system: dx1/dt = -x1 + x1*x2, dx2/dt = -x2 - x1^2
            x1, x2 = x[:, 0:1], x[:, 1:2]
            dx1_dt = -x1 + x1 * x2
            dx2_dt = -x2 - x1**2
            return torch.cat([dx1_dt, dx2_dt], dim=1)
        
        elif self.system_type == "pendulum":
            # Damped pendulum: dx1/dt = x2, dx2/dt = -sin(x1) - 0.5*x2
            x1, x2 = x[:, 0:1], x[:, 1:2]
            dx1_dt = x2
            dx2_dt = -torch.sin(x1) - 0.5 * x2
            return torch.cat([dx1_dt, dx2_dt], dim=1)

class LyapunovTrainer:
    """Training class for the Lyapunov neural network"""
    def __init__(self, system, lr=1e-3, device='cpu'):
        self.system = system
        self.device = device
        self.lyapunov_net = LyapunovNet().to(device)
        self.optimizer = optim.Adam(self.lyapunov_net.parameters(), lr=lr)
        self.losses = []
        
    def compute_gradient(self, x, V):
        """Compute gradient of V with respect to x"""
        grad_outputs = torch.ones_like(V)
        gradients = torch.autograd.grad(
            outputs=V,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        return gradients
    
    def lyapunov_conditions_loss(self, x):
        """Compute loss based on Lyapunov conditions"""
        # Compute V(x)
        V = self.lyapunov_net(x)
        
        # Compute gradient of V
        dV_dx = self.compute_gradient(x, V)
        
        # Compute system dynamics f(x)
        f_x = self.system.dynamics(x)
        
        # Compute Lie derivative: dV/dt = (dV/dx) * f(x)
        lie_derivative = torch.sum(dV_dx * f_x, dim=1, keepdim=True)
        
        # Distance from origin
        distance_from_origin = torch.norm(x, dim=1, keepdim=True)
        
        # Condition 1: V should be positive away from origin
        positive_condition = torch.relu(-V + 1e-6)  # Penalty if V <= 0
        
        # Condition 2: Lie derivative should be negative away from origin
        stability_condition = torch.relu(lie_derivative + 1e-6)  # Penalty if dV/dt >= 0
        
        # Additional condition: V should be small at origin
        origin_condition = V * torch.exp(-distance_from_origin * 10)
        
        # Combine losses
        loss = (positive_condition.mean() + 
                stability_condition.mean() + 
                origin_condition.mean())
        
        return loss, V, lie_derivative
    
    def generate_training_data(self, n_samples=1000, radius=3.0):
        """Generate random training points in a circle around origin"""
        # Generate points in polar coordinates
        r = torch.sqrt(torch.rand(n_samples)) * radius
        theta = torch.rand(n_samples) * 2 * np.pi
        
        # Convert to Cartesian coordinates
        x1 = r * torch.cos(theta)
        x2 = r * torch.sin(theta)
        
        x = torch.stack([x1, x2], dim=1).to(self.device)
        # Enable gradient tracking
        x.requires_grad_(True)
        return x
    
    def train(self, epochs=5000, n_samples=1000):
        """Train the Lyapunov neural network"""
        print("Starting Lyapunov function training...")
        
        for epoch in range(epochs):
            # Generate training data with gradient tracking enabled
            x = self.generate_training_data(n_samples)
            
            # Compute loss
            self.optimizer.zero_grad()
            loss, V, lie_derivative = self.lyapunov_conditions_loss(x)
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(loss.item())
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
                
                # CRITICAL FIX: Proper gradient handling for testing
                # Generate separate test data for validation
                test_r = torch.sqrt(torch.rand(100)) * 2.5
                test_theta = torch.rand(100) * 2 * np.pi
                test_x1 = test_r * torch.cos(test_theta)
                test_x2 = test_r * torch.sin(test_theta)
                test_x = torch.stack([test_x1, test_x2], dim=1).to(self.device)
                test_x.requires_grad_(True)
                
                # Compute test values WITHOUT torch.no_grad()
                test_V = self.lyapunov_net(test_x)
                test_dV_dx = self.compute_gradient(test_x, test_V)
                test_f_x = self.system.dynamics(test_x)
                test_lie_derivative = torch.sum(test_dV_dx * test_f_x, dim=1)
                
                # Convert to numpy for counting violations
                with torch.no_grad():
                    positive_violations = (test_V <= 0).sum().item()
                    stability_violations = (test_lie_derivative >= 0).sum().item()
                
                print(f"  Positive violations: {positive_violations}/100")
                print(f"  Stability violations: {stability_violations}/100")
    
    def visualize_results(self):
        """Visualize the learned Lyapunov function and system behavior"""
        # Create a grid for visualization
        x_range = np.linspace(-3, 3, 50)
        y_range = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Flatten for neural network input
        grid_points = torch.tensor(
            np.column_stack([X.flatten(), Y.flatten()]), 
            dtype=torch.float32
        ).to(self.device)
        
        # Enable gradient tracking for grid points
        grid_points.requires_grad_(True)
        
        # Compute Lyapunov function values and gradients
        V_grid_tensor = self.lyapunov_net(grid_points)
        dV_dx = self.compute_gradient(grid_points, V_grid_tensor)
        f_grid = self.system.dynamics(grid_points)
        lie_derivative = torch.sum(dV_dx * f_grid, dim=1)
        
        # Convert to numpy for plotting
        with torch.no_grad():
            V_values = V_grid_tensor.cpu().numpy()
            f_values = f_grid.cpu().numpy()
            lie_values = lie_derivative.cpu().numpy()
        
        # Reshape for plotting
        V_grid = V_values.reshape(X.shape)
        U = f_values[:, 0].reshape(X.shape)
        V_flow = f_values[:, 1].reshape(X.shape)
        lie_grid = lie_values.reshape(X.shape)
        
        # Create subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Lyapunov function contours with vector field
        ax1 = plt.subplot(2, 3, 1)
        contour = plt.contour(X, Y, V_grid, levels=20, colors='blue', alpha=0.6)
        plt.contourf(X, Y, V_grid, levels=20, alpha=0.3, cmap='viridis')
        plt.colorbar(label='V(x)')
        
        # Add vector field
        skip = 3
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  U[::skip, ::skip], V_flow[::skip, ::skip], 
                  alpha=0.7, color='red', scale=20)
        
        plt.title('Lyapunov Function Contours with Vector Field')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: 3D surface of Lyapunov function
        ax2 = plt.subplot(2, 3, 2, projection='3d')
        surf = ax2.plot_surface(X, Y, V_grid, cmap='viridis', alpha=0.8)
        ax2.set_title('3D Lyapunov Function')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('V(x)')
        
        # Plot 3: Lie derivative
        ax3 = plt.subplot(2, 3, 3)
        contour_lie = plt.contourf(X, Y, lie_grid, levels=20, cmap='RdBu_r')
        plt.colorbar(contour_lie, label='dV/dt')
        plt.title('Lie Derivative (dV/dt)')
        plt.xlabel('x1')
        plt.ylabel('x2')
        
        # Plot 4: Training loss
        ax4 = plt.subplot(2, 3, 4)
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Phase portrait with trajectories
        ax5 = plt.subplot(2, 3, 5)
        plt.contour(X, Y, V_grid, levels=10, colors='blue', alpha=0.4)
        
        # Simulate some trajectories
        initial_conditions = [
            [2.0, 1.0], [-2.0, -1.0], [1.5, -1.5], [-1.5, 1.5],
            [0.5, 2.0], [-0.5, -2.0]
        ]
        
        for ic in initial_conditions:
            trajectory = self.simulate_trajectory(ic, dt=0.01, steps=500)
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.7, linewidth=2)
            plt.plot(ic[0], ic[1], 'go', markersize=8)
        
        plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  U[::skip, ::skip], V_flow[::skip, ::skip], 
                  alpha=0.5, scale=20)
        
        plt.title('Phase Portrait with Trajectories')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Validation of Lyapunov conditions
        ax6 = plt.subplot(2, 3, 6)
        
        # Generate test points for validation
        test_r = torch.sqrt(torch.rand(1000)) * 2.5
        test_theta = torch.rand(1000) * 2 * np.pi
        test_x1 = test_r * torch.cos(test_theta)
        test_x2 = test_r * torch.sin(test_theta)
        test_points = torch.stack([test_x1, test_x2], dim=1).to(self.device)
        test_points.requires_grad_(True)
        
        test_V = self.lyapunov_net(test_points)
        test_dV_dx = self.compute_gradient(test_points, test_V)
        test_f_x = self.system.dynamics(test_points)
        test_lie = torch.sum(test_dV_dx * test_f_x, dim=1)
        
        with torch.no_grad():
            test_points_np = test_points.cpu().numpy()
            test_V_np = test_V.cpu().numpy()
            test_lie_np = test_lie.cpu().numpy()
        
        # Scatter plot: V values vs distance from origin
        distances = np.linalg.norm(test_points_np, axis=1)
        scatter = plt.scatter(distances, test_V_np.flatten(), 
                            c=test_lie_np, cmap='RdBu_r', alpha=0.6)
        plt.colorbar(scatter, label='dV/dt')
        plt.xlabel('Distance from Origin')
        plt.ylabel('V(x)')
        plt.title('Lyapunov Conditions Validation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print validation statistics
        positive_violations = (test_V_np <= 0).sum()
        stability_violations = (test_lie_np >= 0).sum()
        
        print(f"\nValidation Results:")
        print(f"Positive definiteness violations: {positive_violations}/1000 ({positive_violations/10:.1f}%)")
        print(f"Stability violations: {stability_violations}/1000 ({stability_violations/10:.1f}%)")
        print(f"Mean V value: {test_V_np.mean():.4f}")
        print(f"Mean Lie derivative: {test_lie_np.mean():.4f}")
    
    def simulate_trajectory(self, initial_condition, dt=0.01, steps=1000):
        """Simulate a trajectory of the dynamical system"""
        trajectory = np.zeros((steps, 2))
        x = np.array(initial_condition)
        
        for i in range(steps):
            trajectory[i] = x
            
            # Convert to torch tensor for dynamics computation
            x_tensor = torch.tensor(x.reshape(1, -1), dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                dx_dt = self.system.dynamics(x_tensor).cpu().numpy().flatten()
            
            # Simple Euler integration
            x = x + dt * dx_dt
            
            # Stop if trajectory goes too far
            if np.linalg.norm(x) > 10:
                trajectory = trajectory[:i+1]
                break
        
        return trajectory

def main():
    """Main function to run the Lyapunov learning example"""
    print("Neural Network Lyapunov Function Learning")
    print("=" * 50)
    
    # Choose system type
    system_types = ["van_der_pol", "simple_nonlinear", "pendulum"]
    system_type = "simple_nonlinear"  # Change this to try different systems
    
    print(f"Selected system: {system_type}")
    
    # Create system and trainer
    system = NonlinearSystem(system_type)
    trainer = LyapunovTrainer(system, lr=1e-3)
    
    # Train the network
    trainer.train(epochs=3000, n_samples=1000)
    
    # Visualize results
    trainer.visualize_results()
    
    print("\nTraining completed! Check the plots for results.")

if __name__ == "__main__":
    main()
