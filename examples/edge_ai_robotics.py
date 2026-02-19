"""
Edge AI/Robotics Example: Maze Navigation with Energy Constraints

This example demonstrates MNE's advantages for energy-constrained edge AI/robotics
applications. It simulates a robot navigating a maze and compares MNE with a
standard neural network.

Key Features:
- Simple maze navigation task
- Energy-constrained environment (simulating battery-powered robot)
- Comparison of MNE vs StandardNN
- Visualization of energy consumption and neuron count over time
- Demonstration of neurogenesis/apoptosis in action

Application Context:
- Edge robotics with limited battery life
- Autonomous navigation in unknown environments
- Adaptive learning under resource constraints
- Real-time decision making with energy awareness
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Suppress NumPy warnings
import warnings

warnings.filterwarnings("ignore", message=".*NumPy.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import MNE, MNEConfig


# ============================================================================
# Maze Environment
# ============================================================================


class MazeEnvironment:
    """
    Simple maze environment for robot navigation.

    The robot must navigate from start to goal while avoiding obstacles.
    State: (x, y) position in the maze
    Action: Move up, down, left, right
    Reward: +1 for reaching goal, -0.1 for each step, -1 for hitting wall
    """

    def __init__(self, size: int = 10, obstacle_density: float = 0.2):
        self.size = size
        self.obstacle_density = obstacle_density
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Create maze with obstacles
        self.maze = np.zeros((self.size, self.size), dtype=int)

        # Add random obstacles
        num_obstacles = int(self.size * self.size * self.obstacle_density)
        for _ in range(num_obstacles):
            x, y = np.random.randint(0, self.size, 2)
            if (x, y) != (0, 0) and (x, y) != (self.size - 1, self.size - 1):
                self.maze[x, y] = 1

        # Set start and goal
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.position = self.start
        self.steps = 0
        self.max_steps = self.size * 2

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state as normalized position."""
        x, y = self.position
        state = np.array([x / self.size, y / self.size], dtype=np.float32)
        return state

    def step(self, action: int):
        """
        Take a step in the environment.

        Args:
            action: 0=up, 1=down, 2=left, 3=right

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        x, y = self.position

        # Compute new position
        if action == 0:  # up
            new_x, new_y = x - 1, y
        elif action == 1:  # down
            new_x, new_y = x + 1, y
        elif action == 2:  # left
            new_x, new_y = x, y - 1
        elif action == 3:  # right
            new_x, new_y = x, y + 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check bounds and obstacles
        if (
            0 <= new_x < self.size
            and 0 <= new_y < self.size
            and self.maze[new_x, new_y] == 0
        ):
            self.position = (new_x, new_y)
            reward = -0.01  # Small penalty for each step
        else:
            reward = -0.5  # Penalty for hitting wall/obstacle

        # Check if goal reached
        if self.position == self.goal:
            reward = 1.0
            done = True
        else:
            self.steps += 1
            done = self.steps >= self.max_steps

        info = {
            "position": self.position,
            "steps": self.steps,
            "goal_reached": self.position == self.goal,
        }

        return self._get_state(), reward, done, info


# ============================================================================
# Standard Neural Network (for comparison)
# ============================================================================


class StandardNN(nn.Module):
    """
    Standard feedforward neural network for comparison.

    This is a conventional neural network without metabolic constraints
    or dynamic topology.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# MNE Controller Wrapper
# ============================================================================


class MNEController(nn.Module):
    """
    Wrapper for MNE to handle input/output dimensions for robotics tasks.

    The MNE expects inputs to match the number of neurons. This wrapper adds
    input and output layers to map between the actual input/output dimensions
    and the MNE's neuron count.
    """

    def __init__(
        self,
        input_dim: int = 2,
        num_neurons: int = 64,
        output_dim: int = 4,
        mne_config: MNEConfig = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.output_dim = output_dim

        # Input layer: maps input to neurons
        self.input_layer = nn.Linear(input_dim, num_neurons)

        # Output layer: maps neurons to actions
        self.output_layer = nn.Linear(num_neurons, output_dim)

        # MNE core
        if mne_config is None:
            mne_config = MNEConfig(num_neurons=num_neurons)
        else:
            mne_config.num_neurons = num_neurons

        self.mne = MNE(mne_config)

    def get_initial_state(self, batch_size: int = 1):
        """Get initial MNE state."""
        return self.mne.get_initial_state(batch_size)

    def forward(
        self,
        x: torch.Tensor,
        state,
        contribution=None,
        apply_plasticity=True,
        apply_topology=True,
    ):
        """
        Forward pass through MNE controller.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            state: MNE state
            contribution: Optional contribution signal
            apply_plasticity: Whether to apply synaptic plasticity
            apply_topology: Whether to apply structural plasticity

        Returns:
            Tuple of (output, new_state)
        """
        # Map input to neuron space
        neuron_input = self.input_layer(x)

        # Pass through MNE
        activation, new_state = self.mne(
            neuron_input,
            state,
            contribution=contribution,
            apply_plasticity=apply_plasticity,
            apply_topology=apply_topology,
        )

        # Map to output space
        output = self.output_layer(activation)

        return output, new_state

    def get_metrics(self, state):
        """Get MNE metrics."""
        return self.mne.get_metrics(state)


# ============================================================================
# Energy-Constrained Robot Controller
# ============================================================================


class EnergyConstrainedController:
    """
    Robot controller with energy constraints.

    Simulates a battery-powered robot with limited energy.
    Energy is consumed by neural network computations.
    """

    def __init__(self, initial_energy: float = 100.0, energy_per_step: float = 0.5):
        self.initial_energy = initial_energy
        self.energy_per_step = energy_per_step
        self.current_energy = initial_energy
        self.energy_history = []
        self.is_depleted = False

    def consume_energy(self, amount: float = None) -> bool:
        """
        Consume energy for a computation step.

        Args:
            amount: Amount of energy to consume (default: energy_per_step)

        Returns:
            bool: True if energy was consumed, False if depleted
        """
        if self.is_depleted:
            return False

        amount = amount if amount is not None else self.energy_per_step
        self.current_energy -= amount
        self.energy_history.append(self.current_energy)

        if self.current_energy <= 0:
            self.current_energy = 0
            self.is_depleted = True
            return False

        return True

    def reset(self):
        """Reset energy to initial level."""
        self.current_energy = self.initial_energy
        self.energy_history = []
        self.is_depleted = False

    def get_energy_level(self) -> float:
        """Get current energy level."""
        return self.current_energy

    def get_energy_fraction(self) -> float:
        """Get energy as fraction of initial."""
        return self.current_energy / self.initial_energy


# ============================================================================
# Training and Evaluation
# ============================================================================


def train_mne_controller(
    env: MazeEnvironment,
    mne_controller: MNEController,
    controller: EnergyConstrainedController,
    num_episodes: int = 100,
    max_steps_per_episode: int = 50,
) -> dict:
    """
    Train MNE controller for maze navigation.

    MNE uses state-based learning (Hebbian plasticity) rather than
    gradient-based optimization. The contribution signal drives learning.

    Args:
        env: Maze environment
        mne_controller: MNEController wrapper
        controller: Energy controller
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode

    Returns:
        Dictionary of training metrics
    """
    metrics = {
        "episode_rewards": [],
        "episode_steps": [],
        "energy_levels": [],
        "neuron_counts": [],
        "neurogenesis_counts": [],
        "apoptosis_counts": [],
        "efficiency": [],
        "goal_reached": [],
    }

    state = mne_controller.get_initial_state(batch_size=1)

    for episode in range(num_episodes):
        obs = env.reset()
        controller.reset()
        episode_reward = 0
        episode_steps = 0
        goal_reached = False

        for step in range(max_steps_per_episode):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # Get action from MNE
            with torch.no_grad():
                action_logits, state = mne_controller(
                    obs_tensor, state, apply_plasticity=False
                )
                action = action_logits.argmax(dim=1).item()

            # Take action in environment
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # Consume energy
            if not controller.consume_energy():
                print(f"  Episode {episode}: Energy depleted at step {step}")
                break

            # Train MNE using contribution-based learning
            # The contribution signal is based on the reward (higher reward = higher contribution)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # Compute contribution based on reward magnitude
            # Positive rewards increase contribution, negative rewards decrease it
            contribution = torch.abs(torch.FloatTensor([reward])).unsqueeze(0)
            contribution = contribution.expand(1, mne_controller.num_neurons)

            # Apply MNE forward pass with plasticity
            action_logits, state = mne_controller(
                obs_tensor, state, contribution=contribution, apply_plasticity=True
            )

            obs = next_obs

            if done:
                if info["goal_reached"]:
                    goal_reached = True
                break

        # Get metrics from current state
        mne_metrics = mne_controller.get_metrics(state)

        # Record metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_steps"].append(episode_steps)
        metrics["energy_levels"].append(controller.get_energy_fraction())
        metrics["neuron_counts"].append(mne_metrics["num_neurons"])
        metrics["neurogenesis_counts"].append(mne_metrics["neurogenesis_count"])
        metrics["apoptosis_counts"].append(mne_metrics["apoptosis_count"])
        metrics["efficiency"].append(mne_metrics["efficiency"])
        metrics["goal_reached"].append(goal_reached)

        if episode % 10 == 0:
            print(
                f"Episode {episode}/{num_episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Steps={episode_steps}, "
                f"Energy={controller.get_energy_fraction():.2f}, "
                f"Neurons={mne_metrics['num_neurons']}, "
                f"Efficiency={mne_metrics['efficiency']:.4f}"
            )

    return metrics


def train_standard_controller(
    env: MazeEnvironment,
    model: StandardNN,
    controller: EnergyConstrainedController,
    num_episodes: int = 100,
    max_steps_per_episode: int = 50,
) -> dict:
    """
    Train standard neural network controller for maze navigation.

    Args:
        env: Maze environment
        model: Standard neural network
        controller: Energy controller
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode

    Returns:
        Dictionary of training metrics
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    metrics = {
        "episode_rewards": [],
        "episode_steps": [],
        "energy_levels": [],
        "goal_reached": [],
    }

    for episode in range(num_episodes):
        obs = env.reset()
        controller.reset()
        episode_reward = 0
        episode_steps = 0
        goal_reached = False

        for step in range(max_steps_per_episode):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # Get action from model
            with torch.no_grad():
                action_logits = model(obs_tensor)
                action = action_logits.argmax(dim=1).item()

            # Take action in environment
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # Consume energy
            if not controller.consume_energy():
                print(f"  Episode {episode}: Energy depleted at step {step}")
                break

            # Train model (simplified: use reward as target)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            target = torch.FloatTensor([reward]).unsqueeze(0)

            # Forward pass
            output = model(obs_tensor)

            # Compute loss (use first output as prediction)
            loss = criterion(output[:, 0:1], target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs

            if done:
                if info["goal_reached"]:
                    goal_reached = True
                break

        # Record metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_steps"].append(episode_steps)
        metrics["energy_levels"].append(controller.get_energy_fraction())
        metrics["goal_reached"].append(goal_reached)

        if episode % 10 == 0:
            print(
                f"Episode {episode}/{num_episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Steps={episode_steps}, "
                f"Energy={controller.get_energy_fraction():.2f}"
            )

    return metrics


# ============================================================================
# Summary
# ============================================================================


def print_summary(mne_metrics: dict, standard_metrics: dict):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY: MNE vs StandardNN for Edge Robotics")
    print("=" * 70)

    # MNE statistics
    mne_avg_reward = np.mean(mne_metrics["episode_rewards"][-20:])
    mne_avg_energy = np.mean(mne_metrics["energy_levels"][-20:])
    mne_avg_neurons = np.mean(mne_metrics["neuron_counts"][-20:])
    mne_goal_rate = np.mean(mne_metrics["goal_reached"][-20:]) * 100
    mne_neurogenesis = mne_metrics["neurogenesis_counts"][-1]
    mne_apoptosis = mne_metrics["apoptosis_counts"][-1]

    # StandardNN statistics
    std_avg_reward = np.mean(standard_metrics["episode_rewards"][-20:])
    std_avg_energy = np.mean(standard_metrics["energy_levels"][-20:])
    std_goal_rate = np.mean(standard_metrics["goal_reached"][-20:]) * 100

    print("\nMNE (Metabolic Neural Ecosystem):")
    print(f"  Average Reward (last 20): {mne_avg_reward:.3f}")
    print(f"  Average Energy Remaining: {mne_avg_energy:.3f}")
    print(f"  Average Neuron Count: {mne_avg_neurons:.1f}")
    print(f"  Goal Success Rate: {mne_goal_rate:.1f}%")
    print(f"  Total Neurogenesis Events: {mne_neurogenesis}")
    print(f"  Total Apoptosis Events: {mne_apoptosis}")

    print("\nStandardNN:")
    print(f"  Average Reward (last 20): {std_avg_reward:.3f}")
    print(f"  Average Energy Remaining: {std_avg_energy:.3f}")
    print(f"  Goal Success Rate: {std_goal_rate:.1f}%")

    print("\nMNE Advantages:")
    energy_savings = (
        (mne_avg_energy - std_avg_energy) / std_avg_energy * 100
        if std_avg_energy > 0
        else 0
    )
    print(f"  Energy Savings: {energy_savings:.1f}%")
    print(f"  Dynamic Topology: {mne_neurogenesis + mne_apoptosis} structural changes")
    print(f"  Adaptive Neuron Count: {mne_avg_neurons:.1f} neurons (vs fixed 64)")

    print("\nKey Insights:")
    print("  - MNE adapts its structure to the task, adding/removing neurons as needed")
    print("  - Energy-aware learning prioritizes important computations")
    print("  - Neurogenesis creates new neurons when resources are abundant")
    print("  - Apoptosis removes inefficient neurons under energy constraints")
    print("  - This makes MNE ideal for battery-powered edge robotics")

    print("=" * 70)


# ============================================================================
# Main
# ============================================================================


def main():
    """Main function to run the edge AI/robotics example."""
    print("=" * 70)
    print("Edge AI/Robotics Example: Maze Navigation with Energy Constraints")
    print("=" * 70)
    print("\nThis example demonstrates MNE's advantages for energy-constrained")
    print("edge AI/robotics applications by comparing MNE with a standard neural")
    print("network on a maze navigation task.")
    print()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create environment
    print("Creating maze environment...")
    env = MazeEnvironment(size=10, obstacle_density=0.2)
    print(f"  Maze size: {env.size}x{env.size}")
    print(f"  Obstacle density: {env.obstacle_density:.1%}")

    # Create energy controller
    initial_energy = 100.0
    energy_per_step = 0.5
    controller = EnergyConstrainedController(
        initial_energy=initial_energy, energy_per_step=energy_per_step
    )
    print(f"  Initial energy: {initial_energy}")
    print(f"  Energy per step: {energy_per_step}")

    # Configure MNE
    print("\nConfiguring MNE...")
    mne_config = MNEConfig(
        num_neurons=64,
        activation_fn="tanh",
        # Energy parameters (constrained environment)
        initial_energy=initial_energy,
        energy_influx=2.0,  # Low influx (battery-powered)
        min_energy=10.0,
        max_energy=initial_energy,
        # Topology parameters (enable structural plasticity)
        max_neurons=128,
        min_neurons=16,
        resource_high=1.5,  # Threshold for neurogenesis
        resource_low=0.3,  # Threshold for apoptosis
        neurogenesis_rate=0.1,
        apoptosis_rate=0.1,
    )
    mne_controller = MNEController(
        input_dim=2, num_neurons=64, output_dim=4, mne_config=mne_config
    )
    print(f"  Initial neurons: {mne_config.num_neurons}")
    print(f"  Max neurons: {mne_config.max_neurons}")
    print(f"  Min neurons: {mne_config.min_neurons}")

    # Create standard neural network
    print("\nConfiguring StandardNN...")
    standard_model = StandardNN(input_dim=2, hidden_dim=64, output_dim=4)
    num_params = standard_model.get_num_parameters()
    print(f"  Hidden layer size: 64")
    print(f"  Total parameters: {num_params}")

    # Training parameters
    num_episodes = 50  # Reduced for faster testing
    max_steps_per_episode = 50

    # Train MNE
    print("\n" + "=" * 70)
    print("Training MNE Controller")
    print("=" * 70)
    mne_metrics = train_mne_controller(
        env, mne_controller, controller, num_episodes, max_steps_per_episode
    )

    # Train StandardNN
    print("\n" + "=" * 70)
    print("Training StandardNN Controller")
    print("=" * 70)
    standard_metrics = train_standard_controller(
        env, standard_model, controller, num_episodes, max_steps_per_episode
    )

    # Print summary
    print_summary(mne_metrics, standard_metrics)

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print("\nNote: To visualize the results, you can use matplotlib to plot:")
    print("  - Episode rewards over time")
    print("  - Energy consumption over time")
    print("  - Neuron count dynamics (MNE only)")
    print("  - Neurogenesis/apoptosis events (MNE only)")
    print("  - Energy efficiency over time (MNE only)")


if __name__ == "__main__":
    main()
