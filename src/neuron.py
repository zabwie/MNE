"""
Neuron module for Metabolic Neural Ecosystem (MNE).

This module implements individual neurons with metabolic state tracking,
activation dynamics, and energy consumption calculations.

Mathematical Foundation:
-----------------------
The neuron implements the following dynamics:

1. Activation update:
   a_i(t+1) = f(∑_j w_ij(t) a_j(t) + I_i(t) - θ_i(t))

   where:
   - a_i(t): activation of neuron i at time t
   - w_ij(t): synaptic weight from neuron j to i
   - I_i(t): external input to neuron i
   - θ_i(t): homeostatic threshold
   - f(·): activation function (typically tanh or ReLU)

2. Contribution (gradient-based resource allocation):
   contrib_i(t) = |∂L/∂a_i|

   where L is the loss function. This measures how much the neuron
   contributes to the overall task performance.

3. Energy consumption:
   consume_i(t) = κ a_i(t)² + ∑_j γ |w_ij(t)| a_j(t)

   where:
   - κ: baseline metabolic cost coefficient
   - γ: synaptic transmission cost coefficient

4. Resource update:
   r_i(t+1) = r_i(t) + α·contrib_i(t) - β·consume_i(t) - δ r_i(t)

   where:
   - r_i(t): metabolic resource level
   - α: resource gain coefficient
   - β: resource consumption coefficient
   - δ: resource decay coefficient

5. Homeostatic threshold update:
   θ_i(t+1) = θ_i(t) + ρ(a_i(t) - a_target)

   where:
   - ρ: homeostatic learning rate
   - a_target: target activation level

References:
-----------
1. Buzsáki, G. (2019). The Brain from Inside Out. Oxford University Press.
2. Sterling, P., & Laughlin, S. (2015). Principles of Neural Design.
   MIT Press.
3. Levy, W. B., & Baxter, R. A. (1996). Energy efficient neural codes.
   Neural Computation, 8(3), 531-543.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class NeuronState:
    """
    State container for a single neuron.

    Attributes:
        activation: Current activation value a_i(t)
        resource: Current metabolic resource level r_i(t)
        threshold: Homeostatic threshold θ_i(t)
        contribution: Gradient-based contribution |∂L/∂a_i|
        consumption: Energy consumption consume_i(t)
        is_active: Whether neuron is currently active
        age: Number of time steps neuron has been alive
    """

    activation: torch.Tensor
    resource: torch.Tensor
    threshold: torch.Tensor
    contribution: torch.Tensor
    consumption: torch.Tensor
    is_active: torch.Tensor
    age: torch.Tensor


class MNENeuron(nn.Module):
    """
    Metabolic Neural Ecosystem Neuron.

    Implements a biologically inspired neuron with:
    - Activation dynamics with homeostatic threshold
    - Metabolic resource tracking
    - Energy consumption calculation
    - Gradient-based contribution measurement

    Args:
        num_neurons: Number of neurons in the population
        activation_fn: Activation function ('tanh', 'relu', 'sigmoid', or 'leaky_relu')
        kappa: Baseline metabolic cost coefficient (default: 0.1)
        gamma: Synaptic transmission cost coefficient (default: 0.05)
        alpha: Resource gain coefficient (default: 0.5)
        beta: Resource consumption coefficient (default: 1.0)
        delta: Resource decay coefficient (default: 0.01)
        rho: Homeostatic learning rate (default: 0.01)
        target_activation: Target activation level for homeostasis (default: 0.5)
        initial_resource: Initial metabolic resource level (default: 1.0)
        initial_threshold: Initial homeostatic threshold (default: 0.0)
        device: Device to place tensors on (default: 'cpu')

    Example:
        >>> neuron = MNENeuron(num_neurons=100, activation_fn='tanh')
        >>> inputs = torch.randn(32, 100)  # Batch of 32
        >>> weights = torch.randn(100, 100)
        >>> state = neuron.get_initial_state(batch_size=32)
        >>> output, new_state = neuron(inputs, weights, state)
    """

    def __init__(
        self,
        num_neurons: int,
        activation_fn: str = "tanh",
        kappa: float = 0.1,
        gamma: float = 0.05,
        alpha: float = 0.5,
        beta: float = 1.0,
        delta: float = 0.01,
        rho: float = 0.01,
        target_activation: float = 0.5,
        initial_resource: float = 1.0,
        initial_threshold: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.kappa = kappa
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.rho = rho
        self.target_activation = target_activation
        self.initial_resource = initial_resource
        self.initial_threshold = initial_threshold
        self.device = device

        # Set activation function
        if activation_fn == "tanh":
            self.activation_fn = torch.tanh
        elif activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "sigmoid":
            self.activation_fn = torch.sigmoid
        elif activation_fn == "leaky_relu":
            self.activation_fn = lambda x: F.leaky_relu(x, negative_slope=0.01)
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

        # Move to device
        self.to(device)

    def get_initial_state(self, batch_size: int) -> NeuronState:
        """
        Initialize neuron state for a batch.

        Args:
            batch_size: Batch size

        Returns:
            NeuronState: Initial state for all neurons
        """
        device = self.device
        shape = (batch_size, self.num_neurons)

        return NeuronState(
            activation=torch.zeros(shape, device=device),
            resource=torch.full(shape, self.initial_resource, device=device),
            threshold=torch.full(shape, self.initial_threshold, device=device),
            contribution=torch.zeros(shape, device=device),
            consumption=torch.zeros(shape, device=device),
            is_active=torch.ones(shape, dtype=torch.bool, device=device),
            age=torch.zeros(shape, device=device),
        )

    def compute_activation(
        self, inputs: torch.Tensor, weights: torch.Tensor, state: NeuronState
    ) -> torch.Tensor:
        """
        Compute new activation based on inputs and synaptic weights.

        Implements: a_i(t+1) = f(∑_j w_ij(t) a_j(t) + I_i(t) - θ_i(t))

        Args:
            inputs: External input I_i(t) of shape (batch_size, num_neurons)
            weights: Synaptic weights w_ij(t) of shape (num_neurons, num_neurons)
            state: Current neuron state

        Returns:
            torch.Tensor: New activation a_i(t+1) of shape (batch_size, num_neurons)
        """
        # Compute synaptic input: ∑_j w_ij(t) a_j(t)
        synaptic_input = torch.matmul(state.activation, weights.T)

        # Total input: synaptic + external - threshold
        total_input = synaptic_input + inputs - state.threshold

        # Apply activation function
        new_activation = self.activation_fn(total_input)

        # Zero out inactive neurons
        new_activation = new_activation * state.is_active.float()

        return new_activation

    def compute_consumption(
        self, activation: torch.Tensor, weights: torch.Tensor, state: NeuronState
    ) -> torch.Tensor:
        """
        Compute energy consumption for each neuron.

        Implements: consume_i(t) = κ a_i(t)² + ∑_j γ |w_ij(t)| a_j(t)

        Args:
            activation: Current activation a_i(t)
            weights: Synaptic weights w_ij(t)
            state: Current neuron state

        Returns:
            torch.Tensor: Energy consumption for each neuron
        """
        # Baseline metabolic cost: κ a_i(t)²
        baseline_cost = self.kappa * activation**2

        # Synaptic transmission cost: ∑_j γ |w_ij(t)| a_j(t)
        # For each neuron i, sum over incoming weights from j
        abs_weights = torch.abs(weights)
        synaptic_cost = self.gamma * torch.matmul(state.activation, abs_weights.T)

        total_consumption = baseline_cost + synaptic_cost

        # Zero out inactive neurons
        total_consumption = total_consumption * state.is_active.float()

        return total_consumption

    def update_resource(
        self, state: NeuronState, consumption: torch.Tensor
    ) -> torch.Tensor:
        """
        Update metabolic resource levels.

        Implements: r_i(t+1) = r_i(t) + α·contrib_i(t) - β·consume_i(t) - δ r_i(t)

        Args:
            state: Current neuron state
            consumption: Energy consumption

        Returns:
            torch.Tensor: Updated resource levels
        """
        # Resource gain from contribution
        resource_gain = self.alpha * state.contribution

        # Resource loss from consumption
        resource_loss = self.beta * consumption

        # Natural decay
        decay = self.delta * state.resource

        # Update resource
        new_resource = state.resource + resource_gain - resource_loss - decay

        # Ensure non-negative resources
        new_resource = torch.clamp(new_resource, min=0.0)

        # Zero out inactive neurons
        new_resource = new_resource * state.is_active.float()

        return new_resource

    def update_threshold(
        self, state: NeuronState, activation: torch.Tensor
    ) -> torch.Tensor:
        """
        Update homeostatic threshold.

        Implements: θ_i(t+1) = θ_i(t) + ρ(a_i(t) - a_target)

        Args:
            state: Current neuron state
            activation: Current activation

        Returns:
            torch.Tensor: Updated threshold
        """
        # Compute error from target
        error = activation - self.target_activation

        # Update threshold
        new_threshold = state.threshold + self.rho * error

        # Zero out inactive neurons
        new_threshold = new_threshold * state.is_active.float()

        return new_threshold

    def forward(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        state: NeuronState,
        contribution: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, NeuronState]:
        """
        Perform one time step of neuron dynamics.

        Args:
            inputs: External input I_i(t) of shape (batch_size, num_neurons)
            weights: Synaptic weights w_ij(t) of shape (num_neurons, num_neurons)
            state: Current neuron state
            contribution: Optional gradient-based contribution |∂L/∂a_i|
                        If None, defaults to zeros

        Returns:
            Tuple[torch.Tensor, NeuronState]:
                - New activation a_i(t+1)
                - Updated neuron state
        """
        if contribution is None:
            contribution = torch.zeros_like(state.activation)

        # Compute new activation
        new_activation = self.compute_activation(inputs, weights, state)

        # Compute energy consumption
        consumption = self.compute_consumption(new_activation, weights, state)

        # Update resource
        new_resource = self.update_resource(state, consumption)

        # Update threshold
        new_threshold = self.update_threshold(state, new_activation)

        # Update age
        new_age = state.age + state.is_active.float()

        # Create new state
        new_state = NeuronState(
            activation=new_activation,
            resource=new_resource,
            threshold=new_threshold,
            contribution=contribution,
            consumption=consumption,
            is_active=state.is_active,
            age=new_age,
        )

        return new_activation, new_state

    def set_contribution(
        self, state: NeuronState, contribution: torch.Tensor
    ) -> NeuronState:
        """
        Set the gradient-based contribution for the current state.

        Args:
            state: Current neuron state
            contribution: Gradient-based contribution |∂L/∂a_i|

        Returns:
            NeuronState: Updated state with new contribution
        """
        state.contribution = contribution
        return state

    def deactivate_neurons(self, state: NeuronState, mask: torch.Tensor) -> NeuronState:
        """
        Deactivate neurons based on a mask.

        Args:
            state: Current neuron state
            mask: Boolean mask of neurons to deactivate (True = deactivate)

        Returns:
            NeuronState: Updated state with deactivated neurons
        """
        state.is_active = state.is_active & ~mask
        state.activation = state.activation * state.is_active.float()
        state.resource = state.resource * state.is_active.float()
        return state

    def get_active_neurons(self, state: NeuronState) -> torch.Tensor:
        """
        Get indices of active neurons.

        Args:
            state: Current neuron state

        Returns:
            torch.Tensor: Indices of active neurons
        """
        return torch.where(state.is_active[0])[0]  # Use first batch element

    def get_resource_levels(self, state: NeuronState) -> torch.Tensor:
        """
        Get resource levels for all neurons.

        Args:
            state: Current neuron state

        Returns:
            torch.Tensor: Resource levels
        """
        return state.resource

    def get_consumption_levels(self, state: NeuronState) -> torch.Tensor:
        """
        Get consumption levels for all neurons.

        Args:
            state: Current neuron state

        Returns:
            torch.Tensor: Consumption levels
        """
        return state.consumption
