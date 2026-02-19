"""
Core module for Metabolic Neural Ecosystem (MNE).

This module implements the main MNE orchestrator that integrates all components:
neurons, synapses, energy management, and topology.

Mathematical Foundation:
-----------------------
The MNE implements the complete dynamics:

1. Activation update:
   a_i(t+1) = f(∑_j w_ij(t) a_j(t) + I_i(t) - θ_i(t))

2. Contribution (gradient-based):
   contrib_i(t) = |∂L/∂a_i|

3. Consumption:
   consume_i(t) = κ a_i(t)² + ∑_j γ |w_ij(t)| a_j(t)

4. Resource update:
   r_i(t+1) = r_i(t) + α·contrib_i(t) - β·consume_i(t) - δ r_i(t)

5. Weight update (energy-aware Hebbian):
   w_ij(t+1) = w_ij(t) + η·contrib_i(t)·a_i(t)a_j(t) - μγ|w_ij(t)|a_j(t)w_ij(t)

6. Homeostasis:
   θ_i(t+1) = θ_i(t) + ρ(a_i(t) - a_target)

7. Neurogenesis:
   If r_i(t) > R_high, split neuron

8. Apoptosis:
   If r_i(t) < R_low, kill neuron

9. Global energy:
   E_total(t+1) = E_total(t) + E_influx - ∑_i consume_i(t)

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
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from .neuron import MNENeuron, NeuronState
from .synapse import MNESynapse, SynapseState
from .energy import MNEEnergyManager, EnergyState
from .topology import MNETopology, TopologyState, HomeostaticRegulator


@dataclass
class MNEConfig:
    """
    Configuration for MNE.

    Attributes:
        num_neurons: Number of neurons
        activation_fn: Activation function
        # Neuron parameters
        kappa: Baseline metabolic cost coefficient
        gamma: Synaptic transmission cost coefficient
        alpha: Resource gain coefficient
        beta: Resource consumption coefficient
        delta: Resource decay coefficient
        rho: Homeostatic learning rate
        target_activation: Target activation level
        # Synapse parameters
        eta: Learning rate for Hebbian plasticity
        mu: Metabolic penalty coefficient
        weight_init_std: Standard deviation for weight initialization
        sparsity: Initial connection sparsity
        # Energy parameters
        initial_energy: Initial total energy
        energy_influx: Energy influx per time step
        min_energy: Minimum energy threshold
        max_energy: Maximum energy capacity
        # Topology parameters
        max_neurons: Maximum number of neurons
        min_neurons: Minimum number of neurons
        resource_high: High resource threshold for neurogenesis
        resource_low: Low resource threshold for apoptosis
        # System parameters
        device: Device to place tensors on
    """

    num_neurons: int = 100
    activation_fn: str = "tanh"

    # Neuron parameters
    kappa: float = 0.1
    gamma: float = 0.05
    alpha: float = 0.5
    beta: float = 1.0
    delta: float = 0.01
    rho: float = 0.01
    target_activation: float = 0.5
    initial_resource: float = 1.0
    initial_threshold: float = 0.0

    # Synapse parameters
    eta: float = 0.01
    mu: float = 0.1
    weight_init_std: float = 0.1
    weight_clip_min: float = -1.0
    weight_clip_max: float = 1.0
    formation_threshold: float = 0.5
    elimination_threshold: float = 0.01
    sparsity: float = 0.8

    # Energy parameters
    initial_energy: float = 100.0
    energy_influx: float = 10.0
    min_energy: float = 20.0
    max_energy: float = 200.0
    efficiency_window: int = 10
    history_length: int = 100

    # Topology parameters
    max_neurons: int = 1000
    min_neurons: int = 10
    resource_high: float = 2.0
    resource_low: float = 0.1
    neurogenesis_rate: float = 0.1
    apoptosis_rate: float = 0.1

    # System parameters
    device: str = "cpu"


@dataclass
class MNEState:
    """
    Complete state of the MNE system.

    Attributes:
        neuron_state: Neuron state
        synapse_state: Synapse state
        energy_state: Energy state
        topology_state: Topology state
        time_step: Current time step
    """

    neuron_state: NeuronState
    synapse_state: SynapseState
    energy_state: EnergyState
    topology_state: TopologyState
    time_step: int = 0


class MNE(nn.Module):
    """
    Metabolic Neural Ecosystem.

    A biologically inspired neural network architecture with metabolic constraints,
    energy-aware plasticity, and structural plasticity.

    Args:
        config: MNEConfig instance with all parameters

    Example:
        >>> config = MNEConfig(num_neurons=100)
        >>> mne = MNE(config)
        >>> state = mne.get_initial_state(batch_size=32)
        >>> inputs = torch.randn(32, 100)
        >>> output, new_state = mne(inputs, state)
    """

    def __init__(self, config: MNEConfig):
        super().__init__()
        self.config = config

        # Initialize components
        self.neuron = MNENeuron(
            num_neurons=config.num_neurons,
            activation_fn=config.activation_fn,
            kappa=config.kappa,
            gamma=config.gamma,
            alpha=config.alpha,
            beta=config.beta,
            delta=config.delta,
            rho=config.rho,
            target_activation=config.target_activation,
            initial_resource=config.initial_resource,
            initial_threshold=config.initial_threshold,
            device=config.device,
        )

        self.synapse = MNESynapse(
            num_neurons=config.num_neurons,
            eta=config.eta,
            mu=config.mu,
            gamma=config.gamma,
            weight_init_std=config.weight_init_std,
            weight_clip_min=config.weight_clip_min,
            weight_clip_max=config.weight_clip_max,
            formation_threshold=config.formation_threshold,
            elimination_threshold=config.elimination_threshold,
            sparsity=config.sparsity,
            device=config.device,
        )

        self.energy_manager = MNEEnergyManager(
            initial_energy=config.initial_energy,
            energy_influx=config.energy_influx,
            min_energy=config.min_energy,
            max_energy=config.max_energy,
            efficiency_window=config.efficiency_window,
            history_length=config.history_length,
            device=config.device,
        )

        self.topology = MNETopology(
            max_neurons=config.max_neurons,
            min_neurons=config.min_neurons,
            resource_high=config.resource_high,
            resource_low=config.resource_low,
            neurogenesis_rate=config.neurogenesis_rate,
            apoptosis_rate=config.apoptosis_rate,
            device=config.device,
        )

        self.to(config.device)

    def get_initial_state(self, batch_size: int) -> MNEState:
        """
        Initialize MNE state.

        Args:
            batch_size: Batch size

        Returns:
            MNEState: Initial state
        """
        neuron_state = self.neuron.get_initial_state(batch_size)
        synapse_state = self.synapse.get_initial_state()
        energy_state = self.energy_manager.get_initial_state()
        topology_state = self.topology.get_initial_state(self.config.num_neurons)

        return MNEState(
            neuron_state=neuron_state,
            synapse_state=synapse_state,
            energy_state=energy_state,
            topology_state=topology_state,
            time_step=0,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        state: MNEState,
        contribution: Optional[torch.Tensor] = None,
        apply_plasticity: bool = True,
        apply_topology: bool = True,
    ) -> Tuple[torch.Tensor, MNEState]:
        """
        Perform one time step of MNE dynamics.

        Args:
            inputs: External input of shape (batch_size, num_neurons)
            state: Current MNE state
            contribution: Optional gradient-based contribution |∂L/∂a_i|
            apply_plasticity: Whether to apply synaptic plasticity (default: True)
            apply_topology: Whether to apply structural plasticity (default: True)

        Returns:
            Tuple[torch.Tensor, MNEState]:
                - Output activation
                - Updated state
        """
        # Get weights
        weights = self.synapse.get_weights(state.synapse_state)

        # Update neuron dynamics
        activation, neuron_state = self.neuron(
            inputs, weights, state.neuron_state, contribution
        )

        # Update synapse plasticity
        if apply_plasticity and contribution is not None:
            synapse_state = self.synapse.update(
                activation, activation, contribution, state.synapse_state
            )
        else:
            synapse_state = state.synapse_state

        # Update energy
        consumption = self.neuron.get_consumption_levels(neuron_state)
        synapse_consumption = self.synapse.get_energy_cost(synapse_state)

        if contribution is not None:
            energy_state = self.energy_manager.update(
                consumption, contribution, state.energy_state, synapse_consumption
            )
        else:
            energy_state = state.energy_state

        # Update topology
        if apply_topology:
            neuron_state, synapse_state, topology_state = self.topology.update(
                neuron_state, synapse_state, state.topology_state
            )
        else:
            topology_state = state.topology_state

        # Create new state
        new_state = MNEState(
            neuron_state=neuron_state,
            synapse_state=synapse_state,
            energy_state=energy_state,
            topology_state=topology_state,
            time_step=state.time_step + 1,
        )

        return activation, new_state

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor, state: MNEState
    ) -> torch.Tensor:
        """
        Compute loss and gradient-based contribution.

        Args:
            outputs: Network outputs
            targets: Target values
            state: Current MNE state

        Returns:
            torch.Tensor: Loss value
        """
        # Mean squared error loss
        loss = nn.MSELoss()(outputs, targets)

        return loss

    def compute_contribution(
        self, outputs: torch.Tensor, targets: torch.Tensor, state: MNEState
    ) -> torch.Tensor:
        """
        Compute gradient-based contribution |∂L/∂a_i|.

        Args:
            outputs: Network outputs
            targets: Target values
            state: Current MNE state

        Returns:
            torch.Tensor: Contribution of shape (batch_size, num_neurons)
        """
        # Compute loss
        loss = self.compute_loss(outputs, targets, state)

        # Compute gradient w.r.t. activation
        activation = state.neuron_state.activation
        activation.requires_grad_(True)

        # Recompute loss with gradient tracking
        loss_recomputed = self.compute_loss(activation, targets, state)

        # Compute gradient
        grad = torch.autograd.grad(
            loss_recomputed, activation, create_graph=False, retain_graph=False
        )[0]

        # Contribution is absolute gradient
        contribution = torch.abs(grad)

        return contribution

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        state: MNEState,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[torch.Tensor, MNEState, Dict[str, Any]]:
        """
        Perform one training step.

        Args:
            inputs: Input data
            targets: Target data
            state: Current MNE state
            optimizer: Optional optimizer for weight updates

        Returns:
            Tuple of (loss, new_state, metrics)
        """
        # Forward pass
        outputs, state = self.forward(inputs, state, apply_plasticity=False)

        # Compute loss
        loss = self.compute_loss(outputs, targets, state)

        # Compute contribution
        contribution = self.compute_contribution(outputs, targets, state)

        # Backward pass if optimizer provided
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update with plasticity using contribution
        outputs, state = self.forward(
            inputs, state, contribution=contribution, apply_plasticity=True
        )

        # Compute metrics
        metrics = self.get_metrics(state)

        return loss, state, metrics

    def get_metrics(self, state: MNEState) -> Dict[str, Any]:
        """
        Get current metrics.

        Args:
            state: Current MNE state

        Returns:
            Dict of metrics
        """
        metrics = {
            "time_step": state.time_step,
            "num_neurons": self.topology.get_num_neurons(state.topology_state),
            "num_active": self.topology.get_num_active(state.topology_state),
            "total_energy": self.energy_manager.get_total_energy(
                state.energy_state
            ).item(),
            "efficiency": self.energy_manager.get_efficiency(state.energy_state).item(),
            "is_constrained": self.energy_manager.is_constrained(state.energy_state),
            "neurogenesis_count": self.topology.get_neurogenesis_count(
                state.topology_state
            ),
            "apoptosis_count": self.topology.get_apoptosis_count(state.topology_state),
        }

        # Add energy statistics
        energy_stats = self.energy_manager.get_energy_statistics(state.energy_state)
        metrics.update(energy_stats)

        # Add topology statistics
        topology_stats = self.topology.get_topology_statistics(state.topology_state)
        metrics.update(topology_stats)

        return metrics

    def get_weights(self, state: MNEState) -> torch.Tensor:
        """
        Get synaptic weight matrix.

        Args:
            state: Current MNE state

        Returns:
            torch.Tensor: Weight matrix
        """
        return self.synapse.get_weights(state.synapse_state)

    def get_activations(self, state: MNEState) -> torch.Tensor:
        """
        Get neuron activations.

        Args:
            state: Current MNE state

        Returns:
            torch.Tensor: Activations
        """
        return state.neuron_state.activation

    def get_resources(self, state: MNEState) -> torch.Tensor:
        """
        Get neuron resource levels.

        Args:
            state: Current MNE state

        Returns:
            torch.Tensor: Resource levels
        """
        return self.neuron.get_resource_levels(state.neuron_state)

    def reset_state(self, batch_size: int) -> MNEState:
        """
        Reset MNE state to initial values.

        Args:
            batch_size: Batch size

        Returns:
            MNEState: Reset state
        """
        return self.get_initial_state(batch_size)

    def set_energy_influx(self, state: MNEState, influx: float) -> MNEState:
        """
        Set energy influx rate.

        Args:
            state: Current MNE state
            influx: New energy influx rate

        Returns:
            MNEState: Updated state
        """
        state.energy_state = self.energy_manager.set_energy_influx(
            state.energy_state, influx
        )
        return state

    def add_energy(self, state: MNEState, amount: float) -> MNEState:
        """
        Add energy to the system.

        Args:
            state: Current MNE state
            amount: Amount of energy to add

        Returns:
            MNEState: Updated state
        """
        state.energy_state = self.energy_manager.add_energy(state.energy_state, amount)
        return state
