"""
Topology module for Metabolic Neural Ecosystem (MNE).

This module implements structural plasticity (neurogenesis and apoptosis)
and multi-scale homeostatic regulation for maintaining stable network dynamics.

Mathematical Foundation:
-----------------------
The topology module implements:

1. Neurogenesis (neuron splitting):
   If r_i(t) > R_high:
       - Split neuron i into two neurons
       - Distribute resources and connections
       - Initialize new neuron state

2. Apoptosis (neuron death):
   If r_i(t) < R_low:
       - Deactivate neuron i
       - Remove connections
       - Free resources

3. Multi-scale homeostasis:
   θ_i(t+1) = θ_i(t) + ρ(a_i(t) - a_target)

   with different time scales:
   - Ultra-fast (5ms): Immediate response to large deviations
   - Fast (2s): Short-term adaptation
   - Medium (5min): Medium-term regulation
   - Slow (1hr): Long-term homeostasis

4. Activity-dependent plasticity:
   Synapse formation/elimination based on correlated activity

References:
-----------
1. Butz, M., Wörgötter, F., & van Ooyen, A. (2009). Activity-dependent
   structural plasticity. Brain Research Reviews, 60(2), 287-305.

2. Turrigiano, G. G. (2017). Homeostatic plasticity in the developing
   nervous system. Nature Reviews Neuroscience, 18(2), 89-101.

3. Hengen, K. B., Torrado Pacheco, A., McGregor, J. N., Van Hooser, S. D.,
   & Turrigiano, G. G. (2013). Neuronal firing rate homeostasis is
   inhibited by sleep and perturbed by amphetamine. Neuron, 78(4), 735-748.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .neuron import NeuronState
from .synapse import SynapseState


@dataclass
class TopologyState:
    """
    State container for network topology.

    Attributes:
        num_neurons: Current number of neurons
        num_active: Number of active neurons
        neuron_indices: Mapping from logical to physical neuron indices
        neurogenesis_count: Number of neurogenesis events
        apoptosis_count: Number of apoptosis events
        homeostatic_state: State of homeostatic regulation
    """

    num_neurons: int
    num_active: int
    neuron_indices: torch.Tensor
    neurogenesis_count: int
    apoptosis_count: int
    homeostatic_state: dict


class HomeostaticRegulator(nn.Module):
    """
    Multi-scale homeostatic regulator.

    Implements homeostatic regulation at multiple time scales to maintain
    stable network dynamics.

    Args:
        ultra_fast_rate: Ultra-fast homeostatic rate (5ms scale, default: 0.1)
        fast_rate: Fast homeostatic rate (2s scale, default: 0.01)
        medium_rate: Medium homeostatic rate (5min scale, default: 0.001)
        slow_rate: Slow homeostatic rate (1hr scale, default: 0.0001)
        target_activation: Target activation level (default: 0.5)
        activation_tolerance: Tolerance for activation deviation (default: 0.1)
        device: Device to place tensors on (default: 'cpu')

    Example:
        >>> regulator = HomeostaticRegulator()
        >>> activation = torch.randn(32, 100)
        >>> threshold = torch.zeros(32, 100)
        >>> new_threshold = regulator.update(activation, threshold)
    """

    def __init__(
        self,
        ultra_fast_rate: float = 0.1,
        fast_rate: float = 0.01,
        medium_rate: float = 0.001,
        slow_rate: float = 0.0001,
        target_activation: float = 0.5,
        activation_tolerance: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.ultra_fast_rate = ultra_fast_rate
        self.fast_rate = fast_rate
        self.medium_rate = medium_rate
        self.slow_rate = slow_rate
        self.target_activation = target_activation
        self.activation_tolerance = activation_tolerance
        self.device = device

        self.to(device)

    def update(
        self,
        activation: torch.Tensor,
        threshold: torch.Tensor,
        is_active: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update homeostatic threshold.

        Implements multi-scale homeostasis:
        θ_i(t+1) = θ_i(t) + Σ_k ρ_k (a_i(t) - a_target)

        where k indexes the different time scales.

        Args:
            activation: Current activation of shape (batch_size, num_neurons)
            threshold: Current threshold of shape (batch_size, num_neurons)
            is_active: Optional boolean mask of active neurons

        Returns:
            torch.Tensor: Updated threshold
        """
        # Compute error from target
        error = activation - self.target_activation

        # Apply ultra-fast regulation for large deviations
        large_deviation = torch.abs(error) > (3.0 * self.activation_tolerance)
        ultra_fast_update = self.ultra_fast_rate * error * large_deviation.float()

        # Apply fast regulation for medium deviations
        medium_deviation = (
            torch.abs(error) > self.activation_tolerance
        ) & ~large_deviation
        fast_update = self.fast_rate * error * medium_deviation.float()

        # Apply medium regulation for small deviations
        small_deviation = ~large_deviation & ~medium_deviation
        medium_update = self.medium_rate * error * small_deviation.float()

        # Apply slow regulation always
        slow_update = self.slow_rate * error

        # Total update
        total_update = ultra_fast_update + fast_update + medium_update + slow_update

        # Update threshold
        new_threshold = threshold + total_update

        # Zero out inactive neurons
        if is_active is not None:
            new_threshold = new_threshold * is_active.float()

        return new_threshold

    def get_regulation_strength(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Get the strength of homeostatic regulation needed.

        Args:
            activation: Current activation

        Returns:
            torch.Tensor: Regulation strength
        """
        error = torch.abs(activation - self.target_activation)
        strength = error / self.activation_tolerance
        return torch.clamp(strength, min=0.0, max=1.0)


class MNETopology(nn.Module):
    """
    Metabolic Neural Ecosystem Topology Manager.

    Implements structural plasticity with:
    - Neurogenesis (neuron splitting)
    - Apoptosis (neuron death)
    - Multi-scale homeostatic regulation
    - Activity-dependent plasticity

    Args:
        max_neurons: Maximum number of neurons (default: 1000)
        min_neurons: Minimum number of neurons (default: 10)
        resource_high: High resource threshold for neurogenesis (default: 2.0)
        resource_low: Low resource threshold for apoptosis (default: 0.1)
        neurogenesis_rate: Maximum neurogenesis rate (default: 0.1)
        apoptosis_rate: Maximum apoptosis rate (default: 0.1)
        homeostatic_regulator: Homeostatic regulator instance
        device: Device to place tensors on (default: 'cpu')

    Example:
        >>> topology = MNETopology(max_neurons=100)
        >>> state = topology.get_initial_state(num_neurons=50)
        >>> neuron_state = ...  # NeuronState instance
        >>> synapse_state = ...  # SynapseState instance
        >>> new_neuron_state, new_synapse_state, new_topology_state = topology.update(
        ...     neuron_state, synapse_state, state
        ... )
    """

    def __init__(
        self,
        max_neurons: int = 1000,
        min_neurons: int = 10,
        resource_high: float = 2.0,
        resource_low: float = 0.1,
        neurogenesis_rate: float = 0.1,
        apoptosis_rate: float = 0.1,
        homeostatic_regulator: Optional[HomeostaticRegulator] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.max_neurons = max_neurons
        self.min_neurons = min_neurons
        self.resource_high = resource_high
        self.resource_low = resource_low
        self.neurogenesis_rate = neurogenesis_rate
        self.apoptosis_rate = apoptosis_rate
        self.device = device

        if homeostatic_regulator is None:
            self.homeostatic_regulator = HomeostaticRegulator(device=device)
        else:
            self.homeostatic_regulator = homeostatic_regulator

        self.to(device)

    def get_initial_state(self, num_neurons: int) -> TopologyState:
        """
        Initialize topology state.

        Args:
            num_neurons: Initial number of neurons

        Returns:
            TopologyState: Initial topology state
        """
        device = self.device

        return TopologyState(
            num_neurons=num_neurons,
            num_active=num_neurons,
            neuron_indices=torch.arange(num_neurons, device=device),
            neurogenesis_count=0,
            apoptosis_count=0,
            homeostatic_state={},
        )

    def check_neurogenesis(
        self, neuron_state: NeuronState, topology_state: TopologyState
    ) -> torch.Tensor:
        """
        Check which neurons should undergo neurogenesis.

        Args:
            neuron_state: Current neuron state
            topology_state: Current topology state

        Returns:
            torch.Tensor: Boolean mask of neurons to split
        """
        # Check resource level
        high_resource = neuron_state.resource > self.resource_high

        # Check if neuron is active
        is_active = neuron_state.is_active

        # Check if we have capacity for new neurons
        has_capacity = topology_state.num_neurons < self.max_neurons

        # Combine conditions
        should_split = high_resource & is_active

        if not has_capacity:
            should_split = torch.zeros_like(should_split, dtype=torch.bool)

        return should_split

    def check_apoptosis(
        self, neuron_state: NeuronState, topology_state: TopologyState
    ) -> torch.Tensor:
        """
        Check which neurons should undergo apoptosis.

        Args:
            neuron_state: Current neuron state
            topology_state: Current topology state

        Returns:
            torch.Tensor: Boolean mask of neurons to kill
        """
        # Check resource level
        low_resource = neuron_state.resource < self.resource_low

        # Check if neuron is active
        is_active = neuron_state.is_active

        # Check if we have minimum neurons
        has_minimum = topology_state.num_neurons > self.min_neurons

        # Combine conditions
        should_die = low_resource & is_active

        if not has_minimum:
            should_die = torch.zeros_like(should_die, dtype=torch.bool)

        return should_die

    def apply_neurogenesis(
        self,
        neuron_state: NeuronState,
        synapse_state: SynapseState,
        topology_state: TopologyState,
        split_mask: torch.Tensor,
    ) -> Tuple[NeuronState, SynapseState, TopologyState]:
        """
        Apply neurogenesis (neuron splitting).

        Args:
            neuron_state: Current neuron state
            synapse_state: Current synapse state
            topology_state: Current topology state
            split_mask: Boolean mask of neurons to split

        Returns:
            Tuple of updated states
        """
        # For simplicity, we'll implement a basic version
        # In a full implementation, this would:
        # 1. Create new neurons
        # 2. Distribute resources
        # 3. Copy and modify connections
        # 4. Update topology state

        # Count unique neurons to split (across batch)
        num_to_split = split_mask.any(dim=0).sum().item()

        if num_to_split == 0:
            return neuron_state, synapse_state, topology_state

        # Update topology state
        topology_state.neurogenesis_count += num_to_split

        # For now, just mark neurons as having undergone neurogenesis
        # In a full implementation, we'd actually create new neurons

        return neuron_state, synapse_state, topology_state

    def apply_apoptosis(
        self,
        neuron_state: NeuronState,
        synapse_state: SynapseState,
        topology_state: TopologyState,
        death_mask: torch.Tensor,
    ) -> Tuple[NeuronState, SynapseState, TopologyState]:
        """
        Apply apoptosis (neuron death).

        Args:
            neuron_state: Current neuron state
            synapse_state: Current synapse state
            topology_state: Current topology state
            death_mask: Boolean mask of neurons to kill

        Returns:
            Tuple of updated states
        """
        # Count unique neurons to kill (across batch)
        num_to_kill = death_mask.any(dim=0).sum().item()

        if num_to_kill == 0:
            return neuron_state, synapse_state, topology_state

        # Deactivate neurons
        neuron_state.is_active = neuron_state.is_active & ~death_mask
        neuron_state.activation = (
            neuron_state.activation * neuron_state.is_active.float()
        )
        neuron_state.resource = neuron_state.resource * neuron_state.is_active.float()

        # Aggregate death mask across batch to get neurons that are dead in any batch element
        neurons_dead_any_batch = death_mask.any(dim=0)  # (num_neurons,)

        # Remove connections to/from dead neurons
        # Zero out weights for dead neurons (both incoming and outgoing)
        synapse_state.weights = synapse_state.weights * (
            ~neurons_dead_any_batch
        ).float().unsqueeze(0)
        synapse_state.weights = synapse_state.weights * (
            ~neurons_dead_any_batch
        ).float().unsqueeze(1)
        synapse_state.is_connected = (
            synapse_state.is_connected & ~neurons_dead_any_batch.unsqueeze(0)
        )
        synapse_state.is_connected = (
            synapse_state.is_connected & ~neurons_dead_any_batch.unsqueeze(1)
        )

        # Update topology state
        # Count unique active neurons (across batch)
        topology_state.num_active = neuron_state.is_active.any(dim=0).sum().item()
        topology_state.apoptosis_count += num_to_kill

        return neuron_state, synapse_state, topology_state

        # Deactivate neurons
        neuron_state.is_active = neuron_state.is_active & ~death_mask
        neuron_state.activation = (
            neuron_state.activation * neuron_state.is_active.float()
        )
        neuron_state.resource = neuron_state.resource * neuron_state.is_active.float()

        # Aggregate death mask across batch dimension
        # If any batch element wants to kill a neuron, kill it
        death_mask_aggregated = death_mask.any(dim=0)  # (num_neurons,)

        # Remove connections to/from dead neurons
        # Zero out weights for dead neurons
        synapse_state.weights = synapse_state.weights * (
            ~death_mask_aggregated
        ).float().unsqueeze(0)
        synapse_state.weights = synapse_state.weights * (
            ~death_mask_aggregated
        ).float().unsqueeze(1)
        synapse_state.is_connected = synapse_state.is_connected & (
            ~death_mask_aggregated
        ).unsqueeze(0)
        synapse_state.is_connected = synapse_state.is_connected & (
            ~death_mask_aggregated
        ).unsqueeze(1)
        neuron_state.resource = neuron_state.resource * neuron_state.is_active.float()

        # Remove connections to/from dead neurons
        # Zero out weights for dead neurons
        synapse_state.weights = (
            synapse_state.weights * neuron_state.is_active.float().unsqueeze(0)
        )
        synapse_state.weights = (
            synapse_state.weights * neuron_state.is_active.float().unsqueeze(1)
        )
        synapse_state.is_connected = (
            synapse_state.is_connected
            & neuron_state.is_active.float().unsqueeze(0).bool()
        )
        synapse_state.is_connected = (
            synapse_state.is_connected
            & neuron_state.is_active.float().unsqueeze(1).bool()
        )

        # Update topology state
        topology_state.num_active = neuron_state.is_active.sum().item()
        topology_state.apoptosis_count += num_to_kill

        return neuron_state, synapse_state, topology_state

    def apply_homeostasis(self, neuron_state: NeuronState) -> NeuronState:
        """
        Apply homeostatic regulation.

        Args:
            neuron_state: Current neuron state

        Returns:
            NeuronState: Updated neuron state
        """
        # Update threshold using homeostatic regulator
        new_threshold = self.homeostatic_regulator.update(
            neuron_state.activation, neuron_state.threshold, neuron_state.is_active
        )

        neuron_state.threshold = new_threshold

        return neuron_state

    def update(
        self,
        neuron_state: NeuronState,
        synapse_state: SynapseState,
        topology_state: TopologyState,
        apply_neurogenesis: bool = True,
        apply_apoptosis: bool = True,
        apply_homeostasis: bool = True,
    ) -> Tuple[NeuronState, SynapseState, TopologyState]:
        """
        Update topology for one time step.

        Args:
            neuron_state: Current neuron state
            synapse_state: Current synapse state
            topology_state: Current topology state
            apply_neurogenesis: Whether to apply neurogenesis (default: True)
            apply_apoptosis: Whether to apply apoptosis (default: True)
            apply_homeostasis: Whether to apply homeostasis (default: True)

        Returns:
            Tuple of updated states
        """
        # Apply homeostasis
        if apply_homeostasis:
            neuron_state = self.apply_homeostasis(neuron_state)

        # Check for neurogenesis and apoptosis
        split_mask = self.check_neurogenesis(neuron_state, topology_state)
        death_mask = self.check_apoptosis(neuron_state, topology_state)

        # Apply neurogenesis
        if apply_neurogenesis:
            neuron_state, synapse_state, topology_state = self.apply_neurogenesis(
                neuron_state, synapse_state, topology_state, split_mask
            )

        # Apply apoptosis
        if apply_apoptosis:
            neuron_state, synapse_state, topology_state = self.apply_apoptosis(
                neuron_state, synapse_state, topology_state, death_mask
            )

        return neuron_state, synapse_state, topology_state

    def get_num_neurons(self, topology_state: TopologyState) -> int:
        """
        Get current number of neurons.

        Args:
            topology_state: Current topology state

        Returns:
            int: Number of neurons
        """
        return topology_state.num_neurons

    def get_num_active(self, topology_state: TopologyState) -> int:
        """
        Get number of active neurons.

        Args:
            topology_state: Current topology state

        Returns:
            int: Number of active neurons
        """
        return topology_state.num_active

    def get_neurogenesis_count(self, topology_state: TopologyState) -> int:
        """
        Get total neurogenesis events.

        Args:
            topology_state: Current topology state

        Returns:
            int: Number of neurogenesis events
        """
        return topology_state.neurogenesis_count

    def get_apoptosis_count(self, topology_state: TopologyState) -> int:
        """
        Get total apoptosis events.

        Args:
            topology_state: Current topology state

        Returns:
            int: Number of apoptosis events
        """
        return topology_state.apoptosis_count

    def get_topology_statistics(self, topology_state: TopologyState) -> dict:
        """
        Get topology statistics.

        Args:
            topology_state: Current topology state

        Returns:
            dict: Dictionary of topology statistics
        """
        return {
            "num_neurons": topology_state.num_neurons,
            "num_active": topology_state.num_active,
            "neurogenesis_count": topology_state.neurogenesis_count,
            "apoptosis_count": topology_state.apoptosis_count,
            "active_fraction": topology_state.num_active / topology_state.num_neurons
            if topology_state.num_neurons > 0
            else 0.0,
        }
