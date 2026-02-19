"""
Synapse module for Metabolic Neural Ecosystem (MNE).

This module implements energy-aware synaptic connections with plasticity
rules that incorporate metabolic costs.

Mathematical Foundation:
-----------------------
The synapse implements energy-aware Hebbian plasticity:

1. Weight update (energy-aware Hebbian):
   w_ij(t+1) = w_ij(t) + η·contrib_i(t)·a_i(t)a_j(t) - μγ|w_ij(t)|a_j(t)w_ij(t)

   where:
   - w_ij(t): synaptic weight from neuron j to i
   - η: learning rate
   - contrib_i(t): gradient-based contribution of postsynaptic neuron
   - a_i(t), a_j(t): activations of postsynaptic and presynaptic neurons
   - μ: metabolic penalty coefficient
   - γ: synaptic transmission cost coefficient
   - The second term penalizes weights that consume energy

2. Synaptic energy cost:
   E_synapse_ij = γ |w_ij(t)| a_j(t)

   This represents the metabolic cost of transmitting signals through
   the synapse, proportional to weight magnitude and presynaptic activity.

3. Structural plasticity:
   - Synapse formation: If correlated activity exceeds threshold
   - Synapse elimination: If weight magnitude falls below threshold

References:
-----------
1. Bi, G. Q., & Poo, M. M. (2001). Synaptic modification by correlated
   activity: Hebb's postulate revisited. Annual Review of Neuroscience,
   24(1), 139-166.

2. Levy, W. B., & Baxter, R. A. (1996). Energy efficient neural codes.
   Neural Computation, 8(3), 531-543.

3. Chklovskii, D. B., Mel, B. W., & Svoboda, K. (2004). Cortical
   rewiring and information storage. Nature, 431(7010), 782-788.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SynapseState:
    """
    State container for synaptic connections.

    Attributes:
        weights: Synaptic weight matrix w_ij of shape (num_neurons, num_neurons)
        energy_cost: Energy cost of each synapse
        age: Age of each synapse (number of time steps)
        is_connected: Boolean mask of active connections
    """

    weights: torch.Tensor
    energy_cost: torch.Tensor
    age: torch.Tensor
    is_connected: torch.Tensor


class MNESynapse(nn.Module):
    """
    Metabolic Neural Ecosystem Synapse.

    Implements energy-aware synaptic plasticity with:
    - Hebbian learning modulated by metabolic cost
    - Structural plasticity (formation/elimination)
    - Energy cost tracking

    Args:
        num_neurons: Number of neurons
        eta: Learning rate for Hebbian plasticity (default: 0.01)
        mu: Metabolic penalty coefficient (default: 0.1)
        gamma: Synaptic transmission cost coefficient (default: 0.05)
        weight_init_std: Standard deviation for weight initialization (default: 0.1)
        weight_clip_min: Minimum weight value (default: -1.0)
        weight_clip_max: Maximum weight value (default: 1.0)
        formation_threshold: Activity correlation threshold for synapse formation (default: 0.5)
        elimination_threshold: Weight magnitude threshold for synapse elimination (default: 0.01)
        sparsity: Initial connection sparsity (default: 0.8, i.e., 20% connected)
        device: Device to place tensors on (default: 'cpu')

    Example:
        >>> synapse = MNESynapse(num_neurons=100)
        >>> state = synapse.get_initial_state()
        >>> presynaptic = torch.randn(32, 100)
        >>> postsynaptic = torch.randn(32, 100)
        >>> contribution = torch.abs(torch.randn(32, 100))
        >>> new_state = synapse.update(presynaptic, postsynaptic, contribution, state)
    """

    def __init__(
        self,
        num_neurons: int,
        eta: float = 0.01,
        mu: float = 0.1,
        gamma: float = 0.05,
        weight_init_std: float = 0.1,
        weight_clip_min: float = -1.0,
        weight_clip_max: float = 1.0,
        formation_threshold: float = 0.5,
        elimination_threshold: float = 0.01,
        sparsity: float = 0.8,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.eta = eta
        self.mu = mu
        self.gamma = gamma
        self.weight_init_std = weight_init_std
        self.weight_clip_min = weight_clip_min
        self.weight_clip_max = weight_clip_max
        self.formation_threshold = formation_threshold
        self.elimination_threshold = elimination_threshold
        self.sparsity = sparsity
        self.device = device

        self.to(device)

    def get_initial_state(self) -> SynapseState:
        """
        Initialize synaptic state.

        Returns:
            SynapseState: Initial synaptic state
        """
        device = self.device
        shape = (self.num_neurons, self.num_neurons)

        # Initialize weights with random values
        weights = torch.randn(shape, device=device) * self.weight_init_std

        # Create sparse connection mask
        connection_prob = 1.0 - self.sparsity
        is_connected = torch.rand(shape, device=device) < connection_prob

        # Zero out non-connected weights
        weights = weights * is_connected.float()

        # Initialize energy cost and age
        energy_cost = torch.zeros(shape, device=device)
        age = torch.zeros(shape, device=device)

        return SynapseState(
            weights=weights, energy_cost=energy_cost, age=age, is_connected=is_connected
        )

    def compute_hebbian_update(
        self,
        presynaptic: torch.Tensor,
        postsynaptic: torch.Tensor,
        contribution: torch.Tensor,
        state: SynapseState,
    ) -> torch.Tensor:
        """
        Compute Hebbian weight update.

        Implements: Δw_ij = η·contrib_i(t)·a_i(t)a_j(t)

        Args:
            presynaptic: Presynaptic activations a_j(t) of shape (batch_size, num_neurons)
            postsynaptic: Postsynaptic activations a_i(t) of shape (batch_size, num_neurons)
            contribution: Gradient-based contribution contrib_i(t) of shape (batch_size, num_neurons)
            state: Current synaptic state

        Returns:
            torch.Tensor: Hebbian weight update of shape (num_neurons, num_neurons)
        """
        batch_size = presynaptic.shape[0]

        # Compute outer product for each sample in batch
        # contrib_i(t)·a_i(t)a_j(t)
        hebbian_update = torch.zeros(
            (self.num_neurons, self.num_neurons), device=self.device
        )

        for b in range(batch_size):
            # Outer product: a_i(t) * a_j(t)
            outer = torch.outer(postsynaptic[b], presynaptic[b])

            # Scale by contribution
            scaled = outer * contribution[b].unsqueeze(1)

            hebbian_update += scaled

        # Average over batch
        hebbian_update = hebbian_update / batch_size

        # Scale by learning rate
        hebbian_update = self.eta * hebbian_update

        return hebbian_update

    def compute_metabolic_penalty(
        self, presynaptic: torch.Tensor, state: SynapseState
    ) -> torch.Tensor:
        """
        Compute metabolic penalty for weight update.

        Implements: -μγ|w_ij(t)|a_j(t)w_ij(t)

        Args:
            presynaptic: Presynaptic activations a_j(t) of shape (batch_size, num_neurons)
            state: Current synaptic state

        Returns:
            torch.Tensor: Metabolic penalty of shape (num_neurons, num_neurons)
        """
        batch_size = presynaptic.shape[0]

        # Compute average presynaptic activity
        avg_presynaptic = presynaptic.mean(dim=0)  # (num_neurons,)

        # Metabolic penalty: -μγ|w_ij(t)|a_j(t)w_ij(t)
        abs_weights = torch.abs(state.weights)
        metabolic_penalty = -self.mu * self.gamma * abs_weights * state.weights

        # Scale by presynaptic activity
        metabolic_penalty = metabolic_penalty * avg_presynaptic.unsqueeze(0)

        return metabolic_penalty

    def compute_energy_cost(
        self, presynaptic: torch.Tensor, state: SynapseState
    ) -> torch.Tensor:
        """
        Compute energy cost for each synapse.

        Implements: E_synapse_ij = γ |w_ij(t)| |a_j(t)|

        Args:
            presynaptic: Presynaptic activations a_j(t) of shape (batch_size, num_neurons)
            state: Current synaptic state

        Returns:
            torch.Tensor: Energy cost of shape (num_neurons, num_neurons)
        """
        # Compute average presynaptic activity
        avg_presynaptic = presynaptic.mean(dim=0)  # (num_neurons,)

        # Energy cost: γ |w_ij(t)| |a_j(t)|
        # Use absolute value of presynaptic activity to ensure non-negative energy cost
        energy_cost = (
            self.gamma
            * torch.abs(state.weights)
            * torch.abs(avg_presynaptic).unsqueeze(0)
        )

        return energy_cost

    def apply_structural_plasticity(
        self, presynaptic: torch.Tensor, postsynaptic: torch.Tensor, state: SynapseState
    ) -> SynapseState:
        """
        Apply structural plasticity: synapse formation and elimination.

        Args:
            presynaptic: Presynaptic activations
            postsynaptic: Postsynaptic activations
            state: Current synaptic state

        Returns:
            SynapseState: Updated state with structural changes
        """
        # Compute correlation between pre- and post-synaptic activity
        batch_size = presynaptic.shape[0]

        # Average correlation over batch
        correlation = torch.zeros(
            (self.num_neurons, self.num_neurons), device=self.device
        )

        for b in range(batch_size):
            outer = torch.outer(postsynaptic[b], presynaptic[b])
            correlation += outer

        correlation = correlation / batch_size

        # Synapse formation: if correlation exceeds threshold and not connected
        new_connections = (correlation > self.formation_threshold) & (
            ~state.is_connected
        )

        # Initialize new weights with small random values
        new_weights = torch.randn_like(state.weights) * self.weight_init_std * 0.1
        state.weights = torch.where(new_connections, new_weights, state.weights)
        state.is_connected = state.is_connected | new_connections

        # Synapse elimination: if weight magnitude falls below threshold
        weak_connections = (
            torch.abs(state.weights) < self.elimination_threshold
        ) & state.is_connected
        state.weights = torch.where(
            weak_connections, torch.zeros_like(state.weights), state.weights
        )
        state.is_connected = state.is_connected & ~weak_connections

        return state

    def update(
        self,
        presynaptic: torch.Tensor,
        postsynaptic: torch.Tensor,
        contribution: torch.Tensor,
        state: SynapseState,
        apply_structural: bool = True,
    ) -> SynapseState:
        """
        Update synaptic weights based on energy-aware Hebbian plasticity.

        Implements: w_ij(t+1) = w_ij(t) + η·contrib_i(t)·a_i(t)a_j(t) - μγ|w_ij(t)|a_j(t)w_ij(t)

        Args:
            presynaptic: Presynaptic activations a_j(t) of shape (batch_size, num_neurons)
            postsynaptic: Postsynaptic activations a_i(t) of shape (batch_size, num_neurons)
            contribution: Gradient-based contribution contrib_i(t) of shape (batch_size, num_neurons)
            state: Current synaptic state
            apply_structural: Whether to apply structural plasticity (default: True)

        Returns:
            SynapseState: Updated synaptic state
        """
        # Compute Hebbian update
        hebbian_update = self.compute_hebbian_update(
            presynaptic, postsynaptic, contribution, state
        )

        # Compute metabolic penalty
        metabolic_penalty = self.compute_metabolic_penalty(presynaptic, state)

        # Total weight update
        weight_update = hebbian_update + metabolic_penalty

        # Apply update only to connected synapses
        weight_update = weight_update * state.is_connected.float()

        # Update weights
        new_weights = state.weights + weight_update

        # Clip weights to valid range
        new_weights = torch.clamp(
            new_weights, self.weight_clip_min, self.weight_clip_max
        )

        # Zero out non-connected weights
        new_weights = new_weights * state.is_connected.float()

        # Compute energy cost
        energy_cost = self.compute_energy_cost(presynaptic, state)

        # Update age
        new_age = state.age + state.is_connected.float()

        # Create new state
        new_state = SynapseState(
            weights=new_weights,
            energy_cost=energy_cost,
            age=new_age,
            is_connected=state.is_connected,
        )

        # Apply structural plasticity
        if apply_structural:
            new_state = self.apply_structural_plasticity(
                presynaptic, postsynaptic, new_state
            )

        return new_state

    def get_weights(self, state: SynapseState) -> torch.Tensor:
        """
        Get synaptic weight matrix.

        Args:
            state: Current synaptic state

        Returns:
            torch.Tensor: Weight matrix
        """
        return state.weights

    def get_energy_cost(self, state: SynapseState) -> torch.Tensor:
        """
        Get energy cost matrix.

        Args:
            state: Current synaptic state

        Returns:
            torch.Tensor: Energy cost matrix
        """
        return state.energy_cost

    def get_connection_mask(self, state: SynapseState) -> torch.Tensor:
        """
        Get connection mask.

        Args:
            state: Current synaptic state

        Returns:
            torch.Tensor: Boolean connection mask
        """
        return state.is_connected

    def get_sparsity(self, state: SynapseState) -> float:
        """
        Get current sparsity (fraction of non-connected synapses).

        Args:
            state: Current synaptic state

        Returns:
            float: Sparsity value
        """
        num_connections = state.is_connected.sum().item()
        total_possible = self.num_neurons * self.num_neurons
        return 1.0 - (num_connections / total_possible)

    def prune_synapses(
        self, state: SynapseState, keep_fraction: float = 0.5
    ) -> SynapseState:
        """
        Prune synapses to maintain a target sparsity.

        Keeps the strongest connections based on absolute weight magnitude.

        Args:
            state: Current synaptic state
            keep_fraction: Fraction of connections to keep (default: 0.5)

        Returns:
            SynapseState: Updated state with pruned synapses
        """
        # Get absolute weights
        abs_weights = torch.abs(state.weights)

        # Flatten and get threshold
        flat_weights = abs_weights[state.is_connected]
        if flat_weights.numel() > 0:
            num_keep = int(len(flat_weights) * keep_fraction)
            if num_keep > 0:
                threshold = torch.kthvalue(flat_weights, num_keep).values

                # Create mask for synapses to keep
                keep_mask = abs_weights >= threshold

                # Update state
                state.weights = state.weights * keep_mask.float()
                state.is_connected = state.is_connected & keep_mask

        return state
