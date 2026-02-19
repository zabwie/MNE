"""
Energy module for Metabolic Neural Ecosystem (MNE).

This module implements global energy budget management for the neural network,
tracking total energy consumption and enforcing energy constraints.

Mathematical Foundation:
-----------------------
The energy manager implements global energy dynamics:

1. Global energy update:
   E_total(t+1) = E_total(t) + E_influx - ∑_i consume_i(t)

   where:
   - E_total(t): Total energy available at time t
   - E_influx: Energy influx per time step (e.g., from glucose metabolism)
   - consume_i(t): Energy consumption of neuron i

2. Energy constraint enforcement:
   If E_total(t) < E_min:
       - Reduce neuron activity
       - Prioritize high-contribution neurons
       - Trigger apoptosis for low-resource neurons

3. Energy allocation:
   Energy is allocated to neurons based on their contribution to task
   performance, ensuring efficient resource utilization.

4. Energy efficiency metric:
   η_energy = (∑_i contrib_i(t)) / (∑_i consume_i(t))

   This measures how much task performance is achieved per unit of
   energy consumed.

References:
-----------
1. Attwell, D., & Laughlin, S. B. (2001). An energy budget for signaling
   in the grey matter of the brain. Journal of Cerebral Blood Flow &
   Metabolism, 21(10), 1133-1145.

2. Lennie, P. (2003). The cost of cortical computation. Current Biology,
   13(6), 493-497.

3. Sterling, P., & Laughlin, S. (2015). Principles of Neural Design.
   MIT Press.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class EnergyState:
    """
    State container for global energy management.

    Attributes:
        total_energy: Total energy available E_total(t)
        energy_influx: Energy influx per time step E_influx
        total_consumption: Total energy consumption ∑_i consume_i(t)
        efficiency: Energy efficiency η_energy
        energy_history: History of energy levels over time
        consumption_history: History of consumption over time
        is_energy_constrained: Whether energy constraints are active
    """

    total_energy: torch.Tensor
    energy_influx: torch.Tensor
    total_consumption: torch.Tensor
    efficiency: torch.Tensor
    energy_history: List[torch.Tensor]
    consumption_history: List[torch.Tensor]
    is_energy_constrained: bool


class MNEEnergyManager(nn.Module):
    """
    Metabolic Neural Ecosystem Energy Manager.

    Implements global energy budget management with:
    - Energy tracking and history
    - Energy constraint enforcement
    - Energy efficiency monitoring
    - Adaptive energy allocation

    Args:
        initial_energy: Initial total energy (default: 100.0)
        energy_influx: Energy influx per time step (default: 10.0)
        min_energy: Minimum energy threshold (default: 20.0)
        max_energy: Maximum energy capacity (default: 200.0)
        efficiency_window: Window size for efficiency calculation (default: 10)
        history_length: Maximum length of energy history (default: 100)
        device: Device to place tensors on (default: 'cpu')

    Example:
        >>> energy_manager = MNEEnergyManager(initial_energy=100.0)
        >>> state = energy_manager.get_initial_state()
        >>> consumption = torch.tensor([5.0, 3.0, 2.0])
        >>> contribution = torch.tensor([0.8, 0.5, 0.3])
        >>> new_state = energy_manager.update(consumption, contribution, state)
    """

    def __init__(
        self,
        initial_energy: float = 100.0,
        energy_influx: float = 10.0,
        min_energy: float = 20.0,
        max_energy: float = 200.0,
        efficiency_window: int = 10,
        history_length: int = 100,
        device: str = "cpu",
    ):
        super().__init__()
        self.initial_energy = initial_energy
        self.energy_influx = energy_influx
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.efficiency_window = efficiency_window
        self.history_length = history_length
        self.device = device

        self.to(device)

    def get_initial_state(self) -> EnergyState:
        """
        Initialize energy state.

        Returns:
            EnergyState: Initial energy state
        """
        device = self.device

        return EnergyState(
            total_energy=torch.tensor(self.initial_energy, device=device),
            energy_influx=torch.tensor(self.energy_influx, device=device),
            total_consumption=torch.tensor(0.0, device=device),
            efficiency=torch.tensor(0.0, device=device),
            energy_history=[],
            consumption_history=[],
            is_energy_constrained=False,
        )

    def compute_total_consumption(
        self,
        neuron_consumption: torch.Tensor,
        synapse_consumption: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute total energy consumption.

        Args:
            neuron_consumption: Neuron energy consumption of shape (batch_size, num_neurons)
            synapse_consumption: Optional synapse energy consumption of shape (num_neurons, num_neurons)

        Returns:
            torch.Tensor: Total consumption (scalar)
        """
        # Sum over neurons and batch
        total_neuron_consumption = neuron_consumption.sum()

        # Add synapse consumption if provided
        if synapse_consumption is not None:
            total_synapse_consumption = synapse_consumption.sum()
            total_consumption = total_neuron_consumption + total_synapse_consumption
        else:
            total_consumption = total_neuron_consumption

        return total_consumption

    def compute_efficiency(
        self, contribution: torch.Tensor, consumption: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy efficiency.

        Implements: η_energy = (∑_i contrib_i(t)) / (∑_i consume_i(t))

        Args:
            contribution: Neuron contributions of shape (batch_size, num_neurons)
            consumption: Neuron consumptions of shape (batch_size, num_neurons)

        Returns:
            torch.Tensor: Energy efficiency (scalar)
        """
        total_contribution = contribution.sum()
        total_consumption = consumption.sum()

        # Avoid division by zero
        if total_consumption > 1e-6:
            efficiency = total_contribution / total_consumption
        else:
            efficiency = torch.tensor(0.0, device=self.device)

        return efficiency

    def update_energy(
        self, total_consumption: torch.Tensor, state: EnergyState
    ) -> torch.Tensor:
        """
        Update total energy level.

        Implements: E_total(t+1) = E_total(t) + E_influx - ∑_i consume_i(t)

        Args:
            total_consumption: Total energy consumption
            state: Current energy state

        Returns:
            torch.Tensor: Updated total energy
        """
        # Update energy: E_total(t+1) = E_total(t) + E_influx - ∑_i consume_i(t)
        new_energy = state.total_energy + state.energy_influx - total_consumption

        # Clamp to valid range
        new_energy = torch.clamp(new_energy, min=0.0, max=self.max_energy)

        return new_energy

    def check_energy_constraint(self, state: EnergyState) -> bool:
        """
        Check if energy constraints are active.

        Args:
            state: Current energy state

        Returns:
            bool: True if energy is constrained (below minimum threshold)
        """
        return state.total_energy < self.min_energy

    def compute_energy_scaling(
        self, state: EnergyState, contribution: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy scaling factors for neurons.

        When energy is constrained, scale down activity while prioritizing
        high-contribution neurons.

        Args:
            state: Current energy state
            contribution: Neuron contributions of shape (batch_size, num_neurons)

        Returns:
            torch.Tensor: Scaling factors of shape (batch_size, num_neurons)
        """
        if not state.is_energy_constrained:
            # No constraint, full scaling
            return torch.ones_like(contribution)

        # Compute energy deficit
        energy_deficit = self.min_energy - state.total_energy

        # Normalize contribution to get priority weights
        total_contribution = contribution.sum()
        if total_contribution > 1e-6:
            priority = contribution / total_contribution
        else:
            priority = torch.ones_like(contribution) / contribution.numel()

        # Scale based on priority and energy deficit
        # Higher contribution neurons get higher scaling
        scaling = 1.0 - (energy_deficit / self.min_energy) * (1.0 - priority)

        # Ensure minimum scaling
        scaling = torch.clamp(scaling, min=0.1, max=1.0)

        return scaling

    def update(
        self,
        neuron_consumption: torch.Tensor,
        contribution: torch.Tensor,
        state: EnergyState,
        synapse_consumption: Optional[torch.Tensor] = None,
    ) -> EnergyState:
        """
        Update energy state for one time step.

        Args:
            neuron_consumption: Neuron energy consumption of shape (batch_size, num_neurons)
            contribution: Neuron contributions of shape (batch_size, num_neurons)
            state: Current energy state
            synapse_consumption: Optional synapse energy consumption

        Returns:
            EnergyState: Updated energy state
        """
        # Compute total consumption
        total_consumption = self.compute_total_consumption(
            neuron_consumption, synapse_consumption
        )

        # Update total energy
        new_total_energy = self.update_energy(total_consumption, state)

        # Compute efficiency
        efficiency = self.compute_efficiency(contribution, neuron_consumption)

        # Check energy constraint
        is_constrained = new_total_energy < self.min_energy

        # Update history
        energy_history = state.energy_history + [new_total_energy.clone()]
        consumption_history = state.consumption_history + [total_consumption.clone()]

        # Trim history to max length
        if len(energy_history) > self.history_length:
            energy_history = energy_history[-self.history_length :]
            consumption_history = consumption_history[-self.history_length :]

        # Create new state
        new_state = EnergyState(
            total_energy=new_total_energy,
            energy_influx=state.energy_influx,
            total_consumption=total_consumption,
            efficiency=efficiency,
            energy_history=energy_history,
            consumption_history=consumption_history,
            is_energy_constrained=is_constrained,
        )

        return new_state

    def get_total_energy(self, state: EnergyState) -> torch.Tensor:
        """
        Get total energy level.

        Args:
            state: Current energy state

        Returns:
            torch.Tensor: Total energy
        """
        return state.total_energy

    def get_efficiency(self, state: EnergyState) -> torch.Tensor:
        """
        Get energy efficiency.

        Args:
            state: Current energy state

        Returns:
            torch.Tensor: Energy efficiency
        """
        return state.efficiency

    def get_energy_history(self, state: EnergyState) -> List[torch.Tensor]:
        """
        Get energy history.

        Args:
            state: Current energy state

        Returns:
            List[torch.Tensor]: Energy history
        """
        return state.energy_history

    def get_consumption_history(self, state: EnergyState) -> List[torch.Tensor]:
        """
        Get consumption history.

        Args:
            state: Current energy state

        Returns:
            List[torch.Tensor]: Consumption history
        """
        return state.consumption_history

    def is_constrained(self, state: EnergyState) -> bool:
        """
        Check if energy is constrained.

        Args:
            state: Current energy state

        Returns:
            bool: True if energy is constrained
        """
        return state.is_energy_constrained

    def set_energy_influx(self, state: EnergyState, influx: float) -> EnergyState:
        """
        Set energy influx rate.

        Args:
            state: Current energy state
            influx: New energy influx rate

        Returns:
            EnergyState: Updated state
        """
        state.energy_influx = torch.tensor(influx, device=self.device)
        return state

    def add_energy(self, state: EnergyState, amount: float) -> EnergyState:
        """
        Add energy to the system.

        Args:
            state: Current energy state
            amount: Amount of energy to add

        Returns:
            EnergyState: Updated state
        """
        state.total_energy = torch.clamp(
            state.total_energy + amount, min=0.0, max=self.max_energy
        )
        return state

    def get_average_efficiency(
        self, state: EnergyState, window: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get average efficiency over a window.

        Args:
            state: Current energy state
            window: Window size (default: efficiency_window from init)

        Returns:
            torch.Tensor: Average efficiency
        """
        if window is None:
            window = self.efficiency_window

        # We need to compute efficiency from history
        # This is a simplified version - in practice, you'd store efficiency history
        return state.efficiency

    def get_energy_statistics(self, state: EnergyState) -> dict:
        """
        Get energy statistics.

        Args:
            state: Current energy state

        Returns:
            dict: Dictionary of energy statistics
        """
        stats = {
            "total_energy": state.total_energy.item(),
            "energy_influx": state.energy_influx.item(),
            "total_consumption": state.total_consumption.item(),
            "efficiency": state.efficiency.item(),
            "is_constrained": state.is_energy_constrained,
            "energy_level": state.total_energy.item() / self.max_energy,
        }

        if len(state.energy_history) > 0:
            energy_values = [e.item() for e in state.energy_history]
            stats["energy_mean"] = sum(energy_values) / len(energy_values)
            stats["energy_std"] = torch.tensor(energy_values).std().item()
            stats["energy_min"] = min(energy_values)
            stats["energy_max"] = max(energy_values)

        return stats
