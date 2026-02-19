"""
Unit tests for MNE energy module.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.energy import MNEEnergyManager, EnergyState


class TestMNEEnergyManager:
    """Test cases for MNEEnergyManager class."""

    @pytest.fixture
    def energy_manager(self):
        """Create an energy manager instance for testing."""
        return MNEEnergyManager(
            initial_energy=100.0,
            energy_influx=10.0,
            min_energy=20.0,
            max_energy=200.0,
            device="cpu",
        )

    @pytest.fixture
    def state(self, energy_manager):
        """Create initial energy state."""
        return energy_manager.get_initial_state()

    def test_initialization(self, energy_manager):
        """Test energy manager initialization."""
        assert energy_manager.initial_energy == 100.0
        assert energy_manager.energy_influx == 10.0
        assert energy_manager.min_energy == 20.0
        assert energy_manager.max_energy == 200.0
        assert energy_manager.efficiency_window == 10
        assert energy_manager.history_length == 100

    def test_get_initial_state(self, energy_manager, state):
        """Test initial state creation."""
        assert state.total_energy.item() == 100.0
        assert state.energy_influx.item() == 10.0
        assert state.total_consumption.item() == 0.0
        assert state.efficiency.item() == 0.0
        assert len(state.energy_history) == 0
        assert len(state.consumption_history) == 0
        assert not state.is_energy_constrained

    def test_compute_total_consumption(self, energy_manager):
        """Test total consumption computation."""
        neuron_consumption = torch.rand(4, 10) * 5.0

        total = energy_manager.compute_total_consumption(neuron_consumption)

        assert isinstance(total, torch.Tensor)
        assert total.item() > 0

    def test_compute_total_consumption_with_synapse(self, energy_manager):
        """Test total consumption computation with synapse consumption."""
        neuron_consumption = torch.rand(4, 10) * 5.0
        synapse_consumption = torch.rand(10, 10) * 2.0

        total = energy_manager.compute_total_consumption(
            neuron_consumption, synapse_consumption
        )

        assert isinstance(total, torch.Tensor)
        assert total.item() > 0

    def test_compute_efficiency(self, energy_manager):
        """Test efficiency computation."""
        contribution = torch.rand(4, 10) * 0.5
        consumption = torch.rand(4, 10) * 2.0

        efficiency = energy_manager.compute_efficiency(contribution, consumption)

        assert isinstance(efficiency, torch.Tensor)
        assert efficiency.item() >= 0

    def test_compute_efficiency_zero_consumption(self, energy_manager):
        """Test efficiency computation with zero consumption."""
        contribution = torch.rand(4, 10) * 0.5
        consumption = torch.zeros(4, 10)

        efficiency = energy_manager.compute_efficiency(contribution, consumption)

        assert efficiency.item() == 0.0

    def test_update_energy(self, energy_manager, state):
        """Test energy update."""
        total_consumption = torch.tensor(5.0)

        new_energy = energy_manager.update_energy(total_consumption, state)

        # Energy should increase by influx - consumption
        expected = 100.0 + 10.0 - 5.0
        assert new_energy.item() == expected

    def test_update_energy_clamping(self, energy_manager, state):
        """Test that energy is clamped to valid range."""
        # Test upper bound
        state.total_energy = torch.tensor(195.0)
        total_consumption = torch.tensor(0.0)

        new_energy = energy_manager.update_energy(total_consumption, state)
        assert new_energy.item() <= energy_manager.max_energy

        # Test lower bound
        state.total_energy = torch.tensor(5.0)
        total_consumption = torch.tensor(100.0)

        new_energy = energy_manager.update_energy(total_consumption, state)
        assert new_energy.item() >= 0.0

    def test_check_energy_constraint(self, energy_manager, state):
        """Test energy constraint checking."""
        # Above minimum
        state.total_energy = torch.tensor(50.0)
        assert not energy_manager.check_energy_constraint(state)

        # Below minimum
        state.total_energy = torch.tensor(10.0)
        assert energy_manager.check_energy_constraint(state)

    def test_compute_energy_scaling(self, energy_manager, state):
        """Test energy scaling computation."""
        contribution = torch.rand(4, 10)

        # Not constrained
        state.is_energy_constrained = False
        scaling = energy_manager.compute_energy_scaling(state, contribution)
        assert torch.all(scaling == 1.0)

        # Constrained
        state.is_energy_constrained = True
        state.total_energy = torch.tensor(10.0)
        scaling = energy_manager.compute_energy_scaling(state, contribution)
        assert torch.all(scaling >= 0.1)
        assert torch.all(scaling <= 1.0)

    def test_update(self, energy_manager, state):
        """Test energy state update."""
        neuron_consumption = torch.rand(4, 10) * 5.0
        contribution = torch.rand(4, 10) * 0.5

        new_state = energy_manager.update(neuron_consumption, contribution, state)

        assert new_state.total_energy.shape == ()
        assert new_state.energy_influx.shape == ()
        assert new_state.total_consumption.shape == ()
        assert new_state.efficiency.shape == ()
        assert len(new_state.energy_history) == 1
        assert len(new_state.consumption_history) == 1

    def test_update_with_synapse_consumption(self, energy_manager, state):
        """Test energy state update with synapse consumption."""
        neuron_consumption = torch.rand(4, 10) * 5.0
        synapse_consumption = torch.rand(10, 10) * 2.0
        contribution = torch.rand(4, 10) * 0.5

        new_state = energy_manager.update(
            neuron_consumption, contribution, state, synapse_consumption
        )

        assert new_state.total_consumption.item() > 0

    def test_get_total_energy(self, energy_manager, state):
        """Test getting total energy."""
        energy = energy_manager.get_total_energy(state)

        assert energy.item() == 100.0

    def test_get_efficiency(self, energy_manager, state):
        """Test getting efficiency."""
        efficiency = energy_manager.get_efficiency(state)

        assert efficiency.item() == 0.0

    def test_get_energy_history(self, energy_manager, state):
        """Test getting energy history."""
        history = energy_manager.get_energy_history(state)

        assert isinstance(history, list)
        assert len(history) == 0

    def test_get_consumption_history(self, energy_manager, state):
        """Test getting consumption history."""
        history = energy_manager.get_consumption_history(state)

        assert isinstance(history, list)
        assert len(history) == 0

    def test_is_constrained(self, energy_manager, state):
        """Test checking if energy is constrained."""
        assert not energy_manager.is_constrained(state)

        state.is_energy_constrained = True
        assert energy_manager.is_constrained(state)

    def test_set_energy_influx(self, energy_manager, state):
        """Test setting energy influx."""
        new_state = energy_manager.set_energy_influx(state, 20.0)

        assert new_state.energy_influx.item() == 20.0

    def test_add_energy(self, energy_manager, state):
        """Test adding energy."""
        new_state = energy_manager.add_energy(state, 50.0)

        assert new_state.total_energy.item() == 150.0

    def test_add_energy_clamping(self, energy_manager, state):
        """Test that added energy is clamped to max."""
        new_state = energy_manager.add_energy(state, 200.0)

        assert new_state.total_energy.item() == energy_manager.max_energy

    def test_history_trimming(self, energy_manager, state):
        """Test that history is trimmed to max length."""
        neuron_consumption = torch.rand(4, 10) * 5.0
        contribution = torch.rand(4, 10) * 0.5

        # Add more entries than history_length
        for _ in range(150):
            state = energy_manager.update(neuron_consumption, contribution, state)

        assert len(state.energy_history) <= energy_manager.history_length
        assert len(state.consumption_history) <= energy_manager.history_length

    def test_multiple_updates(self, energy_manager, state):
        """Test multiple energy updates."""
        neuron_consumption = torch.rand(4, 10) * 5.0
        contribution = torch.rand(4, 10) * 0.5

        initial_energy = state.total_energy.clone()

        # Run for several steps
        for _ in range(10):
            state = energy_manager.update(neuron_consumption, contribution, state)

        # Energy should change over time
        assert not torch.allclose(state.total_energy, initial_energy, atol=0.01)

    def test_energy_constraint_activation(self, energy_manager, state):
        """Test that energy constraint is activated when energy is low."""
        neuron_consumption = torch.ones(4, 10) * 10.0  # High consumption
        contribution = torch.rand(4, 10) * 0.5

        # Run until energy is low
        for _ in range(20):
            state = energy_manager.update(neuron_consumption, contribution, state)
            if state.is_energy_constrained:
                break

        # Eventually, energy should become constrained
        assert (
            state.is_energy_constrained
            or state.total_energy.item() < energy_manager.min_energy
        )

    def test_get_energy_statistics(self, energy_manager, state):
        """Test getting energy statistics."""
        stats = energy_manager.get_energy_statistics(state)

        assert "total_energy" in stats
        assert "energy_influx" in stats
        assert "total_consumption" in stats
        assert "efficiency" in stats
        assert "is_constrained" in stats
        assert "energy_level" in stats

    def test_get_energy_statistics_with_history(self, energy_manager, state):
        """Test getting energy statistics with history."""
        neuron_consumption = torch.rand(4, 10) * 5.0
        contribution = torch.rand(4, 10) * 0.5

        # Add some history
        for _ in range(10):
            state = energy_manager.update(neuron_consumption, contribution, state)

        stats = energy_manager.get_energy_statistics(state)

        assert "energy_mean" in stats
        assert "energy_std" in stats
        assert "energy_min" in stats
        assert "energy_max" in stats

    def test_different_initial_energy(self):
        """Test different initial energy levels."""
        for initial_energy in [50.0, 100.0, 150.0]:
            manager = MNEEnergyManager(initial_energy=initial_energy)
            state = manager.get_initial_state()

            assert state.total_energy.item() == initial_energy

    def test_different_energy_influx(self):
        """Test different energy influx rates."""
        for influx in [5.0, 10.0, 20.0]:
            manager = MNEEnergyManager(energy_influx=influx)
            state = manager.get_initial_state()

            assert state.energy_influx.item() == influx
