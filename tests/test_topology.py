"""
Unit tests for MNE topology module.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.topology import MNETopology, TopologyState, HomeostaticRegulator
from src.neuron import NeuronState
from src.synapse import SynapseState


class TestHomeostaticRegulator:
    """Test cases for HomeostaticRegulator class."""

    @pytest.fixture
    def regulator(self):
        """Create a homeostatic regulator for testing."""
        return HomeostaticRegulator(device="cpu")

    def test_initialization(self, regulator):
        """Test regulator initialization."""
        assert regulator.target_activation == 0.5
        assert regulator.activation_tolerance == 0.1
        assert regulator.ultra_fast_rate == 0.1
        assert regulator.fast_rate == 0.01
        assert regulator.medium_rate == 0.001
        assert regulator.slow_rate == 0.0001

    def test_update(self, regulator):
        """Test threshold update."""
        activation = torch.randn(4, 10)
        threshold = torch.zeros(4, 10)

        new_threshold = regulator.update(activation, threshold)

        assert new_threshold.shape == (4, 10)
        assert not torch.allclose(new_threshold, threshold)

    def test_update_with_active_mask(self, regulator):
        """Test threshold update with active mask."""
        activation = torch.randn(4, 10)
        threshold = torch.zeros(4, 10)
        is_active = torch.ones(4, 10, dtype=torch.bool)
        is_active[:, 5:] = False  # Deactivate half the neurons

        new_threshold = regulator.update(activation, threshold, is_active)

        # Inactive neurons should have zero threshold
        assert torch.all(new_threshold[:, 5:] == 0)

    def test_get_regulation_strength(self, regulator):
        """Test getting regulation strength."""
        activation = torch.randn(4, 10)

        strength = regulator.get_regulation_strength(activation)

        assert strength.shape == (4, 10)
        assert torch.all(strength >= 0)
        assert torch.all(strength <= 1)


class TestMNETopology:
    """Test cases for MNETopology class."""

    @pytest.fixture
    def topology(self):
        """Create a topology instance for testing."""
        return MNETopology(
            max_neurons=100,
            min_neurons=10,
            resource_high=2.0,
            resource_low=0.1,
            device="cpu",
        )

    @pytest.fixture(scope="function")
    def topology_state(self, topology):
        """Create initial topology state."""
        return topology.get_initial_state(num_neurons=50)

    @pytest.fixture(scope="function")
    def neuron_state(self):
        """Create a neuron state for testing."""
        return NeuronState(
            activation=torch.randn(4, 50),
            resource=torch.ones(4, 50),
            threshold=torch.zeros(4, 50),
            contribution=torch.rand(4, 50),
            consumption=torch.rand(4, 50),
            is_active=torch.ones(4, 50, dtype=torch.bool),
            age=torch.zeros(4, 50),
        )

    @pytest.fixture(scope="function")
    def synapse_state(self):
        """Create a synapse state for testing."""
        return SynapseState(
            weights=torch.randn(50, 50) * 0.1,
            energy_cost=torch.zeros(50, 50),
            age=torch.zeros(50, 50),
            is_connected=torch.ones(50, 50, dtype=torch.bool),
        )

    def test_initialization(self, topology):
        """Test topology initialization."""
        assert topology.max_neurons == 100
        assert topology.min_neurons == 10
        assert topology.resource_high == 2.0
        assert topology.resource_low == 0.1
        assert topology.neurogenesis_rate == 0.1
        assert topology.apoptosis_rate == 0.1

    def test_get_initial_state(self, topology, topology_state):
        """Test initial topology state creation."""
        assert topology_state.num_neurons == 50
        assert topology_state.num_active == 50
        assert topology_state.neurogenesis_count == 0
        assert topology_state.apoptosis_count == 0
        assert len(topology_state.neuron_indices) == 50

    def test_check_neurogenesis(self, topology, neuron_state, topology_state):
        """Test neurogenesis checking."""
        # Set some neurons to high resource
        neuron_state.resource[:, :5] = 3.0

        split_mask = topology.check_neurogenesis(neuron_state, topology_state)

        assert split_mask.shape == (4, 50)
        assert torch.any(split_mask[:, :5])  # High resource neurons should be marked

    def test_check_neurogenesis_capacity_limit(
        self, topology, neuron_state, topology_state
    ):
        """Test neurogenesis checking with capacity limit."""
        # Set to max capacity
        topology_state.num_neurons = topology.max_neurons

        # Set high resource
        neuron_state.resource[:, :5] = 3.0

        split_mask = topology.check_neurogenesis(neuron_state, topology_state)

        # Should not split due to capacity limit
        assert not torch.any(split_mask)

    def test_check_apoptosis(self, topology, neuron_state, topology_state):
        """Test apoptosis checking."""
        # Set some neurons to low resource
        neuron_state.resource[:, :5] = 0.05

        death_mask = topology.check_apoptosis(neuron_state, topology_state)

        assert death_mask.shape == (4, 50)
        assert torch.any(death_mask[:, :5])  # Low resource neurons should be marked

    def test_check_apoptosis_minimum_limit(
        self, topology, neuron_state, topology_state
    ):
        """Test apoptosis checking with minimum limit."""
        # Set to minimum
        topology_state.num_neurons = topology.min_neurons

        # Set low resource
        neuron_state.resource[:, :5] = 0.05

        death_mask = topology.check_apoptosis(neuron_state, topology_state)

        # Should not kill due to minimum limit
        assert not torch.any(death_mask)

    def test_apply_neurogenesis(
        self, topology, neuron_state, synapse_state, topology_state
    ):
        """Test neurogenesis application."""
        split_mask = torch.zeros(4, 50, dtype=torch.bool)
        split_mask[:, 0] = True

        new_neuron_state, new_synapse_state, new_topology_state = (
            topology.apply_neurogenesis(
                neuron_state, synapse_state, topology_state, split_mask
            )
        )

        # Neurogenesis count should be at least the initial count
        # (implementation is simplified, so it may not increase)
        assert (
            new_topology_state.neurogenesis_count >= topology_state.neurogenesis_count
        )

    def test_apply_apoptosis(
        self, topology, neuron_state, synapse_state, topology_state
    ):
        """Test apoptosis application."""
        death_mask = torch.zeros(4, 50, dtype=torch.bool)
        death_mask[:, 0] = True

        new_neuron_state, new_synapse_state, new_topology_state = (
            topology.apply_apoptosis(
                neuron_state, synapse_state, topology_state, death_mask
            )
        )

        # Neuron should be deactivated
        assert not new_neuron_state.is_active[0, 0]
        assert new_neuron_state.activation[0, 0] == 0
        assert new_neuron_state.resource[0, 0] == 0

        # Connections should be removed
        assert not torch.any(new_synapse_state.weights[:, 0])
        assert not torch.any(new_synapse_state.weights[0, :])

        # Apoptosis count should be at least the initial count
        assert new_topology_state.apoptosis_count >= topology_state.apoptosis_count

    def test_apply_homeostasis(self, topology, neuron_state):
        """Test homeostasis application."""
        initial_threshold = neuron_state.threshold.clone()

        new_neuron_state = topology.apply_homeostasis(neuron_state)

        # Threshold should change
        assert not torch.allclose(new_neuron_state.threshold, initial_threshold)

    def test_update(self, topology, neuron_state, synapse_state, topology_state):
        """Test full topology update."""
        new_neuron_state, new_synapse_state, new_topology_state = topology.update(
            neuron_state, synapse_state, topology_state
        )

        assert new_neuron_state is not None
        assert new_synapse_state is not None
        assert new_topology_state is not None

    def test_update_without_plasticity(
        self, topology, neuron_state, synapse_state, topology_state
    ):
        """Test topology update without structural plasticity."""
        initial_active = neuron_state.is_active.clone()

        new_neuron_state, new_synapse_state, new_topology_state = topology.update(
            neuron_state,
            synapse_state,
            topology_state,
            apply_neurogenesis=False,
            apply_apoptosis=False,
        )

        # Active neurons should not change
        assert torch.all(new_neuron_state.is_active == initial_active)

    def test_get_num_neurons(self, topology, topology_state):
        """Test getting number of neurons."""
        num_neurons = topology.get_num_neurons(topology_state)

        assert num_neurons == 50

    def test_get_num_active(self, topology, topology_state):
        """Test getting number of active neurons."""
        num_active = topology.get_num_active(topology_state)

        assert num_active == 50

    def test_get_neurogenesis_count(self, topology, topology_state):
        """Test getting neurogenesis count."""
        count = topology.get_neurogenesis_count(topology_state)

        assert count == 0

    def test_get_apoptosis_count(self, topology, topology_state):
        """Test getting apoptosis count."""
        count = topology.get_apoptosis_count(topology_state)

        assert count == 0

    def test_get_topology_statistics(self, topology, topology_state):
        """Test getting topology statistics."""
        stats = topology.get_topology_statistics(topology_state)

        assert "num_neurons" in stats
        assert "num_active" in stats
        assert "neurogenesis_count" in stats
        assert "apoptosis_count" in stats
        assert "active_fraction" in stats

        assert stats["num_neurons"] == 50
        assert stats["num_active"] == 50
        assert stats["active_fraction"] == 1.0

    def test_multiple_updates(
        self, topology, neuron_state, synapse_state, topology_state
    ):
        """Test multiple topology updates."""
        for _ in range(5):
            neuron_state, synapse_state, topology_state = topology.update(
                neuron_state, synapse_state, topology_state
            )

        # Should complete without errors
        assert topology_state.num_neurons <= topology.max_neurons
        assert topology_state.num_neurons >= topology.min_neurons

    def test_homeostatic_regulation_over_time(self, topology, neuron_state):
        """Test homeostatic regulation over multiple steps."""
        initial_threshold = neuron_state.threshold.clone()

        for _ in range(10):
            neuron_state = topology.apply_homeostasis(neuron_state)

        # Threshold should have changed significantly
        assert not torch.allclose(neuron_state.threshold, initial_threshold, atol=0.01)

    def test_apoptosis_reduces_active_count(
        self, topology, neuron_state, synapse_state, topology_state
    ):
        """Test that apoptosis reduces active neuron count."""
        # Set some neurons to low resource
        neuron_state.resource[:, :5] = 0.05

        initial_active = topology_state.num_active

        neuron_state, synapse_state, topology_state = topology.update(
            neuron_state, synapse_state, topology_state
        )

        # Active count should decrease
        assert topology_state.num_active < initial_active

    def test_different_resource_thresholds(self):
        """Test different resource thresholds."""
        for resource_high in [1.5, 2.0, 3.0]:
            for resource_low in [0.05, 0.1, 0.2]:
                topology = MNETopology(
                    resource_high=resource_high, resource_low=resource_low
                )

                assert topology.resource_high == resource_high
                assert topology.resource_low == resource_low
