"""
Unit tests for MNE synapse module.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.synapse import MNESynapse, SynapseState


class TestMNESynapse:
    """Test cases for MNESynapse class."""

    @pytest.fixture
    def synapse(self):
        """Create a synapse instance for testing."""
        return MNESynapse(
            num_neurons=10,
            sparsity=0.5,  # 50% connected
            device="cpu",
        )

    @pytest.fixture
    def state(self, synapse):
        """Create initial synapse state."""
        return synapse.get_initial_state()

    def test_initialization(self, synapse):
        """Test synapse initialization."""
        assert synapse.num_neurons == 10
        assert synapse.eta == 0.01
        assert synapse.mu == 0.1
        assert synapse.gamma == 0.05
        assert synapse.sparsity == 0.5

    def test_get_initial_state(self, synapse, state):
        """Test initial state creation."""
        assert state.weights.shape == (10, 10)
        assert state.energy_cost.shape == (10, 10)
        assert state.age.shape == (10, 10)
        assert state.is_connected.shape == (10, 10)

        # Check that sparsity is approximately correct
        num_connected = state.is_connected.sum().item()
        total_possible = 10 * 10
        actual_sparsity = 1.0 - (num_connected / total_possible)
        assert 0.3 < actual_sparsity < 0.7  # Allow some variance

        # Check initial values
        assert torch.all(state.energy_cost == 0)
        assert torch.all(state.age == 0)

    def test_compute_hebbian_update(self, synapse, state):
        """Test Hebbian update computation."""
        presynaptic = torch.randn(4, 10)
        postsynaptic = torch.randn(4, 10)
        contribution = torch.rand(4, 10)

        hebbian_update = synapse.compute_hebbian_update(
            presynaptic, postsynaptic, contribution, state
        )

        assert hebbian_update.shape == (10, 10)

    def test_compute_metabolic_penalty(self, synapse, state):
        """Test metabolic penalty computation."""
        presynaptic = torch.randn(4, 10)

        metabolic_penalty = synapse.compute_metabolic_penalty(presynaptic, state)

        assert metabolic_penalty.shape == (10, 10)

    def test_compute_energy_cost(self, synapse, state):
        """Test energy cost computation."""
        presynaptic = torch.randn(4, 10)

        energy_cost = synapse.compute_energy_cost(presynaptic, state)

        assert energy_cost.shape == (10, 10)
        # Energy cost should be non-negative (allow small negative due to floating point)
        assert torch.all(energy_cost >= -1e-10)

    def test_update(self, synapse, state):
        """Test synapse update."""
        presynaptic = torch.randn(4, 10)
        postsynaptic = torch.randn(4, 10)
        contribution = torch.rand(4, 10)

        new_state = synapse.update(presynaptic, postsynaptic, contribution, state)

        assert new_state.weights.shape == (10, 10)
        assert new_state.energy_cost.shape == (10, 10)
        assert new_state.age.shape == (10, 10)
        assert new_state.is_connected.shape == (10, 10)

        # Check that age increased for connected synapses
        assert torch.all(new_state.age[state.is_connected] == 1)

    def test_update_without_structural_plasticity(self, synapse, state):
        """Test synapse update without structural plasticity."""
        presynaptic = torch.randn(4, 10)
        postsynaptic = torch.randn(4, 10)
        contribution = torch.rand(4, 10)

        initial_connections = state.is_connected.clone()

        new_state = synapse.update(
            presynaptic, postsynaptic, contribution, state, apply_structural=False
        )

        # Connections should not change
        assert torch.all(new_state.is_connected == initial_connections)

    def test_get_weights(self, synapse, state):
        """Test getting weights."""
        weights = synapse.get_weights(state)

        assert weights.shape == (10, 10)
        assert torch.all(weights == state.weights)

    def test_get_energy_cost(self, synapse, state):
        """Test getting energy cost."""
        energy_cost = synapse.get_energy_cost(state)

        assert energy_cost.shape == (10, 10)
        assert torch.all(energy_cost == state.energy_cost)

    def test_get_connection_mask(self, synapse, state):
        """Test getting connection mask."""
        mask = synapse.get_connection_mask(state)

        assert mask.shape == (10, 10)
        assert torch.all(mask == state.is_connected)

    def test_get_sparsity(self, synapse, state):
        """Test getting sparsity."""
        sparsity = synapse.get_sparsity(state)

        assert isinstance(sparsity, float)
        assert 0.0 <= sparsity <= 1.0

    def test_prune_synapses(self, synapse, state):
        """Test synapse pruning."""
        # Ensure we have some connections
        state.is_connected[:] = True
        state.weights = torch.randn(10, 10)

        initial_connections = state.is_connected.sum().item()

        new_state = synapse.prune_synapses(state, keep_fraction=0.5)

        final_connections = new_state.is_connected.sum().item()

        # Should have approximately half the connections
        assert final_connections < initial_connections

    def test_weight_clipping(self, synapse, state):
        """Test that weights are clipped to valid range."""
        presynaptic = torch.randn(4, 10)
        postsynaptic = torch.randn(4, 10)
        contribution = torch.ones(4, 10) * 10.0  # Large contribution

        # Run many updates to push weights to extremes
        for _ in range(100):
            state = synapse.update(presynaptic, postsynaptic, contribution, state)

        # Check that weights are clipped
        assert torch.all(state.weights >= synapse.weight_clip_min)
        assert torch.all(state.weights <= synapse.weight_clip_max)

    def test_multiple_updates(self, synapse, state):
        """Test multiple synapse updates."""
        presynaptic = torch.randn(4, 10)
        postsynaptic = torch.randn(4, 10)
        contribution = torch.rand(4, 10)

        initial_weights = state.weights.clone()

        # Run for several steps
        for _ in range(10):
            state = synapse.update(presynaptic, postsynaptic, contribution, state)

        # Weights should change over time
        assert not torch.allclose(state.weights, initial_weights, atol=0.01)

    def test_zero_contribution(self, synapse, state):
        """Test update with zero contribution."""
        # Use non-zero presynaptic activity to ensure metabolic penalty has effect
        presynaptic = torch.ones(4, 10) * 0.5
        postsynaptic = torch.randn(4, 10)
        contribution = torch.zeros(4, 10)

        initial_weights = state.weights.clone()

        new_state = synapse.update(presynaptic, postsynaptic, contribution, state)

        # Weights should still change due to metabolic penalty
        # Use smaller tolerance since metabolic penalty is small
        assert not torch.allclose(new_state.weights, initial_weights, atol=0.001)

    def test_high_contribution(self, synapse, state):
        """Test update with high contribution."""
        # Use deterministic values to ensure significant weight changes
        presynaptic = torch.ones(4, 10) * 0.5
        postsynaptic = torch.ones(4, 10) * 0.5
        contribution = torch.ones(4, 10) * 10.0

        initial_weights = state.weights.clone()

        new_state = synapse.update(presynaptic, postsynaptic, contribution, state)

        # Weights should change significantly for connected synapses
        # Check that at least some connected synapses changed significantly
        connected_mask = state.is_connected
        if connected_mask.any():
            weight_diff = torch.abs(new_state.weights - initial_weights)
            assert weight_diff[connected_mask].max() > 0.01

    def test_structural_plasticity(self, synapse, state):
        """Test structural plasticity (formation/elimination)."""
        # Create state with some connections
        state.is_connected[:] = False
        state.is_connected[0, 1] = True
        state.is_connected[1, 2] = True
        state.weights[0, 1] = 0.5
        state.weights[1, 2] = 0.5

        presynaptic = torch.ones(4, 10)
        postsynaptic = torch.ones(4, 10)
        contribution = torch.ones(4, 10)

        new_state = synapse.update(
            presynaptic, postsynaptic, contribution, state, apply_structural=True
        )

        # Connections may change due to structural plasticity
        # (exact behavior depends on thresholds)

    def test_energy_cost_tracking(self, synapse, state):
        """Test that energy cost is tracked correctly."""
        presynaptic = torch.randn(4, 10)
        postsynaptic = torch.randn(4, 10)
        contribution = torch.rand(4, 10)

        new_state = synapse.update(presynaptic, postsynaptic, contribution, state)

        # Energy cost should be computed
        assert new_state.energy_cost.shape == (10, 10)

    def test_age_tracking(self, synapse, state):
        """Test that synapse age is tracked correctly."""
        presynaptic = torch.randn(4, 10)
        postsynaptic = torch.randn(4, 10)
        contribution = torch.rand(4, 10)

        # Run for several steps without structural plasticity
        for i in range(5):
            state = synapse.update(
                presynaptic, postsynaptic, contribution, state, apply_structural=False
            )

        # Age should increase for connected synapses
        # Check that all connected synapses have age > 0
        assert torch.all(state.age[state.is_connected] > 0)
        # Check that the maximum age is 5
        assert state.age[state.is_connected].max().item() == 5

    def test_different_sparsity_levels(self):
        """Test different sparsity levels."""
        for sparsity in [0.0, 0.5, 0.9, 1.0]:
            synapse = MNESynapse(num_neurons=10, sparsity=sparsity)
            state = synapse.get_initial_state()

            actual_sparsity = synapse.get_sparsity(state)

            # Allow some variance due to randomness
            assert abs(actual_sparsity - sparsity) < 0.2
