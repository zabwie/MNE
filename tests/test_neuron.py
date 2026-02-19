"""
Unit tests for MNE neuron module.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.neuron import MNENeuron, NeuronState


class TestMNENeuron:
    """Test cases for MNENeuron class."""

    @pytest.fixture
    def neuron(self):
        """Create a neuron instance for testing."""
        return MNENeuron(num_neurons=10, activation_fn="tanh", device="cpu")

    @pytest.fixture
    def state(self, neuron):
        """Create initial neuron state."""
        return neuron.get_initial_state(batch_size=4)

    def test_initialization(self, neuron):
        """Test neuron initialization."""
        assert neuron.num_neurons == 10
        assert neuron.kappa == 0.1
        assert neuron.gamma == 0.05
        assert neuron.alpha == 0.5
        assert neuron.beta == 1.0
        assert neuron.delta == 0.01
        assert neuron.rho == 0.01
        assert neuron.target_activation == 0.5

    def test_get_initial_state(self, neuron, state):
        """Test initial state creation."""
        assert state.activation.shape == (4, 10)
        assert state.resource.shape == (4, 10)
        assert state.threshold.shape == (4, 10)
        assert state.contribution.shape == (4, 10)
        assert state.consumption.shape == (4, 10)
        assert state.is_active.shape == (4, 10)
        assert state.age.shape == (4, 10)

        # Check initial values
        assert torch.all(state.activation == 0)
        assert torch.all(state.resource == 1.0)
        assert torch.all(state.threshold == 0.0)
        assert torch.all(state.contribution == 0)
        assert torch.all(state.consumption == 0)
        assert torch.all(state.is_active)
        assert torch.all(state.age == 0)

    def test_compute_activation(self, neuron, state):
        """Test activation computation."""
        inputs = torch.randn(4, 10)
        weights = torch.randn(10, 10)

        activation = neuron.compute_activation(inputs, weights, state)

        assert activation.shape == (4, 10)
        assert torch.all(activation >= -1)  # tanh range
        assert torch.all(activation <= 1)

    def test_compute_consumption(self, neuron, state):
        """Test energy consumption computation."""
        activation = torch.randn(4, 10)
        weights = torch.randn(10, 10)

        consumption = neuron.compute_consumption(activation, weights, state)

        assert consumption.shape == (4, 10)
        assert torch.all(consumption >= 0)  # Consumption should be non-negative

    def test_update_resource(self, neuron, state):
        """Test resource update."""
        consumption = torch.rand(4, 10) * 0.5

        new_resource = neuron.update_resource(state, consumption)

        assert new_resource.shape == (4, 10)
        assert torch.all(new_resource >= 0)  # Resources should be non-negative

    def test_update_threshold(self, neuron, state):
        """Test threshold update."""
        activation = torch.randn(4, 10)

        new_threshold = neuron.update_threshold(state, activation)

        assert new_threshold.shape == (4, 10)

    def test_forward(self, neuron, state):
        """Test forward pass."""
        inputs = torch.randn(4, 10)
        weights = torch.randn(10, 10)
        contribution = torch.rand(4, 10)

        activation, new_state = neuron.forward(inputs, weights, state, contribution)

        assert activation.shape == (4, 10)
        assert new_state.activation.shape == (4, 10)
        assert new_state.resource.shape == (4, 10)
        assert new_state.threshold.shape == (4, 10)
        assert new_state.contribution.shape == (4, 10)
        assert new_state.consumption.shape == (4, 10)
        assert new_state.is_active.shape == (4, 10)
        assert new_state.age.shape == (4, 10)

        # Check that age increased
        assert torch.all(new_state.age == 1)

    def test_forward_without_contribution(self, neuron, state):
        """Test forward pass without contribution."""
        inputs = torch.randn(4, 10)
        weights = torch.randn(10, 10)

        activation, new_state = neuron.forward(inputs, weights, state)

        assert activation.shape == (4, 10)
        assert torch.all(new_state.contribution == 0)

    def test_set_contribution(self, neuron, state):
        """Test setting contribution."""
        contribution = torch.rand(4, 10)

        new_state = neuron.set_contribution(state, contribution)

        assert torch.all(new_state.contribution == contribution)

    def test_deactivate_neurons(self, neuron, state):
        """Test neuron deactivation."""
        mask = (
            torch.tensor(
                [True, False, False, False, False, False, False, False, False, False]
            )
            .unsqueeze(0)
            .repeat(4, 1)
        )

        new_state = neuron.deactivate_neurons(state, mask)

        assert not new_state.is_active[0, 0]
        assert torch.all(new_state.activation[0, 0] == 0)
        assert torch.all(new_state.resource[0, 0] == 0)

    def test_get_active_neurons(self, neuron, state):
        """Test getting active neuron indices."""
        active_indices = neuron.get_active_neurons(state)

        assert len(active_indices) == 10
        assert torch.all(active_indices == torch.arange(10))

    def test_get_resource_levels(self, neuron, state):
        """Test getting resource levels."""
        resources = neuron.get_resource_levels(state)

        assert resources.shape == (4, 10)
        assert torch.all(resources == 1.0)

    def test_get_consumption_levels(self, neuron, state):
        """Test getting consumption levels."""
        consumptions = neuron.get_consumption_levels(state)

        assert consumptions.shape == (4, 10)
        assert torch.all(consumptions == 0)

    def test_activation_functions(self):
        """Test different activation functions."""
        for activation_fn in ["tanh", "relu", "sigmoid", "leaky_relu"]:
            neuron = MNENeuron(num_neurons=10, activation_fn=activation_fn)
            state = neuron.get_initial_state(batch_size=4)
            inputs = torch.randn(4, 10)
            weights = torch.randn(10, 10)

            activation = neuron.compute_activation(inputs, weights, state)

            assert activation.shape == (4, 10)

    def test_invalid_activation_function(self):
        """Test that invalid activation function raises error."""
        with pytest.raises(ValueError):
            MNENeuron(num_neurons=10, activation_fn="invalid")

    def test_multiple_time_steps(self, neuron, state):
        """Test multiple time steps."""
        inputs = torch.randn(4, 10)
        weights = torch.randn(10, 10)
        contribution = torch.rand(4, 10)

        for _ in range(5):
            activation, state = neuron.forward(inputs, weights, state, contribution)

        assert torch.all(state.age == 5)

    def test_resource_dynamics(self, neuron, state):
        """Test resource dynamics over time."""
        inputs = torch.randn(4, 10)
        weights = torch.randn(10, 10)
        contribution = torch.ones(4, 10) * 0.5  # Constant contribution

        initial_resources = state.resource.clone()

        # Run for several steps
        for _ in range(10):
            activation, state = neuron.forward(inputs, weights, state, contribution)

        # Resources should change over time
        assert not torch.allclose(state.resource, initial_resources, atol=0.01)

    def test_threshold_dynamics(self, neuron, state):
        """Test threshold dynamics over time."""
        inputs = torch.randn(4, 10)
        weights = torch.randn(10, 10)
        contribution = torch.rand(4, 10)

        initial_thresholds = state.threshold.clone()

        # Run for several steps
        for _ in range(10):
            activation, state = neuron.forward(inputs, weights, state, contribution)

        # Thresholds should change over time
        assert not torch.allclose(state.threshold, initial_thresholds, atol=0.01)
