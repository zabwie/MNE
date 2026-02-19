"""
Basic usage example for Metabolic Neural Ecosystem (MNE).

This example demonstrates how to create, train, and use an MNE network
for a simple classification task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import MNE components
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core import MNE, MNEConfig


def create_synthetic_data(num_samples=1000, num_features=20, num_classes=3):
    """Create synthetic classification data."""
    # Generate random features
    X = torch.randn(num_samples, num_features)

    # Create simple linear separation
    y = torch.zeros(num_samples, dtype=torch.long)
    for i in range(num_samples):
        if X[i, 0] + X[i, 1] > 0:
            y[i] = 0
        elif X[i, 2] + X[i, 3] > 0:
            y[i] = 1
        else:
            y[i] = 2

    return X, y


def main():
    """Main example function."""
    print("=== Metabolic Neural Ecosystem (MNE) Example ===")
    print("Creating synthetic dataset...")

    # Create synthetic data
    X_train, y_train = create_synthetic_data(800, 20, 3)
    X_test, y_test = create_synthetic_data(200, 20, 3)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Configure MNE
    config = MNEConfig(
        num_neurons=50,
        input_dim=20,
        output_dim=3,
        activation_function="tanh",
        energy_influx=10.0,
        resource_high_threshold=8.0,
        resource_low_threshold=2.0,
        enable_neurogenesis=True,
        enable_apoptosis=True,
        enable_homeostasis=True,
        enable_energy_constraint=True,
    )

    # Create MNE model
    print("\nCreating MNE model...")
    model = MNE(config)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("\nStarting training...")
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output, loss, metrics = model.train_step(data, target, optimizer)

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Print progress
            if batch_idx % 10 == 0:
                print(
                    f"  Batch {batch_idx}: Loss = {loss.item():.4f}, "
                    f"Energy = {metrics['total_energy']:.2f}, "
                    f"Efficiency = {metrics['efficiency']:.4f}"
                )

        # Calculate epoch statistics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%"
        )

        # Print MNE-specific metrics
        mne_metrics = model.get_metrics()
        print(
            f"  Neurons: {mne_metrics['num_neurons']}, "
            f"Active: {mne_metrics['num_active']}, "
            f"Neurogenesis: {mne_metrics['neurogenesis_count']}, "
            f"Apoptosis: {mne_metrics['apoptosis_count']}"
        )

    # Evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model.forward(data)
            _, predicted = torch.max(output, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Print final MNE state
    print("\n=== Final MNE State ===")
    final_metrics = model.get_metrics()
    for key, value in final_metrics.items():
        print(f"{key}: {value}")

    # Show energy dynamics
    print("\n=== Energy Dynamics ===")
    energy_stats = model.energy_manager.get_energy_statistics()
    print(f"Total Energy Consumed: {energy_stats['total_energy']:.2f}")
    print(f"Average Efficiency: {energy_stats['avg_efficiency']:.4f}")
    print(f"Energy Constraint Active: {energy_stats['is_constrained']}")

    # Show topology changes
    print("\n=== Topology Changes ===")
    topology_state = model.topology.get_topology_statistics()
    print(f"Initial Neurons: {config.num_neurons}")
    print(f"Final Neurons: {topology_state['num_neurons']}")
    print(f"Neurogenesis Events: {topology_state['neurogenesis_count']}")
    print(f"Apoptosis Events: {topology_state['apoptosis_count']}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
