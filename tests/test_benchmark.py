"""
Quick verification test for the comprehensive benchmark.
This tests basic import and instantiation without running full benchmarks.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required modules can be imported."""
    print("[TEST] Testing imports...")
    try:
        from tests.benchmark import (
            ComprehensiveBenchmark,
            BenchmarkResults,
            StandardMLP,
            StandardCNN,
        )

        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_benchmark_instantiation():
    """Test that the benchmark can be instantiated."""
    print("\n[TEST] Testing benchmark instantiation...")
    try:
        from tests.benchmark import ComprehensiveBenchmark

        benchmark = ComprehensiveBenchmark(device="cpu")
        print(f"[OK] Benchmark instantiated successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Instantiation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_baseline_models():
    """Test that baseline models can be created."""
    print("\n[TEST] Testing baseline model creation...")
    try:
        from tests.benchmark import StandardMLP, StandardCNN
        import torch

        # StandardMLP takes hidden_dims (list of ints) and output_dim
        mlp = StandardMLP(input_dim=10, hidden_dims=[20], output_dim=5)
        # StandardCNN takes input_channels (not num_channels), no height/width needed (adaptive pooling)
        cnn = StandardCNN(input_channels=3, num_classes=10)

        # Test forward pass
        x_mlp = torch.randn(4, 10)
        output_mlp = mlp(x_mlp)
        assert output_mlp.shape == (4, 5), (
            f"MLP output shape mismatch: {output_mlp.shape}"
        )

        # CNN uses adaptive pooling so it can handle any input size
        x_cnn = torch.randn(4, 3, 28, 28)
        output_cnn = cnn(x_cnn)
        assert output_cnn.shape == (4, 10), (
            f"CNN output shape mismatch: {output_cnn.shape}"
        )

        print(f"[OK] Baseline models created successfully")
        print(f"  - MLP parameters: {mlp.get_num_parameters():,}")
        print(f"  - CNN parameters: {cnn.get_num_parameters():,}")
        return True
    except Exception as e:
        print(f"[FAIL] Baseline models failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_synthetic_data_creation():
    """Test that synthetic data can be created."""
    print("\n[TEST] Testing synthetic data creation...")
    try:
        from tests.benchmark import ComprehensiveBenchmark

        benchmark = ComprehensiveBenchmark(device="cpu")

        # create_synthetic_data returns (train_loader, val_loader, test_loader)
        train_loader, val_loader, test_loader = benchmark.create_synthetic_data(
            num_samples=500,
            input_dim=10,
            num_classes=5,
        )

        # Check that loaders are not None
        assert train_loader is not None, "train_loader is None"
        assert val_loader is not None, "val_loader is None"
        assert test_loader is not None, "test_loader is None"

        # Check that data can be loaded
        train_inputs, train_targets = next(iter(train_loader))
        assert train_inputs.shape[0] > 0, "No training samples"
        assert train_inputs.shape[1] == 10, (
            f"Input dimension mismatch: {train_inputs.shape[1]}"
        )
        assert train_targets.shape[0] == train_inputs.shape[0], (
            "Targets dimension mismatch"
        )

        print(f"[OK] Synthetic data created successfully")
        print(f"  - Input dimension: 10")
        print(f"  - Number of classes: 5")
        print(f"  - Train batch size: {train_inputs.shape[0]}")
        print(f"  - Train input shape: {train_inputs.shape}")
        print(f"  - Train target shape: {train_targets.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] Synthetic data creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK VERIFICATION TEST")
    print("=" * 70)

    tests = [
        test_imports,
        test_benchmark_instantiation,
        test_baseline_models,
        test_synthetic_data_creation,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n[FAIL] Test {test.__name__} crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] All tests passed! The benchmark is ready to use.")
        return 0
    else:
        print(f"\n[FAILURE] {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
