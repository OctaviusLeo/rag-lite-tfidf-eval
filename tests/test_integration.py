"""Simple integration test to verify imports and basic functionality."""
import sys

sys.path.insert(0, "src")

# Test basic imports
print("Testing imports...")
try:
    import rag
    import evaluate
    import benchmark
    import io_utils

    print("✓ All core modules import successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test that key functions exist
print("\nTesting function availability...")
assert hasattr(rag, "build_index")
assert hasattr(rag, "retrieve")
assert hasattr(rag, "retrieve_hybrid")
assert hasattr(rag, "retrieve_grounded")
assert hasattr(evaluate, "calculate_recall_at_k")
assert hasattr(evaluate, "calculate_mrr_at_k")
assert hasattr(benchmark, "Benchmark")
print("✓ All key functions available")

# Test basic functionality
print("\nTesting basic functionality...")
test_passages = ["This is a test passage about Python programming."]

index = rag.build_index(test_passages, use_chunking=False)
assert index is not None
assert "passages" in index
print("✓ Index build works")

results = rag.retrieve(index, "Python", k=1)
assert len(results) > 0
print("✓ Retrieval works")

# Test metrics
recall = evaluate.calculate_recall_at_k([0], [0, 1, 2])
assert recall == 1.0
print("✓ Metrics calculation works")

print("\n" + "=" * 50)
print("All basic integration tests passed! ✅")
print("=" * 50)
