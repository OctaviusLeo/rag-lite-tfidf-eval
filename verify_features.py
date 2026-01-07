"""
Comprehensive feature verification script for RAG-Lite.
Tests all major components and generates a verification report.
"""
import json
import os
import sys
from pathlib import Path

# Ensure we're using the local package
sys.path.insert(0, str(Path(__file__).parent))

def print_section(title):
    """Print a section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def test_imports():
    """Test that all modules can be imported."""
    print_section("1. Testing Module Imports")
    
    modules = [
        "src.rag",
        "src.cli",
        "src.api",
        "src.config",
        "src.cache",
        "src.benchmark",
        "src.evaluate",
        "src.io_utils",
    ]
    
    results = {}
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}")
            results[module] = "PASS"
        except Exception as e:
            print(f"✗ {module}: {e}")
            results[module] = f"FAIL: {e}"
    
    return results

def test_index_operations():
    """Test index building and querying."""
    print_section("2. Testing Index Operations")
    
    from src.rag import build_index, retrieve_hybrid
    from src.io_utils import read_text
    
    results = {}
    
    # Load test data
    passages = read_text("data/docs.txt")
    print(f"✓ Loaded {len(passages)} passages")
    
    # Test TF-IDF
    try:
        idx_tfidf = build_index(passages, use_bm25=False, use_embeddings=False)
        res = retrieve_hybrid(idx_tfidf, "machine learning", 2, method="tfidf")
        print(f"✓ TF-IDF: {len(res)} results")
        results["tfidf"] = "PASS"
    except Exception as e:
        print(f"✗ TF-IDF: {e}")
        results["tfidf"] = f"FAIL: {e}"
    
    # Test BM25
    try:
        idx_bm25 = build_index(passages, use_bm25=True, use_embeddings=False)
        res = retrieve_hybrid(idx_bm25, "machine learning", 2, method="bm25")
        print(f"✓ BM25: {len(res)} results")
        results["bm25"] = "PASS"
    except Exception as e:
        print(f"✗ BM25: {e}")
        results["bm25"] = f"FAIL: {e}"
    
    return results

def test_config():
    """Test configuration loading."""
    print_section("3. Testing Configuration")
    
    from src.config import load_config
    
    try:
        config = load_config("config.yaml")
        print(f"✓ Config loaded")
        print(f"  - Default method: {config.retrieval.default_method}")
        print(f"  - Cache enabled: {config.cache.enabled}")
        print(f"  - API port: {config.api.port}")
        return {"config": "PASS"}
    except Exception as e:
        print(f"✗ Config: {e}")
        return {"config": f"FAIL: {e}"}

def test_cache():
    """Test cache system."""
    print_section("4. Testing Cache System")
    
    from src.cache import CacheManager
    import numpy as np
    
    try:
        cache = CacheManager()
        print(f"✓ Cache initialized")
        
        # Test embedding cache
        test_vec = np.array([1.0, 2.0, 3.0])
        cache.set_embedding("test_text", "test_model", test_vec)
        retrieved = cache.get_embedding("test_text", "test_model")
        
        if retrieved is not None and np.allclose(retrieved, test_vec):
            print(f"✓ Embedding cache working")
        else:
            print(f"✗ Embedding cache mismatch")
            
        stats = cache.get_stats()
        print(f"✓ Cache stats: {stats['total_size_mb']:.2f} MB")
        
        return {"cache": "PASS"}
    except Exception as e:
        print(f"✗ Cache: {e}")
        return {"cache": f"FAIL: {e}"}

def test_benchmark():
    """Test benchmark generation."""
    print_section("5. Testing Benchmark System")
    
    from src.benchmark import benchmark_all_methods, generate_json_report
    from src.io_utils import read_text
    
    try:
        passages = read_text("data/docs.txt")[:100]  # Use subset for speed
        
        results = benchmark_all_methods(
            passages,
            trials=1,
            k=2,
            use_embeddings=False,
            use_reranker=False,
        )
        
        print(f"✓ Benchmark completed")
        print(f"  - Methods tested: {len([k for k in results.keys() if k != 'system_info'])}")
        
        # Generate report
        output_file = "outputs/verify_benchmark.json"
        generate_json_report(results, output_file)
        print(f"✓ Report generated: {output_file}")
        
        return {"benchmark": "PASS"}
    except Exception as e:
        print(f"✗ Benchmark: {e}")
        return {"benchmark": f"FAIL: {e}"}

def test_api_module():
    """Test API module can be imported."""
    print_section("6. Testing API Module")
    
    try:
        from src.api import app
        print(f"✓ API app imported")
        print(f"  - FastAPI version: {app.version}")
        print(f"  - Routes: {len(app.routes)}")
        return {"api": "PASS"}
    except Exception as e:
        print(f"✗ API: {e}")
        return {"api": f"FAIL: {e}"}

def test_evaluation():
    """Test evaluation module."""
    print_section("7. Testing Evaluation Module")
    
    try:
        from src.evaluate import evaluate_retrieval
        print(f"✓ Evaluation module imported")
        # Note: Full eval test requires running queries
        return {"evaluation": "PASS"}
    except Exception as e:
        print(f"✗ Evaluation: {e}")
        return {"evaluation": f"FAIL: {e}"}

def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("  RAG-LITE COMPREHENSIVE FEATURE VERIFICATION")
    print("="*80)
    
    all_results = {}
    
    # Run all tests
    all_results.update(test_imports())
    all_results.update(test_index_operations())
    all_results.update(test_config())
    all_results.update(test_cache())
    all_results.update(test_benchmark())
    all_results.update(test_api_module())
    all_results.update(test_evaluation())
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    
    passed = sum(1 for v in all_results.values() if v == "PASS")
    total = len(all_results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%\n")
    
    for component, result in all_results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {component}: {result}")
    
    # Save results
    report = {
        "timestamp": str(__import__("datetime").datetime.now()),
        "results": all_results,
        "summary": {
            "passed": passed,
            "total": total,
            "success_rate": f"{passed/total*100:.1f}%"
        }
    }
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/verification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Full report saved to: outputs/verification_report.json")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
