#!/usr/bin/env python3
"""
Tests for Ukrainian Stress Service

Comprehensive testing of stress lookup functionality.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.stress_service.stress_service import UkrainianStressService


def test_basic_lookup():
    """Test basic word lookup"""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Word Lookup")
    print("=" * 80)
    
    with UkrainianStressService() as service:
        # Simple word with one stress variant
        word = "кіт"
        result = service.lookup(word)
        
        print(f"\n🔍 Lookup: '{word}'")
        if result:
            print(f"  ✓ Found {len(result)} form(s)")
            for i, form in enumerate(result, 1):
                print(f"  [{i}] Stress: {form.get('stress_variants')}, POS: {form.get('pos')}")
        else:
            print("  ✗ Not found")
        
        assert result is not None, f"Word '{word}' should be found"
        print("\n✅ Test passed")


def test_heteronyms():
    """Test words with multiple stress patterns (heteronyms)"""
    print("\n" + "=" * 80)
    print("TEST 2: Heteronyms (Multiple Stress Patterns)")
    print("=" * 80)
    
    with UkrainianStressService() as service:
        # Known heteronyms in Ukrainian
        heteronyms = ["замок", "атлас", "мука"]
        
        for word in heteronyms:
            result = service.lookup(word)
            variants = service.get_stress_variants(word)
            is_het = service.is_heteronym(word)
            
            print(f"\n🔍 Word: '{word}'")
            print(f"  Heteronym: {is_het}")
            if variants:
                print(f"  Stress variants: {', '.join(variants)}")
            
            if result:
                print(f"  Forms found: {len(result)}")
                for i, form in enumerate(result[:2], 1):  # Show first 2
                    print(f"    [{i}] Stress: {form.get('stress_variants')}, "
                          f"POS: {form.get('pos')}, "
                          f"Feats: {list(form.get('feats', {}).keys())}")
        
        print("\n✅ Test passed")


def test_morphology():
    """Test morphological features"""
    print("\n" + "=" * 80)
    print("TEST 3: Morphological Features")
    print("=" * 80)
    
    with UkrainianStressService() as service:
        word = "книга"
        result = service.lookup(word)
        
        print(f"\n🔍 Word: '{word}'")
        if result:
            for i, form in enumerate(result[:3], 1):  # Show first 3
                print(f"\n  Form {i}:")
                print(f"    Stress: {form.get('stress_variants')}")
                print(f"    POS: {form.get('pos')}")
                
                feats = form.get('feats', {})
                if feats:
                    print(f"    Features:")
                    for key, values in feats.items():
                        print(f"      {key}: {values}")
        
        print("\n✅ Test passed")


def test_pos_tags():
    """Test POS tag extraction"""
    print("\n" + "=" * 80)
    print("TEST 4: Part-of-Speech Tags")
    print("=" * 80)
    
    with UkrainianStressService() as service:
        test_words = ["іти", "швидкий", "добре", "і"]
        
        for word in test_words:
            pos_tags = service.get_pos_tags(word)
            print(f"\n  '{word}': {pos_tags}")
        
        print("\n✅ Test passed")


def test_normalization():
    """Test word normalization"""
    print("\n" + "=" * 80)
    print("TEST 5: Word Normalization")
    print("=" * 80)
    
    with UkrainianStressService() as service:
        # Test different apostrophe types
        variants = ["кров'ю", "кров'ю", "КРОВ'Ю"]
        
        print("\n  Testing apostrophe normalization:")
        for word in variants:
            result = service.lookup(word)
            print(f"    '{word}' -> Found: {result is not None}")
        
        # All variants should find the same word
        results = [service.lookup(w) for w in variants]
        assert all(r is not None for r in results), "All variants should be found"
        
        print("\n✅ Test passed")


def test_not_found():
    """Test handling of non-existent words"""
    print("\n" + "=" * 80)
    print("TEST 6: Non-existent Words")
    print("=" * 80)
    
    with UkrainianStressService() as service:
        fake_words = ["zzzzzzz", "qwerty123", "фівфівфів"]
        
        for word in fake_words:
            result = service.lookup(word)
            print(f"  '{word}': {'Found' if result else 'Not found'} ✓")
            assert result is None, f"Fake word '{word}' should not be found"
        
        print("\n✅ Test passed")


def test_stress_variants():
    """Test stress variant display"""
    print("\n" + "=" * 80)
    print("TEST 7: Stress Variant Display")
    print("=" * 80)
    
    with UkrainianStressService() as service:
        test_words = ["замок", "мука", "атлас", "хліб", "вода"]
        
        print("\n  Word stress variants:")
        for word in test_words:
            variants = service.get_stress_variants(word)
            if variants:
                print(f"    {word:10} → {', '.join(variants)}")
            else:
                print(f"    {word:10} → Not found")
        
        print("\n✅ Test passed")


def test_performance():
    """Test lookup performance"""
    print("\n" + "=" * 80)
    print("TEST 8: Performance Benchmark")
    print("=" * 80)
    
    import time
    
    with UkrainianStressService() as service:
        test_words = ["кіт", "собака", "дім", "вода", "хліб", "сонце", "місяць"]
        iterations = 1000
        
        print(f"\n  Running {iterations} iterations × {len(test_words)} words = {iterations * len(test_words):,} lookups")
        
        start = time.time()
        for _ in range(iterations):
            for word in test_words:
                service.lookup(word)
        elapsed = time.time() - start
        
        total_queries = iterations * len(test_words)
        qps = total_queries / elapsed
        latency = (elapsed / total_queries) * 1000
        
        print(f"\n  Results:")
        print(f"    Total time:    {elapsed:.3f}s")
        print(f"    Queries/sec:   {qps:,.0f}")
        print(f"    Avg latency:   {latency:.4f}ms")
        
        print("\n✅ Test passed")


def test_database_stats():
    """Test database statistics"""
    print("\n" + "=" * 80)
    print("TEST 9: Database Statistics")
    print("=" * 80)
    
    with UkrainianStressService() as service:
        stats = service.get_stats()
        
        print(f"\n  Database Statistics:")
        print(f"    Entries:    {stats['entries']:,}")
        print(f"    Page size:  {stats['page_size']:,} bytes")
        print(f"    Tree depth: {stats['depth']}")
        print(f"    Size:       {stats['size_bytes'] / (1024*1024):.2f} MB")
        
        assert stats['entries'] > 0, "Database should have entries"
        
        print("\n✅ Test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("UKRAINIAN STRESS SERVICE - TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_basic_lookup,
        test_heteronyms,
        test_morphology,
        test_pos_tags,
        test_normalization,
        test_not_found,
        test_stress_variants,
        test_performance,
        test_database_stats,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  Total tests: {len(tests)}")
    print(f"  ✅ Passed:   {passed}")
    print(f"  ❌ Failed:   {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
