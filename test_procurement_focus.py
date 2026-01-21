#!/usr/bin/env python3
"""
Test script to verify procurement-focused PDF extraction improvements.
Tests both the focus on product data sheets and the listed manufacturer prioritization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PDF_link_extraction import extract_pdf_links
import json

def test_procurement_focus():
    """Test that the system returns procurement-relevant PDFs."""
    
    print("🧪 Testing Procurement-Focused PDF Extraction")
    print("=" * 50)
    
    # Test product with specific manufacturers
    test_product = {
        'product_name': 'Steel Rebar',
        'manufacturers': ['Nucor Corporation', 'Gerdau', 'Commercial Metals Company'],
        'specifications': [
            'Grade 60',
            'ASTM A615',
            '#4 rebar',
            'Deformed bars'
        ]
    }
    
    print(f"🔍 Searching for: {test_product['product_name']}")
    print(f"📋 Listed Manufacturers: {', '.join(test_product['manufacturers'])}")
    print(f"📝 Specifications: {', '.join(test_product['specifications'])}")
    print()
    
    # Extract PDF links
    results = extract_pdf_links(test_product)
    
    if not results or 'results' not in results:
        print("❌ No results returned!")
        return
    
    pdf_results = results['results']
    print(f"📊 Found {len(pdf_results)} verified PDF results")
    print(f"💰 Cost: {results.get('prompt_tokens', 0)} prompt + {results.get('completion_tokens', 0)} completion tokens")
    print()
    
    # Analyze results
    listed_mfg_count = 0
    procurement_focused_count = 0
    
    print("📄 PDF Analysis Results:")
    print("-" * 30)
    
    for i, result in enumerate(pdf_results[:10], 1):  # Show first 10
        heading = result.get('heading', 'No heading')
        summary = result.get('pdf_summary', 'No summary')
        confidence = result.get('confidence_score', 0)
        from_listed = result.get('from_listed_manufacturer', 0)
        justification = result.get('justification', 'No justification')
        
        if from_listed == 1:
            listed_mfg_count += 1
        
        # Check if it's procurement-focused (contains specs, model numbers, purchasing info)
        procurement_keywords = ['specification', 'astm', 'grade', 'data sheet', 'technical', 'model', 'part number', 'dimensions', 'performance']
        if any(keyword in summary.lower() or keyword in heading.lower() for keyword in procurement_keywords):
            procurement_focused_count += 1
        
        print(f"{i}. {heading}")
        print(f"   📊 Confidence: {confidence:.1%}")
        print(f"   🏭 Listed Mfg: {'✅ Yes' if from_listed == 1 else '❌ No'}")
        print(f"   📝 Summary: {summary[:100]}...")
        print(f"   💡 Why Relevant: {justification[:80]}...")
        print()
    
    # Summary statistics
    print("📈 Summary Statistics:")
    print("-" * 20)
    print(f"📄 Total PDFs found: {len(pdf_results)}")
    print(f"🏭 From listed manufacturers: {listed_mfg_count} ({listed_mfg_count/len(pdf_results)*100:.1f}%)")
    print(f"🎯 Procurement-focused: {procurement_focused_count} ({procurement_focused_count/len(pdf_results)*100:.1f}%)")
    
    # Check sorting (should prioritize listed manufacturers first, then confidence)
    print(f"\n🔄 Sorting Verification:")
    print("-" * 20)
    
    sorted_correctly = True
    for i in range(len(pdf_results) - 1):
        current = pdf_results[i]
        next_item = pdf_results[i + 1]
        
        current_mfg = current.get('from_listed_manufacturer', 0)
        next_mfg = next_item.get('from_listed_manufacturer', 0)
        current_conf = current.get('confidence_score', 0)
        next_conf = next_item.get('confidence_score', 0)
        
        # Listed manufacturer should come first
        if next_mfg > current_mfg:
            sorted_correctly = False
            break
        # If same manufacturer status, higher confidence should come first
        elif current_mfg == next_mfg and next_conf > current_conf:
            sorted_correctly = False
            break
    
    print(f"✅ Sorting is correct: {sorted_correctly}")
    
    if listed_mfg_count > 0:
        print("✅ SUCCESS: Found PDFs from listed manufacturers")
    else:
        print("⚠️  WARNING: No PDFs from listed manufacturers found")
    
    if procurement_focused_count >= len(pdf_results) * 0.8:  # 80% should be procurement-focused
        print("✅ SUCCESS: Most PDFs are procurement-focused")
    else:
        print("⚠️  WARNING: Low percentage of procurement-focused PDFs")

if __name__ == "__main__":
    test_procurement_focus() 