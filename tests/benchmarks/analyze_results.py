"""
GAAP Benchmark Results Analyzer
================================

Automatically analyzes benchmark results and generates comprehensive reports.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from compare_official import compare_results, generate_report, OFFICIAL_SCORES


def parse_benchmark_output(output: str) -> Dict[str, Any]:
    """Parse benchmark script output (Python dict format)"""
    try:
        # Try to parse as JSON-like Python dict
        result = eval(output)
        return result
    except Exception as e:
        print(f"Error parsing output: {e}")
        return {}


def analyze_single_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single benchmark result"""
    
    dataset = result.get("dataset", "unknown")
    samples = result.get("samples", 0)
    
    direct = result.get("direct", {})
    gaap = result.get("gaap", {})
    
    analysis = {
        "dataset": dataset,
        "samples": samples,
        "accuracy": {
            "direct": direct.get("accuracy", 0) * 100,
            "gaap": gaap.get("accuracy", 0) * 100,
            "improvement_absolute": (gaap.get("accuracy", 0) - direct.get("accuracy", 0)) * 100,
            "improvement_relative": ((gaap.get("accuracy", 0) / direct.get("accuracy", 1)) - 1) * 100 if direct.get("accuracy", 0) > 0 else 0,
        },
        "latency": {
            "direct_avg": direct.get("latency_ms", {}).get("avg", 0),
            "gaap_avg": gaap.get("latency_ms", {}).get("avg", 0),
            "overhead": gaap.get("latency_ms", {}).get("avg", 0) - direct.get("latency_ms", {}).get("avg", 0),
            "overhead_ratio": gaap.get("latency_ms", {}).get("avg", 1) / direct.get("latency_ms", {}).get("avg", 1) if direct.get("latency_ms", {}).get("avg", 0) > 0 else 0,
        },
        "cost": {
            "direct_total": direct.get("total_cost_usd", 0),
            "gaap_total": gaap.get("total_cost_usd", 0),
            "direct_per_question": direct.get("total_cost_usd", 0) / max(samples, 1),
            "gaap_per_question": gaap.get("total_cost_usd", 0) / max(samples, 1),
        },
        "tokens": {
            "direct_total": direct.get("total_tokens", 0),
            "gaap_total": gaap.get("total_tokens", 0),
            "direct_per_question": direct.get("total_tokens", 0) / max(samples, 1),
            "gaap_per_question": gaap.get("total_tokens", 0) / max(samples, 1),
        },
    }
    
    return analysis


def generate_summary(analyses: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive summary report"""
    
    lines = []
    lines.append("=" * 80)
    lines.append("GAAP BENCHMARK RESULTS - COMPREHENSIVE ANALYSIS")
    lines.append("=" * 80)
    lines.append("")
    
    total_samples = sum(a["samples"] for a in analyses)
    avg_direct_acc = sum(a["accuracy"]["direct"] for a in analyses) / len(analyses)
    avg_gaap_acc = sum(a["accuracy"]["gaap"] for a in analyses) / len(analyses)
    avg_improvement = sum(a["accuracy"]["improvement_relative"] for a in analyses) / len(analyses)
    
    lines.append("📊 OVERALL SUMMARY")
    lines.append(f"Total Datasets Tested: {len(analyses)}")
    lines.append(f"Total Questions: {total_samples}")
    lines.append(f"Average Direct Accuracy: {avg_direct_acc:.1f}%")
    lines.append(f"Average GAAP Accuracy: {avg_gaap_acc:.1f}%")
    lines.append(f"Average Improvement: {avg_improvement:.1f}%")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    
    for analysis in analyses:
        dataset = analysis["dataset"].upper()
        samples = analysis["samples"]
        acc = analysis["accuracy"]
        lat = analysis["latency"]
        cost = analysis["cost"]
        tokens = analysis["tokens"]
        
        lines.append(f"📈 {dataset} ({samples} samples)")
        lines.append("")
        
        lines.append("ACCURACY:")
        lines.append(f"  Direct:      {acc['direct']:.2f}%")
        lines.append(f"  GAAP:        {acc['gaap']:.2f}%")
        lines.append(f"  ➕ Absolute: +{acc['improvement_absolute']:.2f}%")
        lines.append(f"  ✖️ Relative:  {acc['improvement_relative']:.1f}x improvement")
        lines.append("")
        
        lines.append("LATENCY:")
        lines.append(f"  Direct:      {lat['direct_avg']:.0f}ms")
        lines.append(f"  GAAP:        {lat['gaap_avg']:.0f}ms")
        lines.append(f"  Overhead:    +{lat['overhead']:.0f}ms ({lat['overhead_ratio']:.1f}x)")
        lines.append("")
        
        lines.append("COST:")
        lines.append(f"  Direct Total:   ${cost['direct_total']:.4f}")
        lines.append(f"  GAAP Total:     ${cost['gaap_total']:.4f}")
        lines.append(f"  Per Question:   ${cost['gaap_per_question']:.4f}")
        lines.append("")
        
        lines.append("TOKENS:")
        lines.append(f"  Direct Total:   {tokens['direct_total']}")
        lines.append(f"  GAAP Total:     {tokens['gaap_total']}")
        lines.append(f"  Per Question:   {tokens['gaap_per_question']:.0f}")
        lines.append("")
        
        lines.append("-" * 80)
        lines.append("")
    
    # Value proposition
    lines.append("💡 VALUE PROPOSITION")
    lines.append("")
    avg_overhead = sum(a["latency"]["overhead_ratio"] for a in analyses) / len(analyses)
    avg_cost_per_q = sum(a["cost"]["gaap_per_question"] for a in analyses) / len(analyses)
    
    lines.append(f"GAAP achieves {avg_improvement:.1f}x better accuracy")
    lines.append(f"with {avg_overhead:.1f}x latency overhead")
    lines.append(f"at ${avg_cost_per_q:.4f} per question")
    lines.append("")
    lines.append("✅ Best for: Tasks requiring high accuracy (reasoning, analysis)")
    lines.append("⚠️  Trade-off: Slower but significantly more accurate")
    lines.append("")
    
    return "\n".join(lines)


def main():
    """Main analyzer entry point"""
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <benchmark_output_file_or_string>")
        print("\nExample:")
        print("  python analyze_results.py benchmark_results.json")
        print("  python analyze_results.py \"{'dataset': 'mmlu', ...}\"")
        return
    
    input_arg = sys.argv[1]
    
    # Try to load from file first
    if Path(input_arg).exists():
        with open(input_arg, 'r') as f:
            content = f.read()
    else:
        content = input_arg
    
    # Parse the result(s)
    try:
        data = json.loads(content) if content.startswith('[') or content.startswith('{') else eval(content)
    except:
        print(f"Error: Could not parse input as JSON or Python dict")
        return
    
    # Handle single result or list of results
    if isinstance(data, dict):
        results = [data]
    else:
        results = data
    
    # Analyze each result
    analyses = [analyze_single_result(r) for r in results]
    
    # Generate summary
    summary = generate_summary(analyses)
    print(summary)
    
    # Save to file
    output_file = "benchmark_analysis.txt"
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"📁 Full analysis saved to: {output_file}")
    
    # Compare with official benchmarks if possible
    print("\n" + "=" * 80)
    print("OFFICIAL BENCHMARK COMPARISON")
    print("=" * 80)
    print("")
    
    comparisons = []
    for result in results:
        dataset = result.get("dataset", "")
        # Assume gemini-2.5-flash for now
        comparison = compare_results(
            dataset=dataset,
            base_model="gemini-2.5-flash",
            direct_accuracy=result.get("direct", {}).get("accuracy", 0),
            gaap_accuracy=result.get("gaap", {}).get("accuracy", 0)
        )
        comparisons.append(comparison)
    
    official_report = generate_report(comparisons)
    print(official_report)
    
    with open("official_comparison.txt", 'w') as f:
        f.write(official_report)
    
    print("📁 Official comparison saved to: official_comparison.txt")


if __name__ == "__main__":
    main()
