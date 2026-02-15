"""
Official Benchmark Comparison
==============================

Compares GAAP results against official benchmark scores for various models.

Official scores from:
- MMLU: https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu
- ARC: https://paperswithcode.com/sota/common-sense-reasoning-on-arc-challenge
- HumanEval: https://paperswithcode.com/sota/code-generation-on-humaneval
"""

from typing import Dict, Any, List
import json

# Official benchmark scores (% accuracy)
# Updated as of February 2026
OFFICIAL_SCORES = {
    "mmlu": {
        "gpt-4": 86.4,
        "gpt-4-turbo": 87.1,
        "claude-3-opus": 86.8,
        "claude-3-sonnet": 79.0,
        "gemini-1.5-pro": 85.9,
        "gemini-1.5-flash": 78.9,
        "gemini-2.0-flash": 82.5,
        "llama-3-70b": 79.5,
        "llama-3-8b": 66.0,
        "mixtral-8x7b": 70.6,
        "gpt-3.5-turbo": 70.0,
    },
    "arc_challenge": {
        "gpt-4": 96.3,
        "gpt-4-turbo": 96.7,
        "claude-3-opus": 95.4,
        "claude-3-sonnet": 92.1,
        "gemini-1.5-pro": 94.2,
        "gemini-1.5-flash": 89.8,
        "llama-3-70b": 93.0,
        "llama-3-8b": 83.4,
        "mixtral-8x7b": 87.3,
        "gpt-3.5-turbo": 85.2,
    },
    "arc_easy": {
        "gpt-4": 96.8,
        "claude-3-opus": 96.0,
        "gemini-1.5-pro": 95.6,
        "llama-3-70b": 94.8,
        "gpt-3.5-turbo": 92.5,
    },
    "openbookqa": {
        "gpt-4": 92.8,
        "claude-3-opus": 91.2,
        "gemini-1.5-pro": 89.6,
        "llama-3-70b": 88.4,
        "gpt-3.5-turbo": 84.0,
    },
}


def get_base_model_name(model: str) -> str:
    """Extract base model name from GAAP model string"""
    if "gemini" in model.lower():
        if "2.5-flash" in model:
            return "gemini-2.5-flash"
        elif "2.0-flash" in model:
            return "gemini-2.0-flash"
        elif "1.5-flash" in model:
            return "gemini-1.5-flash"
        elif "1.5-pro" in model:
            return "gemini-1.5-pro"
    elif "gpt" in model.lower():
        if "gpt-4" in model:
            return "gpt-4"
        elif "gpt-3.5" in model:
            return "gpt-3.5-turbo"
    elif "claude" in model.lower():
        if "opus" in model:
            return "claude-3-opus"
        elif "sonnet" in model:
            return "claude-3-sonnet"
    elif "llama" in model.lower():
        if "70b" in model:
            return "llama-3-70b"
        elif "8b" in model:
            return "llama-3-8b"
    elif "mixtral" in model.lower():
        return "mixtral-8x7b"
    
    return model


def compare_results(
    dataset: str,
    base_model: str,
    direct_accuracy: float,
    gaap_accuracy: float
) -> Dict[str, Any]:
    """
    Compare GAAP results with official benchmarks
    
    Args:
        dataset: Dataset name (mmlu, arc_challenge, etc)
        base_model: Base model used (gemini-2.5-flash, gpt-4, etc)
        direct_accuracy: Raw model accuracy (0-1)
        gaap_accuracy: GAAP-enhanced accuracy (0-1)
    
    Returns:
        Comparison dict with rankings and improvements
    """
    
    dataset_key = dataset.lower().replace("-", "_")
    
    if dataset_key not in OFFICIAL_SCORES:
        return {
            "error": f"No official scores available for dataset: {dataset}",
            "available_datasets": list(OFFICIAL_SCORES.keys())
        }
    
    official = OFFICIAL_SCORES[dataset_key]
    
    # Convert to percentage
    direct_pct = direct_accuracy * 100
    gaap_pct = gaap_accuracy * 100
    
    # Find base model official score
    base_official = official.get(base_model, None)
    
    # Sort all models by score
    sorted_models = sorted(official.items(), key=lambda x: x[1], reverse=True)
    
    # Find where GAAP would rank
    gaap_rank = 1
    direct_rank = 1
    
    for model, score in sorted_models:
        if score > gaap_pct:
            gaap_rank += 1
        if score > direct_pct:
            direct_rank += 1
    
    # Calculate improvements
    result = {
        "dataset": dataset,
        "base_model": base_model,
        "scores": {
            "direct": round(direct_pct, 2),
            "gaap": round(gaap_pct, 2),
            "base_official": base_official,
        },
        "improvement": {
            "absolute": round(gaap_pct - direct_pct, 2),
            "relative": round((gaap_pct / direct_pct - 1) * 100, 2) if direct_pct > 0 else 0,
        },
        "rankings": {
            "total_models": len(official),
            "direct_rank": f"{direct_rank}/{len(official)}",
            "gaap_rank": f"{gaap_rank}/{len(official)}",
            "rank_improvement": direct_rank - gaap_rank,
        },
        "comparison_table": sorted_models,
    }
    
    # Add context
    if base_official:
        diff_from_official = gaap_pct - base_official
        result["vs_official_base"] = {
            "difference": round(diff_from_official, 2),
            "note": "GAAP vs official benchmark of base model"
        }
    
    return result


def generate_report(results: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive comparison report"""
    
    lines = []
    lines.append("=" * 80)
    lines.append("GAAP vs Official Benchmarks - Comprehensive Report")
    lines.append("=" * 80)
    lines.append("")
    
    for result in results:
        if "error" in result:
            lines.append(f"⚠️  {result['error']}")
            continue
        
        dataset = result["dataset"]
        base_model = result["base_model"]
        scores = result["scores"]
        improvement = result["improvement"]
        rankings = result["rankings"]
        
        lines.append(f"📊 Dataset: {dataset.upper()}")
        lines.append(f"🤖 Base Model: {base_model}")
        lines.append("")
        
        lines.append("Accuracy Scores:")
        lines.append(f"  Direct (raw):        {scores['direct']}%")
        lines.append(f"  GAAP (enhanced):     {scores['gaap']}%")
        if scores['base_official']:
            lines.append(f"  Official Benchmark:  {scores['base_official']}%")
        lines.append("")
        
        lines.append("Improvement:")
        lines.append(f"  Absolute:  +{improvement['absolute']}%")
        lines.append(f"  Relative:  +{improvement['relative']}%")
        lines.append("")
        
        lines.append("Rankings:")
        lines.append(f"  Direct rank:  {rankings['direct_rank']}")
        lines.append(f"  GAAP rank:    {rankings['gaap_rank']}")
        if rankings['rank_improvement'] > 0:
            lines.append(f"  Improved by:  {rankings['rank_improvement']} positions ⬆️")
        elif rankings['rank_improvement'] < 0:
            lines.append(f"  Dropped by:   {abs(rankings['rank_improvement'])} positions ⬇️")
        else:
            lines.append(f"  No change in rank")
        lines.append("")
        
        # Show top models for context
        lines.append("Top 5 Models (Official Benchmarks):")
        for i, (model, score) in enumerate(result["comparison_table"][:5], 1):
            marker = "👑" if i == 1 else f"{i}."
            gaap_marker = " ← GAAP" if score < scores['gaap'] else ""
            lines.append(f"  {marker} {model}: {score}%{gaap_marker}")
        
        lines.append("")
        lines.append("-" * 80)
        lines.append("")
    
    return "\n".join(lines)


def main():
    """Example usage"""
    
    # Example: MMLU results
    results = []
    
    # Result 1: MMLU with Gemini 2.5 Flash
    mmlu_result = compare_results(
        dataset="mmlu",
        base_model="gemini-2.5-flash",
        direct_accuracy=0.20,  # 20% direct
        gaap_accuracy=0.90     # 90% with GAAP
    )
    results.append(mmlu_result)
    
    # Result 2: ARC-Challenge
    arc_result = compare_results(
        dataset="arc_challenge",
        base_model="gemini-2.5-flash",
        direct_accuracy=0.30,
        gaap_accuracy=0.90
    )
    results.append(arc_result)
    
    # Generate and print report
    report = generate_report(results)
    print(report)
    
    # Also save as JSON
    with open("benchmark_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("📁 Detailed results saved to: benchmark_comparison.json")


if __name__ == "__main__":
    main()
