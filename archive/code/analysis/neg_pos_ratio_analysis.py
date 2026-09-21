"""
分析 DeFi 曝露网络数据中的真实负正边比例
"""

import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


def load_network_data(path: str) -> Dict:
    """加载网络数据"""
    with open(path, "r") as f:
        payload = json.load(f)
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        return payload


def analyze_neg_pos_ratio(
    network_data: Dict,
) -> Tuple[Dict[str, List], Dict[str, float]]:
    """
    分析所有快照的真实负正边比例

    Returns:
        ratios_by_date: 每个日期的详细统计
        summary: 汇总统计
    """
    ratios = []
    densities = []

    for date, edges_data in network_data.items():
        num_nodes = len(edges_data["nodes"])
        num_edges = len(edges_data["edges"])

        # 有向图的最大边数
        max_edges = num_nodes * (num_nodes - 1)

        # 真实的负边数
        num_neg = max_edges - num_edges

        # 真实的负正比
        neg_pos_ratio = num_neg / max(1, num_edges)

        # 密度
        density = num_edges / max(1, max_edges)

        ratios.append(
            {
                "date": date,
                "num_nodes": num_nodes,
                "num_pos": num_edges,
                "num_neg": num_neg,
                "neg_pos_ratio": neg_pos_ratio,
                "density": density,
            }
        )

        densities.append(density)

    # 汇总统计
    ratios_array = np.array([r["neg_pos_ratio"] for r in ratios])
    densities_array = np.array(densities)

    summary = {
        "num_snapshots": len(ratios),
        "neg_pos_ratio_mean": float(np.mean(ratios_array)),
        "neg_pos_ratio_std": float(np.std(ratios_array)),
        "neg_pos_ratio_min": float(np.min(ratios_array)),
        "neg_pos_ratio_max": float(np.max(ratios_array)),
        "neg_pos_ratio_median": float(np.median(ratios_array)),
        "density_mean": float(np.mean(densities_array)),
        "density_std": float(np.std(densities_array)),
    }

    return ratios, summary


def compare_with_config(summary: Dict, config_neg_ratio: int = 5):
    """
    对比真实比例与配置的负采样比例
    """
    print("\n" + "=" * 60)
    print("真实负正边比例分析")
    print("=" * 60)

    print(f"\n数据集统计 ({summary['num_snapshots']} 个快照):")
    print(f"  负正比均值:     {summary['neg_pos_ratio_mean']:.2f}")
    print(f"  负正比中位数:   {summary['neg_pos_ratio_median']:.2f}")
    print(
        f"  负正比范围:     [{summary['neg_pos_ratio_min']:.2f}, {summary['neg_pos_ratio_max']:.2f}]"
    )
    print(f"  负正比标准差:   {summary['neg_pos_ratio_std']:.2f}")

    print(f"\n图密度统计:")
    print(f"  密度均值:       {summary['density_mean']:.6f}")
    print(f"  密度标准差:     {summary['density_std']:.6f}")

    print(f"\n与当前配置对比:")
    print(f"  当前负采样比例: {config_neg_ratio}:1")
    print(f"  真实负正比:     {summary['neg_pos_ratio_median']:.1f}:1")
    print(
        f"  差异倍数:       {summary['neg_pos_ratio_median'] / config_neg_ratio:.1f}x"
    )

    if summary["neg_pos_ratio_median"] / config_neg_ratio > 10:
        print(
            f"\n⚠️  严重偏差: 真实负样本是当前配置的 {summary['neg_pos_ratio_median'] / config_neg_ratio:.0f} 倍"
        )
        print(f"   这会导致 AUPRC 显著虚高！")
    elif summary["neg_pos_ratio_median"] / config_neg_ratio > 5:
        print(
            f"\n⚠️  中等偏差: 真实负样本是当前配置的 {summary['neg_pos_ratio_median'] / config_neg_ratio:.1f} 倍"
        )
        print(f"   建议: 调整负采样比例或使用 AUPRC 校正")
    else:
        print(f"\n✓ 负采样比例较为合理")


def recommend_sampling_strategy(summary: Dict) -> Dict[str, any]:
    """
    根据数据特性推荐负采样策略
    """
    print("\n" + "=" * 60)
    print("推荐的负采样策略")
    print("=" * 60)

    real_ratio = summary["neg_pos_ratio_median"]

    # 策略1: 渐进式采样
    recommendations = {}

    print("\n策略 1: 渐进式负采样 (推荐)")
    ratios = [5, 10, 20, 50]
    for r in ratios:
        print(f"  - 训练/验证: {r}:1, 测试: {min(100, real_ratio):.0f}:1 (接近真实)")
    recommendations["progressive"] = {
        "train_neg_ratio": 10,
        "test_neg_ratio": min(100, int(real_ratio)),
        "description": "训练使用适中比例平衡计算，测试使用真实比例评估",
    }

    # 策略2: 度偏置采样
    print("\n策略 2: 度偏置负采样")
    print(f"  - 按节点度数概率采样负边: P(neg) ∝ deg(u) + deg(v)")
    print(f"  - 难负样本: 优先采样高度节点对之间的缺失边")
    recommendations["degree_biased"] = {
        "method": "weighted_sampling",
        "weight_func": "deg_sum",
        "description": "采样更接近真实图结构的难负样本",
    }

    # 策略3: 多策略混合
    print("\n策略 3: 多策略混合")
    print(f"  - 70% uniform random + 30% degree-biased")
    print(f"  - 同时报告两种策略的结果")
    recommendations["mixed"] = {
        "uniform_ratio": 0.7,
        "biased_ratio": 0.3,
        "description": "平衡多样性和难度",
    }

    # 策略4: 校正 AUPRC
    print("\n策略 4: AUPRC 校正 (适用于实验已运行)")
    print(f"  - 使用公式: AUPRC_true ≈ AUPRC_pred * (neg_pos_ratio / config_ratio)")
    print(f"  - 校正系数: {real_ratio / 5:.2f}")
    recommendations["correction"] = {
        "correction_factor": real_ratio / 5,
        "description": "对已报告的 AUPRC 进行校正",
    }

    return recommendations


if __name__ == "__main__":
    # 分析数据
    data_path = "data/historical-network_week_2020-03-30.json"

    print(f"加载数据: {data_path}")
    network_data = load_network_data(data_path)

    ratios, summary = analyze_neg_pos_ratio(network_data)

    # 显示分析结果
    compare_with_config(summary, config_neg_ratio=5)

    # 推荐策略
    recommendations = recommend_sampling_strategy(summary)

    # 保存结果
    output_file = Path("analysis/neg_pos_ratio_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "summary": summary,
                "recommendations": recommendations,
                "detailed_ratios": ratios,
            },
            f,
            indent=2,
        )

    print(f"\n详细结果已保存至: {output_file}")
