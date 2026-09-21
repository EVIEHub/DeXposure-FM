"""
说明为什么 neg_ratio=5 下的 AUPRC 0.74 是虚高的
"""

# 在不同负正比下，Random Baseline 的期望 AUPRC

# 1. neg_ratio = 5:1 的情况
#    正样本比例 = 1 / (1 + 5) = 1/6 ≈ 0.167
#    Random baseline AUPRC ≈ 0.167

# 2. 真实负正比 = 1230:1 的情况
#    正样本比例 = 1 / (1 + 1230) = 1/1231 ≈ 0.0008
#    Random baseline AUPRC ≈ 0.0008

# 你的模型在 5:1 下得到 AUPRC = 0.74
# 相对于 baseline 0.167 的提升：0.74 / 0.167 ≈ 4.4x

# 但如果在真实 1230:1 下评估，期望的 AUPRC 应该大约是：
# 0.74 * (0.0008 / 0.167) ≈ 0.0035

print("=" * 60)
print("AUPRC Baseline 对比")
print("=" * 60)

# Scenario 1: 你当前的评估（neg_ratio=5）
neg_ratio_train = 5
pos_ratio_train = 1 / (1 + neg_ratio_train)
print(f"\n场景1：训练和测试都用 neg_ratio={neg_ratio_train}")
print(f"  正样本比例: {pos_ratio_train:.4f}")
print(f"  Random baseline AUPRC: {pos_ratio_train:.4f}")
print(f"  你的模型 AUPRC: 0.7400")
print(f"  相对提升: {0.74 / pos_ratio_train:.2f}x")

# Scenario 2: 真实世界评估
neg_ratio_real = 1230
pos_ratio_real = 1 / (1 + neg_ratio_real)
print(f"\n场景2：真实世界评估（neg_ratio={neg_ratio_real}）")
print(f"  正样本比例: {pos_ratio_real:.6f}")
print(f"  Random baseline AUPRC: {pos_ratio_real:.6f}")
print(f"  如果保持相同的相对性能：")
expected_auprc_real = 0.74 * (pos_ratio_real / pos_ratio_train)
print(f"    期望 AUPRC ≈ {expected_auprc_real:.6f}")
print(f"    (0.74 × {pos_ratio_real / pos_ratio_train:.4f})")

# 校正公式
print("\n" + "=" * 60)
print("AUPRC 校正公式（粗略估计）")
print("=" * 60)
correction_factor = neg_ratio_real / neg_ratio_train
print(f"校正系数 = 真实负正比 / 训练负正比")
print(f"         = {neg_ratio_real} / {neg_ratio_train}")
print(f"         = {correction_factor:.1f}")
print(f"\n校正后的 AUPRC ≈ 当前AUPRC / 校正系数")
print(f"                = 0.74 / {correction_factor:.1f}")
print(f"                ≈ {0.74 / correction_factor:.6f}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print("""
1. 在 5:1 采样下，AUPRC 0.74 看起来不错
2. 但这是在"简化任务"下的表现（正样本占16.7%）
3. 在真实场景下（正样本占0.08%），预期性能会大幅下降
4. 真实 AUPRC 可能只有 0.0035 左右

建议：
- 使用更高的 neg_ratio（如20-50）进行训练
- 在接近真实比例（如200:1）下进行测试评估
- 或者报告 Recall@K 等不受类别不平衡影响的指标
""")
