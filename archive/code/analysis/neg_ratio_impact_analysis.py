"""
分析改变 neg_ratio 对模型训练的影响
"""

print("=" * 70)
print("改变 neg_ratio 的影响分析")
print("=" * 70)

# 场景对比
scenarios = [
    {"name": "当前设置", "neg_ratio": 5, "train": 5, "test": 5},
    {"name": "方案1: 统一提高", "neg_ratio": 20, "train": 20, "test": 20},
    {"name": "方案2: 训练测试分离", "neg_ratio": 20, "train": 20, "test": 200},
    {"name": "方案3: 大幅提高", "neg_ratio": 50, "train": 50, "test": 50},
]

for i, scenario in enumerate(scenarios, 1):
    print(f"\n{'='*70}")
    print(f"方案 {i}: {scenario['name']}")
    print(f"{'='*70}")

    train_ratio = scenario['train']
    test_ratio = scenario['test']

    # 计算正样本比例
    train_pos_ratio = 1 / (1 + train_ratio)
    test_pos_ratio = 1 / (1 + test_ratio)

    print(f"训练集设置：neg_ratio = {train_ratio}")
    print(f"  - 正样本比例: {train_pos_ratio:.2%}")
    print(f"  - 每个 batch: 1正 + {train_ratio}负 = {1+train_ratio} 个样本")

    print(f"\n测试集设置：neg_ratio = {test_ratio}")
    print(f"  - 正样本比例: {test_pos_ratio:.2%}")
    print(f"  - Random baseline AUPRC: {test_pos_ratio:.6f}")

    # 影响分析
    print(f"\n影响分析：")

    if train_ratio == 5:
        print(f"  ✓ 训练快（负样本少）")
        print(f"  ✓ 模型容易学习（任务简单）")
        print(f"  ✗ 不符合真实分布（1230:1）")
        print(f"  ✗ AUPRC 虚高（baseline={test_pos_ratio:.3f}）")
    elif train_ratio == 20:
        print(f"  ✓ 相对平衡的训练开销")
        print(f"  ✓ 更接近真实任务难度")
        print(f"  ~ 训练时间增加 ~{train_ratio/5:.1f}x")
        if test_ratio == 200:
            print(f"  ✓ 测试更接近真实场景")
            print(f"  ⚠️  训练测试分布不匹配（distribution shift）")
        else:
            print(f"  ✓ 训练测试分布一致")
    elif train_ratio == 50:
        print(f"  ✓ 非常接近真实任务难度")
        print(f"  ~ 训练时间增加 ~{train_ratio/5:.1f}x")
        print(f"  ~ 模型可能收敛更慢")
        print(f"  ✓ 更真实的性能评估")

print("\n" + "=" * 70)
print("关键权衡")
print("=" * 70)

print("""
1. **计算开销 vs 真实性**
   - neg_ratio 越高，训练越慢（负样本越多）
   - 但评估越真实（更接近1230:1的真实场景）

2. **类别不平衡问题**
   当前代码没有使用 pos_weight，所以：
   - neg_ratio=5: 负样本"压倒"正样本的程度低
   - neg_ratio=200: 负样本"压倒"正样本的程度高
   - 模型可能学会"全预测负类"来最小化损失

3. **训练测试分布不匹配（Distribution Shift）**
   - 如果训练用20:1，测试用200:1
   - 模型在训练时学到的正负样本分布和测试时不同
   - 可能导致性能下降

推荐解决方案：
""")

print("\n方案A: 保守策略（最小改动）")
print("-" * 70)
print("  训练: neg_ratio = 10  (温和提升)")
print("  测试: neg_ratio = 50  (部分接近真实)")
print("  优点: 改动小，风险低")
print("  缺点: 仍未完全反映真实场景")

print("\n方案B: 折中策略（推荐）⭐")
print("-" * 70)
print("  训练: neg_ratio = 20  (适中难度)")
print("  测试: neg_ratio = 100 (较真实)")
print("  同时添加 pos_weight 到损失函数（重要！）")
print("  优点: 平衡计算开销和真实性")
print("  缺点: 需要修改损失函数")

print("\n方案C: 激进策略（最真实）")
print("-" * 70)
print("  训练: neg_ratio = 50")
print("  测试: neg_ratio = 200")
print("  必须添加 pos_weight 和其他平衡技术")
print("  优点: 最接近真实场景")
print("  缺点: 训练慢，可能需要调参")

print("\n方案D: 仅修改测试（风险！）")
print("-" * 70)
print("  训练: neg_ratio = 5  (不变)")
print("  测试: neg_ratio = 200 (接近真实)")
print("  优点: 不影响训练，直接看真实性能")
print("  缺点: 严重的 distribution shift，性能可能极差")
print("  ⚠️  会暴露当前模型在真实场景下的糟糕表现")
