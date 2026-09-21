"""
分析改变 neg_ratio 对训练时间的实际影响
"""

print("=" * 70)
print("训练时间分析")
print("=" * 70)

print("""
训练循环的计算步骤：

for sample in pairs:  # 283 个时间快照
    # 步骤1：图编码（GNN） - 对整个图的所有节点
    h = model.encode(graph, features, device)
    时间复杂度：O(|V| + |E|) × L（L是GNN层数）

    # 步骤2：节点预测头
    node_pred = model.node_head(h)
    时间复杂度：O(|V|)

    # 步骤3：链接预测头 - 对采样的边对
    logits, w_pred = model.link_scorer(h, src, dst)
    时间复杂度：O(num_pairs) = O(num_pos × (1 + neg_ratio))

    # 步骤4：计算损失和反向传播
    loss.backward()
""")

print("\n" + "=" * 70)
print("neg_ratio 从 5 变到 20 的影响")
print("=" * 70)

# 假设数据规模
num_snapshots = 283
avg_nodes = 5676
avg_pos_edges = 30423
gnn_layers = 3

print(f"\n数据规模（基于你的实验结果）：")
print(f"  快照数量: {num_snapshots}")
print(f"  平均节点数: {avg_nodes}")
print(f"  平均正边数: {avg_pos_edges}")
print(f"  GNN层数: {gnn_layers}（假设）")

print(f"\n步骤1: 图编码（每个快照一次）")
print(f"  neg_ratio=5:  {num_snapshots} 次编码")
print(f"  neg_ratio=20: {num_snapshots} 次编码")
print(f"  时间变化: 0x（不变）")

print(f"\n步骤2: 节点预测头（每个快照一次）")
print(f"  neg_ratio=5:  {num_snapshots} × {avg_nodes} = {num_snapshots * avg_nodes:,} 次预测")
print(f"  neg_ratio=20: {num_snapshots} × {avg_nodes} = {num_snapshots * avg_nodes:,} 次预测")
print(f"  时间变化: 0x（不变）")

print(f"\n步骤3: 链接预测头（对每个样本对）")
num_pairs_5 = avg_pos_edges * 6  # 1正+5负
num_pairs_20 = avg_pos_edges * 21  # 1正+20负
print(f"  neg_ratio=5:  {avg_pos_edges} × 6  = {num_pairs_5:,} 个边对/快照")
print(f"  neg_ratio=20: {avg_pos_edges} × 21 = {num_pairs_20:,} 个边对/快照")
print(f"  时间变化: {num_pairs_20 / num_pairs_5:.1f}x（增加 {num_pairs_20 / num_pairs_5:.1f} 倍）")

print("\n" + "=" * 70)
print("实际训练时间估计")
print("=" * 70)

print("""
假设各步骤耗时占比（这是关键！）：

情况A：图编码是瓶颈（大图、深GNN）
  - 步骤1（图编码）: 70%
  - 步骤2（节点预测）: 10%
  - 步骤3（链接预测）: 15%
  - 步骤4（反向传播）: 5%

  总时间增加 = 70% × 0 + 10% × 0 + 15% × 3.5 + 5% × 2
             ≈ 0.53 + 0.1
             ≈ 1.63x（增加约60%）

情况B：链接预测是瓶颈（小图、大量边对）
  - 步骤1（图编码）: 20%
  - 步骤2（节点预测）: 10%
  - 步骤3（链接预测）: 60%
  - 步骤4（反向传播）: 10%

  总时间增加 = 20% × 0 + 10% × 0 + 60% × 3.5 + 10% × 2
             ≈ 2.1 + 0.2
             ≈ 2.3x（增加约130%）

情况C：均衡（各步骤耗时相当）
  - 步骤1（图编码）: 40%
  - 步骤2（节点预测）: 15%
  - 步骤3（链接预测）: 35%
  - 步骤4（反向传播）: 10%

  总时间增加 = 40% × 0 + 15% × 0 + 35% × 3.5 + 10% × 2
             ≈ 1.225 + 0.2
             ≈ 1.42x（增加约42%）
""")

print("\n" + "=" * 70)
print("结论")
print("=" * 70)

print("""
1. 我之前说的"4倍"是错误的！
   - 样本数增加是 21/6 = 3.5倍，不是4倍
   - 但这只影响链接预测头和反向传播

2. 实际训练时间增加：1.4x - 2.3x
   - 取决于哪个步骤是瓶颈
   - 大多数情况下在 1.5x - 2x 之间

3. 如何验证？
   - 在训练循环中加计时器，看各步骤耗时
   - 或者先用小数据集（一个月的数据）测试

4. 是否值得？
   - 训练时间增加 1.5-2x
   - 但评估更真实（从5:1变到20:1）
   - 如果有GPU，1.5-2x是可以接受的
""")

print("\n" + "=" * 70)
print("建议：先做小规模测试")
print("=" * 70)

print("""
步骤1：用一个月的数据测试
  python run_full_experiment.py --test_mode --neg_ratio 5
  # 记录训练时间 T1

  python run_full_experiment.py --test_mode --neg_ratio 20
  # 记录训练时间 T2

  实际时间倍数 = T2 / T1

步骤2：根据结果决定
  如果 T2/T1 < 2.0: 可以接受，全量数据也用 neg_ratio=20
  如果 T2/T1 > 3.0: 太慢了，考虑只在测试时提高 neg_ratio
""")
