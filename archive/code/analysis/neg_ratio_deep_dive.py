"""
深入理解 neg_ratio 在链接预测中的作用
"""

print("=" * 80)
print("第一部分：neg_ratio 到底是什么？")
print("=" * 80)

print("""
链接预测任务：给定图 G(t)，预测 G(t+1) 中哪些边会出现

数据构造：
  正样本：G(t+1) 中实际存在的边  (u, v) ∈ E(t+1)
  负样本：G(t+1) 中不存在的边    (u, v) ∉ E(t+1)

neg_ratio = 负样本数量 / 正样本数量

例子：
  正边数量：1000
  neg_ratio=5:  采样 5000 条负边
  neg_ratio=20: 采样 20000 条负边
  neg_ratio=1:  采样 1000 条负边

训练过程：
  model(u, v) → score ∈ [0, 1]
  如果是正边，希望 score 接近 1
  如果是负边，希望 score 接近 0
""")

print("\n" + "=" * 80)
print("第二部分：不同领域的标准做法")
print("=" * 80)

print("""
1. 知识图谱补全（Knowledge Graph Completion）
   例如：OGBL-WikiKG2, FB15k-237

   标准做法：
     - 训练：neg_ratio = 1:1 或固定数量（如50个）
     - 测试：对每个正边，用"所有其他可能的边"作为负样本
     - 指标：MRR (Mean Reciprocal Rank), Hits@K

   为什么1:1？
     - 这些数据集本身就很稀疏（已经只保留了高质量的三元组）
     - 测试用全集排序，不依赖neg_ratio
     - 训练时1:1足够让模型学到pattern

2. 社交网络链接预测（Social Network Link Prediction）
   例如：Facebook, Twitter 好友推荐

   标准做法：
     - 训练：neg_ratio = 1:1 到 10:1
     - 测试：neg_ratio = 相同或略高
     - 指标：AUC, Precision@K

   为什么不高？
     - 社交网络的实际应用中，候选集通常是预筛选的（朋友的朋友）
     - 不是从"所有可能的人"中推荐

3. 推荐系统（Recommendation Systems）
   例如：商品推荐、视频推荐

   标准做法：
     - 训练：neg_ratio = 1:1 到 10:1
     - 测试：neg_ratio = 100:1 甚至更高（或全集）
     - 指标：NDCG@K, Recall@K

   为什么训练低、测试高？
     - 训练时：用适中的ratio让模型收敛
     - 测试时：用高ratio或全集，模拟真实场景
     - 使用sampling + re-ranking策略

4. Open Graph Benchmark (OGB)

   标准做法：
     - OGBL-COLLAB: 训练时正负样本均衡采样，测试时全集排序
     - OGBL-DDI: 药物相互作用，类似做法
     - 指标：Hits@K
""")

print("\n" + "=" * 80)
print("第三部分：你的情况分析")
print("=" * 80)

print("""
你的任务：DeFi 网络边预测
  - 节点：5000+
  - 正边：30000+
  - 可能的边：5000 × 4999 ≈ 2500万
  - 真实负正比：1230:1

当前设置：
  - 训练：neg_ratio = 5:1
  - 测试：neg_ratio = 5:1（和训练相同）
  - 指标：AUPRC = 0.74

问题在哪？
  ✓ neg_ratio=5 本身不是问题
  ✗ 训练和测试都用5:1，但真实应用场景是1230:1
  ✗ AUPRC是依赖类别比例的指标，5:1下的0.74不能反映1230:1下的性能
""")

print("\n" + "=" * 80)
print("第四部分：为什么不是所有模型都虚高？")
print("=" * 80)

print("""
关键区别：评估指标的选择

指标1：Ranking-based（不受neg_ratio影响）
  - MRR, Hits@K, NDCG@K
  - 对每条正边，排序所有候选边
  - 只看正边的排名，不关心负边的绝对数量

  例子：
    正边 (A, B)
    neg_ratio=5:   排序 6 个候选边，A-B排第1 → Hits@1=1
    neg_ratio=1230: 排序 1231 个候选边，A-B排第1 → Hits@1=1
    结果相同！

指标2：Classification-based（受neg_ratio影响）
  - AUC, AUPRC, F1
  - 把链接预测当二分类问题
  - 需要设定阈值或计算曲线

  例子：
    neg_ratio=5:   正样本占16.7%, random baseline AUPRC=0.167
    neg_ratio=1230: 正样本占0.08%, random baseline AUPRC=0.0008
    差了200倍！

大部分顶会论文（KDD, WWW, ICLR）使用 Ranking-based 指标！
  - 因为更符合实际应用（推荐Top-K）
  - 不受neg_ratio影响
  - 所以训练时用1:1或5:1没问题

你使用 AUPRC，所以受影响！
""")

print("\n" + "=" * 80)
print("第五部分：解决方案")
print("=" * 80)

print("""
方案A：改变评估指标（推荐）⭐
  不再用 AUPRC，改用 Recall@K 或 Precision@K

  Recall@100 = (Top-100预测中的正边数) / (总正边数)

  优点：
    - 不受neg_ratio影响
    - 更符合实际应用（推荐Top-K条边）
    - 训练可以继续用5:1

  实现：
    对每个时间步的预测边，按score排序
    取Top-K，计算有多少是真实正边

方案B：训练测试分离 + pos_weight
  训练：neg_ratio=5或10，加pos_weight补偿类别不平衡
  测试：neg_ratio=100或200，接近真实场景

  优点：
    - 仍然可以报告AUPRC（但会降低，这是真实性能）
    - 训练时计算可控

  缺点：
    - 训练测试分布不匹配（distribution shift）
    - AUPRC会显著下降

方案C：全程使用高neg_ratio
  训练和测试都用neg_ratio=50或更高

  优点：
    - 完全匹配真实分布
    - AUPRC真实反映性能

  缺点：
    - 训练非常慢
    - 需要大量调参（学习率、batch size等）

方案D：使用weighted sampling
  不是uniform采样负边，而是按节点度数加权采样
  采样"难负样本"（高度节点间的缺失边）

  优点：
    - 即使低neg_ratio，也能学到更多信息
    - 更接近真实分布的难度
""")

print("\n" + "=" * 80)
print("第六部分：你应该怎么做？")
print("=" * 80)

print("""
我的建议（按优先级）：

1. 【立即做】添加 Recall@K 指标
   修改 evaluate_predictions 函数，加入：
     - Recall@100
     - Recall@500
     - Recall@1000

   这样你可以和其他论文对比，而不受neg_ratio影响

2. 【可选】测试真实性能
   只改测试集的neg_ratio为100或200
   看看AUPRC降低到多少

   如果降得不多（还有0.3+），说明模型确实学到了东西
   如果降到很低（<0.05），说明确实虚高了

3. 【长期】考虑重新训练
   如果步骤2发现性能很差，再考虑：
     - 提高训练neg_ratio到20
     - 添加pos_weight
     - 使用weighted sampling

不要：
  ✗ 不要直接把neg_ratio改到100+重新训练（太冒险）
  ✗ 不要只看AUPRC一个指标
  ✗ 不要假设所有论文都用相同的设置
""")
