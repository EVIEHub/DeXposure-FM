"""INTERNAL NOTES: summarize the audited outputs without editing manuscript files."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
r=json.loads((P/'comparison.json').read_text()); img=json.loads((P/'figure4_raster_comparison.json').read_text()); matrix=json.loads((P/'figure4_matrix.json').read_text())
event_audit=json.loads((P/'table5_event_universe.json').read_text())
for name,d in event_audit.items():
 r['table5'][name]['new']=d['variants']['common_with_previous_week']['metrics']
 r['table5'][name]['match_one_decimal']=d['variants']['common_with_previous_week']['matches']
paper={1:[-.95,3.04,1.],4:[-1.71,1.42,.83],8:[-.17,2.60,1.],12:[-.55,2.21,.83]}
lines=['# INTERNAL NOTES｜Tables 4–5 / Figures 2–6 核验报告','','对象：20260919_fm_sequential_h1_h4_h8_h12。核验日：2026-09-21。',
'本次核验使用原 run 的 107 个已保存预测文件，无新训练、无云实例。对预测和输入文件校验 SHA256，并逐周核对节点、时间跨度、真实边及 log1p 权重。',
'','## 结果概览','','| 论文对象 | 核验结果 |','|---|---|',
'| Table 4 / Figures 2–3 | 已补算。压力测试误差与论文不匹配；h4、h12 的尾部改进方向也未重现。 |',
'| Table 5 | '+('两次事件的 8 个印刷数值全部匹配（保留一位小数）。' if all(all(v['match_one_decimal']) for v in r['table5'].values()) else '重新核算完成；详见下表，存在印刷值差异。')+' |',
'| Figure 4 | 15×15 热力图 使用原图指定的补充数据文件，225 个单元格的中心 RGB 与论文图一致；这是图像层面的核验，不是原始矩阵逐数值认证。 |',
'| Figure 5 | 已重算 107 个预测/真实网络的风险指标。相关性数值并不相同，详见下表。 |',
'| Figure 6 | 本次 run 的训练数据包含 2022 年事件，不能用于无泄漏的事件预警核验。未将其输出伪装为事前预测。 |',
'','## Table 4 / Figures 2–3','','ΔMAE = 基准 MAE − 模型 MAE，单位为系统损失百分点，正数更好。',
'旧脚本默认每跨度抽 10 个测试周（seed=42+h），每周 3 个冲击，共 30 项，取最差 6 项；其实际基准网络保留原节点集合。论文文字描述全测试期并要求共同节点。两种评估设置均保留，原始运行命令未知。',
'','| h | 来源 | 全体 ΔMAE | 最差20% ΔMAE | 最差20% 胜率 |','|---|---|---:|---:|---:|']
for h in paper:
 v=paper[h]; lines.append(f'| h{h} | 论文 | {v[0]:+.3f} | {v[1]:+.3f} | {100*v[2]:.1f}% |')
 for mode,label in [('legacy_default10','本次 / 旧脚本默认抽样'),('paper_common_nodes_all','本次 / 全测试周共同节点')]:
  d=r['table4'][mode][f'h={h}']; lines.append(f'| h{h} | {label} | {d["delta_mae_all"]:+.3f} | {d["delta_mae_worst"]:+.3f} | {100*d["win_rate_worst"]:.1f}% |')
lines+=['','全测试周计算 32/29/25/21 个起点，共 321 个冲击样本；各跨度单独按基准绝对误差选择最差20%。抽样版和全测试周版都不支持“完整复现 Table 4”的结论。',
'','## Table 5','','| 事件 | 来源 | 事件前 TVL ($B) | 事件后 TVL ($B) | TVL 变化 | 边数变化 |','|---|---|---:|---:|---:|---:|']
for name,d in r['table5'].items():
 for key,label in [('paper','论文'),('new','重算')]:
  v=d[key];lines.append(f'| {name} | {label} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:+.1f}% | {v[3]:+.1f}% |')
 # Event dates and population definition are documented below the table.
lines+=['','Table 5 使用原事件脚本口径：事件前为 anchor 整张图；事件末周仅保留与紧邻前一周共同存在的节点，使用旧版边记录计数。Terra/Luna 为 2022-04-25 → 2022-05-23（共同节点参考 05-16）；FTX 为 2022-10-24 → 2022-11-21（共同节点参考 11-14）。8 个值按论文一位小数全部匹配。直接比较两张完整网络或使用本次训练的去重边数，会得到不同的统计值；详见 table5_event_universe.json 和 table5_data_versions.json。','','## Figure 4','',f'日期 2025-06-30。按原图脚本指定的 historical-network_week_2025-07-01.json 重算。该文件与本次训练主数据 historical-network_week_2020-03-30.json 的同日矩阵不同（行业集合也有差异），所以只能将结果绑定到原图指定数据源。原论文 PDF 内嵌热力图和重算 PDF 采样 225 个格子，RGB 全部一致；两幅嵌入图宽度相差 1 像素，属于渲染差异。原始数值矩阵未找到，因此没有把“颜色一致”表述为每个敞口数值精确一致。',
'','## Figure 5','','论文图中标注的 R² 实际由原脚本计算为 Pearson 相关系数平方。以下沿用该定义对照，并额外报告真正的回归 R²（1−SSE/SST）；后者为负表示数值误差大于直接预测真实值均值。',
'','| 指标 | 论文图 r² | 本次 r² | 本次 MAE | 本次回归 R² |','|---|---:|---:|---:|---:|']
labels={'tvl_hhi':'TVL 集中度','density':'网络密度','mean_sis':'平均系统重要性','spillover_index':'跨行业敞口集中度'}
for k,d in r['figure5'].items():lines.append(f'| {labels[k]} | {d["paper_pearson_r_squared"]:.3f} | {d["new_pearson_r_squared"]:.3f} | {d["mae"]:.6g} | {d["regression_r_squared"]:.3f} |')
lines+=['','因此，相关性较强也不等于风险指标水平校准准确。新旧 Figure 5 的数值不相同，不应只依据某几个更高的相关性声称整幅图复现。',
'','## Figure 6：适用性核验未通过','','本次训练范围覆盖 2022 年之后，论文则要求每个事件训练单独的事件前模型。当前 checkpoint 不能用于验证 2022 年事前预警。',
'本地确有另一批 2026-08-06 的 pre2022 h4 模型，日志记录训练至 2021-10-11、验证至 2022-03-28，但它是另一任务且跨度为 h4。pilot_pre2022/h1 文件仅含 model、feature_schema、horizon，没有绑定事件截止日期的配置。均不足以认证论文要求的两个事件专用 h1 模型；本次未替换模型或启动新训练。',
'','## 证据与限制','','- 保留原脚本的阈值 0.5、log1p 反变换、传播规则和 Top-N 冲击选择（按预测 TVL，三张图使用同一冲击集合）。',
'- 保存预测覆盖未来正边和 5:1 负采样候选对。下游重算也沿用这一候选集合，不能当作完整全节点对预测。',
'- 旧脚本 PageRank 异常会静默退化成零分。本次最终运行显式检查 SciPy，并使 PageRank 错误立即停止；最终运行未使用退化结果。',
'- 原始 Task II 命令、逐样本结果和模型绑定未找回。上述是确定配置下的重新评估，不是确认了原始运行设置的逐位重放。',
'- 此核验不改变论文文件、模型参数和原 run 产物。',
'','## 可复跑命令','','```sh','PYTHONHASHSEED=0 MPLBACKEND=Agg uv run --no-project --with numpy --with matplotlib --with networkx --with ijson --with scipy python '+str(P/'verify_downstream.py'),'```',
'','最终汇总：verification_summary.json。comparison.json 保留初次完整网络统计，Table 5 的最终共同节点口径见 table5_event_universe.json；逐样本压力测试：legacy_default10/exp2_predictive_contagion.json 和 paper_common_nodes_all/exp2_predictive_contagion.json；风险指标：exp1_forward_risk.json；最终日志：validation_final.log。']
lines += ['', '生成事件口径对照运行 check_event_universe.py；图像格子核对运行 check_figure4.py；最后运行 make_report.py。均使用上列相同 uv 依赖。Figure 4 重绘复用 DeXposure_FM_V2/scripts/make_spillover_matrix_figure.py，显式指定该 run 的 2025-07-01 输入和 metadata，输出至此目录。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
# One self-contained quantitative comparison figure.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
fig,axes=plt.subplots(1,2,figsize=(11,4.2));hs=list(paper);x=np.arange(4);w=.25
for ax,key,pi,title in [(axes[0],'delta_mae_all',0,'All cases'),(axes[1],'delta_mae_worst',1,'Worst 20% of baseline errors')]:
 for off,vals,label,c in [(-w,[paper[h][pi] for h in hs],'Paper','#304e65'),(0,[r['table4']['legacy_default10'][f'h={h}'][key] for h in hs],'New: legacy 10-week sample','#dc8053'),(w,[r['table4']['paper_common_nodes_all'][f'h={h}'][key] for h in hs],'New: all weeks, common nodes','#248b83')]:
  bars=ax.bar(x+off,vals,width=w*.92,color=c,label=label)
  ax.bar_label(bars,labels=[f'{v:+.2f}' for v in vals],fontsize=8,padding=(11 if off==0 else 3))
 ax.axhline(0,color='#52616c',lw=1);ax.set_xticks(x,[f'h{h}' for h in hs]);ax.set_title(title);ax.set_ylim(-7.1,4.2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Baseline MAE − model MAE (percentage points)')
fig.suptitle('Table 4 verification: paper vs. new stress-test results',fontsize=12,y=1.01)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.055),fontsize=9)
fig.tight_layout(); (P/'figures').mkdir(exist_ok=True)
for ext in ['pdf','png']:fig.savefig(P/f'figures/table4_comparison.{ext}',bbox_inches='tight')
plt.close(fig)
print('Report written:',P/'REPORT.md')
# Correctly labelled figure 5, distinct from the archived plotter's legacy R2 label.
risk=json.loads((P/'exp1_forward_risk.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(10,9));colors=['#2a5674','#398b9a','#7b983d','#bb7650']
for ax,(key,title) in zip(axs.flat,[('tvl_hhi','TVL concentration'),('density','Network density'),('mean_sis','Mean systemic importance'),('spillover_index','Cross-sector concentration')]):
 allvals=[]
 for (h,d),c in zip(risk['horizons'].items(),colors):
  actual=[m[key] for m in d['actual_metrics']];pred=[m[key] for m in d['predicted_metrics']]
  ax.scatter(actual,pred,s=20,alpha=.65,color=c,label=h);allvals.extend(actual+pred)
 lo,hi=min(allvals),max(allvals);margin=(hi-lo)*.08;lo-=margin;hi+=margin
 ax.plot([lo,hi],[lo,hi],ls='--',color='#777',lw=1);ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Actual',ylabel='Predicted',title=title)
 d=r['figure5'][key];ax.text(.04,.96,f'Pearson r² = {d["new_pearson_r_squared"]:.3f}\nRegression R² = {d["regression_r_squared"]:.3f}',transform=ax.transAxes,va='top',fontsize=9,bbox={'facecolor':'white','edgecolor':'#ddd','alpha':.85})
 ax.grid(alpha=.15);ax.legend(loc='lower right',fontsize=8,frameon=False)
fig.suptitle('Figure 5 verification · all 107 saved test predictions',fontsize=13)
fig.tight_layout()
for ext in ['pdf','png']:fig.savefig(P/f'figures/figure5_verified.{ext}',bbox_inches='tight')
plt.close(fig)
summary={'run':r['run'],'new_cloud_instances':0,'predictions_checked':107,'stress_cases_full':321,
 'table4_figures2_3':'not_matched_under_both_evaluated_protocols',
 'table5':{'status':'all_8_printed_values_matched','definition':'legacy event evaluation population, end date restricted to nodes present the previous week','events':event_audit},
 'figure4':{'status':'225_heatmap_cell_colors_match_with_original_figure_input','run_primary_data_matrix_differs':not matrix['equal'],'raster':img},
 'figure5':{'status':'not_numerically_identical','metrics':r['figure5']},
 'figure6':'current_2025_run_ineligible_for_pre2022_event_validation',
 'limits':r['protocol']}
(P/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
