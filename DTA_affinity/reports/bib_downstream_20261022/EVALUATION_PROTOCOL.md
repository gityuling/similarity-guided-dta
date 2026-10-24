# 两项下游任务：评分前冻结协议

日期：2026-10-22。所有队列规则、主指标及对照在首次模型评分前确定。复用五个 KIBA identity-disjoint 模型，不重新训练、不依据外部终点调参。该分析是回顾性外部评价，不是预注册或盲法前瞻试验。

## 数据与范围

A：PubChem AID 1963317，2025 年论文 DOI 10.1038/s41467-025-59658-6。128 个预先富集的激酶抑制剂、ABL1 D233–S500 kinase-domain N-HiBiT 构建体、87.5 μM ATP、四次 Kinase-Glo Plus 功能测量。映射到 P00519，以模型的 canonical ABL1 全长序列输入；实验构建体不是输入序列本身，不能声称构建体完全相同，也不能把抑制读出解释成直接 Kd。异构/变构作用以及报告系统干扰是限制。只评价这个候选池内的相对优选，不代表无偏开放库筛选。

源汇总标签有 47 Active / 63 Inactive / 18 Inconclusive。逐重复编码为 1/2/3；实际汇总将 2 和 3 都计为非阴性，至少三个非阴性重复时汇总 Active。描述中的“active repeats”不能误读为只计编码 2。保留官方标签作为主分析，并预设 strict sensitivity：至少三个编码 2 才算 Active，四个编码 1 为 Inactive，其余 Unknown。未确定结果不当作实测阴性。

B：LigUnity 的 JACS/Merck 实验系列，仓库 commit 8ae35edba5481aa13de8a5bac6b03c1c1860ef3c。使用 fep_labels.json 中 act，逐条核对原始 SDF 的 r_exp_dg 字段；act = -r_exp_dg/1.364，最大容差 1e-4 kcal/mol。全标准化 parent InChIKey（含立体信息）匹配，多个构象的实验值必须一致。这里评价实验来源标签，不使用 FEP 预测值。原始测定类型可能为 Ki/IC50 等，不把变换值统一称为实测 pKd；只在各系列内部比较。未重建每篇原始药化论文的全部实验元数据，基准来源与构建体差异需保留为限制。

主来源：LigUnity DOI 10.1016/j.patter.2025.101371；JACS DOI 10.1021/ja512751q；Merck DOI 10.1021/acs.jcim.0c00900；整理版 DOI 10.1038/s42004-023-01019-9。

## 纳排与泄漏控制

标准化 fragment parent、去电荷；所有 Davis/KIBA parent connectivity 重叠排除，包括不同盐型和立体异构。此前 Reinecke、NanoBRET、BRAF 候选及 G2CP annotation-review 化合物也排除；读取这些文件时只读结构列。系列内重复 parent 全部排除，不挑选有利测量；A/B 共用 parent 排除。报告 scaffold 重叠但不据此排除主分析。B 必须精确匹配 KIBA 的 accession/全长序列，剩余至少 10 个分子、act 动态范围至少 1；A 至少 30 个候选、至少 5 个有标签阳性和 5 个阴性。以上只使用数据覆盖和标签质控，不读取模型预测。

A 纳入 80 个候选：26 Active、42 Inactive、12 Unknown。B 纳入 6 个系列 148 个分子：CDK2 16、CDK8 30、MET 24、JNK1 18、SYK 44、TYK2 16。p38 因重叠排除后仅 2 个，整体不纳入。其他 9 个非支持靶点不纳入。源 16 系列及全部排除记录保留。

这不证明预训练编码器从未接触过这些结构/蛋白，也不证明同药化系列不在训练集；JNK1 全部 18 个分子的 Murcko scaffold 与 KIBA 重叠。因此不称 novel-target / novel-series zero-shot。LigUnity 自己的排除和预训练不同，不直接与其论文数值作公平模型优劣比较。

## 推理与对照

同一候选池、同一输入、同一外部盲标签 query。deep、neighbor_prior、fixed_blend、similarity_gate、equal_blend；额外 pure_similarity 按同靶点训练分子的最大 Morgan Tanimoto 排名。五折内应用原固定权重/门控后做等权均值。记录每折预测及其排序稳定性。预测分数越高代表预测效力越强；不将 KIBA 分数差转换成倍数。每折先复现八条历史预测，误差门槛 0.02（prior 2e-6）。共享 GPU lock 保证排他执行。

## A 指标

主指标：Top-10 官方 Active 命中数/10（确认命中率下界，非把 Unknown 定为阴性），对照随机期望 10×26/80。报告 Unknown 个数、命中率上界 (Active+Unknown)/10、确认阳性召回率、相对随机确认命中富集倍数。主要对比 gate−deep，fixed 和 neighbor 为关键额外对照。Top-5、Top-20、全预算曲线为次要结果，不能根据其结果更换主预算。严格重复敏感性沿用相同候选排序和预算。

AP/ROC-AUC 仅在已知标签子集计算并明确这是条件性评价。随机 Top-K 的单侧超几何概率仅描述相对随机的个案结果，不能代替 gate 相对其他模型的统计检验。只有一个靶点，不给出跨靶点泛化置信区间。

## B 指标

主指标：六个系列内 Spearman 的等权宏平均，主要对比 gate−deep。次要：|Δact|≥1 的配对方向准确率（预测相等算 0.5）、预测 Top-3 中实际最好四分位分子的比例及召回率。真值阈值使用 numpy quantile(0.75)，阈值同分全部计为优质，报告每组优质数；预测同分用 source_id 稳定打破。随机期望分别为 0、0.5、系列优质比例、3/n。

以系列为独立单位做 10,000 次配对 bootstrap（seed 20261022），报告宏平均及差值 95% percentile 区间；六系列的主差值另给穷举符号翻转双侧 p 值。小样本区间仅描述探索性不确定性。不得以全部分子对当独立样本增加显著性。

全部系列和方法都报告，门控相对固定融合无提升时明确陈述。结果只支持实验标签上的回顾优选/排序，不支持新分子活性确认、选择性改善或因果机制。

## 排除的数据路线

LIT-PCBA MAPK1（AID 995）是 HEK293 ERK 通路磷酸化读出；MTORC1（AID 493208）是细胞 phospho-S6 读出。均不作为直接靶点 DTA 筛选的主验证。AID 624303 是小规模 RIN1-ABL 跟进反筛，全部阴性，不适合作为排序验证。保留下载原始描述以便复核。
