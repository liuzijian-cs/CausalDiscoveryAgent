# src/utils/data_check.py
# 作者: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# 时间: 2025-09-19 10:20
# 描述: 数据检查工具 (Data Preprocessing Utilities)

# 计划: linearity_check

# 连续-连续：只在连续列子集上做线性性与高斯误差检验（沿用你现有的 linearity_check / gaussian_check）。
# 类别-类别：做关联性检验（卡方、Cramér’s V/互信息），并做多重检验校正。
# 类别-连续：检验不同类别下连续变量分布是否有差异（ANOVA 或更稳健的 Kruskal–Wallis），并做多重校正。
