# 数据来源与结果文件

本项目使用[PPCNet](https://huggingface.co/datasets/ppcnet-dataset/PPCNet)，
DOI为10.57967/hf/9944，其上游包括VerSe和CTSpine1K。
数据需通过官方页面申请，使用时遵守PPCNet和上游数据的许可。

## 输入与标签

每例使用正位、侧位DRR及对应的3×4投影矩阵。标签中的L1–L5重映射为1–5，
背景为0。模型输入256²，标签输出128³，完整接口见
[data_contract.json](../configs/data_contract.json)。

## 仓库保存的结果

- `core_results.csv`：最终方法比较。
- `selection_curves.csv`：三种条件模型的45个选择集评价点。
- `diagnostics.json`：病例配对的汇总、AE敏感性、条件替换及配准结果。
- `prior_sampling.json`：独立先验的两种采样设置比较。

原始影像、标签、病例明细、潜码与权重不在仓库中分发。
代码中的合成张量用于检查接口，与研究结果无关。
[数据许可](https://huggingface.co/datasets/ppcnet-dataset/PPCNet/blob/main/LICENSE)
对数据再分发有单独要求；代码的许可不替代数据许可。
