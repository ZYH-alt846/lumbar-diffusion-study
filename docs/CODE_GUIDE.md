# 代码说明

我将共用的模型和数值计算整理在`src/lumbar_diffusion/`，训练配置与结果分别放在
`configs/`和`data/aggregate/`。下面按数据流说明各模块。

## 输入与几何

`networks.ImageEncoder2d`共享两视图的编码器，每个视图输出16通道特征。
`geometry.project_grid`将三维网格坐标投影到二维像素中心，`lift_features`通过
`grid_sample`采样，并附加两路有效性掩膜。最终条件体为34×32³。

数组顺序为`D=z, H=y, W=x`，投影矩阵作用于`(x,y,z,1)`。
潜网格与128³标签网格的对应为`4*i+1.5`。图像和投影矩阵必须来自同一裁剪。
完整输入形状见[数据契约](../configs/data_contract.json)。

## 标签AE

`autoencoder.SubvoxelAE`将六类标签编码为8通道空间潜码，再解码为六类logits。
`decode_logits`给背景通道加固定4.45偏置；`decode_labels`在相同logits上取argmax。
`standardize`与`unstandardize`使用逐通道训练统计。

## 回归与条件扩散

- `networks.Small3DUNet`：两级下采样、三线性上采样和跳接，可选时间嵌入。
- `networks.DeterministicRegressor`：早期A使用的base32潜回归结构。
- `conditioning.DeterministicConditional`：CR，二维编码器加base64回归U-Net。
- `conditioning.ConditionalPrior`：复制先验的编码部分，通过零初始化卷积注入条件。
- `conditioning.ConditionalDiffusion`：二维编码器、几何提升和条件去噪器的组合。

CD-F冻结先验主干并训练条件分支；CD-S使用相同条件拓扑，从零训练全部参数。
参数量分别为CD-F总4,328,048/可训练1,743,464，CD-S全训练4,328,048，
CR全训练2,611,864；这些数值不包括公共AE。

下面的代码只构造网络；正式使用CD-F时需要先载入训练好的先验权重。

```python
from lumbar_diffusion.networks import Small3DUNet
from lumbar_diffusion.conditioning import ConditionalDiffusion

prior = Small3DUNet(8, 8, base=64, with_time=True)
model = ConditionalDiffusion(prior, freeze_backbone=True)
```

## 训练损失和采样

`training.diffusion_objective`使用前向加噪与v-MSE。
`training.deterministic_objective`将预测潜码送入冻结AE，以CE加前景软Dice训练。
各阶段的预算、学习率和选点间隔记录在配置文件中。

`diffusion.py`包括噪声表、v目标和DDIM/DDPM采样。DDPM采用后验方差，最后一步
直接返回预测干净码。条件特征在采样循环前计算一次，循环内只调用去噪器。
初始噪声与每步反向噪声分别使用固定随机流。

`bridge.BrownianBridge`保留粗到细实验中的桥过程，预测目标为`tau=x_t-x0`。
采样从粗码`y`出发，首个非平凡跳步使用随机入口，后续内部跳步使用确定性更新。

## 指标

`metrics.evaluate_labels`逐椎体计算Dice和HD95。预测漏检时Dice记0、HD95记缺失，
另报漏检数。HD95使用两方向表面距离P95中的较大者。

汇总时先对每个样本的五个椎体平均，再对同患者采样结果和患者平均。
配准分析使用GT，未加入模型推理。

## 检查与数值来源

`scripts/smoke.py`用合成张量走通前向、梯度、参数更新与短表采样。
`tests/`检查坐标、零注入、冻结梯度、v转换、桥恢复、解码与漏检处理。
与原实验模块使用相同随机权重和合成输入时，U-Net、AE、几何提升、条件网络和
短表DDPM输出逐元素相同。原文件与函数的映射保存在`code_provenance.json`。
