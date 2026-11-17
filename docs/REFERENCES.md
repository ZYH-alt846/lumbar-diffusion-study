# 方法参考与数据引用

这里列的是思想来源，不声称复现官方主结果，不包含作者仓库或论文PDF副本。

- DDPM: Ho et al. (2020). https://arxiv.org/abs/2006.11239
- DDIM: Song et al. (2021). https://arxiv.org/abs/2010.02502
- v parameterization / progressive distillation: Salimans and Ho (2022). https://arxiv.org/abs/2202.00512
- Latent Diffusion: Rombach et al. (CVPR2022). https://arxiv.org/abs/2112.10752
- ControlNet: Zhang et al. (ICCV2023). https://arxiv.org/abs/2302.05543
- PC²: Melas-Kyriazi et al. (CVPR2023). https://arxiv.org/abs/2302.10668
- BBDM: Li et al. (CVPR2023). https://openaccess.thecvf.com/content/CVPR2023/html/Li_BBDM_Image-to-Image_Translation_With_Brownian_Bridge_Diffusion_Models_CVPR_2023_paper.html
- Swin-X2S: https://arxiv.org/abs/2501.05961
- AXON: https://arxiv.org/abs/2603.26509
- PPCNet Dataset: https://huggingface.co/datasets/ppcnet-dataset/PPCNet (DOI 10.57967/hf/9944)
- VerSe upstream: https://github.com/anjany/verse
- CTSpine1K upstream: https://github.com/MIRACLE-Center/CTSpine1K

ControlNet基于大规模图像先验；PC²处理RGB表面点云投影，DRR含穿透叠加。
AXON的粗到细CT生成与本项目的标签潜码精修并非同一配置。
这些论文成功不构成当前数据、表示、预算下必然超确定性参考的证据。
