# Sparse-Prior-Guided-MVS

Demo code for "[Sparse Prior Guided Deep Multi-view Stereo](https://www.sciencedirect.com/science/article/abs/pii/S0097849322001157)" (Computers and Graphics 2022)

### Abstract

Recently, the learning-based multi-view stereo (MVS) methods have achieved notable progress. Despite this, the cost volumes of those methods are not robust enough for depth inference with the existence of noise, outlier and occlusion. In this work, we aim to **reduce the matching ambiguities and increase the discrimination of the cost volume**. For this purpose, we first propose a sparse prior guidance strategy which **incorporates the sparse points of structure-from-motion (SfM) into cost  volume construction**. Further, we introduce a geometry-aware regularization module to enhance the representative power of cost volume regularization, which could adaptively fit the local geometric shapes. The two modules are straightforward yet effective, resulting in robust and accurate depth estimation. The detailed experiments are conducted on two public datasets (DTU and Tanks & Temples). The experimental results show that with the two components, the representative methods (e.g. MVSNet and CasMVSNet) make a significant gain in reconstruction quality. 

### Framework
![framework](/figures/Sparse_Guided_MVS_Framework_EN.png)
<center> Overview of our sparse prior guided multi-view stereo framework. We propose 1) Cost Volume Modulation that utilizes the sparse prior to modulate the cost distribution along the depth dimension, leading to unimodal distribution peaked at correct depth hypothesis, and 2) Geometry-aware Regularization which enhances the representative power of cost aggregation by additionally learned offsets to better fit the local geometric shape.
</center>

### Data preprocessing
![sparse_preproc](/figures/sparse_preproc.png)
<center> The sparse points produced by structure-from-motion is the sparse representation of the scene. The sparse points are first projected into image plane to generate the sparse depth, then the sparse priors are used to modulate the cost volume by  using the Gaussian function. 
</center> 

This step is implemented by [colmap_sparse_recon](https://github.com/XYZ-qiyh/colmap_sparse_recon).


<!--
### How to use
0. Dependencies
   + ```   pip install -r requirements.txt   ```
1. Data Preprocessing
   + apply [colmap_sparse_recon](https://github.com/XYZ-qiyh/colmap-sparse-recon) to recover sparse points and convert the sparse points to sparse depth map.

2. Depth inference using sparse points guidance
   + modify the `sparse_filename` in `dtu_yao_eval.py`
   + Enable `--use_guided` in `eval.sh`


### Qualitative Comparison

![depth_results](/figures/depth_results.png)
-->

### Acknowledgement
   + This code is based on the [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch)/[CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet), thank xy-guo/Gu Xiaodong@Alibaba for their excellent code.

### References
[1] [Guided Stereo Matching](https://github.com/mattpoggi/guided-stereo), Matteo Poggi, Davide Pallotti, Fabio Tosi and Stefano Mattoccia, CVPR 2019.

[2] 3D Deformable Convolutions for MRI classification [[paper](https://arxiv.org/pdf/1911.01898.pdf)] [[code](https://github.com/kondratevakate/3d-deformable-convolutions)] 

