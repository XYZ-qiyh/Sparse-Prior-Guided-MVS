# Sparse-Prior-Guided-MVS

Demo code for "Sparse Prior Guided Deep Multi-view Stereo" (In submission)

### Abstract

Recently, the learning-based multi-view stereo (MVS) methods have achieved notable progress. Despite this, the cost volumes of those methods are not robust enough for depth inference with the existence of noise, outlier and occlusion. In this work, we aim to **reduce the matching ambiguities and increase the discrimination of the cost volume**. For this purpose, we first propose a sparse prior guidance strategy which **incorporates the sparse points of structure-from-motion (SfM) into cost  volume construction**. Further, we introduce a geometry-aware regularization module to enhance the representative power of cost volume regularization, which could adaptively fit the local geometric shapes. The two modules are straightforward yet effective, resulting in robust and accurate depth estimation. The detailed experiments are conducted on two public datasets (DTU and Tanks & Temples). The experimental results show that with the two components, the top-performing networks (e.g. MVSNet and CasMVSNet) make a significant gain in reconstruction quality. 

### Framework
![framework](/figures/Sparse_Guided_MVS_Framework_EN.png)
<center> Overview of our sparse prior guided multi-view stereo framework. We propose 1) Cost Volume Modulation that utilizes the sparse prior to modulate the cost distribution along the depth dimension, leading to unimodal distribution peaked at correct depth hypothesis, and 2) Geometry-aware Regularization which enhances the representative power of cost aggregation by additionally learned offsets to better fit the local geometric shape.
</center>

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

### Acknowledgement
   + This code is based on the [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch), thank xy-guo for his excellent code.
