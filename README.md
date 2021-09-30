# Sparse-Guided-MVS

Demo code for "Deep Multi-view Stereo with Sparse Points Guidance and Adaptive Cost Aggregation"


## How to use
1. Data Preprocessing
   + Use [colmap-sparse-recon](https://github.com/XYZ-qiyh/colmap-sparse-recon) to recover sparse points and convert the sparse points to sparse depthmap

2. Depth inference using sparse points guidance

   + modify the `sparse_filename` in `dtu_yao_eval.py`
   + Enable `--use_guided` in `eval.sh`
