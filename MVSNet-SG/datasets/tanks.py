from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from PIL import Image
from datasets.data_io import *


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        # self.depthpath = "/mnt/B/MVS_GT/dtu_training/data1/mvs_training/dtu/"
        self.listfile = listfile
        self.mode = mode
        assert self.mode == "test"

        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        # max_h, max_w
        # self.subset = kwargs['subset']
        # assert self.subset in ['training', 'intermediate', 'advanced']

        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        # depth_max = float(lines[11].split()[3])
        # depth_interval = (depth_max - depth_min) / self.ndepths
        # depth_interval = depth_interval * self.interval_scale
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        # assert np_img.shape[:2] == (1080, 1920) or np_img.shape[:2] == (1080, 2048)

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)


    def resize_input(self, img, intrinsic):
        h, w = img.shape[:2]
        new_h = 1056
        new_w = w if w == 2048 else 1920
        # new_w, new_h (2048, 1056) or (1920, 1056)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # print("after resize: {}".format(img.shape))

        intrinsic[0] *= (new_w/w)
        intrinsic[1] *= (new_h/h)

        return img, intrinsic


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        proj_matrices_new = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images_post/{:0>8}.jpg'.format(scan, vid)) # colmap preprocessed
            if not os.path.exists(img_filename):
                img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, '{}/cams_{}/{:0>8}_cam.txt'.format(scan, scan.lower(), vid)) # short range
            if not os.path.exists(proj_mat_filename):
                proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            # imgs.append(self.read_img(img_filename))
            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            img, intrinsics = self.resize_input(img, intrinsics)
            imgs.append(img)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            # to save new_cam in eval.py
            proj_mat_new = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat_new[0, :4, :4] = extrinsics
            proj_mat_new[1, :3, :3] = intrinsics
            proj_matrices_new.append(proj_mat_new)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)

                sparse_filename = os.path.join(self.datapath, "{}/sparse_colmap/sparse_depth/{:0>8}_sparse.pfm".format(scan, vid))
                sparse_depth = self.read_depth(sparse_filename)
                # print("sparse_depth: {}".format(sparse_depth.shape))
                resize_w = img.shape[1] // 4
                assert sparse_depth.shape[:2] == (264, resize_w), "sparse_depth: {}".format(sparse_depth.shape)


        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        proj_matrices_new = np.stack(proj_matrices_new)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "proj_matrices_new": proj_matrices_new,
                "depth_values": depth_values,
                "sparse_depth": sparse_depth,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("/mnt/B/qiyh/training_preproc", '../lists/tanks/training.txt', 'test', 5,
                         128)
    item = dataset[554]
    for key, value in item.items():
        print(key, type(value))

    print("depth_values: {}".format(item["depth_values"]))
    print(item['filename'])