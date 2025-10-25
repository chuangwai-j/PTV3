# pointcept/datasets/wind_shear.py
import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
from pointcept.utils.logger import get_root_logger
from scipy.spatial import KDTree # 保留
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transforms.builder import build_transform


@DATASETS.register_module()
class WindShearDataset(Dataset):
    def __init__(self, split='train', data_root="D:/model/wind_datas/csv_labels",
                 transform=None, k_neighbors=16, # 【修改】移除 radius
                 min_points=50,
                 filter_full_paths=None,
                 min_altitude=0.0,
                 max_altitude=1000.0):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.transform = build_transform(transform)
        self.k_neighbors = k_neighbors # 用于 _compute_neighborhood_features
        self.min_points = min_points
        self.filter_full_paths = filter_full_paths if filter_full_paths is not None else []
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude

        if self.filter_full_paths:
            logging.info(f"将过滤{len(self.filter_full_paths)}个低价值样本（完整路径）")

        self.data_list = self._get_data_list()
        logger = get_root_logger()
        logging.info(f"WindShearDataset {split} split: {len(self.data_list)} scenes. "
                     f"Altitude filter: [{self.min_altitude}m, {self.max_altitude}m]. "
                     f"Neighbor k={self.k_neighbors}")

    def _get_data_list(self):
        # ... (此方法未修改，与 相同) ...
        if self.split == 'train':
            dates = [f"202303{i:02d}" for i in range(1, 23)]
        elif self.split == 'val':
            dates = [f"202303{i:02d}" for i in range(23, 29)]
        elif self.split == 'test':
            dates = [f"202303{i:02d}" for i in range(29, 32)]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        data_list = []
        for date in dates:
            date_path = os.path.join(self.data_root, date)
            if not os.path.exists(date_path): continue
            datas_dirs = glob.glob(os.path.join(date_path, "datas*"))
            for datas_dir in datas_dirs:
                csv_files = glob.glob(os.path.join(datas_dir, "*_labeled.csv"))
                data_list.extend(csv_files)
        return data_list

    def _compute_neighborhood_features(self, coord, feat, label):
        """
        (功能同上一步)
        - K-NN 基于 *物理坐标* (coord)。
        - 输入 *4D 特征* (u,v,sin,cos)。
        - 输出 *6D 特征* (u,v,sin,cos, u_std, v_std)。
        """
        spatial_features = coord

        if len(spatial_features) < self.k_neighbors:
            extra_zeros = np.zeros((len(feat), 2)) # (N, 2)
            new_feat = np.concatenate([feat, extra_zeros], axis=1) # (N, 4+2=6)
            return new_feat, label.copy(), coord

        tree = KDTree(spatial_features)
        _, indices = tree.query(spatial_features, k=self.k_neighbors)

        new_feat_dim = feat.shape[1] + 2 # 6D
        new_feat = np.zeros((len(spatial_features), new_feat_dim), dtype=np.float32)
        new_label = np.zeros(len(spatial_features), dtype=np.int64)

        for i in range(len(spatial_features)):
            neighbor_indices = indices[i]
            neighbor_feat = feat[neighbor_indices] # (k, 4)

            std_feat = np.std(neighbor_feat, axis=0)  # (4,)
            u_std = std_feat[0]
            v_std = std_feat[1]

            feat_i = feat[i].squeeze() # (4,)
            new_feat[i] = np.concatenate([feat_i, [u_std], [v_std]]) # (6,)

            neighbor_labels = label[neighbor_indices]
            counts = np.bincount(neighbor_labels, minlength=5)
            most_common_label = np.argmax(counts)
            new_label[i] = most_common_label

        if np.isnan(new_feat).any() or np.isinf(new_feat).any():
            nan_mask = np.isnan(new_feat).any(axis=1) | np.isinf(new_feat).any(axis=1)
            new_feat = new_feat[~nan_mask]
            new_label = new_label[~nan_mask]
            coord = coord[~nan_mask]
            logging.warning(f"邻域计算后过滤了{nan_mask.sum()}个含NaN/inf的点")

        return new_feat, new_label, coord


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx, warnings=None):
        csv_path = self.data_list[idx]
        try:
            data = pd.read_csv(csv_path)
        except Exception as e:
            return None # 静默失败

        if csv_path in self.filter_full_paths:
            return None

        # 提取 x,y,z, u,v,beamaz, label
        try:
            coord = data[["x", "y", "z"]].values.astype(np.float32)
            u = data["u"].values.astype(np.float32)
            v = data["v"].values.astype(np.float32)
            beamaz = data["BeamAz"].values.astype(np.float32)
            label = data["label"].values.astype(np.int64)
        except KeyError:
            try:
                coord = data[[" x", " y", " z"]].values.astype(np.float32)
                u = data[" u"].values.astype(np.float32)
                v = data[" v"].values.astype(np.float32)
                beamaz = data["BeamAz"].values.astype(np.float32)
                label = data[" label"].values.astype(np.int64)
            except Exception as e:
                logging.warning(f"读取 {csv_path} 列失败: {e}")
                return None

        # --- 步骤 1: 高度过滤 ---
        altitude_mask = (coord[:, 2] >= self.min_altitude) & (coord[:, 2] <= self.max_altitude)
        coord = coord[altitude_mask]
        u = u[altitude_mask]
        v = v[altitude_mask]
        beamaz = beamaz[altitude_mask]
        label = label[altitude_mask]
        if len(coord) == 0:
            return None

        # --- 步骤 2: 清洗 NaN / Inf ---
        coord_nan_mask = np.isnan(coord).any(axis=1) | np.isinf(coord).any(axis=1)
        feat_nan_mask = np.isnan(u) | np.isnan(v) | np.isnan(beamaz) | \
                        np.isinf(u) | np.isinf(v) | np.isinf(beamaz)
        valid_mask = ~(coord_nan_mask | feat_nan_mask)
        if not np.all(valid_mask):
            coord = coord[valid_mask]
            u = u[valid_mask]
            v = v[valid_mask]
            beamaz = beamaz[valid_mask]
            label = label[valid_mask]

        # --- 步骤 3: 转换 beamaz 为 sin/cos ---
        beamaz_rad = np.deg2rad(beamaz)
        beamaz_sin = np.sin(beamaz_rad)
        beamaz_cos = np.cos(beamaz_rad)
        feat_4d = np.column_stack([u, v, beamaz_sin, beamaz_cos])

        # --- 步骤 4: 清洗无效标签 ---
        valid_label_mask = (label >= 0) & (label <= 4)
        if not np.all(valid_label_mask):
            label = label[valid_label_mask]
            feat_4d = feat_4d[valid_label_mask]
            coord = coord[valid_label_mask]
        if len(coord) == 0:
            return None

        # --- 步骤 5: 计算 6D 邻域特征 ---
        feat_6d, label, coord = self._compute_neighborhood_features(coord, feat_4d, label)
        if len(coord) == 0:
            return None

        # --- 步骤 6: 构建字典，交给 Transform ---
        data_dict = {
            'coord': coord,
            'feat': feat_6d,
            'generate_label': label,
            'path': csv_path,
        }

        # --- 步骤 7: 执行 Transform ---
        if self.transform is not None:
            try:
                data_dict = self.transform(data_dict)
            except Exception as e:
                logging.error(f"样本{csv_path}变换失败：{str(e)}，已跳过")
                return None

        if data_dict is None:
            return None

        data_dict['path'] = csv_path
        if len(data_dict['coord']) < self.min_points:
            return None

        # 【添加】最终 NaN 检查（以防万一）
        if np.isnan(data_dict['coord']).any() or np.isnan(data_dict['feat']).any() or np.isnan(data_dict['generate_label']).any():
             logging.error(f"样本{csv_path}最终数据含 NaN，已跳过")
             return None

        return data_dict