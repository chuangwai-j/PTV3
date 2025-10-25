# pointcept/datasets/transforms/wind_shear.py
import os
import logging
import numpy as np
import torch
from .builder import TRANSFORMS
from scipy.spatial import cKDTree
from .smote import smote_pointcloud
from pointcept.utils.registry import Registry

@TRANSFORMS.register_module()
class NormalizeFeatures(object):
    """
    归一化坐标(3D)和特征(6D)。在采样之后运行。
    """
    def __init__(self, coord_mean, coord_std, feat_mean, feat_std):
        self.coord_mean = np.array(coord_mean, dtype=np.float32).reshape(1, 3)
        self.coord_std = np.array(coord_std, dtype=np.float32).reshape(1, 3)
        self.feat_mean = np.array(feat_mean, dtype=np.float32).reshape(1, 6) # 6D
        self.feat_std = np.array(feat_std, dtype=np.float32).reshape(1, 6) # 6D

        self.coord_std[self.coord_std < 1e-6] = 1e-6
        self.feat_std[self.feat_std < 1e-6] = 1e-6

    def __call__(self, data_dict):
        if 'coord' not in data_dict or 'feat' not in data_dict:
            return data_dict

        data_dict['coord'] = (data_dict['coord'] - self.coord_mean) / self.coord_std
        data_dict['feat'] = (data_dict['feat'] - self.feat_mean) / self.feat_std

        data_dict['coord'] = np.clip(data_dict['coord'], -5.0, 5.0)
        data_dict['feat'] = np.clip(data_dict['feat'], -5.0, 5.0)

        data_dict['coord'] = np.nan_to_num(data_dict['coord'], nan=0.0, posinf=5.0, neginf=-5.0)
        data_dict['feat'] = np.nan_to_num(data_dict['feat'], nan=0.0, posinf=5.0, neginf=-5.0)

        return data_dict


@TRANSFORMS.register_module()
class WindShearGridSample:
    """
    在 *归一化之前* 运行，处理 *原始* 尺度数据 (N, 6)。
    SMOTE 基于 *原始 6D 特征空间* 查找近邻。
    """
    def __init__(self, grid_size=80.0, min_points=50, adaptive=True,
                 balance_data=True,
                 rare_classes=[0, 3, 4],     #
                 dominant_classes=[1],    #
                 smote_ratio=0.5,
                 undersample_ratio=2.0):

        self.grid_size = grid_size
        self.min_points = min_points
        self.adaptive = adaptive
        self.balance_data = balance_data
        self.rare_classes = rare_classes
        self.dominant_classes = dominant_classes
        self.smote_ratio = smote_ratio
        self.undersample_ratio = undersample_ratio

        log_msg = f"WindShearGridSample Initialized: Adaptive={self.adaptive}, MinPoints={self.min_points}, "
        if self.balance_data:
            log_msg += f"Balancing ON (SMOTE Cls={self.rare_classes}, Ratio={self.smote_ratio}; Undersample Cls={self.dominant_classes}, Ratio={self.undersample_ratio})"
        else:
            log_msg += "Balancing OFF (Val/Test Mode)"
        logging.info(log_msg)


    def __call__(self, data_dict):
        original_path = data_dict.get('path', '未知路径')
        coord = data_dict["coord"] # (N, 3) 原始坐标
        feat = data_dict["feat"]   # (N, 6) 原始 6D 特征
        label = data_dict["generate_label"]

        # --- 1. 网格采样 ---
        # (基于 'coord' (原始坐标) 进行)
        if len(coord) == 0: return None # 初始点为空
        coord_min = coord.min(axis=0, keepdims=True)
        coord_min = np.nan_to_num(coord_min, nan=0.0)
        coord_quantized = (coord - coord_min).round().astype(np.int32)
        coord_quantized = np.clip(coord_quantized, 0, 100000)
        x_min_global, y_min_global, z_min_global = coord_quantized.min(axis=0)
        x_max, y_max, z_max = coord_quantized.max(axis=0)

        if self.adaptive:
            # 防止范围为0导致除零
            range_x = max(1.0, x_max - x_min_global)
            range_y = max(1.0, y_max - y_min_global)
            range_z = max(1.0, z_max - z_min_global)
            grid_size_x = max(self.grid_size, range_x / 50)
            grid_size_y = max(self.grid_size, range_y / 50)
            grid_size_z = max(self.grid_size, range_z / 10)
        else:
            # ... (非自适应逻辑) ...
            if isinstance(self.grid_size, (list, np.ndarray)) and len(self.grid_size) == 3:
                grid_size_x, grid_size_y, grid_size_z = self.grid_size
            else:
                grid_size_x = grid_size_y = grid_size_z = self.grid_size
        grid_size_x, grid_size_y, grid_size_z = max(grid_size_x, 1e-3), max(grid_size_y, 1e-3), max(grid_size_z, 1e-3)

        grid_idx = np.floor(
            (coord_quantized - [x_min_global, y_min_global, z_min_global])
            / [grid_size_x, grid_size_y, grid_size_z]
        ).astype(int)
        grid_idx = np.clip(grid_idx, 0, None)
        max_z = grid_idx[:, 2].max() + 1
        max_y = grid_idx[:, 1].max() + 1
        grid_id = grid_idx[:, 0] * max_y * max_z + grid_idx[:, 1] * max_z + grid_idx[:, 2]

        unique_gids, unique_indices = np.unique(grid_id, return_index=True)
        # 使用 unique_indices 直接进行采样，效率更高
        sampled_indices = unique_indices

        # --- 2. 同步采样 ---
        sampled_coord = coord[sampled_indices]
        sampled_feat = feat[sampled_indices]
        sampled_label = label[sampled_indices]

        # --- 3. 数据平衡 (仅 'train' 模式) ---
        if self.balance_data:
            # SMOTE 过采样
            for cls in self.rare_classes: #
                if np.sum(sampled_label == cls) > 3: # k=3 for SMOTE
                    sampled_coord, sampled_feat, sampled_label = smote_pointcloud(
                        sampled_coord, sampled_feat, sampled_label,
                        target_class=cls, k=3, generate_ratio=self.smote_ratio
                    )
            # 随机欠采样
            dominant_mask = np.isin(sampled_label, self.dominant_classes) #
            indices_dominant = np.where(dominant_mask)[0]
            indices_non_dominant = np.where(~dominant_mask)[0]

            if len(indices_non_dominant) == 0 and len(indices_dominant) > 0:
                 max_dominant_points = max(self.min_points, int(len(indices_dominant) * 0.1))
                 final_indices = np.random.choice(indices_dominant, max_dominant_points, replace=False)
            else:
                max_dominant_points = int(len(indices_non_dominant) * self.undersample_ratio)
                if len(indices_dominant) > max_dominant_points and max_dominant_points >= 0: # 允许 max_dominant_points 为 0
                    undersampled_indices_dominant = np.random.choice(
                        indices_dominant, max_dominant_points, replace=False
                    )
                    final_indices = np.concatenate([indices_non_dominant, undersampled_indices_dominant])
                else: # 优势类不够多，或非优势类为0
                    final_indices = np.concatenate([indices_non_dominant, indices_dominant])

            # 检查 final_indices 是否为空
            if len(final_indices) > 0:
                sampled_coord = sampled_coord[final_indices]
                sampled_feat = sampled_feat[final_indices]
                sampled_label = sampled_label[final_indices]
            else:
                # 如果平衡后点数为0，提前返回 None
                 logging.warning(f"样本{original_path}在平衡后点数为0，已跳过")
                 return None


        # --- 4. 补点至384的倍数 ---
        sampled_num = len(sampled_coord)
        if sampled_num < self.min_points:
            logging.debug(f"样本{original_path}采样/平衡后点数({sampled_num})不足 {self.min_points}，已跳过")
            return None # 点数不足，丢弃

        min_multiple = 384
        target_num = max(min_multiple, ((sampled_num + min_multiple - 1) // min_multiple) * min_multiple)
        pad_num = target_num - sampled_num

        if pad_num > 0:
            pad_indices = np.random.choice(sampled_num, pad_num, replace=True)
            sampled_coord = np.concatenate([sampled_coord, sampled_coord[pad_indices]], axis=0)
            sampled_feat = np.concatenate([sampled_feat, sampled_feat[pad_indices]], axis=0)
            sampled_label = np.concatenate([sampled_label, sampled_label[pad_indices]], axis=0)

        # --- 5. 最终组装 (原始尺度) ---
        return {
            "coord": sampled_coord.astype(np.float32), # 确保 float32
            "feat": sampled_feat.astype(np.float32),   # 确保 float32
            "generate_label": sampled_label.astype(np.int64), # 确保 int64
            "grid_size": np.array([grid_size_x, grid_size_y, grid_size_z], dtype=np.float32),
            "path": original_path
        }