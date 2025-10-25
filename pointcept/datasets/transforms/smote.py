# pointcept/datasets/transforms/smote.py
import numpy as np
from scipy.spatial import cKDTree
import logging # 添加日志

def smote_pointcloud(coord, feat, label, target_class, k=5, generate_ratio=1.0):
    """
    - K-NN 搜索基于 'feat' (特征空间, 6D)。
    - 插值 'coord' (N,3) 和 'feat' (N,6)。
    """
    target_indices = np.where(label == target_class)[0]
    num_target_points = len(target_indices)

    if num_target_points == 0:
        # logging.debug(f"SMOTE: Class {target_class} has 0 points, skipping.")
        return coord, feat, label # 没有目标点，无法生成

    num_to_generate = int(num_target_points * generate_ratio)
    if num_to_generate == 0:
        # logging.debug(f"SMOTE: Class {target_class} num_to_generate is 0, skipping.")
        return coord, feat, label # 不需要生成

    if num_target_points < k:
        # 点数不足 k，改为随机复制
        # logging.debug(f"SMOTE: Class {target_class} has {num_target_points} points (< k={k}), using random duplication.")
        new_indices = np.random.choice(target_indices, num_to_generate, replace=True)
        new_coords = coord[new_indices]
        new_feats = feat[new_indices]
        new_labels = label[new_indices] # 复制标签
    else:
        # 正常执行 SMOTE
        # logging.debug(f"SMOTE: Class {target_class} has {num_target_points} points (>= k={k}), generating {num_to_generate} new points.")
        target_coords = coord[target_indices]
        target_feats = feat[target_indices]

        try:
            tree = cKDTree(target_feats) # K-NN 基于特征空间
        except ValueError as e:
            logging.error(f"SMOTE: Error creating KDTree for class {target_class} (maybe NaN/inf in feats?): {e}")
            # KDTree 创建失败，回退到复制
            new_indices = np.random.choice(target_indices, num_to_generate, replace=True)
            new_coords = coord[new_indices]
            new_feats = feat[new_indices]
            new_labels = label[new_indices]
            return coord, feat, label # 返回，不再继续

        base_indices = np.random.choice(target_indices, num_to_generate, replace=True)

        try:
             # 查找基点在 *特征空间* 的近邻
            _, nn_indices_in_target = tree.query(feat[base_indices], k=k)
        except ValueError as e:
            logging.error(f"SMOTE: Error querying KDTree for class {target_class}: {e}")
            # 查询失败，回退到复制
            new_indices = np.random.choice(target_indices, num_to_generate, replace=True)
            new_coords = coord[new_indices]
            new_feats = feat[new_indices]
            new_labels = label[new_indices]
            return coord, feat, label # 返回，不再继续

        new_coords_list = []
        new_feats_list = []

        for i in range(num_to_generate):
            base_idx = base_indices[i]
            # 确保邻居索引有效且不包含自身
            valid_neighbors = nn_indices_in_target[i, 1:] # 排除自身
            if len(valid_neighbors) == 0:
                 # 如果没有有效邻居（可能k=1或重复点），就复制基点
                 nn_idx_global = base_idx
            else:
                nn_idx_in_target = np.random.choice(valid_neighbors)
                nn_idx_global = target_indices[nn_idx_in_target]

            gap = np.random.rand()

            # 插值 coord 和 feat (6D)
            new_c = coord[base_idx] + (coord[nn_idx_global] - coord[base_idx]) * gap
            new_f = feat[base_idx] + (feat[nn_idx_global] - feat[base_idx]) * gap

            new_coords_list.append(new_c)
            new_feats_list.append(new_f)

        if not new_coords_list:
             # logging.debug(f"SMOTE: Class {target_class} generated 0 points unexpectedly.")
             return coord, feat, label

        new_coords = np.array(new_coords_list)
        new_feats = np.array(new_feats_list)
        new_labels = np.full(num_to_generate, target_class, dtype=label.dtype)

    # 拼接新生成的数据
    # logging.debug(f"SMOTE: Class {target_class} generated {len(new_labels)} points. Original points: {len(coord)}")
    coord = np.concatenate([coord, new_coords], axis=0)
    feat = np.concatenate([feat, new_feats], axis=0)
    label = np.concatenate([label, new_labels], axis=0)

    return coord, feat, label