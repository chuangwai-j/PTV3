# pointcept/datasets/transforms/smote.py
import numpy as np
from scipy.spatial import cKDTree


def smote_pointcloud(coord, feat, label, target_class, k=5, generate_ratio=1.0):
    """
    (功能同上一步)
    - K-NN 搜索基于 'feat' (特征空间, 6D)。
    - 插值 'coord' (N,3) 和 'feat' (N,6)。
    """

    target_indices = np.where(label == target_class)[0]

    if len(target_indices) < k:
        # 点数不足 k，改为随机复制
        num_to_generate = int(len(target_indices) * generate_ratio)
        if num_to_generate == 0 or len(target_indices) == 0:
            return coord, feat, label
        new_indices = np.random.choice(target_indices, num_to_generate, replace=True)

        new_coords = coord[new_indices]
        new_feats = feat[new_indices]
        new_labels = label[new_indices]
    else:
        # 正常执行 SMOTE
        num_to_generate = int(len(target_indices) * generate_ratio)
        if num_to_generate == 0:
            return coord, feat, label

        # 【修改】K-NN 基于特征空间
        target_feats = feat[target_indices]
        tree = cKDTree(target_feats)  #

        base_indices = np.random.choice(target_indices, num_to_generate, replace=True)

        # 查找基点在 *特征空间* 的近邻
        _, nn_indices_in_target = tree.query(feat[base_indices], k=k)

        new_coords = []
        new_feats = []

        for i in range(num_to_generate):
            base_idx = base_indices[i]
            nn_idx_in_target = np.random.choice(nn_indices_in_target[i, 1:])
            nn_idx_global = target_indices[nn_idx_in_target]

            gap = np.random.rand()

            # 插值 coord 和 feat (6D)
            new_c = coord[base_idx] + (coord[nn_idx_global] - coord[base_idx]) * gap
            new_f = feat[base_idx] + (feat[nn_idx_global] - feat[base_idx]) * gap

            new_coords.append(new_c)
            new_feats.append(new_f)

        if not new_coords:
            return coord, feat, label

        new_labels = np.full(num_to_generate, target_class, dtype=label.dtype)

    # 拼接新生成的数据
    coord = np.concatenate([coord, np.array(new_coords)], axis=0)
    feat = np.concatenate([feat, np.array(new_feats)], axis=0)
    label = np.concatenate([label, new_labels], axis=0)

    return coord, feat, label