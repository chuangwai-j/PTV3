# pointcept/datasets/transforms/transform.py
import torch
import numpy as np
from .builder import TRANSFORMS
import logging # 添加日志

@TRANSFORMS.register_module()
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_dict):
        for t in self.transforms:
            try:
                data_dict = t(data_dict)
                if data_dict is None:
                    # logging.debug(f"Transform {t.__class__.__name__} returned None, stopping pipeline.")
                    return None # 允许变换中途过滤样本
            except Exception as e:
                path = data_dict.get('path', 'Unknown Path')
                logging.error(f"Error during transform {t.__class__.__name__} for sample {path}: {e}", exc_info=True)
                return None # 出错则过滤该样本
        return data_dict

@TRANSFORMS.register_module()
class ToTensor(object):
    """
    将 numpy 数组转换为 PyTorch Tensors。
    """
    def __call__(self, data_dict):
        if data_dict is None:
            return None
        try:
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    if key in ['generate_label']:
                        # 确保标签是 LongTensor
                        data_dict[key] = torch.from_numpy(value).long()
                    elif key in ['coord', 'feat', 'grid_size']:
                        # 特征和坐标是 FloatTensor
                        data_dict[key] = torch.from_numpy(value).float()
                    # 其他键 (如 'path') 保持不变
            return data_dict
        except Exception as e:
            path = data_dict.get('path', 'Unknown Path')
            logging.error(f"Error during ToTensor for sample {path}: {e}", exc_info=True)
            return None # ToTensor 出错则过滤