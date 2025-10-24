# pointcept/datasets/dataloader.py
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .builder import build_dataset

# 【修改】从 misc.py 导入的辅助函数现在直接定义在这里
import warnings
import logging
import numpy as np
import os


def collate_fn(batch):
    """
    【新增】
    适配风切变 6D 方案的本地批处理函数。
    此函数基于 misc.py 中的版本修改而来：
    1. 移除了对 'beamaz' 键的所有引用和断言。
    2. 将 'feat' 的维度断言更新为 6 维。
    """

    # -------------------------- 第一步：过滤None样本 --------------------------
    batch = [item for item in batch if item is not None]
    if not batch:
        warnings.warn("当前batch所有样本均为无效（点数不足），返回空batch，需在训练循环中跳过")
        return {'path': [], 'coord': torch.empty((0, 3)), 'offset': torch.tensor([0])}

    # -------------------------- 第二步：检查并补全path --------------------------
    for idx, item in enumerate(batch):
        if 'path' not in item:
            item['path'] = f"缺失路径样本_{idx}"
            warnings.warn(f"batch中样本{idx}丢失path，已用默认名称兜底")

        # (日志打印保持不变)
        num_points = item['coord'].shape[0] if isinstance(item['coord'], torch.Tensor) else len(item['coord'])
        if isinstance(item['coord'], torch.Tensor):
            # 确保在 CPU 上执行 min/max，或处理可能的空 Tensor
            if num_points > 0:
                coord_min = item['coord'].min(axis=0).values
                coord_max = item['coord'].max(axis=0).values
            else:
                coord_min = [0, 0, 0];
                coord_max = [0, 0, 0]
        else:
            if num_points > 0:
                coord_min = item['coord'].min(axis=0)
                coord_max = item['coord'].max(axis=0)
            else:
                coord_min = [0, 0, 0];
                coord_max = [0, 0, 0]

        logging.debug(
            f"样本{idx}（{os.path.basename(item['path'])}）：点数={num_points}，"
            f"coord范围=x[{coord_min[0]:.0f}~{coord_max[0]:.0f}], "
            f"y[{coord_min[1]:.0f}~{coord_max[1]:.0f}], z[{coord_min[2]:.0f}~{coord_max[2]:.0f}]"
        )

    # -------------------------- 第三步：过滤无效样本 --------------------------
    valid_batch = []
    for idx, item in enumerate(batch):
        if not isinstance(item, dict) or 'path' not in item or 'coord' not in item:
            warnings.warn(f"过滤无效样本{idx}：非dict或缺失path/coord字段", UserWarning)
            continue
        num_points = item['coord'].shape[0] if isinstance(item['coord'], torch.Tensor) else len(item['coord'])
        if num_points <= 0:
            warnings.warn(
                f"过滤空样本{idx}（{os.path.basename(item['path'])}）：点数={num_points}",
                UserWarning
            )
            continue
        valid_batch.append(item)
    if not valid_batch:
        raise ValueError("当前batch无有效样本！请检查数据预处理/补点流程")

    # -------------------------- 第四步：初始化变量+校验单样本合法性 --------------------------
    result = {}
    offsets = [0]
    total_points = 0
    sample_sizes = []
    sample_paths = []
    device = valid_batch[0]['coord'].device if isinstance(valid_batch[0]['coord'], torch.Tensor) else torch.device(
        'cpu')

    for idx, item in enumerate(valid_batch):
        # 4.1 校验点数（保持不变）
        num_points = item['coord'].shape[0] if isinstance(item['coord'], torch.Tensor) else len(item['coord'])
        if num_points % 384 != 0:
            raise ValueError(
                f"样本{idx}（{os.path.basename(item['path'])}）点数异常：{num_points}（需为384的倍数）"
            )
        sample_sizes.append(num_points)
        total_points += num_points
        offsets.append(total_points)
        sample_paths.append(item['path'])

        # 4.2 统一字段类型+维度校验
        # 【修改】移除 'beamaz'
        for key in ['coord', 'feat', 'generate_label', 'grid_size']:
            if key not in item:
                raise KeyError(f"样本{idx}（{os.path.basename(item['path'])}）缺失必要字段：{key}")

            if isinstance(item[key], np.ndarray):
                dtype = torch.long if key == 'generate_label' else torch.float32
                item[key] = torch.from_numpy(item[key]).to(dtype).to(device)
            elif not isinstance(item[key], torch.Tensor):
                raise TypeError(f"样本{idx}（{os.path.basename(item['path'])}）的{key}类型异常：需Tensor/numpy数组")

            # 维度校验
            if key == 'coord':
                if item[key].dim() != 2 or item[key].shape != (num_points, 3):
                    raise ValueError(f"样本{idx} coord异常：形状={item[key].shape}，需 ({num_points}, 3)")
            elif key == 'feat':
                # 【修改】现在 feat 是 6D
                if item[key].dim() != 2 or item[key].shape[0] != num_points or item[key].shape[1] != 6:
                    raise ValueError(
                        f"样本{idx} feat异常：形状={item[key].shape}\n"
                        f"需为 (点数, 6)，当前点数={num_points}，应满足 ({num_points}, 6)"
                    )
            elif key == 'generate_label':  # 【修改】移除 'beamaz'
                if item[key].dim() != 1 or item[key].shape[0] != num_points:
                    raise ValueError(
                        f"样本{idx} {key}异常：形状={item[key].shape}\n"
                        f"需为 (点数,)，当前点数={num_points}"
                    )
            elif key == 'grid_size':
                if item[key].dim() != 1 or item[key].shape[0] != 3:
                    raise ValueError(f"样本{idx} grid_size异常：形状={item[key].shape}，需 (3,)")

    # -------------------------- 第五步：生成offset并校验 --------------------------
    result['offset'] = torch.tensor(offsets, dtype=torch.int64, device=device)
    if not (torch.diff(result['offset']) > 0).all():
        raise ValueError(f"offset生成异常（需严格递增）：{result['offset'].tolist()}")
    if result['offset'][-1] != total_points:
        raise ValueError(
            f"offset总点数不匹配：offset[-1]={result['offset'][-1]}，实际总点数={total_points}"
        )

    # -------------------------- 第六步：拼接点级字段 --------------------------
    result['coord'] = torch.cat([item['coord'] for item in valid_batch], dim=0)
    result['feat'] = torch.cat([item['feat'] for item in valid_batch], dim=0)
    result['generate_label'] = torch.cat([item['generate_label'] for item in valid_batch], dim=0)
    # 【修改】移除 'beamaz' 的拼接

    # 6.2 检查标签有效性（保持不变）
    if torch.isnan(result['generate_label']).any():
        nan_indices = torch.where(torch.isnan(result['generate_label']))[0]
        print(f"⚠️ 标签含NaN！首批NaN索引：{nan_indices[:5]}，样本路径：{sample_paths}")
    if (result['generate_label'] < 0).any() or (result['generate_label'] >= 5).any():
        invalid_indices = torch.where((result['generate_label'] < 0) | (result['generate_label'] >= 5))[0]
        invalid_values = result['generate_label'][invalid_indices[:5]]
        print(f"⚠️ 标签越界！首批越界值：{invalid_values}，样本路径：{sample_paths}")

    # -------------------------- 第七步：校验拼接后维度 --------------------------
    assert result['coord'].shape == (total_points, 3), f"coord拼接异常：{result['coord'].shape} != ({total_points}, 3)"
    assert result['feat'].shape == (total_points,
                                    6), f"feat拼接异常：{result['feat'].shape} != ({total_points}, 6)"  # 【修改】断言 6D
    assert result['generate_label'].shape == (
        total_points,), f"label拼接异常：{result['generate_label'].shape} != ({total_points},)"
    # 【修改】移除 'beamaz' 断言

    # -------------------------- 第八步：处理grid_size --------------------------
    grid_sizes = [item['grid_size'] for item in valid_batch]
    result['grid_size'] = grid_sizes[0].clone().detach().float()
    if not all(torch.equal(gs, result['grid_size']) for gs in grid_sizes):
        logging.debug(
            f"当前batch样本grid_size不一致（已取第一个样本的{result['grid_size']}作为统一值）\n"
            f"各样本grid_size：{[gs.tolist() for gs in grid_sizes]}, 样本路径：{sample_paths}"
        )

    # -------------------------- 第九步：拼接path --------------------------
    result['path'] = sample_paths
    logging.debug(f"Batch路径列表：{[os.path.basename(p) for p in result['path']]}，共{len(result['path'])}个样本")

    # -------------------------- 第十步：调试日志 --------------------------
    batch_coord_min = result['coord'].min(axis=0).values
    batch_coord_max = result['coord'].max(axis=0).values
    logging.info(f"✅ Batch生成成功：样本数={len(valid_batch)}，总点数={total_points}")
    logging.info(f"   各样本信息：点数={sample_sizes}，路径={[os.path.basename(p) for p in sample_paths]}")
    logging.info(
        f"   整体coord范围：x[{batch_coord_min[0]:.0f}~{batch_coord_max[0]:.0f}], y[{batch_coord_min[1]:.0f}~{batch_coord_max[1]:.0f}], z[{batch_coord_min[2]:.0f}~{batch_coord_max[2]:.0f}]")
    logging.info(
        f"   拼接后维度：coord={result['coord'].shape}，feat={result['feat'].shape}，generate_label={result['generate_label'].shape}")
    logging.info(f"   Offset：{result['offset'].tolist()}")

    return result


def offset2bincount(offset, check_padding=True):
    """
    【新增】
    从 misc.py 迁移而来。
    从offset计算样本点数（增强鲁棒性）。
    """
    # 1. 基础类型/维度校验
    if not isinstance(offset, torch.Tensor):
        raise TypeError(f"offset必须为torch.Tensor，实际类型：{type(offset)}")
    if offset.dim() != 1:
        raise ValueError(f"offset必须为1维张量，实际维度：{offset.dim()}")
    if offset.shape[0] < 2:
        raise ValueError(f"offset长度必须≥2（如[0, 1536]），实际长度：{offset.shape[0]}")

    # 2. 计算样本点数，校验有效性
    bincount = offset[1:] - offset[:-1]
    invalid_mask = bincount <= 0
    if invalid_mask.any():
        invalid_indices = torch.where(invalid_mask)[0].tolist()
        invalid_values = bincount[invalid_mask].tolist()
        raise ValueError(
            f"存在点数≤0的样本：索引={invalid_indices}，点数={invalid_values}\n"
            f"完整offset：{offset.tolist()}，完整样本点数：{bincount.tolist()}"
        )

    # 3. 可选：校验补点逻辑（保持 384 倍数检查）
    if check_padding:
        # (补点后的最小点数检查，可以放宽，只检查 384 的倍数)
        # small_mask = bincount < 384 # 至少要有一个 chunk
        # if small_mask.any(): ...

        if (bincount % 384 != 0).any():
            wrong_indices = torch.where(bincount % 384 != 0)[0].tolist()
            wrong_values = bincount[wrong_indices].tolist()
            raise ValueError(
                f"原始样本点数非384的倍数：索引={wrong_indices}，点数={wrong_values}\n"
                "需与补点逻辑（384倍数）保持一致"
            )
    else:
        min_points = bincount.min().item()
        if min_points < 384:
            logging.debug(f"[注意] 下采样后样本最小点数为{min_points}（小于384），offset={offset.tolist()}")

    return bincount.to(offset.device)


def build_dataloader(cfg, mode='train', dist=False):
    """
    【修改】
    此函数现在使用本文件中定义的 *本地* `collate_fn`。
    """
    dataset = build_dataset(cfg)

    if dist:
        sampler = DistributedSampler(dataset, shuffle=(mode == 'train'))
    else:
        sampler = None

    if mode == 'train':
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collate_fn,  # 【修改】使用本地 collate_fn
            pin_memory=cfg.train.pin_memory,
            prefetch_factor=cfg.train.prefetch_factor if hasattr(cfg.train, 'prefetch_factor') else 2,
            persistent_workers=True if cfg.train.num_workers > 0 else False
        )
    elif mode == 'val':
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.evaluation.batch_size,
            num_workers=cfg.evaluation.num_workers,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_fn,  # 【修改】使用本地 collate_fn
            pin_memory=cfg.evaluation.pin_memory,
            prefetch_factor=cfg.evaluation.prefetch_factor if hasattr(cfg.evaluation, 'prefetch_factor') else 2,
            persistent_workers=True if cfg.evaluation.num_workers > 0 else False
        )
    elif mode == 'test':
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_fn,  # 【修改】使用本地 collate_fn
            pin_memory=cfg.test.pin_memory,
            prefetch_factor=cfg.test.prefetch_factor if hasattr(cfg.test, 'prefetch_factor') else 2,
            persistent_workers=True if cfg.test.num_workers > 0 else False
        )
    else:
        raise Exception("Unknown mode.")
    return dataloader