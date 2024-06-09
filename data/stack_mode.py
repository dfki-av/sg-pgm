from functools import partial
import math
import numpy as np
import torch
from modules.ops import grid_subsample, radius_search
from data.torch_utils import build_dataloader
from torch_geometric.data import Batch
import torch


# Stack mode utilities

def precompute_data_stack_mode(points, insts, lengths, num_stages, voxel_size, radius, neighbor_limits):
    assert num_stages == len(neighbor_limits)

    points_list = []
    insts_list = []
    lengths_list = []
    neighbors_list = []
    subsampling_list = []
    upsampling_list = []

    # grid subsampling
    for i in range(num_stages):
        if i > 0:
            points, insts, lengths = grid_subsample(points, insts, lengths, voxel_size=voxel_size)
        points_list.append(points)
        insts_list.append(insts)
        lengths_list.append(lengths)
        voxel_size *= 2

    # radius search
    for i in range(num_stages):
        cur_points = points_list[i]
        cur_lengths = lengths_list[i]

        neighbors = radius_search(
            cur_points,
            cur_points,
            cur_lengths,
            cur_lengths,
            radius,
            neighbor_limits[i],
        )
        neighbors_list.append(neighbors)

        if i < num_stages - 1:
            sub_points = points_list[i + 1]
            sub_lengths = lengths_list[i + 1]

            subsampling = radius_search(
                sub_points,
                cur_points,
                sub_lengths,
                cur_lengths,
                radius,
                neighbor_limits[i],
            )
            subsampling_list.append(subsampling)

            upsampling = radius_search(
                cur_points,
                sub_points,
                cur_lengths,
                sub_lengths,
                radius * 2,
                neighbor_limits[i + 1],
            )
            upsampling_list.append(upsampling)

        radius *= 2

    return {
        'points': points_list,
        'insts': insts_list,
        'lengths': lengths_list,
        'neighbors': neighbors_list,
        'subsampling': subsampling_list,
        'upsampling': upsampling_list,
    }


def single_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for single point cloud in stack mode.

    Points are organized in the following order: [P_1, ..., P_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool=True)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: feats, points, normals
    if 'normals' in collated_dict:
        normals = torch.cat(collated_dict.pop('normals'), dim=0)
    else:
        normals = None
    feats = torch.cat(collated_dict.pop('feats'), dim=0)
    points_list = collated_dict.pop('points')
    #points_list = collated_dict.pop('insts')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)


    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            collated_dict[key] = value[0]

    if normals is not None:
        collated_dict['normals'] = normals
    collated_dict['features'] = feats
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        collated_dict.update(input_dict)
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size

    return collated_dict


def registration_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True, point_limits=None
):
    r"""Collate function for registration in stack mode.

    Points are organized in the following order: [ref_1, ..., ref_B, src_1, ..., src_B].
    The correspondence indices are within each point cloud without accumulation.

    Args:
        data_dicts (List[Dict])
        num_stages (int)
        voxel_size (float)
        search_radius (float)
        neighbor_limits (List[int])
        precompute_data (bool)

    Returns:
        collated_dict (Dict)
    """
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)

    # handle special keys: [ref_feats, src_feats] -> feats, [ref_points, src_points] -> points, lengths
    feats = torch.cat(collated_dict.pop('ref_feats') + collated_dict.pop('src_feats'), dim=0)
    points_list = collated_dict.pop('ref_points') + collated_dict.pop('src_points')
    insts_list = collated_dict.pop('ref_insts') + collated_dict.pop('src_insts')
    lengths = torch.LongTensor([points.shape[0] for points in points_list])
    points = torch.cat(points_list, dim=0)
    insts = torch.cat(insts_list, dim=0).int()
    if 'sg' in collated_dict.keys():
        follow_batch=['x_q', 'x_t']
        exclude_keys=None
        sg_pair_batched = Batch.from_data_list(collated_dict['sg'], follow_batch, exclude_keys)
        collated_dict['sg'] = sg_pair_batched
        
    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            if key != 'sg' and key != 'sg_match':
                collated_dict[key] = value[0]

    collated_dict['features'] = feats
    
    if precompute_data:
        input_dict = precompute_data_stack_mode(points, insts, lengths, num_stages, voxel_size, search_radius, neighbor_limits)
        voxel_size_ = voxel_size
        if point_limits > 0:
            while input_dict['points'][-1].shape[0] > point_limits:
                voxel_size_ = voxel_size_*math.sqrt(2)
                input_dict = precompute_data_stack_mode(points, insts, lengths, num_stages, voxel_size_, search_radius, neighbor_limits)
        collated_dict['voxel_size'] = voxel_size_
        collated_dict.update(input_dict)
        
    else:
        collated_dict['points'] = points
        collated_dict['lengths'] = lengths
    collated_dict['batch_size'] = batch_size
    #print(collated_dict.keys())

    return collated_dict


def calibrate_neighbors_stack_mode(
    dataset, collate_fn, num_stages, voxel_size, search_radius, keep_ratio=0.8, sample_threshold=20000, point_limits=-1
):
    # Compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (search_radius / voxel_size + 1) ** 3))
    neighbor_hists = np.zeros((num_stages, hist_n), dtype=np.int32)
    max_neighbor_limits = [hist_n] * num_stages

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i, sample in enumerate(dataset):
        data_dict = collate_fn(
            [sample], num_stages, voxel_size, search_radius, max_neighbor_limits, precompute_data=True, point_limits=point_limits
        )

        # update histogram
        counts = [np.sum(neighbors.numpy() < neighbors.shape[0], axis=1) for neighbors in data_dict['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighbor_hists += np.vstack(hists)

        if np.min(np.sum(neighbor_hists, axis=1)) > sample_threshold:
            break

    cum_sum = np.cumsum(neighbor_hists.T, axis=0)
    neighbor_limits = np.sum(cum_sum < (keep_ratio * cum_sum[hist_n - 1, :]), axis=0)

    return neighbor_limits


def build_dataloader_stack_mode(
    dataset,
    collate_fn,
    num_stages,
    voxel_size,
    search_radius,
    neighbor_limits,
    batch_size=1,
    num_workers=1,
    shuffle=False,
    drop_last=False,
    distributed=False,
    precompute_data=True,
    reproducibility=True,
    point_limits=None
):

    
    if reproducibility:
        g = torch.Generator()
        g.manual_seed(0)
    else:
        g = None

    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        generator=g,
        shuffle=shuffle,
        collate_fn=partial(
            collate_fn,
            num_stages=num_stages,
            voxel_size=voxel_size,
            search_radius=search_radius,
            neighbor_limits=neighbor_limits,
            precompute_data=precompute_data,
            point_limits=point_limits
        ),
        drop_last=drop_last,
        distributed=distributed,
    )
    return dataloader

def get_dataloader(train_dataset, val_dataset, cfg, args):

    '''
    if isinstance(train_dataset, torch.utils.data.IterableDataset):
        neighbor_limits = cfg.dataset.neighbor_limits
    else:
        neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius
    )
    '''
    neighbor_limits = cfg.dataset.neighbor_limits

    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        distributed=False,
        reproducibility=args.reproducibility,
        point_limits=cfg.dataset.max_c_points
    )

    val_loader = build_dataloader_stack_mode(
        val_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        distributed=False,
        reproducibility=args.reproducibility,
        point_limits=cfg.dataset.max_c_points
    )
    return train_loader, val_loader


def collate_fn_pointnet(data_dicts):
    batch_size = len(data_dicts)
    # merge data with the same key from different samples into a list
    collated_dict = {}
    for data_dict in data_dicts:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if key not in collated_dict:
                collated_dict[key] = []
            collated_dict[key].append(value)
    
    if 'sg' in collated_dict.keys():
        follow_batch=['x_q', 'x_t']
        exclude_keys=None
        sg_pair_batched = Batch.from_data_list(collated_dict['sg'], follow_batch, exclude_keys)
        collated_dict['sg'] = sg_pair_batched
        
    if batch_size == 1:
        # remove wrapping brackets if batch_size is 1
        for key, value in collated_dict.items():
            if key != 'sg' and key != 'sg_match':
                collated_dict[key] = value[0]
    
    collated_dict['batch_size'] = batch_size
    return collated_dict

def get_dataloader_pointnet(train_dataset, val_dataset, cfg, args):

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            generator=g,
            shuffle=True,
            collate_fn=collate_fn_pointnet,
            pin_memory=False,
            drop_last=False,
    )
    
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            generator=g,
            shuffle=True,
            collate_fn=collate_fn_pointnet,
            pin_memory=False,
            drop_last=False,
    )

    return train_loader, val_loader
