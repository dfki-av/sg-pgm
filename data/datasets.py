if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    sys.path.append('../../')

import os, random, torch
import os.path as osp
import re
import numpy as np
from torch_geometric.data import Data
from utils.pointcloud import random_sample_rotation,random_sample_translation, get_transform_from_rotation_translation
from utils import common, scan3r

def project_to_binary_dimension(number, dimension):
    binary_representation = bin(number)[2:]  # Convert to binary and remove the '0b' prefix
    binary_representation = binary_representation.zfill(dimension)  # Pad with zeros to achieve the desired dimension

    # Convert the binary string to a list of integers
    binary_list = [int(bit) for bit in binary_representation]

    return binary_list

class PairData(Data): # For packing target and query scene graph
    def __inc__(self, key, value, *args):
        if bool(re.search('index_q', key)):
            return self.x_q.size(0)
        if bool(re.search('index_t', key)):
            return self.x_t.size(0)
        else:
            return 0

class Scan3RDataset_pointnet(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        use_augmentation=True,
        augmentation_noise=0.01,
        augmentation_rotation=1,
        augmentation_translation=0.5,
        debug=False,
        anchor_type_name=''
        ):
        self.mode='train'
        self.dataset_root = dataset_root
        self.split = split
        self.pc_resolution = 512
        self.anchor_type_name = anchor_type_name
        self.scans_dir = os.path.join(dataset_root)
        self.scans_scenes_dir = os.path.join(self.scans_dir, 'scenes')
        self.scans_files_dir = os.path.join(self.scans_dir, 'files')

        self.subscans_dir = osp.join(dataset_root, 'out')
        self.subscans_scenes_dir = osp.join(self.subscans_dir, 'scenes')
        self.subscans_files_dir = osp.join(self.subscans_dir, 'files')

        self.mode = 'orig'
        self.anchor_data_filename = os.path.join(self.subscans_files_dir, '{}/anchors{}_{}.json'.format(self.mode, self.anchor_type_name, split))
        
        self.anchor_data = common.load_json(self.anchor_data_filename)
        random.shuffle(self.anchor_data)
        self.is_training = self.split == 'train'

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.augmentation_translation = augmentation_translation
        self.predicted = False
        self.label_file_name = 'labels.instances.align.annotated.v2.ply' if not self.predicted else 'inseg_filtered.ply'

        self.debug = debug

    def __len__(self):
        return len(self.anchor_data)
    
    def _augment_point_cloud(self, ref_points, src_points, rotation, translation, split):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        aug_translation = random_sample_translation()
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T) + aug_translation
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation) + aug_translation

        else:
            src_points = np.matmul(src_points + aug_translation, aug_rotation.T) 
            rotation = np.matmul(rotation, aug_rotation.T)
            translation =  - aug_translation#np.matmul(aug_rotation, aug_translation)

        if split == 'train':
            ref_points += (np.random.rand(ref_points.shape[0],ref_points.shape[1] , 3) - 0.5) * self.aug_noise
            src_points += (np.random.rand(src_points.shape[0],src_points.shape[1],  3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation
    
    def _collate_entity_idxs(self, batch):
        e1i = np.concatenate([data['e1i'] for data in batch])
        e2i = np.concatenate([data['e2i'] for data in batch])
        e1j = np.concatenate([data['e1j'] for data in batch])
        e2j = np.concatenate([data['e2j'] for data in batch])

        e1i_start_idx = 0 
        e2i_start_idx = 0 
        e1j_start_idx = 0 
        e2j_start_idx = 0 
        prev_obj_cnt = 0
        
        for idx in range(len(batch)):
            e1i_end_idx = e1i_start_idx + batch[idx]['e1i_count']
            e2i_end_idx = e2i_start_idx + batch[idx]['e2i_count']
            e1j_end_idx = e1j_start_idx + batch[idx]['e1j_count']
            e2j_end_idx = e2j_start_idx + batch[idx]['e2j_count']

            e1i[e1i_start_idx : e1i_end_idx] += prev_obj_cnt
            e2i[e2i_start_idx : e2i_end_idx] += prev_obj_cnt
            e1j[e1j_start_idx : e1j_end_idx] += prev_obj_cnt
            e2j[e2j_start_idx : e2j_end_idx] += prev_obj_cnt
            
            e1i_start_idx, e2i_start_idx, e1j_start_idx, e2j_start_idx = e1i_end_idx, e2i_end_idx, e1j_end_idx, e2j_end_idx
            prev_obj_cnt += batch[idx]['tot_obj_count']
        
        e1i = e1i.astype(np.int32)
        e2i = e2i.astype(np.int32)
        e1j = e1j.astype(np.int32)
        e2j = e2j.astype(np.int32)

        return e1i, e2i, e1j, e2j

    def _collate_feats(self, batch, key):
        feats = torch.cat([data[key] for data in batch])
        return feats
    
    def collate_fn(self, batch):
        tot_object_points = self._collate_feats(batch, 'tot_obj_pts')
        tot_bow_vec_object_attr_feats = self._collate_feats(batch, 'tot_bow_vec_object_attr_feats')
        tot_bow_vec_object_edge_feats = self._collate_feats(batch, 'tot_bow_vec_object_edge_feats')    
        tot_rel_pose = self._collate_feats(batch, 'tot_rel_pose')
        
        data_dict = {}
        data_dict['tot_obj_pts'] = tot_object_points
        data_dict['e1i'], data_dict['e2i'], data_dict['e1j'], data_dict['e2j'] = self._collate_entity_idxs(batch)

        data_dict['e1i_count'] = np.stack([data['e1i_count'] for data in batch])
        data_dict['e2i_count'] = np.stack([data['e2i_count'] for data in batch])
        data_dict['e1j_count'] = np.stack([data['e1j_count'] for data in batch])
        data_dict['e2j_count'] = np.stack([data['e2j_count'] for data in batch])
        data_dict['tot_obj_count'] = np.stack([data['tot_obj_count'] for data in batch])
        data_dict['global_obj_ids'] = np.concatenate([data['global_obj_ids'] for data in batch])
        
        data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_object_attr_feats.double()
        data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_object_edge_feats.double()
        data_dict['tot_rel_pose'] = tot_rel_pose.double()
        data_dict['graph_per_obj_count'] = np.stack([data['graph_per_obj_count'] for data in batch])
        data_dict['graph_per_edge_count'] = np.stack([data['graph_per_edge_count'] for data in batch])
        data_dict['edges'] = self._collate_feats(batch, 'edges')
        data_dict['scene_ids'] = np.stack([data['scene_ids'] for data in batch])
        data_dict['obj_ids'] = np.concatenate([data['obj_ids'] for data in batch])
        data_dict['pcl_center'] = np.stack([data['pcl_center'] for data in batch])
        
        data_dict['overlap'] = np.stack([data['overlap'] for data in batch])
        data_dict['batch_size'] = data_dict['overlap'].shape[0]

        return data_dict

    def __getitem__(self, idx):
        graph_data = self.anchor_data[idx]
        src_scan_id = graph_data['src']
        ref_scan_id = graph_data['ref']
        overlap = graph_data['overlap']
        
        # Centering
        src_points = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(src_scan_id)), obj_ids = None)
        ref_points = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(ref_scan_id)), obj_ids = None)

        if self.split == 'train':
            if np.random.rand(1)[0] > 0.5:
                pcl_center = np.mean(src_points, axis=0)
            else:
                pcl_center = np.mean(ref_points, axis=0)
        else:
            pcl_center = np.mean(src_points, axis=0)

        src_data_dict = common.load_pkl_data(osp.join(self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, src_scan_id)))
        ref_data_dict = common.load_pkl_data(osp.join(self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, ref_scan_id)))
        
        src_object_ids = src_data_dict['objects_id']
        ref_object_ids = ref_data_dict['objects_id']
        anchor_obj_ids = graph_data['anchorIds']
        global_object_ids = np.concatenate((src_data_dict['objects_cat'], ref_data_dict['objects_cat']))
        
        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id != 0]
        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id in src_object_ids and anchor_obj_id in ref_object_ids]

        src_edges = src_data_dict['edges']
        ref_edges = ref_data_dict['edges']

        src_object_points = src_data_dict['obj_points'][self.pc_resolution] - pcl_center
        ref_object_points = ref_data_dict['obj_points'][self.pc_resolution] - pcl_center

        edges = torch.cat([torch.from_numpy(src_edges), torch.from_numpy(ref_edges)])

        src_object_id2idx = src_data_dict['object_id2idx']
        e1i_idxs = np.array([src_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]) # e1i
        e1j_idxs = np.array([src_object_id2idx[object_id] for object_id in src_data_dict['objects_id'] if object_id not in anchor_obj_ids]) # e1j
        
        ref_object_id2idx = ref_data_dict['object_id2idx']
        e2i_idxs = np.array([ref_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]) + src_object_points.shape[0] # e2i
        e2j_idxs = np.array([ref_object_id2idx[object_id] for object_id in ref_data_dict['objects_id'] if object_id not in anchor_obj_ids]) + src_object_points.shape[0] # e2j

        tot_object_points = torch.cat([torch.from_numpy(src_object_points), torch.from_numpy(ref_object_points)]).type(torch.FloatTensor)
        tot_bow_vec_obj_attr_feats = torch.cat([torch.from_numpy(src_data_dict['bow_vec_object_attr_feats']), torch.from_numpy(ref_data_dict['bow_vec_object_attr_feats'])])
        tot_bow_vec_obj_edge_feats = torch.cat([torch.from_numpy(src_data_dict['bow_vec_object_edge_feats']), torch.from_numpy(ref_data_dict['bow_vec_object_edge_feats'])])
        tot_rel_pose = torch.cat([torch.from_numpy(src_data_dict['rel_trans']), torch.from_numpy(ref_data_dict['rel_trans'])])

        data_dict = {} 
        data_dict['obj_ids'] = np.concatenate([src_object_ids, ref_object_ids])
        data_dict['tot_obj_pts'] = tot_object_points
        data_dict['graph_per_obj_count'] = np.array([src_object_points.shape[0], ref_object_points.shape[0]])
        data_dict['graph_per_edge_count'] = np.array([src_edges.shape[0], ref_edges.shape[0]])
        
        data_dict['e1i'] = e1i_idxs
        data_dict['e1i_count'] = e1i_idxs.shape[0]
        data_dict['e2i'] = e2i_idxs
        data_dict['e2i_count'] = e2i_idxs.shape[0]
        data_dict['e1j'] = e1j_idxs
        data_dict['e1j_count'] = e1j_idxs.shape[0]
        data_dict['e2j'] = e2j_idxs
        data_dict['e2j_count'] = e2j_idxs.shape[0]
        
        data_dict['tot_obj_count'] = tot_object_points.shape[0]
        data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_obj_attr_feats
        data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_obj_edge_feats
        data_dict['tot_rel_pose'] = tot_rel_pose
        data_dict['edges'] = edges    

        data_dict['global_obj_ids'] = global_object_ids
        data_dict['scene_ids'] = [src_scan_id, ref_scan_id]        
        data_dict['pcl_center'] = pcl_center
        data_dict['overlap'] = overlap

        out_dict = {}
        out_dict['scene_name'] = src_scan_id.split('_')[0]
        out_dict['ref_frame'] = ref_scan_id
        out_dict['src_frame'] = src_scan_id
        out_dict['__key__'] = idx

        pts_all = data_dict['tot_obj_pts']
        src_obj_count, ref_obj_count = data_dict['graph_per_obj_count']

        sg_match = np.vstack((np.asarray([i - src_obj_count for i in e2i_idxs]), np.asarray([i for i in e1i_idxs]))).T
        out_dict['sg_match'] = torch.from_numpy(sg_match)
        
        '''if self.use_augmentation:
            rotation = np.identity(3)
            translation = np.zeros(3)
            ref_object_points, src_object_points, rotation, translation = self._augment_point_cloud(ref_object_points, src_object_points, rotation, translation, self.split)
            transform = get_transform_from_rotation_translation(rotation, translation)
        else: 
            transform = np.identity(4)'''
        transform = np.identity(4)
        x_t, x_q = data_dict['tot_bow_vec_object_attr_feats'][:src_obj_count,:], data_dict['tot_bow_vec_object_attr_feats'][src_obj_count:,:]
        node_edge_attr_t, node_edge_attr_q = tot_bow_vec_obj_edge_feats[:src_obj_count,:], tot_bow_vec_obj_edge_feats[src_obj_count:,:]
        ref_edges_attr = torch.concat([node_edge_attr_q[ref_edges[:,0], :], node_edge_attr_q[ref_edges[:,1],:]], dim=1)
        src_edges_attr = torch.concat([node_edge_attr_t[src_edges[:,0], :], node_edge_attr_t[src_edges[:,1],:]], dim=1)
        ref_edges = torch.from_numpy(ref_edges.T).long()
        src_edges = torch.from_numpy(src_edges.T).long()
        sg_pair = PairData(x_q=x_q.float(), edge_index_q=ref_edges, edge_attr_q=ref_edges_attr.float(), x_t=x_t.float(), edge_index_t=src_edges, edge_attr_t=src_edges_attr.float())
        out_dict['sg'] = sg_pair
              
        out_dict['ref_points'] = ref_object_points.astype(np.float32)
        out_dict['src_points'] = src_object_points.astype(np.float32)
        out_dict['transform'] = transform.astype(np.float32)
        out_dict['overlap'] = overlap

        return out_dict

    def _collate_entity_idxs(self, batch):
        e1i = np.concatenate([data['e1i'] for data in batch])
        e2i = np.concatenate([data['e2i'] for data in batch])
        e1j = np.concatenate([data['e1j'] for data in batch])
        e2j = np.concatenate([data['e2j'] for data in batch])

        e1i_start_idx = 0 
        e2i_start_idx = 0 
        e1j_start_idx = 0 
        e2j_start_idx = 0 
        prev_obj_cnt = 0
        
        for idx in range(len(batch)):
            e1i_end_idx = e1i_start_idx + batch[idx]['e1i_count']
            e2i_end_idx = e2i_start_idx + batch[idx]['e2i_count']
            e1j_end_idx = e1j_start_idx + batch[idx]['e1j_count']
            e2j_end_idx = e2j_start_idx + batch[idx]['e2j_count']

            e1i[e1i_start_idx : e1i_end_idx] += prev_obj_cnt
            e2i[e2i_start_idx : e2i_end_idx] += prev_obj_cnt
            e1j[e1j_start_idx : e1j_end_idx] += prev_obj_cnt
            e2j[e2j_start_idx : e2j_end_idx] += prev_obj_cnt
            
            e1i_start_idx, e2i_start_idx, e1j_start_idx, e2j_start_idx = e1i_end_idx, e2i_end_idx, e1j_end_idx, e2j_end_idx
            prev_obj_cnt += batch[idx]['tot_obj_count']
        
        e1i = e1i.astype(np.int32)
        e2i = e2i.astype(np.int32)
        e1j = e1j.astype(np.int32)
        e2j = e2j.astype(np.int32)

        return e1i, e2i, e1j, e2j

    def _collate_feats(self, batch, key):
        feats = torch.cat([data[key] for data in batch])
        return feats
    
    def collate_fn(self, batch):
        tot_object_points = self._collate_feats(batch, 'tot_obj_pts')
        tot_bow_vec_object_attr_feats = self._collate_feats(batch, 'tot_bow_vec_object_attr_feats')
        tot_bow_vec_object_edge_feats = self._collate_feats(batch, 'tot_bow_vec_object_edge_feats')    
        tot_rel_pose = self._collate_feats(batch, 'tot_rel_pose')
        
        data_dict = {}
        data_dict['tot_obj_pts'] = tot_object_points
        data_dict['e1i'], data_dict['e2i'], data_dict['e1j'], data_dict['e2j'] = self._collate_entity_idxs(batch)

        data_dict['e1i_count'] = np.stack([data['e1i_count'] for data in batch])
        data_dict['e2i_count'] = np.stack([data['e2i_count'] for data in batch])
        data_dict['e1j_count'] = np.stack([data['e1j_count'] for data in batch])
        data_dict['e2j_count'] = np.stack([data['e2j_count'] for data in batch])
        data_dict['tot_obj_count'] = np.stack([data['tot_obj_count'] for data in batch])
        data_dict['global_obj_ids'] = np.concatenate([data['global_obj_ids'] for data in batch])
        
        data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_object_attr_feats.double()
        data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_object_edge_feats.double()
        data_dict['tot_rel_pose'] = tot_rel_pose.double()
        data_dict['graph_per_obj_count'] = np.stack([data['graph_per_obj_count'] for data in batch])
        data_dict['graph_per_edge_count'] = np.stack([data['graph_per_edge_count'] for data in batch])
        data_dict['edges'] = self._collate_feats(batch, 'edges')
        data_dict['scene_ids'] = np.stack([data['scene_ids'] for data in batch])
        data_dict['obj_ids'] = np.concatenate([data['obj_ids'] for data in batch])
        data_dict['pcl_center'] = np.stack([data['pcl_center'] for data in batch])
        
        data_dict['overlap'] = np.stack([data['overlap'] for data in batch])
        data_dict['batch_size'] = data_dict['overlap'].shape[0]

        return data_dict


class Scan3RDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        use_augmentation=True,
        augmentation_noise=0.01,
        augmentation_rotation=1,
        augmentation_translation=0.5,
        debug=False,
        anchor_type_name='',
        predicted=False
        ):

        self.dataset_root = dataset_root
        self.split = split
        self.pc_resolution = 512
        self.anchor_type_name = anchor_type_name
        self.predicted = predicted
        #self.scans_dir = os.path.join(dataset_root)
        if self.predicted:
            self.scans_dir = os.path.join(dataset_root, 'predicted')
        else:
            self.scans_dir = os.path.join(dataset_root)
        self.scans_scenes_dir = os.path.join(dataset_root, 'scenes')
        self.scans_files_dir = os.path.join(self.scans_dir, 'files')

        self.subscans_dir = osp.join(self.scans_dir, 'out')
        #self.subscans_dir = osp.join(dataset_root,'predicted')
        self.subscans_scenes_dir = osp.join(self.subscans_dir, 'scenes')
        self.subscans_files_dir = osp.join(self.subscans_dir, 'files')

        self.mode = 'orig'#'node_removed'#'orig' #if self.split == 'train' else cfg.val.data_mode
        self.anchor_data_filename = os.path.join(self.subscans_files_dir, '{}/anchors{}_{}.json'.format(self.mode, self.anchor_type_name, split))

        self.anchor_data = common.load_json(self.anchor_data_filename)
        random.shuffle(self.anchor_data)
        self.is_training = self.split == 'train'

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.augmentation_translation = augmentation_translation
        self.predicted = False
        self.label_file_name = 'labels.instances.align.annotated.v2.ply' if not self.predicted else 'inseg_filtered.ply'

        self.debug = debug

    def __getitem__(self, idx):
        graph_data = self.anchor_data[idx]
        src_scan_id = graph_data['src']
        ref_scan_id = graph_data['ref']
        scan_id = src_scan_id[:src_scan_id.index('_')]
        overlap = graph_data['overlap']
        
        src_points, src_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(src_scan_id)), obj_ids = None, return_ply_data=True)
        ref_points, ref_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(ref_scan_id)), obj_ids = None, return_ply_data=True)
        #print(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(ref_scan_id)))
        if self.split == 'train':
            if np.random.rand(1)[0] > 0.5:
                pcl_center = np.mean(src_points, axis=0)
            else:
                pcl_center = np.mean(ref_points, axis=0)
        else:
            pcl_center = np.mean(src_points, axis=0)

        src_data_dict = common.load_pkl_data(osp.join(self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, src_scan_id)))
        ref_data_dict = common.load_pkl_data(osp.join(self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, ref_scan_id)))
        
        src_object_ids = src_data_dict['objects_id']
        ref_object_ids = ref_data_dict['objects_id']
        anchor_obj_ids = graph_data['anchorIds']
        global_object_ids = np.concatenate((src_data_dict['objects_cat'], ref_data_dict['objects_cat']))
        
        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id != 0]
        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id in src_object_ids and anchor_obj_id in ref_object_ids]
        
        src_edges = src_data_dict['edges']
        ref_edges = ref_data_dict['edges']

        src_object_points = src_data_dict['obj_points'][self.pc_resolution] - pcl_center
        ref_object_points = ref_data_dict['obj_points'][self.pc_resolution] - pcl_center

        edges = torch.cat([torch.from_numpy(src_edges), torch.from_numpy(ref_edges)])

        src_object_id2idx = src_data_dict['object_id2idx']
        e1i_idxs = np.array([src_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]) # e1i
        e1j_idxs = np.array([src_object_id2idx[object_id] for object_id in src_data_dict['objects_id'] if object_id not in anchor_obj_ids]) # e1j
        
        ref_object_id2idx = ref_data_dict['object_id2idx']
        e2i_idxs = np.array([ref_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]) + src_object_points.shape[0] # e2i
        e2j_idxs = np.array([ref_object_id2idx[object_id] for object_id in ref_data_dict['objects_id'] if object_id not in anchor_obj_ids]) + src_object_points.shape[0] # e2j

        # Get and clean inst map
        inst_t = np.zeros_like(src_plydata['objectId'])
        inst_q = np.zeros_like(ref_plydata['objectId'])
        mask_t = np.ones_like(src_plydata['objectId']).astype(np.bool_)
        mask_q = np.ones_like(ref_plydata['objectId']).astype(np.bool_)
        
        for obj_id in np.unique(src_plydata['objectId']):
            if obj_id in src_object_id2idx.keys():
                obj_idx = src_object_id2idx[obj_id]
                inst_t[src_plydata['objectId'] == obj_id] = obj_idx
            else:
                mask_t[src_plydata['objectId'] == obj_id] = False
        
        for obj_id in np.unique(ref_plydata['objectId']):
            if obj_id in ref_object_id2idx.keys():
                obj_idx = ref_object_id2idx[obj_id]
                inst_q[ref_plydata['objectId'] == obj_id] = obj_idx
            else:
                mask_q[ref_plydata['objectId'] == obj_id] = False
        
        tot_object_points = torch.cat([torch.from_numpy(src_object_points), torch.from_numpy(ref_object_points)]).type(torch.FloatTensor)
        #print(src_data_dict['objects_cat'])
        if self.predicted:
            zeros_to_pad = 164-41
            src_fake_object_attr = torch.from_numpy(src_data_dict['bow_vec_object_edge_feats'])
            ref_fake_object_attr = torch.from_numpy(ref_data_dict['bow_vec_object_edge_feats'])
            src_fake_object_attr = torch.cat([src_fake_object_attr, torch.zeros(src_fake_object_attr.size(0), zeros_to_pad)], dim=1)
            ref_fake_object_attr = torch.cat([ref_fake_object_attr, torch.zeros(ref_fake_object_attr.size(0), zeros_to_pad)], dim=1)
            tot_bow_vec_obj_attr_feats =  torch.cat([src_fake_object_attr, ref_fake_object_attr])
        else:
            tot_bow_vec_obj_attr_feats = torch.cat([torch.from_numpy(src_data_dict['bow_vec_object_attr_feats']), 
                                                torch.from_numpy(ref_data_dict['bow_vec_object_attr_feats'])])

        tot_bow_vec_obj_edge_feats = torch.cat([torch.from_numpy(src_data_dict['bow_vec_object_edge_feats']), 
                                                torch.from_numpy(ref_data_dict['bow_vec_object_edge_feats'])])
        tot_rel_pose = torch.cat([torch.from_numpy(src_data_dict['rel_trans']), torch.from_numpy(ref_data_dict['rel_trans'])])

        data_dict = {} 
        data_dict['obj_ids'] = np.concatenate([src_object_ids, ref_object_ids])
        data_dict['tot_obj_pts'] = tot_object_points
        data_dict['graph_per_obj_count'] = np.array([src_object_points.shape[0], ref_object_points.shape[0]])
        data_dict['graph_per_edge_count'] = np.array([src_edges.shape[0], ref_edges.shape[0]])
        
        data_dict['e1i'] = e1i_idxs
        data_dict['e1i_count'] = e1i_idxs.shape[0]
        data_dict['e2i'] = e2i_idxs
        data_dict['e2i_count'] = e2i_idxs.shape[0]
        data_dict['e1j'] = e1j_idxs
        data_dict['e1j_count'] = e1j_idxs.shape[0]
        data_dict['e2j'] = e2j_idxs
        data_dict['e2j_count'] = e2j_idxs.shape[0]
        
        data_dict['tot_obj_count'] = tot_object_points.shape[0]
        data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_obj_attr_feats
        data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_obj_edge_feats
        data_dict['tot_rel_pose'] = tot_rel_pose
        data_dict['edges'] = edges    

        data_dict['global_obj_ids'] = global_object_ids

        out_dict = {}
        out_dict['scene_name'] = src_scan_id.split('_')[0]
        out_dict['ref_frame'] = ref_scan_id
        out_dict['src_frame'] = src_scan_id
        out_dict['__key__'] = idx
        out_dict['overlap'] = overlap

        src_obj_count, ref_obj_count = src_object_points.shape[0], ref_object_points.shape[0]

        xyz_q_np = ref_points[mask_q]
        xyz_t_np = src_points[mask_t]
        inst_q = inst_q[mask_q]
        inst_t = inst_t[mask_t]
        
        sg_match = np.vstack((np.asarray([i - src_obj_count for i in e2i_idxs]), np.asarray([i for i in e1i_idxs]))).T
        out_dict['sg_match'] = torch.from_numpy(sg_match)

        if self.use_augmentation:
            rotation = np.identity(3)
            translation = np.zeros(3)
            xyz_q_np, xyz_t_np, rotation, translation = self._augment_point_cloud(xyz_q_np, xyz_t_np, rotation, translation, self.split)
            transform = get_transform_from_rotation_translation(rotation, translation)
        else: 
            transform = np.identity(4)
        
        x_t, x_q = data_dict['tot_bow_vec_object_attr_feats'][:src_obj_count,:], data_dict['tot_bow_vec_object_attr_feats'][src_obj_count:,:]
        node_edge_attr_t, node_edge_attr_q = tot_bow_vec_obj_edge_feats[:src_obj_count,:], tot_bow_vec_obj_edge_feats[src_obj_count:,:]
        ref_edges_attr = torch.concat([node_edge_attr_q[ref_edges[:,0], :], node_edge_attr_q[ref_edges[:,1],:]], dim=1)
        src_edges_attr = torch.concat([node_edge_attr_t[src_edges[:,0], :], node_edge_attr_t[src_edges[:,1],:]], dim=1)
        ref_edges = torch.from_numpy(ref_edges.T).long()
        src_edges = torch.from_numpy(src_edges.T).long()
        sg_pair = PairData(x_q=x_q.float(), edge_index_q=ref_edges, edge_attr_q=ref_edges_attr.float(), x_t=x_t.float(), edge_index_t=src_edges, edge_attr_t=src_edges_attr.float())
        out_dict['sg'] = sg_pair
        
        out_dict['ref_points'] = xyz_q_np.astype(np.float32)
        out_dict['src_points'] = xyz_t_np.astype(np.float32)

        out_dict['ref_insts'] = inst_q.astype(np.int32)
        out_dict['src_insts'] = inst_t.astype(np.int32)

        out_dict['ref_feats'] = np.ones((xyz_q_np.shape[0], 1), dtype=np.float32)
        out_dict['src_feats'] = np.ones((xyz_t_np.shape[0], 1), dtype=np.float32)
        out_dict['transform'] = transform.astype(np.float32)

        if self.split == 'val':
            ply_data = scan3r.load_ply_data(self.scans_scenes_dir, scan_id, self.label_file_name)
            scene_pts = np.stack((ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z'])).transpose()
            out_dict['raw_points'] = scene_pts.astype(np.float32)
            #out_dict['src_ply'] = src_plydata
            #out_dict['ref_ply'] = ref_plydata
        
        return out_dict
    
    def _augment_point_cloud(self, ref_points, src_points, rotation, translation, split):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        aug_translation = random_sample_translation()
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T) + aug_translation
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation) + aug_translation

        else:
            src_points = np.matmul(src_points + aug_translation, aug_rotation.T) 
            rotation = np.matmul(rotation, aug_rotation.T)
            translation =  - aug_translation#np.matmul(aug_rotation, aug_translation)

        if split == 'train':
            ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
            src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation
    
    def __len__(self):
        return len(self.anchor_data)


class Scan3RDataset_Dynamics(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        use_augmentation=False,
        augmentation_noise=0.01,
        augmentation_rotation=1,
        augmentation_translation=0.5,
        debug=False,
        anchor_type_name='subscan_to_refscan_changed'
        ):
        self.mode='train'
        self.dataset_root = dataset_root
        self.split = split
        self.pc_resolution = 512
        self.anchor_type_name = anchor_type_name
        self.scans_dir = os.path.join(dataset_root)
        self.scans_scenes_dir = os.path.join(self.scans_dir, 'scenes')
        self.scans_files_dir = os.path.join(self.scans_dir, 'files')

        self.subscans_dir = osp.join(dataset_root, 'out')
        self.subscans_scenes_dir = osp.join(self.subscans_dir, 'scenes')
        self.subscans_files_dir = osp.join(self.subscans_dir, 'files')

        self.mode = 'orig' #if self.split == 'train' else cfg.val.data_mode

        self.anchor_data_filename = os.path.join(self.subscans_files_dir, '{}/anchors_{}_{}.json'.format(self.mode, self.anchor_type_name, split))
        #self.anchor_data_filename = os.path.join(self.subscans_files_dir, '{}/anchors_{}.json'.format(self.mode, split))
        
        self.anchor_data = common.load_json(self.anchor_data_filename)
        random.shuffle(self.anchor_data)
        self.is_training = self.split == 'train'

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        self.augmentation_translation = augmentation_translation
        self.predicted = False
        self.label_file_name = 'labels.instances.align.annotated.v2.ply' if not self.predicted else 'inseg_filtered.ply'

        
        self.meta_file = common.load_json(os.path.join(self.scans_files_dir, '3RScan.json'))
        self.meta_dict = {}
        for meta_ele in self.meta_file:
            key = meta_ele['reference']
            self.meta_dict[key] = meta_ele
        
        self.rescan_data_filename = os.path.join(self.scans_files_dir, 'anchors_rescans_to_refscans_{}.json'.format(split))
        self.rescan_data = common.load_json(self.rescan_data_filename)
        rescan_ref_dict = {}
        for pair in self.rescan_data:
            rescan_ref_dict[pair['src']] = pair['ref']
            #rescan_dict[pair['ref']] = pair['src']
            #TODO: we should filter out the pairs from "rescan", the reference is the standard one.
        self.rescan_ref_dict = rescan_ref_dict
        self.debug = debug
        self.clear_anchor()
        print(self.new_anchor_data[0])
        print(len(self.anchor_data))

        self.filter_anchors = []
        self.used_src_scans = []
        
    
    def _augment_point_cloud(self, ref_points, src_points, rotation, translation, split):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        aug_translation = random_sample_translation()
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T) + aug_translation
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation) + aug_translation

        else:
            src_points = np.matmul(src_points + aug_translation, aug_rotation.T) 
            rotation = np.matmul(rotation, aug_rotation.T)
            translation =  - aug_translation#np.matmul(aug_rotation, aug_translation)

        if split == 'train':
            ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
            src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation
    
    def __len__(self):
        return len(self.anchor_data)
        #return len(self.new_anchor_data)
    def clear_anchor(self):
        self.new_anchor_data = []
        for i in range(len(self.anchor_data)):
            graph_data = self.anchor_data[i]
            src_scan_id = graph_data['src']
            scan_id = src_scan_id[:src_scan_id.index('_')]
            if scan_id in self.rescan_ref_dict.keys():
                rescan_id = self.rescan_ref_dict[scan_id]
                new_anchor_ele = {'src': src_scan_id, 'ref': rescan_id, 'overlap': 1.0}
                self.new_anchor_data.append(new_anchor_ele)
    
    def __getitem__(self, idx):
        graph_data = self.anchor_data[idx]

        '''src_scan_id = graph_data['ref']
        ref_scan_id = graph_data['src']
        scan_id = ref_scan_id[:ref_scan_id.index('_')]'''

        src_scan_id = graph_data['src']
        ref_scan_id = graph_data['ref']
        scan_id = src_scan_id[:src_scan_id.index('_')]

        overlap = graph_data['overlap']
        
        src_points, src_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(src_scan_id)), obj_ids = None, return_ply_data=True)
        ref_points, ref_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(ref_scan_id)), obj_ids = None, return_ply_data=True)
        
        if self.split == 'train':
            if np.random.rand(1)[0] > 0.5:
                pcl_center = np.mean(src_points, axis=0)
            else:
                pcl_center = np.mean(ref_points, axis=0)
        else:
            pcl_center = np.mean(src_points, axis=0)

        src_data_dict = common.load_pkl_data(osp.join(self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, src_scan_id)))
        ref_data_dict = common.load_pkl_data(osp.join(self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, ref_scan_id)))
        
        src_object_ids = src_data_dict['objects_id']
        ref_object_ids = ref_data_dict['objects_id']
        anchor_obj_ids = graph_data['anchorIds']
        global_object_ids = np.concatenate((src_data_dict['objects_cat'], ref_data_dict['objects_cat']))
        
        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id != 0]
        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id in src_object_ids and anchor_obj_id in ref_object_ids]
        
        src_edges = src_data_dict['edges']
        ref_edges = ref_data_dict['edges']

        src_object_points = src_data_dict['obj_points'][self.pc_resolution] - pcl_center
        ref_object_points = ref_data_dict['obj_points'][self.pc_resolution] - pcl_center

        edges = torch.cat([torch.from_numpy(src_edges), torch.from_numpy(ref_edges)])

        src_object_id2idx = src_data_dict['object_id2idx']
        e1i_idxs = np.array([src_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]) # e1i
        e1j_idxs = np.array([src_object_id2idx[object_id] for object_id in src_data_dict['objects_id'] if object_id not in anchor_obj_ids]) # e1j
        
        ref_object_id2idx = ref_data_dict['object_id2idx']
        e2i_idxs = np.array([ref_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]) + src_object_points.shape[0] # e2i
        e2j_idxs = np.array([ref_object_id2idx[object_id] for object_id in ref_data_dict['objects_id'] if object_id not in anchor_obj_ids]) + src_object_points.shape[0] # e2j
        
        # Get and clean inst map
        inst_t = np.zeros_like(src_plydata['objectId'])
        inst_q = np.zeros_like(ref_plydata['objectId'])
        mask_t = np.ones_like(src_plydata['objectId']).astype(np.bool_)
        mask_q = np.ones_like(ref_plydata['objectId']).astype(np.bool_)
        
        for obj_id in np.unique(src_plydata['objectId']):
            if obj_id in src_object_id2idx.keys():
                obj_idx = src_object_id2idx[obj_id]
                inst_t[src_plydata['objectId'] == obj_id] = obj_idx
            else:
                mask_t[src_plydata['objectId'] == obj_id] = False
        
        for obj_id in np.unique(ref_plydata['objectId']):
            if obj_id in ref_object_id2idx.keys():
                obj_idx = ref_object_id2idx[obj_id]
                inst_q[ref_plydata['objectId'] == obj_id] = obj_idx
            else:
                mask_q[ref_plydata['objectId'] == obj_id] = False
        
        tot_object_points = torch.cat([torch.from_numpy(src_object_points), torch.from_numpy(ref_object_points)]).type(torch.FloatTensor)
        tot_bow_vec_obj_attr_feats = torch.cat([torch.from_numpy(src_data_dict['bow_vec_object_attr_feats']), 
                                                torch.from_numpy(ref_data_dict['bow_vec_object_attr_feats'])])
        tot_bow_vec_obj_edge_feats = torch.cat([torch.from_numpy(src_data_dict['bow_vec_object_edge_feats']), 
                                                torch.from_numpy(ref_data_dict['bow_vec_object_edge_feats'])])
        tot_rel_pose = torch.cat([torch.from_numpy(src_data_dict['rel_trans']), torch.from_numpy(ref_data_dict['rel_trans'])])

        data_dict = {} 
        data_dict['obj_ids'] = np.concatenate([src_object_ids, ref_object_ids])
        data_dict['tot_obj_pts'] = tot_object_points
        data_dict['graph_per_obj_count'] = np.array([src_object_points.shape[0], ref_object_points.shape[0]])
        data_dict['graph_per_edge_count'] = np.array([src_edges.shape[0], ref_edges.shape[0]])
        
        data_dict['e1i'] = e1i_idxs
        data_dict['e1i_count'] = e1i_idxs.shape[0]
        data_dict['e2i'] = e2i_idxs
        data_dict['e2i_count'] = e2i_idxs.shape[0]
        data_dict['e1j'] = e1j_idxs
        data_dict['e1j_count'] = e1j_idxs.shape[0]
        data_dict['e2j'] = e2j_idxs
        data_dict['e2j_count'] = e2j_idxs.shape[0]
        
        data_dict['tot_obj_count'] = tot_object_points.shape[0]
        data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_obj_attr_feats
        data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_obj_edge_feats
        data_dict['tot_rel_pose'] = tot_rel_pose
        data_dict['edges'] = edges    

        data_dict['global_obj_ids'] = global_object_ids

        out_dict = {}
        out_dict['scene_name'] = src_scan_id.split('_')[0]
        out_dict['ref_frame'] = ref_scan_id
        out_dict['src_frame'] = src_scan_id
        out_dict['__key__'] = idx
        out_dict['overlap'] = overlap

        src_obj_count, ref_obj_count = src_object_points.shape[0], ref_object_points.shape[0]

        xyz_q_np = ref_points[mask_q]
        xyz_t_np = src_points[mask_t]
        inst_q = inst_q[mask_q]
        inst_t = inst_t[mask_t]
        
        sg_match = np.vstack((np.asarray([i - src_obj_count for i in e2i_idxs]), np.asarray([i for i in e1i_idxs]))).T
        out_dict['sg_match'] = torch.from_numpy(sg_match)

        if self.use_augmentation:
            rotation = np.identity(3)
            translation = np.zeros(3)
            xyz_q_np, xyz_t_np, rotation, translation = self._augment_point_cloud(xyz_q_np, xyz_t_np, rotation, translation, self.split)
            transform = get_transform_from_rotation_translation(rotation, translation)
        else: 
            transform = np.identity(4)
        
        x_t, x_q = data_dict['tot_bow_vec_object_attr_feats'][:src_obj_count,:], data_dict['tot_bow_vec_object_attr_feats'][src_obj_count:,:]
        node_edge_attr_t, node_edge_attr_q = tot_bow_vec_obj_edge_feats[:src_obj_count,:], tot_bow_vec_obj_edge_feats[src_obj_count:,:]
        ref_edges_attr = torch.concat([node_edge_attr_q[ref_edges[:,0], :], node_edge_attr_q[ref_edges[:,1],:]], dim=1)
        src_edges_attr = torch.concat([node_edge_attr_t[src_edges[:,0], :], node_edge_attr_t[src_edges[:,1],:]], dim=1)
        ref_edges = torch.from_numpy(ref_edges.T).long()
        src_edges = torch.from_numpy(src_edges.T).long()
        sg_pair = PairData(x_q=x_q.float(), edge_index_q=ref_edges, edge_attr_q=ref_edges_attr.float(), x_t=x_t.float(), edge_index_t=src_edges, edge_attr_t=src_edges_attr.float())
        out_dict['sg'] = sg_pair
        
        out_dict['ref_points'] = xyz_q_np.astype(np.float32)
        out_dict['src_points'] = xyz_t_np.astype(np.float32)

        out_dict['ref_insts'] = inst_q.astype(np.int32)
        out_dict['src_insts'] = inst_t.astype(np.int32)

        out_dict['ref_feats'] = np.ones((xyz_q_np.shape[0], 1), dtype=np.float32)
        out_dict['src_feats'] = np.ones((xyz_t_np.shape[0], 1), dtype=np.float32)
        out_dict['transform'] = transform.astype(np.float32)

        if self.split == 'val':
            ply_data = scan3r.load_ply_data(self.scans_scenes_dir, scan_id, self.label_file_name)
            scene_pts = np.stack((ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z'])).transpose()
            out_dict['raw_points'] = scene_pts.astype(np.float32)
        
        return out_dict
    
def get_datasets(args, cfg):
    if args.dataset=='sgm':
        train_set = Scan3RDataset(dataset_root=cfg.dataset.root, split='train')
        val_set = Scan3RDataset(dataset_root=cfg.dataset.root, split='val', use_augmentation=False)
    elif args.dataset=='sgm_pointnet':
        train_set = Scan3RDataset_pointnet(dataset_root=cfg.dataset.root, split='train')
        val_set = Scan3RDataset_pointnet(dataset_root=cfg.dataset.root, split='val', use_augmentation=False) 
    return train_set, val_set

if __name__ == "__main__":
    dataset_root = "/home/xie/Documents/datasets/3RScan/"
    #dataset = Scan3RDataset_pointnet(dataset_root=dataset_root, split='train')
    dataset = Scan3RDataset(dataset_root=dataset_root, split='val', use_augmentation=False)

    for sample in dataset:
        print(sample.keys())
        for key in sample.keys():
            if isinstance(sample[key], np.ndarray):
                print(key, sample[key].shape)
            else: print(key, type(sample[key]))
        break