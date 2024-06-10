if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    sys.path.append('../../')

import os, random, torch, json, copy, glob
import os.path as osp
import re
import json
import numpy as np
import open3d as o3d
from torch_geometric.data import Data
from utils.pointcloud import *
from utils.util import read_classes, read_relationships, read_txt_to_list
from utils.dcputil import npmat2euler
from utils import common, scan3r
from utils.net_args import parse_args

class PairData(Data): # For packing target and query scene graph
    def __inc__(self, key, value, *args):
        if bool(re.search('index_q', key)):
            return self.x_q.size(0)
        if bool(re.search('index_t', key)):
            return self.x_t.size(0)
        else:
            return 0

class Gen3RScan_Dynamics(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        debug=False,
        dynamics='sub2rescan',
        anchor_type_name='subscan_to_refscan_changed',
        vis = False
        ):
        self.dataset_root = dataset_root
        self.split = split
        self.anchor_type_name = anchor_type_name
        self.dynamics = dynamics
        self.vis = vis

        self.scans_dir = os.path.join(dataset_root)
        self.scans_scenes_dir = os.path.join(self.scans_dir, 'scenes')
        self.scans_files_dir = os.path.join(self.scans_dir, 'files')

        self.subscans_dir = osp.join(dataset_root, 'out')
        self.subscans_scenes_dir = osp.join(self.subscans_dir, 'scenes')
        self.subscans_files_dir = osp.join(self.subscans_dir, 'files')

        self.mode = 'orig' #if self.split == 'train' else cfg.val.data_mode
        self.anchor_data_filename = os.path.join(self.subscans_files_dir, '{}/anchors_{}.json'.format(self.mode, split))
        self.anchor_data = common.load_json(self.anchor_data_filename)
        self.is_training = self.split == 'train'

        if self.dynamics == 'sub2rescan':
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
            self.rescan_ref_dict = rescan_ref_dict
            self.debug = debug
            self.clear_anchor()
            self.filter_anchors = []
            self.used_src_scans = []
            print("src2refscan", len(self.new_anchor_data))
            self.anchor_data = self.new_anchor_data
        elif self.dynamics == 'sub2scan':
            self.used_src_scans = []
            pass
        elif self.dynamics == 'sub2sub':
            self.all_scans = read_txt_to_list(os.path.join(self.subscans_files_dir, '{}/{}_scans_subscenes.txt'.format(self.mode, split)))
            self.used_src_scans = []
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
            self.rescan_ref_dict = rescan_ref_dict
            self.clear_anchor_sub2sub()
            self.anchor_data = self.new_anchor_data
            
    
    def __len__(self):
        return len(self.anchor_data)

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
    
    def clear_anchor_sub2sub(self):
        self.new_anchor_data = []
        for i in range(len(self.anchor_data)):
            graph_data = self.anchor_data[i]
            src_scan_id = graph_data['src']
            scan_id = src_scan_id[:src_scan_id.index('_')]
            if src_scan_id not in self.used_src_scans:
                self.used_src_scans.append(src_scan_id)
            else: continue
            if scan_id in self.rescan_ref_dict.keys():
                rescan_id = self.rescan_ref_dict[scan_id]
                refsubscan_ids = [sub_ref_id for sub_ref_id  in self.all_scans if rescan_id in sub_ref_id]
                for refsubscan_id in refsubscan_ids:
                    new_anchor_ele = {'src': src_scan_id, 'ref': refsubscan_id}
                    self.new_anchor_data.append(new_anchor_ele)
        print('sub2sub pairs', len(self.new_anchor_data))
    
    def who_has_refscan(self):
        new_anchor_data = []
        for i in range(len(self.anchor_data)):
            graph_data = self.anchor_data[i]
            src_scan_id = graph_data['src']
            scan_id = src_scan_id[:src_scan_id.index('_')]

    
    def __getitem__(self, index):
        if self.dynamics == 'sub2rescan':
            anchor_data = self.__getitem__sub2rescan(index)
        elif self.dynamics == 'sub2scan':
            anchor_data = self.__getitem__sub2scan(index)
        elif self.dynamics == 'sub2sub':
            anchor_data = self.__getitem__sub2sub(index)
        
        return anchor_data

    def __getitem__sub2scan(self, idx):
        graph_data = self.anchor_data[idx]
        src_scan_id = graph_data['src']
        ref_scan_id = graph_data['ref']
        #print(src_scan_id, self.used_src_scans)
        #TODO: wait, this used src scan is not right??
        if src_scan_id in self.used_src_scans:
            if ref_scan_id in self.used_src_scans:
                return None
            else:
                self.used_src_scans.append(ref_scan_id)
                src_scan_id = ref_scan_id
        else:
            self.used_src_scans.append(src_scan_id)

        scan_id = src_scan_id[:src_scan_id.index('_')]
        
        src_points, src_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(src_scan_id)), obj_ids = None, return_ply_data=True)
        scan_points, scan_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(scan_id)), obj_ids = None, return_ply_data=True)
        
        src_data_dict = common.load_pkl_data(osp.join(self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, src_scan_id)))
        scan_data_dict = common.load_pkl_data(osp.join(self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, scan_id)))
        
        src_object_ids = src_data_dict['objects_id']
        scan_object_ids = scan_data_dict['objects_id']

        new_anchor_data = {'src': src_scan_id, 
                           'ref': scan_id, 
                           'overlap': 1.0, 
                           'anchorIds': np.asarray(src_object_ids, dtype=np.int32).tolist()}

        unmatched_obj_ids = [id for id in scan_object_ids if id not in src_object_ids]
        if self.vis:
            unmatched_mask = np.zeros((scan_points.shape[0],1))
            for id in unmatched_obj_ids:
                unmatched_mask[scan_plydata['objectId'] == id] = 1
           
            unmatched_parts = scan_points[unmatched_mask.squeeze().astype(np.bool_),:]
            pcd1 = make_open3d_point_cloud(src_points)
            pcd2 = make_open3d_point_cloud(remove_ceiling(scan_points))
            pcd3 = make_open3d_point_cloud(remove_ceiling(unmatched_parts))
            pcd1.estimate_normals()
            pcd2.estimate_normals()
            pcd3.estimate_normals()
            pcd1.paint_uniform_color([1, 0.706, 0])
            pcd2.paint_uniform_color([0, 0.651, 0.929])
            pcd3.paint_uniform_color([0,1,0])
            o3d.visualization.draw_geometries([pcd2, pcd1, pcd3])

        return new_anchor_data
      
    def __getitem__sub2rescan(self, idx):
        graph_data = self.new_anchor_data[idx]
        src_scan_id = graph_data['src']
        if src_scan_id in self.used_src_scans:
            return None
        else:
            self.used_src_scans.append(src_scan_id)
        rescan_ref_id = graph_data['ref']
        scan_id = src_scan_id[:src_scan_id.index('_')]
        
        rescans = self.meta_dict[rescan_ref_id]['scans']
        rescan_ids = [rescans[i]['reference'] for i in range(len(rescans))]
        rescan_trans = [rescans[i]['transform'] for i in range(len(rescans))]
        rescan_index = rescan_ids.index(scan_id)
        target_rescan = rescans[rescan_index]
        
        
        src_points, src_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(src_scan_id)), obj_ids = None, return_ply_data=True)
        rescan_points, rescan_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(rescan_ref_id)), obj_ids = None, return_ply_data=True)
        
        src_data_dict = common.load_pkl_data(osp.join(self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, src_scan_id)))
        ref_data_dict = common.load_pkl_data(osp.join(self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, rescan_ref_id)))
        
        src_object_ids = src_data_dict['objects_id']
        ref_object_ids = ref_data_dict['objects_id']

        rescan_trans = np.asarray(rescan_trans[rescan_index]).reshape(4,4)
        no_big_trans_item = []
        for i, rigid_item in enumerate(target_rescan['rigid']):
            item_transform = np.asarray(rigid_item['transform']).reshape(4,4)[np.newaxis,:,:]
            item_recovered_rot = np.matmul(item_transform[:,:3,:3], rescan_trans[:3,:3])
            item_recovered_rot_euler = npmat2euler(item_recovered_rot[:,:3,:3])
            if np.abs(item_recovered_rot_euler).sum() < 3:
                no_big_trans_item.append(target_rescan['rigid'][i]['instance_reference'])

        changed_obj_ids = [rigid_item['instance_reference'] for rigid_item in target_rescan['rigid'] if rigid_item['instance_reference'] not in no_big_trans_item]  \
                            + target_rescan['removed'] #+ target_rescan['nonrigid']
        
        changed_obj_ids_local = []
        anchor_obj_ids = []
        for ids in src_object_ids:
            if ids in changed_obj_ids:
                changed_obj_ids_local.append(ids)
            else:
                anchor_obj_ids.append(ids)

        if len(changed_obj_ids_local) > 0 and len(anchor_obj_ids) > 0:
            filtered_anchor = {'src': src_scan_id, 'ref': rescan_ref_id, 'overlap': float(1), 
                               'anchorIds': np.asarray(anchor_obj_ids, dtype=np.int32).tolist(), 
                               'changedIds': np.asarray(changed_obj_ids_local, dtype=np.int32).tolist()}
        else: return None

        
        if self.vis:
            print('removed', target_rescan['removed'])
            print('rigid', [rigid_item['instance_reference'] for rigid_item in target_rescan['rigid'] if rigid_item['instance_reference'] not in no_big_trans_item])
            print('picked', changed_obj_ids_local)
            print('common', anchor_obj_ids)
            print('src sub', src_object_ids)
            print('ref', ref_object_ids)
            changes_mask_rigid = np.zeros((rescan_points.shape[0],1))
            for changed_obj_id in [rigid_item['instance_reference'] for rigid_item in target_rescan['rigid'] if rigid_item['instance_reference'] not in no_big_trans_item]:
                changes_mask_rigid[rescan_plydata['objectId'] == changed_obj_id] = 1

            src_points, src_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, '{}/data.npy'.format(scan_id)), obj_ids = None, return_ply_data=True)
            changes_mask_removed = np.zeros((src_points.shape[0],1))
            for changed_obj_id in (target_rescan['removed']):
                changes_mask_removed[src_plydata['objectId'] == changed_obj_id] = 1
            
            pcd1 = make_open3d_point_cloud(remove_ceiling(src_points, 1))
            pcd2 = make_open3d_point_cloud(remove_ceiling(rescan_points, 1))
            changed_parts_removed = src_points[changes_mask_removed.squeeze().astype(np.bool_),:]
            changed_parts_rigid = rescan_points[changes_mask_rigid.squeeze().astype(np.bool_),:]
            pcd3 = make_open3d_point_cloud(changed_parts_removed)
            pcd4 = make_open3d_point_cloud(changed_parts_rigid)
            pcd1.estimate_normals()
            pcd2.estimate_normals()
            pcd3.estimate_normals()
            pcd4.estimate_normals()
            pcd1.paint_uniform_color([1, 0.706, 0])
            pcd2.paint_uniform_color([0, 0.651, 0.929])
            pcd3.paint_uniform_color([1,0,0])
            pcd4.paint_uniform_color([0,1,0])
            # reference scan: blue, src sub-scan: yellow, changed obj: green, chanage obj loacl: red
            o3d.visualization.draw_geometries([pcd2, pcd1, pcd4, pcd3])
            # remove object is the object that is removed in the reference scene, which is there in the src scene
        
        return filtered_anchor

    def __getitem__sub2sub(self, idx):
        graph_data = self.anchor_data[idx]
        src_scan_id = graph_data['src']
        ref_scan_id = graph_data['ref']
        scan_id = src_scan_id[:src_scan_id.index('_')]
        rescan_ref_id = ref_scan_id[:ref_scan_id.index('_')]
        rescans = self.meta_dict[rescan_ref_id]['scans']
        rescan_ids = [rescans[i]['reference'] for i in range(len(rescans))]
        rescan_trans = [rescans[i]['transform'] for i in range(len(rescans))]
        rescan_index = rescan_ids.index(scan_id)
        target_rescan = rescans[rescan_index]
        
        src_points, src_plydata = scan3r.load_plydata_npy(osp.join(
            self.subscans_scenes_dir, '{}/data.npy'.format(src_scan_id)), obj_ids = None, return_ply_data=True)
        ref_points, ref_plydata = scan3r.load_plydata_npy(osp.join(
            self.subscans_scenes_dir,'{}/data.npy'.format(ref_scan_id)), obj_ids = None, return_ply_data=True)

        src_data_dict = common.load_pkl_data(osp.join(
            self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, src_scan_id)))
        ref_data_dict = common.load_pkl_data(osp.join(
            self.subscans_files_dir, '{}/data/{}.pkl'.format(self.mode, ref_scan_id)))
        
        src_object_ids = src_data_dict['objects_id']
        ref_object_ids = ref_data_dict['objects_id']

        rescan_trans = np.asarray(rescan_trans[rescan_index]).reshape(4,4)
        no_big_trans_item = []
        for i, rigid_item in enumerate(target_rescan['rigid']):
            item_transform = np.asarray(rigid_item['transform']).reshape(4,4)[np.newaxis,:,:]
            item_recovered_rot = np.matmul(item_transform[:,:3,:3], rescan_trans[:3,:3])
            item_recovered_rot_euler = npmat2euler(item_recovered_rot[:,:3,:3])
            if np.abs(item_recovered_rot_euler).sum() < 3:
                no_big_trans_item.append(target_rescan['rigid'][i]['instance_reference'])

        rigid_changed_ids = [rigid_item['instance_reference'] for rigid_item in target_rescan['rigid'] if rigid_item['instance_reference'] not in no_big_trans_item]
        removed_ids = target_rescan['removed']
        
        changed_obj_ids_local = []
        anchor_obj_ids = []
        for rigid_changed_id in rigid_changed_ids:
            if rigid_changed_id in src_object_ids and rigid_changed_id in ref_object_ids:
                changed_obj_ids_local.append(rigid_changed_id)
        
        for removed_id in removed_ids:
            if removed_id in src_object_ids or removed_id in ref_object_ids:
                changed_obj_ids_local.append(removed_id)

        anchor_obj_ids = [src_id for src_id in src_object_ids if src_id in ref_object_ids]

        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id not in changed_obj_ids_local]


        if len(changed_obj_ids_local) == 0 or len(anchor_obj_ids)==0:
            return None
        
        overlap = compute_overlap(src_points, ref_points, np.identity(4), positive_radius=0.05)

        print("overlap", overlap)
        if overlap < 0.1 or overlap > 0.9:
            return None
        filtered_anchor = {'src': src_scan_id, 'ref': ref_scan_id, 'overlap': float(overlap), 
                               'anchorIds': np.asarray(anchor_obj_ids, dtype=np.int32).tolist(), 
                               'changedIds': np.asarray(changed_obj_ids_local, dtype=np.int32).tolist()}
        return filtered_anchor
    
if __name__ == "__main__":

    import torch.utils.data
    import json

    DYNAMICS = ['sub2rescan', 'sub2scan', 'sub2sub']
    ANCHOR_TYPE_NAME = ["subscan_to_refscan_changed", "subscan_to_scan", "subscan_to_subscan_changed"]

    args, cfg = parse_args() 
    
    dataset = Gen3RScan_Dynamics(dataset_root=cfg.dataset.root, split=args.split, dynamics=args.dynamics, vis=args.vis)
    anchor_type_name = ANCHOR_TYPE_NAME[DYNAMICS.index(args.dynamics)]
    new_anchor_data = []

    print("Dataset created, starting sampling...")
    print()
    for idx, sample in enumerate(dataset):
        print('\r print("Starting sampling...[{}/{}]")'.format(idx+1, dataset.__len__()), end='')
        if sample is not None:
            new_anchor_data.append(sample)
    print()
    print("Generated data for {} with {} samples. ".format(args.dynamics, len(new_anchor_data)))
    
    new_anchor_data_filename = os.path.join(dataset.subscans_files_dir, 
                                            '{}/anchors_{}_{}.json'.format(dataset.mode, anchor_type_name, args.split))
    common.write_json(new_anchor_data, new_anchor_data_filename)
    print(new_anchor_data_filename)


