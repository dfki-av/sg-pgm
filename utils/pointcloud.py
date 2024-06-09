# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import copy
import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional, Union, Any

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points

def get_nearest_neighbor(
    q_points: np.ndarray,
    s_points: np.ndarray,
    return_index: bool = False,
):
    r"""Compute the nearest neighbor for the query points in support points."""
    s_tree = cKDTree(s_points)
    distances, indices = s_tree.query(q_points, k=1)#, n_jobs=-1)
    if return_index:
        return distances, indices
    else:
        return distances

def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation

def make_open3d_point_cloud(xyz, color=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    pcd.estimate_normals()
    return pcd

def random_sample_rotation(rotation_factor: float = 1.0) -> np.ndarray:
    # angle_z, angle_y, angle_x
    euler = np.random.rand(3) * np.pi * 2 / rotation_factor  # (0, 2 * pi / rotation_range)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    return rotation

def random_sample_translation(translation_magnitude: float = 0.5):
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    return translation

def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Get rigid transform matrix from rotation matrix and translation vector.

    Args:
        rotation (array): (3, 3)
        translation (array): (3,)

    Returns:
        transform: (4, 4)
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform

def random_sample_transform(rotation_magnitude: float, translation_magnitude: float) -> np.ndarray:
    euler = np.random.rand(3) * np.pi * rotation_magnitude / 180.0  # (0, rot_mag)
    rotation = Rotation.from_euler('zyx', euler).as_matrix()
    translation = np.random.uniform(-translation_magnitude, translation_magnitude, 3)
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform

def compute_overlap(ref_points, src_points, transform=None, positive_radius=0.1):
    r"""Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances = get_nearest_neighbor(ref_points, src_points)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap

def get_dis(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def get_pt_box_aspect_ratio(pcd):
    box = pcd.get_oriented_bounding_box()
    pt8 = np.asarray(box.get_box_points())
    p1, p2, p3, p4, p5, p6, p7, p8 = pt8
    bwh = [get_dis(p1, p2), get_dis(p1, p3), get_dis(p1, p4)]
    return max(bwh)/min(bwh)

def draw_registration_result(source, target, transformation=None, recolor=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if transformation is not None:
        source_temp.transform(transformation)
    if recolor:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([target_temp, source_temp])

def draw_point_cloud(pcd, color=None):
    pcd = copy.deepcopy(pcd)
    if color is not None:
        pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd])

def remove_ceiling(points, ceiling_thr=0.2):
    points_mask = points[..., 2] < np.max(points[..., 2]) - ceiling_thr
    points = points[points_mask]
    return points

def draw_point_clouds_with_correspondences(pcd_q, pcd_t, corr, trans_=None, color=[0.1,0.9,0]):
    #pcd_q.paint_uniform_color([1, 0.706, 0])
    #pcd_t.paint_uniform_color([0, 0.651, 0.929])
    trans = trans_.copy()
    trans[1,3] = trans[1,3] + 1
    trans[2,3] = trans[2,3] + 1
    if trans is not None:
      pcd_q.transform(trans)
    
    #slide = np.eye(4,4).astype(np.float32)
    #slide[3,0] = 1.0
    #slide[3,1] = 1.0
    #pcd_q.transform(slide)
    line_set = o3d.geometry.LineSet()
    corr_set = line_set.create_from_point_cloud_correspondences(pcd_q, pcd_t, corr)
    corr_set.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd_q, pcd_t, corr_set])
    #o3d.visualization.draw_geometries([mesh_q, mesh_t, corr_set])

def draw_scene_graph_with_pcd(pts_, pcd_colors, inst, sg_edge, draw_pcd=True):
    
    insts_label = np.unique(inst)
    if -1 in insts_label:
        insts_label = insts_label[1:]
    geometry_list = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_)
    pts = np.array(pcd.points)
    centers = o3d.utility.Vector3dVector()

    for i, label in enumerate(insts_label):
        pcd_seg = o3d.geometry.PointCloud()
        pcd_seg.points = o3d.utility.Vector3dVector(pts[inst.squeeze()==label])
        center = pcd_seg.get_center()
        centers.append(center)
        sphere_node = o3d.geometry.TriangleMesh()
        sphere_node = sphere_node.create_sphere(radius=0.1)
        node_color = pcd_colors[inst.squeeze()==label].mean(axis=0)
        sphere_node.paint_uniform_color(node_color)
        sphere_node.translate(center)
        sphere_node.compute_vertex_normals()
        geometry_list.append(sphere_node)
    
    line_indices = o3d.utility.Vector2iVector(sg_edge.T)
    edge_set = o3d.geometry.LineSet(centers, line_indices)
    edge_set.paint_uniform_color([0.1,0.9,0])
    line_mesh1 = LineMesh(edge_set.points, lines=edge_set.lines,radius=0.0075)
    line_mesh1_geoms = line_mesh1.cylinder_segments
    geometry_list.extend(line_mesh1_geoms)

    if draw_pcd:
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        pcd.estimate_normals()
        geometry_list.append(pcd)

    o3d.visualization.draw_geometries(geometry_list)
    return 

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
                #cylinder_segment = cylinder_segment.rotate(
                   #axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)


def compute_registration_error(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transform (array): ground truth transformation matrix (4, 4)
        est_transform (array): estimated transformation matrix (4, 4)

    Returns:
        rre (float): relative rotation error.
        rte (float): relative translation error.
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    return rre, rte


def compute_rotation_mse_and_mae(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute anisotropic rotation error (MSE and MAE)."""
    gt_euler_angles = Rotation.from_dcm(gt_rotation).as_euler('xyz', degrees=True)  # (3,)
    est_euler_angles = Rotation.from_dcm(est_rotation).as_euler('xyz', degrees=True)  # (3,)
    mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
    mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
    return mse, mae


def compute_translation_mse_and_mae(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute anisotropic translation error (MSE and MAE)."""
    mse = np.mean((gt_translation - est_translation) ** 2)
    mae = np.mean(np.abs(gt_translation - est_translation))
    return mse, mae


def compute_transform_mse_and_mae(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute anisotropic rotation and translation error (MSE and MAE)."""
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    r_mse, r_mae = compute_rotation_mse_and_mae(gt_rotation, est_rotation)
    t_mse, t_mae = compute_translation_mse_and_mae(gt_translation, est_translation)
    return r_mse, r_mae, t_mse, t_mae


def compute_registration_rmse(src_points: np.ndarray, gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute re-alignment error (approximated RMSE in 3DMatch).

    Used in Rotated 3DMatch.

    Args:
        src_points (array): source point cloud. (N, 3)
        gt_transform (array): ground-truth transformation. (4, 4)
        est_transform (array): estimated transformation. (4, 4)

    Returns:
        error (float): root mean square error.
    """
    gt_points = apply_transform(src_points, gt_transform)
    est_points = apply_transform(src_points, est_transform)
    error = np.linalg.norm(gt_points - est_points, axis=1).mean()
    return error


def compute_modified_chamfer_distance(
    raw_points: np.ndarray,
    ref_points: np.ndarray,
    src_points: np.ndarray,
    gt_transform: np.ndarray,
    est_transform: np.ndarray,
):
    r"""Compute the modified chamfer distance (RPMNet)."""
    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, est_transform)
    chamfer_distance_p_q = get_nearest_neighbor(aligned_src_points, raw_points).mean()
    # Q -> P_raw
    composed_transform = np.matmul(est_transform, np.linalg.inv(gt_transform))
    aligned_raw_points = apply_transform(raw_points, composed_transform)
    chamfer_distance_q_p = get_nearest_neighbor(ref_points, aligned_raw_points).mean()
    # sum up
    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p
    return chamfer_distance


def compute_correspondence_residual(ref_corr_points, src_corr_points, transform):
    r"""Computing the mean distance between a set of correspondences."""
    src_corr_points = apply_transform(src_corr_points, transform)
    residuals = np.sqrt(((ref_corr_points - src_corr_points) ** 2).sum(1))
    mean_residual = np.mean(residuals)
    return mean_residual


def compute_inlier_ratio(ref_corr_points, src_corr_points, transform, positive_radius=0.1):
    r"""Computing the inlier ratio between a set of correspondences."""
    src_corr_points = apply_transform(src_corr_points, transform)
    residuals = np.sqrt(((ref_corr_points - src_corr_points) ** 2).sum(1))
    inlier_ratio = np.mean(residuals < positive_radius)
    return inlier_ratio


def compute_overlap(ref_points, src_points, transform=None, positive_radius=0.1):
    r"""Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances = get_nearest_neighbor(ref_points, src_points)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap


# Ground Truth Utilities


def get_correspondences(ref_points, src_points, transform, matching_radius):
    r"""Find the ground truth correspondences within the matching radius between two point clouds.

    Return correspondence indices [indices in ref_points, indices in src_points]
    """
    src_points = apply_transform(src_points, transform)
    src_tree = cKDTree(src_points)
    indices_list = src_tree.query_ball_point(ref_points, matching_radius)
    corr_indices = np.array(
        [(i, j) for i, indices in enumerate(indices_list) for j in indices],
        dtype=np.long,
    )
    return corr_indices


# Matching Utilities


def extract_corr_indices_from_feats(
    ref_feats: np.ndarray,
    src_feats: np.ndarray,
    mutual: bool = False,
    bilateral: bool = False,
):
    r"""Extract correspondence indices from features.

    Args:
        ref_feats (array): (N, C)
        src_feats (array): (M, C)
        mutual (bool = False): whether use mutual matching
        bilateral (bool = False): whether use bilateral non-mutual matching, ignored if `mutual` is True.

    Returns:
        ref_corr_indices: (M,)
        src_corr_indices: (M,)
    """
    ref_nn_indices = get_nearest_neighbor(ref_feats, src_feats, return_index=True)[1]
    if mutual or bilateral:
        src_nn_indices = get_nearest_neighbor(src_feats, ref_feats, return_index=True)[1]
        ref_indices = np.arange(ref_feats.shape[0])
        if mutual:
            ref_masks = np.equal(src_nn_indices[ref_nn_indices], ref_indices)
            ref_corr_indices = ref_indices[ref_masks]
            src_corr_indices = ref_nn_indices[ref_corr_indices]
        else:
            src_indices = np.arange(src_feats.shape[0])
            ref_corr_indices = np.concatenate([ref_indices, src_nn_indices], axis=0)
            src_corr_indices = np.concatenate([ref_nn_indices, src_indices], axis=0)
    else:
        ref_corr_indices = np.arange(ref_feats.shape[0])
        src_corr_indices = ref_nn_indices
    return ref_corr_indices, src_corr_indices


def extract_correspondences_from_feats(
    ref_points: np.ndarray,
    src_points: np.ndarray,
    ref_feats: np.ndarray,
    src_feats: np.ndarray,
    mutual: bool = False,
    return_feat_dist: bool = False,
):
    r"""Extract correspondences from features."""
    ref_corr_indices, src_corr_indices = extract_corr_indices_from_feats(ref_feats, src_feats, mutual=mutual)

    ref_corr_points = ref_points[ref_corr_indices]
    src_corr_points = src_points[src_corr_indices]
    outputs = [ref_corr_points, src_corr_points]
    if return_feat_dist:
        ref_corr_feats = ref_feats[ref_corr_indices]
        src_corr_feats = src_feats[src_corr_indices]
        feat_dists = np.linalg.norm(ref_corr_feats - src_corr_feats, axis=1)
        outputs.append(feat_dists)
    return outputs


# Evaluation Utilities


def evaluate_correspondences(ref_points, src_points, transform, positive_radius=0.1):
    overlap = compute_overlap(ref_points, src_points, transform, positive_radius=positive_radius)
    inlier_ratio = compute_inlier_ratio(ref_points, src_points, transform, positive_radius=positive_radius)
    residual = compute_correspondence_residual(ref_points, src_points, transform)

    return {
        'overlap': overlap,
        'inlier_ratio': inlier_ratio,
        'residual': residual,
        'num_corr': ref_points.shape[0],
    }


def evaluate_sparse_correspondences(ref_points, src_points, ref_corr_indices, src_corr_indices, gt_corr_indices):
    ref_gt_corr_indices = gt_corr_indices[:, 0]
    src_gt_corr_indices = gt_corr_indices[:, 1]

    gt_corr_mat = np.zeros((ref_points.shape[0], src_points.shape[0]))
    gt_corr_mat[ref_gt_corr_indices, src_gt_corr_indices] = 1.0
    num_gt_correspondences = gt_corr_mat.sum()

    pred_corr_mat = np.zeros_like(gt_corr_mat)
    pred_corr_mat[ref_corr_indices, src_corr_indices] = 1.0
    num_pred_correspondences = pred_corr_mat.sum()

    pos_corr_mat = gt_corr_mat * pred_corr_mat
    num_pos_correspondences = pos_corr_mat.sum()

    precision = num_pos_correspondences / (num_pred_correspondences + 1e-12)
    recall = num_pos_correspondences / (num_gt_correspondences + 1e-12)

    pos_corr_mat = pos_corr_mat > 0
    gt_corr_mat = gt_corr_mat > 0
    ref_hit_ratio = np.any(pos_corr_mat, axis=1).sum() / (np.any(gt_corr_mat, axis=1).sum() + 1e-12)
    src_hit_ratio = np.any(pos_corr_mat, axis=0).sum() / (np.any(gt_corr_mat, axis=0).sum() + 1e-12)
    hit_ratio = 0.5 * (ref_hit_ratio + src_hit_ratio)

    return {
        'precision': precision,
        'recall': recall,
        'hit_ratio': hit_ratio,
    }
