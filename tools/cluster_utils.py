
import numpy as np
from sklearn.linear_model import RANSACRegressor
from third_party.OpenPCDet.tools.visual_utils.pcd_preprocess import *
from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V

REJECT={'too_few_pts': 1,
        'vol_too_big': 2,
        'vol_too_small': 3,
        'floating': 4,
        'below_ground': 5,
        'h_too_small': 6,
        'h_too_big': 7,
        'lw_ratio_off': 8}
def get_continuous_labels(labels):
    labels = labels.flatten()
    # Keep labels continuous
    label_mapping = sorted(list(set(labels)))
    label_mapping = {x:i for i, x in enumerate(label_mapping, -1)}
    
    for x, i in label_mapping.items():
        labels[labels==x] = i
    return labels

def distance_to_plane(ptc, plane, directional=False):
    d = ptc @ plane[:3] + plane[3]
    if not directional:
        d = np.abs(d)
    d /= np.sqrt((plane[:3]**2).sum())
    return d


def is_valid_cluster(
        ptc,
        ground_o3d, ground_tree,
        min_points=10,
        max_volume=70, #120
        min_volume=0.3,
        max_height_for_lowest_point=1,
        min_height_for_highest_point=0.5):
    
    if ptc.shape[0] < min_points:
        return False, REJECT['too_few_pts']
    volume = np.prod(ptc.max(axis=0) - ptc.min(axis=0))
    if max_volume is not None and volume > max_volume: # volume too big 
        return False, REJECT['vol_too_big']
    if volume < min_volume: # volume too small
        return False, REJECT['vol_too_small']
    
    if ground_o3d is not None:
        cluster_centroid = ptc[:,:3].mean(axis = 0)
        lowest_point_idx = np.argmin(ptc[:,2])
        lowest_pt = ptc[lowest_point_idx]
        
        # Method 1. project to the closest ground normal
        # dist_centroid_ground = np.linalg.norm(np.asarray(ground_o3d.points) - cluster_centroid.reshape((-1,3)), axis = 1)
        # closest_ground_pt_idx = np.argmin(dist_centroid_ground)
        # g_pt = ground_o3d.points[closest_ground_pt_idx]
        # g_normal = ground_o3d.normals[closest_ground_pt_idx]
        # signed_distance_to_ground = np.dot(ptc - g_pt, g_normal[...,np.newaxis])

        # Method 2. Subtract z values assuming flat ground i.e. normal of [0,0,1]
        [_, idx, _] = ground_tree.search_knn_vector_3d(lowest_pt, 20)
        #g_pt = np.asarray(ground_o3d.points)[idx[10]] #median of 20 closest points
        g_pt = np.asarray(ground_o3d.points)[idx].mean(axis = 0)
        signed_distance_to_ground = ptc[:,-1] - g_pt[-1] # subtract z value of closest ground pt

        if signed_distance_to_ground.min() > max_height_for_lowest_point: #if min distance to plane > 1m, object floating in air
            return False, REJECT['floating']
        if signed_distance_to_ground.max() < min_height_for_highest_point: #if max distance to plane < 0.5, object underground or too small
            return False, REJECT['below_ground']

    h = ptc[:,2].max() - ptc[:,2].min()
    # w = ptc[:,1].max() - ptc[:,1].min()
    # l = ptc[:,0].max() - ptc[:,0].min()
    # if height is not much, make this cluster background
    if h < 0.4:
        return False, REJECT['h_too_small']
    # # Remove walls
    # if h > 3:
    #     return False, REJECT['vol_too_big']
    # if l/w >= 4 or w/l >=4:
    #     return False, REJECT['lw_ratio_off']
    return True, 0

def filter_labels(
    pc,
    labels,
    max_volume = 70,
    min_volume = 0.3,
    max_height_for_lowest_point=1,
    min_height_for_highest_point=0.5,
    ground_mask = None
    ):
    
    labels = labels.flatten().copy()
    num_obj_labels = int(labels.max()+1)
    rejection_tag = np.zeros(int(num_obj_labels)) #rejection tag for labels 0,1, ... (exclude label -1)

    ground_o3d = None 
    ground_tree = None
    if ground_mask is not None:
        ground = pc[ground_mask,:3]
        ground_o3d = o3d.geometry.PointCloud()
        ground_o3d.points = o3d.utility.Vector3dVector(ground)
        # ground_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
        #                                                   max_nn=30))
        # ground_o3d.orient_normals_to_align_with_direction()
        ground_tree = o3d.geometry.KDTreeFlann(ground_o3d)


    for i in np.unique(labels):
        if i == -1:
            continue
        
        # if i == labels[178343] or i == labels[113573]:
        #     visualize_selected_labels(pc, labels, [i])
        is_valid, tag = is_valid_cluster(pc[labels == i, :3],
                                         ground_o3d, ground_tree, 
                                         max_volume = max_volume, min_volume=min_volume,
                                         max_height_for_lowest_point=max_height_for_lowest_point,
                                         min_height_for_highest_point=min_height_for_highest_point)
        if not is_valid:
            labels[labels == i] = -1 #set as background
            rejection_tag[int(i)] = tag
    
    return labels, rejection_tag

def cluster(xyz, non_ground_mask, eps=0.2):
    
    labels_non_ground = clusters_hdbscan(xyz[non_ground_mask], n_clusters=-1, eps=eps)[...,np.newaxis] #np.ones((non_ground_mask.sum(), 1))

    labels = np.ones(xyz.shape[0]) * -1

    labels[non_ground_mask] = labels_non_ground.flatten() #(N all points, 1)
    assert (labels.max() < np.finfo('float16').max), 'max segment id overflow float16 number'

    return labels


def save_labels(labels, path):
    labels = labels.astype(np.float16)
    labels.tofile(path)

def load_labels(path):
    labels = np.fromfile(path, dtype=np.float16)
    return labels