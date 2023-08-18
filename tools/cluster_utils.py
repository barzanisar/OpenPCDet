
import numpy as np
from sklearn.linear_model import RANSACRegressor
from visual_utils.pcd_preprocess import *
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

def above_plane(ptc, plane, offset=0.05, only_range=((-30, 30), (-30, 30))):
    mask = distance_to_plane(ptc, plane, directional=True) < offset
    # if only_range is not None:
    #     range_mask = (ptc[:, 0] < only_range[0][1]) * (ptc[:, 0] > only_range[0][0]) * \
    #         (ptc[:, 1] < only_range[1][1]) * (ptc[:, 1] > only_range[1][0])
    #     mask *= range_mask
    return np.logical_not(mask) # returns mask of all non-plane points

def estimate_plane(origin_ptc, max_hs=0.05, it=1, ptc_range=((-70, 70), (-20, 20))):
    mask = np.ones(origin_ptc.shape[0], dtype=bool)
    if ptc_range is not None:
        mask = (origin_ptc[:, 0] > np.min(ptc_range[0])) & \
            (origin_ptc[:, 0] < np.max(ptc_range[0])) & \
            (origin_ptc[:, 1] > np.min(ptc_range[1])) & \
            (origin_ptc[:, 1] < np.max(ptc_range[1]))
        if max_hs is not None:
            if (origin_ptc[:, 2] < max_hs).sum() > 50:
                mask = mask & (origin_ptc[:, 2] < max_hs)
        if mask.sum() < 50: # if few valid points, don't estimate plane
            return None
    for _ in range(it):
        ptc = origin_ptc[mask]
        reg = RANSACRegressor().fit(ptc[:, [0, 1]], ptc[:, 2])
        w = np.zeros(3)
        w[0] = reg.estimator_.coef_[0]
        w[1] = reg.estimator_.coef_[1]
        w[2] = -1.0
        h = reg.estimator_.intercept_
        norm = np.linalg.norm(w)
        w /= norm
        h = h / norm
        result = np.array((w[0], w[1], w[2], h))
        result *= -1
        mask = np.logical_not(above_plane(
            origin_ptc[:, :3], result, offset=0.2)) # mask of all plane-points
    return result # minus [a,b,c,d] = -plane coeff

def is_valid_cluster(
        ptc,
        plane=None,
        min_points=10,
        max_volume=70, #120
        min_volume=0.3,
        max_min_height=1,
        min_max_height=0.5):
    if ptc.shape[0] < min_points:
        return False, REJECT['too_few_pts']
    volume = np.prod(ptc.max(axis=0) - ptc.min(axis=0))
    if volume > max_volume: # volume too big 
        #V.draw_scenes(ptc)
        return False, REJECT['vol_too_big']
    if volume < min_volume: # volume too small
        #V.draw_scenes(ptc)
        return False, REJECT['vol_too_small']
    
    if plane is not None:
        distance_to_ground = distance_to_plane(ptc, plane, directional=True) #signed distance to ground
        if distance_to_ground.min() > max_min_height: #if min distance to plane > 1m, object floating in air
            #V.draw_scenes(ptc)
            return False, REJECT['floating']
        if distance_to_ground.max() < min_max_height: #if max distance to plane < 0.5, object underground or too small
            #V.draw_scenes(ptc)
            return False, REJECT['below_ground']
    h = ptc[:,2].max() - ptc[:,2].min()
    w = ptc[:,1].max() - ptc[:,1].min()
    l = ptc[:,0].max() - ptc[:,0].min()
    # if height is not much, make this cluster background
    if h < 0.4:
        return False, REJECT['h_too_small']
    # Remove walls
    if h > 3:
        return False, REJECT['vol_too_big']
    if l/w >= 4 or w/l >=4:
        return False, REJECT['lw_ratio_off']
    return True, 0

def filter_labels(
    pc,
    labels,
    num_obj_labels,
    max_volume = 70,
    estimate_dist_to_plane = True
    ):
    
    labels = labels.flatten().copy()
    rejection_tag = np.zeros((int(num_obj_labels), 1)) #rejection tag for labels 0,1, ... (exclude label -1)
    plane = None
    if estimate_dist_to_plane:
        plane = estimate_plane(pc, max_hs=0.05, ptc_range=((-70, 70), (-30, 30)))
    # plane = estimate_plane(pc[ground_mask], max_hs=0.05, ptc_range=((-70, 70), (-30, 30)))
    # above_plane_mask1 = above_plane(
    #                     pc[:,:3], plane1,
    #                     offset=0.1,
    #                     only_range=None)
    # above_plane_mask = above_plane(
    #                     pc[:,:3], plane,
    #                     offset=0.1,
    #                     only_range=None)
                
    # V.draw_scenes(pc[:,:3][above_plane_mask1])
    # V.draw_scenes(pc[:,:3][above_plane_mask])
    for i in np.unique(labels): #range(int(labels.max())+1):
        if i == -1:
            continue
        # if estimate_dist_to_plane:
        #     cluster_center = pc[labels == i, :3].mean(axis=0)
        #     ptc_range = ((cluster_center[0] + 20, cluster_center[0] - 20), (cluster_center[1] - 10, cluster_center[1] + 10))
        #     plane = estimate_plane(pc, max_hs=0.05, ptc_range=ptc_range)

        is_valid, tag = is_valid_cluster(pc[labels == i, :3], plane, max_volume = max_volume)
        if not is_valid:
            labels[labels == i] = -1 #set as background
            rejection_tag[int(i)] = tag
    
    return labels, rejection_tag

def cluster(xyz, non_ground_mask):
    
    labels_non_ground = clusters_hdbscan(xyz[non_ground_mask], n_clusters=-1)[...,np.newaxis] #np.ones((non_ground_mask.sum(), 1))

    labels = np.ones((xyz.shape[0], 1)) * -1

    labels[non_ground_mask] = labels_non_ground #(N all points, 1)

    return labels

def visualize_selected_labels(pc, labels, selected):
    selected_labels_mask = np.zeros(labels.shape[0], dtype = bool)
    lbls_to_show = -1 *np.ones(labels.shape[0]) 
    for i in selected:
        selected_labels_mask[labels==i] = True
    lbls_to_show[selected_labels_mask] = labels[selected_labels_mask]
    visualize_pcd_clusters(pc[:,:3], lbls_to_show.reshape((-1,1)))