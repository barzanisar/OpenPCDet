import os
import sys
import open3d as o3d
import numpy as np
from third_party.OpenPCDet.tools.cluster_utils import *
from third_party.OpenPCDet.tools import pypatchworkpp


def get_plane_mask(ptc, plane, offset=0.05, only_range=None):
    mask = distance_to_plane(ptc, plane, directional=True) < offset #plane points mask
    if only_range is not None:
        range_mask = (ptc[:, 0] < only_range[0][1]) * (ptc[:, 0] > only_range[0][0]) * \
            (ptc[:, 1] < only_range[1][1]) * (ptc[:, 1] > only_range[1][0])
        mask *= range_mask
    return mask # returns mask of all non-plane points

def estimate_plane_RANSAC(origin_ptc, max_hs=0.05, it=1, ptc_range=((-70, 70), (-20, 20))):
    mask = np.ones(origin_ptc.shape[0], dtype=bool)
    if ptc_range is not None:
        mask = (origin_ptc[:, 2] < max_hs) & \
            (origin_ptc[:, 0] > np.min(ptc_range[0])) & \
            (origin_ptc[:, 0] < np.max(ptc_range[0])) & \
            (origin_ptc[:, 1] > np.min(ptc_range[1])) & \
            (origin_ptc[:, 1] < np.max(ptc_range[1]))
        # if max_hs is not None:
        #     if (origin_ptc[:, 2] < max_hs).sum() > 50:
        #         mask = mask & (origin_ptc[:, 2] < max_hs)
        if mask.sum() < 50: # if few valid points, don't estimate plane
            return None
    #for _ in range(it):
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
        # mask = get_plane_mask(
        #     origin_ptc[:, :3], result, offset=0.2, only_range=((-30,30), (-30,30))) # mask of all plane-points
    return result # minus [a,b,c,d] = -plane coeff

def visualize_ground(ground_o3d, nonground, centers=None, normals=None):
        # Visualize
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width = 600, height = 400)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    nonground_o3d = o3d.geometry.PointCloud()
    nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
    nonground_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[1.0, 0.0, 0.0] for _ in range(nonground.shape[0])], dtype=float) #RGB
    )


    vis.add_geometry(mesh)
    vis.add_geometry(ground_o3d)
    vis.add_geometry(nonground_o3d)

    if centers is not None:
        centers_o3d = o3d.geometry.PointCloud()
        centers_o3d.points = o3d.utility.Vector3dVector(centers)
        centers_o3d.normals = o3d.utility.Vector3dVector(normals)
        centers_o3d.colors = o3d.utility.Vector3dVector(
            np.array([[1.0, 1.0, 0.0] for _ in range(centers.shape[0])], dtype=float) #RGB
        )
        vis.add_geometry(centers_o3d)

    vis.run()
    vis.destroy_window()

def estimate_ground(pc, show_plots=False):
    # Patchwork++ initialization
    params = pypatchworkpp.Parameters()
    params.verbose = False
    params.enable_RNR = True
    params.enable_RVPF = True
    params.enable_TGR = True #impo

    params.num_iter = 3              # Number of iterations for ground plane estimation using PCA.
    params.num_lpr = 50              # Maximum number of points to be selected as lowest points representative.
    params.num_min_pts = 10          # Minimum number of points to be estimated as ground plane in each patch.
    params.num_zones = 4             # Setting of Concentric Zone Model(CZM)
    params.num_rings_of_interest = 4 # Number of rings to be checked with elevation and flatness values.

    params.RNR_ver_angle_thr = -15.0 # Noise points vertical angle threshold. Downward rays of LiDAR are more likely to generate severe noise points.
    params.RNR_intensity_thr = 0.2   # Noise points intensity threshold. The reflected points have relatively small intensity than others.

    params.sensor_height = 0                 
    params.th_seeds = 0.5                        # threshold for lowest point representatives using in initial seeds selection of ground points.
    params.th_dist = 0.125                       # threshold for thickenss of ground.
    params.th_seeds_v = 0.25                     # threshold for lowest point representatives using in initial seeds selection of vertical structural points.
    params.th_dist_v = 0.1                       # threshold for thickenss of vertical structure.
    params.max_range = 80.0                      # max_range of ground estimation area
    params.min_range = 0.5                       # min_range of ground estimation area
    params.uprightness_thr = 0.707               # threshold of uprightness using in Ground Likelihood Estimation(GLE). Please refer paper for more information about GLE.
    params.adaptive_seed_selection_margin = -1.2 # parameter using in initial seeds selection

    params.num_sectors_each_zone = [16, 32, 54, 32] # Setting of Concentric Zone Model(CZM)
    params.num_rings_each_zone = [2, 4, 4, 4]       # Setting of Concentric Zone Model(CZM)

    params.max_flatness_storage = 1000  # The maximum number of flatness storage
    params.max_elevation_storage = 1000 # The maximum number of elevation storage
    params.elevation_thr = [0, 0, 0, 0] # threshold of elevation for each ring using in GLE. Those values are updated adaptively.
    params.flatness_thr = [0, 0, 0, 0]  # threshold of flatness for each ring using in GLE. Those values are updated adaptively.

    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    # Estimate Ground
    PatchworkPLUSPLUS.estimateGround(pc)

    # Get Ground and Nonground
    ground      = PatchworkPLUSPLUS.getGround()
    nonground   = PatchworkPLUSPLUS.getNonground()
    ground_idx      = PatchworkPLUSPLUS.getGroundIndices()
    nonground_idx   = PatchworkPLUSPLUS.getNongroundIndices()
    centers     = PatchworkPLUSPLUS.getCenters()
    normals     = PatchworkPLUSPLUS.getNormals()

    # print('1. Patchwork')
    # visualize_ground(ground, nonground)

    ground_mask = np.zeros(pc.shape[0], dtype=bool)
    ground_mask[ground_idx] = True

    plane = estimate_plane_RANSAC(nonground, max_hs=0.5, ptc_range=((-20, 20), (-10, 10)))
    if plane is not None:
        ground_mask_for_nonground_pts = get_plane_mask(
            nonground, plane,
            offset=0.15,
            only_range=((-30, 30), (-20, 20)))

    ground_mask[nonground_idx[ground_mask_for_nonground_pts]] = True    
    ground = pc[ground_mask,:3]
    nonground =  pc[np.logical_not(ground_mask),:3]

    # ground = np.vstack([ground, nonground[ground_mask_for_nonground_pts]])
    # nonground = nonground[np.logical_not(ground_mask_for_nonground_pts)]
    # print('2. After removing close range ground')
    # visualize_ground(ground, nonground)

    ground_o3d = o3d.geometry.PointCloud()
    ground_o3d.points = o3d.utility.Vector3dVector(ground)
    ground_o3d.colors = o3d.utility.Vector3dVector(
        np.array([[0.0, 1.0, 0.0] for _ in range(ground.shape[0])], dtype=float) # RGB
    )
    ground_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                          max_nn=30))
    ground_o3d.orient_normals_to_align_with_direction()

    if show_plots:
        print('Showing Ground segmentation ...')
        #visualize_ground(pc[ground_mask,:3], pc[np.logical_not(ground_mask),:3], centers, normals)
        visualize_ground(ground_o3d, nonground)


    ground_tree = o3d.geometry.KDTreeFlann(ground_o3d)
    # [_, idx, _] = pcd_ground_tree.search_knn_vector_3d(np.array([2,2,2]), 1)
    # normal = normals[idx[0]] / np.linalg.norm(normals[idx[0]])
    # signed_distance = np.dot(normals[idx[0]], np.array([2,2,2]))




    return ground_mask, ground_o3d, ground_tree


def main():
    import glob
    seq_name = 'segment-10061305430875486848_1080_000_1100_000_with_camera_labels'
    pc_files = glob.glob(f'/home/barza/DepthContrast/data/waymo/waymo_processed_data_10_short/{seq_name}/*.npy')
    for file in pc_files:
        pc = np.load(file)
        pc[:,3] = np.tanh(pc[:, 3]) * 255.0
        
        print(file)
        
        estimate_ground(pc)

if __name__ == "__main__":
    main()
