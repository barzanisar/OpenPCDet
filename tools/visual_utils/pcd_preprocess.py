import numpy as np
import open3d as o3d
import hdbscan
import matplotlib.pyplot as plt
from third_party.OpenPCDet.tools.visual_utils.mouse_and_point_coord import vis_mouse_and_point_coord


np.random.seed(100)
def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


def overlap_clusters(cluster_i, cluster_j, min_cluster_point=20):
    # get unique labels from pcd_i and pcd_j
    unique_i = np.unique(cluster_i)
    unique_j = np.unique(cluster_j)

    # get labels present on both pcd (intersection)
    unique_ij = np.intersect1d(unique_i, unique_j)[1:]

    # also remove clusters with few points
    for cluster in unique_ij.copy():
        ind_i = np.where(cluster_i == cluster)
        ind_j = np.where(cluster_j == cluster)

        if len(ind_i[0]) < min_cluster_point or len(ind_j[0]) < min_cluster_point:
            unique_ij = np.delete(unique_ij, unique_ij == cluster)
        
    # labels not intersecting both pcd are assigned as -1 (unlabeled)
    cluster_i[np.in1d(cluster_i, unique_ij, invert=True)] = -1
    cluster_j[np.in1d(cluster_j, unique_ij, invert=True)] = -1

    return cluster_i, cluster_j

def clusters_hdbscan(points_set, n_clusters, eps=0.2):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None,
                                cluster_selection_method='eom', cluster_selection_epsilon=eps,
                            core_dist_n_jobs=1) #cluster_selection_epsilon=0.05, 0.07 also work

    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    if n_clusters == -1:
        clusters_labels = cluster_info[::-1][:, 0]
    else:
        clusters_labels = cluster_info[::-1][:n_clusters, 0]
    
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusters_from_pcd(pcd, n_clusters, eps=0.25):
    # clusterize pcd points
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=20))
    lbls, counts = np.unique(labels, return_counts=True)
    num_clusters_found = lbls.shape[0]
    if num_clusters_found > 1:
        cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
        cluster_info = cluster_info[cluster_info[:,1].argsort()]

        clusters_labels = cluster_info[::-1][:n_clusters, 0]
        labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels, num_clusters_found

def clusterize_pcd(points, n_clusters, dist_thresh=0.25, eps=0.5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # segment plane (ground)
    _, inliers = pcd.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=200)

    # # vis plane
    # inlier_cloud = pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    pcd_ = pcd.select_by_index(inliers, invert=True)

    l, num_clusters_found = clusters_from_pcd(pcd_, n_clusters, eps)

    labels_ = np.expand_dims(l, axis=-1)

    # that is a blessing of array handling
    # pcd are an ordered list of points
    # in a list [a, b, c, d, e] if we get the ordered indices [1, 3]
    # we will get [b, d], however if we get ~[1, 3] we will get the opposite indices
    # still ordered, i.e., [a, c, e] which means listing the inliers indices and getting
    # the invert we will get the outliers ordered indices (a sort of indirect indices mapping)
    labels = np.ones((points.shape[0], 1)) * -1
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[inliers] = False

    labels[mask] = labels_

    return np.concatenate((points, labels), axis=-1), num_clusters_found

def vis_mode(geometries, mode=""):
    if mode == 'pick_points':
        # print("")
        # print("1) Please pick at least three correspondences using [shift + left click]")
        # print("   Press [shift + right click] to undo point picking")
        # print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        #vis.get_render_option().line_width = 10.0
        for geometry in geometries:
            vis.add_geometry(geometry)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()
    elif mode == 'mouse_and_point_coord':
        vis_mouse_and_point_coord(geometries)
        return None
    else:
        o3d.visualization.draw_geometries(geometries)
        return None

def visualize_pcd_clusters(points, labels, boxes=None, mode='pick_points'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    
    colors = np.zeros((len(labels), 4))
    flat_indices = np.unique(labels[:,-1])
    max_instance = len(flat_indices)
    colors_instance = plt.get_cmap("prism")(np.arange(len(flat_indices)) / (max_instance if max_instance > 0 else 1))

    for idx in range(len(flat_indices)):
        colors[labels[:,-1] == flat_indices[int(idx)]] = colors_instance[int(idx)]

    colors[labels[:,-1] == -1] = [0.,0.,0.,0.]

    pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
    visuals = [pcd]

    if boxes is not None:
        for i in range(boxes.shape[0]):
            line_set, box3d = translate_boxes_to_open3d_instance(boxes[i])
            line_set.paint_uniform_color((0, 1, 0))
            visuals.append(line_set)

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    visuals.append(axis_pcd)
    picked_pts = vis_mode(visuals, mode)
    if picked_pts is not None:
        for i in picked_pts:
            print(f"{i}: Point picked: {points[i]}, with label: {labels[i]}")
    return picked_pts


def visualize_selected_labels(pc, labels, selected, boxes=None, mode='pick_points'):
    selected_labels_mask = np.zeros(labels.shape[0], dtype = bool)
    lbls_to_show = -1 *np.ones(labels.shape[0]) 
    for i in selected:
        selected_labels_mask[labels==i] = True
    lbls_to_show[selected_labels_mask] = labels[selected_labels_mask]
    picked_pts = visualize_pcd_clusters(pc[:,:3], lbls_to_show.reshape((-1,1)), boxes, mode)
    if picked_pts is not None:
        for i in picked_pts:
            print(f"{i}: Point picked: {pc[i]}, with label: {labels[i]}")
    return picked_pts  


def visualize_pcd_clusters_compare(point_set, pi, pj):
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(point_set[:,:3])

    pi[:,-1], pj[:,-1] = overlap_clusters(pi[:,-1], pj[:,-1])
    point_set[:,-1], pi[:,-1] = overlap_clusters(point_set[:,-1], pi[:,-1])

    labels = point_set[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd_.colors = o3d.utility.Vector3dVector(np.zeros_like(colors[:, :3]))
    o3d.visualization.draw_geometries([pcd_])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set[:,:3])

    labels = point_set[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

    pcd_i = o3d.geometry.PointCloud()
    pcd_i.points = o3d.utility.Vector3dVector(pi[:,:3])

    labels = pi[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd_i.colors = o3d.utility.Vector3dVector(np.zeros_like(colors[:, :3]))
    o3d.visualization.draw_geometries([pcd_i])
    pcd_i.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd_i])

    pcd_j = o3d.geometry.PointCloud()
    pcd_j.points = o3d.utility.Vector3dVector(pj[:,:3])

    labels = pj[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd_j.colors = o3d.utility.Vector3dVector(np.zeros_like(colors[:, :3]))
    o3d.visualization.draw_geometries([pcd_j])
    pcd_j.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd_j])

    # pcd_2 = o3d.geometry.PointCloud()
    # point_set_[:,2] += 10.
    # pcd_2.points = o3d.utility.Vector3dVector(point_set_[:,:3])

    # labels = point_set_[:, -1]
    # import matplotlib.pyplot as plt
    # colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    # colors[labels < 0] = 0

    # pcd_2.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([pcd_i])
    o3d.visualization.draw_geometries([pcd_j])
    #return pcd_
