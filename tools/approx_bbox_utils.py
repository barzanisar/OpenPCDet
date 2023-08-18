import numpy as np
import sklearn
from scipy.spatial import ConvexHull
import open3d as o3d
import matplotlib.pyplot as plt
#https://logicatcore.github.io/scratchpad/lidar/sensor-fusion/jupyter/2021/04/20/3D-Oriented-Bounding-Box.html
#https://github.com/logicatcore/scratchpad/blob/master/_notebooks/2021-04-20-2D-Oriented-Bounding-Box.ipynb
#https://colab.research.google.com/github/logicatcore/scratchpad/blob/master/_notebooks/2021-04-20-3D-Oriented-Bounding-Box.ipynb
def draw2DRectangle(ax, rectangleCoordinates, color, label=None):
    # diagonal line
    # plt.plot([x1, x2], [y1, y2], linestyle='dashed')
    # four sides of the rectangle
    ax.plot(rectangleCoordinates[0, 0:2], rectangleCoordinates[1, 0:2], color=color, label=label) # | (up)
    ax.plot(rectangleCoordinates[0, 1:3], rectangleCoordinates[1, 1:3], color=color) # -->
    ax.plot(rectangleCoordinates[0, 2:], rectangleCoordinates[1, 2:], color=color)    # | (down)
    ax.plot([rectangleCoordinates[0, 3], rectangleCoordinates[0, 0]], [rectangleCoordinates[1, 3], rectangleCoordinates[1, 0]], color=color)    # <--

def get_box_corners(cxyz, lwh, heading):
    #box: [xyz,lwh,rz]
    rot = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,0,1]])
    
    box3d = o3d.geometry.OrientedBoundingBox(cxyz, rot(heading), lwh)
    corners = np.asarray(box3d.get_box_points()) # 8 x 3
    bev_corners = np.array([corners[1,:2],
                            corners[0,:2],
                            corners[2,:2],
                            corners[7,:2]]) # 4 x 2
    
    """ open3d corners returned from new_corners = np.asarray(box3d.get_box_points()) # 8 x 3
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0

          ||0-1|| is length
          0-2 is width
          0-3 is height

          0th corner = 

          rval = np.array([
        [max_x, min_y], -> corner 1
        [min_x, min_y], -> corner 0
        [min_x, max_y], -> corner 2
        [max_x, max_y], -> corner 7
            ])

        """
    return bev_corners
    
def refine_box(anchor, cx, cy, cz, l,w,h, heading, max_l, max_w):
    rot = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,0,1]])
    cxcy = np.array([[cx], [cy]])
    anchor = anchor.reshape((-1,1))

    Rot_lidar_from_cc =  rot(heading)[:2,:2]
    Rot_cc_from_lidar = Rot_lidar_from_cc.T
    center_in_cc = Rot_cc_from_lidar @ (cxcy - anchor)
    sign_center_in_cc = np.sign(center_in_cc)

    new_center_in_cc= np.array([center_in_cc[0] + sign_center_in_cc[0] * 0.5*(max_l - l), 
                                    center_in_cc[1] + sign_center_in_cc[1] * 0.5*(max_w - w)])

    new_cxy_in_lidar = Rot_lidar_from_cc @ new_center_in_cc + anchor# cxy
    
    # Get new corners from center, lwh and rot
    cxyz = np.array([new_cxy_in_lidar[0,0], new_cxy_in_lidar[1,0], cz])
    lwh = np.array([max_l, max_w, h])

    bev_corners = get_box_corners(cxyz, lwh, heading)
    
    return cxyz, lwh, heading, bev_corners

def refine_boxes(boxes_this_label, max_l, max_w):
    refined_boxes = np.empty((0,15))
    num_boxes = boxes_this_label.shape[0]
    corners = boxes_this_label[:, 7:15]
    corners = corners.reshape((num_boxes,-1,2)) #(num boxes, 4, 2)
    norms = np.linalg.norm(corners, axis = -1, keepdims=True) #(numboxes, 4, 1)
    closest_corners_ind = np.argmin(norms, axis=1) #(numboxes, 1)
    #closest_corners = corners[:,closest_corners_ind, :][:,0,:,:]

    for i in range(num_boxes):
        closest_corner_ind = closest_corners_ind[i]
        closest_corner = corners[i, closest_corner_ind, :]

        cxyz, lwh, heading, bev_corners = refine_box(anchor=closest_corner, 
                                                     cx=boxes_this_label[i, 0], cy=boxes_this_label[i, 1], cz=boxes_this_label[i, 2], 
                                                     l=boxes_this_label[i, 3], w=boxes_this_label[i, 4], h=boxes_this_label[i, 5], 
                                                     heading=boxes_this_label[i, 6], max_l=max_l, max_w=max_w)
       
        box3d_full = np.concatenate((cxyz, lwh, np.array([heading]), bev_corners.flatten()))
        refined_boxes = np.vstack([refined_boxes, box3d_full])
        
    return refined_boxes

def test_refine():
    center = np.array([3.5, -2.5, 1])
    u = np.sqrt(0.5**2 + 0.5**2)
    lwh = np.array([4*u, 2*u, 1])
    heading = np.deg2rad(45)
    max_l = 6*u
    max_w = 3*u

    # Get corners from center, lwh and rot
    bev_corners = get_box_corners(center, lwh, heading)

    anchor=bev_corners[3]

    new_cxyz, new_lwh, heading, new_bev_corners = refine_box(anchor=anchor, 
               cx=center[0], cy=center[1], cz=center[2], 
               l=lwh[0],w=lwh[1],h=lwh[2], heading=heading, max_l=max_l, max_w=max_w)
    
    #visualize
    fig=plt.figure()
    ax=fig.add_subplot(111) #,projection='3d'
    ax.scatter(anchor[0], anchor[1], color='r', label='anchor')
    ax.scatter(center[0], center[1], marker='x', color='k')
    ax.scatter(new_cxyz[0], new_cxyz[1], marker='x', color='m')
    draw2DRectangle(ax, bev_corners.T, color='k', label='old')
    draw2DRectangle(ax, new_bev_corners.T, color='m', label='new')
    ax.legend()
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval, angles[best_idx], areas[best_idx]

def naive_rectangle(cluster_ptc):
    min_x = cluster_ptc[:,0].min()
    max_x = cluster_ptc[:,0].max()
    min_y = cluster_ptc[:,1].min()
    max_y = cluster_ptc[:,1].max()
    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])

    angle=0.0

    return rval, angle, area


def PCA_rectangle(cluster_ptc): #Nx2
    components = sklearn.decomposition.PCA(
        n_components=2).fit(cluster_ptc).components_
    on_component_ptc = cluster_ptc @ components.T #components 0th row is 1st component i.e. has max eval: [[x_1st, y_1st], [x_2nd, y_2nd]]
    min_x, max_x = on_component_ptc[:, 0].min(), on_component_ptc[:, 0].max()
    min_y, max_y = on_component_ptc[:, 1].min(), on_component_ptc[:, 1].max()
    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    angle = np.arctan2(components[0, 1], components[0, 0])
    return rval, angle, area

def closeness_rectangle(cluster_ptc, delta=0.1, d0=1e-2):
    max_beta = -float('inf')
    choose_angle = None
    for angle in np.arange(0, 90+delta, delta): #from 0 to 90 deg, step 0.1
        angle = angle / 180. * np.pi # convert to rad
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ]) #rectangles orthogonal edge directions e1 = [np.cos(angle), np.sin(angle)], e2=[-np.sin(angle), np.cos(angle)]
        projection = cluster_ptc @ components.T #project points to the rectangle's edges Nx2 i.e. C1 = Nx1 and C2 = Nx2
        min_x, max_x = projection[:,0].min(), projection[:,0].max() #boundaries of projections along axis e1 i.e. min/max projection len
        min_y, max_y = projection[:,1].min(), projection[:,1].max() #boundaries of projections along axis e2 i.e. min/max projection len
        Dx = np.vstack((projection[:, 0] - min_x, max_x - projection[:, 0])).min(axis=0) # distance of all point projections to closest corner/boundary of e1
        Dy = np.vstack((projection[:, 1] - min_y, max_y - projection[:, 1])).min(axis=0) # distance of all point projections to closest corner/boundary of e2
        beta = np.vstack((Dx, Dy)).min(axis=0) #smallest distance between projected point and closest rectangle's edge
        beta = np.maximum(beta, d0) 
        beta = 1 / beta # closeness score
        beta = beta.sum()
        if beta > max_beta:
            max_beta = beta
            choose_angle = angle
    angle = choose_angle
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ]) # corners of rectangle in e1 and e2 i.e. rectangle edges frame i.e e.vectors frame
    rval = rval @ components
    return rval, angle, area


def variance_rectangle(cluster_ptc, delta=0.1):
    max_var = -float('inf')
    choose_angle = None
    for angle in np.arange(0, 90+delta, delta):
        angle = angle / 180. * np.pi
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
        Dx = np.vstack((projection[:, 0] - min_x,
                       max_x - projection[:, 0])).min(axis=0)
        Dy = np.vstack((projection[:, 1] - min_y,
                       max_y - projection[:, 1])).min(axis=0)
        Ex = Dx[Dx < Dy]
        Ey = Dy[Dy < Dx]
        var = 0
        if (Dx < Dy).sum() > 0:
            var += -np.var(Ex)
        if (Dy < Dx).sum() > 0:
            var += -np.var(Ey)
        # print(angle, var)
        if var > max_var:
            max_var = var
            choose_angle = angle
    # print(choose_angle, max_var)
    angle = choose_angle
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ])
    rval = rval @ components
    return rval, angle, area

def get_highest_and_lowest_point_rect(pc, xy_center, l, w, rz):
    pc_xy_from_center_in_lidar_frame = pc[:, [0, 1]] - xy_center #vectors from center to points in xz cam axis
    rot_lidar_from_center = np.array([
        [np.cos(rz), -np.sin(rz)],
        [np.sin(rz), np.cos(rz)]
    ]) # rot of center x axis wrt lidar x axis i.e. lidar_from_objcenterframe
    pc_xy_from_center_in_center_frame = (rot_lidar_from_center.T @ pc_xy_from_center_in_lidar_frame.T).T
    mask = (pc_xy_from_center_in_center_frame[:, 0] > -l/2) & \
        (pc_xy_from_center_in_center_frame[:, 0] < l/2) & \
        (pc_xy_from_center_in_center_frame[:, 1] > -w/2) & \
        (pc_xy_from_center_in_center_frame[:, 1] < w/2)
    z_in_rect = pc[mask, 2]
    return z_in_rect.max(), z_in_rect.min() # top point of the cluster i.e. max y corrdinate in cam frame

def fit_box(ptc, fit_method='closeness_to_edge', full_pc = None):
    if fit_method == 'min_zx_area_fit':
        corners, rz, area = minimum_bounding_rectangle(ptc[:, [0, 1]])
    elif fit_method == 'PCA':
        corners, rz, area = PCA_rectangle(ptc[:, [0, 1]])
    elif fit_method == 'variance_to_edge':
        corners, rz, area = variance_rectangle(ptc[:, [0, 1]])
    elif fit_method == 'closeness_to_edge':
        corners, rz, area = closeness_rectangle(ptc[:, [0, 1]]) #corners in lidar x and y and heading of box's x axis wrt lidar x axis
    elif fit_method == 'naive_min_max':
        corners, rz, area = naive_rectangle(ptc[:, [0, 1]])

    else:
        raise NotImplementedError(fit_method)
    l = np.linalg.norm(corners[0] - corners[1])
    w = np.linalg.norm(corners[0] - corners[-1])
    cxy = 0.5*(corners[0] + corners[2]) # center in xy lidar axis

    if full_pc is not None:
        z_max_in_rect, _ = get_highest_and_lowest_point_rect(full_pc, cxy, l, w, rz)
        h = z_max_in_rect -  ptc[:, 2].min()
        cz = 0.5*(z_max_in_rect +  ptc[:, 2].min())
    else:
        h = ptc[:, 2].max() - ptc[:, 2].min()
        cz = 0.5*(ptc[:, 2].max() + ptc[:, 2].min())

    box = np.array([cxy[0], cxy[1], cz, l, w, h, rz])
    return box, corners, area #corners 4x2