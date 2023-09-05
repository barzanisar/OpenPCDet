import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment
from third_party.OpenPCDet.tools import approx_bbox_utils

def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0: # if num tracks = M == 0
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]): # loop over N curr det
    j = dist[i].argmin() # jth track has min dist with ith detection
    if dist[i][j] < 1e16: # if the min dist is below inf
      dist[:, j] = 1e18 # fill all other det's dist with this j track to be inf i.e. this jth track cannot be assigned to any other detection
      matched_indices.append([i, j]) #(ith det, jth track matched)
  return np.array(matched_indices, np.int32).reshape(-1, 2)

def hungarian_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0: # if num tracks = M == 0
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    # Solve the assignment problem
    row_indices, col_indices = linear_sum_assignment(dist)
    for row, col in zip(row_indices, col_indices):
      if dist[row][col] < 1e16:
        matched_indices.append([row, col])

    return np.array(matched_indices, np.int32).reshape(-1, 2)

MATCHER = {'greedy': greedy_assignment, 'hungarian': hungarian_assignment}


def visualize_tracks_3d(visualize, tracks, dets, invalid, dist, time_lag, matches=[]):
  pc_last = visualize['last_pc']
  pc_curr = visualize['curr_pc']
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(pc_last[:,0], pc_last[:,1], pc_last[:,2], color='k', label="pc_last")
  ax.scatter(pc_curr[:,0], pc_curr[:,1], pc_curr[:,2], color='r', label="pc_curr")
  for i, track in enumerate(tracks):
    #Plot prev bbox in tracks
    approx_bbox_utils.draw3Dbox(ax, track['corners3d'], color='k', label='prev_tracked_boxes' if i == 0 else None)
    
    # plot predicted ct of tracks
    prev_ct = track['ct']
    predicted_ct = prev_ct + track['velocity']*time_lag
    ax.plot([prev_ct[0], predicted_ct[0]], [prev_ct[1], predicted_ct[1]], '--k')
  

  for i, det in enumerate(dets):
    #Plot curr dets
    approx_bbox_utils.draw3Dbox(ax, det['corners3d'], color='g', label='curr_dets' if i == 0 else None)
    
  #plot valid distances between predicted ct of tracks and curr det's ct
  for det_i in range(invalid.shape[0]):
    valid_tracks_idx_for_det_i = np.logical_not(invalid).nonzero()[0]
    # if matches is not None and det_i in matches[:, 0]:
    #   idx, = np.where(matches[:, 0] == det_i)
    #   matched_track_j = matches[idx, 1]
    for track_j in valid_tracks_idx_for_det_i:
      track = tracks[track_j]
      predicted_ct = track['ct'] + track['velocity']*time_lag
      det = dets[det_i]

      # if matches is not None and track_j == matched_track_j:
      #   ax.plot([predicted_ct[0], det['ct'][0]], [predicted_ct[1], det['ct'][1]], '--r', linewidth = 1)
      # else:
      ax.plot([predicted_ct[0], det['ct'][0]], [predicted_ct[1], det['ct'][1]], '--g')
      text_x = 0.5 * (predicted_ct[0] + det['ct'][0])
      text_y = 0.5 * (predicted_ct[1] + det['ct'][1])
      text_z = 0.5 * (track['translation'][2] + det['translation'][2])
      ax.text(text_x, text_y, text_z, dist[det_i, track_j], color='k', fontsize = 10, bbox=dict(facecolor='yellow', alpha=0.5))

  for m in matches:
    det = dets[m[0]]
    track = tracks[m[1]]
    predicted_ct = track['ct'] + track['velocity']*time_lag
    ax.plot([predicted_ct[0], det['ct'][0]], [predicted_ct[1], det['ct'][1]], '--r', linewidth = 4)
    text_x = 0.5 * (predicted_ct[0] + det['ct'][0])
    text_y = 0.5 * (predicted_ct[1] + det['ct'][1])
    text_z = 0.5 * (track['translation'][2] + det['translation'][2])
    ax.text(text_x, text_y, text_z, dist[m[0], m[1]], color='k', fontsize = 10, bbox=dict(facecolor='yellow', alpha=0.5))

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.legend()
  plt.show()


class PubTracker(object):
  def __init__(self, max_age=0, max_dist=1.5, matcher = 'greedy'):
    self.max_age = max_age

    #self.WAYMO_CLS_VELOCITY_ERROR = max_dist 
    self.max_dist_thresh = max_dist # 1m

    self.reset()

    self.matcher_func = MATCHER[matcher]
  
  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, dets_to_track, time_lag, visualize=None):
    if len(dets_to_track) == 0:
      self.tracks = []
      return []
    else:
      temp = []
      for det in dets_to_track:

        det['ct'] = np.array(det['translation'][:2]) # curr det box center x and center y in world frame
        temp.append(det)

      dets_to_track = temp

    N = len(dets_to_track) # current detections
    M = len(self.tracks) # previous tracked detections

    # dets: N X 2 center xy of all curr frame dets (in world frame)
    dets = np.array([det['ct'] for det in dets_to_track], np.float32) 

    max_diff = np.array([min(1.5, 3 * max(box['dxdydz'][0], box['dxdydz'][1])) for box in dets_to_track], np.float32) # N max acceptable diff for curr detections

    # move tprev detections center xy forward to curr time t to match with curr det  
    tracks = np.array(
      [pre_det['ct'] + pre_det['velocity']*time_lag for pre_det in self.tracks], np.float32) # M x 2 previous detections cx and cy in the prev frame (at time t-1)

    if len(tracks) > 0:  # NOT FIRST FRAME
      # Visualize current and prev pc in world frame
      # visualize curr det boxes in w (dark green) and self.tracks boxes in world frame (black)
      # visualize projections of tracks in new frame (dotted black arrow from black box to pred cxyz)
      # Visualize valid det boxes for all tracks and colour invalid boxes with yellow
      # draw solid green line connecting valid dets and predicted cxcy location of tracks, with distance annotated
      dist = (((tracks.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N=500 dets x M=19 tracks    (19 tracks ct - 500 det ct)**2 -> (500, 19, 2) -> sum -> (500, 19)
      dist = np.sqrt(dist) # absolute distance in meter (500, 19) distance matrix

      invalid = dist > max_diff.reshape(N, 1) # dist > self.max_dist_thresh #dist (500 dets, 19 tracks) > (500 max dist, 1) -> (500, 19) matrix with true values if dist is bigger 
      dist = dist  + invalid * 1e18 # fill invalid matches with inf dist
      if visualize is not None:
        visualize_tracks_3d(visualize, self.tracks, dets_to_track)
      
      matched_indices = self.matcher_func(copy.deepcopy(dist)) #greedy_assignment(copy.deepcopy(dist))
    else:  # first few frame
      assert M == 0
      matched_indices = np.array([], np.int32).reshape(-1, 2) #(0,2) matched index in curr dets, matched index in tracks

    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]

    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]
    
    matches = matched_indices

    ret = []
    for m in matches: # Add matched dets as existing tracks using old tracking id
      track = dets_to_track[m[0]] # matched det
      track['tracking_id'] = self.tracks[m[1]]['tracking_id'] # matched track's id      
      track['age'] = 1 
      track['active'] = self.tracks[m[1]]['active'] + 1 # number of frames this obj has been consecutively seen so far
      track['velocity'] = (dets_to_track[m[0]]['ct'] - self.tracks[m[1]]['ct'])/time_lag
      ret.append(track)

    for i in unmatched_dets: # Add unmatched high conf dets as new tracks using new tracking id 
      track = dets_to_track[i]
      # initialize tracks with high condfidence pred boxes not matched with previous tracks
      self.id_count += 1
      track['tracking_id'] = self.id_count
      track['age'] = 1
      track['active'] =  1
      track['velocity'] = np.zeros(2)
      ret.append(track)

    # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
    # the object in current frame 
    for i in unmatched_tracks:
      track = self.tracks[i]
      if track['age'] < self.max_age:
        track['age'] += 1
        track['active'] = 0
        ct = track['ct']

        # movement in the last second
        # move forward  (i.e. predict ct in curr frame t so that detections in frame t+1 can match with these tracks )
        track['ct'] = ct + track['velocity']*time_lag 
        ret.append(track)

    self.tracks = ret
    return ret
