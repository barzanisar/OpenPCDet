import numpy as np
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

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

class PubTracker(object):
  def __init__(self, max_age=0, max_dist=1, matcher = 'greedy'):
    self.max_age = max_age

    #self.WAYMO_CLS_VELOCITY_ERROR = max_dist 
    self.max_dist_thresh = max_dist # 1m

    self.reset()

    self.matcher_func = MATCHER[matcher]
  
  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, results, time_lag):
    if len(results) == 0:
      self.tracks = []
      return []
    else:
      temp = []
      for det in results:

        det['ct'] = np.array(det['translation'][:2]) # curr det box center x and center y in world frame
        temp.append(det)

      results = temp

    N = len(results) # current detections
    M = len(self.tracks) # previous tracked detections

    # dets: N X 2 center xy of all curr frame dets (in world frame)
    dets = np.array([det['ct'] for det in results], np.float32) 

    #max_diff = np.array([self.WAYMO_CLS_VELOCITY_ERROR[box['detection_name']] for box in results], np.float32) # N max acceptable diff for curr detections

    # move tprev detections center xy forward to curr time t to match with curr det  
    tracks = np.array(
      [pre_det['ct'] + pre_det['velocity']*time_lag for pre_det in self.tracks], np.float32) # M x 2 previous detections cx and cy in the prev frame (at time t-1)

    if len(tracks) > 0:  # NOT FIRST FRAME
      dist = (((tracks.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N=500 x M=19     (19 tracks ct - 500 det ct)**2 -> (500, 19, 2) -> sum -> (500, 19)
      dist = np.sqrt(dist) # absolute distance in meter (500, 19) distance matrix

      invalid = dist > self.max_dist_thresh #dist > max_diff.reshape(N, 1) #dist (500 dets, 19 tracks) > (500 max dist, 1) -> (500, 19) matrix with true values if dist is bigger 
      dist = dist  + invalid * 1e18 # fill invalid matches with inf dist
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
      track = results[m[0]] # matched det
      track['tracking_id'] = self.tracks[m[1]]['tracking_id'] # matched track's id      
      track['age'] = 1 
      track['active'] = self.tracks[m[1]]['active'] + 1 # number of frames this obj has been consecutively seen so far
      track['velocity'] = (track['ct'] - self.tracks[m[1]]['ct'])/time_lag
      ret.append(track)

    for i in unmatched_dets: # Add unmatched high conf dets as new tracks using new tracking id 
      track = results[i]
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
