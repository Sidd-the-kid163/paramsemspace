"""
Standalone motioncode extraction script.
Extracts the 208-d motioncode vector from joint coordinates.

Input:  coords tensor of shape (num_frames, num_joints, 3)
        - The last 2 "joints" should be orientation (Euler angles in degrees)
          and translation (xyz in meters), appended by your data loader.

Output: A flat 208-d vector (Option B: pairwise slots duplicated under both joints).
        -1 means no motion detected for that slot.

To add/remove categories:
  - Edit POSECODE_OPERATORS_VALUES / MOTIONCODE_OPERATORS_VALUES for thresholds
  - Edit ALL_ELEMENTARY_POSECODES / ALL_ELEMENTARY_MOTIONCODES for joint sets
  - The pipeline auto-adapts to whatever is defined there.
"""

import math, copy, json, os, shutil
import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################################################################
# SINGLE PATH FINDER (from MS_Algorithms.py)
################################################################################
def single_path_finder(time_series_signal, threshold=1):
    for i in range(len(time_series_signal)):
        if time_series_signal[i] > 0: time_series_signal[i] = 1
        if time_series_signal[i] < 0: time_series_signal[i] = -1
    start_i, end_i = 0, 0
    last_few_neg, last_few_pos = [], []
    Direction = 0
    positions = time_series_signal
    Final_output = []
    while start_i < len(positions) - 1:
        start_i += 1
        diff = positions[start_i]
        if diff < 0: last_few_neg.append(start_i); Direction -= 1
        elif diff > 0: last_few_pos.append(start_i); Direction += 1
        if abs(Direction) > threshold:
            if Direction > 0:
                if len(last_few_pos) == 0: pass
                elif len(last_few_pos) < threshold + 1: start_i = last_few_pos[0]
                else: start_i = last_few_pos[-(threshold + 1)]
            if Direction < 0:
                if len(last_few_neg) == 0: pass
                elif len(last_few_neg) < threshold + 1: start_i = last_few_neg[0]
                else: start_i = last_few_neg[-(threshold + 1)]
            max_prev = {'start': start_i, 'end': start_i, 'intensity': positions[start_i], 'velocity': 1}
            Current_intensity = positions[start_i]
            inner_direction = 1 if Direction > 0 else -1
            end_i = start_i
            while end_i < len(positions) - 1:
                end_i += 1
                diff = positions[end_i]
                if diff == 0: continue
                Current_intensity += diff
                Current_velocity = round(Current_intensity / (end_i - start_i + 1), 2)
                if (abs(max_prev['intensity']) < abs(Current_intensity)) or \
                   (abs(max_prev['intensity']) == abs(Current_intensity) and abs(max_prev['velocity']) < abs(Current_velocity)):
                    max_prev = {'start': start_i, 'end': end_i, 'intensity': Current_intensity, 'velocity': Current_velocity}
                if Direction > 0:
                    if diff > 0: inner_direction = min(threshold, inner_direction + diff)
                    if diff < 0:
                        inner_direction += diff
                        if abs(inner_direction) >= threshold: break
                if Direction < 0:
                    if diff > 0:
                        inner_direction += diff
                        if abs(inner_direction) >= threshold: break
                    if diff < 0: inner_direction = max(-threshold, inner_direction + diff)
            Current_intensity = max_prev['intensity']
            end_i = max_prev['end']
            while start_i < end_i:
                start_i += 1
                diff = positions[start_i - 1]
                Current_intensity -= diff
                Current_velocity = round(Current_intensity / (end_i - start_i + 1), 2)
                if (abs(max_prev['intensity']) < abs(Current_intensity)) or \
                   (abs(max_prev['intensity']) == abs(Current_intensity) and abs(max_prev['velocity']) < abs(Current_velocity)):
                    max_prev = {'start': start_i, 'end': end_i, 'intensity': Current_intensity, 'velocity': Current_velocity}
            Final_output.append(max_prev)
            start_i = max_prev['end']
            Direction = 0
            last_few_neg, last_few_pos = [], []
    return Final_output

################################################################################
# OPERATOR VALUES — Edit thresholds/categories here
################################################################################
POSECODE_OPERATORS_VALUES = {
    'angle': {'category_names': ['completely bent','bent more','right angle','bent less','slightly bent','straight'],
              'category_thresholds': [45,75,105,135,160], 'random_max_offset': 5},
    'distance': {'category_names': ['close','shoulder width','spread','wide'],
                 'category_thresholds': [0.20,0.50,0.80], 'random_max_offset': 0.05},
    'relativePosX': {'category_names': ['at_right','ignored_relpos0','at_left'],
                     'category_thresholds': [-0.15,0.15], 'random_max_offset': 0.05},
    'relativePosY': {'category_names': ['below','ignored_relpos1','above'],
                     'category_thresholds': [-0.15,0.15], 'random_max_offset': 0.05},
    'relativePosZ': {'category_names': ['behind','ignored_relpos2','front'],
                     'category_thresholds': [-0.15,0.15], 'random_max_offset': 0.05},
    'position_x': {'category_names': [f'x_{x/100.0}' for x in range(-200,205,5)],
                   'category_thresholds': [x/100.0 for x in range(-200,200,5)], 'random_max_offset': 0.05},
    'position_y': {'category_names': [f'x_{x/100.0}' for x in range(-200,205,5)],
                   'category_thresholds': [x/100.0 for x in range(-200,200,5)], 'random_max_offset': 0.05},
    'position_z': {'category_names': [f'x_{x/100.0}' for x in range(-200,205,5)],
                   'category_thresholds': [x/100.0 for x in range(-200,200,5)], 'random_max_offset': 0.05},
    'orientation_pitch': {'category_names': ['upside_down_backward','lying_flat_backward','leaning_backward',
                          'slightly_leaning_backward','neutral_pitch','slightly_leaning_forward',
                          'leaning_forward','lying_flat_forward','upside_down_forwardt'],
                          'category_thresholds': [35,55,75,85,95,105,125,145], 'random_max_offset': 5},
    'orientation_roll': {'category_names': ['upside_down_right','lying_right','leaning_right',
                         'moderately_leaning_right','slightly_leaning_right','neutral',
                         'slightly_leaning_left','moderately_leaning_left','leaning_left','lying_left','upside_down_left'],
                         'category_thresholds': [-90,-45,-30,-15,5,5,15,30,45,90], 'random_max_offset': 5},
    'orientation_yaw': {'category_names': ['about-face_turned_clockwise','completely_turned_clockwise',
                        'moderately_turned_clockwise','slightly_turned_clockwise','neutral',
                        'slightly_turned_counterclockwise','moderately_turned_counterclockwise',
                        'completely_turned_counterclockwise','about-face_turned_counterclockwise'],
                        'category_thresholds': [-135,-90,-60,-25,25,60,90,135], 'random_max_offset': 5},
}

_vel = ['very_slow','slow','moderate','fast','very_fast']
_vel_t = [0.05,0.1,0.5,0.8]
MOTIONCODE_OPERATORS_VALUES = {
    'angular':          {'category_names': ['significant_bend','moderate_bend','slight_bend','no_action','slight_extension','moderate_extension','significant_extension'],
                         'category_thresholds': [-4,-3,-2,0,2,3], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 1},
    'proximity':        {'category_names': ['significant_closing','moderate_closing','stationary','moderate_spreading','significant_spreading'],
                         'category_thresholds': [-2.1,-1,1,2.1], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 0.05},
    'spatial_relation_x':{'category_names': ['left-to-right','stationary','right-to-left'],
                         'category_thresholds': [-1,1], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 0.05},
    'spatial_relation_y':{'category_names': ['above-to-below','stationary','below-to-above'],
                         'category_thresholds': [-1,1], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 0.05},
    'spatial_relation_z':{'category_names': ['front-to-behind','stationary','behind-to-front'],
                         'category_thresholds': [-1,1], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 0.05},
    'displacement_x':   {'category_names': ['very_long_left','long_left','moderate_left','short_left','very_short_left','no_action','very_short_right','short_right','moderate_right','long_right','very_long_right'],
                         'category_thresholds': [-10,-8,-5,-3,-1,1,3,5,8,10], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 0.05},
    'displacement_y':   {'category_names': ['very_long_down','long_down','moderate_down','short_down','very_short_down','no_action','very_short_up','short_up','moderate_up','long_up','very_long_up'],
                         'category_thresholds': [-10,-8,-5,-3,-1,1,3,5,8,10], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 0.05},
    'displacement_z':   {'category_names': ['very_long_backward','long_backward','moderate_backward','short_backward','very_short_backward','no_action','very_short_forward','short_forward','moderate_forward','long_forward','very_long_forward'],
                         'category_thresholds': [-10,-8,-5,-3,-1,1,3,5,8,10], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 0.05},
    'rotation_pitch':   {'category_names': ['significant_leaning_backward','moderate_leaning_backward','slight_leaning_backward','no_action','slight_leaning_forward','moderate_leaning_forward','significant_leaning_forward'],
                         'category_thresholds': [-4,-3,-2,0,2,3], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 0.05},
    'rotation_roll':    {'category_names': ['significant_leaning_right','moderate_leaning_right','slight_leaning_right','no_action','slight_leaning_left','moderate_leaning_left','significant_leaning_left'],
                         'category_thresholds': [-4,-3,-2,0,2,3], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 0.05},
    'rotation_yaw':     {'category_names': ['significant_turn_clockwise','moderate_turn_clockwise','slight_turn_clockwise','no_action','slight_turn_counterclockwise','moderate_turn_counterclockwise','significant_turn_counterclockwise'],
                         'category_thresholds': [-4,-3,-2,0,2,3], 'category_names_velocity': _vel, 'category_thresholds_velocity': _vel_t, 'random_max_offset': 0.05},
}

################################################################################
# JOINT NAMES
################################################################################
ALL_JOINT_NAMES = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist',
    'left_index1','left_index2','left_index3','left_middle1','left_middle2','left_middle3','left_pinky1','left_pinky2','left_pinky3','left_ring1','left_ring2','left_ring3','left_thumb1','left_thumb2','left_thumb3',
    'right_index1','right_index2','right_index3','right_middle1','right_middle2','right_middle3','right_pinky1','right_pinky2','right_pinky3','right_ring1','right_ring2','right_ring3','right_thumb1','right_thumb2','right_thumb3',
    'orientation','translation']
ALL_JOINT_NAMES2ID = {jn: i for i, jn in enumerate(ALL_JOINT_NAMES)}
VIRTUAL_JOINTS = ["left_hand", "right_hand", "torso"]
JOINT_NAMES = ALL_JOINT_NAMES[:22] + ALL_JOINT_NAMES[-2:] + ['left_middle2', 'right_middle2'] + VIRTUAL_JOINTS
JOINT_NAMES2ID = {jn: i for i, jn in enumerate(JOINT_NAMES)}

################################################################################
# ELEMENTARY POSECODES — Edit joint sets here
################################################################################
ANGLE_POSECODES = [
    [('left_hip','left_knee','left_ankle'), 'left_knee', [], ['completely bent'], [('completely bent',2)]],       #L
    [('right_hip','right_knee','right_ankle'), 'right_knee', [], ['completely bent'], [('completely bent',2)]],   #L
    [('left_shoulder','left_elbow','left_wrist'), 'left_elbow', [], ['completely bent'], []],
    [('right_shoulder','right_elbow','right_wrist'), 'right_elbow', [], ['completely bent'], []],
]
DISTANCE_POSECODES = [
    [('left_elbow','right_elbow'), None, ["close","shoulder width","wide"], ["close"], [('shoulder width',1)]],
    [('left_hand','right_hand'), None, ["close","shoulder width","spread","wide"], [], [('shoulder width',1)]],
    [('left_knee','right_knee'), None, ["close","shoulder width","wide"], ["wide"], [('shoulder width',1)]],       #L
    [('left_foot','right_foot'), None, ["close","shoulder width","wide"], ["close"], [('shoulder width',1)]],       #L
    [('left_hand','left_shoulder'), 'left_hand', ['close'], ['close'], []],
    [('left_hand','right_shoulder'), 'left_hand', ['close'], ['close'], []],
    [('right_hand','right_shoulder'), 'right_hand', ['close'], ['close'], []],
    [('right_hand','left_shoulder'), 'right_hand', ['close'], ['close'], []],
    [('left_hand','right_elbow'), 'left_hand', ['close'], ['close'], []],
    [('right_hand','left_elbow'), 'right_hand', ['close'], ['close'], []],
    [('left_hand','left_knee'), 'left_hand', ['close'], ['close'], []],       #L
    [('left_hand','right_knee'), 'left_hand', ['close'], ['close'], []],      #L
    [('right_hand','right_knee'), 'right_hand', ['close'], ['close'], []],    #L
    [('right_hand','left_knee'), 'right_hand', ['close'], ['close'], []],     #L
    [('left_hand','left_ankle'), 'left_hand', ['close'], ['close'], []],      #L
    [('left_hand','right_ankle'), 'left_hand', ['close'], ['close'], []],     #L
    [('right_hand','right_ankle'), 'right_hand', ['close'], ['close'], []],   #L
    [('right_hand','left_ankle'), 'right_hand', ['close'], ['close'], []],    #L
    [('left_hand','left_foot'), 'left_hand', ['close'], ['close'], []],       #L
    [('left_hand','right_foot'), 'left_hand', ['close'], ['close'], []],      #L
    [('right_hand','right_foot'), 'right_hand', ['close'], ['close'], []],    #L
    [('right_hand','left_foot'), 'right_hand', ['close'], ['close'], []],     #L
]
RELATIVEPOS_POSECODES = [
    [('left_shoulder','right_shoulder'), None, [None,['below','above'],['behind','front']], [[],[],[]], [[],[],[]]],
    [('left_elbow','right_elbow'), None, [None,['below','above'],['behind','front']], [[],[],[]], [[],[],[]]],
    [('left_hand','right_hand'), None, [['at_right'],['below','above'],['behind','front']], [['at_right'],[],[]], [[],[],[]]],
    [('left_knee','right_knee'), None, [None,['below','above'],['behind','front']], [[],[],[]], [[],[('above',2)],[]]],                #L
    [('left_foot','right_foot'), None, [['at_right'],['below','above'],['behind','front']], [['at_right'],[],[]], [[],[],[]]],          #L
    [('neck','pelvis'), 'body', [['at_right','at_left'],None,['behind','front']], [[],[],[]], [[('at_right',1),('at_left',1)],[],[('behind',1),('front',1)]]],  #L
    [('left_ankle','neck'), 'left_ankle', [None,['below'],None], [[],[],[]], [[],[('below',1)],[]]],    #L
    [('right_ankle','neck'), 'right_ankle', [None,['below'],None], [[],[],[]], [[],[('below',1)],[]]],  #L
    [('left_hip','left_knee'), 'left_hip', [None,['above'],None], [[],[],[]], [[],[('above',1)],[]]],   #L
    [('right_hip','right_knee'), 'left_hip', [None,['above'],None], [[],[],[]], [[],[('above',1)],[]]],  #L
    [('left_hand','left_shoulder'), 'left_hand', [['at_right'],['above'],None], [[],[],[]], [[],[],[]]],
    [('right_hand','right_shoulder'), 'right_hand', [['at_left'],['above'],None], [[],[],[]], [[],[],[]]],
    [('left_foot','left_hip'), 'left_foot', [['at_right'],['above'],None], [['at_right'],['above'],[]], [[],[],[]]],    #L
    [('right_foot','right_hip'), 'right_foot', [['at_left'],['above'],None], [['at_left'],['above'],[]], [[],[],[]]],   #L
    [('left_wrist','neck'), 'left_hand', [None,['above'],None], [[],[],[]], [[],[],[]]],
    [('right_wrist','neck'), 'right_hand', [None,['above'],None], [[],[],[]], [[],[],[]]],
    [('left_hand','left_hip'), 'left_hand', [None,['below'],None], [[],[],[]], [[],[],[]]],    #L
    [('right_hand','right_hip'), 'right_hand', [None,['below'],None], [[],[],[]], [[],[],[]]],  #L
    [('left_hand','torso'), 'left_hand', [None,None,['behind']], [[],[],[]], [[],[],[]]],
    [('right_hand','torso'), 'right_hand', [None,None,['behind']], [[],[],[]], [[],[],[]]],
    [('left_foot','torso'), 'left_foot', [None,None,['behind','front']], [[],[],[]], [[],[],[]]],    #L
    [('right_foot','torso'), 'right_foot', [None,None,['behind','front']], [[],[],[]], [[],[],[]]],  #L
]
_sl_x = [(x,1) for x in POSECODE_OPERATORS_VALUES['position_x']['category_names']]
_sl_y = [(x,1) for x in POSECODE_OPERATORS_VALUES['position_y']['category_names']]
_sl_z = [(x,1) for x in POSECODE_OPERATORS_VALUES['position_z']['category_names']]
POSITION_POSECODES_X, POSITION_POSECODES_Y, POSITION_POSECODES_Z = [], [], []
for _j in JOINT_NAMES:
    _f = 'body' if _j == 'translation' else _j
    POSITION_POSECODES_X.append([(_j,), _f, [], [], _sl_x])
    POSITION_POSECODES_Y.append([(_j,), _f, [], [], _sl_y])
    POSITION_POSECODES_Z.append([(_j,), _f, [], [], _sl_z])
ORIENTATION_PITCH_POSECODES = [[('orientation',),'body',[],[],[]],[('translation',),'root_translation(Ignore)',[],[],[]]]
ORIENTATION_ROLL_POSECODES  = [[('orientation',),'body',[],[],[]],[('translation',),'root_translation(Ignore)',[],[],[]]]
ORIENTATION_YAW_POSECODES   = [[('orientation',),'body',[],[],[]],[('translation',),'root_translation(Ignore)',[],[],[]]]

ALL_ELEMENTARY_POSECODES = {
    "angle": ANGLE_POSECODES, "distance": DISTANCE_POSECODES,
    "relativePosX": [[p[0],p[1],p[2][0],p[3][0],p[4][0]] for p in RELATIVEPOS_POSECODES if p[2][0]],
    "relativePosY": [[p[0],p[1],p[2][1],p[3][1],p[4][1]] for p in RELATIVEPOS_POSECODES if p[2][1]],
    "relativePosZ": [[p[0],p[1],p[2][2],p[3][2],p[4][2]] for p in RELATIVEPOS_POSECODES if p[2][2]],
    "position_x": POSITION_POSECODES_X, "position_y": POSITION_POSECODES_Y, "position_z": POSITION_POSECODES_Z,
    'orientation_pitch': ORIENTATION_PITCH_POSECODES, 'orientation_roll': ORIENTATION_ROLL_POSECODES, 'orientation_yaw': ORIENTATION_YAW_POSECODES,
}

################################################################################
# ELEMENTARY MOTIONCODES — Edit joint sets here
################################################################################
_acc_ang = ['significant_bend','moderate_bend','moderate_extension','significant_extension']
_acc_prox = ['significant_closing','moderate_closing','moderate_spreading','significant_spreading']
_acc_prox_sig = ['significant_closing','significant_spreading']
_vel_acc = ['moderate','fast','very_fast']

ANGULAR_MOTIONCODES = [
    [('left_hip','left_knee','left_ankle'), 'left_knee', _acc_ang, [], [], _vel_acc, [], []],       #L
    [('right_hip','right_knee','right_ankle'), 'right_knee', _acc_ang, [], [], _vel_acc, [], []],   #L
    [('left_shoulder','left_elbow','left_wrist'), 'left_elbow', _acc_ang, [], [], _vel_acc, [], []],
    [('right_shoulder','right_elbow','right_wrist'), 'right_elbow', _acc_ang, [], [], _vel_acc, [], []],
]
PROXIMITY_MOTIONCODES = [
    [('left_elbow','right_elbow'), None, _acc_prox, [], [], _vel_acc, [], []],
    [('left_hand','right_hand'), None, _acc_prox_sig, [], [], _vel_acc, [], []],
    [('left_knee','right_knee'), None, _acc_prox_sig, [], [], _vel_acc, [], []],       #L
    [('left_foot','right_foot'), None, _acc_prox_sig, [], [], _vel_acc, [], []],       #L
    [('left_hand','left_shoulder'), 'left_hand', _acc_prox, [], [], _vel_acc, [], []],
    [('left_hand','right_shoulder'), 'left_hand', _acc_prox, [], [], _vel_acc, [], []],
    [('right_hand','right_shoulder'), 'right_hand', _acc_prox, [], [], _vel_acc, [], []],
    [('right_hand','left_shoulder'), 'right_hand', _acc_prox, [], [], _vel_acc, [], []],
    [('left_hand','right_elbow'), 'left_hand', _acc_prox, [], [], _vel_acc, [], []],
    [('right_hand','left_elbow'), 'right_hand', _acc_prox, [], [], _vel_acc, [], []],
    [('left_hand','left_knee'), 'left_hand', _acc_prox_sig, [], [], _vel_acc, [], []],       #L
    [('left_hand','right_knee'), 'left_hand', _acc_prox_sig, [], [], _vel_acc, [], []],      #L
    [('right_hand','right_knee'), 'right_hand', _acc_prox_sig, [], [], _vel_acc, [], []],    #L
    [('right_hand','left_knee'), 'right_hand', _acc_prox_sig, [], [], _vel_acc, [], []],     #L
    [('left_hand','left_ankle'), 'left_hand', _acc_prox_sig, [], [], _vel_acc, [], []],      #L
    [('left_hand','right_ankle'), 'left_hand', _acc_prox_sig, [], [], _vel_acc, [], []],     #L
    [('right_hand','right_ankle'), 'right_hand', _acc_prox_sig, [], [], _vel_acc, [], []],   #L
    [('right_hand','left_ankle'), 'right_hand', _acc_prox_sig, [], [], _vel_acc, [], []],    #L
    [('left_hand','left_foot'), 'left_hand', _acc_prox, [], [], _vel_acc, [], []],           #L
    [('left_hand','right_foot'), 'left_hand', _acc_prox, [], [], _vel_acc, [], []],          #L
    [('right_hand','right_foot'), 'right_hand', _acc_prox, [], [], _vel_acc, [], []],        #L
    [('right_hand','left_foot'), 'right_hand', _acc_prox, [], [], _vel_acc, [], []],         #L
]
SPATIAL_RELATION_X_MOTIONCODES = [
    [('left_hand','right_hand'), None, [], [], [], None, [], []],
    [('left_hand','left_shoulder'), 'left_hand', [], [], [], None, [], []],
    [('right_hand','right_shoulder'), 'right_hand', [], [], [], None, [], []],
    [('left_foot','left_hip'), 'left_foot', [], [], [], None, [], []],       #L
    [('right_foot','right_hip'), 'right_foot', [], [], [], None, [], []],    #L
]
SPATIAL_RELATION_Y_MOTIONCODES = [
    [('left_shoulder','right_shoulder'), None, [], [], [], None, [], []],
    [('left_elbow','right_elbow'), None, [], [], [], None, [], []],
    [('left_hand','right_hand'), None, [], [], [], None, [], []],
    [('left_knee','right_knee'), None, [], [], [], None, [], []],            #L
    [('left_foot','right_foot'), None, [], [], [], None, [], []],            #L
    [('left_hand','left_shoulder'), 'left_hand', [], [], [], None, [], []],
    [('right_hand','right_shoulder'), 'right_hand', [], [], [], None, [], []],
    [('left_foot','left_hip'), 'left_foot', [], ['above-to-below','below-to-above'], [], None, [], []],    #L
    [('right_foot','right_hip'), 'right_foot', [], ['above-to-below','below-to-above'], [], None, [], []], #L
    [('left_wrist','neck'), 'left_hand', [], [], [], None, [], []],
    [('right_wrist','neck'), 'right_hand', [], [], [], None, [], []],
    [('left_hand','left_hip'), 'left_hand', [], [], [], None, [], []],       #L
    [('right_hand','right_hip'), 'right_hand', [], [], [], None, [], []],    #L
]
SPATIAL_RELATION_Z_MOTIONCODES = [
    [('left_shoulder','right_shoulder'), None, [], [], [], None, [], []],
    [('left_elbow','right_elbow'), None, [], [], [], None, [], []],
    [('left_hand','right_hand'), None, [], [], [], None, [], []],
    [('left_hand','torso'), 'left_hand', [], [], [], None, [], []],
    [('right_hand','torso'), 'right_hand', [], [], [], None, [], []],
]
DISPLACEMENT_X = [[('translation',), 'body', [], [], [], _vel_acc, [], []]]
DISPLACEMENT_Y = [[('translation',), 'body', [], [], [], _vel_acc, [], []]]
DISPLACEMENT_Z = [[('translation',), 'body', [], [], [], _vel_acc, [], []]]
ROTATION_PITCH = [[('orientation',), 'body', [], [], [], _vel_acc, [], []]]
ROTATION_ROLL  = [[('orientation',), 'body', [], [], [], _vel_acc, [], []]]
ROTATION_YAW   = [[('orientation',), 'body', [], [], [], _vel_acc, [], []]]

ALL_ELEMENTARY_MOTIONCODES = {
    "angular": ANGULAR_MOTIONCODES, "proximity": PROXIMITY_MOTIONCODES,
    "spatial_relation_x": SPATIAL_RELATION_X_MOTIONCODES, "spatial_relation_y": SPATIAL_RELATION_Y_MOTIONCODES,
    "spatial_relation_z": SPATIAL_RELATION_Z_MOTIONCODES,
    "displacement_x": DISPLACEMENT_X, "displacement_y": DISPLACEMENT_Y, "displacement_z": DISPLACEMENT_Z,
    "rotation_pitch": ROTATION_PITCH, "rotation_roll": ROTATION_ROLL, "rotation_yaw": ROTATION_YAW,
}
MOTION2POSE_MAP = {
    'angular':'angle', 'proximity':'distance',
    'spatial_relation_x':'relativePosX', 'spatial_relation_y':'relativePosY', 'spatial_relation_z':'relativePosZ',
    'displacement_x':'position_x', 'displacement_y':'position_y', 'displacement_z':'position_z',
    'rotation_pitch':'orientation_pitch', 'rotation_roll':'orientation_roll', 'rotation_yaw':'orientation_yaw',
}

################################################################################
# POSECODE OPERATOR CLASSES
################################################################################
deg2rad = lambda t: math.pi * t / 180.0
rad2deg = lambda t: 180.0 * t / math.pi
torch_cos2deg = lambda c: rad2deg(torch.acos(c))

def distance_between_joint_pairs(joint_ids, joint_coords):
    if type(joint_ids) == list: joint_ids = torch.tensor(joint_ids)
    joint_ids = joint_ids.view(-1, 2)
    return torch.linalg.norm(joint_coords[:,joint_ids[:,0],:] - joint_coords[:,joint_ids[:,1],:], axis=2)

class Posecode:
    def __init__(self): self.category_names = self.category_thresholds = self.random_max_offset = None
    def eval(self, joint_ids, joint_coords): raise NotImplementedError
    def interprete(self, val, ct=None):
        if ct is None: ct = self.category_thresholds
        ret = torch.ones(val.shape) * len(ct)
        for i in range(len(ct)-1,-1,-1): ret[val<=ct[i]] = i
        return ret.int()

class PosecodeAngle(Posecode):
    def __init__(self):
        p = POSECODE_OPERATORS_VALUES['angle']
        self.category_names, self.category_thresholds, self.random_max_offset = p['category_names'], p['category_thresholds'], p['random_max_offset']
    def eval(self, jids, coords):
        v1 = torch.nn.functional.normalize(coords[:,jids[:,2]] - coords[:,jids[:,1]], dim=2)
        v2 = torch.nn.functional.normalize(coords[:,jids[:,0]] - coords[:,jids[:,1]], dim=2)
        return torch_cos2deg((v1*v2).sum(2))

class PosecodeDistance(Posecode):
    def __init__(self):
        p = POSECODE_OPERATORS_VALUES['distance']
        self.category_names, self.category_thresholds, self.random_max_offset = p['category_names'], p['category_thresholds'], p['random_max_offset']
    def eval(self, jids, coords): return distance_between_joint_pairs(jids, coords)

class PosecodeRelativePos(Posecode):
    def __init__(self, axis):
        p = POSECODE_OPERATORS_VALUES[['relativePosX','relativePosY','relativePosZ'][axis]]
        self.category_names, self.category_thresholds, self.random_max_offset = p['category_names'], p['category_thresholds'], p['random_max_offset']
        self.axis = axis
    def eval(self, jids, coords): return coords[:,jids[:,0],self.axis] - coords[:,jids[:,1],self.axis]

class PosecodePosition(Posecode):
    def __init__(self, axis):
        p = POSECODE_OPERATORS_VALUES[f'position_{axis}']
        self.category_names, self.category_thresholds, self.random_max_offset = p['category_names'], p['category_thresholds'], p['random_max_offset']
        self.axis = {'x':0,'y':1,'z':2}[axis]
    def eval(self, jids, coords): return coords[:,jids,self.axis].squeeze()

class PosecodeOrientation(Posecode):
    def __init__(self, axis):
        p = POSECODE_OPERATORS_VALUES[f'orientation_{["pitch","roll","yaw"][axis]}']
        self.category_names, self.category_thresholds, self.random_max_offset = p['category_names'], p['category_thresholds'], p['random_max_offset']
        self.axis = axis
    def eval(self, jids, coords): return coords[:,jids,self.axis].squeeze()

class MotioncodeGeneral:
    def __init__(self, kind):
        p = MOTIONCODE_OPERATORS_VALUES[kind]
        self.category_names, self.category_thresholds = p['category_names'], p['category_thresholds']
        self.category_names_velocity, self.category_thresholds_velocity = p['category_names_velocity'], p['category_thresholds_velocity']
        self.random_max_offset = p['random_max_offset']
    def eval(self, pm_pairs, coords, p_interp):
        out = []
        for pair in pm_pairs:
            ts = p_interp[:, pair['pj_id']].cpu().numpy()
            delta = [0] + [ts[i+1]-ts[i] for i in range(len(ts)-1)]
            out.append((pair['m_js'], single_path_finder(delta)))
        return out
    def interprete(self, val):
        ct_s, ct_t = self.category_thresholds, self.category_thresholds_velocity
        result = [[] for _ in range(len(val))]
        for iq in range(len(val)):
            for m in val[iq][1]:
                cs, ct = len(ct_s), len(ct_t)
                for i in range(len(ct_s)-1,-1,-1):
                    if m['intensity'] <= ct_s[i]: cs = i
                for i in range(len(ct_t)-1,-1,-1):
                    if abs(m['velocity']) <= ct_t[i]: ct = i
                result[iq].append({'spatial':cs,'temporal':ct,'start':m['start'],'end':m['end'],'intensity':m['intensity'],'velocity':m['velocity']})
        return result

POSECODE_OPERATORS = {"angle":PosecodeAngle(),"distance":PosecodeDistance(),
    "relativePosX":PosecodeRelativePos(0),"relativePosY":PosecodeRelativePos(1),"relativePosZ":PosecodeRelativePos(2),
    "position_x":PosecodePosition('x'),"position_y":PosecodePosition('y'),"position_z":PosecodePosition('z'),
    "orientation_pitch":PosecodeOrientation(0),"orientation_roll":PosecodeOrientation(1),"orientation_yaw":PosecodeOrientation(2)}
MOTIONCODE_OPERATORS = {k: MotioncodeGeneral(k) for k in MOTIONCODE_OPERATORS_VALUES}

################################################################################
# PIPELINE FUNCTIONS
################################################################################
def prepare_input(coords):
    nb = coords.shape[1]
    body, ot = coords[:,:-2,:], coords[:,-2:,:]
    nb_body = body.shape[1]
    if nb_body == 22:
        x = 0.1367
        lv = body[:,ALL_JOINT_NAMES2ID["left_wrist"]] - body[:,ALL_JOINT_NAMES2ID["left_elbow"]]
        rv = body[:,ALL_JOINT_NAMES2ID["right_wrist"]] - body[:,ALL_JOINT_NAMES2ID["right_elbow"]]
        aj = [x*lv/torch.linalg.norm(lv,axis=1).view(-1,1)+body[:,ALL_JOINT_NAMES2ID["left_wrist"]],
              x*rv/torch.linalg.norm(rv,axis=1).view(-1,1)+body[:,ALL_JOINT_NAMES2ID["right_wrist"]]]
        body = torch.cat([body]+[a.view(-1,1,3) for a in aj], axis=1)
    if nb_body >= 52:
        keep = [ALL_JOINT_NAMES2ID[jn] for jn in JOINT_NAMES[:-len(VIRTUAL_JOINTS)] if ALL_JOINT_NAMES2ID[jn] < nb_body]
        body = body[:,keep]
    coords = torch.cat([body, ot], axis=1)
    reorder = list(range(22)) + [24,25,22,23]
    coords = coords[:,reorder,:]
    vj = [0.5*(coords[:,JOINT_NAMES2ID["left_wrist"]]+coords[:,JOINT_NAMES2ID["left_middle2"]]),
          0.5*(coords[:,JOINT_NAMES2ID["right_wrist"]]+coords[:,JOINT_NAMES2ID["right_middle2"]]),
          1/3*(coords[:,JOINT_NAMES2ID["pelvis"]]+coords[:,JOINT_NAMES2ID["neck"]]+coords[:,JOINT_NAMES2ID["spine3"]])]
    coords = torch.cat([coords]+[v.view(-1,1,3) for v in vj], axis=1)
    return coords

def prepare_posecode_queries():
    pq = {}; offset = 0
    for pk, pl in ALL_ELEMENTARY_POSECODES.items():
        acc = [p[2] if p[2] else POSECODE_OPERATORS_VALUES[pk]['category_names'] for p in pl]
        n2id = {n:i+offset for i,n in enumerate(POSECODE_OPERATORS_VALUES[pk]['category_names'])}
        jids = torch.tensor([[JOINT_NAMES2ID[j] for j in p[0]] if type(p[0])!=str else [JOINT_NAMES2ID[p[0]]] for p in pl]).view(len(pl),-1)
        pq[pk] = {"joint_ids":jids, "acceptable_intptt_ids":[[n2id[n] for n in a] for a in acc], "focus_body_part":[p[1] for p in pl], "offset":offset}
        offset += len(POSECODE_OPERATORS_VALUES[pk]['category_names'])
    return pq

def prepare_motioncode_queries():
    mq = {}; offset = 0
    vel_off = sum(len(MOTIONCODE_OPERATORS_VALUES[k]['category_names']) for k in MOTIONCODE_OPERATORS_VALUES)
    for mk, ml in ALL_ELEMENTARY_MOTIONCODES.items():
        pov = MOTIONCODE_OPERATORS_VALUES[mk]
        n2id = {n:i+offset for i,n in enumerate(pov['category_names'])}
        vn2id = {n:i+vel_off for i,n in enumerate(pov['category_names_velocity'])}
        jids = torch.tensor([[JOINT_NAMES2ID[j] for j in p[0]] if type(p[0])!=str else [JOINT_NAMES2ID[p[0]]] for p in ml]).view(len(ml),-1)
        sa = [p[2] if p[2] else pov['category_names'] for p in ml]
        ta = []
        for p in ml:
            if p[5] is None: ta.append([])
            elif p[5] == []: ta.append(pov['category_names_velocity'])
            else: ta.append(p[5])
        mq[mk] = {"joint_ids":jids, "spatial_acceptable_intptt_ids":[[n2id[n] for n in a] for a in sa],
                   "temporal_acceptable_intptt_ids":[[vn2id[n] for n in a] for a in ta],
                   "spatial_rare_intptt_ids":[[n2id.get(n,0) for n in p[3]] for p in ml],
                   "temporal_rare_intptt_ids":[[vn2id.get(n,0) for n in p[6]] for p in ml],
                   "focus_body_part":[p[1] for p in ml], "offset":offset}
        offset += len(pov['category_names'])
    return mq

def infer_posecodes(coords, pq):
    pi = {}
    for pk, op in POSECODE_OPERATORS.items():
        val = op.eval(pq[pk]["joint_ids"], coords)
        # Random noise removed for deterministic results
        # val += (torch.rand(val.shape)*2-1).to(coords.device) * op.random_max_offset
        pi[pk] = op.interprete(val) + pq[pk]["offset"]
    return pi

def infer_motioncodes(coords, pi, pq, mq):
    mi = {}
    for mk, op in MOTIONCODE_OPERATORS.items():
        pk = MOTION2POSE_MAP[mk]
        pairs = []
        for mid in range(mq[mk]["joint_ids"].shape[0]):
            mjs = mq[mk]["joint_ids"][mid]
            match = torch.all(pq[pk]['joint_ids']==mjs, dim=1)
            pid = torch.where(match)[0].cpu().numpy().item()
            pairs.append({'m_js':mjs,'mjs_id':mid,'pj_id':pid})
        val = op.eval(pairs, coords, pi[pk])
        mint = op.interprete(val)
        for i in range(len(mint)):
            for j in range(len(mint[i])):
                mint[i][j]['spatial'] += mq[mk]["offset"]
        mi[mk] = mint
    return mi

################################################################################
# VECTOR BUILDER & HELPERS
################################################################################
def build_slot_definitions():
    slots = []
    for mk, ml in ALL_ELEMENTARY_MOTIONCODES.items():
        for entry in ml:
            joints = tuple(entry[0]) if isinstance(entry[0], tuple) else (entry[0],)
            slots.append({'kind':mk, 'joints':joints, 'focus':entry[1], 'is_pairwise':len(joints)>1})
    return slots

def build_vector(mi, slots, option_b=True):
    values = []
    for slot in slots:
        kind = slot['kind']
        ml = ALL_ELEMENTARY_MOTIONCODES[kind]
        js_idx = None
        for idx, entry in enumerate(ml):
            ej = tuple(entry[0]) if isinstance(entry[0], tuple) else (entry[0],)
            if ej == slot['joints']: js_idx = idx; break
        if js_idx is not None and js_idx < len(mi[kind]):
            motions = mi[kind][js_idx]
            if motions:
                best = max(motions, key=lambda m: abs(m['intensity']))
                values.append(best['spatial']); values.append(best['temporal'])
            else: values.append(-1); values.append(-1)
        else: values.append(-1); values.append(-1)
    if option_b:
        for i, slot in enumerate(slots):
            if slot['is_pairwise']:
                values.append(values[i*2]); values.append(values[i*2+1])
    return np.array(values, dtype=np.int32)

def extract_motioncode_vector(coords, option_b=True, verbose=False):
    coords = coords.to(device)
    if verbose: print("Preparing input...")
    coords = prepare_input(coords)
    if verbose: print("Preparing queries...")
    pq = prepare_posecode_queries(); mq = prepare_motioncode_queries()
    if verbose: print("Inferring posecodes...")
    pi = infer_posecodes(coords, pq)
    if verbose: print("Inferring motioncodes...")
    mi = infer_motioncodes(coords, pi, pq, mq)
    if verbose: print("Building vector...")
    slots = build_slot_definitions()
    vec = build_vector(mi, slots, option_b=option_b)
    if verbose: print(f"Vector shape: {vec.shape}, Non-zero: {np.count_nonzero(vec != -1)}/{len(vec)}")
    return vec

def get_slot_labels(option_b=True):
    slots = build_slot_definitions()
    labels = []
    for s in slots:
        j = '+'.join(s['joints'])
        labels.append(f"{j}|{s['kind']}|intensity"); labels.append(f"{j}|{s['kind']}|velocity")
    if option_b:
        for s in slots:
            if s['is_pairwise']:
                j = '+'.join(reversed(s['joints']))
                labels.append(f"{j}|{s['kind']}|intensity(dup)"); labels.append(f"{j}|{s['kind']}|velocity(dup)")
    return labels

def get_category_names():
    info = {}
    for k, p in MOTIONCODE_OPERATORS_VALUES.items():
        info[k] = {'intensity_categories':p['category_names'], 'intensity_thresholds':p['category_thresholds']}
    info['_velocity'] = {'categories':_vel, 'thresholds':_vel_t}
    return info

################################################################################
# LOWER-BODY FILTER
################################################################################
LOWER_BODY_JOINTS = {'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot', 'pelvis'}

def get_lower_body_indices(option_b=True):
    labels = get_slot_labels(option_b=option_b)
    indices = []
    for i, label in enumerate(labels):
        joints_in_slot = set(label.split('|')[0].split('+'))
        if joints_in_slot & LOWER_BODY_JOINTS: indices.append(i)
    return indices

def has_lower_body_motion(vector, lb_indices):
    return any(vector[i] != -1 for i in lb_indices)

def filter_motions_by_lower_body(motion_files, load_fn, output_dir, texts_dir=None, option_b=True):
    """
    Filter motions by lower-body activity. Copies qualifying files to output_dir.
    Optionally syncs a texts_dir to only keep text files matching kept motions.
    """
    os.makedirs(output_dir, exist_ok=True)
    lb_idx = get_lower_body_indices(option_b=option_b)
    kept, skipped, failed = 0, 0, 0
    kept_ids = set()
    for item in motion_files:
        if isinstance(item, (list, tuple)): mid, fpath = item
        else: fpath = item; mid = os.path.splitext(os.path.basename(fpath))[0]
        try:
            coords = load_fn(fpath)
            if coords.shape[0] < 2: failed += 1; continue
            vec = extract_motioncode_vector(coords, option_b=option_b, verbose=False)
            if has_lower_body_motion(vec, lb_idx):
                shutil.copy2(fpath, os.path.join(output_dir, os.path.basename(fpath))); kept += 1
                kept_ids.add(mid)
            else: skipped += 1
        except Exception as e: print(f"  Failed {mid}: {e}"); failed += 1
    print(f"\nLower-body filter: Kept={kept}, Skipped={skipped}, Failed={failed}, Output={output_dir}/")

    # Sync texts folder: remove text files that don't match kept motions
    if texts_dir and os.path.isdir(texts_dir):
        removed_texts = 0
        for tf in os.listdir(texts_dir):
            tid = os.path.splitext(tf)[0]
            if tid not in kept_ids:
                os.remove(os.path.join(texts_dir, tf))
                removed_texts += 1
        print(f"Texts sync: removed {removed_texts} from {texts_dir}/, kept {len(os.listdir(texts_dir))}")

################################################################################
# LOWER-BODY SCORING
################################################################################
def score_lower_body(motion_dir, texts_dir, output_json, option_b=True):
    """
    For each motion in motion_dir, extract lower-body slot values and
    pair with the first sentence from the matching text file.
    Saves results to a JSON file. Excludes Option B duplicates from scoring.
    """
    # Get lower-body indices from the NON-duplicated portion only
    labels_no_dup = get_slot_labels(option_b=False)
    lb_indices_no_dup = []
    for i, label in enumerate(labels_no_dup):
        joints_in_slot = set(label.split('|')[0].split('+'))
        if joints_in_slot & LOWER_BODY_JOINTS:
            lb_indices_no_dup.append(i)
    lb_labels = [(i, labels_no_dup[i]) for i in lb_indices_no_dup]

    files = sorted([f for f in os.listdir(motion_dir) if f.endswith('.npy')])
    results = {}
    failed = 0

    for fi, fname in enumerate(files):
        mid = os.path.splitext(fname)[0]
        if fi % 500 == 0:
            print(f"  Processing {fi}/{len(files)}...")
        try:
            coords = load_humanml3d(mid)
            if coords.shape[0] < 2:
                failed += 1
                continue
            # Use option_b=False for scoring to avoid duplicates
            vec = extract_motioncode_vector(coords, option_b=False, verbose=False)

            # Extract lower-body scores (only active slots)
            lb_scores = {}
            for idx, label in lb_labels:
                val = int(vec[idx])
                if val != -1:
                    lb_scores[label] = val

            # Get first sentence from text file
            description = ""
            txt_path = os.path.join(texts_dir, f"{mid}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        description = first_line.split('#')[0].strip()

            results[mid] = {
                "lower_body_scores": lb_scores,
                "active_lb_slots": len(lb_scores),
                "description": description
            }
        except Exception as e:
            print(f"  Failed {mid}: {e}")
            failed += 1

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nLower-body scoring complete:")
    print(f"  Processed: {len(results)}")
    print(f"  Failed: {failed}")
    print(f"  Max possible LB slots: {len(lb_indices_no_dup)}")
    print(f"  Saved to: {output_json}")


################################################################################
# BATCH PROCESSING & SAVING
################################################################################
def process_motion_files(motion_files, load_fn, option_b=True, verbose=False):
    vectors, motion_ids, failed = [], [], []
    for item in motion_files:
        if isinstance(item, (list, tuple)): mid, fpath = item
        else: fpath = item; mid = os.path.splitext(os.path.basename(fpath))[0]
        try:
            coords = load_fn(fpath)
            if coords.shape[0] < 2: failed.append((mid,"too few frames")); continue
            vec = extract_motioncode_vector(coords, option_b=option_b, verbose=False)
            vectors.append(vec); motion_ids.append(mid)
            if verbose: print(f"  {mid}: {np.count_nonzero(vec!=-1)}/{len(vec)} active")
        except Exception as e: failed.append((mid, str(e)))
    if failed:
        print(f"\nFailed: {len(failed)}")
        for m,r in failed[:10]: print(f"  {m}: {r}")
    result = {'vectors':np.array(vectors,dtype=np.int32) if vectors else np.empty((0,0),dtype=np.int32),
              'motion_ids':motion_ids, 'labels':get_slot_labels(option_b), 'categories':get_category_names()}
    print(f"Processed {len(vectors)} motions → {result['vectors'].shape}")
    return result

def save_space(space, out_dir="motioncode_output"):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir,'vectors.npz'), vectors=space['vectors'], motion_ids=np.array(space['motion_ids'],dtype=str))
    with open(os.path.join(out_dir,'metadata.json'),'w') as f:
        json.dump({'labels':space['labels'],'categories':space['categories'],'num_motions':len(space['motion_ids']),
                   'num_dimensions':space['vectors'].shape[1] if len(space['vectors'])>0 else 0}, f, indent=2)
    print(f"Saved to {out_dir}/")

def load_space(out_dir="motioncode_output"):
    d = np.load(os.path.join(out_dir,'vectors.npz'), allow_pickle=True)
    with open(os.path.join(out_dir,'metadata.json'),'r') as f: meta = json.load(f)
    return {'vectors':d['vectors'], 'motion_ids':d['motion_ids'].tolist(), 'labels':meta['labels'], 'categories':meta['categories']}

################################################################################
# MAIN
################################################################################

# Paths — edit these to match your layout
NEW_JOINTS_DIR = "new_joints"
NEW_JOINT_VECS_DIR = "new_joint_vecs"


def _qrot(q, v):
    """Rotate vector(s) v about quaternion(s) q. q: (...,4), v: (...,3)."""
    assert q.shape[-1] == 4 and v.shape[-1] == 3
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2 * (q[..., :1] * uv + uuv)


def _qinv(q):
    mask = torch.ones_like(q)
    mask[..., 1:] = -1
    return q * mask


def _qefix(q):
    """Enforce quaternion continuity. q: (L, J, 4)."""
    result = q.copy()
    dot = np.sum(q[1:] * q[:-1], axis=2)
    mask = (np.cumsum(dot < 0, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def recover_root_rot_pos(data):
    """Recover root rotation quaternion, position, and Euler angles from 263-d HumanML3D features."""
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,))
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,))
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]

    # Convert quaternion to Euler angles (xyz order, radians)
    q0, q1, q2, q3 = r_rot_quat[...,0], r_rot_quat[...,1], r_rot_quat[...,2], r_rot_quat[...,3]
    ex = torch.atan2(2*(q0*q1 - q2*q3), 1 - 2*(q1*q1 + q2*q2))
    ey = torch.asin(torch.clamp(2*(q1*q3 + q0*q2), -1, 1))
    ez = torch.atan2(2*(q0*q3 - q1*q2), 1 - 2*(q2*q2 + q3*q3))
    euler = torch.stack((ex, ey, ez), dim=-1)  # radians

    # Enforce continuity
    euler_np = _qefix(euler[:, None, :].cpu().numpy()).squeeze(1)
    euler = torch.tensor(euler_np, dtype=torch.float32)

    # Convert to degrees
    euler_deg = euler * (180.0 / math.pi)

    return r_rot_quat, r_pos, euler_deg


def load_humanml3d(motion_id):
    """
    Load a HumanML3D motion by ID. Combines new_joints (positions) with
    new_joint_vecs (for root orientation recovery).

    Returns: torch.tensor of shape (num_frames, 24, 3)
      joints 0-21:  body joint positions
      joint 22:     orientation (Euler angles in degrees: pitch, roll, yaw)
      joint 23:     translation (pelvis xyz in meters)
    """
    joints_path = os.path.join(NEW_JOINTS_DIR, f"{motion_id}.npy")
    vecs_path = os.path.join(NEW_JOINT_VECS_DIR, f"{motion_id}.npy")

    joints = np.load(joints_path)  # (frames, 22, 3)
    vecs = np.load(vecs_path)      # (frames, 263)

    # Handle single-frame files that lost the time dimension
    if joints.ndim == 2:
        joints = joints[np.newaxis, :]  # (1, 22, 3)
    if vecs.ndim == 1:
        vecs = vecs[np.newaxis, :]      # (1, 263)

    frames = min(joints.shape[0], vecs.shape[0])
    joints = joints[:frames]
    vecs = vecs[:frames]

    # Recover root orientation from 263-d features
    vecs_t = torch.tensor(vecs, dtype=torch.float32)
    _, r_pos, euler_deg = recover_root_rot_pos(vecs_t)

    # Normalize orientation: subtract first frame yaw so body starts facing forward
    euler_deg[:, 2] = euler_deg[:, 2] - euler_deg[0, 2]

    # Build coords: 22 joints + orientation + translation
    orientation = euler_deg.numpy()[:, np.newaxis, :]     # (frames, 1, 3)
    translation = r_pos.numpy()[:, np.newaxis, :]         # (frames, 1, 3)

    coords = np.concatenate([joints, orientation, translation], axis=1)  # (frames, 24, 3)
    return torch.tensor(coords, dtype=torch.float32)


if __name__ == "__main__":
    import glob
    """
    print("=" * 60)
    print("Testing with random data...")
    print("=" * 60)
    coords = torch.randn(60, 24, 3)
    vector = extract_motioncode_vector(coords, option_b=True, verbose=True)
    labels = get_slot_labels(option_b=True)
    print(f"\nNon-zero dimensions:")
    for i, (label, val) in enumerate(zip(labels, vector)):
        if val != -1: print(f"  [{i:3d}] {label}: {val}")

    # =====================================================================
    # Test with real HumanML3D data
    # =====================================================================
    
    print("\n" + "=" * 60)
    print("Testing with HumanML3D motion 000000...")
    print("=" * 60)
    try:
        coords = load_humanml3d("000000")
        print(f"Loaded coords shape: {coords.shape}")
        vector = extract_motioncode_vector(coords, option_b=True, verbose=True)
        labels = get_slot_labels(option_b=True)
        print(f"\nNon-zero dimensions:")
        for i, (label, val) in enumerate(zip(labels, vector)):
            if val != -1: print(f"  [{i:3d}] {label}: {val}")
    except Exception as e:
        print(f"Could not load: {e}")
    """
    # =====================================================================
    # OPTION 2: Filter motions by lower-body activity
    # =====================================================================
    print("\n" + "=" * 60)
    print("Filtering motions by lower-body activity...")
    print("=" * 60)
    files = [os.path.splitext(f)[0] for f in sorted(os.listdir(NEW_JOINTS_DIR)) if f.endswith('.npy')]
    filter_motions_by_lower_body(
        [(mid, os.path.join(NEW_JOINTS_DIR, f"{mid}.npy")) for mid in files],
        lambda fpath: load_humanml3d(os.path.splitext(os.path.basename(fpath))[0]),
        output_dir="lower_body_motions",
        texts_dir="texts"
    )

    # =====================================================================
    # OPTION 3: Score lower-body activity and save with descriptions
    # =====================================================================
    print("\n" + "=" * 60)
    print("Scoring lower-body activity...")
    print("=" * 60)
    score_lower_body("lower_body_motions", "texts", "lower_body_scores.json")

    # =====================================================================
    # OPTION 4: Build motioncode space
    # =====================================================================
    # files = [os.path.splitext(f)[0] for f in sorted(os.listdir(NEW_JOINTS_DIR)) if f.endswith('.npy')]
    # motion_items = [(mid, os.path.join(NEW_JOINTS_DIR, f"{mid}.npy")) for mid in files]
    # space = process_motion_files(motion_items,
    #     lambda fpath: load_humanml3d(os.path.splitext(os.path.basename(fpath))[0]),
    #     option_b=True, verbose=True)
    # save_space(space, out_dir="motioncode_output")

"""
python -c "
import os
from extract_motioncodes import load_humanml3d, process_motion_files, save_space

files = [os.path.splitext(f)[0] for f in sorted(os.listdir('style_motions')) if f.endswith('.npy')]
motion_items = [(mid, os.path.join('style_motions', f'{mid}.npy')) for mid in files]

space = process_motion_files(
    motion_items,
    lambda fpath: load_humanml3d(os.path.splitext(os.path.basename(fpath))[0]),
    option_b=True, verbose=False
)
save_space(space, out_dir='motioncode_output')
"
"""