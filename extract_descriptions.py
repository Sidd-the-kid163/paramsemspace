"""
Full motioncode-to-description pipeline for lower-body focused motions.
Extends extract_motioncodes.py with: onGround, relativeVAxis, super-posecodes,
support logic, aggregation, timecodes, and text conversion.
No random skipping. Lower-body joints + body-global categories only.

Usage:
    python extract_descriptions.py
"""

import math, copy, json, os, re, random
import torch
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Import the base pipeline from extract_motioncodes.py
from extract_motioncodes import (
    single_path_finder,
    POSECODE_OPERATORS_VALUES, MOTIONCODE_OPERATORS_VALUES,
    ALL_JOINT_NAMES, ALL_JOINT_NAMES2ID, VIRTUAL_JOINTS, JOINT_NAMES, JOINT_NAMES2ID,
    ANGLE_POSECODES, DISTANCE_POSECODES, RELATIVEPOS_POSECODES,
    POSITION_POSECODES_X, POSITION_POSECODES_Y, POSITION_POSECODES_Z,
    ORIENTATION_PITCH_POSECODES, ORIENTATION_ROLL_POSECODES, ORIENTATION_YAW_POSECODES,
    ALL_ELEMENTARY_MOTIONCODES, MOTION2POSE_MAP,
    PosecodeAngle, PosecodeDistance, PosecodeRelativePos, PosecodePosition, PosecodeOrientation,
    MotioncodeGeneral,
    distance_between_joint_pairs, deg2rad, rad2deg, torch_cos2deg,
    prepare_input, prepare_motioncode_queries,
    load_humanml3d,
    _vel, _vel_t,
)

################################################################################
# ADDITIONAL POSECODE OPERATORS (onGround, relativeVAxis)
################################################################################

# Add to POSECODE_OPERATORS_VALUES
POSECODE_OPERATORS_VALUES['relativeVAxis'] = {
    'category_names': ['vertical', 'ignored_relVaxis', 'horizontal'],
    'category_thresholds': [10, 80],
    'random_max_offset': 5
}
POSECODE_OPERATORS_VALUES['onGround'] = {
    'category_names': ['on_ground', 'ignored_onGround'],
    'category_thresholds': [0.10],
    'random_max_offset': 0.05
}

class PosecodeRelativeVAxis:
    def __init__(self):
        p = POSECODE_OPERATORS_VALUES['relativeVAxis']
        self.category_names, self.category_thresholds, self.random_max_offset = p['category_names'], p['category_thresholds'], p['random_max_offset']
        self.vertical_vec = torch.tensor([0.0, 1.0, 0.0]).to(device)
    def eval(self, jids, coords):
        bpv = torch.nn.functional.normalize(coords[:,jids[:,1]] - coords[:,jids[:,0]], dim=2)
        c = (self.vertical_vec * bpv).sum(2).abs()
        return torch_cos2deg(c)
    def interprete(self, val):
        ct = self.category_thresholds
        ret = torch.ones(val.shape) * len(ct)
        for i in range(len(ct)-1,-1,-1): ret[val<=ct[i]] = i
        return ret.int()

class PosecodeOnGround:
    def __init__(self):
        p = POSECODE_OPERATORS_VALUES['onGround']
        self.category_names, self.category_thresholds, self.random_max_offset = p['category_names'], p['category_thresholds'], p['random_max_offset']
    def eval(self, jids, coords):
        return coords[:, jids, 1].squeeze() - coords[:,:,1].min(1)[0].view(-1,1)
    def interprete(self, val):
        ct = self.category_thresholds
        ret = torch.ones(val.shape) * len(ct)
        for i in range(len(ct)-1,-1,-1): ret[val<=ct[i]] = i
        return ret.int()

################################################################################
# LOWER-BODY FOCUSED POSECODE DEFINITIONS
################################################################################

# relativeVAxis — lower body + torso only
RELATIVEVAXIS_POSECODES = [
    [('left_hip','left_knee'), 'left_thigh', ['horizontal','vertical'], ['horizontal'], []],
    [('right_hip','right_knee'), 'right_thigh', ['horizontal','vertical'], ['horizontal'], []],
    [('left_knee','left_ankle'), 'left_calf', ['horizontal','vertical'], ['horizontal'], []],
    [('right_knee','right_ankle'), 'right_calf', ['horizontal','vertical'], ['horizontal'], []],
    [('pelvis','left_shoulder'), 'left_backdiag', ['horizontal'], [], [('horizontal',1)]],
    [('pelvis','right_shoulder'), 'right_backdiag', ['horizontal'], [], [('horizontal',1)]],
    [('pelvis','neck'), 'torso', ['vertical'], [], []],
    [('left_foot','right_foot'), '<plural>_feet', ['horizontal'], [], [('horizontal',1)]],
]

# onGround — lower body only
ONGROUND_POSECODES = [
    [('left_knee',), 'left_knee', ['on_ground'], [], [('on_ground',1)]],
    [('right_knee',), 'right_knee', ['on_ground'], [], [('on_ground',1)]],
    [('left_foot',), 'left_foot', ['on_ground'], [], [('on_ground',1)]],
    [('right_foot',), 'right_foot', ['on_ground'], [], [('on_ground',1)]],
]

# Full posecode set including new operators
ALL_POSECODES = {
    "angle": ANGLE_POSECODES,
    "distance": DISTANCE_POSECODES,
    "relativePosX": [[p[0],p[1],p[2][0],p[3][0],p[4][0]] for p in RELATIVEPOS_POSECODES if p[2][0]],
    "relativePosY": [[p[0],p[1],p[2][1],p[3][1],p[4][1]] for p in RELATIVEPOS_POSECODES if p[2][1]],
    "relativePosZ": [[p[0],p[1],p[2][2],p[3][2],p[4][2]] for p in RELATIVEPOS_POSECODES if p[2][2]],
    "relativeVAxis": RELATIVEVAXIS_POSECODES,
    "onGround": ONGROUND_POSECODES,
    "position_x": POSITION_POSECODES_X,
    "position_y": POSITION_POSECODES_Y,
    "position_z": POSITION_POSECODES_Z,
    'orientation_pitch': ORIENTATION_PITCH_POSECODES,
    'orientation_roll': ORIENTATION_ROLL_POSECODES,
    'orientation_yaw': ORIENTATION_YAW_POSECODES,
}

# All posecode operators
ALL_POSECODE_OPERATORS = {
    "angle": PosecodeAngle(), "distance": PosecodeDistance(),
    "relativePosX": PosecodeRelativePos(0), "relativePosY": PosecodeRelativePos(1), "relativePosZ": PosecodeRelativePos(2),
    "relativeVAxis": PosecodeRelativeVAxis(), "onGround": PosecodeOnGround(),
    "position_x": PosecodePosition('x'), "position_y": PosecodePosition('y'), "position_z": PosecodePosition('z'),
    "orientation_pitch": PosecodeOrientation(0), "orientation_roll": PosecodeOrientation(1), "orientation_yaw": PosecodeOrientation(2),
}

MOTIONCODE_OPERATORS = {k: MotioncodeGeneral(k) for k in MOTIONCODE_OPERATORS_VALUES}

################################################################################
# SUPER-POSECODES (lower-body focused)
################################################################################

PLURAL_KEY = '<plural>'
def flatten_list(l): return [item for sublist in l for item in sublist]

# Only lower-body relevant super-posecodes
SUPER_POSECODES = [
    ['torso_horizontal', [('torso',), 'horizontal'], True],
    ['body_bent_forward', [('body',), 'bent_forward'], False],
    ['body_bent_backward', [('body',), 'bent_backward'], True],
    ['body_bent_left', [('body',), 'bent_left'], False],
    ['body_bent_right', [('body',), 'bent_right'], False],
    ['kneel_on_left', [('body',), 'kneel_on_left'], True],
    ['kneel_on_right', [('body',), 'kneel_on_right'], True],
    ['kneeling', [('body',), 'kneeling'], True],
    ['feet_shoulder_width', [(f'{PLURAL_KEY}_feet',), 'shoulder width'], False],
]

SUPER_POSECODES_REQUIREMENTS = {
    'torso_horizontal': [
        [['relativeVAxis', ('pelvis','left_shoulder'), 'horizontal'],
         ['relativeVAxis', ('pelvis','right_shoulder'), 'horizontal']]],
    'body_bent_forward': [
        [['relativePosY', ('left_ankle','neck'), 'below'],
         ['relativePosZ', ('neck','pelvis'), 'front']],
        [['relativePosY', ('right_ankle','neck'), 'below'],
         ['relativePosZ', ('neck','pelvis'), 'front']]],
    'body_bent_backward': [
        [['relativePosY', ('left_ankle','neck'), 'below'],
         ['relativePosZ', ('neck','pelvis'), 'behind']],
        [['relativePosY', ('right_ankle','neck'), 'below'],
         ['relativePosZ', ('neck','pelvis'), 'behind']]],
    'body_bent_left': [
        [['relativePosY', ('left_ankle','neck'), 'below'],
         ['relativePosX', ('neck','pelvis'), 'at_left']],
        [['relativePosY', ('right_ankle','neck'), 'below'],
         ['relativePosX', ('neck','pelvis'), 'at_left']]],
    'body_bent_right': [
        [['relativePosY', ('left_ankle','neck'), 'below'],
         ['relativePosX', ('neck','pelvis'), 'at_right']],
        [['relativePosY', ('right_ankle','neck'), 'below'],
         ['relativePosX', ('neck','pelvis'), 'at_right']]],
    'kneel_on_left': [
        [['relativePosY', ('left_knee','right_knee'), 'below'],
         ['onGround', ('left_knee',), 'on_ground'],
         ['onGround', ('right_foot',), 'on_ground']]],
    'kneel_on_right': [
        [['relativePosY', ('left_knee','right_knee'), 'above'],
         ['onGround', ('right_knee',), 'on_ground'],
         ['onGround', ('left_foot',), 'on_ground']]],
    'kneeling': [
        [['relativePosY', ('left_hip','left_knee'), 'above'],
         ['relativePosY', ('right_hip','right_knee'), 'above'],
         ['onGround', ('left_knee',), 'on_ground'],
         ['onGround', ('right_knee',), 'on_ground']],
        [['angle', ('left_hip','left_knee','left_ankle'), 'completely bent'],
         ['angle', ('right_hip','right_knee','right_ankle'), 'completely bent'],
         ['onGround', ('left_knee',), 'on_ground'],
         ['onGround', ('right_knee',), 'on_ground']]],
    'feet_shoulder_width': [
        [['distance', ('left_foot','right_foot'), 'shoulder width'],
         ['relativeVAxis', ('left_foot','right_foot'), 'horizontal']]],
}

################################################################################
# INTERPRETATION SETS (global ID mapping)
################################################################################

# Build global interpretation ID mapping for posecodes
INTERPRETATION_SET = flatten_list([POSECODE_OPERATORS_VALUES[k]['category_names'] for k in ALL_POSECODES if k in POSECODE_OPERATORS_VALUES])
# Add super-posecode interpretations
sp_intptts = [sp[1][1] for sp in SUPER_POSECODES if sp[1][1] not in INTERPRETATION_SET]
INTERPRETATION_SET += list(dict.fromkeys(sp_intptts))  # preserve order, deduplicate
INTPTT_NAME2ID = {n: i for i, n in enumerate(INTERPRETATION_SET)}

# Motioncode interpretation set
INTERPRETATION_SET_MOTION = flatten_list([MOTIONCODE_OPERATORS_VALUES[k]['category_names'] for k in MOTIONCODE_OPERATORS_VALUES])
GLOBAL_VELOCITY_OFFSET = len(INTERPRETATION_SET_MOTION)
INTERPRETATION_SET_MOTION += MOTIONCODE_OPERATORS_VALUES['angular']['category_names_velocity']
INTPTT_NAME2ID_MOTION = {n: i for i, n in enumerate(INTERPRETATION_SET_MOTION)}

# Timecode
TIMECODE_VALUES = {
    'ChronologicalOrder': {
        'category_names': ["preceding_a_moment","soon_before","shortly_before","immediately_before",
                           "simultaneously","immediately_after","shortly_after","soon_after","after_a_moment"],
        'category_thresholds': [-3,-1.7,-0.9,-0.2,0.2,0.9,1.7,3],
        'random_max_offset': 0.1,
    }
}
INTERPRETATION_SET_TIME = TIMECODE_VALUES['ChronologicalOrder']['category_names']
INTPTT_NAME2ID_TIME = {n: i for i, n in enumerate(INTERPRETATION_SET_TIME)}

# Opposite correspondences for subject/object swapping
OPPOSITE_CORRESP_ID = {}
_opp = {'at_right':'at_left','at_left':'at_right','below':'above','above':'below',
        'behind':'front','front':'behind','close':'close','shoulder width':'shoulder width',
        'spread':'spread','wide':'wide'}
for k, v in _opp.items():
    if k in INTPTT_NAME2ID and v in INTPTT_NAME2ID:
        OPPOSITE_CORRESP_ID[INTPTT_NAME2ID[k]] = INTPTT_NAME2ID[v]

OPPOSITE_CORRESP_MOTION = {'right-to-left':'left-to-right','above-to-below':'below-to-above','front-to-behind':'behind-to-front'}
OPPOSITE_CORRESP_ID_MOTION = {}
for k, v in OPPOSITE_CORRESP_MOTION.items():
    if k in INTPTT_NAME2ID_MOTION and v in INTPTT_NAME2ID_MOTION:
        OPPOSITE_CORRESP_ID_MOTION[INTPTT_NAME2ID_MOTION[k]] = INTPTT_NAME2ID_MOTION[v]

################################################################################
# QUERY PREPARATION (with super-posecodes and support logic)
################################################################################

def prepare_posecode_queries_full():
    pq = {}; offset = 0
    for pk, pl in ALL_POSECODES.items():
        all_names = POSECODE_OPERATORS_VALUES[pk]['category_names']
        acc = [p[2] if p[2] else all_names for p in pl]
        n2id = {n: i+offset for i, n in enumerate(all_names)}
        jids = torch.tensor([[JOINT_NAMES2ID[j] for j in p[0]] if type(p[0])!=str else [JOINT_NAMES2ID[p[0]]] for p in pl]).view(len(pl),-1)
        acc_ids = [[n2id[n] for n in a] for a in acc]
        rare_ids = [[n2id[n] for n in p[3]] for p in pl]
        sup1_ids = [[n2id[s[0]] for s in p[4] if s[1]==1] for p in pl]
        sup2_ids = [[n2id[s[0]] for s in p[4] if s[1]==2] for p in pl]
        pq[pk] = {"joint_ids": jids, "acceptable_intptt_ids": acc_ids, "rare_intptt_ids": rare_ids,
                   "support_intptt_ids_typeI": sup1_ids, "support_intptt_ids_typeII": sup2_ids,
                   "focus_body_part": [p[1] for p in pl], "offset": offset}
        offset += len(all_names)
    return pq

def prepare_super_posecode_queries(pq):
    spq = {}
    for sp in SUPER_POSECODES:
        sp_id = sp[0]
        required = []
        for w in SUPER_POSECODES_REQUIREMENTS[sp_id]:
            w_info = []
            for req in w:
                req_js = torch.tensor([JOINT_NAMES2ID[j] for j in req[1]] if type(req[1])!=str else [JOINT_NAMES2ID[req[1]]]).view(1,-1)
                try:
                    req_ind = torch.where((pq[req[0]]['joint_ids'] == req_js).all(1))[0][0].item()
                except IndexError:
                    print(f"Warning: posecode {req} not found for super-posecode {sp_id}")
                    req_ind = 0
                w_info.append([req[0], req_ind, INTPTT_NAME2ID[req[2]]])
            required.append(w_info)
        spq[sp_id] = {"required_posecodes": required, "is_rare": sp[2],
                       "intptt_id": INTPTT_NAME2ID[sp[1][1]], "focus_body_part": sp[1][0]}
    return spq

################################################################################
# POSECODE INFERENCE (with super-posecodes + support suppression)
################################################################################

def infer_posecodes_full(coords, pq, spq):
    nb_frames = len(coords)
    pi, pe = {}, {}
    for pk, op in ALL_POSECODE_OPERATORS.items():
        val = op.eval(pq[pk]["joint_ids"], coords)
        intptt = op.interprete(val) + pq[pk]["offset"]
        elig = torch.zeros(intptt.shape)
        for js in range(intptt.shape[1]):
            ia = torch.tensor(pq[pk]["acceptable_intptt_ids"][js])
            ir = torch.tensor(pq[pk]["rare_intptt_ids"][js])
            elig[:, js] = (intptt[:, js].view(-1,1) == ia).sum(1) + (intptt[:, js].view(-1,1) == ir).sum(1)
        pi[pk] = intptt
        pe[pk] = elig

    # Super-posecodes
    sp_elig = torch.zeros(nb_frames, len(spq))
    for sp_ind, sp_id in enumerate(spq):
        for w in spq[sp_id]["required_posecodes"]:
            col = torch.ones(nb_frames)
            for ep in w:
                col = torch.logical_and(col, (pi[ep[0]][:, ep[1]] == ep[2]))
            sp_elig[:, sp_ind] = torch.logical_or(sp_elig[:, sp_ind], col.view(-1))
        if spq[sp_id]["is_rare"]:
            sp_elig[:, sp_ind] *= 2

    # Support suppression
    for sp_ind, sp_id in enumerate(spq):
        for w in spq[sp_id]["required_posecodes"]:
            for ep in w:
                if ep[2] in pq[ep[0]]["support_intptt_ids_typeI"][ep[1]]:
                    sel = (pi[ep[0]][:, ep[1]] == ep[2])
                    pe[ep[0]][sel, ep[1]] = 0
                elif ep[2] in pq[ep[0]]["support_intptt_ids_typeII"][ep[1]]:
                    sel = torch.logical_and(sp_elig[:, sp_ind].bool(), (pi[ep[0]][:, ep[1]] == ep[2]))
                    pe[ep[0]][sel, ep[1]] = 0

    # Additional support-I cleanup
    for pk in pe:
        if pk == 'superPosecodes': continue
        intptt = pi[pk]
        for js in range(intptt.shape[1]):
            sup1 = torch.tensor(pq[pk]["support_intptt_ids_typeI"][js])
            if len(sup1) > 0:
                mask = 1 - (intptt[:, js].view(-1,1) == sup1).sum(1)
                pe[pk][:, js] *= mask

    pe["superPosecodes"] = sp_elig
    return pi, pe

################################################################################
# MOTIONCODE INFERENCE (same as extract_motioncodes but with offsets)
################################################################################

def infer_motioncodes_full(coords, pi, pq, mq):
    mi = {}
    for mk, op in MOTIONCODE_OPERATORS.items():
        pk = MOTION2POSE_MAP[mk]
        pairs = []
        for mid in range(mq[mk]["joint_ids"].shape[0]):
            mjs = mq[mk]["joint_ids"][mid]
            match = torch.all(pq[pk]['joint_ids'] == mjs, dim=1)
            pid = torch.where(match)[0].cpu().numpy().item()
            pairs.append({'m_js': mjs, 'mjs_id': mid, 'pj_id': pid})
        val = op.eval(pairs, coords, pi[pk])
        mint = op.interprete(val)
        for i in range(len(mint)):
            for j in range(len(mint[i])):
                mint[i][j]['spatial'] += mq[mk]["offset"]
                mint[i][j]['temporal'] += GLOBAL_VELOCITY_OFFSET
                mint[i][j]['posecode'] = [pk, pairs[i]['pj_id']]
        mi[mk] = mint
    return mi

################################################################################
# FORMAT MOTIONCODES (no random skip)
################################################################################

POSECODE_KIND_FOCUS = ['angular','relativeVAxis','onGround',
                       'displacement_x','displacement_y','displacement_z',
                       'rotation_pitch','rotation_roll','rotation_yaw',
                       'position_x','position_y','position_z',
                       'orientation_pitch','orientation_roll','orientation_yaw']

def parse_joint(jn):
    x = jn.split("_")
    return x if len(x) == 2 else [None] + x

def parse_posecode_joints(p_ind, p_kind, queries):
    focus = queries[p_kind]['focus_body_part'][p_ind]
    if focus is None:
        bp1_name = JOINT_NAMES[queries[p_kind]['joint_ids'][p_ind][0]]
        s1, b1 = parse_joint(bp1_name)
    else:
        s1, b1 = parse_joint(focus)
    if p_kind in POSECODE_KIND_FOCUS:
        s2, b2 = None, None
    else:
        bp2_name = JOINT_NAMES[queries[p_kind]['joint_ids'][p_ind][1]]
        s2, b2 = parse_joint(bp2_name)
    return s1, b1, s2, b2

def format_motioncodes(mi, mq):
    data, skipped = [], []
    for mk in mi:
        mint = mi[mk]
        for mc in range(len(mint)):
            if not mint[mc]: continue
            s1, b1, s2, b2 = parse_posecode_joints(mc, mk, mq)
            for m in range(len(mint[mc])):
                info = mint[mc][m]
                data.append([s1, b1,
                    {'spatial': info['spatial'], 'temporal': info['temporal'],
                     'start': info['start'], 'end': info['end'],
                     'posecode': [tuple(info['posecode'])],
                     'mc_info': {'m_kind': mk, 'mc_index': mc, 'focus_body_part': mq[mk]['focus_body_part'][mc]}},
                    s2, b2])
    return data, skipped

def format_posecodes(pi, pe, pq, spq):
    nb_frames = len(pi[list(pi.keys())[0]])
    data = [[] for _ in range(nb_frames)]
    for pk in pi:
        intptt = pi[pk]; elig = pe[pk]
        for pc in range(intptt.shape[1]):
            s1, b1, s2, b2 = parse_posecode_joints(pc, pk, pq)
            for f in range(nb_frames):
                if elig[f, pc] > 0:
                    data[f].append([s1, b1, intptt[f, pc].item(), s2, b2, [(pk, pc)]])
    # Super-posecodes
    sp_elig = pe['superPosecodes']
    for sp_ind, sp_id in enumerate(spq):
        fbp = spq[sp_id]['focus_body_part']
        if isinstance(fbp, tuple): fbp = fbp[0]
        s1, b1 = parse_joint(fbp)
        for f in range(nb_frames):
            if sp_elig[f, sp_ind] > 0:
                data[f].append([s1, b1, spq[sp_id]["intptt_id"], None, None, [('Super', sp_id)]])
    return data

################################################################################
# TIMECODES
################################################################################

class ChronologicalOrder:
    def __init__(self, bin_length):
        self.category_thresholds = TIMECODE_VALUES['ChronologicalOrder']['category_thresholds']
        self.bin_length = bin_length
    def eval(self, bin_diff): return bin_diff * self.bin_length
    def interprete(self, val):
        ct = self.category_thresholds
        ret = np.ones(val.shape) * len(ct)
        for i in range(len(ct)-1,-1,-1): ret[val < ct[i]] = i
        return ret.astype(int)

def assign_timecodes(motioncodes_binned, time_bin_info):
    t_op = ChronologicalOrder(0.5)
    last_tc = 0
    first_done = False
    for win in range(len(motioncodes_binned)):
        random.shuffle(motioncodes_binned[win])
        for m_ind in range(len(motioncodes_binned[win])):
            mc = motioncodes_binned[win][m_ind]
            if not isinstance(mc[2], list):
                mc[2] = {'spatial': mc[2]['spatial'], 'temporal': mc[2]['temporal'],
                         'start': mc[2]['start'], 'end': mc[2]['end'],
                         'posecode': mc[2].get('posecode', []), 'mc_info': mc[2].get('mc_info', {}),
                         'bin_diff': 0}
                mc = ['<SINGLE>', [[mc[0], mc[1]]], [mc[2]], [[mc[3], mc[4]]], set(mc[2]['posecode'])]
                motioncodes_binned[win][m_ind] = mc
            cur_tc = mc[2][0].get('bin_diff', 0) + win
            mc[2][0]['bin_diff'] = cur_tc - last_tc
            last_tc = mc[2][-1].get('bin_diff', 0) + win if len(mc[2]) > 1 else win
            bd = np.array([x.get('bin_diff', 0) for x in mc[2]])
            t_vals = t_op.eval(bd)
            t_intptt = t_op.interprete(t_vals)
            for t in range(len(t_intptt)):
                if not first_done:
                    mc[2][t]['chronological_order'] = None
                    first_done = True
                else:
                    mc[2][t]['chronological_order'] = int(t_intptt[t])
    return motioncodes_binned

################################################################################
# TEXT CONVERSION
################################################################################

DETERMINER_KEY = "<determiner>"
NO_VERB_KEY = "<no_verb>"
VERB_TENSE = "<TENSE>"
VELOCITY_TERM = '<VELOCITY_TERM>'
AND_VELOCITY_TERM = '<AND_VELOCITY_TERM>'
TIME_RELATION_TERM = '<TIME_RELATION>'
INITIAL_POSE_TERM = '<INITIAL_POSE_TERM>'
FINAL_POSE_TERM = '<FINAL_POSE_TERM>'
MULTIPLE_SUBJECTS_KEY = '<multiple_subjects>'
JOINT_BASED_AGGREG_KEY = '<joint_based_aggreg>'
sj_obj = "<SECOND_JOINT_OBJECT>"

subj_t = "{} %s"

SENTENCE_START = ['Someone', 'The person', 'A person', 'The body']

VELOCITY_ADJECTIVES = {
    'very_slow': ['very slowly', 'extremely slowly'],
    'slow': ['slowly', 'steadily', 'gently'],
    'moderate': [''],
    'fast': ['quickly', 'rapidly', 'briskly'],
    'very_fast': ['very fast', 'extremely fast', 'very rapidly'],
}

CHRONOLOGICAL_ADJECTIVE = {
    "preceding_a_moment": ["much earlier"],
    "soon_before": ["just before", "a moment earlier"],
    "shortly_before": ["moments prior", "a few seconds earlier"],
    "immediately_before": ["right before"],
    "simultaneously": ["at the same time", "meanwhile"],
    "immediately_after": ["right after", "a second later"],
    "shortly_after": ["a moment later", "shortly after"],
    "soon_after": ["soon after"],
    "after_a_moment": ["after a while", "eventually"],
}

# Motion templates (1-component)
MOTION_TEMPLATES_1 = {
    "significant_bend": [f"{subj_t} bends{{TENSE}} significantly{{VEL}}"],
    "moderate_bend": [f"{subj_t} bends{{TENSE}}{{VEL}}"],
    "slight_bend": [f"{subj_t} bends{{TENSE}} slightly{{VEL}}"],
    "no_action": [f"{subj_t} remains stationary"],
    "slight_extension": [f"{subj_t} extends{{TENSE}} slightly{{VEL}}"],
    "moderate_extension": [f"{subj_t} extends{{TENSE}}{{VEL}}"],
    "significant_extension": [f"{subj_t} extends{{TENSE}} significantly{{VEL}}"],
    "significant_closing": [f"{subj_t} draws{{TENSE}} significantly together{{VEL}}"],
    "moderate_closing": [f"{subj_t} moves{{TENSE}} together{{VEL}}"],
    "stationary": [f"{subj_t} stays{{TENSE}} still"],
    "moderate_spreading": [f"{subj_t} spreads{{TENSE}} apart{{VEL}}"],
    "significant_spreading": [f"{subj_t} spreads{{TENSE}} significantly apart{{VEL}}"],
    "left-to-right": [f"{subj_t} shifts{{TENSE}} from left to right{{VEL}}"],
    "right-to-left": [f"{subj_t} shifts{{TENSE}} from right to left{{VEL}}"],
    "above-to-below": [f"{subj_t} lowers{{TENSE}}{{VEL}}"],
    "below-to-above": [f"{subj_t} raises{{TENSE}}{{VEL}}"],
    "front-to-behind": [f"{subj_t} moves{{TENSE}} to behind{{VEL}}"],
    "behind-to-front": [f"{subj_t} comes{{TENSE}} forward{{VEL}}"],
    "very_long_left": [f"{subj_t} moves{{TENSE}} far to the left{{VEL}}"],
    "long_left": [f"{subj_t} moves{{TENSE}} considerably to the left{{VEL}}"],
    "moderate_left": [f"{subj_t} moves{{TENSE}} to the left{{VEL}}"],
    "short_left": [f"{subj_t} moves{{TENSE}} slightly to the left{{VEL}}"],
    "very_short_left": [f"{subj_t} shifts{{TENSE}} subtly to the left{{VEL}}"],
    "very_short_right": [f"{subj_t} shifts{{TENSE}} subtly to the right{{VEL}}"],
    "short_right": [f"{subj_t} moves{{TENSE}} slightly to the right{{VEL}}"],
    "moderate_right": [f"{subj_t} moves{{TENSE}} to the right{{VEL}}"],
    "long_right": [f"{subj_t} moves{{TENSE}} considerably to the right{{VEL}}"],
    "very_long_right": [f"{subj_t} moves{{TENSE}} far to the right{{VEL}}"],
    "very_long_down": [f"{subj_t} moves{{TENSE}} far downwards{{VEL}}"],
    "long_down": [f"{subj_t} moves{{TENSE}} considerably downwards{{VEL}}"],
    "moderate_down": [f"{subj_t} moves{{TENSE}} downwards{{VEL}}"],
    "short_down": [f"{subj_t} moves{{TENSE}} slightly downwards{{VEL}}"],
    "very_short_down": [f"{subj_t} shifts{{TENSE}} subtly downwards{{VEL}}"],
    "very_short_up": [f"{subj_t} shifts{{TENSE}} subtly upwards{{VEL}}"],
    "short_up": [f"{subj_t} moves{{TENSE}} slightly upwards{{VEL}}"],
    "moderate_up": [f"{subj_t} moves{{TENSE}} upwards{{VEL}}"],
    "long_up": [f"{subj_t} moves{{TENSE}} considerably upwards{{VEL}}"],
    "very_long_up": [f"{subj_t} moves{{TENSE}} far upwards{{VEL}}"],
    "very_long_backward": [f"{subj_t} moves{{TENSE}} far to the back{{VEL}}"],
    "long_backward": [f"{subj_t} moves{{TENSE}} considerably to the back{{VEL}}"],
    "moderate_backward": [f"{subj_t} moves{{TENSE}} to the back{{VEL}}"],
    "short_backward": [f"{subj_t} moves{{TENSE}} slightly to the back{{VEL}}"],
    "very_short_backward": [f"{subj_t} shifts{{TENSE}} subtly to the back{{VEL}}"],
    "very_short_forward": [f"{subj_t} shifts{{TENSE}} subtly to the front{{VEL}}"],
    "short_forward": [f"{subj_t} moves{{TENSE}} slightly to the front{{VEL}}"],
    "moderate_forward": [f"{subj_t} moves{{TENSE}} to the front{{VEL}}"],
    "long_forward": [f"{subj_t} moves{{TENSE}} considerably to the front{{VEL}}"],
    "very_long_forward": [f"{subj_t} moves{{TENSE}} far to the front{{VEL}}"],
    "significant_leaning_backward": [f"{subj_t} leans{{TENSE}} significantly backward{{VEL}}"],
    "moderate_leaning_backward": [f"{subj_t} leans{{TENSE}} backward{{VEL}}"],
    "slight_leaning_backward": [f"{subj_t} leans{{TENSE}} slightly backward{{VEL}}"],
    "slight_leaning_forward": [f"{subj_t} leans{{TENSE}} slightly forward{{VEL}}"],
    "moderate_leaning_forward": [f"{subj_t} leans{{TENSE}} forward{{VEL}}"],
    "significant_leaning_forward": [f"{subj_t} leans{{TENSE}} significantly forward{{VEL}}"],
    "significant_leaning_right": [f"{subj_t} leans{{TENSE}} significantly to the right{{VEL}}"],
    "moderate_leaning_right": [f"{subj_t} leans{{TENSE}} to the right{{VEL}}"],
    "slight_leaning_right": [f"{subj_t} leans{{TENSE}} slightly to the right{{VEL}}"],
    "slight_leaning_left": [f"{subj_t} leans{{TENSE}} slightly to the left{{VEL}}"],
    "moderate_leaning_left": [f"{subj_t} leans{{TENSE}} to the left{{VEL}}"],
    "significant_leaning_left": [f"{subj_t} leans{{TENSE}} significantly to the left{{VEL}}"],
    "significant_turn_clockwise": [f"{subj_t} turns{{TENSE}} significantly clockwise{{VEL}}"],
    "moderate_turn_clockwise": [f"{subj_t} turns{{TENSE}} clockwise{{VEL}}"],
    "slight_turn_clockwise": [f"{subj_t} turns{{TENSE}} slightly clockwise{{VEL}}"],
    "slight_turn_counterclockwise": [f"{subj_t} turns{{TENSE}} slightly counterclockwise{{VEL}}"],
    "moderate_turn_counterclockwise": [f"{subj_t} turns{{TENSE}} counterclockwise{{VEL}}"],
    "significant_turn_counterclockwise": [f"{subj_t} turns{{TENSE}} significantly counterclockwise{{VEL}}"],
}

# 2-component templates
MOTION_TEMPLATES_2 = {
    "significant_closing": [f"{subj_t} draws{{TENSE}} significantly near to {{OBJ}}{{VEL}}"],
    "moderate_closing": [f"{subj_t} gets{{TENSE}} closer to {{OBJ}}{{VEL}}"],
    "stationary": [f"{subj_t} stays{{TENSE}} still with {{OBJ}}"],
    "moderate_spreading": [f"{subj_t} spreads{{TENSE}} away from {{OBJ}}{{VEL}}"],
    "significant_spreading": [f"{subj_t} spreads{{TENSE}} significantly apart from {{OBJ}}{{VEL}}"],
    "left-to-right": [f"{subj_t} shifts{{TENSE}} from left to right of {{OBJ}}{{VEL}}"],
    "right-to-left": [f"{subj_t} shifts{{TENSE}} from right to left of {{OBJ}}{{VEL}}"],
    "above-to-below": [f"{subj_t} lowers{{TENSE}} from above {{OBJ}} to below{{VEL}}"],
    "below-to-above": [f"{subj_t} raises{{TENSE}} from below {{OBJ}} to above{{VEL}}"],
    "front-to-behind": [f"{subj_t} moves{{TENSE}} from front of {{OBJ}} to behind{{VEL}}"],
    "behind-to-front": [f"{subj_t} moves{{TENSE}} from behind {{OBJ}} to front{{VEL}}"],
}

SPELLING_FIXES = [('moveing','moving'),('closeing','closing'),('geting','getting'),
                  ('reachs','reaches'),('drawsing','drawing'),('bendsing','bending'),
                  ('extendsing','extending'),('spreadsing','spreading'),
                  ('shiftsing','shifting'),('lowersing','lowering'),('raisesing','raising'),
                  ('comesing','coming'),('leansing','leaning'),('turnsing','turning'),
                  ('getsing','getting'),('staysing','staying')]


def motioncode_to_sentence(bp1, verb, intptt_info, bp2, time_intptt):
    spatial_name = INTERPRETATION_SET_MOTION[intptt_info['spatial']]
    temporal_name = INTERPRETATION_SET_MOTION[intptt_info['temporal']] if intptt_info.get('temporal') is not None else ''

    # Time relation
    time_text = ''
    if time_intptt is not None and time_intptt in range(len(INTERPRETATION_SET_TIME)):
        time_key = INTERPRETATION_SET_TIME[time_intptt]
        time_text = random.choice(CHRONOLOGICAL_ADJECTIVE.get(time_key, [''])) + ', '

    # Velocity
    vel_text = ''
    if temporal_name and temporal_name in VELOCITY_ADJECTIVES:
        v = random.choice(VELOCITY_ADJECTIVES[temporal_name])
        if v: vel_text = ' ' + v

    # Pick template
    if bp2 is None:
        templates = MOTION_TEMPLATES_1.get(spatial_name, [f"{subj_t} performs{{TENSE}} {spatial_name}{{VEL}}"])
    else:
        templates = MOTION_TEMPLATES_2.get(spatial_name, MOTION_TEMPLATES_1.get(spatial_name, [f"{subj_t} performs{{TENSE}} {spatial_name}{{VEL}}"]))

    tmpl = random.choice(templates)
    # Fill subject — use % formatting since templates use {TENSE}/{VEL}/{OBJ} for later replacement
    try:
        tmpl = tmpl % verb
    except TypeError:
        pass
    # Replace {0} style subject placeholder manually
    tmpl = tmpl.replace("{} %s", bp1 + " " + verb).replace("{}", bp1)
    # Fill object
    tmpl = tmpl.replace('{OBJ}', bp2 if bp2 else '')
    # Fill tense
    tmpl = tmpl.replace('{TENSE}', '')
    # Fill velocity
    tmpl = tmpl.replace('{VEL}', vel_text)
    # Prepend time
    sentence = time_text + tmpl
    return sentence


def generate_description(motioncodes_binned, time_bin_info):
    """Convert binned motioncodes to a text description."""
    sentences = []
    for win in range(len(motioncodes_binned)):
        for mc in motioncodes_binned[win]:
            # Determine subject
            if mc[0] == '<SINGLE>':
                subjects = mc[1]
                interps = mc[2]
                objects = mc[3]
            else:
                subjects = mc[1] if isinstance(mc[1], list) else [[mc[0], mc[1]]]
                interps = mc[2] if isinstance(mc[2], list) else [mc[2]]
                objects = mc[3] if isinstance(mc[3], list) else [[mc[3], mc[4]]]

            for i, info in enumerate(interps):
                # Subject text
                if len(subjects) == 1:
                    s, b = subjects[0]
                    bp1 = f"the {s + ' ' if s and s not in [PLURAL_KEY, None] else ''}{b if b else 'body'}"
                    verb = 'is'
                else:
                    parts = [f"the {s + ' ' if s and s not in [PLURAL_KEY, None] else ''}{b}" for s, b in subjects]
                    bp1 = ', '.join(parts[:-1]) + ' and ' + parts[-1]
                    verb = 'are'

                # Object text
                bp2 = None
                if i < len(objects):
                    os_, ob_ = objects[i] if isinstance(objects[i], (list, tuple)) else (None, None)
                    if ob_:
                        bp2 = f"the {os_ + ' ' if os_ and os_ not in [PLURAL_KEY, None] else ''}{ob_}"

                time_rel = info.get('chronological_order', None)
                sent = motioncode_to_sentence(bp1, verb, info, bp2, time_rel)
                sentences.append(sent)

    # Join and clean
    text = '. '.join(sentences)
    text = re.sub(r'\s+', ' ', text).strip()
    for old, new in SPELLING_FIXES:
        text = text.replace(old, new)
    if text and not text.endswith('.'):
        text += '.'
    text = '. '.join(s.strip().capitalize() for s in text.split('. '))
    return text

################################################################################
# MAIN PIPELINE
################################################################################

def extract_description(coords, verbose=False):
    """
    Full pipeline using installed text2pose: coords → full captioning with
    aggregation, timecodes, text. No random skipping.
    Returns description string.
    """
    from text2pose.posescript.captioning import main as captioning_main
    import tempfile, shutil as _shutil

    coords = coords.to(device)
    if verbose: print("Preparing input...")
    coords = prepare_input(coords)

    if verbose: print("Running full captioning pipeline...")
    save_dir = tempfile.mkdtemp()
    try:
        result = captioning_main(
            coords, save_dir=save_dir, babel_info=False,
            simplified_captions=False,
            apply_transrel_ripple_effect=False,  # FIXME: disabled — crashes on posecode address lists
            apply_stat_ripple_effect=False,       # FIXME: disabled — depends on stat rules from PoseScript
            random_skip=False,
            motion_tracking=True, verbose=verbose, ablations=[])
        # result = (binning_details, motioncodes4vis, desc_non_agg, desc_agg)
        _, _, _, desc_agg = result
        description = " ".join([s for s in desc_agg if s.strip()])
    except Exception as e:
        if verbose: print(f"Pipeline error: {e}")
        import traceback; traceback.print_exc()
        description = ""
    finally:
        _shutil.rmtree(save_dir, ignore_errors=True)

    return description


################################################################################
# BATCH PROCESSING
################################################################################

def process_descriptions_to_folder(motion_dir, output_dir, max_files=None):
    """Process motions and save each description as a separate .txt file."""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(motion_dir) if f.endswith('.npy')])
    if max_files: files = files[:max_files]
    done, failed = 0, 0
    for fi, fname in enumerate(files):
        mid = os.path.splitext(fname)[0]
        if fi % 100 == 0: print(f"  Processing {fi}/{len(files)}...")
        try:
            coords = load_humanml3d(mid)
            if coords.shape[0] < 3: failed += 1; continue
            description = extract_description(coords, verbose=False)
            if description.strip():
                with open(os.path.join(output_dir, f"{mid}.txt"), 'w', encoding='utf-8') as f:
                    f.write(description)
                done += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Failed {mid}: {e}"); failed += 1
    print(f"\nSaved {done} descriptions to {output_dir}/, failed: {failed}")


def process_descriptions(motion_dir, texts_dir, output_json, max_files=None):
    """Process motions and generate descriptions. Save alongside original text."""
    files = sorted([f for f in os.listdir(motion_dir) if f.endswith('.npy')])
    if max_files: files = files[:max_files]

    results = {}
    for fi, fname in enumerate(files):
        mid = os.path.splitext(fname)[0]
        if fi % 100 == 0: print(f"  Processing {fi}/{len(files)}...")
        try:
            coords = load_humanml3d(mid)
            if coords.shape[0] < 3: continue
            result = extract_description(coords, verbose=False)

            # Get original text
            orig = ""
            txt_path = os.path.join(texts_dir, f"{mid}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                    if line: orig = line.split('#')[0].strip()

            results[mid] = {
                'generated': result['description'],
                'original': orig,
                'initial_pose': result['initial_pose'],
                'num_motioncodes': result['num_motioncodes'],
            }
        except Exception as e:
            print(f"  Failed {mid}: {e}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} descriptions to {output_json}")


################################################################################
# MAIN
################################################################################

if __name__ == "__main__":
    # Test with one motion
    print("=" * 60)
    print("Testing description generation...")
    print("=" * 60)

    try:
        coords = load_humanml3d("000000")
        description = extract_description(coords, verbose=True)
        print(f"\nGenerated description:")
        print(f"  {description}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()

    # Batch process style_motions to individual text files
    process_descriptions_to_folder('style_motions', 'style_descriptions')