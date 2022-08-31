import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.patches import PathPatch

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys 
sys.path.append("./parse_utils")
import time
from glob import glob
import pretty_midi 
import h5py
from decimal import Decimal, getcontext, ROUND_HALF_UP
import pandas as pd
import shutil
import copy
import subprocess
import importlib
import warnings
import gc
from djitw.djitw import dtw
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import scipy.stats as stats
from scipy.stats import pearsonr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from process_data \
    import make_align_matrix, make_note_based, make_onset_based_pick, \
           corrupt_to_onset, corrupt_to_beat, make_notenum_onehot, \
           make_pianoroll_x, get_vertical_position
from parse_utils import *
from parse_features \
    import parse_test_cond, parse_test_features, \
        parse_test_x_features, parse_test_y_features, GaussianFeatures
from model import PerformGenerator as pg


def pianoroll(x, m):
    roll = torch.matmul(
        x.transpose(1, 2), m).transpose(1, 2)
    return roll


def inverse_rendering_art(
    input_notes=None, save_dir=None, cond=None, features=None, 
    tempo=None, tempo_rate=1., same_onset_ind=None, savename=None, 
    save_perform=True, save_score=False, return_notes=False):

    midi_notes = input_notes
    tempo_rate = Decimal(str(tempo_rate))
    quarter = Decimal(str(60. / tempo)) * tempo_rate

    # make deadpan MIDI
    score_notes_norm = list()
    prev_note = None
    prev_new_note = None
    for i, note in enumerate(midi_notes): 

        if prev_note is None:
            note_dur = Decimal(str(note.end)) - Decimal(str(note.start))
            note_onset = Decimal(str(note.start))
            note_offset = note_onset + note_dur * tempo_rate
            new_note = pretty_midi.containers.Note(velocity=64,
                                                   pitch=note.pitch,
                                                   start=float(note_onset),
                                                   end=float(note_offset))
        elif prev_note is not None:
            if prev_note.start < note.start:
                note_dur = Decimal(str(note.end)) - Decimal(str(note.start))
                note_ioi = Decimal(str(note.start)) - Decimal(str(prev_note.start))
                note_onset = Decimal(str(prev_new_note.start)) + note_ioi * tempo_rate
                note_offset = note_onset + note_dur * tempo_rate
                new_note = pretty_midi.containers.Note(velocity=64,
                                                       pitch=note.pitch,
                                                       start=float(note_onset),
                                                       end=float(note_offset))
            elif prev_note.start == note.start:
                note_dur = Decimal(str(note.end)) - Decimal(str(note.start))
                note_onset = note_onset
                note_offset = note_onset + note_dur * tempo_rate
                new_note = pretty_midi.containers.Note(velocity=64,
                                                       pitch=note.pitch,
                                                       start=float(note_onset),
                                                       end=float(note_offset))
        score_notes_norm.append(new_note)
        prev_note = note
        prev_new_note = new_note
    score_notes_norm = make_midi_start_zero(score_notes_norm)

    if save_score is True:
        save_new_midi(score_notes_norm, new_midi_path=os.path.join(save_dir, "score_" + savename))

    prev_note = None
    prev_new_note = None

    _vel = features[0]
    _loc = features[1]
    _art = features[2]
    _ioi = features[3] # current - prev
    _ioi_expand = make_note_based(cond, _ioi, same_onset_ind=same_onset_ind)
    new_onsets = list()
    prev_onsets = list()
    score_iois = list()

    # get locs and iois
    for i in range(len(_vel)):
        note = score_notes_norm[i]
        each_in = cond[i]
        # loc_ratio = Decimal(str(_loc[i]))
        loc = Decimal(str(_loc[i]))
        ioi_ratio = Decimal(str(_ioi_expand[i]))
        same_onset = np.argmax(each_in[same_onset_ind[0]:same_onset_ind[1]]) # 0-False, 1-True
        # print(same_onset)

        dur_16th = Decimal(str(quarter)) / 4
        dur_16th = round(dur_16th, 3)

        if same_onset == 0: # False
            if prev_note is None: # first note
                old_ioi = dur_16th
                onset_for_ioi = -dur_16th
                
            elif prev_note is not None:
                onset_for_ioi = Decimal(str(np.mean(same_onset_notes)))
                # onset_for_ioi = Decimal(str(same_onset_notes[-1])) # highest_voice
                old_ioi = Decimal(str(note.start)) - Decimal(str(prev_note.start))

            new_ioi = old_ioi * ioi_ratio
            # loc = new_ioi * loc_ratio
            new_onset = onset_for_ioi + new_ioi + loc

            score_iois.append(old_ioi)
            prev_onsets.append(onset_for_ioi)

            # update prev onset
            same_onset_notes = [new_onset]

        elif same_onset == 1: # True
            # loc = new_ioi * loc_ratio
            new_onset = onset_for_ioi + new_ioi + loc
            same_onset_notes.append(new_onset)  
        
        new_onsets.append(new_onset)
        prev_note = note
        prev_new_note = new_note
    
    
    onset_for_ioi = Decimal(str(np.mean(same_onset_notes)))
    prev_onsets.append(onset_for_ioi)
    mean_onsets = prev_onsets[1:] # current mean onset
    # assert len(mean_onsets) == len(_ioi)
    
    j = -1
    rendered_notes = list()
    for i in range(len(_vel)):
        note = score_notes_norm[i]
        each_in = cond[i]
        vel = _vel[i]
        # loc = Decimal(str(_loc[i]))
        art = Decimal(str(_art[i]))
        new_onset = new_onsets[i]
        same_onset = np.argmax(each_in[same_onset_ind[0]:same_onset_ind[1]]) # 0-False, 1-True
        if same_onset == 0:
            j += 1
            mean_onset = mean_onsets[j]
        elif same_onset == 1:
            mean_onset = mean_onset
        # print(mean_onset)

        new_vel = np.min([int(vel), 127])

        if i < len(_vel)-1:
            if j < len(mean_onsets)-1:
                next_mean_onset = Decimal(str(mean_onsets[j+1]))
                score_ioi = score_iois[j+1]
                ioi2 = Decimal(str(next_mean_onset - mean_onset))
            elif j == len(mean_onsets)-1:
                ioi2 = score_ioi
            ioi2_ratio = ioi2 / score_ioi
            # print(ioi2_ratio)
            dur_ratio = ioi2_ratio * art
            # print(ioi2_ratio, art)
        elif i == len(_vel)-1:
            dur_ratio = Decimal(1.)
        new_dur = (Decimal(str(note.end)) - Decimal(str(note.start))) * dur_ratio
        # print(dur_ratio, (Decimal(str(note.end)) - Decimal(str(note.start))))
        new_dur = np.max([Decimal(0.026), new_dur])
        new_offset = new_onset + new_dur

        new_note = pretty_midi.containers.Note(velocity=new_vel,
                                               pitch=note.pitch,
                                               start=float(new_onset),
                                               end=float(new_offset))

        rendered_notes.append(new_note)
    
    rendered_notes = make_midi_start_zero(rendered_notes)

    if save_perform == True:
        save_new_midi(rendered_notes, new_midi_path=os.path.join(
            save_dir, savename))
    if return_notes == True:
        return rendered_notes


def inverse_rendering_art_note(
    input_notes=None, save_dir=None, cond=None, features=None, 
    tempo=None, tempo_rate=1., same_onset_ind=None, savename=None, 
    save_perform=True, save_score=False, return_notes=False):

    midi_notes = input_notes
    tempo_rate = Decimal(str(tempo_rate))
    quarter = Decimal(str(60. / tempo)) * tempo_rate

    # make deadpan MIDI (BPM 120)
    score_notes_norm = list()
    prev_note = None
    prev_new_note = None
    for i, note in enumerate(midi_notes): 

        if prev_note is None:
            note_dur = Decimal(str(note.end)) - Decimal(str(note.start))
            note_onset = Decimal(str(note.start))
            note_offset = note_onset + note_dur * tempo_rate
            new_note = pretty_midi.containers.Note(velocity=64,
                                                   pitch=note.pitch,
                                                   start=float(note_onset),
                                                   end=float(note_offset))
        elif prev_note is not None:
            if prev_note.start < note.start:
                note_dur = Decimal(str(note.end)) - Decimal(str(note.start))
                note_ioi = Decimal(str(note.start)) - Decimal(str(prev_note.start))
                note_onset = Decimal(str(prev_new_note.start)) + note_ioi * tempo_rate
                note_offset = note_onset + note_dur * tempo_rate
                new_note = pretty_midi.containers.Note(velocity=64,
                                                       pitch=note.pitch,
                                                       start=float(note_onset),
                                                       end=float(note_offset))
            elif prev_note.start == note.start:
                note_dur = Decimal(str(note.end)) - Decimal(str(note.start))
                note_onset = note_onset
                note_offset = note_onset + note_dur * tempo_rate
                new_note = pretty_midi.containers.Note(velocity=64,
                                                       pitch=note.pitch,
                                                       start=float(note_onset),
                                                       end=float(note_offset))
        score_notes_norm.append(new_note)
        prev_note = note
        prev_new_note = new_note
    score_notes_norm = make_midi_start_zero(score_notes_norm)

    if save_score is True:
        save_new_midi(score_notes_norm, new_midi_path=os.path.join(save_dir, "score_" + savename))

    prev_note = None
    prev_new_note = None

    _vel = features[0]
    _art = features[1]
    _ioi = features[2] # current - prev
    new_onsets = list()
    prev_onsets = list()
    score_iois = list()

    # get locs and iois
    for i in range(len(_vel)):
        note = score_notes_norm[i]
        each_in = cond[i]
        # loc_ratio = Decimal(str(_loc[i]))
        # loc = Decimal(str(_loc[i]))
        ioi_ratio = Decimal(str(_ioi[i]))
        same_onset = np.argmax(each_in[same_onset_ind[0]:same_onset_ind[1]]) # 0-False, 1-True
        # print(same_onset)

        dur_16th = Decimal(str(quarter)) / 4
        dur_16th = round(dur_16th, 3)

        if same_onset == 0: # False
            if prev_note is None: # first note
                old_ioi = dur_16th
                onset_for_ioi = -dur_16th
                
            elif prev_note is not None:
                onset_for_ioi = Decimal(str(np.mean(same_onset_notes)))
                # onset_for_ioi = Decimal(str(same_onset_notes[-1])) # highest_voice
                old_ioi = Decimal(str(note.start)) - Decimal(str(prev_note.start))

            new_ioi = old_ioi * ioi_ratio
            # loc = new_ioi * loc_ratio
            new_onset = onset_for_ioi + new_ioi

            score_iois.append(old_ioi)
            prev_onsets.append(onset_for_ioi)

            # update prev onset
            same_onset_notes = [new_onset]

        elif same_onset == 1: # True
            # loc = new_ioi * loc_ratio
            new_ioi = old_ioi * ioi_ratio
            new_onset = onset_for_ioi + new_ioi
            same_onset_notes.append(new_onset)  
        
        new_onsets.append(new_onset)
        prev_note = note
        prev_new_note = new_note
    
    
    onset_for_ioi = Decimal(str(np.mean(same_onset_notes)))
    prev_onsets.append(onset_for_ioi)
    mean_onsets = prev_onsets[1:] # current mean onset
    # assert len(mean_onsets) == len(_ioi)
    
    j = -1
    rendered_notes = list()
    prev_dur_ratio = None
    for i in range(len(_vel)):
        note = score_notes_norm[i]
        each_in = cond[i]
        vel = _vel[i]
        # loc = Decimal(str(_loc[i]))
        art = Decimal(str(_art[i]))
        new_onset = new_onsets[i]
        same_onset = np.argmax(each_in[same_onset_ind[0]:same_onset_ind[1]]) # 0-False, 1-True
        if same_onset == 0:
            j += 1
            mean_onset = mean_onsets[j]
        elif same_onset == 1:
            mean_onset = mean_onset
        # print(mean_onset)

        new_vel = np.min([int(round(vel)), 127])

        if i < len(_vel)-1:
            if j < len(mean_onsets)-1:
                next_mean_onset = Decimal(str(mean_onsets[j+1]))
                score_ioi = score_iois[j+1]
                ioi2 = Decimal(str(next_mean_onset - new_onset))
            elif j == len(mean_onsets)-1:
                ioi2 = score_ioi
            ioi2_ratio = ioi2 / score_ioi
            # print(ioi2_ratio)
            dur_ratio = ioi2_ratio * art
            # print(ioi2_ratio, art)
        elif i == len(_vel)-1:
            if same_onset == 1:
                dur_ratio = prev_dur_ratio
            elif same_onset == 0:
                dur_ratio = Decimal(1.)
        new_dur = (Decimal(str(note.end)) - Decimal(str(note.start))) * dur_ratio
        # print(i, dur_ratio, note.pitch, Decimal(str(note.end)) - Decimal(str(note.start)))
        new_dur = np.max([Decimal(0.026), new_dur])
        new_offset = new_onset + new_dur

        new_note = pretty_midi.containers.Note(velocity=new_vel,
                                               pitch=note.pitch,
                                               start=float(new_onset),
                                               end=float(new_offset))

        rendered_notes.append(new_note)
        prev_dur_ratio = dur_ratio
    
    rendered_notes = make_midi_start_zero(rendered_notes)

    if save_perform == True:
        save_new_midi(rendered_notes, new_midi_path=os.path.join(
            save_dir, savename))
    if return_notes == True:
        return rendered_notes


def inverse_feature(
    sampled_y, sampled_y2, art=False, numpy=False, interp=None, to_raw=True):
    vel = sampled_y[0,:,0]
    mic = sampled_y[0,:,1]
    dur = sampled_y[0,:,2]
    ioi = sampled_y2[0,:,-1] # group-wise

    if numpy is False:
        vel = vel.cpu().data.numpy()
        mic = mic.cpu().data.numpy()
        dur = dur.cpu().data.numpy()
        ioi = ioi.cpu().data.numpy()

    if interp == "tanh":
        vel = np.interp(vel, [-1, 1], [1, 127])
        mic = np.interp(mic, [-1, 1], [-0.1, 0.1])
        if art is True:
            dur = np.interp(dur, [-1, 1], [-0.6, 0.6])
        else:
            dur = np.interp(dur, [-1, 1], [-0.6, 0.6])
        ioi = np.interp(ioi, [-1, 1], [-0.9, 0.9])

    if to_raw is True:
        vel = vel
        mic = mic
        dur = np.power(10, dur)
        ioi = np.power(10, ioi)

    return vel, mic, dur, ioi


def get_feature(
    y, x=None, art=False, same_onset_ind=None):
    _y = y.copy()
    vel = _y[:,0]
    mic = _y[:,1]
    dur = _y[:,2]
    ioi1 = _y[:,3]
    ioi2 = _y[:,4]

    _vel = vel 
    _mic = mic
    if art is True:
        _art = np.log10(dur) - np.log10(ioi2)
        _dur = _art
    else:
        _dur = np.log10(dur)       
    _ioi = np.log10(ioi1) # ioi 1

    _y[:,0] = np.interp(_vel, [1, 127], [-1, 1])
    _y[:,1] = np.interp(_mic, [-0.1, 0.1], [-1, 1])
    if art is True:
        _y[:,2] = np.interp(_dur, [-0.6, 0.6], [-1, 1])
    else: # dur ratio
        _y[:,2] = np.interp(_dur, [-0.6, 0.6], [-1, 1])
    _y[:,3] = np.interp(_ioi, [-0.9, 0.9], [-1, 1])
      
    return _y[:,:4]


def features_by_condition(
    y, x=None, cond=None, art=None, ratio=0.2, same_onset_ind=None):

    # copy original features
    y_ = y.copy()

    # manipulate performance 
    if cond is not None:
        if type(cond) != list: 
            cond = [cond]

        # tempo
        if "fast" in cond:
            y_[:,1:3] = y_[:,1:3] * (1-ratio) # ld
            y_[:,3] = y_[:,3] * (1-ratio) # ioi1
            y_[:,4] = y_[:,4] * (1-ratio) # ioi2
        elif "slow" in cond:
            y_[:,1:3] = y_[:,1:3] * (1+ratio) # ld
            y_[:,3] = y_[:,3] * (1+ratio) # ioi1
            y_[:,4] = y_[:,4] * (1+ratio) # ioi2
        
        # dynamics
        if "quiet" in cond:
            y_[:,0] = y_[:,0] * ((1-ratio)) # v
        elif "loud" in cond:
            y_[:,0] = y_[:,0] * (1+ratio) # v
        
        # articulation
        if "stac" in cond:
            y_[:,2] = y_[:,2] * (1-ratio) # d
        elif "legato" in cond:
            y_[:,2] = y_[:,2] * (1+ratio) # d    

    elif cond is None:
        y_ = y 

    # get features
    y_new = get_feature(y_, x=x, art=art, same_onset_ind=same_onset_ind)

    return y_new


def inverse_feature_note(
    sampled_y, art=False, numpy=False, interp=None, to_raw=True):
    vel = sampled_y[0,:,0]
    dur = sampled_y[0,:,1]
    ioi = sampled_y[0,:,-1] # note-wise

    if numpy is False:
        vel = vel.cpu().data.numpy()
        dur = dur.cpu().data.numpy()
        ioi = ioi.cpu().data.numpy()

    if interp == "tanh":
        vel = np.interp(vel, [-1, 1], [1, 127])
        if art is True:
            dur = np.interp(dur, [-1, 1], [-0.6, 0.6])
        else:
            dur = np.interp(dur, [-1, 1], [-0.6, 0.6])
        ioi = np.interp(ioi, [-1, 1], [-0.9, 0.9])

    if to_raw is True:
        vel = vel
        dur = np.power(10, dur)
        ioi = np.power(10, ioi)

    return vel, dur, ioi


def get_feature_note(
    y, x=None, art=False, same_onset_ind=None):
    _y = y.copy()
    vel = _y[:,0]
    dur = _y[:,1]
    ioi1 = _y[:,2]
    ioi2 = _y[:,3]

    _vel = vel
    if art is True:
        _art = np.log10(dur) - np.log10(ioi2)
        _dur = _art
    else:
        _dur = np.log10(dur)       
    _ioi = np.log10(ioi1) # ioi 1

    _y[:,0] = np.interp(_vel, [1, 127], [-1, 1])
    if art is True:
        _y[:,1] = np.interp(_dur, [-0.6, 0.6], [-1, 1])
    else: # dur ratio
        _y[:,1] = np.interp(_dur, [-0.6, 0.6], [-1, 1])
    _y[:,2] = np.interp(_ioi, [-0.9, 0.9], [-1, 1])
      
    return _y[:,:3]


def features_by_condition_note(
    y, x=None, cond=None, ratio=0.3, art=None, same_onset_ind=None):

    # copy original features
    y_ = y.copy()

    # manipulate performance 
    if cond is not None:
        if type(cond) != list: 
            cond = [cond]

        # tempo
        if "fast" in cond:
            y_[:,1] = y_[:,1] * (1-ratio) # ld
            y_[:,2] = y_[:,2] * (1-ratio) # ioi1
            y_[:,3] = y_[:,3] * (1-ratio) # ioi2
        elif "slow" in cond:
            y_[:,1] = y_[:,1] * (1+ratio) # ld
            y_[:,2] = y_[:,2] * (1+ratio) # ioi1
            y_[:,3] = y_[:,3] * (1+ratio) # ioi2
        
        # dynamics
        if "quiet" in cond:
            y_[:,0] = y_[:,0] * (1-ratio) # v
        elif "loud" in cond:
            y_[:,0] = y_[:,0] * (1+ratio) # v
        
        # articulation
        if "stac" in cond:
            y_[:,1] = y_[:,1] * (1-ratio) # d
        elif "legato" in cond:
            y_[:,1] = y_[:,1] * (1+ratio) # d    

    elif cond is None:
        y_ = y_

    # get features
    y_new = get_feature_note(y_, x=x, art=art, same_onset_ind=same_onset_ind)

    return y_new



class GetData(object):

    def __init__(self, null_tempo=120, same_onset_ind=None, stat=np.mean):
        self.soi = same_onset_ind
        self.null_tempo = null_tempo
        self.stat = stat
        self.clab_note = None
        self.zlab_note = None

    def file2data(self, files, measures, mode=None, pair_path=None, save_mid=False):

        xml, score, perform = files
        p_name = '__'.join(score.split("/")[-3:-1])

        # parse data
        if measures is None:
            first_measure = 0 
        else:
            first_measure = measures[0]
        tempo, time_sig, key_sig = get_signatures_from_xml(xml, first_measure)
        # _, notes = parse_test_x_features(score=score, tempo=tempo)
        test_y, test_x, pairs, note_ind = \
            parse_test_y_features(xml=xml, score=score, perform=perform,
            mode=mode, measures=measures, tempo=tempo, pair_path=pair_path,
            null_tempo=self.null_tempo, same_onset_ind=self.soi)
        cond = parse_test_cond(pair=pairs, small_ver=False, 
            tempo=tempo, time_sig=time_sig, key_sig=key_sig) 
        # test_notes = [notes[n] for n in note_ind]
        # test_notes = make_midi_start_zero(test_notes)
        xml_notes_raw = [p['xml_note'][1] for p in pairs]
        xml_notes = xml_to_midi_notes(xml_notes_raw)
    
        # ## make note-based roll (Maezawa et al., 2020)
        # xml_dict = dict()
        # for n in [p['xml_note'] for p in pairs]:
        #     xml_dict[n[0]] = n[1]
        # _unit = (60 / tempo) * (1/8) # 1/32
        # roll, _, _ = make_pianoroll(xml_notes, unit=_unit)
        # # 7-beat-radius pianoroll 
        # in_ind = note_ind
        # roll_list = list()
        # rad = 7 # 7 beats in 1/32th note resolution
        # beats = rad * 8
        # '''
        # 1 beat in 32 frames --> 8 frames (quarter)
        # 3 beats backwards + 4 beats forwards 
        # --> 56 * 88 
        # '''
        # for ind in in_ind: 
        #     base = xml_dict[ind].note_duration.time_position
        #     base_ind = quantize_to_frame(base, unit=_unit)

        #     min_ind = np.max([0, base_ind - (beats//2)])
        #     max_ind = np.min([roll.shape[1], base_ind + (beats//2)])
        #     # print(base, min_ind, max_ind)
            
        #     rad_roll = np.zeros([88,56])
        #     sub_roll = roll[:,min_ind:max_ind]
        #     rad_roll[:,rad_roll.shape[1]-sub_roll.shape[1]:] = sub_roll
        #     roll_list.append(rad_roll.T)
        # roll_list = np.asarray(roll_list)

        # ## quantized dynamics/tempo (Maezawa et al., 2020)    
        # out_beat, out_beat_num = corrupt_to_beat(test_y, beat=cond[:,1:13])
        # out_beat_ma = moving_average(out_beat, win_len=17) # 8 beats back/forth
        # # tile averaged attr by note
        # out_beat_tile = list()
        # for v, n in zip(out_beat_ma, out_beat_num):
        #     out_beat_tile.append(np.tile(np.reshape(v, (1, -1)), (n, 1)))
        # out_beat_tile = np.concatenate(out_beat_tile, axis=0)
        # assert len(test_y) == len(out_beat_tile)
        # # quantize attr
        # '''
        # Ref: Maezawa et al., 2020 (Rendering Music Performance~)
        # '''
        # q_tempo = [q // 30 for q in (120 * (1/out_beat_tile[:,2]))] # features based on 120 BPM
        # q_tempo = np.clip(q_tempo, 1, 10)
        # q_dynamics = [q // 10 for q in out_beat_tile[:,0]]
        # q_dynamics = np.clip(q_dynamics, 1, 12)
        # dyn = np.zeros([len(q_tempo), 12])
        # tpo = np.zeros([len(q_tempo), 10])
        # for i in range(len(q_tempo)):
        #     dyn[i, int(q_dynamics[i])-1] = 1
        #     tpo[i, int(q_tempo[i])-1] = 1

        # make more features
        test_r = make_pianoroll_x(test_x, same_onset_ind=self.soi)
        test_m = make_align_matrix(test_x, test_r, same_onset_ind=self.soi)
        notenum = make_notenum_onehot(test_m)
        notepos = get_vertical_position(test_x, same_onset_ind=self.soi)
        test_x = np.concatenate([test_x, notenum, notepos, cond[:,19:23]], axis=-1) #, cond

        perform_notes = [p['perform_midi'][1] for p in pairs \
            if p['perform_midi'] is not None]
        perform_notes = make_midi_start_zero(perform_notes)

        # save perform midi
        if save_mid is True:
            save_new_midi(xml_notes, 
                new_midi_path='orig_{}_mm{}-{}_score.mid'.format(p_name, measures[0], measures[1]))
            # save_new_midi(test_notes, 
                # new_midi_path='orig_{}_mm{}-{}_score.mid'.format(p_name, measures[0], measures[1]))
            save_new_midi(perform_notes, 
                new_midi_path='orig_{}_mm{}-{}_perform.mid'.format(p_name, measures[0], measures[1]))

        return test_x, test_y, test_m, pairs, xml_notes, perform_notes, tempo, p_name

    def file2data_noY(self, files, measures, mode=None, save_mid=False):

        xml, score = files
        p_name = '__'.join(score.split("/")[-3:-1])

        # parse data
        if measures is None:
            first_measure = 0 
        else:
            first_measure = measures[0]
        tempo, time_sig, key_sig = get_signatures_from_xml(xml, first_measure)
        test_x, pairs, note_ind = \
            parse_test_features(xml=xml, score=score,
            mode=mode, measures=measures, tempo=tempo,
            null_tempo=self.null_tempo, same_onset_ind=self.soi)
        cond = parse_test_cond(pair=pairs, small_ver=False, 
            tempo=tempo, time_sig=time_sig, key_sig=key_sig) 
        # test_notes = [notes[n] for n in note_ind]
        # test_notes = make_midi_start_zero(test_notes)
        xml_notes_raw = [p['xml_note'][1] for p in pairs]
        xml_notes = xml_to_midi_notes(xml_notes_raw)
    
        # ## make note-based roll (Maezawa et al., 2020)
        # xml_dict = dict()
        # for n in [p['xml_note'] for p in pairs]:
        #     xml_dict[n[0]] = n[1]
        # _unit = (60 / tempo) * (1/8) # 1/32
        # roll, _, _ = make_pianoroll(xml_notes, unit=_unit)
        # # 7-beat-radius pianoroll 
        # in_ind = note_ind
        # roll_list = list()
        # rad = 7 # 7 beats in 1/32th note resolution
        # beats = rad * 8
        # '''
        # 1 beat in 32 frames --> 8 frames (quarter)
        # 3 beats backwards + 4 beats forwards 
        # --> 56 * 88 
        # '''
        # for ind in in_ind: 
        #     base = xml_dict[ind].note_duration.time_position
        #     base_ind = quantize_to_frame(base, unit=_unit)

        #     min_ind = np.max([0, base_ind - (beats//2)])
        #     max_ind = np.min([roll.shape[1], base_ind + (beats//2)])
        #     # print(base, min_ind, max_ind)
            
        #     rad_roll = np.zeros([88,56])
        #     sub_roll = roll[:,min_ind:max_ind]
        #     rad_roll[:,rad_roll.shape[1]-sub_roll.shape[1]:] = sub_roll
        #     roll_list.append(rad_roll.T)
        # roll_list = np.asarray(roll_list)

        # ## quantized dynamics/tempo (Maezawa et al., 2020)    
        # out_beat, out_beat_num = corrupt_to_beat(test_y, beat=cond[:,1:13])
        # out_beat_ma = moving_average(out_beat, win_len=17) # 8 beats back/forth
        # # tile averaged attr by note
        # out_beat_tile = list()
        # for v, n in zip(out_beat_ma, out_beat_num):
        #     out_beat_tile.append(np.tile(np.reshape(v, (1, -1)), (n, 1)))
        # out_beat_tile = np.concatenate(out_beat_tile, axis=0)
        # assert len(test_y) == len(out_beat_tile)
        # # quantize attr
        # '''
        # Ref: Maezawa et al., 2020 (Rendering Music Performance~)
        # '''
        # q_tempo = [q // 30 for q in (120 * (1/out_beat_tile[:,2]))] # features based on 120 BPM
        # q_tempo = np.clip(q_tempo, 1, 10)
        # q_dynamics = [q // 10 for q in out_beat_tile[:,0]]
        # q_dynamics = np.clip(q_dynamics, 1, 12)
        # dyn = np.zeros([len(q_tempo), 12])
        # tpo = np.zeros([len(q_tempo), 10])
        # for i in range(len(q_tempo)):
        #     dyn[i, int(q_dynamics[i])-1] = 1
        #     tpo[i, int(q_tempo[i])-1] = 1

        # make more features
        test_r = make_pianoroll_x(test_x, same_onset_ind=self.soi)
        test_m = make_align_matrix(test_x, test_r, same_onset_ind=self.soi)
        notenum = make_notenum_onehot(test_m)
        notepos = get_vertical_position(test_x, same_onset_ind=self.soi)
        test_x = np.concatenate([test_x, notenum, notepos, cond[:,19:23]], axis=-1) #, cond

        # save perform midi
        if save_mid is True:
            save_new_midi(xml_notes, 
                new_midi_path='orig_{}_mm{}-{}_xml.mid'.format(p_name, measures[0], measures[1]))
            save_new_midi(test_notes, 
                new_midi_path='orig_{}_mm{}-{}_score.mid'.format(p_name, measures[0], measures[1]))

        return test_x, test_m, pairs, xml_notes, tempo, p_name


    def data2input(self, test_x, test_y, test_m, cond=None, ratio=0.3, N=4, art=True, mode=None, device=None):
        if mode == "group":
            y = features_by_condition(
                test_y, x=test_x, cond=cond, art=art, ratio=ratio, same_onset_ind=self.soi)  
            vel, mic, art, ioi = y[:,0], y[:,1], y[:,2], y[:,3]
        elif mode == "note":
            y = features_by_condition_note(
                test_y, x=test_x, cond=cond, art=art, ratio=ratio, same_onset_ind=self.soi)  
            vel, art, ioi = y[:,0], y[:,1], y[:,2]
        y2 = corrupt_to_onset(
            test_x, np.stack([vel, art, ioi], axis=-1), stat=self.stat, same_onset_ind=self.soi)
        # clab = y2[0]
        # clab = np.mean(y, axis=0)
        # clab = np.stack([quantize(n, unit=0.2) for n in np.mean(y, axis=0)], axis=0)
        # clab = moving_average(y2, win_len=5, stat=np.mean, half=False)
        if N > 1:
            clab = poly_predict(y2, N=N)
            clab_ = poly_predict(y, N=N)
        elif N == 1:
            clab, _, _ = linear_predict(y2)
            clab_, _, _ = linear_predict(y)
        # clab_ = copy.deepcopy(np.matmul(clab.T, test_m.T).T) 
        # zlab_ = copy.deepcopy(y)
        # zlab = np.sign(zlab_ - clab_)

        # c label
        clab_m = np.mean(clab, axis=0)
        clab_m = np.where(clab_m > 0, np.ones_like(clab_m), np.zeros_like(clab_m))
        v, a, i = clab_m[0], clab_m[1], clab_m[2]
        label = v * 1 + a * 2 + i * 4
        assert label >= 0 and label < 8
        label = np.array(label)

        # convert to tensor
        test_x_ = torch.from_numpy(test_x.astype(np.float32)).to(device).unsqueeze(0)
        test_m_ = torch.from_numpy(test_m.astype(np.float32)).to(device).unsqueeze(0)
        test_y_ = torch.from_numpy(y.astype(np.float32)).to(device).unsqueeze(0)
        test_y2_ = torch.from_numpy(y2.astype(np.float32)).to(device).unsqueeze(0)
        test_clab_ = torch.from_numpy(clab.astype(np.float32)).to(device).unsqueeze(0)
        test_clab2_ = torch.from_numpy(clab_.astype(np.float32)).to(device).unsqueeze(0)
        test_lab = torch.from_numpy(label.astype(np.int32)).to(device).unsqueeze(0)
        
        self.clab_note = test_clab2_
        self.lab = test_lab
        # self.zlab_note = test_zlab_

        return test_x_, test_y_, test_y2_, test_m_, test_clab_

    def data2input_noY(self, test_x, test_m, mode=None, device=None):

        # convert to tensor
        test_x_ = torch.from_numpy(test_x.astype(np.float32)).to(device).unsqueeze(0)
        test_m_ = torch.from_numpy(test_m.astype(np.float32)).to(device).unsqueeze(0)

        return test_x_, test_m_



def qualitative_results(
    song_name=None, measures=None, device_num=None,
    model_num=None, exp_num=None, epoch_num=None, same_onset_ind=[110,112]):

    pair_path = None

    # test data paths
    # song_name = "Schubert_Impromptu_op.90_D.899__4"
    # song_name = "Mozart_Piano_Sonatas__11-3"
    # song_name = "Beethoven_Piano_Sonatas__8-2"
    # song_name = "Beethoven_Piano_Sonatas__14-3"
    same_onset_ind = [110,112]
    # measures = [209, 114]
    # measures = [217,232]
    # measures = [149,150]
    song_name_ =  '/'.join(song_name.split('__'))
    parent_path = "/data/asap_dataset/exp_data/listening_test/raw"
    perform = [p for p in sorted(glob(os.path.join(parent_path, "{}/01/*.mid".format(
        song_name_)))) if "cleaned" not in p][0]
    xml = os.path.join(parent_path, "{}/musicxml_cleaned_plain.musicxml".format(song_name_))
    score = os.path.join(parent_path, "{}/score_plain.mid".format(song_name_))
    pair = os.path.join(os.path.dirname(perform), 'xml_score_perform_pairs.npy')
    if os.path.exists(pair) is True:
        pair_path = pair 
    else:
        pair_path = None
    # pair_path = None

    ## LOAD DATA ##
    # model_num = 'exp1971'
    # exp_num = 'exp1971'
    # epoch_num = '100'
    print()
    print()
    print("############# TEST ##############")
    print("> MODEL: {}\n> EXP: {}".format(model_num, exp_num))
    print("#################################")
    print()
    print("** Load Data **")

    null_tempo = 120
    get_data = GetData(null_tempo=null_tempo, 
                       same_onset_ind=same_onset_ind,
                       stat=np.mean)
    test_x, test_y, test_m, pairs, xml_notes, perform_notes, tempo, p_name = get_data.file2data(
        files=[xml, score, perform], measures=measures, mode="note", pair_path=pair_path)
    tempo_rate = tempo / null_tempo
    print("     > tempo: {}".format(tempo))
    print()


    ## LOAD MODEL ## 
    print("** Load model **")
    module_name = "piano_cvae_model2_torch_{}".format(model_num)
    model = importlib.import_module(module_name)
    Generator = model.PerformGenerator
    note2group = model.Note2Group()

    model_path = "./model_cvae/piano_cvae_ckpt_{}".format(exp_num) # 
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")
    model = Generator(device=device)
    
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    loss_val = checkpoint["loss_val"]
    checkpoint_num = len(loss_val)
    model.eval()

    ## INFER LATENTS BY CONDITIONS ##
    indices = ["fast", "slow", "loud", "quiet", "stac", "legato", "neutral"]
    c_dict = dict()
    interp = 'tanh'
    n = 0

    y_vel = test_y[:,0]
    y_ioi = test_y[:,2]
    y_art = np.power(10, np.log10(test_y[:,1]) - np.log10(test_y[:,3]))

    for ind in indices:
        # get input data
        test_inputs_ = get_data.data2input(test_x, test_y, test_m, 
            cond=ind, art=True, mode="note", device=device)
        test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs_
        test_inputs = test_inputs_

        _, _, _, \
        _, _, z_moments, \
        c, z, _, _, \
        _, _, _ = model(*test_inputs)

        c_dict[ind] = c


    ## INTERPOLATION ##
    interp_t = dict()
    interp_d = dict()
    interp_a = dict()
    ## CONTROLLING FADER ##
    t = len(test_y2_[0])
    start, end = measures 
    middle = start + ((end+1 - start) // 2)
    pairs_onset = make_onset_pairs(pairs, fmt="xml")
    for n, onset in enumerate(pairs_onset):
        if onset[0]['xml_note'][1].measure_number == middle-1:
            break
    sketch1 = np.linspace(3, -3, num=t, endpoint=True)
    
    # assert len(add_seq_) == t
    # add_seq = torch.from_numpy(add_seq_).to(device).float()

    # dynamics
    c_loud = c_dict["loud"]
    c_quiet = c_dict["quiet"]
    # 0 -> 1 : loud to quiet
    for a in range(3):
        alpha = a / 2.
        # get latent variable
        c_seed_ = alpha * c_quiet + (1-alpha) * c_loud 
        _, _, z_, sampled = model.sample(
            test_x_, test_m_, c_=c_seed_, z_=z)

        # inverse to feature
        vel, dur, ioi = \
            inverse_feature_note(sampled, art=True, numpy=False, interp=interp)
        interp_d[a] = [vel, dur, ioi]
    print("     > sampled by dynamics")

    # articulations
    c_stac = c_dict["stac"]
    c_leg = c_dict["legato"]
    # 0 -> 1 : staccato to legato
    for a in range(3):
        alpha = a / 2.
        # get latent variable
        c_seed_ = alpha * c_leg + (1-alpha) * c_stac 
        _, _, z_, sampled = model.sample(
            test_x_, test_m_, c_=c_seed_, z_=z)

        # inverse to feature
        vel, dur, ioi = \
            inverse_feature_note(sampled, art=True, numpy=False, interp=interp)
        interp_a[a] = [vel, dur, ioi]
    print("     > sampled by articulations")

    # tempo
    c_fast = c_dict["fast"]
    c_slow = c_dict["slow"]
    # 0 -> 1 : fast to slow
    for a in range(3):
        alpha = a / 2.
        # get latent variable
        c_seed_ = alpha * c_slow + (1-alpha) * c_fast
        _, _, z_, sampled = model.sample(
            test_x_, test_m_, c_=c_seed_, z_=z)

        # inverse to feature
        vel, dur, ioi = \
            inverse_feature_note(sampled, art=True, numpy=False, interp=interp)
        interp_t[a] = [vel, dur, ioi]
    print("     > sampled by tempo")


    vel_min = np.min([np.min([np.min(y_vel), np.min(f[0])]) for f in [interp_d[0], interp_d[1], interp_d[2]]])
    art_min = np.min([np.min([np.min(y_art), np.min(f[1])]) for f in [interp_a[0], interp_a[1], interp_a[2]]])
    ioi_min = np.min([np.min([np.min(y_ioi), np.min(f[2])]) for f in [interp_t[0], interp_t[1], interp_t[2]]])
    vel_max = np.max([np.max([np.max(y_vel), np.max(f[0])]) for f in [interp_d[0], interp_d[1], interp_d[2]]])
    art_max = np.max([np.max([np.max(y_art), np.max(f[1])]) for f in [interp_a[0], interp_a[1], interp_a[2]]])
    ioi_max = np.max([np.max([np.max(y_ioi), np.max(f[2])]) for f in [interp_t[0], interp_t[1], interp_t[2]]])

    vel_min = vel_min - 1
    art_min = art_min - 0.1
    ioi_min = ioi_min - 0.1
    vel_max = vel_max + 1
    art_max = art_max + 0.1
    ioi_max = ioi_max + 0.1

    # # sample various versions
    # c_seed_ = torch.empty_like(c_dict["neutral"]).copy_(c_dict["neutral"])
    # for a in range(3):
    #     _, _, z_, sampled = model.sample(test_x_, test_m_, c_=c_seed_) # c_=c_seed_, 
    #     # inverse to feature
    #     vel, dur, ioi = \
    #         inverse_feature_note(sampled, art=True, numpy=False, interp=interp)
    #     styles[a] = [alpha, [vel, dur, ioi], c_seed_, z_]
    # print("     > sampled multiple styles (neutral)")
    # print()

    # c_mean = torch.mean(c_dict["neutral"], dim=1).unsqueeze(1).repeat(1, t, 1)
    # c_rand = torch.empty_like(c_mean).copy_(c_mean)
    # # tempo
    # s_i = torch.empty_like(c_rand).copy_(c_rand)
    # s_i[:,:,8] = s_i[:,:,8] + add_seq.unsqueeze(0)
    # _, _, _, i_sampled = model.sample(
    #     test_x_, test_m_, c_=s_i, trunc=True, threshold=2) 
    # fader_i = inverse_feature_note(i_sampled, art=True, numpy=False, interp=interp)
    # # dynamics 
    # s_d = torch.empty_like(c_rand).copy_(c_rand)
    # s_d[:,:,0] = s_d[:,:,0] + add_seq.unsqueeze(0)
    # _, _, _, d_sampled = model.sample(
    #     test_x_, test_m_, c_=s_d, trunc=True, threshold=2) 
    # fader_d = inverse_feature_note(d_sampled, art=True, numpy=False, interp=interp)
    # # articulation
    # s_a = torch.empty_like(c_rand).copy_(c_rand)
    # s_a[:,:,4] = s_a[:,:,4] + add_seq.unsqueeze(0)
    # _, _, _, a_sampled = model.sample(
    #     test_x_, test_m_, c_=s_a, trunc=True, threshold=2) 
    # fader_a = inverse_feature_note(a_sampled, art=True, numpy=False, interp=interp)
    # print()

    # plt.figure(figsize=(10,12))
    # plt.subplot(311)
    # plt.title("Sampled(fader) velocity")
    # plt.plot(range(len(test_x)), y_vel, label="GT")
    # plt.plot(range(len(test_x)), fader_i[0], label="fader_ioi")
    # plt.plot(range(len(test_x)), fader_d[0], label="fader_vel")
    # plt.plot(range(len(test_x)), fader_a[0], label="fader_art")
    # plt.legend()
    # plt.subplot(312)
    # plt.title("Sampled(fader) articulation")
    # plt.plot(range(len(test_x)), y_art, label="GT")
    # plt.plot(range(len(test_x)), fader_i[1], label="fader_ioi")
    # plt.plot(range(len(test_x)), fader_d[1], label="fader_vel")
    # plt.plot(range(len(test_x)), fader_a[1], label="fader_art")
    # plt.ylim([0, 8])
    # plt.legend()
    # plt.subplot(313)
    # plt.title("Sampled(fader) IOI")
    # plt.plot(range(len(test_x)), y_ioi, label="GT")
    # plt.plot(range(len(test_x)), fader_i[2], label="fader_ioi")
    # plt.plot(range(len(test_x)), fader_d[2], label="fader_vel")
    # plt.plot(range(len(test_x)), fader_a[2], label="fader_art")
    # plt.ylim([0, 8])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join("./gen_fader_{}_{}_{}_mm{}-{}.png".format(song_name, exp_num, checkpoint_num, measures[0], measures[1])))


    ### PLOT ###
    ## CHANGING DYNAMICS ##
    plt.rcParams['font.size'] = 30
    plt.rcParams['axes.linewidth'] = 5
    plt.rcParams['lines.linewidth'] = 7
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, 
        figsize=(15,25)) #sharey=True,
    fig.subplots_adjust(
        left=0.12, right=0.99, bottom=0.33, top=0.99, 
        wspace=0.07, hspace=0.14)
    colormap = cm.copper
    normalize = mcolors.Normalize(vmin=0, vmax=5)

    ax1 = axs[0, 0]
    ax1.plot(range(len(test_x)), y_vel, color="gray", label="GT", linewidth=3)
    ax1.plot(range(len(test_x)), interp_d[0][0], color="#CD5C5C", label="quiet=0 / loud=1") #color=colormap(normalize(3)), 
    ax1.set_ylabel('MIDIVelocity', fontsize=65)
    ax1.axes.xaxis.set_visible(False)
    ax1.set_ylim([vel_min, vel_max])
    
    ax2 = axs[1, 0]
    ax2.plot(range(len(test_x)), y_art, color="gray", label="GT", linewidth=3)
    ax2.plot(range(len(test_x)), interp_d[0][1], color="#CD5C5C", label="quiet=0 / loud=1")
    ax2.set_ylabel('Articulation', fontsize=65) 
    ax2.axes.xaxis.set_visible(False)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_ylim([art_min, art_max])
    
    ax3 = axs[2, 0]
    ax3.plot(range(len(test_x)), y_ioi, color="gray", label="GT", linewidth=3)
    ax3.plot(range(len(test_x)), interp_d[0][2], color="#CD5C5C", label="quiet=0 / loud=1")
    ax3.set_ylabel('IOIRatio', fontsize=65)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.set_ylim([ioi_min, ioi_max])
    # ax3.set_xlabel("Note Index", fontsize=65)
    # ax3.axes.xaxis.set_visible(False)

    ax4 = axs[0, 1]
    ax4.plot(range(len(test_x)), y_vel, color="gray", label="GT", linewidth=3)
    ax4.plot(range(len(test_x)), interp_d[1][0], color="#8B0000", label="quiet=0.5 / loud=0.5")
    ax4.axes.xaxis.set_visible(False)
    ax4.axes.yaxis.set_visible(False)
    ax4.set_ylim([vel_min, vel_max])

    ax5 = axs[1, 1]
    ax5.plot(range(len(test_x)), y_art, color="gray", label="GT", linewidth=3)
    ax5.plot(range(len(test_x)), interp_d[1][1], color="#8B0000", label="quiet=0.5 / loud=0.5")
    ax5.axes.xaxis.set_visible(False)
    ax5.axes.yaxis.set_visible(False)
    ax5.set_ylim([art_min, art_max])

    ax6 = axs[2, 1]
    ax6.plot(range(len(test_x)), y_ioi, color="gray", label="GT", linewidth=3)
    ax6.plot(range(len(test_x)), interp_d[1][2], color="#8B0000", label="quiet=0.5 / loud=0.5")
    ax6.axes.yaxis.set_visible(False)
    ax6.set_ylim([ioi_min, ioi_max])
    # ax6.axes.xaxis.set_visible(False)
    # ax6.set_xlabel("Note Index", fontsize=30)

    ax7 = axs[0, 2]
    ax7.plot(range(len(test_x)), y_vel, color="gray", label="GT", linewidth=3)
    ax7.plot(range(len(test_x)), interp_d[2][0], color="black", label="quiet=1 / loud=0")
    ax7.axes.xaxis.set_visible(False)
    ax7.axes.yaxis.set_visible(False)
    ax7.set_ylim([vel_min, vel_max])

    ax8 = axs[1, 2]
    ax8.plot(range(len(test_x)), y_art, color="gray", label="GT", linewidth=3)
    ax8.plot(range(len(test_x)), interp_d[2][1], color="black", label="quiet=1 / loud=0")
    ax8.axes.xaxis.set_visible(False)
    ax8.axes.yaxis.set_visible(False)
    ax8.set_ylim([art_min, art_max])

    ax9 = axs[2, 2]
    ax9.plot(range(len(test_x)), y_ioi, color="gray", label="GT", linewidth=3)
    ax9.plot(range(len(test_x)), interp_d[2][2], color="black", label="quiet=1 / loud=0")
    ax9.axes.yaxis.set_visible(False)
    ax9.set_ylim([ioi_min, ioi_max])
    # ax9.axes.xaxis.set_visible(False)
    # ax9.set_xlabel("Note Index", fontsize=30)

    fig.align_ylabels(axs[:, 0])
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("Note Index", fontsize=65)

    tick_labels = ["GT", "Loud 100%", "Loud 50% + Quiet 50%", "Quiet 100%"]
    # colors = ["gray"] + [list(colormap(normalize(n))) for n in reversed(range(1, 4))]
    colors = ["gray", "#CD5C5C", "#8B0000", "black"]

    legend_elements = [Patch(facecolor=c, edgecolor=None, label=t) \
        for c, t in zip(colors, tick_labels)]
    fig.legend(handles=legend_elements, loc="lower center", fontsize=65, bbox_to_anchor=(0.5, 0),
          fancybox=True, ncol=1)    

    # for patch in leg.get_patches():
    #     patch.set_height(22)
    #     patch.set_y(-5)

    plt.savefig("qualitative_result_{}_{}-{}_d".format(p_name, measures[0], measures[1]))
    plt.close()




    ### PLOT ###
    ## CHANGING ARTICULATION ##
    plt.rcParams['font.size'] = 30
    plt.rcParams['axes.linewidth'] = 5
    plt.rcParams['lines.linewidth'] = 7
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, 
        figsize=(15,25)) #sharey=True,
    fig.subplots_adjust(
        left=0.12, right=0.99, bottom=0.33, top=0.99, 
        wspace=0.07, hspace=0.14)

    ax1 = axs[0, 0]
    ax1.plot(range(len(test_x)), y_vel, color="gray", label="GT", linewidth=3)
    ax1.plot(range(len(test_x)), interp_a[0][0], color="#CD5C5C", label="legato=0 / stac=1")
    ax1.set_ylabel('MIDIVelocity', fontsize=65)
    ax1.axes.xaxis.set_visible(False)
    ax1.set_ylim([vel_min, vel_max])
    
    ax2 = axs[1, 0]
    ax2.plot(range(len(test_x)), y_art, color="gray", label="GT", linewidth=3)
    ax2.plot(range(len(test_x)), interp_a[0][1], color="#CD5C5C", label="legato=0 / stac=1")
    ax2.set_ylabel('Articulation', fontsize=65) 
    ax2.axes.xaxis.set_visible(False)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_ylim([art_min, art_max])
    
    ax3 = axs[2, 0]
    ax3.plot(range(len(test_x)), y_ioi, color="gray", label="GT", linewidth=3)
    ax3.plot(range(len(test_x)), interp_a[0][2], color="#CD5C5C", label="legato=0 / stac=1")
    ax3.set_ylabel('IOIRatio', fontsize=65)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.set_ylim([ioi_min, ioi_max])
    # ax3.set_xlabel("Note Index", fontsize=30)
    # ax3.axes.xaxis.set_visible(False)

    ax4 = axs[0, 1]
    ax4.plot(range(len(test_x)), y_vel, color="gray", label="GT", linewidth=3)
    ax4.plot(range(len(test_x)), interp_a[1][0], color="#8B0000", label="legato=0.5 / stac=0.5")
    ax4.axes.xaxis.set_visible(False)
    ax4.axes.yaxis.set_visible(False)
    ax4.set_ylim([vel_min, vel_max])

    ax5 = axs[1, 1]
    ax5.plot(range(len(test_x)), y_art, color="gray", label="GT", linewidth=3)
    ax5.plot(range(len(test_x)), interp_a[1][1], color="#8B0000", label="legato=0.5 / stac=0.5")
    ax5.axes.xaxis.set_visible(False)
    ax5.axes.yaxis.set_visible(False)
    ax5.set_ylim([art_min, art_max])

    ax6 = axs[2, 1]
    ax6.plot(range(len(test_x)), y_ioi, color="gray", label="GT", linewidth=3)
    ax6.plot(range(len(test_x)), interp_a[1][2], color="#8B0000", label="legato=0.5 / stac=0.5")
    ax6.axes.yaxis.set_visible(False)
    ax6.set_ylim([ioi_min, ioi_max])
    # ax6.axes.xaxis.set_visible(False)
    # ax6.set_xlabel("Note Index", fontsize=30)

    ax7 = axs[0, 2]
    ax7.plot(range(len(test_x)), y_vel, color="gray", label="GT", linewidth=3)
    ax7.plot(range(len(test_x)), interp_a[2][0], color="black", label="legato=1 / stac=0")
    ax7.axes.xaxis.set_visible(False)
    ax7.axes.yaxis.set_visible(False)
    ax7.set_ylim([vel_min, vel_max])

    ax8 = axs[1, 2]
    ax8.plot(range(len(test_x)), y_art, color="gray", label="GT", linewidth=3)
    ax8.plot(range(len(test_x)), interp_a[2][1], color="black", label="legato=1 / stac=0")
    ax8.axes.xaxis.set_visible(False)
    ax8.axes.yaxis.set_visible(False)
    ax8.set_ylim([art_min, art_max])

    ax9 = axs[2, 2]
    ax9.plot(range(len(test_x)), y_ioi, color="gray", label="GT", linewidth=3)
    ax9.plot(range(len(test_x)), interp_a[2][2], color="black", label="legato=1 / stac=0")
    ax9.axes.yaxis.set_visible(False)
    ax9.set_ylim([ioi_min, ioi_max])
    # ax9.axes.xaxis.set_visible(False)
    # ax9.set_xlabel("Note Index", fontsize=30)

    fig.align_ylabels(axs[:, 0])
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("Note Index", fontsize=65)

    tick_labels = ["GT", "Stac. 100%", "Stac. 50% + Legato 50%", "Legato 100%"]
    # colors = ["gray"] + [list(colormap(normalize(n))) for n in reversed(range(1, 4))]
    colors = ["gray", "#CD5C5C", "#8B0000", "black"]

    legend_elements = [Patch(facecolor=c, edgecolor=None, label=t) \
        for c, t in zip(colors, tick_labels)]
    fig.legend(handles=legend_elements, loc="lower center", fontsize=65, bbox_to_anchor=(0.5, 0),
          fancybox=True, ncol=1)  
    
    plt.savefig("qualitative_result_{}_{}-{}_a".format(p_name, measures[0], measures[1]))
    plt.close()



    ### PLOT ###
    ## CHANGING TEMPO ##
    plt.rcParams['font.size'] = 30
    plt.rcParams['axes.linewidth'] = 5
    plt.rcParams['lines.linewidth'] = 7
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, 
        figsize=(15,25)) #sharey=True,
    fig.subplots_adjust(
        left=0.12, right=0.99, bottom=0.33, top=0.99, 
        wspace=0.07, hspace=0.14)

    ax1 = axs[0, 0]
    ax1.plot(range(len(test_x)), y_vel, color="gray", label="GT", linewidth=3)
    ax1.plot(range(len(test_x)), interp_t[0][0], color="#CD5C5C", label="slow=0 / fast=1")
    ax1.set_ylabel('MIDIVelocity', fontsize=65)
    ax1.axes.xaxis.set_visible(False)
    ax1.set_ylim([vel_min, vel_max])
    
    ax2 = axs[1, 0]
    ax2.plot(range(len(test_x)), y_art, color="gray", label="GT", linewidth=3)
    ax2.plot(range(len(test_x)), interp_t[0][1], color="#CD5C5C", label="slow=0 / fast=1")
    ax2.set_ylabel('Articulation', fontsize=65) 
    ax2.axes.xaxis.set_visible(False)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_ylim([art_min, art_max])
    
    ax3 = axs[2, 0]
    ax3.plot(range(len(test_x)), y_ioi, color="gray", label="GT", linewidth=3)
    ax3.plot(range(len(test_x)), interp_t[0][2], color="#CD5C5C", label="slow=0 / fast=1")
    ax3.set_ylabel('IOIRatio', fontsize=65)
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax3.set_ylim([ioi_min, ioi_max])
    # ax3.set_xlabel("Note Index", fontsize=65)
    # ax3.axes.xaxis.set_visible(False)

    ax4 = axs[0, 1]
    ax4.plot(range(len(test_x)), y_vel, color="gray", label="GT", linewidth=3)
    ax4.plot(range(len(test_x)), interp_t[1][0], color="#8B0000", label="slow=0.5 / fast=0.5")
    ax4.axes.xaxis.set_visible(False)
    ax4.axes.yaxis.set_visible(False)
    ax4.set_ylim([vel_min, vel_max])

    ax5 = axs[1, 1]
    ax5.plot(range(len(test_x)), y_art, color="gray", label="GT", linewidth=3)
    ax5.plot(range(len(test_x)), interp_t[1][1], color="#8B0000", label="slow=0.5 / fast=0.5")
    ax5.axes.xaxis.set_visible(False)
    ax5.axes.yaxis.set_visible(False)
    ax5.set_ylim([art_min, art_max])

    ax6 = axs[2, 1]
    ax6.plot(range(len(test_x)), y_ioi, color="gray", label="GT", linewidth=3)
    ax6.plot(range(len(test_x)), interp_t[1][2], color="#8B0000", label="slow=0.5 / fast=0.5")
    ax6.axes.yaxis.set_visible(False)
    ax6.set_ylim([ioi_min, ioi_max])
    # ax6.axes.xaxis.set_visible(False)
    # ax6.set_xlabel("Note Index", fontsize=30)

    ax7 = axs[0, 2]
    ax7.plot(range(len(test_x)), y_vel, color="gray", label="GT", linewidth=3)
    ax7.plot(range(len(test_x)), interp_t[2][0], color="black", label="slow=1 / fast=0")
    ax7.axes.xaxis.set_visible(False)
    ax7.axes.yaxis.set_visible(False)
    ax7.set_ylim([vel_min, vel_max])

    ax8 = axs[1, 2]
    ax8.plot(range(len(test_x)), y_art, color="gray", label="GT", linewidth=3)
    ax8.plot(range(len(test_x)), interp_t[2][1], color="black", label="slow=1 / fast=0")
    ax8.axes.xaxis.set_visible(False)
    ax8.axes.yaxis.set_visible(False)
    ax8.set_ylim([art_min, art_max])

    ax9 = axs[2, 2]
    ax9.plot(range(len(test_x)), y_ioi, color="gray", label="GT", linewidth=3)
    ax9.plot(range(len(test_x)), interp_t[2][2], color="black", label="slow=1 / fast=0")
    ax9.axes.yaxis.set_visible(False)
    ax9.set_ylim([ioi_min, ioi_max])
    # ax9.axes.xaxis.set_visible(False)
    # ax9.set_xlabel("Note Index", fontsize=30)

    fig.align_ylabels(axs[:, 0])
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("Note Index", fontsize=65)

    tick_labels = ["GT", "Fast 100%", "Fast 50% + Slow 50%", "Slow 100%"]
    # colors = ["gray"] + [list(colormap(normalize(n))) for n in reversed(range(1, 4))]
    colors = ["gray", "#CD5C5C", "#8B0000", "black"]

    legend_elements = [Patch(facecolor=c, edgecolor=None, label=t) \
        for c, t in zip(colors, tick_labels)]
    fig.legend(handles=legend_elements, loc="lower center", fontsize=65, bbox_to_anchor=(0.5, 0),
          fancybox=True, ncol=1)  

    plt.savefig("qualitative_result_{}_{}-{}_t".format(p_name, measures[0], measures[1]))
    plt.close()



    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_t[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind, savename="slow_0_fast_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_t[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="slow_100_fast_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_d[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="quiet_0_loud_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_d[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="quiet_100_loud_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_a[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="stac_0_leg_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_a[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="stac_100_leg_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample1_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[1][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample2_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[2][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample3_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=y_sampled0, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="neutral_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader_i, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader_i_{}_mm{}-{}.mid".format(song_name, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader_d, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader_d_{}_mm{}-{}.mid".format(song_name, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader_a, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader_a_{}_mm{}-{}.mid".format(song_name, measures[0], measures[1]), save_score=False)
    

def qualitative_results_fader(
    song_name=None, measures=None, device_num=None,
    model_num=None, exp_num=None, epoch_num=None, same_onset_ind=[110,112]):

    pair_path = None

    # test data paths
    # song_name = "Schubert_Impromptu_op90_D899__4"
    # song_name = "Mozart_Piano_Sonatas__11-3"
    # song_name = "Beethoven_Piano_Sonatas__8-2"
    # song_name = "Beethoven_Piano_Sonatas__14-3"
    # song_name = "Liszt_Gran_Etudes_de_Paganini__2_La_campanella"
    # measures = [5, 10]
    # same_onset_ind = [110,112]
    # measures = [1, 16]
    # measures = [217,232]
    # measures = [149,153]
    song_name_ =  '/'.join(song_name.split('__'))
    parent_path = "/data/asap_dataset/exp_data/listening_test/raw"
    perform = [p for p in sorted(glob(os.path.join(parent_path, "{}/01/*.mid".format(
        song_name_)))) if "cleaned" not in p][0]
    xml = os.path.join(parent_path, "{}/musicxml_cleaned_plain.musicxml".format(song_name_))
    score = os.path.join(parent_path, "{}/score_plain.mid".format(song_name_))
    pair = os.path.join(os.path.dirname(perform), 'xml_score_perform_pairs.npy')
    if os.path.exists(pair) is True:
        pair_path = pair 
    else:
        pair_path = None
    # pair_path = None

    ## LOAD DATA ##
    # model_num = 'exp1937'
    # exp_num = 'exp1937_p4'
    # epoch_num = '100'
    # device_num = 2
    print()
    print()
    print("############# TEST ##############")
    print("> MODEL: {}\n> EXP: {}".format(model_num, exp_num))
    print("#################################")
    print()
    print("** Load Data **")

    null_tempo = 120
    get_data = GetData(null_tempo=null_tempo, 
                       same_onset_ind=same_onset_ind,
                       stat=np.mean)
    test_x, test_y, test_m, pairs, xml_notes, perform_notes, tempo, p_name = get_data.file2data(
        files=[xml, score, perform], measures=measures, mode="note", pair_path=pair_path)
    tempo_rate = tempo / null_tempo
    print("     > tempo: {}".format(tempo))
    print()


    ## LOAD MODEL ## 
    print("** Load model **")
    module_name = "piano_cvae_model2_torch_{}".format(model_num)
    model = importlib.import_module(module_name)
    Generator = model.PerformGenerator
    note2group = model.Note2Group()

    model_path = "./model_cvae/piano_cvae_ckpt_{}".format(exp_num) # 
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")
    model = Generator(device=device)
    
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    loss_val = checkpoint["loss_val"]
    checkpoint_num = len(loss_val)
    model.eval()

    ## INFER LATENTS BY CONDITIONS ##
    indices = ["fast", "slow", "loud", "quiet", "stac", "legato", "neutral"]
    c_dict = dict()
    interp = 'tanh'
    n = 0

    y_vel = test_y[:,0]
    y_ioi = test_y[:,2]
    y_art = np.power(10, np.log10(test_y[:,1]) - np.log10(test_y[:,3]))

    test_inputs_ = get_data.data2input(test_x, test_y, test_m, 
        cond=None, art=True, mode="note", device=device)
    test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs_
    test_inputs = test_inputs_

    _, _, _, \
    _, _, z_moments, \
    c, z, _, _, \
    _, _, _ = model(*test_inputs)

    # c_mean = torch.mean(c, dim=1).unsqueeze(1).repeat(1, T, 1)
    c_rand = torch.randn_like(c)
    c_mean = torch.empty_like(c_rand).copy_(c_rand)
    c_mean[:,:,0] = torch.zeros_like(c_rand[:,:,0])
    c_mean[:,:,4] = torch.zeros_like(c_rand[:,:,4])
    c_mean[:,:,8] = torch.zeros_like(c_rand[:,:,8])

    _, _, _, inferred = model.sample(
        test_x_, test_m_, c_=c, z_=z)  
    _, _, _, sampled = model.sample(
        test_x_, test_m_, c_=c, trunc=True, threshold=2.)  
    _, _, _, sampled0 = model.sample(
        test_x_, test_m_, c_=c_mean) 
    _, _, _, sampled02 = model.sample(
        test_x_, test_m_, c_=c_mean) 
    inferred_group = note2group(inferred, test_m_)
    sampled_group = note2group(sampled, test_m_)
    sampled0_group = note2group(sampled0, test_m_)
    sampled0_group2 = note2group(sampled02, test_m_)
    
    gt = inverse_feature_note(test_y2_,  art=True, numpy=False, interp=interp)
    infer = inverse_feature_note(inferred_group,  art=True, numpy=False, interp=interp)
    sample = inverse_feature_note(sampled_group,  art=True, numpy=False, interp=interp)
    flat = inverse_feature_note(sampled0_group,  art=True, numpy=False, interp=interp)
    flat2 = inverse_feature_note(sampled0_group2,  art=True, numpy=False, interp=interp)
    infer_raw = inverse_feature_note(inferred,  art=True, numpy=False, interp=interp)
    sample_raw = inverse_feature_note(sampled,  art=True, numpy=False, interp=interp)
    flat_raw = inverse_feature_note(sampled0,  art=True, numpy=False, interp=interp)
    flat_raw2 = inverse_feature_note(sampled02,  art=True, numpy=False, interp=interp)


    T = c.size(1)
    h1, h2 = T//2, T-(T//2)
    t1, t2, t3 = T//3, T//3, T-(T//3)*2
    random1 = np.concatenate([np.random.randn(h1,)-1, np.random.randn(h2,)+1])
    random2 = np.concatenate([np.random.randn(h1,)+1, np.random.randn(h2,)-1])
    # random3 = np.concatenate([np.random.randn(t1,)-1, np.random.randn(t2,)+1, np.random.randn(t3,)-1])
    # random4 = np.concatenate([np.random.randn(t1,)+1, np.random.randn(t2,)-1, np.random.randn(t3,)+1])  
    sketch1 = poly_predict(random1, N=4)
    sketch2 = poly_predict(random2, N=4)
    # sketch3 = poly_predict(random3, N=4)
    # sketch4 = poly_predict(random4, N=4)
    flat_alpha = c_mean[0][:,0].cpu().data.numpy()
    sketch0_v = c[0][:,0].cpu().data.numpy()
    sketch0_a = c[0][:,4].cpu().data.numpy()
    sketch0_i = c[0][:,8].cpu().data.numpy()

    s_min = np.min([np.min(s) for s in [sketch1, sketch2]])
    s_max = np.max([np.max(s) for s in [sketch1, sketch2]])
    s_min = s_min - 0.1
    s_max = s_max + 0.1
    
    # assert len(add_seq_) == t
    sketch1 = torch.from_numpy(sketch1).to(device).float()
    sketch2 = torch.from_numpy(sketch2).to(device).float()
    # sketch3 = torch.from_numpy(sketch3).to(device).float()
    # sketch4 = torch.from_numpy(sketch4).to(device).float()


    # fader 
    s1 = torch.empty_like(c_mean).copy_(c_mean)
    s1[:,:,0] = sketch1.unsqueeze(0)
    s1[:,:,4] = sketch1.unsqueeze(0)
    s1[:,:,8] = sketch1.unsqueeze(0)
    _, _, _, sampled1 = model.sample(test_x_, test_m_, c_=s1)
    _, _, _, sampled12 = model.sample(test_x_, test_m_, c_=s1) 
    sampled1_group = note2group(sampled1, test_m_)
    sampled1_group2 = note2group(sampled12, test_m_)
    fader1 = inverse_feature_note(sampled1_group,  art=True, numpy=False, interp=interp)
    fader12 = inverse_feature_note(sampled1_group2,  art=True, numpy=False, interp=interp)
    fader1_raw = inverse_feature_note(sampled1,  art=True, numpy=False, interp=interp)
    fader1_raw2 = inverse_feature_note(sampled12,  art=True, numpy=False, interp=interp)

    s2 = torch.empty_like(c_mean).copy_(c_mean)
    s2[:,:,0] = sketch2.unsqueeze(0)
    s2[:,:,4] = sketch2.unsqueeze(0)
    s2[:,:,8] = sketch2.unsqueeze(0)
    _, _, _, sampled2 = model.sample(test_x_, test_m_, c_=s2) 
    _, _, _, sampled22 = model.sample(test_x_, test_m_, c_=s2) 
    sampled2_group = note2group(sampled2, test_m_)
    sampled2_group2 = note2group(sampled22, test_m_)
    fader2 = inverse_feature_note(sampled2_group,  art=True, numpy=False, interp=interp)
    fader22 = inverse_feature_note(sampled2_group2,  art=True, numpy=False, interp=interp)
    fader2_raw = inverse_feature_note(sampled2,  art=True, numpy=False, interp=interp)
    fader2_raw2 = inverse_feature_note(sampled22,  art=True, numpy=False, interp=interp)

    # s3 = torch.empty_like(c_mean).copy_(c_mean)
    # s3[:,:,0] = sketch3.unsqueeze(0)
    # s3[:,:,4] = sketch3.unsqueeze(0)
    # s3[:,:,8] = sketch3.unsqueeze(0)
    # _, _, _, sampled3 = model.sample(
    #     test_x_, test_m_, c_=s3, trunc=True, threshold=2.)
    # sampled3_group = note2group(sampled3, test_m_) 
    # fader3 = inverse_feature_note(sampled3_group,  art=True, numpy=False, interp=interp)
    # fader3_raw = inverse_feature_note(sampled3,  art=True, numpy=False, interp=interp)

    # s4 = torch.empty_like(c_mean).copy_(c_mean)
    # s4[:,:,0] = sketch4.unsqueeze(0)
    # s4[:,:,4] = sketch4.unsqueeze(0)
    # s4[:,:,8] = sketch4.unsqueeze(0)
    # _, _, _, sampled4 = model.sample(
    #     test_x_, test_m_, c_=s4, trunc=True, threshold=2.) 
    # sampled4_group = note2group(sampled4, test_m_)
    # fader4 = inverse_feature_note(sampled4_group,  art=True, numpy=False, interp=interp)
    # fader4_raw = inverse_feature_note(sampled4,  art=True, numpy=False, interp=interp)

    vel_min = np.min([np.min(f[0]) for f in [gt, infer, sample, flat, flat2, fader1, fader12, fader2, fader22]])
    art_min = np.min([np.min(f[1]) for f in [gt, infer, sample, flat, flat2, fader1, fader12, fader2, fader22]])
    ioi_min = np.min([np.min(f[2]) for f in [gt, infer, sample, flat, flat2, fader1, fader12, fader2, fader22]])
    vel_max = np.max([np.max(f[0]) for f in [gt, infer, sample, flat, flat2, fader1, fader12, fader2, fader22]])
    art_max = np.max([np.max(f[1]) for f in [gt, infer, sample, flat, flat2, fader1, fader12, fader2, fader22]])
    ioi_max = np.max([np.max(f[2]) for f in [gt, infer, sample, flat, flat2, fader1, fader12, fader2, fader22]])

    vel_min = vel_min - 1
    art_min = art_min - 0.1
    ioi_min = ioi_min - 0.1
    vel_max = vel_max + 1
    art_max = art_max + 0.1
    ioi_max = ioi_max + 0.1


    ### PLOT ###
    ## CHANGING DYNAMICS ##
    plt.rcParams['font.size'] = 25
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['lines.linewidth'] = 5
    fig, axs = plt.subplots(nrows=4, ncols=4, sharex=True, 
        figsize=(18,15), 
        gridspec_kw={'height_ratios': [0.5, 1, 1, 1]}) #sharey=True, gridspec_kw={'height_ratios': [0.5, 1, 1, 1]}
    fig.subplots_adjust(
        left=0.07, right=0.995, bottom=0.20, top=0.99, 
        wspace=0.07, hspace=0.1)
    colormap = cm.Blues
    normalize = mcolors.Normalize(vmin=1, vmax=5)
    colors = ["gray", "#2E8B57", "#69aadb", "black", "#0b5394", "#E87200"] #colormap(normalize(4))
    tick_labels = ["GT", "Sketch", "Reconstructed", "Controlled 1", "Sampled", "Controlled 2"]


    # ax000 = axs[0,0]
    # ax000.plot(range(T), sketch0_v, color="#8B0000", linewidth=4)
    # ax000.plot(range(T), sketch0_a, color="#8B0000", linewidth=4)
    # ax000.plot(range(T), sketch0_v, color="#8B0000", linewidth=4)
    # ax000.set_ylabel(r'$\alpha$', fontsize=35)
    # ax000.axes.xaxis.set_visible(False)
    # ax000.set_ylim([s_min, s_max])

    # make xaxis invisibel
    axs[0,0].xaxis.set_visible(False)
    # make spines (the box) invisible
    plt.setp(axs[0,0].spines.values(), visible=False)
    # remove ticks and labels for the left axis
    axs[0,0].tick_params(left=False, labelleft=False)

    ax001 = axs[1,0]
    ax001.plot(range(T), gt[0], color="gray", linewidth=2)
    ax001.plot(range(T), infer[0], color="#69aadb")
    ax001.plot(range(T), sample[0], color="#0b5394")
    ax001.set_ylabel('MIDIVelocity', fontsize=35)
    ax001.axes.xaxis.set_visible(False)
    ax001.set_ylim([vel_min, vel_max])
    
    ax002 = axs[2,0]
    ax002.plot(range(T), gt[1], color="gray", linewidth=2)
    ax002.plot(range(T), infer[1], color="#69aadb")
    ax002.plot(range(T), sample[1], color="#0b5394")
    ax002.set_ylabel('Articulation', fontsize=35) 
    ax002.axes.xaxis.set_visible(False)
    ax002.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax002.set_ylim([art_min, art_max])
    
    ax003 = axs[3,0]
    ax003.plot(range(T), gt[2], color="gray", linewidth=2)
    ax003.plot(range(T), infer[2], color="#69aadb")
    ax003.plot(range(T), sample[2], color="#0b5394")
    ax003.set_ylabel('IOIRatio', fontsize=35)
    ax003.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax003.set_ylim([ioi_min, ioi_max])


    ax00 = axs[0,1]
    ax00.plot(range(T), flat_alpha, color="#2E8B57", linewidth=7)
    ax00.set_ylabel(r'$\alpha$', fontsize=35)
    ax00.axes.xaxis.set_visible(False)
    ax00.set_ylim([s_min, s_max])

    ax01 = axs[1,1]
    ax01.plot(range(T), gt[0], color="gray", linewidth=2)
    ax01.plot(range(T), flat[0], color="#E87200")
    ax01.plot(range(T), flat2[0], color="black")
    # ax01.set_ylabel('MIDIVelocity', fontsize=35)
    ax01.axes.xaxis.set_visible(False)
    ax01.axes.yaxis.set_visible(False)
    ax01.set_ylim([vel_min, vel_max])
    
    ax02 = axs[2,1]
    ax02.plot(range(T), gt[1], color="gray", linewidth=2)
    ax02.plot(range(T), flat[1], color="#E87200")
    ax02.plot(range(T), flat2[1], color="black")
    # ax02.set_ylabel('Articulation', fontsize=35) 
    ax02.axes.xaxis.set_visible(False)
    ax02.axes.yaxis.set_visible(False)
    ax02.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax02.set_ylim([art_min, art_max])
    
    ax03 = axs[3,1]
    ax03.plot(range(T), gt[2], color="gray", linewidth=2)
    ax03.plot(range(T), flat[2], color="#E87200")
    ax03.plot(range(T), flat2[2], color="black")
    # ax03.set_ylabel('IOIRatio', fontsize=35)
    ax03.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax03.set_ylim([ioi_min, ioi_max])
    # ax3.set_xlabel("Note Index", fontsize=35)
    # ax03.axes.xaxis.set_visible(False)
    ax03.axes.yaxis.set_visible(False)

    # second column
    ax10 = axs[0,2]
    ax10.plot(range(T), sketch1.cpu().data.numpy(), color="#2E8B57", linewidth=7)
    # ax10.set_ylabel(r'$\alpha$', fontsize=35)
    ax10.axes.yaxis.set_visible(False)
    ax10.axes.xaxis.set_visible(False)
    ax10.set_ylim([s_min, s_max])

    ax11 = axs[1,2]
    ax11.plot(range(T), gt[0], color="gray", linewidth=2)
    ax11.plot(range(T), fader1[0], color="#E87200")
    ax11.plot(range(T), fader12[0], color="black")
    # ax11.set_ylabel('MIDI Velocity', fontsize=30)
    ax11.axes.xaxis.set_visible(False)
    ax11.axes.yaxis.set_visible(False)
    ax11.set_ylim([vel_min, vel_max])

    ax12 = axs[2,2]
    ax12.plot(range(T), gt[1], color="gray", linewidth=2)
    ax12.plot(range(T), fader1[1], color="#E87200")
    ax12.plot(range(T), fader12[1], color="black")
    # ax12.set_ylabel('Articulation', fontsize=30) 
    ax12.axes.xaxis.set_visible(False)
    ax12.axes.yaxis.set_visible(False)
    ax12.set_ylim([art_min, art_max])
    # ax12.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    ax13 = axs[3,2]
    ax13.plot(range(T), gt[2], color="gray", linewidth=2)
    ax13.plot(range(T), fader1[2], color="#E87200")
    ax13.plot(range(T), fader12[2], color="black")
    ax13.set_ylabel('IOI', fontsize=30)
    # ax13.axes.xaxis.set_visible(False)
    ax13.axes.yaxis.set_visible(False)
    ax13.set_ylim([ioi_min, ioi_max])
    # ax13.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax13.set_xlabel("Note Index", fontsize=30)
    

    # third column
    ax20 = axs[0,3]
    ax20.plot(range(T), sketch2.cpu().data.numpy(), color="#2E8B57", linewidth=7)
    # ax20.set_ylabel(r'$\alpha$', fontsize=30)
    ax20.axes.xaxis.set_visible(False)
    ax20.axes.yaxis.set_visible(False)
    ax20.set_ylim([s_min, s_max])

    ax21 = axs[1,3]
    ax21.plot(range(T), gt[0], color="gray", linewidth=2)
    ax21.plot(range(T), fader2[0], color="#E87200")
    ax21.plot(range(T), fader22[0], color="black")
    # ax21.set_ylabel('MIDI Velocity', fontsize=30)
    ax21.axes.xaxis.set_visible(False)
    ax21.axes.yaxis.set_visible(False)
    ax21.set_ylim([vel_min, vel_max])

    ax22 = axs[2,3]
    ax22.plot(range(T), gt[1], color="gray", linewidth=2)
    ax22.plot(range(T), fader2[1], color="#E87200")
    ax22.plot(range(T), fader22[1], color="black")
    # ax22.set_ylabel('Articulation', fontsize=30) 
    ax22.axes.xaxis.set_visible(False)
    ax22.axes.yaxis.set_visible(False)
    ax22.set_ylim([art_min, art_max])
    # ax22.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    ax23 = axs[3,3]
    ax23.plot(range(T), gt[2], color="gray", linewidth=2)
    ax23.plot(range(T), fader2[2], color="#E87200")
    ax23.plot(range(T), fader22[2], color="black")
    # ax23.set_ylabel('IOI', fontsize=30)
    # ax23.axes.xaxis.set_visible(False)
    ax23.axes.yaxis.set_visible(False)
    ax23.set_ylim([ioi_min, ioi_max])
    # ax23.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax23.set_xlabel("Note Index", fontsize=30)
    
    fig.align_ylabels(axs[:])
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("Onset Index (Chordwise)", fontsize=35)

    # for patch in leg.get_patches():
    #     patch.set_height(22)
    #     patch.set_y(-5)

    legend_elements = [Patch(facecolor=c, edgecolor=None, label=t) \
        for c, t in zip(colors, tick_labels)]
    fig.legend(handles=legend_elements, loc="lower center", fontsize=28, bbox_to_anchor=(0.5, 0),
          fancybox=True, ncol=3)  

    plt.savefig("demo_qualitative_result_fader_{}_{}-{}_new".format(p_name, measures[0], measures[1]))
    plt.close()


    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader1_raw, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader1_{}_mm{}-{}_new.mid".format(song_name, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader2_raw, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader2_{}_mm{}-{}_new.mid".format(song_name, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader3_raw, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader3_{}_mm{}-{}.mid".format(song_name, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader4_raw, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader4_{}_mm{}-{}.mid".format(song_name, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=flat_raw, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader0_{}_mm{}-{}_new.mid".format(song_name, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader1_raw2, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader12_{}_mm{}-{}_new.mid".format(song_name, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader2_raw2, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader22_{}_mm{}-{}_new.mid".format(song_name, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader3_raw, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader3_{}_mm{}-{}.mid".format(song_name, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader4_raw, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader4_{}_mm{}-{}.mid".format(song_name, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=flat_raw2, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader02_{}_mm{}-{}_new.mid".format(song_name, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=sample_raw, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_sample_{}_mm{}-{}_new.mid".format(song_name, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=infer_raw, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_infer_{}_mm{}-{}_new.mid".format(song_name, measures[0], measures[1]), save_score=False)
    

def qualitative_results_EP(
    song_name=None, measures=None, device_num=None,
    model_num=None, exp_num=None, epoch_num=None, same_onset_ind=[110,112]):

    pair_path = None

    # test data paths
    # song_name = "Schubert_Impromptu_op90_D899__4"
    # song_name = "Mozart_Piano_Sonatas__11-3"
    # song_name = "Beethoven_Piano_Sonatas__8-2"
    # song_name = "Beethoven_Piano_Sonatas__14-3"
    # same_onset_ind = [110,112]
    # measures = [1, 16]
    # measures = [217,232]
    # measures = [149,153]
    song_name_ =  '/'.join(song_name.split('__'))
    parent_path = "/data/asap_dataset/exp_data/listening_test/raw"
    perform = [p for p in sorted(glob(os.path.join(parent_path, "{}/01/*.mid".format(
        song_name_)))) if "cleaned" not in p][0]
    xml = os.path.join(parent_path, "{}/musicxml_cleaned_plain.musicxml".format(song_name_))
    score = os.path.join(parent_path, "{}/score_plain.mid".format(song_name_))
    pair = os.path.join(os.path.dirname(perform), 'xml_score_perform_pairs.npy')
    if os.path.exists(pair) is True:
        pair_path = pair 
    else:
        pair_path = None
    # pair_path = None

    ## LOAD DATA ##
    # model_num = 'exp1937'
    # exp_num = 'exp1937_p4'
    # epoch_num = '100'
    # device_num = 2
    print()
    print()
    print("############# TEST ##############")
    print("> MODEL: {}\n> EXP: {}".format(model_num, exp_num))
    print("#################################")
    print()
    print("** Load Data **")

    null_tempo = 120
    get_data = GetData(null_tempo=null_tempo, 
                       same_onset_ind=same_onset_ind,
                       stat=np.mean)
    test_x, test_y, test_m, pairs, xml_notes, perform_notes, tempo, p_name = get_data.file2data(
        files=[xml, score, perform], measures=measures, mode="note", pair_path=pair_path)
    tempo_rate = tempo / null_tempo
    print("     > tempo: {}".format(tempo))
    print()


    ## LOAD MODEL ## 
    print("** Load model **")
    module_name = "piano_cvae_model2_torch_{}".format(model_num)
    model = importlib.import_module(module_name)
    Generator = model.PerformGenerator
    note2group = model.Note2Group()
    Mask = model.Mask

    model_path = "./model_cvae/piano_cvae_ckpt_{}".format(exp_num) # 
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")
    model = Generator(device=device)
    
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    loss_val = checkpoint["loss_val"]
    checkpoint_num = len(loss_val)
    model.eval()

    ## INFER LATENTS BY CONDITIONS ##
    indices = ["fast", "slow", "loud", "quiet", "stac", "legato", "neutral"]
    c_dict = dict()
    interp = 'tanh'
    n = 0

    if "_p" in exp_num:
        N = int(exp_num.split("_")[-1][1]) 
    elif "exp1937" == exp_num:
        N = 2 
    else:
        N = 4

    y_vel = test_y[:,0]
    y_ioi = test_y[:,2]
    y_art = np.power(10, np.log10(test_y[:,1]) - np.log10(test_y[:,3]))

    test_inputs_ = get_data.data2input(test_x, test_y, test_m, 
        cond=None, art=True, mode="note", device=device)
    test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs_
    test_inputs = test_inputs_
    mask = Mask(m=test_m_)

    # sample c
    if "cvrnn" in model_num or "fader" in model_num:
        c = note2group.reverse(test_clab_, test_m_)
    else:
        _, _, _, \
        _, _, z_moments, \
        c, z, _, _, \
        est_c, _, _ = model(*test_inputs)

    y_c_vel, y_c_art, y_c_ioi = \
        inverse_feature_note(test_clab_, art=True, numpy=False, interp=interp)


    # est_c_ = note2group.reverse(est_c, test_m_) 
    # y_c_vel, y_c_art, y_c_ioi = \
    #     inverse_feature_note(test_clab_, art=True, numpy=False, interp=interp)

    orig = inverse_feature_note(test_y2_, art=True, numpy=False, interp=interp)

    # re-infer c
    _, _, _, sampled = model.sample(
        test_x_, test_m_, c_=c, trunc=True, threshold=2) 
    sampled_c = model.sample_c_only(sampled, test_m_)
    s_c1 = model.predict_c1(sampled_c[:,:,:4], mask)
    s_c2 = model.predict_c2(sampled_c[:,:,4:8], mask)
    s_c3 = model.predict_c3(sampled_c[:,:,8:12], mask)
    est_c = torch.cat([s_c1, s_c2, s_c3], dim=-1)

    # s_c_vel, s_c_art, s_c_ioi = \
        # inverse_feature_note(est_c2, art=True, numpy=False, interp=interp)
    # sampled_group = note2group(sampled, test_m_)
    # new_ep = poly_predict(sampled_group[0].cpu().data.numpy(), N=N)
    # new_ep = np.expand_dims(new_ep, 0)
    features = \
        inverse_feature_note(sampled.cpu().data.numpy(), art=True, numpy=True, interp=interp)
    s_c_vel, s_c_art, s_c_ioi = \
        inverse_feature_note(est_c.cpu().data.numpy(), art=True, numpy=True, interp=interp)

    ### PLOT ###
    ## CHANGING DYNAMICS ##
    plt.rcParams['font.size'] = 30
    plt.rcParams['axes.linewidth'] = 4
    plt.rcParams['lines.linewidth'] = 9
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, 
        figsize=(18,18)) #sharey=True, gridspec_kw={'height_ratios': [0.5, 1, 1, 1]}
    fig.subplots_adjust(
        left=0.09, right=0.995, bottom=0.18, top=0.99, 
        wspace=0.07, hspace=0.12)
    colormap = cm.Blues
    normalize = mcolors.Normalize(vmin=1, vmax=5)
    colors = ["gray", "#a70357"] #colormap(normalize(4))
    tick_labels = ["Raw Value", "Estimated EP"]

    # second column
    ax00 = axs[0]
    ax00.plot(range(len(orig[0])), orig[0], color="gray", linewidth=5)
    ax00.plot(range(len(orig[0])), s_c_vel, color="#a70357")
    # ax00.plot(range(len(orig[0])), s_c_vel, color="orange")
    ax00.set_ylabel('MIDIVelocity', fontsize=50)
    ax00.axes.xaxis.set_visible(False)

    ax01 = axs[1]
    ax01.plot(range(len(orig[0])), orig[1], color="gray", linewidth=5)
    ax01.plot(range(len(orig[0])), s_c_art, color="#a70357")
    # ax01.plot(range(len(orig[0])), s_c_art, color="orange")
    ax01.set_ylabel('Articulation', fontsize=50)
    ax01.axes.xaxis.set_visible(False)
    
    ax02 = axs[2]
    ax02.plot(range(len(orig[0])), orig[2], color="gray", linewidth=5)
    ax02.plot(range(len(orig[0])), s_c_ioi, color="#a70357")
    # ax02.plot(range(len(orig[0])), s_c_ioi, color="orange")
    # ax02.set_ylim([0,2])
    ax02.set_ylabel('IOIRatio', fontsize=50) 
    # ax02.axes.xaxis.set_visible(False)
    ax02.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    fig.align_ylabels(axs[:])
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel("Onset Index (Chordwise)", fontsize=50)

    # for patch in leg.get_patches():
    #     patch.set_height(22)
    #     patch.set_y(-5)

    legend_elements = [Patch(facecolor=c, edgecolor=None, label=t) \
        for c, t in zip(colors, tick_labels)]
    fig.legend(handles=legend_elements, loc="lower center", fontsize=50, bbox_to_anchor=(0.5, 0),
          fancybox=True, ncol=2)  

    plt.savefig("qualitative_result_EP_{}_{}-{}_{}".format(
        p_name, measures[0], measures[1], exp_num))
    plt.close()




    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=features, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind, savename="demo_qualitative_result_EP_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_t[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="slow_100_fast_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_d[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="quiet_0_loud_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_d[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="quiet_100_loud_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_a[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="stac_0_leg_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_a[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="stac_100_leg_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample1_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[1][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample2_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[2][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample3_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=y_sampled0, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="neutral_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader_i, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader_i_{}_mm{}-{}.mid".format(song_name, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader_d, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader_d_{}_mm{}-{}.mid".format(song_name, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader_a, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="qual_fader_a_{}_mm{}-{}.mid".format(song_name, measures[0], measures[1]), save_score=False)
    


def test_model(
    song_name=None, measures=None, device_num=None,
    model_num=None, exp_num=None, epoch_num=None, same_onset_ind=[110,112]):

    pair_path = None

    # test data paths
    # song_name = "Schubert_Impromptu_op.90_D.899__4"
    # song_name = "Mozart_Piano_Sonatas__11-3"
    # song_name = "Beethoven_Piano_Sonatas__8-2"
    # song_name = "Beethoven_Piano_Sonatas__14-3"
    # same_onset_ind = [110,112]
    # measures = [1, 16]
    # measures = [217,232]
    # measures = [149,156]
    song_name_ =  '/'.join(song_name.split('__'))
    parent_path = "/data/asap_dataset/exp_data/listening_test/raw"
    perform = [p for p in sorted(glob(os.path.join(parent_path, "{}/01/*.mid".format(
        song_name_)))) if "cleaned" not in p][0]
    xml = os.path.join(parent_path, "{}/musicxml_cleaned_plain.musicxml".format(song_name_))
    score = os.path.join(parent_path, "{}/score_plain.mid".format(song_name_))
    pair = os.path.join(os.path.dirname(perform), 'xml_score_perform_pairs.npy')
    if os.path.exists(pair) is True:
        pair_path = pair 
    else:
        pair_path = None
    # pair_path = None

    ## LOAD DATA ##
    # model_num = 'exp1941_fader'
    # exp_num = 'exp1941_fader'
    # epoch_num = '100'
    print()
    print()
    print("############# TEST ##############")
    print("> MODEL: {}\n> EXP: {}".format(model_num, exp_num))
    print("#################################")
    print()
    print("** Load Data **")

    null_tempo = 120
    get_data = GetData(null_tempo=null_tempo, 
                       same_onset_ind=same_onset_ind,
                       stat=np.mean)
    test_x, test_y, test_m, pairs, xml_notes, perform_notes, tempo, p_name = get_data.file2data(
        files=[xml, score, perform], measures=measures, mode="note", pair_path=pair_path, save_mid=True)
    tempo_rate = tempo / null_tempo
    print("     > tempo: {}".format(tempo))
    print()


    ## LOAD MODEL ## 
    print("** Load model **")
    # checkpoint_num = epoch_num
    module_name = "piano_cvae_model2_torch_{}".format(model_num)
    model = importlib.import_module(module_name)
    Generator = model.PerformGenerator
    mask = model.Mask
    note2group = model.Note2Group()

    model_path = "./model_cvae/piano_cvae_ckpt_{}".format(exp_num) # 
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")
    model = Generator(device=device)
    
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict']) 
    # model.encoder.load_state_dict(checkpoint['state_dictE']) 
    # model.decoder.load_state_dict(checkpoint['state_dictD']) 
    loss_val = checkpoint["loss_val"]
    checkpoint_num = len(loss_val)
    model.eval()
    print()

    # plot validation loss
    plt.figure()
    plt.plot(range(len(loss_val)), loss_val[:,1], label="recon_loss")
    plt.plot(range(len(loss_val)), loss_val[:,2], label="kld_c")
    plt.plot(range(len(loss_val)), loss_val[:,3], label="kld_z")
    plt.plot(range(len(loss_val)), loss_val[:,4], label="disc_c")
    plt.plot(range(len(loss_val)), loss_val[:,5], label="disc_z")
    plt.plot(range(len(loss_val)), loss_val[:,6], label="disc_d")
    plt.plot(range(len(loss_val)), loss_val[:,7], label="reg_loss")
    plt.ylim([-1,1.5])
    plt.legend(fontsize=5)
    plt.savefig("loss_val_{}_{}.png".format(exp_num, checkpoint_num))


    ## INFER LATENTS BY CONDITIONS ##
    indices = ["fast", "slow", "loud", "quiet", "stac", "legato", "neutral"]
    c_dict = dict()
    interp = 'tanh'
    n = 0
    
    # get latent variables by conditions
    # plt.figure(figsize=(10,20))
    for ind in indices:
        # get input data
        test_inputs_ = get_data.data2input(test_x, test_y, test_m, 
            cond=ind, art=True, mode="note", device=device)
        test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs_
        # test_inputs = test_inputs_[:-1]
        test_inputs = test_inputs_
        # test_inputs = [test_x_, test_y_, test_m_]

        # sample c
        if "cvrnn" in model_num or "fader" in model_num:
            c = note2group.reverse(test_clab_, test_m_)
        elif "pati" in model_num:
            z_moments1, z1 = model.sample_z_only(test_x_, test_y_, test_m_)
            c = z1[:,:,:3]         
        else:
            c = model.sample_c_only(test_y_, test_m_)

        # sample z
        # if "_pati" in model_num:
        #     z_moments, z = z_moments1, z1
        # else:
        #     z_moments, z = model.sample_z_only(x, y, m)

        c_dict[ind] = c

        # plt.subplot(int('71{}'.format(n+1)))
        # plt.imshow(np.transpose(z_moments[0][0].cpu().data.numpy()), aspect='auto')
        # plt.colorbar()
        # n += 1

    # plt.tight_layout()
    # plt.savefig("inferenced_z_mu_by_cond_{}_{}_{}_mm{}-{}.png".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]))        


    ## RECONSTRUCTION ##
    print("** Reconstruction **")
    if "pati" in model_num:
        s_note, _, \
        z_prior_moments, z_moments, \
        z, recon_note = model(*test_inputs)
    elif "cvrnn" in model_num:
        s_note, _, \
        z_prior_moments, z_moments, \
        z, recon_note = model(*test_inputs)
    elif "note" not in model_num:
        s_note, _, _, \
        z_prior_moments, c_moments, z_moments, \
        _, z, recon_note, _, \
        est_c, est_z, _ = model(*test_inputs)
    elif "note" in model_num:
        s_note, _, \
        z_prior_moments, c_moments, z_moments, \
        c, z, recon_note, \
        est_c, est_z, _ = model(test_x_, test_y_, test_m_, test_clab_)    

    # get results
    y_recon_vel, y_recon_dur, y_recon_ioi = \
        inverse_feature_note(recon_note, art=True, numpy=False, interp="tanh")
    if "note" in exp_num:
        est_c_ = est_c
        gt = test_y_[0].cpu().data.numpy() 
        clab = note2group.reverse(test_clab_, test_m_)[0].cpu().data.numpy()
    elif "cvae" in exp_num or "fader" in exp_num:
        est_c_ = torch.zeros(1, s_note.size(1), 3) 
        gt = test_y_[0].cpu().data.numpy()
        clab = note2group.reverse(test_clab_, test_m_)[0].cpu().data.numpy()
    else:
        est_c_ = note2group.reverse(est_c, test_m_) 
        gt = test_y2_[0].cpu().data.numpy()
        clab = test_clab_[0].cpu().data.numpy()
    y_c_vel, y_c_dur, y_c_ioi = \
        inverse_feature_note(est_c_, art=True, numpy=False, interp="tanh")

    est_c = est_c[0].cpu().data.numpy()

    plt.figure(figsize=(10,10))
    plt.subplot(311)
    plt.plot(range(len(est_c)), est_c[:,0], label="est.")
    plt.plot(range(len(est_c)), clab[:,0], label="clab_GT")
    plt.plot(range(len(est_c)), gt[:,0], label="GT")
    plt.legend()
    plt.subplot(312)
    plt.plot(range(len(est_c)), est_c[:,1], label="est.")
    plt.plot(range(len(est_c)), clab[:,1], label="clab_GT")
    plt.plot(range(len(est_c)), gt[:,1], label="GT")
    plt.legend()
    plt.subplot(313)
    plt.plot(range(len(est_c)), est_c[:,2], label="est.")
    plt.plot(range(len(est_c)), clab[:,2], label="clab_GT")
    plt.plot(range(len(est_c)), gt[:,2], label="GT")
    plt.legend()
    plt.tight_layout()
    plt.savefig("est_c_raw_{}_{}_{}_mm{}-{}.png".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]))


    # attribute means
    # est_mean = torch.mean(recon_note[0], dim=0).cpu().data.numpy()
    # test_clab_ = torch.mean(test_y_, dim=1)
    # print("     > data stats: (vel){:.4f} / (art){:.4f} / (ioi){:.4f}".format(
    #     test_clab_[0][0], test_clab_[0][1], test_clab_[0][2]))
    # print("     > gen stats: (vel){:.4f} / (art){:.4f} / (ioi){:.4f}".format(
    #     est_mean[0], est_mean[1], est_mean[2]))
    # print()


    ## SAMPLE ##
    print("** Sampling **")
    # i0 = torch.randn_like(i)

    if "cvrnn" in model_num:
        _, _, _, sampled0_note = model.sample(test_x_, test_m_, c_=c)
    else:
        _, _, z0, \
            sampled0_note = model.sample(test_x_, test_m_)
    y_sampled0 = inverse_feature_note(sampled0_note, art=True, numpy=False, interp=interp)

    # recon_group = recon_group[0].cpu().data.numpy()
    recon_group = corrupt_to_onset(test_x, recon_note[0].cpu().data.numpy(), same_onset_ind=same_onset_ind)
    # sampled0_group = corrupt_to_onset(test_x, sampled0_note[0].cpu().data.numpy(), same_onset_ind=same_onset_ind)

    # plot group-wise results
    plt.figure(figsize=(10,15))
    plt.subplot(311)
    plt.title("velocity")
    plt.plot(range(len(test_y2_[0])), test_y2_[0,:,0].cpu().data.numpy(), label="GT") # [0,:,0].cpu().data.numpy()
    plt.plot(range(len(recon_group)), recon_group[:,0], label="est")
    # plt.plot(range(len(recon_group)), sampled0_group[:,0], label="sampled")
    plt.legend()
    plt.subplot(312)
    plt.title("articulation")
    plt.plot(range(len(test_y2_[0])), test_y2_[0,:,1].cpu().data.numpy(), label="GT")
    plt.plot(range(len(recon_group)), recon_group[:,1], label="est")
    # plt.plot(range(len(recon_group)), sampled0_group[:,1], label="sampled")
    plt.legend()
    plt.subplot(313)
    plt.title("ioi")
    plt.plot(range(len(test_y2_[0])), test_y2_[0,:,-1].cpu().data.numpy(), label="GT")
    plt.plot(range(len(recon_group)), recon_group[:,-1], label="est")
    # plt.plot(range(len(recon_group)), sampled0_group[:,-1], label="sampled")
    plt.legend()
    plt.tight_layout()
    plt.savefig("recon_group_{}_{}_{}_mm{}-{}.png".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]))
    
    # # plot inferred/sampled z 
    roll = model.pianoroll(test_x_[:,:,22:110], test_m_)
    # plt.figure(figsize=(10,15))
    # plt.subplot(611)
    # plt.title("score note")
    # plt.imshow(np.transpose(s_note[0].cpu().data.numpy()), aspect='auto')
    # plt.colorbar()
    # # plt.subplot(612)
    # # plt.title("intermediate-score group")
    # # plt.imshow(np.transpose(s_group_[0].cpu().data.numpy()), aspect='auto')
    # # plt.colorbar()
    # # plt.subplot(613)
    # # plt.title("score group")
    # # plt.imshow(np.transpose(s_group[0].cpu().data.numpy()), aspect='auto')
    # # plt.colorbar()
    # plt.subplot(612)
    # plt.title("score roll")
    # plt.imshow(np.transpose(roll[0].cpu().data.numpy()), aspect="auto")
    # plt.colorbar()
    # # plt.subplot(614)
    # # plt.title("score attn matrix")
    # # plt.imshow(np.transpose(torch.sum(`s_attn`, dim=1)[0].cpu().data.numpy()), aspect="auto")
    # # plt.colorbar()
    # plt.subplot(613)
    # plt.title("z mu (inferenced)")
    # plt.imshow(np.transpose(z_moments[0][0].cpu().data.numpy()), aspect="auto")
    # plt.colorbar()
    # plt.subplot(614)
    # plt.title("z prior mu (inferenced)")
    # plt.imshow(np.transpose(z_prior_moments[0][0].cpu().data.numpy()), aspect="auto")
    # plt.colorbar()
    # plt.subplot(615)
    # plt.title("z sampled mu")
    # plt.imshow(np.transpose(z_moments0[0][0].cpu().data.numpy()), aspect="auto")
    # plt.colorbar()
    # plt.subplot(616)
    # plt.title("z sampled")
    # plt.imshow(np.transpose(z0[0].cpu().data.numpy()), aspect="auto")
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig("score_z_compare_{}_{}_{}_mm{}-{}.png".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]))
    # print()


    ## INTERPOLATION ##
    print("** Interpolation **")
    interp_t = dict()
    interp_d = dict()
    interp_a = dict()
    styles = dict()
    # tempo
    c_fast = c_dict["fast"]
    c_slow = c_dict["slow"]
    # c_fast = torch.mean(c_dict["fast"], dim=1)
    # c_slow = torch.mean(c_dict["slow"], dim=1)
    # c_fast = c_fast.unsqueeze(1).repeat(1, c.size(1), 1)
    # c_slow = c_slow.unsqueeze(1).repeat(1, c.size(1), 1)
    # 0 -> 1 : fast to slow
    for a in range(5):
        alpha = a / 4.
        # get latent variable
        c_seed_ = alpha * c_slow + (1-alpha) * c_fast 
        _, _, z_, sampled = model.sample(
            test_x_, test_m_, c_=c_seed_, z_=z, trunc=False, threshold=2)

        # sampled_ = sampled - torch.mean(sampled, dim=1).unsqueeze(1)

        # inverse to feature
        vel, dur, ioi = \
            inverse_feature_note(sampled, art=True, numpy=False, interp=interp)
        interp_t[a] = [alpha, [vel, dur, ioi], c_seed_, z_]
    print("     > sampled by tempo")
    # dynamics
    c_loud = c_dict["loud"]
    c_quiet = c_dict["quiet"]
    # c_loud = torch.mean(c_dict["loud"], dim=1)
    # c_quiet = torch.mean(c_dict["quiet"], dim=1)
    # c_quiet = c_quiet.unsqueeze(1).repeat(1, c.size(1), 1)
    # c_loud = c_loud.unsqueeze(1).repeat(1, c.size(1), 1)
    # 0 -> 1 : fast to slow
    for a in range(5):
        alpha = a / 4.
        # get latent variable
        c_seed_ = alpha * c_quiet + (1-alpha) * c_loud 
        _, _, z_, sampled = model.sample(
                test_x_, test_m_, c_=c_seed_, z_=z, trunc=False, threshold=2)

        # sampled_ = sampled - torch.mean(sampled, dim=1).unsqueeze(1)

        # inverse to feature
        vel, dur, ioi = \
            inverse_feature_note(sampled, art=True, numpy=False, interp=interp)
        interp_d[a] = [alpha, [vel, dur, ioi], c_seed_, z_]
    print("     > sampled by dynamics")
    # articulations
    c_stac = c_dict["stac"]
    c_leg = c_dict["legato"]
    # c_stac = torch.mean(c_dict["stac"], dim=1)
    # c_leg = torch.mean(c_dict["legato"], dim=1)
    # c_stac = c_stac.unsqueeze(1).repeat(1, c.size(1), 1)
    # c_leg = c_leg.unsqueeze(1).repeat(1, c.size(1), 1)
    # 0 -> 1 : fast to slow
    for a in range(5):
        alpha = a / 4.
        # get latent variable
        c_seed_ = alpha * c_leg + (1-alpha) * c_stac 
        _, _, z_, sampled = model.sample(
                test_x_, test_m_, c_=c_seed_, z_=z, trunc=False, threshold=2)

        # sampled_ = sampled - torch.mean(sampled, dim=1).unsqueeze(1)

        # inverse to feature
        vel, dur, ioi = \
            inverse_feature_note(sampled, art=True, numpy=False, interp=interp)
        interp_a[a] = [alpha, [vel, dur, ioi], c_seed_, z_]
    print("     > sampled by articulations")

    # sample various versions
    # c_neut = torch.mean(c_dict["neutral"], dim=1)
    # c_seed_ = c_neut.unsqueeze(1).repeat(1, c.size(1), 1)
    c_seed_ = torch.empty_like(c_dict["neutral"]).copy_(c_dict["neutral"])
    # c_seed_ = torch.flip(c_seed_, dims=[1])
    for a in range(3):
        _, _, z_, sampled = model.sample(test_x_, test_m_, c_=c_seed_) # c_=c_seed_, 
        # inverse to feature
        vel, dur, ioi = \
            inverse_feature_note(sampled, art=True, numpy=False, interp=interp)
        styles[a] = [alpha, [vel, dur, ioi], c_seed_, z_]
    print("     > sampled multiple styles (neutral)")
    print()


    # ## CONTROLLING FADER ##
    # print("** Controlling fader **")
    # # get fadered constants
    # if "note" not in model_num:
    #     t = len(test_y2_[0])
    # elif "note" in model_num:
    #     t = len(test_y_[0])
    # start, end = measures 
    # middle = start + ((end+1 - start) // 2)
    # pairs_onset = make_onset_pairs(pairs, fmt="xml")
    # for n, onset in enumerate(pairs_onset):
    #     if onset[0]['xml_note'][1].measure_number == middle-1:
    #         break
    # # prev_range = n 
    # # next_range = t - n
    # # prev_linspace = np.linspace(-3, 3, num=prev_range, endpoint=False)
    # # next_linspace = np.linspace(3, -3, num=next_range, endpoint=True)
    # # add_seq_ = np.concatenate([prev_linspace, next_linspace], axis=0)
    # all_linspace = np.linspace(2, -2, num=t, endpoint=True)
    # add_seq_ = all_linspace
    # assert len(add_seq_) == t
    # add_seq = torch.from_numpy(add_seq_).to(device).float()
    # c_rand = torch.empty_like(c_dict["neutral"]).copy_(c_dict["neutral"])
    # # c_rand = c_rand[:,0].unsqueeze(1).repeat(1, t, 1)
    # # tempo
    # s_i = torch.empty_like(c_rand).copy_(c_rand)
    # # i_new = s_i[:,:,8].clone() + add_seq.unsqueeze(0)
    # # i_new = add_seq.unsqueeze(0)
    # i_new = torch.mean(s_i[:,:,8], dim=1).unsqueeze(1).repeat(1, t) + add_seq.unsqueeze(0)
    # s_i_new = torch.cat([s_i[:,:,:8], i_new.unsqueeze(-1), s_i[:,:,9:]], dim=-1)
    # if "noZ" in model_num:
    #     _, i_sampled = model.sample(
    #         test_x_, test_m_, c_=s_i_new, trunc=False, threshold=2) 
    # else:
    #     _, _, _, i_sampled = model.sample(
    #         test_x_, test_m_, c_=s_i_new, trunc=False, threshold=2) 
    # fader_i = inverse_feature_note(i_sampled, art=True, numpy=False, interp=interp)
    # # dynamics 
    # s_d = torch.empty_like(c_rand).copy_(c_rand)
    # # d_new = s_d[:,:,0].clone() + add_seq.unsqueeze(0)
    # # d_new = add_seq.unsqueeze(0)
    # d_new = torch.mean(s_d[:,:,0], dim=1).unsqueeze(1).repeat(1, t) + add_seq.unsqueeze(0)
    # s_d_new = torch.cat([d_new.unsqueeze(-1), s_d[:,:,1:]], dim=-1)
    # if "noZ" in model_num:
    #     _, d_sampled = model.sample(
    #         test_x_, test_m_, c_=s_d_new, trunc=False, threshold=2) 
    # else:
    #     _, _, _, d_sampled = model.sample(
    #         test_x_, test_m_, c_=s_d_new, trunc=False, threshold=2) 
    # fader_d = inverse_feature_note(d_sampled, art=True, numpy=False, interp=interp)
    # # articulation
    # s_a = torch.empty_like(c_rand).copy_(c_rand)
    # # a_new = s_a[:,:,4].clone() + add_seq.unsqueeze(0)
    # # a_new = add_seq.unsqueeze(0)
    # a_new = torch.mean(s_a[:,:,4], dim=1).unsqueeze(1).repeat(1, t) + add_seq.unsqueeze(0)
    # s_a_new = torch.cat([s_a[:,:,:4], a_new.unsqueeze(-1), s_a[:,:,5:]], dim=-1)
    # if "noZ" in model_num:
    #     _, a_sampled = model.sample(
    #         test_x_, test_m_, c_=s_a_new, trunc=False, threshold=2) 
    # else:
    #     _, _, _, a_sampled = model.sample(
    #         test_x_, test_m_, c_=s_a_new, trunc=False, threshold=2) 
    # fader_a = inverse_feature_note(a_sampled, art=True, numpy=False, interp=interp)
    # print()

    ## PLOT ##
    print("** Plot results **")
    y_vel = test_y[:,0]
    y_ioi = test_y[:,2]
    y_art = np.power(10, np.log10(test_y[:,1]) - np.log10(test_y[:,3]))
    # y_norm = test_y_ - torch.mean(test_y_, dim=1).unsqueeze(1)
    # y_all = inverse_feature_note(y_norm, art=True, numpy=False, interp=interp)
    # y_vel = y_all[0]
    # y_art = y_all[1]
    # y_ioi = y_all[2]

    # plt.figure(figsize=(10,12))
    # plt.subplot(311)
    # plt.title("Sampled(fader) velocity")
    # plt.plot(range(len(test_x)), y_vel, label="GT")
    # plt.plot(range(len(test_x)), fader_i[0], label="fader_ioi")
    # plt.plot(range(len(test_x)), fader_d[0], label="fader_vel")
    # plt.plot(range(len(test_x)), fader_a[0], label="fader_art")
    # plt.legend()
    # plt.subplot(312)
    # plt.title("Sampled(fader) articulation")
    # plt.plot(range(len(test_x)), y_art, label="GT")
    # plt.plot(range(len(test_x)), fader_i[1], label="fader_ioi")
    # plt.plot(range(len(test_x)), fader_d[1], label="fader_vel")
    # plt.plot(range(len(test_x)), fader_a[1], label="fader_art")
    # plt.ylim([0, 8])
    # plt.legend()
    # plt.subplot(313)
    # plt.title("Sampled(fader) IOI")
    # plt.plot(range(len(test_x)), y_ioi, label="GT")
    # plt.plot(range(len(test_x)), fader_i[2], label="fader_ioi")
    # plt.plot(range(len(test_x)), fader_d[2], label="fader_vel")
    # plt.plot(range(len(test_x)), fader_a[2], label="fader_art")
    # plt.ylim([0, 8])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join("./gen_fader_{}_{}_{}_mm{}-{}.png".format(song_name, exp_num, checkpoint_num, measures[0], measures[1])))

    # sampled data by conditions (tempo, dynamics)
    plt.figure(figsize=(10,25))
    plt.subplot(711)
    plt.title("Sampled/Predicted velocity (change tempo)")
    plt.plot(range(len(test_x)), y_vel, label="GT")
    plt.plot(range(len(test_x)), y_recon_vel, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[0], label="sampled")
    plt.plot(range(len(test_x)), y_c_vel, label="est_c")
    plt.plot(range(len(test_x)), interp_t[0][1][0], label="slow=0 / fast=1")
    plt.plot(range(len(test_x)), interp_t[1][1][0], label="slow=0.25 / fast=0.75")
    plt.plot(range(len(test_x)), interp_t[2][1][0], label="slow=0.5 / fast=0.5")
    plt.plot(range(len(test_x)), interp_t[3][1][0], label="slow=0.75 / fast=0.25")
    plt.plot(range(len(test_x)), interp_t[4][1][0], label="slow=1 / fast=0")
    plt.legend()
    plt.subplot(712)
    plt.title("Sampled/Predicted duration (change tempo)")
    plt.plot(range(len(test_x)), y_art, label="GT")
    plt.plot(range(len(test_x)), y_recon_dur, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[1], label="sampled")
    plt.plot(range(len(test_x)), y_c_dur, label="est_c")
    plt.plot(range(len(test_x)), interp_t[0][1][1], label="slow=0 / fast=1")
    plt.plot(range(len(test_x)), interp_t[1][1][1], label="slow=0.25 / fast=0.75")
    plt.plot(range(len(test_x)), interp_t[2][1][1], label="slow=0.5 / fast=0.5")
    plt.plot(range(len(test_x)), interp_t[3][1][1], label="slow=0.75 / fast=0.25")
    plt.plot(range(len(test_x)), interp_t[4][1][1], label="slow=1 / fast=0")
    plt.ylim([0, 8])
    plt.legend()
    plt.subplot(713)
    plt.title("Sampled/Predicted IOI (change tempo)")
    plt.plot(range(len(test_x)), y_ioi, label="GT")
    plt.plot(range(len(test_x)), y_recon_ioi, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[2], label="sampled")
    plt.plot(range(len(test_x)), y_c_ioi, label="est_c")
    plt.plot(range(len(test_x)), interp_t[0][1][2], label="slow=0 / fast=1")
    plt.plot(range(len(test_x)), interp_t[1][1][2], label="slow=0.25 / fast=0.75")
    plt.plot(range(len(test_x)), interp_t[2][1][2], label="slow=0.5 / fast=0.5")
    plt.plot(range(len(test_x)), interp_t[3][1][2], label="slow=0.75 / fast=0.25")
    plt.plot(range(len(test_x)), interp_t[4][1][2], label="slow=1 / fast=0")
    plt.ylim([0, 8])
    plt.legend()
    plt.subplot(714)
    plt.title("Sampled/Predicted velocity (change dynamics)")
    plt.plot(range(len(test_x)), y_vel, label="GT")
    plt.plot(range(len(test_x)), y_recon_vel, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[0], label="sampled")
    plt.plot(range(len(test_x)), y_c_vel, label="est_c")
    plt.plot(range(len(test_x)), interp_d[0][1][0], label="quiet=0 / loud=1")
    plt.plot(range(len(test_x)), interp_d[1][1][0], label="quiet=0.25 / loud=0.75")
    plt.plot(range(len(test_x)), interp_d[2][1][0], label="quiet=0.5 / loud=0.5")
    plt.plot(range(len(test_x)), interp_d[3][1][0], label="quiet=0.75 / loud=0.25")
    plt.plot(range(len(test_x)), interp_d[4][1][0], label="quiet=1 / loud=0")
    plt.legend()
    plt.subplot(715)
    plt.title("Sampled/Predicted duration (change dynamics)")
    plt.plot(range(len(test_x)), y_art, label="GT")
    plt.plot(range(len(test_x)), y_recon_dur, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[1], label="sampled")
    plt.plot(range(len(test_x)), y_c_dur, label="est_c")
    plt.plot(range(len(test_x)), interp_d[0][1][1], label="quiet=0 / loud=1")
    plt.plot(range(len(test_x)), interp_d[1][1][1], label="quiet=0.25 / loud=0.75")
    plt.plot(range(len(test_x)), interp_d[2][1][1], label="quiet=0.5 / loud=0.5")
    plt.plot(range(len(test_x)), interp_d[3][1][1], label="quiet=0.75 / loud=0.25")
    plt.plot(range(len(test_x)), interp_d[4][1][1], label="quiet=1 / loud=0")
    plt.legend()
    plt.ylim([0, 8])
    plt.subplot(716)
    plt.title("Sampled/Predicted IOI (change dynamics)")
    plt.plot(range(len(test_x)), y_ioi, label="GT")
    plt.plot(range(len(test_x)), y_recon_ioi, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[2], label="sampled")
    plt.plot(range(len(test_x)), y_c_ioi, label="est_c")
    plt.plot(range(len(test_x)), interp_d[0][1][2], label="quiet=0 / loud=1")
    plt.plot(range(len(test_x)), interp_d[1][1][2], label="quiet=0.25 / loud=0.75")
    plt.plot(range(len(test_x)), interp_d[2][1][2], label="quiet=0.5 / loud=0.5")
    plt.plot(range(len(test_x)), interp_d[3][1][2], label="quiet=0.75 / loud=0.25")
    plt.plot(range(len(test_x)), interp_d[4][1][2], label="quiet=1 / loud=0")
    plt.legend()
    plt.ylim([0, 8])
    plt.subplot(717)
    plt.title("score roll")
    plt.imshow(np.transpose(roll[0].cpu().data.numpy()), aspect='auto')
    plt.tight_layout()
    plt.savefig(os.path.join("./gen_control_tempo_dynamics_{}_{}_{}_mm{}-{}.png".format(song_name, exp_num, checkpoint_num, measures[0], measures[1])))
    # sampled data by conditions (articulation)
    plt.figure(figsize=(10,25))
    plt.subplot(711)
    plt.title("Sampled/Predicted velocity (change articulation)")
    plt.plot(range(len(test_x)), y_vel, label="GT")
    plt.plot(range(len(test_x)), y_recon_vel, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[0], label="sampled")
    plt.plot(range(len(test_x)), y_c_vel, label="est_c")
    plt.plot(range(len(test_x)), interp_a[0][1][0], label="legato=0 / stac=1")
    plt.plot(range(len(test_x)), interp_a[1][1][0], label="legato=0.25 / stac=0.75")
    plt.plot(range(len(test_x)), interp_a[2][1][0], label="legato=0.5 / stac=0.5")
    plt.plot(range(len(test_x)), interp_a[3][1][0], label="legato=0.75 / stac=0.25")
    plt.plot(range(len(test_x)), interp_a[4][1][0], label="legato=1 / stac=0")
    plt.legend()
    plt.subplot(712)
    plt.title("Sampled/Predicted duration (change articulation)")
    plt.plot(range(len(test_x)), y_art, label="GT")
    plt.plot(range(len(test_x)), y_recon_dur, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[1], label="sampled")
    plt.plot(range(len(test_x)), y_c_dur, label="est_c")
    plt.plot(range(len(test_x)), interp_a[0][1][1], label="legato=0 / stac=1")
    plt.plot(range(len(test_x)), interp_a[1][1][1], label="legato=0.25 / stac=0.75")
    plt.plot(range(len(test_x)), interp_a[2][1][1], label="legato=0.5 / stac=0.5")
    plt.plot(range(len(test_x)), interp_a[3][1][1], label="legato=0.75 / stac=0.25")
    plt.plot(range(len(test_x)), interp_a[4][1][1], label="legato=1 / stac=0")
    plt.legend()
    plt.ylim([0, 8])
    plt.subplot(713)
    plt.title("Sampled/Predicted IOI (change articulation)")
    plt.plot(range(len(test_x)), y_ioi, label="GT")
    plt.plot(range(len(test_x)), y_recon_ioi, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[2], label="sampled")
    plt.plot(range(len(test_x)), y_c_ioi, label="est_c")
    plt.plot(range(len(test_x)), interp_a[0][1][2], label="legato=0 / stac=1")
    plt.plot(range(len(test_x)), interp_a[1][1][2], label="legato=0.25 / stac=0.75")
    plt.plot(range(len(test_x)), interp_a[2][1][2], label="legato=0.5 / stac=0.5")
    plt.plot(range(len(test_x)), interp_a[3][1][2], label="legato=0.75 / stac=0.25")
    plt.plot(range(len(test_x)), interp_a[4][1][2], label="legato=1 / stac=0")
    plt.legend()
    plt.ylim([0, 8])
    plt.subplot(714)
    plt.title("Sampled/Predicted velocity (multiple styles)")
    plt.plot(range(len(test_x)), y_vel, label="GT")
    plt.plot(range(len(test_x)), y_recon_vel, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[0], label="sampled")
    plt.plot(range(len(test_x)), y_c_vel, label="est_c")
    plt.plot(range(len(test_x)), styles[0][1][0], label="style 1")
    plt.plot(range(len(test_x)), styles[1][1][0], label="style 2")
    plt.plot(range(len(test_x)), styles[2][1][0], label="style 3")
    plt.legend()
    plt.subplot(715)
    plt.title("Sampled/Predicted duration (multiple styles)")
    plt.plot(range(len(test_x)), y_art, label="GT")
    plt.plot(range(len(test_x)), y_recon_dur, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[1], label="sampled")
    plt.plot(range(len(test_x)), y_c_dur, label="est_c")
    plt.plot(range(len(test_x)), styles[0][1][1], label="style 1")
    plt.plot(range(len(test_x)), styles[1][1][1], label="style 2")
    plt.plot(range(len(test_x)), styles[2][1][1], label="style 3")
    plt.legend()
    plt.ylim([0, 8])
    plt.subplot(716)
    plt.title("Sampled/Predicted IOI (multiple styles)")
    plt.plot(range(len(test_x)), y_ioi, label="GT")
    plt.plot(range(len(test_x)), y_recon_ioi, label="pred")
    # plt.plot(range(len(test_x)), y_sampled0[2], label="sampled")
    plt.plot(range(len(test_x)), y_c_ioi, label="est_c")
    plt.plot(range(len(test_x)), styles[0][1][2], label="style 1")
    plt.plot(range(len(test_x)), styles[1][1][2], label="style 2")
    plt.plot(range(len(test_x)), styles[2][1][2], label="style 3")
    plt.legend()
    plt.ylim([0, 8])
    plt.subplot(717)
    plt.title("score roll")
    plt.imshow(np.transpose(roll[0].cpu().data.numpy()), aspect='auto')
    plt.tight_layout()
    plt.savefig(os.path.join("./gen_control_articulation_{}_{}_{}_mm{}-{}.png".format(song_name, exp_num, checkpoint_num, measures[0], measures[1])))
    
    # sampled C
    plt.figure(figsize=(10,10))
    plt.subplot(311)
    plt.title("C latent variable (ioi)")
    plt.plot(range(c_seed_.size(-1)), interp_t[0][-2][0][0].cpu().data.numpy(), label="slow=0 / fast=1")
    plt.plot(range(c_seed_.size(-1)), interp_t[1][-2][0][0].cpu().data.numpy(), label="slow=0.25 / fast=0.75")
    plt.plot(range(c_seed_.size(-1)), interp_t[2][-2][0][0].cpu().data.numpy(), label="slow=0.5 / fast=0.5")
    plt.plot(range(c_seed_.size(-1)), interp_t[3][-2][0][0].cpu().data.numpy(), label="slow=0.75 / fast=0.25")
    plt.plot(range(c_seed_.size(-1)), interp_t[4][-2][0][0].cpu().data.numpy(), label="slow=1 / fast=0")
    plt.xlim([0,8])
    plt.legend()
    plt.subplot(312)
    plt.title("C latent variable (vel)")
    plt.plot(range(c_seed_.size(-1)), interp_d[0][-2][0][0].cpu().data.numpy(), label="quiet=0 / loud=1")
    plt.plot(range(c_seed_.size(-1)), interp_d[1][-2][0][0].cpu().data.numpy(), label="quiet=0.25 / loud=0.75")
    plt.plot(range(c_seed_.size(-1)), interp_d[2][-2][0][0].cpu().data.numpy(), label="quiet=0.5 / loud=0.5")
    plt.plot(range(c_seed_.size(-1)), interp_d[3][-2][0][0].cpu().data.numpy(), label="quiet=0.75 / loud=0.75")
    plt.plot(range(c_seed_.size(-1)), interp_d[4][-2][0][0].cpu().data.numpy(), label="quiet=1 / loud=0")
    plt.xlim([0,8])
    plt.legend()
    plt.subplot(313)
    plt.title("C latent variable (art)")
    plt.plot(range(c_seed_.size(-1)), interp_a[0][-2][0][0].cpu().data.numpy(), label="legato=0 / stac=1")
    plt.plot(range(c_seed_.size(-1)), interp_a[1][-2][0][0].cpu().data.numpy(), label="legato=0.25 / stac=0.75")
    plt.plot(range(c_seed_.size(-1)), interp_a[2][-2][0][0].cpu().data.numpy(), label="legato=0.5 / stac=0.5")
    plt.plot(range(c_seed_.size(-1)), interp_a[3][-2][0][0].cpu().data.numpy(), label="legato=0.75 / stac=0.75")
    plt.plot(range(c_seed_.size(-1)), interp_a[4][-2][0][0].cpu().data.numpy(), label="legato=1 / stac=0")
    plt.xlim([0,8])
    plt.legend()
    plt.tight_layout()
    plt.savefig("c_sampled_by_cond_{}_{}_{}_mm{}-{}.png".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]))

    # sampled Z
    # plt.figure(figsize=(10,15))
    # plt.subplot(811)
    # plt.title("score roll")
    # plt.imshow(np.transpose(roll[0].cpu().data.numpy()), aspect='auto')
    # plt.colorbar()
    # # plt.subplot(812)
    # # plt.title("score att matrix")
    # # plt.imshow(np.transpose(s_attn[0][0].cpu().data.numpy()), aspect='auto')
    # # plt.colorbar()
    # plt.subplot(813)
    # plt.title("tempo (fast 100%)")
    # plt.imshow(np.transpose(interp_t[0][-1][0].cpu().data.numpy()), aspect='auto')
    # plt.colorbar()
    # plt.subplot(814)
    # plt.title("tempo (slow 100%)")
    # plt.imshow(np.transpose(interp_t[4][-1][0].cpu().data.numpy()), aspect="auto")
    # plt.colorbar()
    # plt.subplot(815)
    # plt.title("dynamics (loud 100%)")
    # plt.imshow(np.transpose(interp_d[0][-1][0].cpu().data.numpy()), aspect="auto")
    # plt.colorbar()
    # plt.subplot(816)
    # plt.title("dynamics (quiet 100%)")
    # plt.imshow(np.transpose(interp_d[4][-1][0].cpu().data.numpy()), aspect="auto")
    # plt.colorbar()
    # plt.subplot(817)
    # plt.title("articulation (stac 100%)")
    # plt.imshow(np.transpose(interp_a[0][-1][0].cpu().data.numpy()), aspect="auto")
    # plt.colorbar()
    # plt.subplot(818)
    # plt.title("articulation (legato 100%)")
    # plt.imshow(np.transpose(interp_a[4][-1][0].cpu().data.numpy()), aspect="auto")
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig("z_sampled_by_cond_{}_{}_{}.png".format(song_name, exp_num, checkpoint_num))
    # print()


    ### RENDER MIDI ###
    print("** Render MIDI files** ")
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_t[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind, savename="slow_0_fast_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_t[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="slow_100_fast_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_d[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="quiet_0_loud_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_d[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="quiet_100_loud_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_a[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="stac_0_leg_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_a[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="stac_100_leg_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample1_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[1][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample2_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[2][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample3_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=y_sampled0, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="neutral_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader_i, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="fader_i_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader_d, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="fader_d_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    # inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=fader_a, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="fader_a_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=True)
    
    # GT
    y_features = [test_y[:,0], y_art, y_ioi]
    gt_notes = inverse_rendering_art_note(input_notes=xml_notes, save_dir="./", cond=test_x, features=y_features, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="gt_features_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False, return_notes=True)
    print()
    print("#################################")
    print()


def test_model_transfer(
    song_name=None, measures=None,
    model_num=None, exp_num=None, epoch_num=None, same_onset_ind=[110,112]):

    pair_path = None

    # test data paths
    # song_name1 = "Schubert_Impromptu_op.90_D.899__4"
    # song_name2 = "Liszt_Gran_Etudes_de_Paganini__2_La_campanella"
    # song_name1 = "Mozart_Piano_Sonatas__11-3"
    # song_name1 = "Beethoven_Piano_Sonatas__8-2"
    # song_name2 = "Beethoven_Piano_Sonatas__14-3"
    # same_onset_ind = [110,112]
    # measures1 = [1,16]
    # measures2 = [1,16]
    # measures1 = [217,232]
    # measures2 = [149,156]
    measures1, measures2 = measures
    song_name1, song_name2 = song_name
    song_name1_ =  '/'.join(song_name1.split('__'))
    song_name2_ =  '/'.join(song_name2.split('__'))
    parent_path = "/data/asap_dataset/exp_data/listening_test/raw"
    # song 1
    perform1 = [p for p in sorted(glob(os.path.join(parent_path, "{}/01/*.mid".format(
        song_name1_)))) if "cleaned" not in p][0]
    xml1 = os.path.join(parent_path, "{}/musicxml_cleaned_plain.musicxml".format(song_name1_))
    score1 = os.path.join(parent_path, "{}/score_plain.mid".format(song_name1_))
    pair1 = os.path.join(os.path.dirname(perform1), 'xml_score_perform_pairs.npy')
    if os.path.exists(pair1) is True:
        pair_path1 = pair1 
    else:
        pair_path1 = None
    # pair_path = None
    # song 2
    perform2 = [p for p in sorted(glob(os.path.join(parent_path, "{}/01/*.mid".format(
        song_name2_)))) if "cleaned" not in p][0]
    xml2 = os.path.join(parent_path, "{}/musicxml_cleaned_plain.musicxml".format(song_name2_))
    score2 = os.path.join(parent_path, "{}/score_plain.mid".format(song_name2_))
    pair2 = os.path.join(os.path.dirname(perform2), 'xml_score_perform_pairs.npy')
    if os.path.exists(pair2) is True:
        pair_path2 = pair2 
    else:
        pair_path2 = None

    ## LOAD DATA ##
    # model_num = 'exp1813'
    # exp_num = 'exp1814_p2'
    # epoch_num = '100'
    print()
    print()
    print("############# TEST ##############")
    print("> MODEL: {}\n> EXP: {}\n> EPOCH: {}".format(model_num, exp_num, epoch_num))
    print("#################################")
    print()
    print("** Load Data **")

    null_tempo = 120
    get_data = GetData(null_tempo=null_tempo, 
                       same_onset_ind=same_onset_ind,
                       stat=np.mean)
    # song 1
    test_x1, test_y1, test_m1, pairs1, xml_notes1, perform_notes1, tempo1, p_name1 = get_data.file2data(
        files=[xml1, score1, perform1], measures=measures1, mode="note", pair_path=pair_path1)
    tempo_rate1 = tempo1 / null_tempo
    print("     > tempo: {}".format(tempo1))
    print()
    # song 2
    test_x2, test_y2, test_m2, pairs2, xml_notes2, perform_notes2, tempo2, p_name2 = get_data.file2data(
        files=[xml2, score2, perform2], measures=measures2, mode="note", pair_path=pair_path2)
    tempo_rate2 = tempo2 / null_tempo
    print("     > tempo: {}".format(tempo2))
    print()


    ## LOAD MODEL ## 
    print("** Load model **")
    checkpoint_num = epoch_num
    module_name = "piano_cvae_model2_torch_{}".format(model_num)
    model = importlib.import_module(module_name)
    Generator = model.PerformGenerator
    note2group = model.Note2Group()

    model_path = "./model_cvae/piano_cvae_ckpt_{}".format(exp_num) # 
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:3" if cuda_condition else "cpu")
    model = Generator(device=device)
    
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict']) 
    loss_val = checkpoint["loss_val"]
    model.eval()
    print()

    ## INFER LATENTS BY CONDITIONS ##
    interp = 'tanh'
    ind = None
    # get input data
    # input 1
    test_inputs1_ = get_data.data2input(test_x1, test_y1, test_m1, 
        cond=ind, art=True, mode="note", device=device)
    test_inputs1 = test_inputs1_
    # input 2
    test_inputs2_ = get_data.data2input(test_x2, test_y2, test_m2, 
        cond=ind, art=True, mode="note", device=device)
    test_inputs2 = test_inputs2_
    test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs2_

    if "exp" in model_num.split("_")[-1]:
        _, _, _, \
        _, c_moments, z_moments, \
        c1, _, _, _, \
        _, _, _ = model(*test_inputs1) #, _, _
    elif "_noZ" in model_num:
        _, _, \
        c_moments, \
        c1, _, _ = model(*test_inputs1) #, _, _
    elif "note" in model_num:
        _, _, \
        _, c_moments, z_moments, \
        c1, _, _, \
        _, _ = model(*test_inputs1) #, _, _

    if "exp" in model_num.split("_")[-1]:
        _, _, _, \
        _, c_moments, z_moments, \
        c2, _, _, _, \
        _, _, _ = model(*test_inputs2) #, _, _
    elif "_noZ" in model_num:
        _, _, \
        c_moments, \
        c2, _, _ = model(*test_inputs2) #, _, _
    elif "note" in model_num:
        _, _, \
        _, c_moments, z_moments, \
        c2, _, _, \
        _, _ = model(*test_inputs2) #, _, _

    c_from = c1 
    c_to = c2  

    ## SAMPLE ##
    print("** Sampling **")
    # i0 = torch.randn_like(i)
    
    c_from_mean = torch.mean(c_from, dim=1)
    c_to_mean = torch.mean(c_to, dim=1)
    
    c_diff = c_from_mean - c_to_mean


    if c_from.size(1) >= c_to.size(1):
        c_from = c_from[:,:c_to.size(1)]
    elif c_from.size(1) < c_to.size(1):
        diff = c_to.size(1) - c_from.size(1)
        c_from_last_tile = c_from[:,-1:].repeat(1, diff, 1)
        c_from = torch.cat([c_from, c_from_last_tile], dim=1)

    c_to = c_to + c_diff.unsqueeze(1).repeat(1, c_to.size(1), 1)    

    if "exp" in model_num.split("_")[-1]:
        _, _, _, \
            sampled1 = model.sample(test_x_, test_m_, c_=c_from, trunc=True)
        _, _, _, \
            sampled2 = model.sample(test_x_, test_m_, c_=c_to, trunc=True)
    elif "_noZ" in model_num:
        _, sampled1 = model.sample(test_x_, test_m_, c_=c_from, trunc=True)
        _, sampled2 = model.sample(test_x_, test_m_, c_=c_to, trunc=True)        

    y_sampled1 = inverse_feature_note(sampled1, art=True, numpy=False, interp=interp)
    y_sampled2 = inverse_feature_note(sampled2, art=True, numpy=False, interp=interp)


    ### RENDER MIDI ###
    print("** Render MIDI files** ")
    inverse_rendering_art_note(input_notes=xml_notes2, save_dir="./", cond=test_x2, features=y_sampled1, tempo=tempo2, tempo_rate=tempo_rate2,  same_onset_ind=same_onset_ind, savename="after_transfer_{}_to_{}_exp{}_{}_mm{}-{}.mid".format(song_name1, song_name2, exp_num, checkpoint_num, measures2[0], measures2[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes2, save_dir="./", cond=test_x2, features=y_sampled2, tempo=tempo2, tempo_rate=tempo_rate2,  same_onset_ind=same_onset_ind, savename="before_transfer_{}_to_{}_exp{}_{}_mm{}-{}.mid".format(song_name1, song_name2, exp_num, checkpoint_num, measures2[0], measures2[1]), save_score=False)
    
    # GT
    y_ioi1 = test_y1[:,2]
    y_art1 = np.power(10, np.log10(test_y1[:,1]) - np.log10(test_y1[:,3]))
    y_features1 = [test_y1[:,0], y_art1, y_ioi1]
    y_ioi2 = test_y2[:,2]
    y_art2 = np.power(10, np.log10(test_y2[:,1]) - np.log10(test_y2[:,3]))
    y_features2 = [test_y2[:,0], y_art2, y_ioi2]
    inverse_rendering_art_note(input_notes=xml_notes1, save_dir="./", cond=test_x1, features=y_features1, tempo=tempo1, tempo_rate=tempo_rate1,  same_onset_ind=same_onset_ind,  savename="gt_features1_{}_exp{}_{}_mm{}-{}.mid".format(song_name1, exp_num, checkpoint_num, measures1[0], measures1[1]), save_score=False)
    inverse_rendering_art_note(input_notes=xml_notes2, save_dir="./", cond=test_x2, features=y_features2, tempo=tempo2, tempo_rate=tempo_rate2,  same_onset_ind=same_onset_ind,  savename="gt_features2_{}_exp{}_{}_mm{}-{}.mid".format(song_name2, exp_num, checkpoint_num, measures2[0], measures2[1]), save_score=False)
    print()
    print("#################################")
    print()



def save_all_test_samples(
    model_num=None, exp_num=None, epoch_num=None, device_num=None, same_onset_ind=[110,112]):

    # model_num = 'DP'
    # exp_num = 'DP'
    # epoch_num = '100'
    # device_num = 3 
    same_onset_ind = [110,112]
    print()
    print()
    print("############# TEST ##############")
    print("> MODEL: {}\n> EXP: {}\n> EPOCH: {}".format(model_num, exp_num, epoch_num))
    print("#################################")
    print()

    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")

    if "GT" in model_num or model_num == "DP":
        model = None 

    else:
        ## LOAD MODEL ## 
        print("** Load model **")
        checkpoint_num = epoch_num
        module_name = "piano_cvae_model2_torch_{}".format(model_num)
        model = importlib.import_module(module_name)
        note2group = model.Note2Group()
        # Generator = model.PerformGenerator
        model_path = "./model_cvae/piano_cvae_ckpt_{}".format(exp_num) # 
        if "exp" in model_num: # proposed(group)
            Generator = model.PerformGenerator
        elif "nms" in model_num: 
            Generator = model.NMSLatentDisentangledDynamic 
        model = Generator(device=device)
        model.to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict']) 
        model.eval()

    parent_path = "/data/asap_dataset/exp_data/listening_test/raw"
    categs = sorted(glob(os.path.join(parent_path, '*/')))
    song_list = list()
    for categ in categs:
        pieces = sorted(glob(os.path.join(categ, '*/')))
        for piece in pieces:
            song_list.append(piece)
    song_list = sorted(song_list)
    indices = np.load("./ACCESS/access_song_start_point_dict.npy", allow_pickle=True).tolist()
    # max_len = 16

    for Set in indices:
        song_indices = indices[Set]
        save_path = "./ACCESS/test_samples/Set{}/{}_Set{}".format(Set+1, exp_num, Set+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        song_num = 1
        for song in song_indices:
            song_path = song_list[song]

            if "Bach" in song_path:
                continue
            # song_name = "Schubert_Impromptu_op90_D899__4"
            # song_name = "Beethoven_Piano_Sonatas__8-2"
            # same_onset_ind = [110,112]
            song_name =  '__'.join(song_path.split('/')[-3:-1])
            perform = [p for p in sorted(glob(
                os.path.join(song_path, "01/*.mid"))) if "cleaned" not in p][0]
            xml = os.path.join(song_path, "musicxml_cleaned_plain.musicxml")
            score = os.path.join(song_path, "score_plain.mid")
            pair = os.path.join(os.path.dirname(perform), 'xml_score_perform_pairs.npy')
            if os.path.exists(pair) is True:
                pair_path = pair 
            else:
                pair_path = None

            ## LOAD DATA ##
            null_tempo = 120
            measures = song_indices[song][0]
            get_data = GetData(null_tempo=null_tempo, 
                            same_onset_ind=same_onset_ind,
                            stat=np.mean)
            test_x, test_y, test_m, pairs, xml_notes, perform_notes, tempo, p_name = \
                get_data.file2data(files=[xml, score, perform], 
                measures=measures, mode="note", pair_path=pair_path, save_mid=False)
            tempo_rate = tempo / null_tempo
            # print("     > tempo: {}".format(tempo))
            # print()

            # get input data
            test_inputs_ = get_data.data2input(
                test_x, test_y, test_m, art=True, mode="note", device=device)
            test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs_
            test_inputs = test_inputs_
            test_inputs2 = [test_x_, test_y_, test_m_, test_clab_]
            y_vel = test_y[:,0]
            y_ioi = test_y[:,2]
            y_art = np.power(10, np.log10(test_y[:,1]) - np.log10(test_y[:,3])) 
            features_gt = [y_vel, y_art, y_ioi]

            if "GT" not in model_num and model_num != "DP":
                # forward
                if "_cvrnn" in model_num or "_fader" in model_num:
                    c_ = note2group.reverse(test_clab_, test_m_)
                elif "_pati" in model_num:
                    _, c_ = model.sample_z_only(test_y_, test_m_)
                else:
                    c_ = model.sample_c_only(test_y_, test_m_)
                
                c = c_
                # c = torch.zeros_like(c_)
                # c = torch.mean(c_, dim=1).unsqueeze(1).repeat(1, c_.size(1), 1)

                # sample 
                if "fader" in model_num or "pati" in model_num:
                    _, _, sampled = model.sample(
                        test_x_, test_m_, c_=c, trunc=True, threshold=2.)
                else:
                    _, _, _, sampled = model.sample(
                        test_x_, test_m_, c_=c, trunc=True, threshold=2.)

                # inverse to feature
                vel, art, ioi = \
                    inverse_feature_note(sampled, art=True, numpy=False, interp='tanh')

                # set the same global attributes to GT
                # vel_ = vel + (np.mean(y_vel) - np.mean(vel))
                # ioi_ = ioi + (np.mean(y_ioi) - np.mean(ioi))
                # art_ = art + (np.mean(y_art) - np.mean(art))
                # vel_ = np.clip(vel_, 1, 127)
                # art_ = np.clip(art_, 0.01, 100)
                # ioi_ = np.clip(ioi_, 0.125, 8)             
                # features = [vel_, art_, ioi_]
                features = [vel, art, ioi]

                ### RENDER MIDI ###
                # print("** Render MIDI files** ")
                inverse_rendering_art_note(
                    input_notes=xml_notes, save_dir="./", cond=test_x, features=features, 
                    tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  
                    savename=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.mid".format(
                        song_name, measures[0], measures[1], exp_num)), 
                    save_score=False, return_notes=False, save_perform=True)

                # new_notes = change_tempo_to_target(result_notes, perform_notes)
                # save_new_midi(new_notes, 
                #     new_midi_path=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.mid".format(
                #         song_name, measures[0], measures[1], exp_num)))

            elif model_num == "GT":
                save_new_midi(perform_notes, 
                    new_midi_path=os.path.join(save_path, 
                    'test_sample.{}__mm{}-{}.GT.mid'.format(song_name, measures[0], measures[1])))

            elif model_num == "GT2":

                vel, art, ioi = \
                    inverse_feature_note(test_y_, art=True, numpy=False, interp='tanh')
                features = [vel, art, ioi]

                inverse_rendering_art_note(
                    input_notes=xml_notes, save_dir="./", cond=test_x, features=features, 
                    tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  
                    savename=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.mid".format(
                        song_name, measures[0], measures[1], exp_num)), 
                    save_score=False, return_notes=False, save_perform=True)


            elif model_num == "DP":
                # vel_dp = np.tile(np.mean(y_vel).reshape(1,), (len(y_vel),))
                # art_dp = np.tile(np.mean(y_art).reshape(1,), (len(y_art),))
                # ioi_dp = np.tile(np.mean(y_ioi).reshape(1,), (len(y_ioi),))
                vel_dp = np.ones_like(y_vel) * 64
                art_dp = np.ones_like(y_art)
                ioi_dp = np.ones_like(y_ioi)
                features_dp = [vel_dp, art_dp, ioi_dp]
                result_notes = inverse_rendering_art_note(
                    input_notes=xml_notes, save_dir="./", cond=test_x, features=features_dp, 
                    tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  
                    savename=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.mid".format(
                        song_name, measures[0], measures[1], exp_num)), 
                    save_score=False, return_notes=True, save_perform=False)

                new_notes = change_tempo_to_target(result_notes, perform_notes)
                save_new_midi(new_notes, 
                    new_midi_path=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.mid".format(
                        song_name, measures[0], measures[1], exp_num)))

            print("saved sample for Set {}: {} ({}/{} th song)".format(
                Set+1, song_name, song_num, len(song_list)))
            song_num += 1


def save_all_test_samples_control(
    model_num=None, exp_num=None, epoch_num=None, device_num=None, same_onset_ind=[110,112]):

    # model_num = 'DP'
    # exp_num = 'DP'
    # epoch_num = '100'
    # device_num = 3 
    same_onset_ind = [110,112]
    print()
    print()
    print("############# TEST ##############")
    print("> MODEL: {}\n> EXP: {}\n> EPOCH: {}".format(model_num, exp_num, epoch_num))
    print("#################################")
    print()

    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")

    if "GT" in model_num or model_num == "DP":
        model = None 

    else:
        ## LOAD MODEL ## 
        print("** Load model **")
        checkpoint_num = epoch_num
        module_name = "piano_cvae_model2_torch_{}".format(model_num)
        model = importlib.import_module(module_name)
        note2group = model.Note2Group()
        # Generator = model.PerformGenerator
        model_path = "./model_cvae/piano_cvae_ckpt_{}".format(exp_num) # 
        if "exp" in model_num: # proposed(group)
            Generator = model.PerformGenerator
        elif "nms" in model_num: 
            Generator = model.NMSLatentDisentangledDynamic 
        model = Generator(device=device)
        model.to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict']) 
        model.eval()

    parent_path = "/data/asap_dataset/exp_data/listening_test/raw"
    categs = sorted(glob(os.path.join(parent_path, '*/')))
    song_list = list()
    for categ in categs:
        pieces = sorted(glob(os.path.join(categ, '*/')))
        for piece in pieces:
            song_list.append(piece)
    song_list = sorted(song_list)
    indices = np.load("./ACCESS/access_song_start_point_dict.npy", allow_pickle=True).tolist()
    # max_len = 16

    for Set in indices:
        song_indices = indices[Set]
        save_path = "./ACCESS/test_samples/Set{}/{}_Set{}_control".format(Set+1, exp_num, Set+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        song_num = 1
        for song in song_indices:
            song_path = song_list[song]

            if "Bach" in song_path:
                continue
            # song_name = "Schubert_Impromptu_op90_D899__4"
            # song_name = "Beethoven_Piano_Sonatas__8-2"
            # same_onset_ind = [110,112]
            song_name =  '__'.join(song_path.split('/')[-3:-1])
            perform = [p for p in sorted(glob(
                os.path.join(song_path, "01/*.mid"))) if "cleaned" not in p][0]
            xml = os.path.join(song_path, "musicxml_cleaned_plain.musicxml")
            score = os.path.join(song_path, "score_plain.mid")
            pair = os.path.join(os.path.dirname(perform), 'xml_score_perform_pairs.npy')
            if os.path.exists(pair) is True:
                pair_path = pair 
            else:
                pair_path = None

            ## LOAD DATA ##
            null_tempo = 120
            measures = song_indices[song][0]
            get_data = GetData(null_tempo=null_tempo, 
                            same_onset_ind=same_onset_ind,
                            stat=np.mean)
            test_x, test_y, test_m, pairs, test_notes, xml_notes, perform_notes, tempo, p_name = \
                get_data.file2data(files=[xml, score, perform], 
                measures=measures, mode="note", pair_path=pair_path, save_mid=False)
            tempo_rate = tempo / null_tempo
            # print("     > tempo: {}".format(tempo))
            # print()

            # get input data
            test_inputs_ = get_data.data2input(
                test_x, test_y, test_m, art=True, mode="note", device=device)
            test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs_
            test_inputs = test_inputs_
            test_inputs2 = [test_x_, test_y_, test_m_, test_clab_]
            y_vel = test_y[:,0]
            y_ioi = test_y[:,2]
            y_art = np.power(10, np.log10(test_y[:,1]) - np.log10(test_y[:,3])) 
            features_gt = [y_vel, y_art, y_ioi]

            if "GT" not in model_num and model_num != "DP":
                # forward
                if "_pati" in model_num:
                    c_ = model.infer_c_only(test_x_, test_y_, test_m_)
                elif "_cvrnn" in model_num or "_fader" in model_num:
                    c_ = note2group.reverse(test_clab_, test_m_)
                else:
                    c_ = model.infer_c_only(test_y_, test_m_)
                
                c_max = torch.max(c_, dim=1).values
                c_min = torch.min(c_, dim=1).values
                c_mean = torch.mean(c_, dim=1)
                c = c_mean.unsqueeze(1).repeat(1, c_.size(1), 1)
                cm = c_mean[0].cpu().data.numpy()
                cmax = c_max[0].cpu().data.numpy()
                cmin = c_min[0].cpu().data.numpy()
                mx = cmax 
                mn = cmin
                # mx = cmax * 0.75 + cmin * 0.25
                # mn = cmax * 0.25 + cmin * 0.75
                
                for _ in range(1):           

                    if "note" in model_num or "_cvrnn" in model_num or "_fader" in model_num:
                        T = test_y_.size(1)
                    else:
                        T = test_y2_.size(1)

                    # EP for fader-control
                    if "_cvrnn" in model_num or "_fader" in model_num:
                        d_ep = np.linspace(mx[0], mn[0], num=T, endpoint=True)
                        a_ep = np.linspace(mx[1], mn[1], num=T, endpoint=True)
                        i_ep = np.linspace(mx[2], mn[2], num=T, endpoint=True)
                        d_diff = mx[0] - mn[0]
                        a_diff = mx[1] - mn[1]
                        i_diff = mx[2] - mn[2]
                    
                    else:
                        d_ep = np.linspace(mx[0], mn[0], num=T, endpoint=True)
                        a_ep = np.linspace(mx[4], mn[4], num=T, endpoint=True)
                        i_ep = np.linspace(mx[8], mn[8], num=T, endpoint=True)
                        d_diff = mx[0] - mn[0]
                        a_diff = mx[4] - mn[4]
                        i_diff = mx[8] - mn[8]

                    d_ep = torch.from_numpy(d_ep).to(device).float()
                    a_ep = torch.from_numpy(a_ep).to(device).float()
                    i_ep = torch.from_numpy(i_ep).to(device).float()

                    c_0 = torch.empty_like(c).copy_(c)

                    if "cvrnn" in model_num or "fader" in model_num:             
                        # dynamics
                        c_d = torch.empty_like(c).copy_(c)
                        # c_d[:,:,0] = c_[:,:,0]
                        c_d[:,:,0] = d_ep
                        # articulation
                        c_a = torch.empty_like(c).copy_(c)
                        # c_a[:,:,1] = c_[:,:,1]
                        c_a[:,:,1] = a_ep
                        # tempo
                        c_i = torch.empty_like(c).copy_(c)
                        # c_i[:,:,2] = c_[:,:,2]
                        c_i[:,:,2] = i_ep
                    else:
                        # dynamics
                        c_d = torch.empty_like(c).copy_(c)
                        # c_d[:,:,0] = c_[:,:,0]
                        c_d[:,:,0] = d_ep
                        # articulation
                        c_a = torch.empty_like(c).copy_(c)
                        # c_a[:,:,4] = c_[:,:,4]
                        c_a[:,:,4] = a_ep
                        # tempo
                        c_i = torch.empty_like(c).copy_(c)
                        # c_i[:,:,8] = c_[:,:,8]
                        c_i[:,:,8] = i_ep

                    for each_c, attr in zip(
                        [c_d, c_a, c_i], ["dyn","art","ioi"]):

                        # sample
                        _, _, _, sampled1 = model.sample(
                            test_x_, test_m_, c_=each_c, trunc=True, threshold=2.)
                        # _, _, _, sampled2 = model.sample(
                            # test_x_, test_m_, c_=c_0, trunc=True, threshold=2.)

                        # inverse to feature
                        vel1, art1, ioi1 = \
                            inverse_feature_note(sampled1, art=True, numpy=False, interp='tanh')
                        # vel2, art2, ioi2 = \
                            # inverse_feature_note(sampled2, art=True, numpy=False, interp='tanh')

                        # set the same global attributes to GT
                        # vel_ = vel + (np.mean(y_vel) - np.mean(vel))
                        # ioi_ = ioi + (np.mean(y_ioi) - np.mean(ioi))
                        # art_ = art + (np.mean(y_art) - np.mean(art))
                        # vel_ = np.clip(vel_, 1, 127)
                        # art_ = np.clip(art_, 0.01, 100)
                        # ioi_ = np.clip(ioi_, 0.125, 8)             
                        # features = [vel_, art_, ioi_]
                        features1 = [vel1, art1, ioi1]
                        # features2 = [vel2, art2, ioi2]

                        ### RENDER MIDI ###
                        # print("** Render MIDI files** ")
                        inverse_rendering_art_note(
                            input_notes=xml_notes, save_dir="./", cond=test_x, features=features1, 
                            tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  
                            savename=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.{}.mid".format(
                                song_name, measures[0], measures[1], attr, exp_num)), 
                            save_score=False, return_notes=False, save_perform=True)

                # new_notes = change_tempo_to_target(result_notes, perform_notes)
                # save_new_midi(new_notes, 
                #     new_midi_path=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.mid".format(
                #         song_name, measures[0], measures[1], exp_num)))

            elif model_num == "GT":
                save_new_midi(perform_notes, 
                    new_midi_path=os.path.join(save_path, 
                    'test_sample.{}__mm{}-{}.GT.mid'.format(song_name, measures[0], measures[1])))

            elif model_num == "GT_control":
                y_mean = torch.mean(
                    test_y2_, dim=1).unsqueeze(1).repeat(1, test_y_.size(1), 1)
                # dynamics
                c_d = torch.empty_like(y_mean).copy_(y_mean)
                c_d[:,:,0] = test_y_[:,:,0]
                # articulation
                c_a = torch.empty_like(y_mean).copy_(y_mean)
                c_a[:,:,1] = test_y_[:,:,1]
                # tempo
                c_i = torch.empty_like(y_mean).copy_(y_mean)
                c_i[:,:,2] = test_y_[:,:,2]

                for each_c, attr in zip(
                    [c_d, c_a, c_i], ["dyn","art","ioi"]):

                    vel, art, ioi = \
                        inverse_feature_note(each_c, art=True, numpy=False, interp='tanh')
                    features = [vel, art, ioi]

                    inverse_rendering_art_note(
                        input_notes=xml_notes, save_dir="./", cond=test_x, features=features, 
                        tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  
                        savename=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.{}.mid".format(
                            song_name, measures[0], measures[1], attr, exp_num)), 
                        save_score=False, return_notes=False, save_perform=True)


            elif model_num == "DP":
                # vel_dp = np.tile(np.mean(y_vel).reshape(1,), (len(y_vel),))
                # art_dp = np.tile(np.mean(y_art).reshape(1,), (len(y_art),))
                # ioi_dp = np.tile(np.mean(y_ioi).reshape(1,), (len(y_ioi),))
                vel_dp = np.ones_like(y_vel) * 64
                art_dp = np.ones_like(y_art)
                ioi_dp = np.ones_like(y_ioi)
                features_dp = [vel_dp, art_dp, ioi_dp]
                result_notes = inverse_rendering_art_note(
                    input_notes=xml_notes, save_dir="./", cond=test_x, features=features_dp, 
                    tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  
                    savename=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.mid".format(
                        song_name, measures[0], measures[1], exp_num)), 
                    save_score=False, return_notes=True, save_perform=False)

                new_notes = change_tempo_to_target(result_notes, perform_notes)
                save_new_midi(new_notes, 
                    new_midi_path=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.mid".format(
                        song_name, measures[0], measures[1], exp_num)))

            print("saved sample for Set {}: {} ({}/{} th song)".format(
                Set+1, song_name, song_num, len(song_list)))
            song_num += 1


def save_all_test_samples2(
    model_num=None, exp_num=None, epoch_num=None, device_num=None, same_onset_ind=[110,112]):

    # model_num = 'exp1915'
    # exp_num = 'exp1915'
    # epoch_num = '100'
    # device_num = 3 
    same_onset_ind = [110,112]
    print()
    print()
    print("############# TEST ##############")
    print("> MODEL: {}\n> EXP: {}\n> EPOCH: {}".format(model_num, exp_num, epoch_num))
    print("#################################")
    print()

    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")

    if model_num == "GT" or model_num == "REF":
        model = None 

    else:
        ## LOAD MODEL ## 
        print("** Load model **")
        checkpoint_num = epoch_num
        module_name = "piano_cvae_model2_torch_{}".format(model_num)
        model = importlib.import_module(module_name)
        # Generator = model.PerformGenerator
        model_path = "./model_cvae/piano_cvae_ckpt_{}".format(exp_num) # 
        if "exp" in model_num: # proposed(group)
            Generator = model.PerformGenerator
        elif "nms" in model_num: 
            Generator = model.NMSLatentDisentangledDynamic 
        model = Generator(device=device)
        model.to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict']) 
        model.eval()

    parent_path = "/data/asap_dataset/exp_data/listening_test/raw"
    categs = sorted(glob(os.path.join(parent_path, '*/')))
    song_list = list()
    for categ in categs:
        pieces = sorted(glob(os.path.join(categ, '*/')))
        for piece in pieces:
            song_list.append(piece)
    song_list = sorted(song_list)
    indices = np.load("./ACCESS/access_song_start_point_dict.npy", allow_pickle=True).tolist()
    d_list = ["loud", "quiet"]
    a_list = ["stac", "legato"]
    i_list = ["fast", "slow"]

    for Set in indices:
        song_indices = indices[Set]
        save_path = "./ACCESS/test_samples_infer/Set{}/{}_Set{}".format(Set+1, exp_num, Set+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        song_num = 1
        for song in song_indices:
            song_path = song_list[song]

            if "Bach" in song_path:
                continue
            # song_name = "Schubert_Impromptu_op90_D899__4"
            # song_name = "Beethoven_Piano_Sonatas__8-2"
            # same_onset_ind = [110,112]
            song_name =  '__'.join(song_path.split('/')[-3:-1])
            perform = [p for p in sorted(glob(
                os.path.join(song_path, "01/*.mid"))) if "cleaned" not in p][0]
            xml = os.path.join(song_path, "musicxml_cleaned_plain.musicxml")
            score = os.path.join(song_path, "score_plain.mid")
            pair = os.path.join(os.path.dirname(perform), 'xml_score_perform_pairs.npy')
            if os.path.exists(pair) is True:
                pair_path = pair 
            else:
                pair_path = None

            ## LOAD DATA ##
            null_tempo = 120
            measures = song_indices[song][0]
            get_data = GetData(null_tempo=null_tempo, 
                            same_onset_ind=same_onset_ind,
                            stat=np.mean)
            test_x, test_y, test_m, pairs, test_notes, xml_notes, perform_notes, tempo, p_name = \
                get_data.file2data(files=[xml, score, perform], 
                measures=measures, mode="note", pair_path=pair_path, save_mid=False)
            tempo_rate = tempo / null_tempo
            # print("     > tempo: {}".format(tempo))
            # print()

            # get input data
            test_inputs_ = get_data.data2input(
                test_x, test_y, test_m, art=True, mode="note", device=device)


            # condition
            cond_num = (Set*23 + (song-2)) % 8
            ii = cond_num // 4 
            aa = (cond_num % 4) // 2 
            dd = cond_num % 2
            cond = [d_list[dd], a_list[aa], i_list[ii]]

            # reference data
            ref_inputs_ = get_data.data2input(
                test_x, test_y, test_m, cond=cond, art=True, mode="note", device=device)

            test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs_
            ref_x_, ref_y_, ref_y2_, ref_m_, ref_clab_ = ref_inputs_
            test_inputs = test_inputs_
            test_inputs2 = [test_x_, test_y_, test_m_, test_clab_]
            y_vel = test_y[:,0]
            y_ioi = test_y[:,2]
            y_art = np.power(10, np.log10(test_y[:,1]) - np.log10(test_y[:,3])) 

            if model_num != "GT" and model_num != "REF":
                # forward
                if "_pati" in model_num:
                    c = model.sample_c_only(ref_x_, ref_y_, ref_m_)
                elif "_cvae" in model_num or "_fader" in model_num:
                    c = ref_clab_
                else:
                    c = model.sample_c_only(ref_y_, ref_m_)

                # sample
                if "_pati" in model_num:
                    c_new = torch.randn_like(c)
                    c_new[:,:,0] = c[:,:,0]
                    c_new[:,:,4] = c[:,:,4]
                    c_new[:,:,8] = c[:,:,8]
                    _, sampled = model.sample(
                        test_x_, test_m_, c_=c_new, trunc=True, threshold=2.)    
                else:
                    _, _, _, sampled = model.sample(
                        test_x_, test_m_, c_=c, trunc=True, threshold=2.)

                # inverse to feature
                vel, art, ioi = \
                    inverse_feature_note(sampled, art=True, numpy=False, interp='tanh')

                # set the same global attributes to GT
                # vel_ = vel + (np.mean(y_vel) - np.mean(vel))
                # ioi_ = ioi + (np.mean(y_ioi) - np.mean(ioi))
                # art_ = art + (np.mean(y_art) - np.mean(art))
                # vel_ = np.clip(vel_, 1, 127)
                # art_ = np.clip(art_, 0.01, 100)
                # ioi_ = np.clip(ioi_, 0.125, 8)             
                # features = [vel_, art_, ioi_]
                features = [vel, art, ioi]

                ### RENDER MIDI ###
                # print("** Render MIDI files** ")
                inverse_rendering_art_note(
                    input_notes=xml_notes, save_dir="./", cond=test_x, features=features, 
                    tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  
                    savename=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.mid".format(
                        song_name, measures[0], measures[1], exp_num)), save_score=False)

            elif model_num == "GT":
                save_new_midi(perform_notes, 
                    new_midi_path=os.path.join(save_path, 
                    'test_sample.{}__mm{}-{}.GT.mid'.format(song_name, measures[0], measures[1])))

            elif model_num == "REF":
                # inverse to feature
                vel, art, ioi = \
                    inverse_feature_note(ref_y_, art=True, numpy=False, interp='tanh')
                features = [vel, art, ioi]
                inverse_rendering_art_note(
                    input_notes=xml_notes, save_dir="./", cond=test_x, features=features, 
                    tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  
                    savename=os.path.join(save_path, "test_sample.{}__mm{}-{}.REF.mid".format(
                        song_name, measures[0], measures[1])), save_score=False)

            print("saved sample for Set {}: {} ({}/{} th song)".format(
                Set+1, song_name, song_num, len(song_list)))
            song_num += 1


def audio_with_timidity():
    midi_paths = sorted(glob("./rendered_test/*.mid"))

    for midi_path in midi_paths:
        filename = '.'.join(os.path.basename(midi_path).split(".")[:-1])
        savepath = os.path.join(os.path.dirname(midi_path), filename + ".wav")
        os.chdir("./")
        subprocess.call(["timidity", midi_path, "-o",  savepath, "-Ow"])



if __name__ == "__main__":
    model_num = sys.argv[1]
    exp_num = sys.argv[2]
    epoch_num = sys.argv[3]    
    mode = sys.argv[4]
    device_num = sys.argv[5]

    if mode == "test":
        # song_type = sys.argv[6]
        song_name = sys.argv[6]
        measure_start = int(sys.argv[7])
        measure_end = int(sys.argv[8])

        if 'exp' not in model_num:
            test_model_base(
                model_num=model_num, exp_num=exp_num, epoch_num=epoch_num, 
                song_name=song_name, measures=[measure_start, measure_end])
        else:
            test_model(
                model_num=model_num, exp_num=exp_num, epoch_num=epoch_num, device_num=device_num,
                song_name=song_name, measures=[measure_start, measure_end])


    elif mode == "qualitative":
        # song_type = sys.argv[6]
        song_name = sys.argv[6]
        measure_start = int(sys.argv[7])
        measure_end = int(sys.argv[8])

        qualitative_results(
            model_num=model_num, exp_num=exp_num, epoch_num=epoch_num, device_num=device_num,
            song_name=song_name, measures=[measure_start, measure_end])


    elif mode == "qualitative_fader":
        # song_type = sys.argv[6]
        song_name = sys.argv[6]
        measure_start = int(sys.argv[7])
        measure_end = int(sys.argv[8])

        qualitative_results_fader(
            model_num=model_num, exp_num=exp_num, epoch_num=epoch_num, device_num=device_num,
            song_name=song_name, measures=[measure_start, measure_end])


    elif mode == "qualitative_EP":
        # song_type = sys.argv[6]
        song_name = sys.argv[6]
        measure_start = int(sys.argv[7])
        measure_end = int(sys.argv[8])

        qualitative_results_EP(
            model_num=model_num, exp_num=exp_num, epoch_num=epoch_num, device_num=device_num,
            song_name=song_name, measures=[measure_start, measure_end])


    elif mode == "all":
        save_all_test_samples(
            model_num=model_num, exp_num=exp_num, epoch_num=epoch_num, device_num=device_num)        


    elif mode == "all_control":
        save_all_test_samples_pianotab_control(
            model_num=model_num, exp_num=exp_num, epoch_num=epoch_num, device_num=device_num)        


    elif mode == "all2":
        save_all_test_samples2(
            model_num=model_num, exp_num=exp_num, epoch_num=epoch_num, device_num=device_num) 


    elif mode == "test_transfer":
        # song_type = sys.argv[6]
        song_name1 = sys.argv[6]
        measure_start1 = int(sys.argv[7])
        measure_end1 = int(sys.argv[8])
        song_name2 = sys.argv[9]
        measure_start2 = int(sys.argv[10])
        measure_end2 = int(sys.argv[11])

        test_model_transfer(
            model_num=model_num, exp_num=exp_num, epoch_num=epoch_num, 
            song_name=[song_name1, song_name2], measures=[[measure_start1, measure_end1], [measure_start2, measure_end2]])
