import numpy as np
import os
import sys 
sys.path.append('./parse_utils')
from glob import glob
import pretty_midi 
import h5py
import pandas as pd 
from decimal import Decimal
import collections

import urllib.request
import requests 
from bs4 import BeautifulSoup as bs

import subprocess

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, mode
from scipy.stats import ttest_ind, ttest_rel # t test
from scipy.stats import bartlett, fligner, levene # equal variance test 
import pandas
# from statsmodels.formula.api import ols
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# import soundfile as sf

# import musicxml_parser
from parse_features import *
from parse_utils.parse_utils import *
from get_id import get_randomname
# from piano_cvae_main2_torch_test import get_feature

def ind2str(ind, n):
    ind_ = str(ind)
    rest = n - len(ind_)
    str_ind = rest*"0" + ind_
    return str_ind 

def split_sets(data=None):
    '''
    < chopin_maestro >
    * Total 19 pieces / 91 performances
    < chopin cleaned >
    * Total 21 pieces / 224 performances
    * Total 34 pieces / 356 performances
    '''
    datapath = '/data/{}/original'.format(data)
    val_songs = [
        "Chopin_Etude/25_2/", # 224
        "Chopin_Etude/25_9/"] # 112
    test_songs = [
        "Chopin_Etude/10_7/", # 126
        "Chopin_Etude/25_3/"] # 120
    train_path = '/data/{}/exp_data/train/raw'.format(data)
    test_path = '/data/{}/exp_data/test/raw'.format(data)
    val_path = '/data/{}/exp_data/val/raw'.format(data)
    num_songs = 0
    num_performs = 0

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    categs = sorted(glob(os.path.join(datapath, '*/')))
    for c in categs:
        pieces = sorted(glob(os.path.join(c, '*/')))
        c_name = c.split('/')[-2]
        for p in pieces:
            players = sorted(glob(os.path.join(p, "*[!pro]/")))
            # if not os.path.exists(os.path.join(p, "cond.npy")):
                # continue
            p_name = p.split('/')[-2]
            num_songs += 1

            if p in [os.path.join(datapath, t) for t in test_songs]:
                savepath = test_path
            elif p in [os.path.join(datapath, v) for v in val_songs]:
                savepath = val_path
            else: savepath = train_path

            savepath_ = os.path.join(savepath, c_name, p_name)
            if not os.path.exists(savepath_):
                os.makedirs(savepath_)

            cond = np.load(os.path.join(p, "cond.npy"), allow_pickle=True)
            # np.save(os.path.join(savepath_, "cond.npy"), cond) 

            for pl in players:
                pl_name = pl.split('/')[-2]
                num_performs += 1
                inp = np.load(os.path.join(pl, "inp.npy"), allow_pickle=True)
                oup = np.load(os.path.join(pl, "oup.npy"), allow_pickle=True)
                oup_v = np.unique([i[1][0] for i in oup])
                if len(oup_v) < 5:
                    print(oup_v)
                # np.save(os.path.join(savepath_, "inp.{}.npy".format(pl_name)), inp)
                # np.save(os.path.join(savepath_, "oup.{}.npy".format(pl_name)), oup)

            print("saved data npy for {}/{}".format(c_name, p_name))

def split_listening_sets():

    '''
    116 performances, 23 pieces, 10 composers
    '''

    datapath = '/data/asap_dataset/original'
    savepath = '/data/asap_dataset/exp_data/listening_test/raw/'
    test_list = [
        'Balakirev_Islamey.0',
        'Beethoven_Piano_Sonatas.14-3',
        'Beethoven_Piano_Sonatas.22-2',
        'Beethoven_Piano_Sonatas.31-2',
        'Beethoven_Piano_Sonatas.8-2',
        'Debussy_Pour_le_Piano.1',
        'Glinka_The_Lark.0',
        'Haydn_Keyboard_Sonatas.31-1',
        'Haydn_Keyboard_Sonatas.39-3',
        'Haydn_Keyboard_Sonatas.49-1',
        'Liszt_Gran_Etudes_de_Paganini.2_La_campanella',
        'Liszt_Transcendental_Etudes.10',
        'Liszt_Transcendental_Etudes.3',
        'Mozart_Piano_Sonatas.11-3',
        'Mozart_Piano_Sonatas.12-1',
        'Mozart_Piano_Sonatas.8-1',
        'Ravel_Miroirs.4_Alborada_del_gracioso',
        'Ravel_Pavane.0',
        'Schubert_Impromptu_op.90_D.899.4',
        'Schubert_Impromptu_op142.3',
        'Schubert_Piano_Sonatas.664-3',
        'Schubert_Wanderer_fantasie.0',
        'Schumann_Kreisleriana.5']
    
    # 'Bach_Fugue.bwv_884',
    # 'Bach_Prelude.bwv_888',

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    num = 0
    categs = sorted(glob(os.path.join(datapath, '*/')))
    for c in categs:
        pieces = sorted(glob(os.path.join(c, '*/')))
        c_name = c.split('/')[-2]
        for p in pieces:
            players = sorted(glob(os.path.join(p, "*/")))
            p_name = p.split('/')[-2]

            full_name = "{}.{}".format(c_name, p_name)
            if full_name not in test_list:
                continue 

            xml = os.path.join(p, 'musicxml_cleaned_plain.musicxml')
            score = os.path.join(p, 'score_plain.mid')
            assert os.path.exists(xml)
            assert os.path.exists(score)

            savepath_ = os.path.join(savepath, c_name, p_name)
            if not os.path.exists(savepath_):
                os.makedirs(savepath_)

            # shutil.copy(xml, os.path.join(savepath_, 'musicxml_cleaned_plain.musicxml'))
            # shutil.copy(score, os.path.join(savepath_, 'score_plain.mid'))

            # os.chmod(os.path.join(savepath_, 'musicxml_cleaned_plain.musicxml'), 0o777)
            # os.chmod(os.path.join(savepath_, 'score_plain.mid'), 0o777)

            for pl in players:
                pl_name = pl.split('/')[-2]
                num += 1

                # shutil.copytree(pl, os.path.join(savepath_, pl_name))

            print("saved data npy for {}/{}".format(c_name, p_name))


def make_onset_based_pick(x_data, out, same_onset_ind=None):
    '''
    get only the lowest note for each onset
    '''
    start, end = same_onset_ind
    same_onset = np.argmax(x_data[:,start:end], axis=-1)
    new_out = list()
    for i in range(x_data.shape[0]):
        o = same_onset[i] 
        if o == 0:
            new_out.append(out[i])
        elif o == 1:
            continue
    return np.asarray(new_out)


def make_onset_based_all(x_data, out, same_onset_ind=None):
    '''
    get all notes in each onset
    '''
    start, end = same_onset_ind
    same_onset = np.argmax(x_data[:,start:end], axis=-1)
    new_out = list()
    is_onset = [out[0]]
    for i in range(1, x_data.shape[0]):
        o = same_onset[i] 
        if o == 0:
            new_out.append(is_onset)
            is_onset = [out[i]]
        elif o == 1:
            is_onset.append(out[i])
    new_out.append(is_onset)
    return new_out


def make_beat_based_all(out, beat=None):
    '''
    get all notes in each beat
    '''
    beat = np.argmax(beat, axis=-1)
    new_out = list()
    prev_b = None 
    is_beat = [out[0]]
    for i in range(1, out.shape[0]):
        b = beat[i] 
        if b != prev_b: # new beat
            new_out.append(is_beat)
            is_beat = [out[i]]
        elif b == prev_b: # same beat
            is_beat.append(out[i])
        prev_b = b
    new_out.append(is_beat)
    return new_out


def make_onset_based_all_index(x_data, out, same_onset_ind=None):
    '''
    get all notes in each onset
    '''
    start, end = same_onset_ind
    same_onset = np.argmax(x_data[:,start:end], axis=-1)
    new_out = list()
    is_onset = [[0, out[0]]]
    for i in range(1, x_data.shape[0]):
        o = same_onset[i] 
        if o == 0:
            new_out.append(is_onset)
            is_onset = [[i, out[i]]]
        elif o == 1:
            is_onset.append([i, out[i]])
    new_out.append(is_onset)
    return new_out


def make_onset_based_top(x_data, out, same_onset_ind=None):
    '''
    get top note in each onset
    '''
    out = make_onset_based_all_index(x_data, out, same_onset_ind)
    new_out = [o[-1] for o in out]
    new_onsets = np.asarray([o[1] for o in new_out])
    new_inds = np.asarray([int(o[0]) for o in new_out])
    return new_onsets, new_inds


def make_note_based(x_data, out, same_onset_ind=None):
    start, end = same_onset_ind
    same_onset = np.argmax(x_data[:,start:end], axis=-1)
    new_out = list()
    j = -1
    for i in range(x_data.shape[0]):
        o = same_onset[i] 
        if o == 0:
            j += 1
            new_out.append(out[j])
        elif o == 1:
            new_out.append(out[j])
    return np.asarray(new_out)


def make_align_matrix(x_data, roll, same_onset_ind=None, rev=False):
    align_mat = np.zeros([x_data.shape[0], roll.shape[0]])
    start, end = same_onset_ind
    same_onset = np.argmax(x_data[:,start:end], axis=-1)
    ind = -1
    if rev == False:
        for i in range(x_data.shape[0]):
            o = same_onset[i]
            if o == 0: # if not same onset:
                ind += 1
                align_mat[i, ind] = 1
            elif o == 1:
                align_mat[i, ind] = 1
        # print(ind)
    elif rev == True:
        prev_o = 0
        for i in reversed(range(x_data.shape[0])):
            o = same_onset[i]
            if o == 1:
                if prev_o == 0: # diff onset
                    ind += 1
                    align_mat[i, ind] = 1
                    # print("o: 1 / prev_o: 0 --> {}th".format(i))
                elif prev_o == 1: # same onset
                    align_mat[i, ind] = 1
            elif o == 0:
                if prev_o == 0: # diff onset
                    ind += 1
                    align_mat[i, ind] = 1
                    # print("o: 0 / prev_o: 0 --> {}th".format(i))
                elif prev_o == 1: # same onset
                    align_mat[i, ind] = 1
            prev_o = o    
        align_mat = np.flip(align_mat, 0)    
    return align_mat.astype(np.float32)


def make_notenum_onehot(m_data):
    '''
    shape = [note-num, onset-num]
    '''
    note_num_seq = np.sum(m_data, axis=0)
    seq_expand = np.matmul(note_num_seq, m_data.T)
    # make onehot
    notenum_onehot = np.zeros([m_data.shape[0], 11])
    for i, s in enumerate(seq_expand):
        assert s > 0
        notenum_onehot[i,int(s-1)] = 1

    return notenum_onehot


def get_vertical_position(x_data, same_onset_ind=None):
    v_pos = np.zeros([len(x_data), 11])
    start, end = same_onset_ind
    for i, d in enumerate(x_data):
        same_onset = np.argmax(d[start:end])
        if same_onset == 0:
            pos = 0
            v_pos[i, pos] = 1
        elif same_onset == 1:
            pos += 1 
            v_pos[i, pos] = 1
    return v_pos 


def make_pianoroll_x(x_data, same_onset_ind=None):
    roll = np.zeros([88,len(x_data)])
    start, end = same_onset_ind
    col = -1
    for i, d in enumerate(x_data):
        pitch = np.argmax(d[start-88:start])
        same_onset = np.argmax(d[start:end])
        if same_onset == 0: # start
            col += 1
            roll[pitch,col] = 1
            # print(i)
        elif same_onset == 1: # cont
            roll[pitch,col] = 1
    # erase empty rolls
    for j in reversed(range(len(x_data))):
        if np.sum(roll[:,j]) > 0:
            break
    roll = np.transpose(roll[:,:j+1])
    return roll


def note_to_onset_ind(inp, same_onset_ind):
    onset_ind = list()
    start, end = same_onset_ind
    for n, each_note in enumerate(inp):
        if np.argmax(each_note[start:end]) == 0: # same onset feature
            onset_ind.append(n)   
    onset_ind.append(n+1)
    assert onset_ind[-1] == len(inp)
    return onset_ind


def save_batches(data=None, same_onset_ind=[110,112]):
    print("Saving batches...")

    orig_parent_path = '/data/{}/original'.format(data)
    parent_path = '/data/{}/exp_data/'.format(data)
    groups = sorted(glob(os.path.join(parent_path, "*/")))
    maxlen, hop = 16, 4
    if data == "chopin_cleaned":
        ex_dir = "01"

    all_out = list()
    all_in = list()
    all_batches = list()
    all_num = 0
    for group in groups: # train/val/test
        datapath = os.path.join(group, 'raw')
        savepath = os.path.join(group, 'batch_onset_16')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        categs = sorted(glob(os.path.join(datapath, '*/')))
        for categ in categs:
            c_name = categ.split('/')[-2]
            # if c_name != "Chopin_Etude":
            #     continue
            pieces = sorted(glob(os.path.join(categ, '*/')))
            for piece in pieces:
                p_name = piece.split('/')[-2]
                cond_ = np.load(os.path.join(piece, 'cond.npy'), allow_pickle=True) # xml 
                cond_ = sorted(cond_, key=lambda x: x[0]) # same order with score
                tempo = cond_[0][1][0] # BPM
                inps = sorted(glob(os.path.join(piece, 'inp.*.npy'))) # score
                oups = sorted(glob(os.path.join(piece, 'oup.*.npy'))) # perform
                if len(inps) < 4:
                    print(c_name, p_name, len(inps), tempo)
                    all_num += len(inps)
                
                # score_path = os.path.join(orig_parent_path, c_name, p_name, 
                #     ex_dir, 'score_plain.cleaned.mid')
                # xml_path = os.path.join(orig_parent_path, c_name, p_name, 
                    # 'musicxml_cleaned_plain.musicxml')
                pair_path = os.path.join(orig_parent_path, c_name, p_name, 
                    ex_dir, 'xml_score_perform_pairs.npy')
                pairs = np.load(pair_path, allow_pickle=True)
                pairs_xml = [p for p in pairs if p['xml_note'] is not None and \
                    p['xml_note'][1].is_grace_note is False]
                # XMLDocument = MusicXMLDocument(xml_path)
                xml_notes = [x['xml_note'] for x in pairs_xml]
                xml_dict = dict()
                for n in xml_notes:
                    xml_dict[n[0]] = n[1]
                score_notes = xml_to_midi_notes([x[1] for x in xml_notes])
                _unit = (60 / tempo) * (1/8) # 1/32
                roll, _, _ = make_pianoroll(score_notes, unit=_unit)
                # score_notes, _ = extract_midi_notes(score_path, clean=True)
                # assert os.path.exists(score_path)

                for inp_path, oup_path in zip(inps, oups):

                    # group = groups[0]
                    # datapath = os.path.join(group, 'raw')
                    # categs = sorted(glob(os.path.join(datapath, '*/')))  
                    # categ = categs[0]
                    # pieces = sorted(glob(os.path.join(categ, '*/')))
                    # piece = pieces[0]                 
                    # cond_ = np.load(os.path.join(piece, 'cond.npy'), allow_pickle=True) # xml 
                    # cond_ = sorted(cond_, key=lambda x: x[0]) # same order with score
                    # inps = sorted(glob(os.path.join(piece, 'inp2.*.npy'))) # score
                    # oups = sorted(glob(os.path.join(piece, 'oup2.*.npy'))) # perform
                    # inp_path, oup_path = inps[0], oups[0]

                    pl_name = os.path.basename(oup_path).split('.')[-2]
                    inp_ = np.load(inp_path, allow_pickle=True)
                    oup_ = np.load(oup_path, allow_pickle=True)
                    inp_ = sorted(inp_, key=lambda x: x[0])
                    oup_ = sorted(oup_, key=lambda x: x[0]) # same order with score midi
                    if not len(inp_) == len(oup_) == len(cond_) == len(pairs_xml):
                        print(c_name, p_name, pl_name)
                        print(len(inp_), len(pairs_xml))
                        raise AssertionError

                    # augmentation
                    '''
                    cond: tempo(1), beat(12), dynamics(6), staff(2), downbeat(2), 
                        time_sig(num)(12), time_sig(denom)(12), key_sig(12)
                    '''
                    cond = np.asarray([c[1] for c in cond_])
                    inp = np.asarray([i[1] for i in inp_])
                    # t_aug = [0.9, 1, 1.1]
                    # d_aug = [0.9, 1, 1.1]
                    # a_aug = [0.9, 1, 1.1] 
                    t_aug = [1]
                    d_aug = [1]

                    print("{}: {}: {}".format(c_name, p_name, pl_name), end="\r")

                    for t in t_aug: # augment tempo
                        for d in d_aug: # augment dynamics
                            # t, d = 1, 1
                            in_batch = inp
                            out_batch = np.asarray([
                                [o[1][0]*d, o[1][1]*t,
                                o[1][2]*t, o[1][3]*t] for o in oup_]) # , o[1][4]*t
                            
                            ## FOR baseline model (Maezawa et al., 2019) ##
                            # quantized tempo/dynamics by every 16 beat
                            # out_beat_raw = make_beat_based_all(out_batch, beat=cond[:,1:13])
                            out_beat, out_beat_num = corrupt_to_beat(out_batch, beat=cond[:,1:13])
                            out_beat_ma = moving_average(out_beat, win_len=17) # 8 beats back/forth
                            # tile averaged attr by note
                            out_beat_tile = list()
                            for v, n in zip(out_beat_ma, out_beat_num):
                                out_beat_tile.append(np.tile(np.reshape(v, (1, -1)), (n, 1)))
                            out_beat_tile = np.concatenate(out_beat_tile, axis=0)
                            assert len(out_batch) == len(out_beat_tile)
                            # quantize attr
                            '''
                            Ref: Maezawa et al., 2020 (Rendering Music Performance~)
                            '''
                            q_tempo = [q // 30 for q in (120 * (1/out_beat_tile[:,2]))] # features based on 120 BPM
                            q_tempo = np.clip(q_tempo, 1, 10)
                            q_dynamics = [q // 10 for q in out_beat_tile[:,0]]
                            q_dynamics = np.clip(q_dynamics, 1, 12)
                            dyn = np.zeros([len(q_tempo), 12])
                            tpo = np.zeros([len(q_tempo), 10])
                            for i in range(len(q_tempo)):
                                dyn[i, int(q_dynamics[i])-1] = 1
                                tpo[i, int(q_tempo[i])-1] = 1
                            # concatenate with input 
                            quan = np.concatenate([dyn, tpo], axis=-1)
                            
                            # in3_num = np.sum(in3_batch, axis=0) 
                            all_out.append(out_batch)
                            # all_in.append(in_batch)

                            onset_ind = note_to_onset_ind(in_batch, same_onset_ind=same_onset_ind)
                            t_ind = ind2str(int(t*100), 3)
                            d_ind = ind2str(int(d*100), 3)

                            # save batch 
                            num = 1
                            # for b in range(0, len(out_batch)-maxlen, hop): # note-based
                            for b in range(0, len(onset_ind)-maxlen, hop): # onset-based
                                
                                b_ind = ind2str(num, 4) 

                                # note-based
                                # minind, maxind = None, None
                                # for k in range(b, b+maxlen+1):
                                #     if k in onset_ind:
                                #         minind = k
                                #         break 
                                # for j in reversed(range(b+maxlen+1)):
                                #     if j in onset_ind:
                                #         maxind = j
                                #         break 

                                # onset_based
                                onset_in_range = onset_ind[b:b+maxlen+1]
                                minind = onset_in_range[0]
                                maxind = onset_in_range[-1]

                                in_ = in_batch[minind:maxind]
                                out_ = out_batch[minind:maxind]
                                con_ = cond[minind:maxind]
                                q_ = quan[minind:maxind]

                                ## FOR baseline model (Maezawa et al., 2019) ##
                                # 7-beat-radius pianoroll
                                in_ind = [i[0] for i in inp_[minind:maxind]]
                                roll_list = list()
                                rad = 7 # 7 beats in 1/32th note resolution
                                beats = rad * 8
                                '''
                                1 beat in 32 frames --> 8 frames (quarter)
                                3 beats backwards + 4 beats forwards 
                                --> 56 * 88 
                                '''
                                for ind in in_ind: 
                                    base = xml_dict[ind].note_duration.time_position
                                    base_ind = quantize_to_frame(base, unit=_unit)

                                    min_ind = np.max([0, base_ind - (beats//2)])
                                    max_ind = np.min([roll.shape[1], base_ind + (beats//2)])
                                    # print(base, min_ind, max_ind)
                                    
                                    rad_roll = np.zeros([88,56])
                                    sub_roll = roll[:,min_ind:max_ind]
                                    rad_roll[:,rad_roll.shape[1]-sub_roll.shape[1]:] = sub_roll
                                    roll_list.append(rad_roll.T)
                                roll_list = np.asarray(roll_list)
                                
                                # for t in range(7):
                                # tp = 0
                                # in_label = np.argmax(in_[:,22:110], axis=-1)
                                # in_label += tp
                                # if len(in_label[in_label<0]) > 0:
                                #     continue
                                # new_pitch = np.zeros([in_.shape[0], 88])
                                # for p, pitch in enumerate(in_label):
                                #     new_pitch[p, int(pitch)] = 1
                                # in_[:,22:110] = new_pitch

                                in2_ = make_pianoroll_x(in_, same_onset_ind=same_onset_ind)

                                if len(in2_) < 4:
                                    continue
                                
                                in3_ = make_align_matrix(in_, in2_, same_onset_ind=same_onset_ind)
                                in3_rev = make_align_matrix(in_, in2_, rev=True, same_onset_ind=same_onset_ind)
                                in4_ = make_notenum_onehot(in3_)
                                in5_ = get_vertical_position(in_, same_onset_ind=same_onset_ind)
                                # out2_1 = make_onset_based_pick(in_, out_[:,3], same_onset_ind=same_onset_ind) # ioi1
                                # out2_2 = make_onset_based_pick(in_, out_[:,4], same_onset_ind=same_onset_ind) # ioi2
                                # assert in3_.shape[1] == len(out2_1) == len(out2_2)
                                assert in_.shape[0] == in3_.shape[0] == in4_.shape[0]
                                assert in2_.shape[0] == in3_.shape[1] == in3_rev.shape[1]

                                in_ = np.concatenate([in_, in4_, in5_], axis=-1)

                                all_batches.append(np.max(np.argmax(in4_, axis=-1)))

                                # # make into onehot
                                # vel_hot = np.zeros([len(out_), 128])
                                # loc_hot = np.zeros([len(out_), 280])
                                # dur_hot = np.zeros([len(out_), 227])
                                # ioi_hot = np.zeros([len(out2_), 196])

                                # for i in range(len(out_)):
                                #     vel = int(out_[i][0]) - 1
                                #     loc = int(out_[i][1] // 0.01) + 116
                                #     dur = int(np.log10(out_[i][2]) // 0.02) + 131
                                #     vel_hot[i, vel] = 1
                                #     loc_hot[i, loc] = 1
                                #     dur_hot[i, dur] = 1

                                # for j in range(len(out2_)):
                                #     ioi = int(np.log10(out2_[j]) // 0.02) + 117
                                #     ioi_hot[j, ioi] = 1

                                # out_new = np.concatenate([vel_hot, loc_hot, dur_hot], axis=-1)
                                # out2_new = ioi_hot

                                # save batch
                                savename_x = os.path.join(savepath, '{}.{}.{}.batch_x.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))
                                savename_m = os.path.join(savepath, '{}.{}.{}.batch_m.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))
                                savename_m2 = os.path.join(savepath, '{}.{}.{}.batch_m2.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))
                                savename_y = os.path.join(savepath, '{}.{}.{}.batch_y.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind)) 
                                savename_y2 = os.path.join(savepath, '{}.{}.{}.batch_y2.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))  
                                savename_q = os.path.join(savepath, '{}.{}.{}.batch_q.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))  
                                savename_c = os.path.join(savepath, '{}.{}.{}.batch_cond.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))
                                savename_r = os.path.join(savepath, '{}.{}.{}.batch_r.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))   

                                assert len(in_) == len(out_) == len(con_)
                                assert in3_.shape == in3_rev.shape

                                np.save(savename_x, in_)
                                np.save(savename_m, in3_)
                                # np.save(savename_m2, in3_rev)
                                np.save(savename_y, out_)
                                # np.save(savename_y2, out2_1)
                                # np.save(savename_y3, out2_2)
                                np.save(savename_c, con_) 
                                np.save(savename_q, q_) 
                                np.save(savename_r, roll_list)

                                # np.save(savename_y, out_new)
                                # np.save(savename_y2, out2_new) 

                                print("saved batches for {} {}: player {} (aug: {}/{}) --> inp{} / oup{} / cond{}.     ".format(
                                    c_name, p_name, pl_name, t, d, in_.shape, out_.shape, con_.shape), end='\r') 
                                num += 1
                        print("saved batches for {} {}: player {}".format(c_name, p_name, pl_name))


def corrupt_to_onset(x, y, stat=np.mean, same_onset_ind=None):
    y_onset = make_onset_based_all(x, y, same_onset_ind=same_onset_ind)
    y_onset_ = list()
    for each_onset in y_onset:
        each = np.asarray(each_onset)
        _features = stat(each, axis=0)
        y_onset_.append(_features)
    return np.asarray(y_onset_)


def corrupt_to_beat(y, stat=np.mean, beat=None):
    y_beat = make_beat_based_all(y, beat=beat)
    y_beat_ = list()
    y_beat_num = list()
    prev_beat = 0
    for each_onset in y_beat:
        each = np.asarray(each_onset)
        _features = stat(each, axis=0)
        y_beat_.append(_features)
        y_beat_num.append(len(each))
    return np.asarray(y_beat_), np.asarray(y_beat_num)


def create_h5_dataset(data=None, dataset=None): # save npy files into one hdf5 dataset
    # batch_path = "/home/seungyeon/DATA/seungyeon_files/Piano/chopin_cleaned/exp_data/{}/batch".format(dataset)
    batch_path = "/data/{}/exp_data/{}/batch_onset_16".format(data, dataset)
    # load filenames
    x_path = [np.string_(x) for x in sorted(glob(os.path.join(batch_path, "*.batch_x.*.t100_d100.npy")))]
    m_path = [np.string_(m) for m in sorted(glob(os.path.join(batch_path, "*.batch_m.*.t100_d100.npy")))]
    y_path = [np.string_(y) for y in sorted(glob(os.path.join(batch_path, "*.batch_y.*.t100_d100.npy")))]
    # y2_path = [np.string_(y) for y in sorted(glob(os.path.join(batch_path, "*.batch_y2.*.t100_d100.npy")))]
    # y3_path = [np.string_(y) for y in sorted(glob(os.path.join(batch_path, "*.batch_y3.*.t100_d100.npy")))]
    c_path = [np.string_(c) for c in sorted(glob(os.path.join(batch_path, "*.batch_cond.*.t100_d100.npy")))]
    r_path = [np.string_(c) for c in sorted(glob(os.path.join(batch_path, "*.batch_r.*.t100_d100.npy")))]
    q_path = [np.string_(c) for c in sorted(glob(os.path.join(batch_path, "*.batch_q.*.t100_d100.npy")))]
    # save h5py dataset
    f = h5py.File("{}_{}_onset_16.h5".format(data, dataset), "w")
    dt = h5py.special_dtype(vlen=str) # save data as string type
    f.create_dataset("x", data=x_path, dtype=dt)
    f.create_dataset("m", data=m_path, dtype=dt)
    f.create_dataset("y", data=y_path, dtype=dt)
    # f.create_dataset("y2", data=y2_path, dtype=dt)
    # f.create_dataset("y3", data=y3_path, dtype=dt)
    f.create_dataset("c", data=c_path, dtype=dt)
    f.create_dataset("r", data=r_path, dtype=dt)
    f.create_dataset("q", data=q_path, dtype=dt)
    f.close()






if __name__ == "__main__":
    split_sets()
    save_batches()


