import numpy as np
import os
import time
from glob import glob
import pretty_midi 
from decimal import Decimal
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F 

from data.make_batches import (
    make_align_matrix, 
    corrupt_to_onset, 
    make_notenum_onehot,
    make_pianoroll_x, 
    get_vertical_position
)
from sketching_piano_expression.utils.parse_utils import *
from data.parse_features import (
    parse_test_cond, 
    parse_test_features, 
    parse_test_features_noY
)
import model.piano_model as model


def rendering_from_notes(
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

def get_feature(
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

def features_by_condition(
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
    y_new = get_feature(y_, x=x, art=art, same_onset_ind=same_onset_ind)

    return y_new


class GetData(object):

    def __init__(self, null_tempo=120, same_onset_ind=None, stat=np.mean):
        self.soi = same_onset_ind
        self.null_tempo = null_tempo
        self.stat = stat

    def file2data(self, files, measures, pair_path=None, save_mid=False):

        xml, score, perform = files
        p_name = '__'.join(score.split("/")[-3:-1])

        # parse data
        if measures is None:
            first_measure = 0 
        else:
            first_measure = measures[0]
        tempo, time_sig, key_sig = get_signatures_from_xml(xml, first_measure)
        test_y, test_x, pairs, note_ind = \
            parse_test_features(
                xml=xml, score=score, perform=perform,
                measures=measures, tempo=tempo, pair_path=pair_path,
                null_tempo=self.null_tempo, same_onset_ind=self.soi)
        cond = parse_test_cond(pair=pairs, small_ver=False, 
            tempo=tempo, time_sig=time_sig, key_sig=key_sig) 
        xml_notes_raw = [p['xml_note'][1] for p in pairs]
        xml_notes = xml_to_midi_notes(xml_notes_raw)
    
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
            save_new_midi(perform_notes, 
                new_midi_path='orig_{}_mm{}-{}_perform.mid'.format(p_name, measures[0], measures[1]))

        return test_x, test_y, test_m, pairs, xml_notes, perform_notes, tempo, p_name

    def file2data_noY(self, files, measures, save_mid=False):

        xml, score = files
        p_name = '__'.join(score.split("/")[-3:-1])

        # parse data
        if measures is None:
            first_measure = 0 
        else:
            first_measure = measures[0]
        tempo, time_sig, key_sig = get_signatures_from_xml(xml, first_measure)
        test_x, pairs, note_ind = \
            parse_test_features_noY(
                xml=xml, score=score,
                measures=measures, tempo=tempo,
                null_tempo=self.null_tempo, same_onset_ind=self.soi)
        cond = parse_test_cond(pair=pairs, small_ver=False, 
            tempo=tempo, time_sig=time_sig, key_sig=key_sig) 
        xml_notes_raw = [p['xml_note'][1] for p in pairs]
        xml_notes = xml_to_midi_notes(xml_notes_raw)
    
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


    def data2input(self, test_x, test_y, test_m, cond=None, ratio=0.3, N=4, art=True, device=None):

        y = features_by_condition(
            test_y, x=test_x, cond=cond, art=art, ratio=ratio, same_onset_ind=self.soi)  
        vel, art, ioi = y[:,0], y[:,1], y[:,2]
        y2 = corrupt_to_onset(
            test_x, np.stack([vel, art, ioi], axis=-1), stat=self.stat, same_onset_ind=self.soi)

        if N > 1:
            clab = poly_predict(y2, N=N)
            clab_ = poly_predict(y, N=N)
        elif N == 1:
            clab, _, _ = linear_predict(y2)
            clab_, _, _ = linear_predict(y)

        # c label
        clab_m = np.mean(clab, axis=0)
        clab_m = np.where(clab_m > 0, np.ones_like(clab_m), np.zeros_like(clab_m))
        v, a, i = clab_m[0], clab_m[1], clab_m[2]
        label = v * 1 + a * 2 + i * 4
        assert label >= 0 and label < 8
        label = np.array(label)

        device_ = device if device is not None else "cpu"
        # convert to tensor
        test_x_ = torch.from_numpy(test_x.astype(np.float32)).to(device_).unsqueeze(0)
        test_m_ = torch.from_numpy(test_m.astype(np.float32)).to(device_).unsqueeze(0)
        test_y_ = torch.from_numpy(y.astype(np.float32)).to(device_).unsqueeze(0)
        test_y2_ = torch.from_numpy(y2.astype(np.float32)).to(device_).unsqueeze(0)
        test_clab_ = torch.from_numpy(clab.astype(np.float32)).to(device_).unsqueeze(0)
        
        return test_x_, test_y_, test_y2_, test_m_, test_clab_

    def data2input_noY(self, test_x, test_m, device=None):

        device_ = device if device is not None else "cpu"
        # convert to tensor
        test_x_ = torch.from_numpy(test_x.astype(np.float32)).to(device_).unsqueeze(0)
        test_m_ = torch.from_numpy(test_m.astype(np.float32)).to(device_).unsqueeze(0)

        return test_x_, test_m_

    def file2yfeature(self, files, measures, pair_path=None):

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
            parse_test_features(
                xml=xml, score=score, perform=perform,
                measures=measures, tempo=tempo, pair_path=pair_path,
                null_tempo=self.null_tempo, same_onset_ind=self.soi)

        y_vel = test_y[:,0]
        y_ioi = test_y[:,2]
        y_art = np.power(10, np.log10(test_y[:,1]) - np.log10(test_y[:,3]))

        y = np.stack([y_vel, y_art, y_ioi], axis=-1)
        y2 = corrupt_to_onset(
            test_x, y, stat=self.stat, same_onset_ind=self.soi)

        return y.T, y2.T

def test_model(
    song_name=None, measures=None, device_num=None,
    exp_num=None, same_onset_ind=[110,112]):

    song_name_ =  '/'.join(song_name.split('__'))
    parent_path = "./data/raw_samples"
    perform = [p for p in sorted(glob(os.path.join(parent_path, "{}/01/*.mid".format(
        song_name_)))) if "cleaned" not in p][0]
    xml = os.path.join(parent_path, "{}/musicxml_cleaned_plain.musicxml".format(song_name_))
    score = os.path.join(parent_path, "{}/score_plain.mid".format(song_name_))
    pair = os.path.join(os.path.dirname(perform), 'xml_score_perform_pairs.npy')
    if os.path.exists(pair) is True:
        pair_path = pair 
    else:
        pair_path = None

    ## LOAD DATA ##
    print()
    print()
    print("############# TEST ##############")
    print("> EXP: {}".format(exp_num))
    print("#################################")
    print()
    print("** Load Data **")

    null_tempo = 120
    get_data = GetData(null_tempo=null_tempo, 
                       same_onset_ind=same_onset_ind,
                       stat=np.mean)
    test_x, test_y, test_m, pairs, xml_notes, perform_notes, tempo, p_name = \
        get_data.file2data(
            files=[xml, score, perform], measures=measures, 
            pair_path=pair_path, save_mid=True)
    tempo_rate = tempo / null_tempo
    print("     > tempo: {}".format(tempo))
    print()


    ## LOAD MODEL ## 
    print("** Load model **")
    Generator = model.PerformGenerator
    mask = model.Mask
    note2group = model.Note2Group()

    model_path = "/workspace/Piano/gen_task/model_cvae/piano_cvae_ckpt_{}".format(exp_num) 
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")
    model = Generator(device=device)
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict']) 
    loss_val = checkpoint["loss_val"]
    checkpoint_num = len(loss_val)
    
    model.eval()
    print()
    with torch.no_grad():

        ## INFER LATENTS BY CONDITIONS ##
        indices = ["fast", "slow", "loud", "quiet", "stac", "legato", "neutral"]
        c_dict = dict()
        interp = 'tanh'
        n = 0
        
        # get latent variables by conditions
        for ind in indices:
            # get input data
            test_inputs_ = get_data.data2input(test_x, test_y, test_m, 
                cond=ind, art=True, device=device)
            test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs_
            test_inputs = test_inputs_
            # sample c
            c = model.sample_c_only(test_y_, test_m_)
            c_dict[ind] = c

        ## RECONSTRUCTION ##
        print("** Reconstruction **")
        s_note, _, _, \
        z_prior_moments, c_moments, z_moments, \
        _, z, recon_note, _, \
        est_c, est_z, _ = model(*test_inputs) 

        # get results
        y_recon_vel, y_recon_dur, y_recon_ioi = \
            inverse_feature(recon_note, art=True, numpy=False, interp="tanh")
        est_c_ = note2group.reverse(est_c, test_m_) 
        gt = test_y2_[0].cpu().data.numpy()
        clab = test_clab_[0].cpu().data.numpy()
        y_c_vel, y_c_dur, y_c_ioi = \
            inverse_feature(est_c_, art=True, numpy=False, interp="tanh")

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


        ## SAMPLE ##
        print("** Sampling **")
        _, _, z0, \
            sampled0_note = model.sample(test_x_, test_m_)
        y_sampled0 = inverse_feature(sampled0_note, art=True, numpy=False, interp=interp)
        recon_group = corrupt_to_onset(test_x, recon_note[0].cpu().data.numpy(), same_onset_ind=same_onset_ind)

        # plot group-wise results
        plt.figure(figsize=(10,15))
        plt.subplot(311)
        plt.title("velocity")
        plt.plot(range(len(test_y2_[0])), test_y2_[0,:,0].cpu().data.numpy(), label="GT") # [0,:,0].cpu().data.numpy()
        plt.plot(range(len(recon_group)), recon_group[:,0], label="est")
        plt.legend()
        plt.subplot(312)
        plt.title("articulation")
        plt.plot(range(len(test_y2_[0])), test_y2_[0,:,1].cpu().data.numpy(), label="GT")
        plt.plot(range(len(recon_group)), recon_group[:,1], label="est")
        plt.legend()
        plt.subplot(313)
        plt.title("ioi")
        plt.plot(range(len(test_y2_[0])), test_y2_[0,:,-1].cpu().data.numpy(), label="GT")
        plt.plot(range(len(recon_group)), recon_group[:,-1], label="est")
        plt.legend()
        plt.tight_layout()
        plt.savefig("recon_group_{}_{}_{}_mm{}-{}.png".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]))
        

        ## INTERPOLATION ##
        print("** Interpolation **")
        interp_t = dict()
        interp_d = dict()
        interp_a = dict()
        styles = dict()

        # tempo
        c_fast = c_dict["fast"]
        c_slow = c_dict["slow"]
        # 0 -> 1 : fast to slow
        for a in range(5):
            alpha = a / 4.
            # get latent variable
            c_seed_ = alpha * c_slow + (1-alpha) * c_fast 
            _, _, z_, sampled = model.sample(
                test_x_, test_m_, c_=c_seed_, z_=z, trunc=False, threshold=2)
            # inverse to feature
            vel, dur, ioi = \
                inverse_feature(sampled, art=True, numpy=False, interp=interp)
            interp_t[a] = [alpha, [vel, dur, ioi], c_seed_, z_]
        print("     > sampled by tempo")

        # dynamics
        c_loud = c_dict["loud"]
        c_quiet = c_dict["quiet"]
        # 0 -> 1 : fast to slow
        for a in range(5):
            alpha = a / 4.
            # get latent variable
            c_seed_ = alpha * c_quiet + (1-alpha) * c_loud 
            _, _, z_, sampled = model.sample(
                    test_x_, test_m_, c_=c_seed_, z_=z, trunc=False, threshold=2)
            # inverse to feature
            vel, dur, ioi = \
                inverse_feature(sampled, art=True, numpy=False, interp=interp)
            interp_d[a] = [alpha, [vel, dur, ioi], c_seed_, z_]
        print("     > sampled by dynamics")

        # articulations
        c_stac = c_dict["stac"]
        c_leg = c_dict["legato"]
        # 0 -> 1 : fast to slow
        for a in range(5):
            alpha = a / 4.
            # get latent variable
            c_seed_ = alpha * c_leg + (1-alpha) * c_stac 
            _, _, z_, sampled = model.sample(
                    test_x_, test_m_, c_=c_seed_, z_=z, trunc=False, threshold=2)
            # inverse to feature
            vel, dur, ioi = \
                inverse_feature(sampled, art=True, numpy=False, interp=interp)
            interp_a[a] = [alpha, [vel, dur, ioi], c_seed_, z_]
        print("     > sampled by articulations")

        # sample various versions
        c_seed_ = torch.empty_like(c_dict["neutral"]).copy_(c_dict["neutral"])
        for a in range(3):
            _, _, z_, sampled = model.sample(test_x_, test_m_, c_=c_seed_) # c_=c_seed_, 
            # inverse to feature
            vel, dur, ioi = \
                inverse_feature(sampled, art=True, numpy=False, interp=interp)
            styles[a] = [alpha, [vel, dur, ioi], c_seed_, z_]
        print("     > sampled multiple styles (neutral)")
        print()


    ## PLOT ##
    print("** Plot results **")
    y_vel = test_y[:,0]
    y_ioi = test_y[:,2]
    y_art = np.power(10, np.log10(test_y[:,1]) - np.log10(test_y[:,3]))
    # piano-roll of score
    roll = model.pianoroll(test_x_[:,:,22:110], test_m_)

    # sampled data by conditions (tempo, dynamics)
    plt.figure(figsize=(10,25))
    plt.subplot(711)
    plt.title("Sampled/Predicted velocity (change tempo)")
    plt.plot(range(len(test_x)), y_vel, label="GT")
    plt.plot(range(len(test_x)), y_recon_vel, label="pred")
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
    plt.plot(range(len(test_x)), y_c_vel, label="est_c")
    plt.plot(range(len(test_x)), styles[0][1][0], label="style 1")
    plt.plot(range(len(test_x)), styles[1][1][0], label="style 2")
    plt.plot(range(len(test_x)), styles[2][1][0], label="style 3")
    plt.legend()
    plt.subplot(715)
    plt.title("Sampled/Predicted duration (multiple styles)")
    plt.plot(range(len(test_x)), y_art, label="GT")
    plt.plot(range(len(test_x)), y_recon_dur, label="pred")
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
    

    ### RENDER MIDI ###
    print("** Render MIDI files** ")
    rendering_from_notes(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_t[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind, savename="slow_0_fast_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    rendering_from_notes(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_t[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="slow_100_fast_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    rendering_from_notes(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_d[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="quiet_0_loud_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    rendering_from_notes(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_d[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="quiet_100_loud_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    rendering_from_notes(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_a[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="stac_0_leg_100_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    rendering_from_notes(input_notes=xml_notes, save_dir="./", cond=test_x, features=interp_a[4][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="stac_100_leg_0_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    rendering_from_notes(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[0][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample1_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    rendering_from_notes(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[1][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample2_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)
    rendering_from_notes(input_notes=xml_notes, save_dir="./", cond=test_x, features=styles[2][1], tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="style_sample3_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False)

    # GT
    y_features = [test_y[:,0], y_art, y_ioi]
    gt_notes = rendering_from_notes(input_notes=xml_notes, save_dir="./", cond=test_x, features=y_features, tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  savename="gt_features_{}_exp{}_{}_mm{}-{}.mid".format(song_name, exp_num, checkpoint_num, measures[0], measures[1]), save_score=False, return_notes=True)
    print()
    print("#################################")
    print()


def generate_scratch(
    song_dir, sketch=[None, None, None], exp_num=None, device_num=None, same_onset_ind=[110,112]):

    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:{}".format(device_num) if cuda_condition else "cpu")

    ## LOAD MODEL ## 
    print("** Load model **")
    note2group = model.Note2Group()
    Generator = model.PerformGenerator
    model_path = "/workspace/Piano/gen_task/model_cvae/piano_cvae_ckpt_{}".format(exp_num)
    model = Generator(device=device)
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict']) 
    model.eval()

    with torch.no_grad():
        xml_list = sorted(glob(os.path.join(song_dir, "*.musicxml")))
        mid_list = sorted(glob(os.path.join(song_dir, "*.mid")))

        file_num = 0
        for xml, mid in zip(xml_list, mid_list):
            file_num += 1
            score = mid

            ## LOAD DATA ##
            null_tempo = 120
            get_data = GetData(null_tempo=null_tempo, 
                               same_onset_ind=same_onset_ind,
                               stat=np.mean)

            test_x, test_m, pairs, xml_notes, tempo, p_name = \
                get_data.file2data_noY(files=[xml, score], measures=measures)
            tempo_rate = tempo / null_tempo

            # get input data
            test_inputs_ = get_data.data2input_noY(test_x, test_m, device=device)
            test_x_, test_m_ = test_inputs_
            test_c_ = torch.randn(1, test_m_.size(2), 12).to(device)

            # load sketch
            for each, ind in zip(sketch, [0, 4, 8]):
                if each is not None:
                    data_len = test_m.shape[1]
                    if each == "zero":
                        line = np.zeros([data_len,])  
                    elif each == "curve_upper":
                        h1, h2 = data_len//2, data_len-data_len//2
                        random = np.concatenate([np.random.randn(h1,)-1, np.random.randn(h2,)+1])
                        line = poly_predict(random, N=4)
                    elif each == "curve_lower":
                        h1, h2 = data_len//2, data_len-data_len//2
                        random = np.concatenate([np.random.randn(h1,)+1, np.random.randn(h2,)-1])
                        line = poly_predict(random, N=4)
                    line_ = torch.from_numpy(line).to(device).float()
                    test_c_[:,:,ind] = line_.unsqueeze(0)
                else:
                    pass

            _, _, _, sampled = model.sample(
                test_x_, test_m_, c_=test_c_, trunc=True, threshold=2.)

            # inverse to feature
            vel, art, ioi = \
                inverse_feature(sampled, art=True, numpy=False, interp='tanh')
            features = [vel, art, ioi]

            rendering_from_notes(
                input_notes=xml_notes, save_dir="./", cond=test_x, features=features, 
                tempo=tempo, tempo_rate=tempo_rate,  same_onset_ind=same_onset_ind,  
                savename=os.path.join(save_path, "test_sample.{}__mm{}-{}.{}.mid".format(
                    song_name, measures[0], measures[1], exp_num)), 
                save_score=False, return_notes=False, save_perform=True)
            
            print("sampled {}/{}th file".format(file_num, len(xml_list)))

