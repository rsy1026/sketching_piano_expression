import os
import numpy as np
import pretty_midi
import csv
import time
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation
import copy
import mido

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures

from .musicxml_parser import MusicXMLDocument

dc = getcontext()
dc.prec = 48
dc.rounding = ROUND_HALF_UP


class GaussianFeatures(BaseEstimator, TransformerMixin):
    '''
    https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html#Gaussian-Basis
    '''
    """Uniformly spaced Gaussian features for one-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)

def ind2str(ind, n):
    ind_ = str(ind)
    rest = n - len(ind_)
    str_ind = rest*"0" + ind_
    return str_ind 

def pad_for_same_length(x1, x2):
    a = copy.deepcopy(x1)
    b = copy.deepcopy(x2)
    assert len(a.shape) == 2 and len(b.shape) == 2
    if len(a) > len(b):
        diff = a - b
        b = np.concatenate([b, np.zeros_like(b[:diff])], axis=0)
    elif len(b) > len(a):
        diff = b - a 
        a = np.concatenate([a, np.zeros_like(a[:diff])], axis=0)
    elif len(a) == len(b):
        pass 
    return a, b

def linear_predict(y):
    x = np.arange(len(y)).reshape(-1, 1)
    y = y
    model = LinearRegression()
    model.fit(x, y)
    yfit = model.predict(x)
    coef = model.coef_
    itct = model.intercept_

    return yfit, coef, itct

def gauss_predict(y, N):
    x = np.arange(len(y)).reshape(-1, 1)
    y = y
    gauss_model = make_pipeline(GaussianFeatures(N), LinearRegression())
    gauss_model.fit(x, y)
    yfit = gauss_model.predict(x)

    return yfit

def poly_predict(y, N):
    x = np.arange(len(y)).reshape(-1, 1)
    y = y
    poly_model = make_pipeline(PolynomialFeatures(degree=N, include_bias=True), LinearRegression())
    poly_model.fit(x, y)
    yfit = poly_model.predict(x)

    return yfit

def moving_average(
    data, win_len=None, stat=np.mean, half=False):
    '''
    data = [timestep, feature]
    '''
    new_data = list()

    if half is False:
        assert win_len % 2 == 1 
        assert win_len > 1
        unit = (win_len - 1) // 2
    elif half is True:
        unit = int(win_len - 1)

    for i in range(len(data)):
        if half is False:
            minind = np.max([0, i-unit])
            maxind = np.min([len(data), i+unit+1])
        elif half is True:
            minind = np.max([0, i-unit])
            maxind = np.min([len(data), i+1])     
        data_in_range = data[minind:maxind]       
        
        in_range = [d for d in data_in_range if d is not None]
        assert len(in_range) > 0
        mean_data = stat(in_range, axis=0)
        new_data.append(mean_data)

    return np.asarray(new_data)

def quantize(x, unit=None):
    div = x // unit
    x_prev = unit * div
    x_next = unit * (div+1)
    _prev = x - x_prev
    _next = x_next - x
    if _prev > _next:
        x_new = x_next
    elif _prev < _next:
        x_new = x_prev
    elif _prev == _next:
        x_new = x_prev
    return float(x_new)

def quantize_to_frame(value, unit):
    # for making pianoroll from MIDI 
    sample = int(round(Decimal(str(value / unit))))
    return sample

def make_pianoroll(
    notes, start=None, maxlen=None, 
    start_pitch=21, num_pitch=88,
    unit=None, front_buffer=0., back_buffer=0., cut_end=True):
    '''
    unit, buffers: in seconds
    start: time to subtract to make roll start at certain time
    '''

    # unit = float(round(Decimal(str(unit)), 3))
    if start is None:
        start = np.min([n.start for n in notes])
    if maxlen is None:
        min_ = np.min([n.start for n in notes])
        max_ = np.max([n.end for n in notes])
        maxlen = max_ - min_
        maxlen = quantize_to_frame(maxlen, unit=unit) 

    front_buffer_sample = quantize_to_frame(front_buffer, unit=unit)
    back_buffer_sample = quantize_to_frame(back_buffer, unit=unit)
    maxlen += back_buffer_sample + front_buffer_sample
    roll = np.zeros([num_pitch, maxlen])
    onset_roll = np.zeros([num_pitch, maxlen])

    onset_list = list()
    offset_list = list()
    for n in notes:
        pitch = n.pitch - start_pitch
        if n.pitch >= 70: # right-hand(temporary)
            hand = 0
        elif n.pitch < 70: # left-hand(temporary)
            hand = 1
        dur_raw = n.end - n.start
        dur = quantize_to_frame(dur_raw, unit=unit) 
        onset = quantize_to_frame(
            n.start - start + front_buffer, unit=unit)  
        offset = onset + dur  
        vel = n.velocity
        onset_list.append([hand, onset])
        offset_list.append([hand, offset]) 
 
        assert onset < maxlen
        roll[pitch, onset:offset] = vel
        onset_roll[pitch, onset] = offset - onset
            
    if cut_end is True:
        last_offset = np.max([o[1] for o in offset_list])
        roll = roll[:,:last_offset+back_buffer_sample] 
        onset_roll = onset_roll[:,:last_offset+back_buffer_sample] 
    elif cut_end is False:
        pass
        
    return roll, onset_list, onset_roll
                                                
def check_note_measure_pair(xml_notes, xml_measures):
    for n, m in zip(xml_notes, xml_measures):
        n_num = n.measure_number
        m_num = m.notes[0].measure_number
        assert n_num == m_num # same measure number

def apply_tied_notes(xml_notes, xml_measures):
    tied_indices = list()
    for i, note in enumerate(xml_notes):
        if note.note_notations.tied_stop is True: # if tied by previous note
            for j in reversed(range(i)): # find the previous note backward 
                if xml_notes[j].note_notations.tied_start is True and \
                    xml_notes[j].pitch == xml_notes[i].pitch and \
                    xml_notes[j].voice == xml_notes[i].voice:
                    xml_notes[j].note_duration.duration += xml_notes[i].note_duration.duration
                    xml_notes[j].note_duration.seconds += xml_notes[i].note_duration.seconds
                    xml_notes[j].note_duration.midi_ticks += xml_notes[i].note_duration.midi_ticks 
                    break
            tied_indices.append(i)
    # disregard tied_stop notes
    xml_notes_ = [xml_notes[k] for k in range(len(xml_notes)) if k not in tied_indices] 
    xml_measures_ = [xml_measures[k] for k in range(len(xml_measures)) if k not in tied_indices] 
    return xml_notes_, xml_measures_

def apply_grace_notes(xml_notes):
    # group by measure number
    measure_group = [[0, xml_notes[0].pitch, xml_notes[0]]]
    measure_group_list = list()
    prev_measure_number = xml_notes[0].measure_number
    for i, note in enumerate(xml_notes[1:]):
        if prev_measure_number == note.measure_number:
            measure_group.append([note.measure_number, note.pitch, note])
        else:
            measure_group_list.append(measure_group)
            measure_group = [[note.measure_number, note.pitch, note]]
        prev_measure_number = note.measure_number
    measure_group_list.append(measure_group)
    # rearrange grace notes within measure group
    '''
    since a grace note does not have time position, 
    its position can be identified by x position(physical)
    '''
    new_xml_notes = list()
    for measure in measure_group_list:
        new_order = sorted(measure, key=lambda x: x[2].x_position)
        non_grace = list()
        for note in new_order:
            if note[2].is_grace_note == False: # collect non-grace notes
                non_grace.append(note)
            elif note[2].is_grace_note == True: # if meet grace notes
                non_grace.sort(key=lambda x: x[2].pitch[1])
                non_grace.sort(key=lambda x: x[2].note_duration.time_position)
                for n in non_grace: # append sorted non-grace notes
                    new_xml_notes.append(n[2])
                non_grace = list() # initialize
                new_xml_notes.append(note[2]) # append following grace note
        # last non-grace list
        non_grace.sort(key=lambda x: x[2].pitch[1])
        non_grace.sort(key=lambda x: x[2].note_duration.time_position)
        for n in non_grace:
            new_xml_notes.append(n[2])
    return new_xml_notes

def check_in_order(xml_notes):
    '''
    check if non-grace notes are in time order
    '''
    # find first non-grace note
    for i, note in enumerate(xml_notes):
        if note.is_grace_note == False:
            break
    prev_note = xml_notes[i]
    assert prev_note.is_grace_note == False
    for i, note in enumerate(xml_notes[i+1:]):
        if note.is_grace_note == False:
            prev_onset = prev_note.note_duration.time_position
            _onset = note.note_duration.time_position
            assert prev_onset <= _onset
            prev_note = note 

def extract_xml_raw(xml_doc, measures=None):
    part = xml_doc.parts[0]
    xml_measures = list()
    xml_notes = list()
    # collect all note/measure objects 
    for measure in part.measures:
        for note in measure.notes:
            xml_notes.append(note)
            xml_measures.append(measure) 
    
    # sort by measure number
    xml_notes.sort(key=lambda x: x.measure_number)

    note_measure_pair = list()
    for n, m in zip(xml_notes, xml_measures):
        if measures is not None:
            if n.measure_number >= measures[0] and \
                n.measure_number < measures[1]:
                pair = {'note': n, 'measure': m}
                note_measure_pair.append(pair)
        elif measures is None:
            pair = {'note': n, 'measure': m}
            note_measure_pair.append(pair)            

    return note_measure_pair

def get_signatures_from_xml(xml_path, measure_start):
    xml_doc = MusicXMLDocument(xml_path)
    xml = extract_xml_raw(xml_doc)
    # parse tempo from xml
    tempo_list = list()
    time_sig = list()
    key_sig = list()
    prev_measure = -1
    for x in xml:
        # get tempo
        if len(x['measure'].directions) > 0:
            tempo_direction = [d.tempo \
                for d in x['measure'].directions if d.tempo is not None]
            if len(tempo_direction) > 0:
                tempo = np.round(float(tempo_direction[0]), 1)
                tempo_list.append([tempo, x['note'].measure_number])
        else:
            pass 

        # get time signature
        if x['measure'].time_signature is not None:
            if prev_measure < x['note'].measure_number:
                time_sig.append([x['note'].measure_number, x['measure'].time_signature])
        # get key signature
        if x['measure'].key_signature is not None:
            if prev_measure < x['note'].measure_number:
                key_sig.append([x['note'].measure_number, x['measure'].key_signature])
        
        # update prev measure number
        prev_measure = x['note'].measure_number
    
    # choose one tempo
    if len(tempo_list) > 0:
        for t in tempo_list:
            if measure_start > t[1]:
                break 
            tempo = t[0]

    elif len(tempo_list) == 0:
        tempo = 120.0

    assert tempo is not None
    assert len(time_sig) > 0
    assert len(key_sig) > 0
    return tempo, np.asarray(time_sig), np.asarray(key_sig)
    
def extract_xml_notes(
    xml_doc, note_only=True, apply_grace=True, apply_tie=True):
    parts = xml_doc.parts
    xml_measures = list()
    xml_notes = list()
    
    # collect all note/measure objects 
    for part in parts:
        for measure in part.measures:
            for note in measure.notes:
                if note_only is True:
                    if apply_grace is False:
                        if note.is_rest is False and note.is_grace_note is False:
                            xml_notes.append(note)
                            xml_measures.append(measure) 
                    elif apply_grace is True:
                        if note.is_rest is False: 
                            xml_notes.append(note)
                            xml_measures.append(measure)  

                elif note_only is False:
                    if apply_grace is False:
                        if note.is_grace_note is False:
                            xml_notes.append(note)
                            xml_measures.append(measure)
                    elif apply_grace is True:
                        xml_notes.append(note)
                        xml_measures.append(measure)   
                                     
    # sort xml notes 
    if note_only is True:
        xml_notes.sort(key=lambda x: x.pitch[1]) 
    xml_notes.sort(key=lambda x: x.note_duration.time_position)
    xml_notes.sort(key=lambda x: x.measure_number) 
    xml_measures.sort(key=lambda x: x.notes[0].measure_number)  

    ## post-process ##
    # for applying grace
    if apply_grace is True:
        xml_notes_ = apply_grace_notes(xml_notes)
    else:
        xml_notes_ = xml_notes
    # check order (for applying grace)
    check_in_order(xml_notes_)
    
    # for applying tie
    if apply_tie is True:
        xml_notes_, xml_measures_ = apply_tied_notes(xml_notes_, xml_measures)
    else:
        xml_notes_ = xml_notes_
        xml_measures_ = xml_measures
    xml_notes_, xml_measures_ = remove_overlaps_xml(xml_notes_, xml_measures_)
    
    # check measure numbers
    check_note_measure_pair(xml_notes_, xml_measures_)

    ## wrap-up ##
    # make note-measure pairs
    note_measure_pair = list()
    for n, m in zip(xml_notes_, xml_measures_):
        pair = {'note': n, 'measure': m}
        note_measure_pair.append(pair)

    return note_measure_pair

def remove_overlaps_xml(xml_notes, xml_measures):
    # find overlapped note groups
    same_notes_list = list()
    for k, note in enumerate(xml_notes):
        if note.pitch is None:
            same_notes_list.append([[k, xml_notes[k]]])
        if note.pitch is not None:
            same_notes = [[k, xml_notes[k]]]
            break 
    prev_note = xml_notes[k]
    for i, note in enumerate(xml_notes[k+1:]):
        if note.pitch is None:
            same_notes_list.append([[i+k+1, note]])
            continue
        if note.is_grace_note is False:
            if prev_note.pitch[1] == note.pitch[1] and \
                prev_note.note_duration.time_position == \
                note.note_duration.time_position: 
                same_notes.append([i+k+1, note])
            else:
                same_notes_list.append(same_notes)
                same_notes = [[i+k+1, note]]
        elif note.is_grace_note is True:
            if prev_note.pitch[1] == note.pitch[1] and \
                prev_note.x_position == note.x_position: 
                same_notes.append([i+k+1, note])
            else:
                same_notes_list.append(same_notes)
                same_notes = [[i+k+1, note]]
        prev_note = note
    same_notes_list.append(same_notes)
    same_notes_list = sorted(same_notes_list, key=lambda x: x[0][0])
    # clean overlapped notes
    cleaned_list = list()
    for j, each_group in enumerate(same_notes_list):
        if len(each_group) > 1:
            max_dur_note = sorted(each_group, 
                key=lambda x: x[1].note_duration._convert_type_to_ratio())[-1] 
            cleaned_list.append(max_dur_note)
        elif len(each_group) == 1:
            cleaned_list.append(each_group[0])
    cleaned_notes = [c[1] for c in cleaned_list]
    cleaned_measures = [xml_measures[c[0]] for c in cleaned_list]
    return cleaned_notes, cleaned_measures

def remove_overlaps_midi(midi_notes):
    midi_notes.sort(key=lambda x: x.pitch)
    midi_notes.sort(key=lambda x: x.start)
    same_notes_list = list()
    same_notes = [[0, midi_notes[0]]]
    prev_note = midi_notes[0]
    num = 0
    for i, note in enumerate(midi_notes[1:]):
        if prev_note.pitch == note.pitch and \
            prev_note.start == note.start: # if overlapped 
            same_notes.append([i+1, note])
        else:
            same_notes_list.append(same_notes)
            same_notes = [[i+1, note]]
        prev_note = note
    same_notes_list.append(same_notes)
    # clean overlapped notes
    cleaned_notes = list()
    for j, each_group in enumerate(same_notes_list):
        if len(each_group) > 1:
            max_dur_note = sorted(each_group, 
                key=lambda x: x[1].end)[-1][1]
            cleaned_notes.append(max_dur_note)
            num += 1
        elif len(each_group) == 1:
            cleaned_notes.append(each_group[0][1])
    # print("__overlapped notes: {}".format(num))
    return cleaned_notes

def erase_track_prettyMIDI(filepath, erase_ind=None):

    midi_data = mido.MidiFile(filepath)
    if type(erase_ind) is not list:
        erase_ind = [erase_ind]
    for i in erase_ind:
        midi_data.tracks[i] = []
    
    # load as PrettyMIDI object 
    PM = pretty_midi.PrettyMIDI()

    for track in midi_data.tracks:
        tick = 0
        for event in track:
            event.time += tick
            tick = event.time

    # Store the resolution for later use
    PM.resolution = midi_data.ticks_per_beat

    # Populate the list of tempo changes (tick scales)
    PM._load_tempo_changes(midi_data)

    # Update the array which maps ticks to time
    MAX_TICK = 1e7
    max_tick = max([max([e.time for e in t])
                    for t in midi_data.tracks if len(t) > 0]) + 1
    # If max_tick is huge, the MIDI file is probably corrupt
    # and creating the __tick_to_time array will thrash memory
    if max_tick > MAX_TICK:
        raise ValueError(('MIDI file has a largest tick of {},'
                            ' it is likely corrupt'.format(max_tick)))

    # Create list that maps ticks to time in seconds
    PM._update_tick_to_time(max_tick)

    # Populate the list of key and time signature changes
    PM._load_metadata(midi_data)

    # Check that there are tempo, key and time change events
    # only on track 0
    if any(e.type in ('set_tempo', 'key_signature', 'time_signature')
            for track in midi_data.tracks[1:] for e in track):
        warnings.warn(
            "Tempo, Key or Time signature change events found on "
            "non-zero tracks.  This is not a valid type 0 or type 1 "
            "MIDI file.  Tempo, Key or Time Signature may be wrong.",
            RuntimeWarning)

    # Populate the list of instruments
    PM._load_instruments(midi_data)

    return PM

def get_cleaned_midi(
    filepath, no_vel=None, no_pedal=None, erase_track=None):
    filename = os.path.basename(filepath).split('.')[0]
    if erase_track is not None:
        midi = erase_track_prettyMIDI(filepath, erase_ind=erase_track)
    else:
        midi = pretty_midi.PrettyMIDI(filepath)

    midi_new = pretty_midi.PrettyMIDI(resolution=10000, initial_tempo=120) # new midi object
    inst_new = pretty_midi.Instrument(0) # new instrument object
    min_pitch, max_pitch = 21, 108
    orig_note_num = 0
    for inst in midi.instruments: # existing object from perform midi
        for note in inst.notes:
            if note.pitch >= min_pitch and note.pitch <= max_pitch:
                inst_new.notes.append(note)
        for cc in inst.control_changes:
            inst_new.control_changes.append(cc)
        orig_note_num += len(inst.notes)
    new_note_num = len(inst_new.notes)
    # append new instrument
    midi_new.instruments.append(inst_new)
    midi_new.remove_invalid_notes()
    # in case of removing velocity/pedals
    for track in midi_new.instruments:
        if no_vel == True:
            for i in range(len(track.notes)):
                track.notes[i].velocity = 64
        if no_pedal == True:
            track.control_changes = list()

    print("{}: {}/{} notes --> plain vel: {}".format(
        filename, new_note_num, orig_note_num, no_vel))

    return midi_new

def extract_midi_notes(
    midi_path, clean=False, erase_track=None, inst_num=None, 
    no_pedal=False, raw=False, save=False, savepath=None):
    if clean is False:
        midi_obj = get_cleaned_midi(
            midi_path, no_vel=False, no_pedal=no_pedal, erase_track=erase_track)

    elif clean is True:
        midi_obj = get_cleaned_midi(
            midi_path, no_vel=True, no_pedal=True, erase_track=erase_track)

    midi_notes = list()
    ccs = list()
    for inst in midi_obj.instruments:
        if inst_num is not None:
            assert type(inst_num) == int
            if inst.program != inst_num:
                continue
        for note in inst.notes:
            note.start = float(round(Decimal(str(note.start)), 6))
            note.end = float(round(Decimal(str(note.end)), 6))
            midi_notes.append(note)
        for cc in inst.control_changes:
            ccs.append(cc)

    midi_notes.sort(key=lambda x: x.start)
    if raw is False:
        midi_notes_ = remove_overlaps_midi(midi_notes)
    else:
        midi_notes_ = midi_notes
    midi_notes_.sort(key=lambda x: x.pitch)
    midi_notes_.sort(key=lambda x: x.start)
    
    if len(midi_notes) != len(midi_notes_):
        print("cleaned duplicated notes: {}/{}".format(
            len(midi_notes_), len(midi_notes)))

    if save is True:
        save_new_midi(midi_notes_, ccs=ccs, 
            new_midi_path=savepath, initial_tempo=120, start_zero=False)
        return None 
    else:
        return midi_notes_, ccs

def read_corresp(corresp):
    lines = list()
    with open(corresp) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            lines.append(row)
    return lines

def check_duplicated_alignment(lines):
    '''
    - in case when multiple performed notes are assigned 
      to the same score note,
    - leave only one performed note that has the closest 
      pitch to the score note,
    - other notes are considered as added notes
    '''
    prev_line = None
    dup_ind_list = list()
    dup_ind = []
    for i, line in enumerate(lines):
        if prev_line is not None:
            # find duplicated notes in reference
            if line[5:] == prev_line[5:] and line[5] != '*':
                dup_ind.append(i-1)
                dup_ind.append(i)
            else:
                if len(dup_ind) > 0:
                    dup_ind_list.append(set(dup_ind))
                dup_ind = []
        prev_line = line
    # decide which note to be survived among duplicated notes
    for dup_group in dup_ind_list:
        score_pitch = int(lines[list(dup_group)[0]][8]) # pitch number
        diff_list = list()
        for each in dup_group:
            # print(lines[each])
            each_pitch = int(lines[each][3])
            pitch_diff = np.abs(score_pitch - each_pitch)
            diff_list.append(pitch_diff)
        less_diff = np.where(diff_list==np.min(diff_list))[0][0]
        # update lines
        for i, each in enumerate(dup_group):
            if i != less_diff:
                lines[each][5:] = ['*', '-1', '*', '-1', '-1', '']
    return lines

def check_corresp_notes(lines, num_score, num_perform):
    score_based = [l for l in lines if l[6] != '-1']
    perform_based = [l for l in lines if l[1] != '-1']
    not_matched = None
    if num_score != len(score_based):
        print("** number of score not matched! **")
        not_matched = True
    if num_perform != len(perform_based):
        print("** number of perform not matched! **")
        not_matched = True
    if not_matched == True:
        raise AssertionError

def extract_corresp(corresp, num_score, num_perform):
    lines = read_corresp(corresp)[1:]
    lines_sorted = sorted(lines, key=lambda x: float(x[6])) # by onset
    # sort notes without added notes(*)
    l = 0
    while lines_sorted[l][5] == '*':
        l += 1
    lines_sorted[l:] = sorted(
        lines_sorted[l:], key=lambda x: float(x[5])) # by index
    # check whether any score note is duplicated
    lines_ = check_duplicated_alignment(lines_sorted)
    # check whether corresp lines matches to number of MIDI notes
    check_corresp_notes(lines_, num_score, num_perform)
    return lines_

def match_XML_to_scoreMIDI_plain(xml_parsed, score_parsed):

    # if len(xml_parsed) != len(score_parsed):
    #     if np.abs(len(xml_parsed) - len(score_parsed)) < 3:
    #         pass 
    #     else:
    #         raise AssertionError(
    #             "** xml and score not matched!! ({} notes)".format(
    #             len(xml_parsed) - len(score_parsed)))

    # check which measures contains grace notes
    grace_measures = [xml_['note'].measure_number 
        for xml_ in xml_parsed if xml_['note'].is_grace_note == True]
    grace_measures = set(grace_measures)

    pairs = list()
    score_matched = list()
    prev_measure_number = -1
    prev_note_start = -1
    for x, xml_ in enumerate(xml_parsed):
        xml_note = xml_['note']
        xml_measure = xml_['measure']
        measure_number = xml_note.measure_number
        xml_note_start = xml_note.note_duration.time_position
        xml_pos = xml_note.x_position
        pair = None
        match = False

        # update measure onset
        '''
        note in new measure should be later 
        than last note in previous measure.
        '''
        if prev_measure_number < measure_number:
            measure_onset = prev_note_start 

        # get onset of the non-grace note in next measure
        '''
        to make sure messed-up onsets are not bigger 
        than next measure onset at least
        '''
        next_measure = False
        for xml__ in xml_parsed[x:]:
            next_note = xml__['note'] 
            if next_note.measure_number == measure_number+1 and \
                next_note.is_grace_note == False:
                next_measure = True
                break
        if next_measure is True:
            next_measure_onset = \
                next_note.note_duration.time_position
        elif next_measure is False:
            next_measure_onset = next_note.note_duration.time_position + \
                next_note.note_duration.seconds

        # search matched score note
        for s, score_note in enumerate(score_parsed):

            if x == 0: # if first xml note
                if xml_note.pitch[1] == score_note.pitch:
                    match = True 

            else:
                # straight forward comparison first
                if np.abs(score_note.start - xml_note_start) < 1e-3 and \
                    xml_note.pitch[1] == score_note.pitch:
                    match = True

                else:
                    # print("__implicit comparison")
                    for r in reversed(range(x)):
                        if xml_pos != xml_parsed[r]['note'].x_position and \
                            pairs[r]['score_midi'] is not None:
                            if xml_parsed[r]['note'].note_duration.time_position == 0: # if grace
                                continue
                            else: # only if not grace!
                                if r > 0:
                                    if xml_parsed[r]['note'].note_duration.time_position > \
                                        xml_parsed[r-1]['note'].note_duration.time_position:
                                        break # stop at note in new onset
                                elif r == 0:
                                    break

                    prev_onset = pairs[r]['score_midi'][1].start 

                    if score_note.start >= np.max([measure_onset, prev_onset]) and \
                        score_note.start < next_measure_onset and \
                        xml_note.pitch[1] == score_note.pitch:
                        match = True                                

            if match is True and s not in score_matched:
                break
            else:
                match = False
                continue

        # update pair list
        if match is True:
            pair = {'xml_note': [x, xml_note], 
                    'xml_measure': xml_measure, 
                    'score_midi': [s, score_note]}
            score_matched.append(s)

            pairs.append(pair)
            
        elif match is False:
            print("not-matched note: (note){}, (measure){}".format(xml_note, xml_measure))
            break
        
        print("matched {}th xml note: matched: {} ".format(x, match), end='\r')
        
        # update prev attributes
        prev_measure_number = xml_note.measure_number
        if xml_note.is_grace_note is False: 
            prev_note_start = xml_note_start

    # assert xml and score is paired without disregarded notes
    print(len(xml_parsed), len(score_parsed), len(pairs))
    assert len(xml_parsed) == len(pairs) or len(score_parsed) == len(pairs)

    return pairs            

def check_alignment_with_1d_plot(
    xml_parsed, score_parsed, pairs, s_name):
    # collect aligned indices
    paired_ind = list()
    for pair in pairs:
        if pair['score_midi'] is not None and \
            pair['xml_note'] is not None:
            xml_ind = pair['xml_note'][0]
            xml_pitch = pair['xml_note'][1].pitch[1] - 21
            score_ind = pair['score_midi'][0]
            score_pitch = pair['score_midi'][1].pitch - 21
            paired_ind.append([[xml_ind, score_ind], [xml_pitch, score_pitch]])
    # get measure ind
    prev = -1
    measure_ind = list()
    for i, xml_ in enumerate(xml_parsed):
        measure_num = xml_['note'].measure_number
        if measure_num > prev:
            measure_ind.append([i, measure_num])
        prev = measure_num
    measure_ind = np.array(measure_ind)

    # make roll of xml notes
    unit = 0.032
    xml_len = len(xml_parsed)
    xml_roll = np.zeros([88, xml_len])
    for i, xml_ in enumerate(xml_parsed):
        pitch = xml_['note'].pitch[1] - 21
        xml_roll[pitch, i] = 1
    # make roll of score midi notes
    score_len = len(score_parsed)
    score_roll = np.zeros([88, score_len])
    for j, score_ in enumerate(score_parsed):
        pitch = score_.pitch - 21
        score_roll[pitch, j] = 1

    # plot all rolls
    fig = plt.figure(figsize=(200, 5))
    # xml notes
    plt.subplot(211)
    plt.title("Score XML notes")
    # plt.imshow(xml_roll[:,:15], aspect='auto')
    xs1 = range(xml_roll.shape[1])
    ys1 = np.argmax(xml_roll, axis=0)
    plt.plot(xs1, ys1, 'bo-')
    plt.vlines(measure_ind[:,0], 0, 88, 
        colors='r', linestyles='dotted', linewidth=0.7)
    # annotate pitch at every note
    for x, y in zip(xs1, ys1):
        label = str(int(y))
        plt.annotate(label, # this is the text
                     (x, y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,5), # distance from text to points (x,y)
                     ha='center',
                     fontsize=5) # horizontal alignment can be left, right or center
    # annotate measure number
    for m in measure_ind:
        label = str(m[1])
        plt.annotate(label, 
                     (m[0], 87), 
                     textcoords="offset points", 
                     xytext=(0,5), 
                     ha='left',
                     fontsize=5)
    plt.xlim([-1,np.max([xml_len,score_len])+1])
    plt.ylim([0,88])
    plt.xlabel("note number")
    plt.ylabel("pitch")
    ax1 = plt.gca()
    # score midi notes
    plt.subplot(212)
    plt.title("Score MIDI notes")
    # plt.imshow(score_roll[:,:15], aspect='auto')
    xs2 = range(score_roll.shape[1])
    ys2 = np.argmax(score_roll, axis=0)
    plt.plot(xs2, ys2, 'go-')
    # annotate pitch at every note
    for x, y in zip(xs2, ys2):
        label = str(int(y))
        plt.annotate(label, 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0,5), 
                     ha='center',
                     fontsize=5) 
    plt.xlim([-1,np.max([xml_len,score_len])+1])
    plt.ylim([0,88])
    plt.xlabel("note number")
    plt.ylabel("pitch")
    ax2 = plt.gca()

    plt.tight_layout()
    trans_figure = fig.transFigure.inverted()
    lines = list()
    for ind, pitch in paired_ind:
        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(
            ax1.transData.transform([ind[0], pitch[0]]))
        coord2 = trans_figure.transform(
            ax2.transData.transform([ind[1], pitch[1]]))
        # draw a line
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                       (coord1[1], coord2[1]),
                                       linewidth=0.5,
                                       transform=fig.transFigure,
                                       color='k')
        lines.append(line)
    fig.lines = lines
    plt.tight_layout()
    plt.savefig("{}.aligned_plot.png".format(s_name))
    plt.close()

def reorder_pairs(pairs):
    new_pairs = copy.deepcopy(pairs) 
    prev_ind = -1 
    for i, pair in enumerate(new_pairs):
        if pair['perform_midi'] is not None:
            ind = pair['perform_midi'][0]
            if prev_ind > ind:
                # find None with the same pitch
                for j in reversed(range(i)):
                    if new_pairs[j]['perform_midi'] is None and \
                        new_pairs[j]['xml_note'][1].pitch[1] == pair['perform_midi'][1].pitch:
                        # find more prev pair to check index order
                        prev_pair = None
                        for k in reversed(range(j)):
                            if new_pairs[k]['perform_midi'] is not None:
                                prev_pair = new_pairs[k]
                                break 
                        # find next pair to check index order
                        next_pair = None
                        for h in range(j+1, len(new_pairs)):
                            if new_pairs[h]['perform_midi'] is not None:
                                next_pair = new_pairs[h]
                                break
                        
                        if prev_pair is not None and next_pair is not None:
                            if prev_pair['perform_midi'][0] < ind and \
                                next_pair['perform_midi'][0] > ind: # check both
                                new_pairs[j]['perform_midi'] = pair['perform_midi']
                                pair['perform_midi'] = None
                        elif prev_pair is None and next_pair is not None:
                            if next_pair['perform_midi'][0] > ind: # check next ind
                                new_pairs[j]['perform_midi'] = pair['perform_midi']
                                pair['perform_midi'] = None
                        elif prev_pair is not None and next_pair is None:
                            if prev_pair['perform_midi'][0] < ind: # check prev ind
                                new_pairs[j]['perform_midi'] = pair['perform_midi']
                                pair['perform_midi'] = None                            
                        else: 
                            continue

                        break # end finding None

            prev_ind = ind # update ind if perform_midi is not None
    
    return new_pairs
        
def match_score_to_performMIDI(
    xml_score_pairs, corresp_parsed, perform_parsed, 
    score_parsed, xml_parsed):
    new_pairs = list()
    for pair in xml_score_pairs:
        if pair['score_midi'] is not None:
            score_ind = pair['score_midi'][0]
            score_pitch = pair['score_midi'][1].pitch
            for line in corresp_parsed:
                score_ind_ = line[5]
                score_pitch_ = line[8]
                perform_ind = line[0]
                if str(score_ind) == score_ind_ and \
                    str(score_pitch) == score_pitch_:
                    if perform_ind == '*':
                        perform_note = None
                    elif perform_ind != '*':
                        perform_note = [int(perform_ind), 
                            perform_parsed[int(perform_ind)]]
                    pair['perform_midi'] = perform_note
                    break
        elif pair['score_midi'] is None:
            pair['perform_midi'] = None
        new_pairs.append(pair)  
    # get matched perform midi indices      
    perform_ind_list = [p['perform_midi'][0] for p in new_pairs 
        if p['perform_midi'] is not None]
    # append pair with no xml/score note matched
    for i, note in enumerate(perform_parsed):
        if i not in perform_ind_list:
            pair = {'xml_note': None, 
                    'xml_measure': None, 
                    'score_midi': None,
                    'perform_midi': [i, note]}
            new_pairs.append(pair)
    # check if no note is disregarded
    only_xml = [p for p in new_pairs if p['xml_note'] is not None]
    only_score = [p for p in new_pairs if p['score_midi'] is not None]
    only_perform = [p for p in new_pairs if p['perform_midi'] is not None]
    if len(xml_parsed) == len(score_parsed):
        assert len(only_xml) == len(xml_parsed)
        assert len(only_score) == len(score_parsed)
        assert len(only_perform) == len(perform_parsed)
    return new_pairs

def save_new_midi(
    notes, ccs=None, new_midi_path=None, 
    initial_tempo=120, program=0, start_zero=False):
    new_obj = pretty_midi.PrettyMIDI(resolution=10000, initial_tempo=initial_tempo)
    new_inst = pretty_midi.Instrument(program=program)
    if start_zero is True:
        notes_ = make_midi_start_zero(notes)
    elif start_zero is False:
        notes_ = notes
    new_inst.notes = notes_
    if ccs is not None:
        new_inst.control_changes = ccs
    new_obj.instruments.append(new_inst)
    new_obj.write(new_midi_path)

def make_midi_start_zero(notes):
    notes_start = np.min([n.start for n in notes])
    new_notes = list()
    for note in notes:
        new_onset = note.start - notes_start
        new_offset = note.end - notes_start
        new_note = pretty_midi.containers.Note(velocity=note.velocity,
                                               pitch=note.pitch,
                                               start=new_onset,
                                               end=new_offset)  
        new_notes.append(new_note)
    return new_notes    

def xml_to_midi_notes(xml_note):
    # xml_note object to MIDI notes
    midi_notes = list()
    for note in xml_note:
        start = Decimal(str(note.note_duration.time_position))
        dur = Decimal(str(note.note_duration.seconds))
        end = start + dur 
        end = np.max([Decimal(str(0.026)), end])
        pitch = note.pitch[1]
        new_note = pretty_midi.containers.Note(velocity=64,
                                               pitch=pitch,
                                               start=float(start),
                                               end=float(end))    
        midi_notes.append(new_note)

    midi_notes = sorted(midi_notes, key=lambda x: x.pitch)
    midi_notes = sorted(midi_notes, key=lambda x: x.start)

    return midi_notes    

def save_xml_to_midi(xml_path, save_path):
    # load xml object 
    XMLDocument = MusicXMLDocument(xml_path)
    # extract all xml notes 
    xml_parsed = extract_xml_notes(XMLDocument) 
    xml_notes = [x['note'] for x in xml_parsed] 
    # xml to midi notes
    midi_notes = xml_to_midi_notes(xml_notes)
    save_new_midi(midi_notes, new_midi_path=save_path, start_zero=False)

def sort_pair(pairs, fmt=None):
    if fmt == "xml":
        pairs_ = sorted(pairs, 
            key=lambda x: x['xml_note'][0])
    elif fmt == "mid":
        pairs_ = sorted(pairs, 
            key=lambda x: x['score_midi'][0])   
    return pairs_ 

def make_onset_pairs(pairs, fmt=None):
    pairs = sort_pair(pairs, fmt=fmt)

    if fmt == "xml":    
        prev_onset = pairs[0]['xml_note'][1].note_duration.time_position
    elif fmt == "mid":
        prev_onset = pairs[0]['score_midi'][1].start
    same_onset = [pairs[0]]
    onset_list = list()
   
    for pair in pairs[1:]:
        if fmt == "xml":
            onset = pair['xml_note'][1].note_duration.time_position
        elif fmt == "mid":
            onset = pair['score_midi'][1].start 
        if onset == prev_onset:
            same_onset.append(pair)
        elif onset > prev_onset:
            # check if all notes have same onset
            if fmt == "xml":
                onsets_only = [note['xml_note'][1].note_duration.time_position \
                    for note in same_onset]    
            elif fmt == "mid":
                onsets_only = [note['score_midi'][1].start \
                    for note in same_onset]
            onset_set = set(onsets_only)
            assert len(onset_set) == 1
            same_onset = sort_pair(same_onset, fmt=fmt)
            onset_list.append(same_onset)
            same_onset = [pair]
        prev_onset = onset

    # check if the last group have same onset
    if fmt == "xml":
        onsets_only = [note['xml_note'][1].note_duration.time_position \
            for note in same_onset]    
    elif fmt == "mid":
        onsets_only = [note['score_midi'][1].start \
            for note in same_onset]
    onset_set = set(onsets_only)
    assert len(onset_set) == 1
    same_onset = sort_pair(same_onset, fmt=fmt)
    onset_list.append(same_onset)    

    return onset_list


