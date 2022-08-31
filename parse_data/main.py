import os
import sys
sys.setrecursionlimit(100000)
import numpy as np
from glob import glob
from fractions import Fraction
import pretty_midi
import csv
import time
import shutil
from decimal import Decimal

# import matplotlib
# matplotlib.use("agg")
# import matplotlib.pyplot as plt
# import seaborn as sns

# import soundfile as sf

from musicxml_parser import MusicXMLDocument
from parse_utils import *
from nakamura_match import *


'''
* The current code includes several steps:
    - should be prepared with perform WAV, score XML, score MIDI, and perform MIDI

    1-1. perform WAV - perform MIDI --> midi file("*_aligned.mid")
        - temporally align performance audio with performance MIDI
        - FUNCTIONS:
            - align_wav_midi()
            - pretty_midi/align_midi.py 
                (https://github.com/craffel/pretty-midi/blob/master/examples/align_midi.py)

    1-2. score MIDI - perform MIDI --> corresp file("*_corresp.text") 
        - match score-performance MIDI with Nakamura algorithm 
        - https://midialignment.github.io/demo.html (Nakamura et al., 2017)
        - FUNCTIONS:
            - save_corresp_files() 

    2. score XML - score MIDI --> var "xml_score_pairs" **IMPERFECT**
        - rule-based note-by-note matching between score XML and score MIDI 
        - FUNCTIONS:
            - match_XML_to_scoreMIDI()
    
    3. score XML- score MIDI - perform MIDI --> var "xml_score_perform_pairs" **IMPERFECT**
        - rule-based note-by-note matching with "xml_score_pairs" and corresp file("*_corresp.txt")
        - FUNCTIONS: 
            - match_score_to_performMIDI()

    4. SPLIT perform WAV by measures --> splitted wav file("*.measure*.wav")
        - split wav according to xml_score_perform_pairs 
        - can split all measures at once & split particular range of measures 
        - FUNCTIONS:
            - split_wav_by_measure()

'''

class XML_SCORE_PERFORM_MATCH(object):
    def __init__(self, 
                 current_dir=None,
                 save_dir=None,
                 program_dir=None):

        self.current_dir = current_dir
        self.save_dir = save_dir
        self.program_dir = program_dir

    def align_wav_midi(self, wav, pmid, filename=None):

        if self.save_dir is None:
            save_dir_ = os.path.dirname(pmid)
        else:
            save_dir_ = self.save_dir

        # for wav in wavs:
        save_name = os.path.join(
            save_dir_, "{}.aligned.mid".format(filename))
        # align
        if not os.path.exists(save_name):
            subprocess.call(
                ['python3', 'align_midi.py', wav, pmid, save_name])
        return save_name

    def align_xml_midi(
        self, xml, score, performs=None, corresps=None, 
        save_pairs=None, plain=True, plot=None):
        
        # load xml object 
        XMLDocument = MusicXMLDocument(xml)
        # extract all xml/midi score notes 
        xml_parsed = extract_xml_notes(XMLDocument)
        score_parsed, _ = extract_midi_notes(score, clean=True) 
        num_score = len(score_parsed)
        # match score xml to score midi
        if plain is True:
            xml_score_pairs = \
                match_XML_to_scoreMIDI_plain(xml_parsed, score_parsed)
        elif plain is False:
            xml_score_pairs = \
                match_XML_to_scoreMIDI(xml_parsed, score_parsed)
        print("** aligned score xml-score midi! **             ")
        
        if plot is True:
            check_alignment_with_1d_plot(
              xml_parsed, score_parsed, xml_score_pairs, 
              s_name=".".join(xml.split("/")[-3:-1]))

        if performs is not None or corresps is not None:
            pairs_all = dict()
            for perform, corresp in zip(performs, corresps):
                # match score pairs with perform midi
                perform_parsed, _ = extract_midi_notes(perform, clean=False, no_pedal=True)
                num_perform = len(perform_parsed)
                corresp_parsed = extract_corresp(corresp, num_score, num_perform)
                xml_score_perform_pairs = match_score_to_performMIDI(
                    xml_score_pairs, corresp_parsed, perform_parsed, score_parsed, xml_parsed)   

                if save_pairs is True:
                    if self.save_dir is None:
                        save_dir_ = os.path.dirname(perform)
                    else:
                        save_dir_ = self.save_dir
                    np.save(os.path.join(save_dir_, 
                        "xml_score_perform_pairs.npy"), xml_score_perform_pairs)
                    pairs_all = None
                    print("saved pairs for {} at {}".format(os.path.basename(perform), save_dir_))
                else:
                    pairs_all[os.path.basename(perform)] = xml_score_perform_pairs
                    print("parsed pairs for {}".format(os.path.basename(perform)))

            print("** aligned score xml-score midi-perform midi! **") 

        else:
            pairs_all = xml_score_pairs   

        return pairs_all

    def __call__(
        self, xml, smid, pmids, wav=None, corresps=None, 
        filenames=None, save_pairs=True, plain=True, plot=False):

        score = smid 
        assert type(pmids) is list
        print()
        print()

        if wav is not None:
            ### PERFORM WAV - PERFORM MIDI ### 
            performs = list()
            for pmid, filename in zip(pmids, filenames):
                perform = self.align_wav_midi(wav, pmid, filename=filename)
                performs.append(perform)
            print("** aligned wav! **                          ")
        else:
            performs = pmids

        ### SCORE MIDI - PERFORM MIDI ### 
        if corresps is None:
            corresps = list()
            for perform in performs:
                perform_name = '.'.join(os.path.basename(perform).split('.')[:-1])
                # get directory for saving
                if self.save_dir is None:
                    save_dir_ = os.path.dirname(perform)
                else:
                    save_dir_ = self.save_dir
                # check if corresp exists
                any_corresp = os.path.join(save_dir_, 
                    "{}.cleaned_corresp.txt".format(perform_name))
                if not os.path.exists(any_corresp): # make new corresp
                    print("saving corresp at {}".format(save_dir_))
                    corresp = save_corresp_file(
                        perform, score, self.program_dir, save_dir_) 
                else: # already exists
                    corresp = any_corresp
                corresps.append(corresp)

            os.chdir(self.current_dir)
            print("** aligned score midi-perform midi! **          ")
       
        elif corresps is not None:
            assert type(corresps) is list
            for corresp in corresps:
                assert os.path.exists(corresp)

        ### SCORE XML - SCORE MIDI - PERFORM MIDI ### 
        pairs = self.align_xml_midi(
            xml, score, performs, corresps,
            save_pairs=save_pairs, plain=plain, plot=plot)  


        return pairs, performs


def split_by_structure(
    mid_path, pair, filename, save_path, wav_path=None, 
    split_by_measure=None, split_by_note=None,
    fade_in=1e-4, fade_out=1e-3, dtype="float32", subtype="PCM_32"):

    # load midi
    mid, ccs = extract_midi_notes(mid_path, clean=False, no_pedal=False)
    # sort pair based on xml notes or score notes
    pair = [p for p in pair if p["xml_note"] is not None]
    pair_xml = sorted(pair, key=lambda x: x["xml_note"][0])
    pair_score = sorted(pair, key=lambda x: x["score_midi"][0])
    # get marker
    marker = get_measure_marker(pair_xml)
    max_key = np.max(list(marker.keys()))
    # initiate
    start_measure, end_measure = None, None
    start_note, end_note = None, None
    split_indices = list()

    ### GET TARGET INDICES ###
    # split all measures
    if split_by_measure == "all": 
        
        for measure in marker:
            # get start onset 
            notes = marker[measure]
            start_onset = np.min([n["perform_midi"][1].start \
                for n in notes if n["perform_midi"] is not None])
            # get end onset
            try:
                all_next_onsets = list()
                add = 0
                while len(all_next_onsets) == 0:
                    next_notes = marker[measure+1+add]
                    all_next_onsets = [n["perform_midi"][1].start \
                        for n in next_notes if n["perform_midi"] is not None]
                    add += 1
                end_onset = np.min(all_next_onsets)

            except KeyError:
                if measure == max_key:
                    end_onset = None
                else:
                    raise AssertionError 

            split_indices.append([start_onset, end_onset, 
                "measure{}-{}".format(measure, measure+1+add)])

        print("splitting all measures...")

    # split measures in part
    elif split_by_measure is not None and split_by_measure != "all":

        # split particular measures
        if split_by_measure is not None and split_by_note is None:
            # get boundaries
            start_measure, end_measure = [int(m) for m in split_by_measure]
            # get start onset 
            notes = marker[start_measure]
            start_onset = np.min([n["perform_midi"][1].start \
                for n in notes if n["perform_midi"] is not None])
            # get end onset
            try:
                all_next_onsets = list()
                add = 0
                while len(all_next_onsets) == 0:
                    next_notes = marker[end_measure+1+add]
                    all_next_onsets = [n["perform_midi"][1].start \
                        for n in next_notes if n["perform_midi"] is not None]
                    add += 1
                end_onset = np.min(all_next_onsets) 

            except KeyError:
                if measure == max_key:
                    end_onset = None
                else:
                    raise AssertionError 

            split_indices.append([start_onset, end_onset, 
                "measure{}-{}".format(measure, measure+1+add)])

            print("splitting measures from {} to {}...".format(start_measure, end_measure))  

        # split particular notes
        elif split_by_note is not None:
            # get boundaries
            start_note, end_note = [int(m) for m in split_by_note]
            # get start onset 
            '''
            find minimum perform onset among onsets that have the corresponding score onset 
            the same to that of the start note 
            '''
            start_onset = np.min([n["perform_midi"][1].start for n in pair_score \
                if n["score_midi"][1].start==pair_score[start_note]["score_midi"][1].start \
                and n["perform_midi"] is not None])
            # get end onset
            '''
            find minimum perform onset among onsets that have the corresponding score onset 
            the same to that of "the next note" of the end note 
            '''
            all_next_onsets = list()
            add = 0
            while len(all_next_onsets) == 0:
                next_note = pair_score[end_note+1+add]
                # assert next_note["score_midi"][1].start > pair_score[end_note]["score_midi"][1].start
                all_next_onsets = [n["perform_midi"][1].start for n in pair_score \
                    if n["score_midi"][1].start==next_note["score_midi"][1].start]
                add += 1
            end_onset = np.min(all_next_onsets)

            split_indices.append([start_onset, end_onset, 
                "note{}-{}".format(start_note, end_note+1+add)])

            print("splitting notes from {} to {}...".format(start_note, end_note))    


    ### SPLIT ### 
    if wav is not None:
        wav, sr = sf.read(wav_path, dtype=dtype) # stereo

    for start_onset, end_onset, name in split_indices:

        # start_onset_ = quantize(float(Decimal(str(start_onset))), unit=0.00005)
        # end_onset_ = quantize(float(Decimal(str(end_onset))), unit=0.00005)
        start_onset_ = start_onset 
        end_onset_ = end_onset
        same_part_note = list()
        same_part_cc = list()

        ### SPLIT MIDI ### 
        # get notes in the same part
        for note in mid:
          if note.start >= start_onset_ and note.start < end_onset_:
              same_part_note.append(note)
          else:
              continue
        # get ccs in the same part
        if len(ccs) > 0:
            for cc in ccs:
              if cc.time >= start_onset_ and cc.time < end_onset_:
                  same_part_cc.append(cc)
              else:
                  continue   
        else: 
            same_part_cc = None         

        if start_measure == 1 or start_note == 1: # if very start
            part_midi = same_part_note
        else: 
            part_midi = make_midi_start_zero(same_part_note) # make start zero

        # save splitted midi
        new_midi_path = os.path.join(save_dir, "{}.{}.mid".format(filename, name))
        save_new_midi(part_midi, ccs=same_part_cc, 
            new_midi_path=new_midi_path, start_zero=False)
        print("saved splitted midi for {}".format(filename))

        ### SPLIT AUDIO ###
        if wav is not None:

            start_sample = int(np.round(start_onset / (1/sr)))
            end_sample = int(np.round(end_onset / (1/sr)))
            same_part_wav = wav[start_sample:end_sample]
    
            print(">RESULT: ")
            print("  --> start_onset: {:.3f} / end_onset: {:.3f}".format(start_onset, end_onset))
            print("  --> sample length: {} / non-zero: {}".format(
                len(same_part_wav), np.sum(np.abs(same_part_wav))>0))
            print("  --> save at {}".format(save_dir))

            # fade-in and fade-out
            if start_measure == 1 or start_note == 1:
                fade_in_len = None
            else:
                fade_in_len = int(sr * fade_in)
            fade_out_len = int(sr * fade_out)
            fade_wav = fade_in_out(same_part_wav, 
                fade_in_len=fade_in_len, fade_out_len=fade_out_len)
            # save splitted audio 
            sf.write(os.path.join(
                save_dir, '{}.{}.wav'.format(filename, name)), 
                fade_wav, sr, subtype=subtype)
            print("saved splitted audio for {}".format(filename))



def main(wav_paths=None, 
         score_paths=None, 
         perform_paths=None, 
         xml_paths=None,
         save_dir=None, 
         program_dir=None,
         target_measure=None,
         target_note=None):

    if wav_paths is None:
        wav_paths = [None] * len(perform_paths)
    elif wav_paths is not None:
        wav_paths = sorted(wav_paths)
    score_paths = sorted(score_paths)
    perform_paths = sorted(perform_paths)
    xml_paths = sorted(xml_paths)

    match = XML_SCORE_PERFORM_MATCH(
        save_dir=save_dir, program_dir=program_dir)

    for wav, score, perform, xml in zip(
        wav_paths, score_paths, perform_paths, xml_paths):

        # wav, score, perform, xml = wav_paths[0], score_paths[0], perform_paths[0], xml_paths[0]

        # make sure right files to match
        s_name = '.'.join(os.path.basename(score).split('.')[:-2])
        p_name = '.'.join(os.path.basename(perform).split('.')[:-2])
        x_name = '.'.join(os.path.basename(xml).split('.')[:-2])
        if wav is None:
            assert s_name == p_name == x_name
        elif wav is not None:
            w_name = '.'.join(os.path.basename(wav).split('.')[:-2])
            assert w_name == s_name == p_name == x_name

        pair_name = os.path.join(os.path.dirname(perform), 
            "{}.xml_score_perform_pairs.npy".format(p_name))

        if target_measure is None and target_note is None:
            if not os.path.exists(pair_name):
                pair, perform2 = match(smid=score, pmid=perform, xml=xml, wav=wav, name=p_name)
                np.save(pair_name, pair)

        elif target_measure is not None or target_note is not None:
            if not os.path.exists(pair_name):
                pair, perform2 = match(smid=score, pmid=perform, xml=xml, wav=wav, name=p_name)
            else:
                pair = np.load(pair_name, allow_pickle=True).tolist()
                perform2 = os.path.join(os.path.dirname(perform), 
                    "{}.perform.aligned.cleaned.mid".format(p_name))
                wav2 = os.path.join(os.path.dirname(perform), 
                    "{}.perform.wav".format(p_name))
                if not os.path.exists(perform2) and not os.path.exists(wav2):
                    perform2 = os.path.join(os.path.dirname(perform), 
                        "{}.perform.cleaned.mid".format(p_name))

            ### split wav by measure ###
            '''
            * To determine dtype & subtype: 
            --> find out original type of audio
                --> terminal command: ffmpeg -i ***.wav
                --> find line containing a term forming like "pcm_f32le" 
                --> above term indicates "PCM 32-bit floating-point little-endian"
                --> dtype(when load wav): "float32" / subtype(when save wav): "PCM_32"

            * Fade_in, fade_out parameter are in millisec
            --> default: fade_in = 0.1ms / fade_out = 1ms 
            --> this is for avoiding tick sounds (due to discontinuity of splitted waveform)
            '''

            if target_measure == "all":
                split_by_measure = None
            else:
                split_by_measure = target_measure

            split_by_note = target_note

            split_wav_by_measure(
                wav_path=wav, mid_path=perform2, pair=pair, save_path=save_dir, 
                split_by_measure=split_by_measure, split_by_note=split_by_note,
                fade_in=1e-4, fade_out=1e-3, dtype="float32", subtype="PCM_32")
        


if __name__ == "__main__":
    '''
    * DEFAULT: split into all measures
    
        python3 main.py /home/user/data /home/user/savedir

    * To split particular measures: 

        python3 main.py /home/user/data /home/user/savedir start_measure end_measure
    
    * measure number starts with 1 
    * unfinished measure(very first measure with shorter length) is counted
    '''
    # data_dir = sys.argv[1]
    # save_dir = sys.argv[2]
    # target_measure = "all"

    # if len(sys.argv) > 3:
        # target_measure = [sys.argv[3], sys.argv[4]]

    # make new directory for saving data
    # if not os.path.exists(save_dir):
        # os.makedirs(save_dir)
    
    # for debugging
    # data_dir = "/home/rsy/Dropbox/RSY/Piano/git_piano_parsing/wav_split_by_score/test"
    # save_dir = "/home/rsy/Dropbox/RSY/Piano/git_piano_parsing/wav_split_by_score/test_result"
    # target_measure = [1, 4]

    
    #---------------------------------------------------------------------------------------------#

    data_dir = "/home/seungyeon/Piano/sarah/recording_data"
    # groups = sorted(glob(os.path.join(data_dir, "*_all/")))
    groups = sorted(glob(os.path.join(data_dir, "*_add/")))
    # group = groups[0]
    for group in groups:
        g_name = group.split('/')[-2]
        
        perform_paths = sorted(glob(os.path.join(group, "scale_*.perform.mid")))
        score_paths = sorted(glob(os.path.join(group, "scale_*.plain.mid")))
        xml_paths = sorted(glob(os.path.join(group, "scale_*.plain.xml"))) 
        # wav_paths = sorted(glob(os.path.join(group, "*.perform.wav")))
        wav_paths = None 
        # save_dir = group
        save_dir = os.path.join(data_dir, "{}_scale".format(g_name))

        if wav_paths is None:
            wav_paths_ = perform_paths
        else:
            wav_paths_ = wav_paths
        assert len(score_paths) == len(xml_paths) == len(perform_paths) == len(wav_paths_)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        program_dir = os.getcwd()

        # main func
        main(wav_paths=wav_paths, 
             score_paths=score_paths, 
             perform_paths=perform_paths,
             xml_paths=xml_paths, 
             save_dir=save_dir, 
             program_dir=program_dir,
             target_note=[1,113])


    # measures = os.path.join(data_dir, "measures.txt")
    # with open(measures, "r") as txt_file:
    #     lines = txt_file.readlines()
    #     pieces = list()
    #     for line in lines:
    #         pieces.append(line.split("\n")[0].split(" "))

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #     for piece in pieces[1:]:
    #         p_name = piece[0][:-1]
    #         # start_measure = int(piece[1][:-1])
    #         # end_measure = int(piece[2])
    #         score_paths = sorted(glob(os.path.join(data_dir, "score", "{}*.mid".format(p_name))))
    #         xml_paths = sorted(glob(os.path.join(data_dir, "musicxml", "{}*.musicxml".format(p_name))))

    #         for score, xml in zip(score_paths, xml_paths):
    #             m_name = '.'.join(os.path.basename(score).split(".")[:-1])
    #             perform_paths = sorted(glob(os.path.join(data_dir, "perform", "{}.*.mid".format(m_name))))
    #             wav_paths = sorted(glob(os.path.join(data_dir, "wav", "{}.*.wav".format(m_name))))
    #             for perform, wav in zip(perform_paths, wav_paths):
    #                 pl_name = '.'.join(os.path.basename(perform).split(".")[:-1])
    #                 new_perform = os.path.join(save_dir, "{}.perform.mid".format(pl_name))
    #                 new_score = os.path.join(save_dir, "{}.score.mid".format(pl_name))
    #                 new_xml = os.path.join(save_dir, "{}.musicxml".format(pl_name))
    #                 new_wav = os.path.join(save_dir, os.path.basename(wav))
    #                 shutil.copy(perform, new_perform)
    #                 shutil.copy(score, new_score)
    #                 shutil.copy(xml, new_xml)
    #                 shutil.copy(wav, new_wav)

    # for piece in pieces[10:11]:
    #     p_name = piece[0][:-1]
    #     start_measure = int(piece[1][:-1])
    #     end_measure = int(piece[2])
    #     score_paths = sorted(glob(os.path.join(data_dir, "parsing_space", "{}*.score.mid".format(p_name))))
    #     xml_paths = sorted(glob(os.path.join(data_dir, "parsing_space", "{}*.musicxml".format(p_name))))
    #     perform_paths = sorted(glob(os.path.join(data_dir, "parsing_space", "{}*.perform.mid".format(p_name))))
    #     wav_paths = sorted(glob(os.path.join(data_dir, "parsing_space", "{}*.wav".format(p_name))))

    #     assert len(score_paths) == len(xml_paths) == len(perform_paths) == len(wav_paths)

    #     program_dir = os.getcwd()

    #     # main func
    #     main(wav_paths=wav_paths, 
    #          score_paths=score_paths, 
    #          perform_paths=perform_paths,
    #          xml_paths=xml_paths, 
    #          save_dir=save_dir, 
    #          program_dir=program_dir,
    #          target_measure=[start_measure, end_measure])





