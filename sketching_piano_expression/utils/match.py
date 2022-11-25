import os
import sys
import numpy as np

from .musicxml_parser import MusicXMLDocument
from .parse_utils import *
from .nakamura_match import *

'''
* The current code includes several steps:
    - should be prepared with perform WAV, score XML, score MIDI, and perform MIDI

    1. score MIDI - perform MIDI --> corresp file("*_corresp.text") 
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
            - match_score_to_performMIDI_plain()

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
        self, xml, smid, pmids, corresps=None, 
        filenames=None, save_pairs=True, plain=True, plot=False):

        score = smid 
        assert type(pmids) is list
        print()
        print()

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

