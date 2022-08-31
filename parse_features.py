import os
import sys
sys.setrecursionlimit(100000)
sys.path.append('workspace/Piano/gen_task/ismir_codes/parse_utils')
import numpy as np
from glob import glob
from fractions import Fraction
from pretty_midi import Note
import csv
import time
import shutil
import collections
import copy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from scipy.stats import pearsonr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation

from io import BytesIO
from zipfile import ZipFile
import urllib.request
import os,sys,requests,csv
from bs4 import BeautifulSoup as bs

from parse_utils.main import XML_SCORE_PERFORM_MATCH as MATCH
from parse_utils.parse_utils import *
import process_data as pdata

dc = getcontext()
dc.prec = 48
dc.rounding = ROUND_HALF_UP

'''
* About This code: 
    - originally from Jaejun Lee(SNU, jjlee0721@gmail.com)
    - modified by Seungyeon Rhyu(SNU, rsy1026@gmail.com)

** version 2: (2019.05~2019.11)

** version 3: (2019.12~)
    - modified for new experiments for score midi-to-performance midi

-----------------------------------------------------------------------------------------

* The current code includes several steps:
    - should be prepared with score XML, score MIDI, and performance MIDI
    1. score-performance matching: --> corresp file
        - score-performance MIDI matching with Nakamura algorithm (Nakamura et al., 2017) 
    2. XML-score matching: --> xml-score pairs
        - rule-based note-by-note matching  
    3. XML-score-performance matching: --> xml-score-perform pairs
        - rule-based note-by-note matching with corresp file 
    4. parse score features: --> score condition input
        - note-by-note parsing according to paired musicXML and score MIDI
    5. parse performance features: --> performance feature output
        - note-by-note parsing according to paired performance	

'''

def make_dir_from_crawled():

    parent_path = '/data/chopin_cleaned/original'
    data_path = '/workspace/Piano/gen_task/parse_features/crawled_data/chopin_cleaned'
    categs = sorted(glob(os.path.join(data_path, "*/")))

    for categ in categs:
        c_name = categ.split('/')[-2]
        pieces = sorted(glob(os.path.join(categ, "*/")))

        for piece in pieces:
            p_name = piece.split("/")[-2]
            if c_name == "Chopin_Etude":
                if p_name.split('_')[0] not in ['10', '25']:
                    continue
            #if "10_3_2" not in p_name:
            #    continue
            players = sorted(glob(os.path.join(data_path, c_name, p_name, '*.*')))
            for n, player in enumerate(players):
                pl_name = os.path.basename(player)
                new_dir = os.path.join(parent_path, c_name, p_name, ind2str(n+1, 2))
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                new_path = os.path.join(new_dir, pl_name)
                if os.path.exists(new_path) is True:
                    os.remove(new_path)
                shutil.copy(player, new_path)
                print("copied {} for {}".format(pl_name, p_name))
                print("	--> {}".format(new_path))

    ################ ERASE COPIED DATA ##################
    # for piece in pieces:    
    #     players = sorted(glob(os.path.join(piece, '*/')))
    #     for player in players:
    #         shutil.rmtree(player[:-1])
    #####################################################


def make_dir_from_raw_dataset():
    data_path = '/data/asap_dataset/original'
    parent_path = '/data/asap_dataset/original_raw'
    composers = sorted(glob(os.path.join(parent_path, "*/")))

    for cp in composers: 
        cp_name = cp.split("/")[-2]
        categs = sorted(glob(os.path.join(cp, "*/")))
        for categ in categs:
            c_name = "{}_{}".format(cp_name, categ.split("/")[-2])
            pieces = sorted(glob(os.path.join(categ, "*/")))
            if len(pieces) == 0:
                pieces = np.copy(categs)
                no_piece_name = True 
            else:
                pieces = pieces
                no_piece_name = False
            for piece in pieces:
                if no_piece_name is False:
                    p_name = piece.split("/")[-2]
                    score = os.path.join(piece, 'midi_score.mid')
                    xml = os.path.join(piece, 'xml_score.musicxml')
                    players = sorted(glob(os.path.join(piece, '*[!midi_score].mid')))
                elif no_piece_name is True:
                    p_name = "0"
                    score = os.path.join(categ, 'midi_score.mid')
                    xml = os.path.join(categ, 'xml_score.musicxml')
                    players = sorted(glob(os.path.join(categ, '*[!midi_score].mid')))
                assert os.path.exists(score)
                assert os.path.exists(xml)
                new_dir = os.path.join(data_path, c_name, p_name)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                shutil.copy(score, os.path.join(new_dir, 'score_plain.mid'))
                shutil.copy(xml, os.path.join(new_dir, 'musicxml_cleaned_plain.musicxml'))                    
                for n, player in enumerate(players):
                    pl_name = os.path.basename(player)
                    new_sub_dir = os.path.join(new_dir, ind2str(n+1, 2))
                    if not os.path.exists(new_sub_dir):
                        os.makedirs(new_sub_dir)
                    new_path = os.path.join(new_sub_dir, pl_name)
                    shutil.copy(player, new_path)
                    print("copied {} for {}/{}".format(pl_name, p_name, c_name))
                    print("	--> {}".format(new_path))


def crawl_yamaha_chopin():
    '''
    Ref: https://stackoverflow.com/questions/57597972/python-web-scraping-and-downloading-specific-zip-files-in-windows
    '''

    home_url = 'https://www.yamahaden.com'
    parent_url = 'https://www.yamahaden.com/midi-files/category/frederic_chopin'
    save_parent_path = "/workspace/Piano/gen_task/parse_features/crawled_data/chopin_cleaned"
    if not os.path.exists(save_parent_path):
        os.makedirs(save_parent_path)

    # get page and setup BeautifulSoup
    r1 = requests.get(parent_url)
    soup1 = bs(r1.content, "html.parser")
    parent_label = soup1.find_all("div", {"class": "category"})
    
    for td1 in parent_label: # each category (Op. 10 or Op. 25)
        if "Ballade" in td1.text: #Étude
            # print(td1.text)
            # pieces = td1.find_all("div", {"class": "sub-categories"})
            pieces = td1.find_all("a")
            cat_name = pieces[0]["title"]
            print()
            print("< {} >".format(cat_name))
            print()
            for each_piece in pieces[1:]: # each piece
                piece_link = each_piece['href']
                piece_name = each_piece.get_text(strip=True)
                if "Op. 38" in piece_name: 
                    print('crawling mids from: {}'.format(piece_name))
                    sub_url = home_url + piece_link
                    r2 = requests.get(sub_url)
                    soup2 = bs(r2.content, "html.parser")
                    mainlabel2 = soup2.find_all("li", {"class": "element element-download"})
                    save_path = os.path.join(save_parent_path, "Chopin_Ballade", piece_name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    for each_pl in mainlabel2: # each player
                        if "Standard MIDI File" in each_pl.text:
                            pl_link = each_pl.find_next('a')
                            pl_name = pl_link['title']
                            print(' --> {}'.format(each_pl.get_text(strip=True)))
                            download_link = home_url + pl_link['href']
                            url = urllib.request.urlopen(download_link)
                            saveData = url.read()
                            outputFilename = os.path.join(save_path, pl_name)
                            # Save zip file to disk
                            print ("    ----> saved to {}".format(outputFilename))
                            output = open(outputFilename,'wb')
                            output.write(saveData)
                            output.close()


def search(dirname):
    """ 
    * Function to search 4 kinds of files in 'dirname'
        - xml(score): 'xml_list'
        - midi(score): 'score_midi_list'
        - corresp file(score-perform): 'corresp_list'
        - midi(perform): 'perform_midi_list'
    """
    # initialize lists 
    xml_list = dict()
    score_midi_list = dict()
    # corresp_list = dict()
    perform_midi_list = dict()

    # collect directories 
    categs = sorted(glob(os.path.join(dirname, "*/"))) # category ex) Ballade
    # categs = [os.path.join(dirname, "Chopin_Etude/")]

    for c in categs:
        # c = categs[0]
        c_name = c.split('/')[-2] # get category name
        # if 'Chopin' in c_name:
        #     continue
        xml_list[c_name] = dict()
        score_midi_list[c_name] = dict()
        perform_midi_list[c_name] = dict()
        pieces = sorted(glob(os.path.join(c, "*/"))) # piece ex) (Ballade) No.1
        for p in pieces: 
            # p = pieces[0]
            p_name = p.split('/')[-2] # get piece name
            players = sorted(glob(os.path.join(p, "[!pro]*/"))) # player 3x) (Ballade No.1) player1
            # get each path of xml, score, performance files
            xml_path = os.path.join(p, "musicxml_cleaned_plain.musicxml")
            score_path = os.path.join(p, "score_plain.mid")
            # assign paths to corresponding piece category
            if os.path.exists(xml_path) is True:
                xml_list[c_name][p_name] = xml_path
                score_midi_list[c_name][p_name] = score_path
                perform_midi_list[c_name][p_name] = list()
                for pl in players:
                    # pl = players[0]
                    pl_name = pl.split('/')[-2]
                    perform_path = glob(os.path.join(pl, '[!score_plain]*.mid'))
                    perform_path += glob(os.path.join(pl, '[!score_plain]*.MID'))
                    perform_path = [p for p in perform_path if os.path.basename(p).split(".")[1] != "cleaned"]
                    perform_midi_list[c_name][p_name].append(perform_path[0])
    return xml_list, score_midi_list, perform_midi_list


def save_matched_files():
    # PARENT DIRECTORY
    # dirname = '/home/rsy/Dropbox/RSY/Piano/data/chopin_maestro/original'
    # dirname = '/data/chopin_cleaned/original'
    # dirname = '/data/chopin_maestro/original'
    dirname = '/data/asap_dataset/exp_data/listening_test/raw'
    program_dir = '/workspace/Piano/gen_task/match_score_midi'
    # get directory lists
    xml_list, score_midi_list, perform_midi_list = search(dirname)
    match = MATCH(current_dir=os.getcwd(), 
                  program_dir=program_dir)	

    # ########### for debugging #############
    # categ = sorted(xml_list)[0]
    # piece = sorted(xml_list[categ])[0] # 57

    # categ = sorted(xml_list)[1]
    # piece = sorted(xml_list[categ])[-4] # 25-5

    # categ = sorted(xml_list)[-2]
    # piece = sorted(xml_list[categ])[-1] # 31-2

    # categ = sorted(xml_list)[1]
    # piece = sorted(xml_list[categ])[4] # 10-3	

    # categ = sorted(xml_list)[-1]
    # piece = sorted(xml_list[categ])[0] # 58-1	
    # #######################################
    
    # start matching
    for categ in sorted(perform_midi_list): 
        for piece in sorted(perform_midi_list[categ]):
            # categ = sorted(xml_list)[0]
            # piece = sorted(xml_list[categ])[-5] # 25-5
            xml = xml_list[categ][piece]
            score = score_midi_list[categ][piece]
            # save_xml_to_midi(xml, score)
            performs = perform_midi_list[categ][piece]
            # print("saved pairs for {}:{}".format(categ, piece))
            # save pairs: xml-score-perform
            pair_path = os.path.join(
                os.path.dirname(performs[0]), "xml_score_perform_pairs.npy")
            if os.path.exists(pair_path) is False:
                # try:
                _, _ = match(xml, score, performs, save_pairs=True, plot=True)	
                # except:
                    # print("** passed due to error: {}: {}".format(categ, piece))
                    # continue

            print("saved pairs for {}:{}".format(categ, piece))


def save_features_xml():
    # dirname = '/data/chopin_maestro/original'
    dirname = '/data/chopin_cleaned/original'
    categs = sorted(glob(os.path.join(dirname, "*/")))
    
    pc = ['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B']

    # ################## for debugging ###################
    # # etude 10-4
    # c_name = categs[0].split('/')[-2]
    # pieces = sorted(glob(os.path.join(categs[0], "*/")))
    # piece = pieces[4]
    # # scherzo 31-2
    # c_name = categs[-1].split('/')[-4]
    # pieces = sorted(glob(os.path.join(categs[-4], "*/")))
    # piece = pieces[0]
    # ####################################################

    for categ in categs:
        c_name = categ.split('/')[-2]
        pieces = sorted(glob(os.path.join(categ, "*/")))
        # if c_name != "Chopin_Etude":
        #     continue
        for piece in pieces:
            p_name = piece.split('/')[-2]
            pair_path = os.path.join(piece, 
                '01', "xml_score_perform_pairs.npy")
            pairs = np.load(pair_path, allow_pickle=True).tolist()
            pairs_xml = [p for p in pairs if p['xml_note'] is not None and \
                p['xml_note'][1].is_grace_note is False]
            pairs_xml = sorted(pairs_xml, key=lambda x: x['xml_note'][0])
            if os.path.exists(os.path.join(piece, 'cond__.npy')) is False:
                # PARSE FEATURES
                cond_list = list()
                note_list = list()
                csv_list = list()
                prev_xml_note = None
                prev_xml_measure = None

                for i in range(len(pairs_xml)):
                    xml_note = pairs_xml[i]['xml_note'][1]
                    xml_measure = pairs_xml[i]['xml_measure']	
                    midi_ind = pairs_xml[i]['score_midi'][0]

                    # parse features for each note
                    parsed_note = XMLFeatures(note=xml_note,
                                              measure=xml_measure,
                                              prev_note=prev_xml_note,
                                              prev_measure=prev_xml_measure,
                                              note_ind=i)

                    _input = parsed_note._input


                    ################# CHECK VIA CSV ####################

                    # add parsed feature informations to csv files 
                    csv_list.append([])
                    csv_list.append(['<{}th note: measure {}>'.format(i, xml_note.measure_number)])
                    csv_list.append(['> tempo: {}'.format(parsed_note.tempo)])
                    csv_list.append(['> time_signature: {}'.format(parsed_note.time_signature_text)])
                    csv_list.append(['> key_signature: {} (num_accidentals: {})'.format(parsed_note.key_signature, parsed_note.key_sig)])
                    csv_list.append(['> onset: {:4f}'.format(parsed_note.time_position)])
                    csv_list.append(['> pitch: {} (pc: {} / octave: {})'.format(
                        parsed_note.pitch_name, parsed_note.pitch_class, parsed_note.octave)])
                    # # csv_list.append(['> pitch/normalized_pitch: {}({})/{} (key: {} {})'.format(
                    # 	# parsed_note.pitch_name, parsed_note.pitch, parsed_note.pitch_norm, pc[int(parsed_note.key_final)], parsed_note.mode)])
                    # csv_list.append(['> type: {}'.format(parsed_note._type)])
                    # csv_list.append(['> is_dot: {}'.format(parsed_note.is_dot)])
                    # csv_list.append(['> voice: {}'.format(parsed_note.voice)])
                    # csv_list.append(["> current_directions: {}".format(
                    # 	[d.type['content'] for d in parsed_note.current_directions])])
                    csv_list.append(["> dynamics: {}".format(parsed_note.dynamics)])
                    csv_list.append(["> is_new_dynamic: {}".format(parsed_note.is_new_dynamics)])
                    csv_list.append(["> next_dynamic: {}".format(parsed_note.next_dynamics)])
                    # csv_list.append(["> wedge: {}".format(str(parsed_note.wedge))])
                    # csv_list.append(["> same_onset: {}".format(parsed_note.same_onset)])
                    csv_list.append(["> beat: {}".format(parsed_note.beat)])
                    csv_list.append(["> is_downbeat: {}".format(parsed_note.is_downbeat)])
                    # csv_list.append(["> is_grace_note: {}".format(parsed_note.is_grace_note)])

                    # # csv_list.append(["> ornament: {}".format(parsed_note.ornament)])
                    # # csv_list.append(["> tuplet: {}".format(parsed_note.tuplet)])
                    # csv_list.append(["> is_tied: {}".format(parsed_note.is_tied)])	

                    csv_list.append(["---------------------------------------------------------------------------"])
                    
                    csv_list.append(["--> tempo input: {}".format(_input[:1])])
                    # csv_list.append(["--> type input: {}".format(_input[1:7])]) # 6 
                    # csv_list.append(["--> dot input: {}".format(_input[7:9])]) # 2
                    csv_list.append(["--> staff input: {}".format(_input[19:21])]) # 2
                    # csv_list.append(["--> grace note input: {}".format(_input[11:13])]) # 2
                    # csv_list.append(["--> voice input: {}".format(_input[13:17])]) # 4
                    csv_list.append(["--> dynamic input: {}".format(_input[13:19])]) # 6
                    # csv_list.append(["--> pitch result: {}".format(_input[23:111])]) # 88
                    # csv_list.append(["--> pitch result 1: pc input: {}".format(_input[111:123])]) # 12
                    # csv_list.append(["--> pitch result 2: octave input: {}".format(_input[123:131])]) # 8
                    # # csv_list.append(["--> ornament input: {}".format(_input[42:46])])
                    # # csv_list.append(["--> tuplet input: {}".format(_input[46:49])])
                    # # csv_list.append(["--> tied input: {}".format(_input[42:44])])				
                    # csv_list.append(["--> same onset input: {}".format(_input[131:133])]) # 2
                    # csv_list.append(["--> wedge input: {}".format(_input[133:138])]) # 5
                    csv_list.append(["--> downbeat input: {}".format(_input[21:])])
                    csv_list.append(["--> beat input: {}".format(_input[1:13])]) # 12
                    csv_list.append(["--> time signature input(num): {}".format(_input[23:35])]) # 12
                    csv_list.append(["--> time signature input(denom): {}".format(_input[35:47])]) # 12
                    csv_list.append(["--> key signature input(num. accidental): {}".format(_input[47:])]) # 12
                    csv_list.append([])
                    csv_list.append([])
                    
                    ####################################################


                    # append input list
                    cond_list.append([midi_ind, _input])

                    # update previous measure number and onset time 
                    prev_xml_note = parsed_note # InputFeatures object
                    prev_xml_measure = parsed_note.measure # xml object

                # save csv file for checking 
                writer = csv.writer(open("./features_check_{}_{}_downbeat.csv".format(c_name, p_name), 'w'))
                for row in csv_list:
                    writer.writerow(row)
                cond_list = np.asarray(cond_list, dtype=object)

                np.save(os.path.join(piece, "cond.npy"), cond_list)
                print()
                print("parsed {}/{} condition input".format(c_name, p_name))


def save_features_midi(mode=None):
    '''
    <Chopin cleaned> (updated 210721)
    - matched note number: 808360 notes (dataset)
    - performance note number: 837613 notes 

    * only Etude: 
    - matched note number: 480611 notes 
    - perform note number: 499812 notes
    '''
    dirname = '/data/chopin_cleaned/original'
    # categs = [os.path.join(dirname, "Chopin_Etude/")]
    categs = sorted(glob(os.path.join(dirname, "*/")))
    all_inputs = list()
    all_outputs = list()
    all_pair_num = 0
    all_matched_num = 0
    all_perform_num = 0

    for categ in categs:
        c_name = categ.split('/')[-2]
        pieces = sorted(glob(os.path.join(categ, "*/")))
        for piece in pieces:
            p_name = piece.split('/')[-2]
            players = sorted(glob(os.path.join(piece, '*/')))
            if os.path.exists(os.path.join(players[0], 'inp.npy')) is False:
                midi_path = os.path.join(piece, "score_plain.mid")
                for player in players:
                    pl_name = player.split('/')[-2]
                    pair_path = os.path.join(player, "xml_score_perform_pairs.npy")
                    if not os.path.exists(pair_path):
                        continue
                    pairs = np.load(pair_path, allow_pickle=True)
                    pair_num = len(pairs)
                    perform_num = len([p for p in pairs if p['perform_midi'] is not None])
                    matched_num = len([p for p in pairs if p['xml_note'] is not None and \
                        p['score_midi'] is not None and p['perform_midi'] is not None])
                    all_pair_num += pair_num
                    all_matched_num += matched_num
                    all_perform_num += perform_num

                    #print("parsed {}/{} output: player {}".format(c_name, p_name, pl_name))
                    # try:
                    input_list, output_list = parse_midi_features(
                        mode=mode, pair_path=pair_path,
                        same_onset_ind=[110,112], null_tempo=120)
                    # except AssertionError:
                    #     print("** passed due to AssetionError: {}: {} (player {})".format(
                    #         c_name, p_name, pl_name))
                    #     continue

                    # save outputs
                    np.save(os.path.join(player, 'inp.npy'), input_list)
                    # np.save(os.path.join(player, 'inp2.npy'), input_list) # more class for dur,ioi
                    # np.save(os.path.join(player, 'oup2.npy'), output_list) # ioi in onset-level
                    np.save(os.path.join(player, 'oup.npy'), output_list) # ioi in note-level
                    # np.save(os.path.join(player, 'oup3.npy'), output_list) # micro-timing ratio
                    # np.save(os.path.join(player, 'oup4.npy'), output_list) # w/o micro-timing
                    print("parsed {}/{} output: player {} (inp len: {} / oup len: {})".format(
                        c_name, p_name, pl_name, len(input_list), len(output_list)))
    print("matched note number: {}".format(all_matched_num))
    print("perform note number: {}".format(all_perform_num))


def parse_test_cond(
    pair=None, pair_path=None, small_ver=True, tempo=None, time_sig=None, key_sig=None):
    if pair is not None:
        pairs = pair 
    elif pair is None and pair_path is not None:
        pairs = np.load(pair_path).tolist()

    pairs_xml = [p for p in pairs if p['xml_note'] is not None]
    pairs_xml = sorted(pairs_xml, key=lambda x: x['xml_note'][0])

    # PARSE FEATURES
    cond_list = list()
    note_list = list()
    csv_list = list()
    prev_xml_note = None
    prev_xml_measure = None
    for i in range(len(pairs_xml)):
        xml_note = pairs_xml[i]['xml_note'][1]
        xml_measure = pairs_xml[i]['xml_measure']	
        midi_ind = pairs_xml[i]['score_midi'][0]

        # parse features for each note
        parsed_note = XMLFeatures(note=xml_note,
                                  measure=xml_measure,
                                  prev_note=prev_xml_note,
                                  prev_measure=prev_xml_measure,
                                  note_ind=i,
                                  tempo=tempo,
                                  time_sig=time_sig,
                                  key_sig=key_sig)

        _input = parsed_note._input
        
        # append input list
        cond_list.append([midi_ind, _input])

        # update previous measure number and onset time 
        prev_xml_note = parsed_note # InputFeatures object
        prev_xml_measure = parsed_note.measure # xml object

    cond_list = np.asarray(cond_list, dtype=object)
    cond_list = sorted(cond_list, key=lambda x: x[0])
    cond = np.asarray([c[1] for c in cond_list])

    if small_ver is True:
        out = cond[:,19:23] # w/o tempo(1)/beat(12)/dynamics(6)
    elif small_ver is False:
        out = cond

    return out


def parse_midi_features(
    mode=None, pairs_score=None, pair_path=None,
    tempo=None, null_tempo=None, same_onset_ind=None):

    '''
    * function for parsing MIDI features
        - always parse in order of SCORE MIDI 
    '''

    ## FOR DEBUGGING ## 
    # pair_path = '/data/chopin_cleaned/original/Chopin_Etude/10_1/02/xml_score_perform_pairs.npy'
    # midi_path = '/data/chopin_cleaned/original/Chopin_Etude/10_1/score_plain.mid'
    # pairs_score = None 
    # midi_notes = None 
    # tempo = 176

    # pair_path = '/data/chopin_cleaned/original/Chopin_Etude/10_8/03/xml_score_perform_pairs.npy'
    # xml = '/data/chopin_cleaned/original/Chopin_Etude/10_8/musicxml_cleaned_plain.musicxml'
    # pair_path = '/data/chopin_cleaned/original/Chopin_Berceuse/57/02/xml_score_perform_pairs.npy'
    # xml = '/data/chopin_cleaned/original/Chopin_Berceuse/57/musicxml_cleaned_plain.musicxml'
        
    # tempo, time_sig = get_tempo_from_xml(xml, measure_start=0)
    # pairs_score = None 
    # midi_notes = None
    # null_tempo = 120
    # mode = "note" 

    ###################


    if pair_path is not None:
        pairs = np.load(pair_path, allow_pickle=True).tolist()
        pairs_score = [p for p in pairs if p['xml_note'] is not None and \
            p['xml_note'][1].is_grace_note is False]
        pairs_score_raw = sorted(pairs_score, key=lambda x: x['xml_note'][0])

        ## FOR DEBUGGING ## 
        # measures = [1,8]
        # start_measure, end_measure = measures[0]-1, measures[1]-1
        # pairs_score = list()
        # for note in pairs_score_raw:
        #     measure_num = note['xml_note'][1].measure_number
        #     if measure_num >= start_measure and measure_num <= end_measure:
        #         pairs_score.append(note)
        # pairs_score_raw = sorted(pairs_score, key=lambda x: x['xml_note'][0])
        ###################

    elif pairs_score is not None:
        pairs_score_raw = pairs_score

    # fix errors in index order
    pairs_score = reorder_pairs(pairs_score_raw)

    # get first onsets
    pairs_score_onset = make_onset_pairs(pairs_score, fmt="xml")
    first_onset_group = [n for n in pairs_score_onset[0] \
        if n['perform_midi'] is not None]
    '''
    make sure at least one note is in the first_note_group
    '''
    if len(first_onset_group) == 0:
        # find first group with performed notes
        o = 1
        while len(first_onset_group) == 0: 
            first_onset_group = [n for n in pairs_score_onset[o] \
                if n['perform_midi'] is not None]
            o += 1	
    elif len(first_onset_group) > 0:
        pass		

    first_onset_perform = np.min(
        [n['perform_midi'][1].start for n in first_onset_group])
    first_onset_score = np.min(
        [n['xml_note'][1].note_duration.time_position for n in pairs_score_onset[0]])

    # print("** first perform onset: {:.4f}".format(first_onset_perform))
    # print("** first score onset: {:.4f}".format(first_onset_score))

    # get score midi for check
    # if midi_path is not None:
    #     midi_notes, _ = extract_midi_notes(midi_path, clean=True)
    # if midi_notes is not None:
    #     midi_notes = midi_notes

    # set first onsets to 0 for score and perform
    for note in pairs_score:
        if note['perform_midi'] is not None:
            note['perform_midi'][1].start -= first_onset_perform
            note['perform_midi'][1].end -= first_onset_perform

        if note['xml_note'] is not None:
            note['xml_note'][1].note_duration.time_position -= first_onset_score

    # parse features per note
    input_list = list()
    output_list = list()
    note_list = list()
    onset_score_list = list()
    base_onset_perform_list = list()
    mean_onset_perform_list = list()
    next_onset_perform_list = list()
    
    prev_note = None
    prev_mean_onset_score = None
    prev_mean_onset_perform = None 

    # make pairs in onset AGAIN --> time changed (- first onset)
    # just for sure
    pairs_onset = make_onset_pairs(pairs_score, fmt="xml")

    for i in range(len(pairs_score)):
        # assign each pair
        note = pairs_score[i]

        if i < len(pairs_score)-1:
            next_note = pairs_score[i+1]
        elif i == len(pairs_score):
            next_note = None
        note_ind = note['xml_note'][0]  

        # parse features for each note
        parsed_note = MIDIFeatures(note=note,
                                   prev_note=prev_note,
                                   next_note=next_note,
                                   note_ind=note_ind,
                                   tempo_=tempo,
                                   null_tempo=null_tempo)

        if prev_note is None: # first note
            '''
            * assume the first note starts after 16th-length rest
                -> since the first note also has micro-timing and non-zero onset
            * same prev mean onset for score and perform
            '''
            prev_mean_onset_score = -parsed_note.dur_16th
            prev_mean_onset_perform = -parsed_note.dur_16th

        if parsed_note.is_same_onset is False: # different note group

            # current onset -> find onset group having current note index
            same_onset = [onset for onset in pairs_onset \
                if note_ind == onset[0]['xml_note'][0]]
            assert len(same_onset) == 1
            same_onset = same_onset[0]
            same_onset_values_perform = [o['perform_midi'][1].start \
                for o in same_onset if o['perform_midi'] is not None]
            if len(same_onset_values_perform) == 0:
                same_mean_onset_perform = None
            elif len(same_onset_values_perform) > 0:
                same_mean_onset_perform = np.mean(same_onset_values_perform) # mean
            
            same_mean_onset_score = parsed_note.score_onset

            '''
            * next note to current group's last note 
                is the first note of the next group
            '''
            same_onset_last_ind = same_onset[-1]['xml_note'][0]
            last_pair_ind = pairs_score[-1]['xml_note'][0]

            if same_onset_last_ind < last_pair_ind:
                next_onset = list()
                n = 1
                while len(next_onset) == 0: # find next note group
                    for onset in pairs_onset:
                        if same_onset_last_ind+n in [o['xml_note'][0] for o in onset]:
                            next_onset.append(onset) 
                    n += 1
                assert len(next_onset) == 1 # next_onset is always length 1
                next_onset = next_onset[0]
                next_onset_values_perform = [o['perform_midi'][1].start \
                    for o in next_onset if o['perform_midi'] is not None]	
                if len(next_onset_values_perform) == 0:
                    next_mean_onset_perform = None 
                elif len(next_onset_values_perform) > 0:
                    next_mean_onset_perform = np.mean(next_onset_values_perform) # mean

                next_mean_onset_score = next_onset[0]['xml_note'][1].note_duration.time_position
            
            elif same_onset_last_ind == last_pair_ind: # if last note group
                next_onset = same_onset
                next_onset_values_perform = [o['perform_midi'][1].end \
                    for o in next_onset if o['perform_midi'] is not None]
                if len(next_onset_values_perform) == 0:
                    next_mean_onset_perform = None 
                elif len(next_onset_values_perform) > 0:
                    next_mean_onset_perform = np.max(next_onset_values_perform)
                next_mean_onset_score = np.max([o['xml_note'][1].note_duration.time_position + \
                    o['xml_note'][1].note_duration.seconds \
                    for o in next_onset]) # notes can have different offsets within group

            # get current mean onset for the note group
            mean_onset_score = same_mean_onset_score
            mean_onset_perform = same_mean_onset_perform

            # get previous mean onset for computing ioi (current - prev)
            base_onset_score = prev_mean_onset_score
            base_onset_perform = prev_mean_onset_perform

            # get next mean onset for computing next ioi (next - current)
            next_onset_score = next_mean_onset_score
            next_onset_perform = next_mean_onset_perform
            
        elif parsed_note.is_same_onset is True: # same note group
            mean_onset_score = mean_onset_score
            mean_onset_perform = mean_onset_perform 
            base_onset_score = base_onset_score	
            base_onset_perform = base_onset_perform	
            next_onset_score = next_onset_score	
            next_onset_perform = next_onset_perform	

        # gather onsets and note object
        base_onset_perform_list.append(base_onset_perform)
        mean_onset_perform_list.append(mean_onset_perform)
        next_onset_perform_list.append(next_onset_perform)	
        note_list.append([note_ind, parsed_note])

        # print()
        # print(same_onset_values)
        # print("{},{},{},{},".format(
            # parsed_note.is_same_onset, base_onset_perform, mean_onset_perform, next_onset_perform))

        _input = parsed_note.get_input_features(
            base_onset_score, next_onset_score)
        _output = parsed_note.get_output_features(
            base_onset_perform, mean_onset_perform, next_onset_perform, mode=mode)

        # ioi1s.append(float(parsed_note.ioi_ratio1))
        # ioi2s.append(float(parsed_note.ioi_ratio2))
        # mode_ioi1 = stats.mode(ioi1s, axis=None).mode[0]
        # mode_ioi2 = stats.mode(ioi2s, axis=None).mode[0]

        # print(_output, parsed_note.is_same_onset)
        if parsed_note.ioi_ratio2 <= 0 or parsed_note.ioi_ratio1 <= 0:
            print(base_onset_score, mean_onset_score, next_onset_score, parsed_note.is_same_onset)
            print(parsed_note.score_ioi_norm1)
            print()
            break

        input_list.append([note_ind, _input])
        output_list.append([note_ind, _output])
        
        # update previous attributes
        prev_note = parsed_note # MIDIFeatures object
        prev_mean_onset_score = mean_onset_score 
        prev_mean_onset_perform = mean_onset_perform

    assert len(output_list) == len(input_list)	

    input_list = sorted(input_list, key=lambda x: x[0])
    output_list = sorted(output_list, key=lambda x: x[0])
    note_list = sorted(note_list, key=lambda x: x[0])
    inp_ind = sorted([i[0] for i in input_list])
    oup_ind = sorted([i[0] for i in output_list])

    # find missing parts for iois (ioi1 & ioi2)
    assert mode == "note"
    output_list_, non_durs = interpolate_feature(output_list, input_list, note_list, 
        f_type="dur", same_onset_ind=same_onset_ind)
    output_list_, non_ioi1s = interpolate_feature(output_list_, input_list, note_list,
        f_type="ioi1", same_onset_ind=same_onset_ind)
    output_list_, non_ioi2s = interpolate_feature(output_list_, input_list, note_list, 
        f_type="ioi2", same_onset_ind=same_onset_ind)

    # for n in non_ioi1s:
    #     print(output_list[n[0][1]-1], nn)
    #     for nn in n:
    #         print(output_list[nn[1]], nn)
    #         print(output_list_[nn[1]], nn)
    #     print(output_list[n[-1][1]+1], nn)
    #     print()

    # to numpy array
    input_list = np.array(input_list, dtype=object)
    output_list = np.array(output_list_, dtype=object)

    inp = np.asarray([i[1] for i in input_list])
    oup = np.asarray([o[1][0] for o in output_list])

    # rearrange mean onsets
    base_onsets = pdata.make_onset_based_pick(
        inp, np.asarray(base_onset_perform_list), same_onset_ind=same_onset_ind)
    mean_onsets = pdata.make_onset_based_pick(
        inp, np.asarray(mean_onset_perform_list), same_onset_ind=same_onset_ind)
    next_onsets = pdata.make_onset_based_pick(
        inp, np.asarray(next_onset_perform_list), same_onset_ind=same_onset_ind)

    assert np.array_equal(base_onsets[1:], mean_onsets[:-1])
    assert np.array_equal(mean_onsets[1:], next_onsets[:-1])

    return input_list, output_list

def parse_score_features(sub_notes, tempo=None, null_tempo=120):
    # parse features
    input_list = list()
    prev_note = None
    prev_mean_onset = None

    for i in range(len(sub_notes)):
        # assign each pair
        note = sub_notes[i]
        if i < len(sub_notes)-1:
            next_note = sub_notes[i+1]
        elif i == len(sub_notes)-1:
            next_note = None
        # parse features for each note
        parsed_note = MIDIFeatures_test2(note=note,
                                        prev_note=prev_note,
                                        next_note=next_note,
                                        note_ind=i,
                                        tempo_=tempo,
                                        null_tempo=null_tempo)

        if prev_note is None: # first note
            prev_mean_onset = -parsed_note.dur_16th

        if parsed_note.is_same_onset is False: # different note group
            mean_onset = float(parsed_note.score_onset)
            
            # update next mean onset for computing ioi
            next_onset = None
            for j in range(i, len(sub_notes)):
                if sub_notes[j].start > mean_onset:
                    next_onset = sub_notes[j].start
                    break
            if next_onset is None and j == len(sub_notes)-1:
                next_onset = sub_notes[j].end 

            next_onset_score = next_onset
            base_onset_score = prev_mean_onset
            
        elif parsed_note.is_same_onset is True: # same note group
            mean_onset = mean_onset
            next_onset_score = next_onset_score
            base_onset_score = base_onset_score

        # parse input features
        _input = parsed_note.get_input_features(base_onset_score, next_onset_score)
        input_list.append(_input)

        # print(base_onset_score, mean_onset, next_onset_score, parsed_note.is_same_onset)
        
        # update previous attributes
        prev_note = parsed_note # MIDIFeatures object
        prev_mean_onset = mean_onset

    assert len(sub_notes) == len(input_list)
    input_list = np.array(input_list)
    
    return input_list

def parse_score_features_simple(sub_notes):
    # parse features
    input_list = list()
    prev_note = None
    prev_mean_onset = None

    for i in range(len(sub_notes)):
        # assign each pair
        note = sub_notes[i]

        # parse features for each note
        parsed_note = MIDIFeatures_simple(note=note, note_ind=i)

        # parse input features
        _input = parsed_note.get_input_features()
        input_list.append(_input)

    assert len(sub_notes) == len(input_list)
    input_list = np.array(input_list)
    
    return input_list


def parse_test_features(
    xml, score, mode=None, measures=None,
    tempo=None, null_tempo=None, same_onset_ind=None):

    '''
    * function for parsing MIDI features
        - always parse in order of SCORE MIDI 
    '''

    ## FOR DEBUGGING ## 
    # pair_path = '/data/chopin_cleaned/original/Chopin_Etude/10_1/02/xml_score_perform_pairs.npy'
    # midi_path = '/data/chopin_cleaned/original/Chopin_Etude/10_1/score_plain.mid'
    # pairs_score = None 
    # midi_notes = None 
    # tempo = 176

    # pair_path = '/data/chopin_cleaned/original/Chopin_Etude/10_8/03/xml_score_perform_pairs.npy'
    # xml = '/data/chopin_cleaned/original/Chopin_Etude/10_8/musicxml_cleaned_plain.musicxml'
    # pair_path = '/data/chopin_cleaned/original/Chopin_Berceuse/57/02/xml_score_perform_pairs.npy'
    # xml = '/data/chopin_cleaned/original/Chopin_Berceuse/57/musicxml_cleaned_plain.musicxml'
        
    # xml = '/data/pianotab/original/starwars.musicxml'
    # score = '/data/pianotab/original/starwars.mid'

    # tempo, time_sig, key_sig = get_signatures_from_xml(xml, measure_start=0)
    # pairs_score = None 
    # midi_notes = None
    # null_tempo = 120
    # mode = "note" 

    ###################

    match = MATCH(
        current_dir=os.getcwd(),
        program_dir="/workspace/Piano/gen_task/match_score_midi")

    pairs = match.align_xml_midi(
        xml, score, performs=None, corresps=None, 
        save_pairs=None, plain=True, plot=None)

    pairs_score_all = [p for p in pairs if p['xml_note'] is not None and \
        p['xml_note'][1].is_grace_note is False]
    pairs_score_all = sorted(pairs_score_all, key=lambda x: x['xml_note'][0])

    if measures is not None:
        start_measure, end_measure = measures[0]-1, measures[1]-1
        pairs_score = list()
        for note in pairs_score_all:
            measure_num = note['xml_note'][1].measure_number
            if measure_num >= start_measure and measure_num <= end_measure:
                pairs_score.append(note)
        pairs_score = sorted(pairs_score, key=lambda x: x['xml_note'][0])
        # note_ind = [n['score_midi'][0] for n in pairs_score]

    # get first onsets
    pairs_score_onset = make_onset_pairs(pairs_score, fmt="xml")
    first_onset_score = np.min(
        [n['xml_note'][1].note_duration.time_position \
            for n in pairs_score_onset[0]])

    # print("** first perform onset: {:.4f}".format(first_onset_perform))
    # print("** first score onset: {:.4f}".format(first_onset_score))

    # get score midi for check
    # if midi_path is not None:
    #     midi_notes, _ = extract_midi_notes(midi_path, clean=True)
    # if midi_notes is not None:
    #     midi_notes = midi_notes

    # set first onsets to 0 for score and perform
    for note in pairs_score:
        if note['xml_note'] is not None:
            note['xml_note'][1].note_duration.time_position -= first_onset_score

    # parse features per note
    input_list = list()
    note_list = list()
    onset_score_list = list()
    base_onset_list = list()
    mean_onset_list = list()
    next_onset_list = list()
    
    prev_note = None
    prev_mean_onset_score = None

    # make pairs in onset AGAIN --> time changed (- first onset)
    # just for sure
    pairs_onset = make_onset_pairs(pairs_score, fmt="xml")

    for i in range(len(pairs_score)):
        # assign each pair
        note = pairs_score[i]

        if i < len(pairs_score)-1:
            next_note = pairs_score[i+1]
        elif i == len(pairs_score):
            next_note = None
        note_ind = note['xml_note'][0]  

        # parse features for each note
        parsed_note = MIDIFeatures_test(note=note, 
                                        prev_note=prev_note, 
                                        next_note=next_note,
                                        note_ind=note_ind,
                                        tempo_=tempo,
                                        fmt="xml",
                                        null_tempo=120)
                 

        if prev_note is None: # first note
            '''
            * assume the first note starts after 16th-length rest
                -> since the first note also has micro-timing and non-zero onset
            * same prev mean onset for score and perform
            '''
            prev_mean_onset_score = -parsed_note.dur_16th

        if parsed_note.is_same_onset is False: # different note group

            # current onset -> find onset group having current note index
            same_onset = [onset for onset in pairs_onset \
                if note_ind == onset[0]['xml_note'][0]]
            assert len(same_onset) == 1
            same_onset = same_onset[0]
            same_mean_onset_score = parsed_note.score_onset

            '''
            * next note to current group's last note 
                is the first note of the next group
            '''
            same_onset_last_ind = same_onset[-1]['xml_note'][0]
            last_pair_ind = pairs_score[-1]['xml_note'][0]

            if same_onset_last_ind < last_pair_ind:
                next_onset = list()
                n = 1
                while len(next_onset) == 0: # find next note group
                    for onset in pairs_onset:
                        if same_onset_last_ind+n in [o['xml_note'][0] for o in onset]:
                            next_onset.append(onset) 
                    n += 1
                assert len(next_onset) == 1 # next_onset is always length 1
                next_onset = next_onset[0]
                next_mean_onset_score = next_onset[0]['xml_note'][1].note_duration.time_position
            
            elif same_onset_last_ind == last_pair_ind: # if last note group
                next_onset = same_onset
                next_mean_onset_score = np.max([o['xml_note'][1].note_duration.time_position + \
                    o['xml_note'][1].note_duration.seconds \
                    for o in next_onset]) # notes can have different offsets within group

            # get current mean onset for the note group
            mean_onset_score = same_mean_onset_score

            # get previous mean onset for computing ioi (current - prev)
            base_onset_score = prev_mean_onset_score

            # get next mean onset for computing next ioi (next - current)
            next_onset_score = next_mean_onset_score
            
        elif parsed_note.is_same_onset is True: # same note group
            mean_onset_score = mean_onset_score
            base_onset_score = base_onset_score
            next_onset_score = next_onset_score

        # gather onsets and note object
        base_onset_list.append(float(base_onset_score))
        mean_onset_list.append(float(mean_onset_score))
        next_onset_list.append(float(next_onset_score))	
        note_list.append([note_ind, parsed_note])

        # print()
        # print(same_onset_values)
        # print("{},{},{},{},".format(
            # parsed_note.is_same_onset, base_onset_perform, mean_onset_perform, next_onset_perform))

        _input = parsed_note.get_input_features(
            base_onset_score, next_onset_score)


        # print(_output, parsed_note.is_same_onset)
        # if parsed_note.ioi_ratio2 <= 0 or parsed_note.ioi_ratio1 <= 0:
        #     print(base_onset_score, mean_onset_score, next_onset_score, parsed_note.is_same_onset)
        #     print(parsed_note.score_ioi_norm1)
        #     print()
        #     break

        input_list.append([note_ind, _input])
        
        # update previous attributes
        prev_note = parsed_note # MIDIFeatures object
        prev_mean_onset_score = mean_onset_score

    assert len(note_list) == len(input_list)	

    input_list = sorted(input_list, key=lambda x: x[0])
    note_list = sorted(note_list, key=lambda x: x[0])
    inp_ind = sorted([i[0] for i in input_list])
    note_ind = sorted([i[0] for i in note_list])

    # to numpy array
    input_list = np.array(input_list, dtype=object)
    inp = np.asarray([i[1] for i in input_list])

    # rearrange mean onsets
    base_onsets = pdata.make_onset_based_pick(
        inp, np.asarray(base_onset_list), same_onset_ind=same_onset_ind)
    mean_onsets = pdata.make_onset_based_pick(
        inp, np.asarray(mean_onset_list), same_onset_ind=same_onset_ind)
    next_onsets = pdata.make_onset_based_pick(
        inp, np.asarray(next_onset_list), same_onset_ind=same_onset_ind)

    assert np.array_equal(base_onsets[1:], mean_onsets[:-1])
    assert np.array_equal(mean_onsets[1:], next_onsets[:-1])


    return inp, pairs_score, note_ind


def parse_test_x_features(
    score=None, xml=None, tempo=None, null_tempo=120, 
    sec=None, vel=None, ioi=None, measures=None):

    midi_notes_, _ = extract_midi_notes(score, clean=True)
    midi_notes = make_midi_start_zero(midi_notes_)

    # estimate quarter length
    # if quarter is None:
    # 	# get ioi histogram
    # 	dur_dict = dict()
    # 	prev_note = midi_notes[0]
    # 	for note in midi_notes[1:]:
    # 		onset = Decimal(str(note.start))
    # 		prev_onset = Decimal(str(prev_note.start))
    # 		dur = round(onset - prev_onset, 3)
    # 		try:
    # 			dur_dict[dur] += 1
    # 		except KeyError:
    # 			dur_dict[dur] = 1
    # 		prev_note = note
    # 	# sort histrogram by num
    # 	quarter_cand = sorted(dur_dict.items(), 
    # 		key=lambda x: x[1], reverse=True)
    # 	# get most frequent duration
    # 	for q, _ in quarter_cand:
    # 		'''
    # 		upper bound: 200 BPM
    # 		lower bound: 20 BPM
    # 		'''
    # 		if float(q) >= 0.3 and float(q) <= 3.0:
    # 			break
    # 	quarter = q

    # elif quarter is not None:
    # 	quarter = quarter
    
    # get 16th length
    # dur_16th = round(quarter / 4, 3)

    # group into onset
    if sec is not None:
        sub_notes = trim_length(midi_notes, sec=sec)
    else:
        sub_notes = midi_notes

    # parse features
    input_list = list()
    prev_note = None
    prev_mean_onset = None

    for i in range(len(sub_notes)):
        # assign each pair
        note = sub_notes[i]
        if i < len(sub_notes)-1:
            next_note = sub_notes[i+1]
        elif i == len(sub_notes)-1:
            next_note = None
        # parse features for each note
        parsed_note = MIDIFeatures_test(note=note,
                                        prev_note=prev_note,
                                        next_note=next_note,
                                        note_ind=i,
                                        tempo_=tempo,
                                        null_tempo=null_tempo)

        if prev_note is None: # first note
            prev_mean_onset = -parsed_note.dur_16th

        if parsed_note.is_same_onset is False: # different note group
            mean_onset = float(parsed_note.score_onset)
            
            # update next mean onset for computing ioi
            next_onset = None
            for j in range(i, len(sub_notes)):
                if sub_notes[j].start > mean_onset:
                    next_onset = sub_notes[j].start
                    break
            if next_onset is None and j == len(sub_notes)-1:
                next_onset = sub_notes[j].end 

            next_onset_score = next_onset
            base_onset_score = prev_mean_onset
            
        elif parsed_note.is_same_onset is True: # same note group
            mean_onset = mean_onset
            next_onset_score = next_onset_score
            base_onset_score = base_onset_score

        # parse input features
        _input = parsed_note.get_input_features(base_onset_score, next_onset_score)
        input_list.append(_input)

        # print(base_onset_score, mean_onset, next_onset_score, parsed_note.is_same_onset)
        
        # update previous attributes
        prev_note = parsed_note # MIDIFeatures object
        prev_mean_onset = mean_onset

    assert len(sub_notes) == len(input_list)
    input_list = np.array(input_list)

    if measures is not None and sec is None:
        # extract all xml score notes 
        XMLDocument = MusicXMLDocument(xml)
        xml_notes = extract_xml_notes(XMLDocument, apply_grace=False)
        pairs_score_ = match_XML_to_scoreMIDI(xml_notes, midi_notes)
        start_measure, end_measure = measures[0]-1, measures[1]-1
        pairs_score = list()
        for note in pairs_score_:
            measure_num = note['xml_note'][1].measure_number
            if measure_num >= start_measure and measure_num <= end_measure:
                pairs_score.append(note)
        pairs_score = sorted(pairs_score, key=lambda x: x['score_midi'][0])
        note_ind = [n['score_midi'][0] for n in pairs_score]
        input_list = np.asarray([input_list[n] for n in note_ind])
        sub_notes = [sub_notes[n] for n in note_ind]

    if vel is not None and ioi is not None:
        vel_list = np.tile(vel, (len(input_list)))
        ioi_list = np.tile(ioi, (len(input_list)))
        cond = np.stack([vel_list, ioi_list], axis=-1)
        input_list = np.concatenate([input_list, cond], axis=-1)

    return input_list, sub_notes


def parse_test_y_features(
    xml=None, score=None, perform=None, corresp=None, mode=None, 
    pair_path=None, measures=None, tempo=None, null_tempo=120, same_onset_ind=None):

    if type(perform) is not list:
        perform = [perform]
    if corresp is not None:
        if type(corresp) is not list:
            corresp = [corresp]

    # get xml_score_perform_pairs
    if pair_path is None:
        save_dir = os.path.dirname(perform[0])
        match = MATCH(
            current_dir=os.getcwd(), save_dir=save_dir,
            program_dir="/workspace/Piano/gen_task/match_score_midi")
        pairs_, _ = match(
            xml, score, perform, corresps=corresp, plot=True, save_pairs=False)
        pairs = [p for k, p in pairs_.items()][0]
    elif pair_path is not None:
        pairs = np.load(pair_path, allow_pickle=True).tolist()

    # collect pairs only with non-grace notes
    # sort by the order of xml_note
    pairs_score_all = [p for p in pairs if p['xml_note'] is not None and \
        p['xml_note'][1].is_grace_note is False]
    pairs_score_all = sorted(pairs_score_all, key=lambda x: x['xml_note'][0])
    mid_note_ind = [n['score_midi'][0] for n in pairs_score_all]
    xml_note_ind = [n['xml_note'][0] for n in pairs_score_all]

    if measures is not None:
        start_measure, end_measure = measures[0]-1, measures[1]-1
        pairs_score = list()
        for note in pairs_score_all:
            measure_num = note['xml_note'][1].measure_number
            if measure_num >= start_measure and measure_num <= end_measure:
                pairs_score.append(note)
        pairs_score = sorted(pairs_score, key=lambda x: x['xml_note'][0])

    elif measures is None:
        pairs_score = pairs_score_all

    # parse output features
    input_list_, output_list_ = parse_midi_features(
        mode=mode, pairs_score=pairs_score,
        tempo=tempo, null_tempo=null_tempo, same_onset_ind=same_onset_ind)

    for i, o in zip(input_list_, output_list_):
        assert i[0] == o[0]

    input_list = np.asarray([o[1] for o in input_list_])
    output_list = np.asarray([o[1] for o in output_list_])
    ind_list = np.asarray([o[0] for o in output_list_])
    
    # output_list = output_list[:len(x_notes)]
    assert len(input_list) == len(output_list) == len(pairs_score)

    return output_list, input_list, pairs_score, ind_list




#----------------------------------------------------------------------------------#

# Class for parsing MIDI features(input/output)
class MIDIFeatures(object):
    def __init__(self, 
                 note=None, 
                 prev_note=None,
                 next_note=None,
                 note_ind=None,
                 tempo_=None,
                 fmt="xml",
                 mode_ioi1=None,
                 mode_ioi2=None,
                 null_tempo=120):

        # Inputs
        self.format = fmt
        self.note = note
        if fmt == "xml":
            self.score_note = note['xml_note'][1]
            self.next_note = next_note['xml_note'][1]
        elif fmt == "mid":
            self.score_note = note['score_midi'][1]
            self.next_note = next_note['score_midi'][1]
        if note['perform_midi'] is not None:
            self.perform_note = note['perform_midi'][1] 
        else:
            self.perform_note = None
        # self.xml_note = note['xml_note'][1]
        assert self.note['xml_note'][1].is_grace_note is False

        self.directions = note['xml_measure'].directions
        self.prev_note = prev_note # parsed_note object
        self.note_ind = note_ind

        # Features to parse
        self.prev_velocity = None
        self.prev_score_note = None
        self.is_same_onset = None
        self.is_top = None
        self.base_offset = None
        self.base_pitch = None
        self.xml_onset = None
        self.score_onset = None
        self.score_offset = None
        self.score_dur = None
        self.score_ioi1 = None
        self.score_ioi2 = None
        self.ioi_units = None
        self.dur_units = None
        self.perform_onset = None
        self.perform_offset = None
        self.perform_dur = None
        self.perform_ioi1 = None
        self.perform_ioi2 = None
        self.local_dev = None 
        self.tempo = tempo_
        self.null_tempo = null_tempo
        self.tempo_ratio = None #Decimal(1.)
        self.dur_16th = None
        self.dur_32th = None
        self.ioi_class = None
        self.dur_class = None
        self.velocity = 0
        self.dur_ratio = 0
        self.ioi_ratio1 = 0
        self.ioi_ratio2 = 0
        self.base_onset_score = None 
        self.base_onset_perform = None
        self.mean_onset = None 
        self.next_onset_score = None 
        self.next_onset_perform = None
        self._input = None
        self._output = None

        # Functions		
        if self.prev_note is not None:
            self.get_prev_attributes()
        self.get_tempo()
        self.get_score_attributes() # onset/offset/dur
        self.get_perform_attributes() # onset/offset/dur


    def get_prev_attributes(self):
        pass

    def get_input_features(
        self, onset_for_ioi, next_onset_for_ioi):
        '''
        inputs based on score attributes 
        '''
        self.get_score_ioi(onset_for_ioi, next_onset_for_ioi)
        self.get_score_duration()
        self.get_ioi_class_16()
        self.get_dur_class_16()
        self.get_pitch()
        self.get_top_voice()
        self.input_to_vector_16()
        return self._input	

    def get_output_features(
        self, onset_for_ioi, mean_onset, next_onset_for_ioi, mode=None): #, pooled_ioi
        '''
        outputs based on performed attributes
        '''
        self.get_velocity() # velocity value v
        if mode == 'note':
            self.get_ioi_ratio_note(onset_for_ioi, next_onset_for_ioi) # ioi ratio
        elif mode == 'group':
            self.get_ioi_ratio(onset_for_ioi, mean_onset, next_onset_for_ioi) # ioi ratio v
            self.get_local_dev_raw(mean_onset)
        self.get_duration_ratio() # duration ratio v
        
        # self.get_local_dev_ratio(mean_onset)
        
        self.output_to_vector(mode=mode)
        return self._output

    def input_to_vector_16(self):
        _score_ioi = np.zeros([11,])
        _score_ioi[self.ioi_class] = 1
        _score_dur = np.zeros([11,])
        _score_dur[self.dur_class] = 1
        _pitch = np.zeros([88,])
        _pitch[self.pitch] = 1
        _same_onset = np.zeros([2,])
        _same_onset[int(self.is_same_onset)] = 1
        _top = np.zeros([2,])
        _top[int(self.is_top)] = 1

        assert np.sum(_score_ioi) == 1
        assert np.sum(_score_dur) == 1
        assert np.sum(_pitch) == 1
        assert np.sum(_same_onset) == 1
        assert np.sum(_top) == 1

        self._input = np.concatenate(
            [_score_ioi, _score_dur, _pitch, _same_onset, _top], axis=-1)

    def input_to_vector_48(self):
        _score_ioi = np.zeros([50,])
        _score_ioi[self.ioi_class] = 1
        _score_dur = np.zeros([50,])
        _score_dur[self.dur_class] = 1
        _pitch = np.zeros([88,])
        _pitch[self.pitch] = 1
        _same_onset = np.zeros([2,])
        _same_onset[int(self.is_same_onset)] = 1
        _top = np.zeros([2,])
        _top[int(self.is_top)] = 1

        assert np.sum(_score_ioi) == 1
        assert np.sum(_score_dur) == 1
        assert np.sum(_pitch) == 1
        assert np.sum(_same_onset) == 1
        assert np.sum(_top) == 1

        self._input = np.concatenate(
            [_score_ioi, _score_dur, _pitch, _same_onset, _top], axis=-1)

    def output_to_vector(self, mode=None):
        _velocity = self.velocity
        if mode == "group":
            _local_dev = float(self.local_dev)
        _dur_ratio = float(self.dur_ratio)
        _ioi_ratio1 = float(self.ioi_ratio1)
        _ioi_ratio2 = float(self.ioi_ratio2)
        
        if mode == "note":
            self._output = np.stack(
                [_velocity, _dur_ratio, _ioi_ratio1, _ioi_ratio2], axis=-1) # note-level
        elif mode == "group":
            self._output = np.stack(
                [_velocity, _local_dev, _dur_ratio, _ioi_ratio1, _ioi_ratio2], axis=-1)

    ## FUNCTIONS FOR FEATURES ##
    def get_tempo(self):
        if self.tempo is None:
            if len(self.directions) > 0:
                for direction in self.directions:
                    if direction.tempo is not None:
                        self.tempo = direction.tempo
                        break

            if self.tempo is None: # nothing in directions
                if self.prev_note is not None:
                    self.tempo = self.prev_note.tempo
                elif self.prev_note is None:
                    raise AssertionError
            elif self.tempo is not None: # tempo in directions
                pass

        elif self.tempo is not None:
            self.tempo = self.tempo

        self.tempo = Decimal(self.tempo)
        self.dur_quarter = round(Decimal(60) / self.tempo, 3) # BPM
        self.dur_16th = self.dur_quarter / Decimal(4)
        self.dur_32th = self.dur_16th / Decimal(2)
        self.dur_48th = self.dur_quarter / Decimal(12) # 12 for one quarter
        self.tempo_ratio = self.tempo / Decimal(self.null_tempo)

    def get_score_attributes(self):
        if self.format == "xml":
            self.score_onset = Decimal(str(self.score_note.note_duration.time_position))
            self.score_offset = Decimal(str(self.score_note.note_duration.time_position + \
                self.score_note.note_duration.seconds))
            if self.next_note is not None:
                self.next_onset = Decimal(str(self.next_note.note_duration.time_position))
            else:
                self.next_onset = None

        elif self.format == "mid":
            self.score_onset = Decimal(str(self.score_note.start))
            self.score_offset = Decimal(str(self.score_note.end))  
            if self.next_note is not None:
                self.next_onset = Decimal(str(self.next_note.start))
            else:
                self.next_onset = None

        self.score_dur = self.score_offset - self.score_onset
        # self.score_dur = round(self.score_dur, 3)
        assert self.score_dur > 0

        # # get note onset in xml 
        # if self.xml_note is not None:
        #     self.xml_onset = Decimal(str(self.xml_note.note_duration.time_position))
        # elif self.xml_note is None:
        #     self.xml_onset = self.score_onset    

        # # get previous note onset in xml
        # if self.prev_xml_note is not None:
        #     self.prev_xml_onset = Decimal(str(self.prev_xml_note.note_duration.time_position))
        # elif self.prev_xml_note is None:
        #     self.prev_xml_onset = None

        # DECIDE is_same_onset
        if self.prev_note is None:
            self.is_same_onset = False

        elif self.prev_note is not None:
            '''
            If XML and MIDI have different note order, 
            possibly it is because the corresponding note 
            follows the grace notes in original score!
            '''
            # if score onset larger than previous note
            if self.score_onset > self.prev_note.score_onset: 
                self.is_same_onset = False # decide only with midi

                # if self.xml_note is None: # no xml data 
                #     self.is_same_onset = False # decide only with midi
                # elif self.xml_note is not None: # with xml data
                #     if self.prev_xml_onset is not None:
                #         if self.xml_onset > self.prev_xml_onset:
                #             self.is_same_onset = False
                #             # print(self.note, self.prev_note.note) 
                #         elif self.xml_onset == self.prev_xml_onset:
                #             self.is_same_onset = True
                #     elif self.prev_xml_onset is None:
                #         self.is_same_onset = False
            
            # if score onset same as previous note
            elif self.score_onset == self.prev_note.score_onset:
                self.is_same_onset = True

                # if self.xml_note is None: # no xml data 
                #     self.is_same_onset = True # decide only with midi
                # elif self.xml_note is not None: # with xml data
                #     if self.prev_xml_onset is not None:
                #         if self.xml_onset > self.prev_xml_onset:
                #             self.is_same_onset = False
                #             # print(self.note, self.prev_note.note) 
                #         elif self.xml_onset == self.prev_xml_onset:
                #             self.is_same_onset = True
                #     elif self.prev_xml_onset is None:
                #         self.is_same_onset = False                
        
        assert self.is_same_onset is not None
        # print(self.note)
        # if self.score_onset is not None and self.prev_note is not None:
            # print("{}, {} / {}, {}".format(
                # self.score_onset, self.prev_note.score_onset, self.xml_onset, self.prev_xml_onset))
        # print(self.is_same_onset)
    
    def get_perform_attributes(self):
        if self.perform_note is None:
            self.perform_onset = None
            self.perform_offset = None
            self.perform_dur = None

        elif self.perform_note is not None:
            self.perform_onset = Decimal(str(self.perform_note.start))
            self.perform_offset = Decimal(str(self.perform_note.end))
            self.perform_dur = self.perform_offset - self.perform_onset
            assert self.perform_dur > 0	

    def get_top_voice(self):
        if self.prev_note is None:
            if self.next_onset is not None:
                if self.score_onset == self.next_onset: # note presents above
                    self.is_top = False
                elif self.score_onset < self.next_onset: 
                    self.is_top = True
                elif self.score_onset > self.next_onset:
                    raise AssertionError("invalid onset! (score_onset > next_onset)")
            elif self.next_onset is None:
                raise AssertionError("** first note doesn't have next note!")
            self.base_offset = self.score_offset
            self.base_pitch = self.pitch
        
        elif self.prev_note is not None:
            if self.next_onset is not None:
                if self.score_onset == self.next_onset: # note presents above
                    self.is_top = False
                elif self.score_onset < self.next_onset: # maybe top?
                    if self.score_onset < self.prev_note.base_offset: # note in the middle
                        if self.pitch < self.prev_note.base_pitch: # top note is long
                            self.is_top = False
                        elif self.pitch >= self.prev_note.base_pitch: # bottom note is long
                            self.is_top = True
                    elif self.score_onset >= self.prev_note.base_offset: # note starts after prev note ends
                        self.is_top = True
                elif self.score_onset > self.next_onset:
                    raise AssertionError("invalid onset! (score_onset > next_onset)")
            
            elif self.next_onset is None:
                self.is_top = True 

            if self.is_top is False:
                self.base_offset = self.prev_note.base_offset
                self.base_pitch = self.prev_note.base_pitch
            elif self.is_top is True:
                self.base_offset = self.score_offset
                self.base_pitch = self.pitch
            else: 
                raise AssertionError("is_top not identified!")

    def get_ioi_class_16(self):
        '''
        score midi duration based on unit of 16th note
        '''
        self.ioi_q = quantize(self.score_ioi1, self.dur_32th)
        self.ioi_units = Decimal(str(self.ioi_q)) / self.dur_16th
        self.ioi_units = float(round(self.ioi_units, 1))
        # assign to certain class
        if self.ioi_units < 1.0:
            self.ioi_class = 0 # shorter
        elif self.ioi_units == 1.0:
            self.ioi_class = 1 # 16th
        elif self.ioi_units > 1.0 and self.ioi_units < 2.0:
            self.ioi_class = 2 # 16th~8th
        elif self.ioi_units == 2.0:
            self.ioi_class = 3 # 8th
        elif self.ioi_units > 2.0 and self.ioi_units < 4.0:
            self.ioi_class = 4 # 8th~4th
        elif self.ioi_units == 4.0:
            self.ioi_class = 5 # quarter(4th)
        elif self.ioi_units > 4.0 and self.ioi_units < 8.0:
            self.ioi_class = 6 # 4th~half
        elif self.ioi_units == 8.0:
            self.ioi_class = 7 # half
        elif self.ioi_units > 8.0 and self.ioi_units < 16.0:
            self.ioi_class = 8 # half~whole
        elif self.ioi_units == 16.0:
            self.ioi_class = 9 # whole
        elif self.ioi_units > 16.0:
            self.ioi_class = 10 # longer

    def get_dur_class_16(self):
        '''
        score midi duration based on unit of 16th note
        '''
        self.dur_q = quantize(self.score_dur, self.dur_32th)
        self.dur_units = Decimal(str(self.dur_q)) / self.dur_16th
        self.dur_units = float(round(self.dur_units, 1))
        # assign to certain class
        if self.dur_units < 1.0:
            self.dur_class = 0 # shorter
        elif self.dur_units == 1.0:
            self.dur_class = 1 # 16th
        elif self.dur_units > 1.0 and self.dur_units < 2.0:
            self.dur_class = 2 # 16th~8th
        elif self.dur_units == 2.0:
            self.dur_class = 3 # 8th
        elif self.dur_units > 2.0 and self.dur_units < 4.0:
            self.dur_class = 4 # 8th~4th
        elif self.dur_units == 4.0:
            self.dur_class = 5 # quarter(4th)
        elif self.dur_units > 4.0 and self.dur_units < 8.0:
            self.dur_class = 6 # 4th~half
        elif self.dur_units == 8.0:
            self.dur_class = 7 # half
        elif self.dur_units > 8.0 and self.dur_units < 16.0:
            self.dur_class = 8 # half~whole
        elif self.dur_units == 16.0:
            self.dur_class = 9 # whole
        elif self.dur_units > 16.0:
            self.dur_class = 10 # longer

    def get_ioi_class_48(self):
    	'''
    	score midi duration based on unit of 48th note
    	'''
    	self.ioi_q = quantize(self.score_ioi1, self.dur_48th)
    	self.ioi_units = round(Decimal(str(self.ioi_q)) / self.dur_48th)
    	self.ioi_class = min(self.ioi_units, 49)

    def get_dur_class_48(self):
    	'''
    	score midi duration based on unit of 48th note
    	'''
    	self.dur_q = quantize(self.score_dur, self.dur_48th)
    	self.dur_units = round(Decimal(str(self.dur_q)) / self.dur_48th)
    	self.dur_class = min(self.dur_units, 49) 

    def get_score_ioi(self, onset_for_ioi, next_onset_for_ioi):
        '''
        score midi ioi based on null tempo
        ''' 
        if onset_for_ioi is not None:
            self.base_onset_score = Decimal(str(onset_for_ioi))
        elif onset_for_ioi is None:
            self.base_onset_score = None

        if next_onset_for_ioi is not None:
            self.next_onset_score = Decimal(str(next_onset_for_ioi))
        elif next_onset_for_ioi is None:
            self.next_onset_score = None

        if self.is_same_onset == False:
            self.score_ioi1 = self.score_onset - self.base_onset_score
            self.score_ioi2 = self.next_onset_score - self.score_onset
        elif self.is_same_onset == True:
            self.score_ioi1 = self.prev_note.score_ioi1
            self.score_ioi2 = self.prev_note.score_ioi2

        if self.score_ioi1 <= 0. or self.score_ioi2 <= 0.:
            print("** score ioi ratio <= 0 !!")
            print(self.note_ind, self.score_ioi1, self.base_onset_score, self.score_onset)
            print(self.note_ind, self.score_ioi2, self.score_onset, self.next_onset_score)

        self.score_ioi_norm1 = self.score_ioi1 * self.tempo_ratio
        self.score_ioi_norm2 = self.score_ioi2 * self.tempo_ratio

    def get_score_duration(self):
        '''
        score midi duration based on null tempo
        '''
        self.score_dur_norm = self.score_dur * self.tempo_ratio

    def get_pitch(self):
        '''
        pitch of score midi
        '''
        if self.format == "xml":
            midi_num = self.score_note.pitch[1]
        elif self.format == "mid":
            midi_num = self.score_note.pitch
        self.pitch = midi_num - 21
        self.pitch_class = np.mod(midi_num, 12) # pitch class
        self.octave = int(midi_num / 12) - 1 # octave

    def get_velocity(self):
        '''
        velocity value: 0~127 / 127
        0: off / 1-16: ppp / 17-32: pp / 33-48: p / 49-64: mp 
        65-80: mf / 81-96: f / 97-112: ff / 113-127: fff
        '''				
        if self.perform_note is not None:
            self.velocity = self.perform_note.velocity

        elif self.perform_note is None:
            if self.prev_note is None: # first note
                self.velocity = 64
            elif self.prev_note is not None:
                self.velocity = self.prev_note.velocity

        # self.q_vel = int(self.velocity // 4) # quantize to 32 classes

    def get_ioi_ratio(self, onset_for_ioi, mean_onset, next_onset_for_ioi):
        '''
        ratio of perform midi ioi / score midi(norm) ioi 
        '''
        # previous onset for computing ioi
        if onset_for_ioi is not None:
            self.base_onset_perform = Decimal(str(onset_for_ioi))
        elif onset_for_ioi is None:
            self.base_onset_perform = None

        # next onset for computing ioi
        if next_onset_for_ioi is not None:
            self.next_onset_perform = Decimal(str(next_onset_for_ioi))
        elif next_onset_for_ioi is None:
            self.next_onset_perform = None

        # current mean onset
        if mean_onset is not None:
            self.mean_onset = Decimal(str(mean_onset))
        elif mean_onset is None:
            self.mean_onset = None		

        # compute IOI 1 (current(chord) - prev(chord))
        if self.base_onset_perform is not None:
            if self.mean_onset is not None:
                self.perform_ioi1 = self.mean_onset - self.base_onset_perform
                # ioi value is not 0 
                if self.perform_ioi1 < Decimal(str(1e-3)):
                    self.perform_ioi1 = Decimal(str(1e-3))
                # compute ioi ratio
                self.ioi_ratio1 = self.perform_ioi1 / self.score_ioi_norm1

            elif self.mean_onset is None:
                if self.prev_note is None:
                    self.ioi_ratio1 = 1 / self.tempo_ratio
                elif self.prev_note is not None:
                    self.ioi_ratio1 = self.prev_note.ioi_ratio1

        elif self.base_onset_perform is None:
            if self.prev_note is None:
                self.ioi_ratio1 = 1 / self.tempo_ratio # (null tempo / real tempo)
            elif self.prev_note is not None:
                self.ioi_ratio1 = self.prev_note.ioi_ratio1 

        # compute IOI 2 (next(chord) - current("note"))
        if self.next_onset_perform is not None:
            if self.perform_onset is not None:
                self.perform_ioi2 = self.next_onset_perform - self.perform_onset
                # ioi value is not 0 
                if self.perform_ioi2 < Decimal(str(1e-3)):
                    self.perform_ioi2 = Decimal(str(1e-3))
                # compute ioi ratio
                self.ioi_ratio2 = self.perform_ioi2 / self.score_ioi_norm2

            elif self.perform_onset is None:
                if self.prev_note is None:
                    self.ioi_ratio2 = 1 / self.tempo_ratio
                elif self.prev_note is not None:
                    self.ioi_ratio2 = self.prev_note.ioi_ratio2

        elif self.next_onset_perform is None:
            if self.prev_note is None:
                self.ioi_ratio2 = 1 / self.tempo_ratio # (null tempo / real tempo)
            elif self.prev_note is not None:
                self.ioi_ratio2 = self.prev_note.ioi_ratio2 

        # self.ioi_ratio = round(self.ioi_ratio, 4)
        if self.ioi_ratio1 <= 0 or self.ioi_ratio2 <= 0.:
            print("** perform ioi ratio <= 0 !!")
            print(self.note_ind, self.ioi_ratio1, self.base_onset_perform, self.mean_onset)
            print(self.note_ind, self.ioi_ratio2, self.mean_onset, self.next_onset_perform)

    def get_ioi_ratio_note(self, onset_for_ioi, next_onset_for_ioi):
        '''
        ratio of perform midi ioi / score midi(norm) ioi 
        '''
        # previous onset for computing ioi
        if onset_for_ioi is not None:
            self.base_onset_perform = Decimal(str(onset_for_ioi))
        elif onset_for_ioi is None:
            self.base_onset_perform = None

        # next onset for computing ioi
        if next_onset_for_ioi is not None:
            self.next_onset_perform = Decimal(str(next_onset_for_ioi))
        elif next_onset_for_ioi is None:
            self.next_onset_perform = None

        # compute IOI 1 (current - prev)
        if self.base_onset_perform is not None:
            if self.perform_onset is not None:
                self.perform_ioi1 = self.perform_onset - self.base_onset_perform
                self.perform_ioi1 = np.max([self.perform_ioi1, Decimal(str(1e-3))]) #self.score_ioi_norm1 / 8
                self.ioi_ratio1 = self.perform_ioi1 / self.score_ioi_norm1 # compute ioi ratio

            elif self.perform_onset is None:
                if self.is_same_onset is False:
                    if self.prev_note is None:
                        self.ioi_ratio1 = 1 / self.tempo_ratio 
                    elif self.prev_note is not None:
                        self.ioi_ratio1 = 99
                elif self.is_same_onset is True:
                    self.ioi_ratio1 = self.prev_note.ioi_ratio1

        elif self.base_onset_perform is None:
            if self.is_same_onset is False:
                if self.prev_note is None:
                    self.ioi_ratio1 = 1 / self.tempo_ratio 
                elif self.prev_note is not None:
                    self.ioi_ratio1 = 99
            elif self.is_same_onset is True:
                self.ioi_ratio1 = self.prev_note.ioi_ratio1 

        # compute IOI 2 (next - current)
        if self.next_onset_perform is not None:
            if self.perform_onset is not None:
                self.perform_ioi2 = self.next_onset_perform - self.perform_onset
                self.perform_ioi2 = np.max([self.perform_ioi2, Decimal(str(1e-3))]) #self.score_ioi_norm2 / 8
                self.ioi_ratio2 = self.perform_ioi2 / self.score_ioi_norm2

            elif self.perform_onset is None:
                if self.is_same_onset is False:
                    if self.prev_note is None:
                        self.ioi_ratio2 = 1 / self.tempo_ratio 
                    elif self.prev_note is not None:
                        self.ioi_ratio2 = 99
                elif self.is_same_onset is True:
                    self.ioi_ratio2 = self.prev_note.ioi_ratio2

        elif self.next_onset_perform is None:
            if self.is_same_onset is False:
                if self.prev_note is None:
                    self.ioi_ratio2 = 1 / self.tempo_ratio
                elif self.prev_note is not None:
                    self.ioi_ratio2 = 99
            elif self.is_same_onset is True:
            # elif self.prev_note is not None:
                self.ioi_ratio2 = self.prev_note.ioi_ratio2 

        # self.ioi_ratio = round(self.ioi_ratio, 4)
        if self.ioi_ratio1 <= 0. or self.ioi_ratio2 <= 0.:
            print("** perform ioi ratio <= 0 !!")
            print(self.note_ind, self.ioi_ratio1, self.score_ioi_norm1, self.base_onset_perform, self.perform_onset)
            print(self.note_ind, self.ioi_ratio2, self.score_ioi_norm2, self.perform_onset, self.next_onset_perform)

    def get_local_dev(self, pooled_ioi):
        '''
        ratio of perform midi ioi / score midi(norm) ioi 
        '''
        # get perform ioi and compute ioi ratio

        self.pooled_ioi_ratio = Decimal(str(pooled_ioi))

        if self.perform_ioi1 is not None:
            self.base_ioi = self.perform_ioi1 * (self.pooled_ioi_ratio / self.ioi_ratio1)
            self.in_tempo_onset = self.base_onset_perform + self.base_ioi

            if self.perform_note is not None:
                self.local_dev = self.perform_onset - self.in_tempo_onset

            elif self.perform_note is None:
                if self.is_same_onset is True:
                    self.local_dev = self.prev_note.local_dev 
                elif self.is_same_onset is False:
                    self.local_dev = Decimal(str(0.))

        elif self.perform_ioi1 is None:
            if self.is_same_onset is True:
                self.local_dev = self.prev_note.local_dev 
            elif self.is_same_onset is False:
                self.local_dev = Decimal(str(0.))

    def get_local_dev_ratio2(self, pooled_ioi):
        '''
        ratio of perform midi ioi / score midi(norm) ioi 
        '''
        # get perform ioi and compute ioi ratio

        self.pooled_ioi_ratio = Decimal(str(pooled_ioi))

        if self.perform_ioi1 is not None:
            self.base_ioi = self.pooled_ioi_ratio * self.score_ioi_norm1
            self.in_tempo_onset = self.base_onset_perform + self.base_ioi

            if self.perform_note is not None:
                self.local_dev_time = self.perform_onset - self.in_tempo_onset
                self.local_dev_ratio = self.local_dev_time / self.base_ioi

            elif self.perform_note is None:
                if self.is_same_onset is True:
                    self.local_dev_ratio = self.prev_note.local_dev_ratio 
                elif self.is_same_onset is False:
                    self.local_dev_ratio = Decimal(str(0.))

        elif self.perform_ioi1 is None:
            if self.is_same_onset is True:
                self.local_dev_ratio = self.prev_note.local_dev_ratio 
            elif self.is_same_onset is False:
                self.local_dev_ratio = Decimal(str(0.))

    def get_local_dev_raw(self, mean_onset):
        '''
        ratio of perform midi ioi / score midi(norm) ioi 
        '''
        # get perform ioi and compute ioi ratio
        if mean_onset is not None:
            self.mean_onset = Decimal(str(mean_onset))
        elif mean_onset is None:
            self.mean_onset = None

        if self.mean_onset is not None:
            if self.perform_note is not None:
                self.local_dev = self.perform_onset - self.mean_onset

            elif self.perform_note is None:
                self.local_dev = Decimal(str(0.))
                # if self.prev_note is None:
                # 	self.local_dev = Decimal(str(0.))
                # elif self.prev_note is not None:
                # 	self.local_dev = self.prev_note.local_dev

        elif self.mean_onset is None:
            self.local_dev = Decimal(str(0.))
            # if self.prev_note is None:
            # 	self.local_dev = Decimal(str(0.))
            # elif self.prev_note is not None:
            # 	self.local_dev = self.prev_note.local_dev 

        # self.local_dev = round(self.local_dev, 6)

    def get_local_dev_ratio(self, mean_onset):
        '''
        ratio of perform midi ioi / score midi(norm) ioi 
        '''
        # get perform ioi and compute ioi ratio
        if mean_onset is not None:
            self.mean_onset = Decimal(str(mean_onset))
        elif mean_onset is None:
            self.mean_onset = None

        if self.mean_onset is not None:
            if self.perform_note is not None:
                self.local_dev = self.perform_onset - self.mean_onset
                self.local_dev_ratio = self.local_dev / self.dur_quarter

            elif self.perform_note is None:
                self.local_dev_ratio = Decimal(str(0.))

        elif self.mean_onset is None:
            self.local_dev_ratio = Decimal(str(0.))

    def get_duration_ratio(self):
        '''
        ratio of perform midi duration / score midi duration 
        '''
        if self.perform_dur is not None:
            self.dur_ratio = self.perform_dur / self.score_dur_norm
        
        elif self.perform_dur is None:
            if self.is_same_onset is False:
                if self.prev_note is None:
                    self.dur_ratio = 1 / self.tempo_ratio
                elif self.prev_note is not None:
                    self.dur_ratio = 99
            elif self.is_same_onset is True:
                    self.dur_ratio = self.prev_note.dur_ratio

        # self.dur_ratio = round(self.dur_ratio, 4)
        if self.dur_ratio <= 0:
            print(self.dur_ratio, self.perform_dur, self.score_dur_norm)
            raise AssertionError


class MIDIFeatures_test(MIDIFeatures):
    def __init__(self, 
                 note=None, 
                 prev_note=None, 
                 next_note=None,
                 note_ind=None,
                 tempo_=None,
                 fmt="xml",
                 null_tempo=120):

        # Inputs
        self.format = fmt
        self.note = note
        self.score_note = note
        self.prev_note = prev_note
        self.next_note = next_note
        self.note_ind = note_ind
        self.tempo = tempo_

        if fmt == "xml":
            self.score_note = note['xml_note'][1]
            self.next_note = next_note['xml_note'][1]
        elif fmt == "mid":
            self.score_note = note['score_midi'][1]
            self.next_note = next_note['score_midi'][1]
        # self.xml_note = note['xml_note'][1]
        assert self.note['xml_note'][1].is_grace_note is False

        # Features to parse
        self.prev_score_note = None
        self.is_same_onset = None
        self.is_top = None
        self.xml_onset = None
        self.score_onset = None
        self.score_offset = None
        self.score_dur = None
        self.score_ioi = None
        self.ioi_units = None
        self.dur_units = None
        self.ioi_class = None
        self.dur_class = None
        self.base_onset_score = None 
        self.null_tempo = null_tempo
        self.dur_16th = None
        self.dur_32th = None
        self.dur_48th = None
        self.tempo_ratio = None
        self._input = None

        # Functions		
        if self.prev_note is not None:
            self.get_prev_attributes()
        self.get_tempo_attributes()
        super().get_score_attributes() # onset/offset/dur

    def get_prev_attributes(self):
        pass

    def get_tempo_attributes(self):
        self.tempo = Decimal(self.tempo)
        self.dur_quarter = round(Decimal(60) / self.tempo, 3) # BPM
        self.dur_16th = self.dur_quarter / Decimal(4)
        self.dur_32th = self.dur_16th / Decimal(2)
        self.dur_48th = self.dur_quarter / Decimal(12) # 12 for one quarter
        self.tempo_ratio = self.tempo / Decimal(self.null_tempo)

    def get_input_features(self, onset_for_ioi, next_onset_for_ioi):
        super().get_score_ioi(onset_for_ioi, next_onset_for_ioi)
        super().get_score_duration()
        super().get_ioi_class_16()
        super().get_dur_class_16()
        super().get_pitch()
        super().get_top_voice()
        super().input_to_vector_16()
        return self._input	


class MIDIFeatures_simple(MIDIFeatures):
    def __init__(self, 
                 note=None,
                 note_ind=None):

        # Inputs
        self.note = note
        self.score_note = note
        self.note_ind = note_ind

        # Features to parse
        self.pitch = None
        self.score_onset = None
        self.score_offset = None
        self._input = None

    def get_input_features(self):
        self.score_onset = self.score_note.start
        self.score_offset = self.score_note.end
        self.pitch = self.score_note.pitch
        self._input = np.stack(
            [self.score_onset, self.score_offset, self.pitch], axis=-1)
        return self._input	


# Class for parsing XML features(cond)
class XMLFeatures(object):
    def __init__(self, 
                 note=None, 
                 measure=None,
                 prev_note=None,
                 prev_measure=None,
                 note_ind=None,
                 tempo=None,
                 time_sig=None,
                 key_sig=None):

        # Inputs
        self.note = note # xml note
        self.measure = measure # xml measure
        self.prev_note = prev_note
        self.prev_measure = prev_measure
        self.note_ind = note_ind
        self.tempo = tempo
        self.time_sig = time_sig
        self.key_sig = key_sig
        # self.prev_wedge_in_middle = prev_wedge_in_middle
        # self.cresc_in_word = cresc_in_word # on-going cresc word 
        # self.dim_in_word = dim_in_word # on-going dim word 
        # self.cresc_in_wedge = cresc_in_wedge # on-going cresc wedges 
        # self.dim_in_wedge = dim_in_wedge # on-going wedges

        # Initialize features to parse
        # self._type = {'shorter': None, '16th': None, 'eighth': None,
                    #   'quarter': None, 'half': None, 'longer': None}
        # self.is_dot = None
        # self.voice = None
        self.dynamics = {'pp': None, 'p': None, 'mp': None,
                         'ff': None, 'f': None, 'mf': None}
        # self.global_wedge = {'none': None, 'cresc': None, 'dim': None} 
        # self.local_wedge = {'none': None, 'cresc': None, 'dim': None} 
        # self.wedge = {'cresc_start': None, 'cresc_end': None, 
                    #   'dim_start': None, 'dim_end': None, 'none': None} 
        # self.cresc_word_start = [False, None, None, None]
        # self.cresc_wedge_start = [False, None, None, None]
        # self.dim_word_start = [False, None, None, None]
        # self.dim_wedge_start = [False, None, None, None]
        # self.wedge_stop = list()
        # self.wedge_in_middle = None
        self.beat = None
        self.is_downbeat = None
        self.is_grace_note = None
        self.time_num = None 
        self.time_denom = None
        # self.accent = {'none': None, 'accent': None, 'strong_accent': None}
        # self.staccato = {'none': None, 'staccato': None, 'strong_staccato': None}
        # self.is_arpeggiate = None
        # self.is_trill = None
        # self.tempo = {'rit': None, 'accel': None, 'a_tempo': None}
        self.same_onset = {'start': None, 'cont': None}
        # self.ornament = {'none': None, 'trill': None, 'mordent': None, 'wavy': None}
        # self.tuplet = {'none': None, 'start': None, 'stop': None} 
        # self.is_tied = None
        self.pitch_name = None
        self.pitch = None
        self.pitch_norm = None
        self.mode = None
        self.key_final = None
        self.pitch_class = None
        self.octave = None
        self.ioi = None
        self._input = None

        # initialize previous note attributes
        self.prev_measure_number = None
        self.prev_time_position = None
        self.prev_dynamics = None
        self.prev_next_dynamics = None
        self.prev_downbeat = None
        self.prev_stem = None
        self.prev_staff = None
        # self.prev_is_global = None
        # self.prev_is_local = None
        self.prev_beat = None

        # initialize current note attributes
        self.time_position = None
        self.xml_position = None
        self.time_signature = None
        self.measure_number = None
        self.measure_num_base = None
        self.x_position = None
        self.y_position = None
        self.stem = None
        self.staff = None
        self.current_directions = list()
        self.is_new_dynamic = None
        self.next_dynamics = None
        self.in_measure_pos = 0
        self.measure_duration = None
        
        # self.is_cresc_global = False
        # self.is_cresc_global_word = False
        # self.is_dim_global = False
        # self.is_dim_global_word = False
        # self.is_cresc_local_1 = False
        # self.is_cresc_local_word_1 = False
        # self.is_dim_local_1 = False
        # self.is_dim_local_word_1 = False
        # self.is_cresc_local_2 = False
        # self.is_cresc_local_word_2 = False
        # self.is_dim_local_2 = False
        # self.is_dim_local_word_2 = False

        # get attributes 
        if self.prev_note is not None:
            self.get_prev_attributes()
        self.get_current_attributes()
        # get input features
        self.get_features()
        # wrap up inputs
        self.features_to_onehot()


    def get_prev_attributes(self):
        self.prev_measure_number = self.prev_note.measure_number
        self.prev_measure_num_base = self.prev_note.measure_num_base
        self.prev_time_position = self.prev_note.time_position 
        self.prev_xml_position = self.prev_note.xml_position 
        self.prev_x_position = self.prev_note.x_position
        self.prev_y_position = self.prev_note.y_position
        self.prev_dynamics = self.prev_note.dynamics
        self.prev_next_dynamics = self.prev_note.next_dynamics
        self.prev_stem = self.prev_note.stem
        self.prev_staff = self.prev_note.staff
        self.prev_is_downbeat = self.prev_note.is_downbeat
        self.prev_key_final = self.prev_note.key_final
        self.prev_beat = self.prev_note.beat
        self.in_measure_pos = self.prev_note.in_measure_pos
 
    def get_current_attributes(self):
        # depending on whether grace note
        if self.note.is_grace_note is True:
            if self.prev_note is not None:
                self.measure_num_base = self.prev_note.measure_num_base
                self.xml_position = self.prev_note.xml_position
                self.time_position = self.prev_note.time_position
            elif self.prev_note is None:
                self.measure_num_base = self.note.measure_number-1
                self.xml_position = self.note.note_duration.xml_position
                self.time_position = self.note.note_duration.time_position
        elif self.note.is_grace_note is False:
            self.measure_num_base = self.note.measure_number
            self.xml_position = self.note.note_duration.xml_position
            self.time_position = self.note.note_duration.time_position

        self.measure_number = self.note.measure_number
        self.x_position = self.note.x_position
        self.y_position = self.note.y_position
        self.stem = self.note.stem
        self.staff = self.note.staff
        self.measure_duration = self.measure.duration
        
        # get time signature
        if self.time_sig is None:
            if self.measure.time_signature is not None:
                self.time_signature_cand = self.measure.time_signature
                if self.time_signature_cand.numerator > 12 or \
                    self.time_signature_cand.denominator > 12: # error
                    if self.prev_note is not None:
                        # print('> using previous time signature instead of current: {}'.format(
                            # self.time_signature_cand))
                        # print('     --> {}'.format(self.prev_note.time_signature))
                        self.time_signature = self.prev_note.time_signature
                    elif self.prev_note is None:
                        self.time_signature = self.time_signature_cand
                        # raise AssertionError("** numerator or denominator larger than 12 -> no time signature!")   
                elif self.time_signature_cand.numerator <= 12 and \
                    self.time_signature_cand.denominator <= 12:
                    self.time_signature = self.time_signature_cand

            elif self.measure.time_signature is None:
                if self.prev_note is not None:
                    self.time_signature = self.prev_note.time_signature
                elif self.prev_note is None:
                    raise AssertionError("** no time signature!")

        elif self.time_sig is not None:
            assert len(self.time_sig.shape) == 2
            # find corresponding time signature among multiple candidates
            for each_time_sig in self.time_sig:
                if self.measure_number >= each_time_sig[0]:
                    self.time_signature_cand = each_time_sig[1]
                    break
            assert self.time_signature_cand is not None

            if self.time_signature_cand.numerator > 12 or \
                self.time_signature_cand.denominator > 12: # error
                if self.prev_note is not None:
                    # print('> using previous time signature instead of current: {}'.format(
                        # self.time_signature_cand))
                    # print('     --> {}'.format(self.prev_note.time_signature))
                    self.time_signature = self.prev_note.time_signature
                elif self.prev_note is None:
                    self.time_signature = self.time_signature_cand
                    # raise AssertionError("** numerator or denominator larger than 12 -> no time signature!")   
            elif self.time_signature_cand.numerator <= 12 and \
                self.time_signature_cand.denominator <= 12:
                self.time_signature = self.time_signature_cand


        # get key signature 
        if self.key_sig is None:
            if self.measure.key_signature is not None:
                self.key_signature = self.measure.key_signature
                key_str = str(self.key_signature)
                key_root = key_str.split(' ')[0]
                key_mode = key_str.split(' ')[1] 
                fifth1 = np.asarray(['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F'])
                fifth2 = np.asarray(['C', 'G', 'D', 'A', 'E', 'Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F'])
                
                if key_mode == 'major':
                    if key_root in fifth1:
                        num_acc = np.where(fifth1 == key_root)[0][0]
                    elif key_root in fifth2:
                        num_acc = np.where(fifth2 == key_root)[0][0]
                elif key_mode == 'minor':
                    if key_root in fifth1:
                        num_acc = (np.where(fifth1 == key_root)[0][0] + 3) % 12
                    elif key_root in fifth2:
                        num_acc = (np.where(fifth2 == key_root)[0][0] + 3) % 12
                self.num_acc = num_acc

            elif self.measure.key_signature is None:
                if self.prev_note is not None:
                    self.key_signature = self.prev_note.key_signature
                    self.key_sig = self.prev_note.key_sig
                elif self.prev_note is None:
                    raise AssertionError("** no key signature!")

        elif self.key_sig is not None:
            assert len(self.key_sig.shape) == 2
            # find corresponding time signature among multiple candidates
            for each_key_sig in self.key_sig:
                if self.measure_number >= each_key_sig[0]:
                    self.key_signature = each_key_sig[1]
                    break
            assert self.key_signature is not None

            key_str = str(self.key_signature)
            key_root = key_str.split(' ')[0]
            key_mode = key_str.split(' ')[1] 
            fifth1 = np.asarray(['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F'])
            fifth2 = np.asarray(['C', 'G', 'D', 'A', 'E', 'Cb', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F'])
            
            if key_mode == 'major':
                if key_root in fifth1:
                    num_acc = np.where(fifth1 == key_root)[0][0]
                elif key_root in fifth2:
                    num_acc = np.where(fifth2 == key_root)[0][0]
            elif key_mode == 'minor':
                if key_root in fifth1:
                    num_acc = (np.where(fifth1 == key_root)[0][0] + 3) % 12
                elif key_root in fifth2:
                    num_acc = (np.where(fifth2 == key_root)[0][0] + 3) % 12
            self.num_acc = num_acc


        # if current measure contains any directions
        if len(self.measure.directions) > 0:
            for direction in self.measure.directions:
                _time_position = direction.time_position
                if _time_position == self.time_position:
                    self.current_directions.append(direction)		

    def get_features(self):
        if self.tempo is None:
            self.get_tempo() # written tempo
        # self.get_type() # type of duration 
        self.get_grace_note() # whether grace note (also for downbeat)
        # self.get_voice() # voice number for each note
        self.get_dynamics() # global dynamics (no wedge)
        # self.get_wedge() # global/local wedges (cresc, dim, none)
        self.get_downbeat() # whether first beat of a measure
        # self.get_accent() # whether accent 
        # self.get_staccato() # whether staccato 
        # self.get_arpeggiate() # whether arpeggiate 
        # self.get_fermata() # whether fermata
        # self.get_tempo_change() # whether tempo changes
        self.get_same_onset() # whether in same onset group: start, cont 
        self.get_pitch() # pitch class and octave
        # self.get_ornament() # whether ornaments
        # self.get_tuplet() # whether tuplet
        # self.get_tied() # whether tied
        self.get_ioi() # xml IOI
        self.get_beat() # beat position (should be parsed last)

    def features_to_onehot(self):
        '''
        Make features into binary vectors
        '''
        # tempo
        _tempo = np.zeros([1,])
        _tempo[0] = self.tempo
        # type
        # _type = np.zeros([6,])
        # for i, key in enumerate(sorted(self._type)):
        # 	if self._type[key] == 1:
        # 		_type[i] = 1
        # 		break
        # dot
        # _dot = np.zeros([2,])
        # _dot[self.is_dot] = 1
        # staff
        _staff = np.zeros([2,])
        _staff[self.staff-1] = 1
        # grace note
        # _grace = np.zeros([2,])
        # _grace[self.is_grace_note] = 1	
        # voice
        # _voice = np.zeros([4,])
        # _voice[self.voice-1] = 1	
        # dynamics
        _dynamics = np.zeros([6,])
        if self.dynamics['pp'] == 1:
        	_dynamics[0] = 1
        elif self.dynamics['p'] == 1:
        	_dynamics[1] = 1
        elif self.dynamics['mp'] == 1:
        	_dynamics[2] = 1
        elif self.dynamics['mf'] == 1:
        	_dynamics[3] = 1
        elif self.dynamics['f'] == 1:
        	_dynamics[4] = 1
        elif self.dynamics['ff'] == 1:
        	_dynamics[5] = 1
        # wedges
        # _wedge = np.zeros([5,])
        # if self.wedge['cresc_start'] == 1:
        # 	_wedge[0] = 1
        # elif self.wedge['cresc_end'] == 1:
        # 	_wedge[1] = 1
        # elif self.wedge['dim_start'] == 1:
        # 	_wedge[2] = 1
        # elif self.wedge['dim_end'] == 1:
        # 	_wedge[3] = 1
        # elif self.wedge['none'] == 1:
        # 	_wedge[4] = 1
        # downbeat 
        _downbeat = np.zeros([2,])
        _downbeat[self.is_downbeat] = 1	
        # beat 
        _beat = np.zeros([12,])
        _beat[self.beat] = 1		
        # time signature 
        _num = np.zeros([12,])
        _denom = np.zeros([12,])
        _num[self.time_num-1] = 1 
        _denom[self.time_denom-1] = 1
        # key signature 
        _key = np.zeros([12,])
        _key[self.num_acc] = 1
        # onset
        # _same_onset = np.zeros([2,])	
        # if self.same_onset['start'] == 1:
        # 	_same_onset[0] = 1
        # elif self.same_onset['cont'] == 1:
        # 	_same_onset[1] = 1		
        # pitch class and octave 
        # _pitch_class = np.zeros([12,])
        # _octave = np.zeros([8,])
        # _pitch_class[self.pitch_class] = 1
        # _octave[self.octave] = 1
        # _pitch2 = np.concatenate([_pitch_class, _octave], axis=-1)
        # _pitch = np.zeros([88,])
        # _pitch[int(self.pitch-21)] = 1
        # ornament
        # _ornament = np.zeros([4,])
        # for i, key in enumerate(sorted(self.ornament)):
        # 	if self.ornament[key] == 1:
        # 		_ornament[i] = 1
        # 		break
        # tuplet
        # _tuplet = np.zeros([3,])
        # for i, key in enumerate(sorted(self.tuplet)):
        # 	if self.tuplet[key] == 1:
        # 		_tuplet[i] = 1
        # 		break
        # tied
        # _tied = np.zeros([2,])
        # _tied[self.is_tied] = 1

        # check if onehot for each feature 
        # assert np.sum(_type) == 1
        assert np.sum(_staff) == 1
        # assert np.sum(_dot) == 1
        # assert np.sum(_grace) == 1
        # assert np.sum(_voice) == 1
        if np.sum(_dynamics) != 1:
        	print(self.dynamics, _dynamics, self.note)
        	raise AssertionError
        assert np.sum(_downbeat) == 1
        assert np.sum(_beat) == 1
        assert np.sum(_num) == 1
        assert np.sum(_denom) == 1
        assert np.sum(_key) == 1
        # assert np.sum(_same_onset) == 1
        # assert np.sum(_pitch_class) == 1
        # assert np.sum(_octave) == 1
        # assert np.sum(_pitch) == 1
        # assert np.sum(_ornament) == 1
        # assert np.sum(_tuplet) == 1
        # assert np.sum(_tied) == 1
        # assert np.sum(_wedge) == 1

        # concatenate all features into one vector 
        # self._input = np.concatenate(
            # [_tempo, _type, _dot, _staff, _grace, _voice, _dynamics, 
            # _pitch, _pitch2, _same_onset, _wedge], axis=-1)
        self._input = np.concatenate([_tempo, _beat, _dynamics, _staff, _downbeat, _num, _denom, _key], axis=-1)

    def get_tempo(self):
        if len(self.measure.directions) > 0:
            for direction in self.measure.directions:
                if direction.tempo is not None:
                    self.tempo = direction.tempo

        if self.tempo is None: # nothing in directions
            if self.prev_note is not None:
                self.tempo = self.prev_note.tempo
            elif self.prev_note is None:
                raise AssertionError
        elif self.tempo is not None: # tempo in directions
            pass

        self.tempo = float(self.tempo)

    def get_ioi(self):
        if self.prev_note is None:
            self.ioi = 0 
        elif self.prev_note is not None:
            if self.same_onset['start'] == 1 and self.is_grace_note == 0:
                self.ioi = self.xml_position - self.prev_xml_position
            else:
                self.ioi = self.prev_note.ioi

    def get_beat(self):
        '''
        get beat position within a measure with note duration in xml 
        '''
        # update measure position (should be updated last)
        if self.is_downbeat == 1: # if start of measure (not grace note)
            self.in_measure_pos = 0
        elif self.is_downbeat == 0:
            if self.same_onset['start'] == 1 and self.is_grace_note == 0: # only if not grace note
                self.in_measure_pos += self.ioi
            else:
                self.in_measure_pos = self.in_measure_pos 

        num = self.time_signature.numerator # beat num (2 in 2/4)
        denom = self.time_signature.denominator # beat type (4 in 2/4)
        self.time_signature_text = "{}/{}".format(num, denom)
        if num > 12 or denom > 12:
            # raise AssertionError
            '''
            the denominator is typically a power of 2
            if denom > 12, maybe denom is one of 16, 32, ..., etc
            '''
            orig_num, orig_denom = copy.deepcopy(num), copy.deepcopy(denom)
            while num > 12 or denom > 12:
                if num % 2 == 0 and denom % 2 == 0:
                    denom = denom // 2
                    num = num // 2 
                else:
                    # print("time sig larger than 12! --> {}".format([num, denom]))
                    raise AssertionError
            # print("time sig {} --> {}".format([orig_num, orig_denom], [num, denom]))
        
        else:
            pass

        self.time_num = num 
        self.time_denom = denom   

        if self.prev_note is None:
            self.beat = 0 # first beat

        elif self.prev_note is not None:            
            # if in different onset group
            beat_unit = self.measure.duration // num # xml duration of each beat
            if self.same_onset['start'] == 1:
                self.beat = self.in_measure_pos // beat_unit 

            # if in same onset group
            elif self.same_onset['cont'] == 1:
                self.beat = self.prev_beat

            # print(self.note_ind, self.note.pitch, self.measure_num_base, self.time_position, 
            #     self.is_downbeat, self.same_onset['start'], self.is_grace_note, 
            #     self.measure.duration, self.time_signature, 
            #     num, beat_unit, self.beat, self.in_measure_pos, self.ioi)
            # print(self.in_measure_pos, beat_unit, self.note_ind)

        if self.beat >= 12:
            self.beat = self.beat % 4
        assert self.beat < num


    def get_type(self):
        '''
        get note type from note.note_duration.type 
        '''
        note_type = self.note.note_duration.type
        shorter_group = ['32nd','64th','128th','256th','512th','1024th']
        longer_group = ['whole','breve','long','maxima']
        if len(note_type) > 0:
            if note_type in shorter_group:
                self._type['shorter'] = 1
            elif note_type == '16th':
                self._type['16th']= 1
            elif note_type == 'eighth':
                self._type['eighth']= 1
            elif note_type == 'quarter':
                self._type['quarter'] = 1
            elif note_type == 'half':
                self._type['half'] = 1
            elif note_type in longer_group:
                self._type['longer'] = 1
        # if note is dotted
        if self.note.note_duration.dots > 0:
            self.is_dot = 1

    def get_grace_note(self):
        '''
        get whether a note is grace note from note.is_grace_note
        '''
        if self.note.is_grace_note is True:
            self.is_grace_note = 1
        elif self.note.is_grace_note is False:
            self.is_grace_note = 0

    def get_voice(self):
        '''
        get voice number for a note from note.voice
        '''
        if self.note.voice <= 4:
            self.voice = self.note.voice
        elif self.note.voice > 4:
            self.voice = self.note.voice - 4

    def get_dynamics(self):
        '''
        get dynamics from measure.directions.type
        * only consider global dynamics (most of dynamics are not local) 
        '''	
        dynamic_list = list(self.dynamics.keys())
        dynamic_candidates = list()

        # parse dynamics within current directions 
        if len(self.current_directions) > 0:
            for direction in self.current_directions:
                '''
                        ff (staff 1-above)
                staff 1 ==================
                        ff (staff 1-below / staff 2-above)
                staff 2 ==================
                        ff (staff 2-below)
                '''
                _staff = int(direction.staff) # staff position of the direction
                _place = direction.placement # whether above/below note.staff
                _dynamic = None
                _next_dynamic = None # in case of "fp" kinds

                if (_staff == 1 and _place == 'below') or \
                    (_staff == 2 and _place == 'above'):
                
                    content = direction.type["content"]
                    # parse dynamics with "dynamic" type
                    if direction.type["type"] == "dynamic":
                        if content in dynamic_list:
                            _dynamic = content 
                        else: # other dynamics other than basic ones
                            if content == "fp":
                                _dynamic = "f" 
                                # _dynamic = "p" 
                                _next_dynamic = "p"
                            elif content == "ffp":
                                _dynamic = "ff" 
                                # _dynamic = "p" 
                                _next_dynamic = "p"
                            elif content == "ppp":
                                _dynamic = "pp"
                            elif content == "fff":
                                _dynamic = "ff"
                            else:
                                # print("** dynamics at {}th note: {}".format(note_ind, content))
                                continue

                    # parse dynamics with "word" type (ex: f con fuoco)
                    elif direction.type["type"] == "words":
                        if 'pp ' in str(content) or ' pp' in str(content):
                            _dynamic = 'pp'
                        elif 'p ' in str(content) or ' p' in str(content):
                            _dynamic = 'p'
                        elif 'mp ' in str(content) or ' mp' in str(content):
                            _dynamic = 'mp'
                        elif 'mf ' in str(content) or ' mf' in str(content):
                            _dynamic = 'mf'
                        elif 'f ' in str(content) or ' f' in str(content):
                            _dynamic = 'f'
                        elif 'ff ' in str(content) or ' ff' in str(content):
                            _dynamic = 'ff'

                    if _dynamic is not None:
                        dynamic_candidates.append(
                            {'dynamic': _dynamic, 'next_dynamic': _next_dynamic})
        # print(dynamic_candidates)

        if len(dynamic_candidates) == 1:
            self.is_new_dynamics = True
            self.dynamics[dynamic_candidates[0]['dynamic']] = 1
            self.next_dynamics = dynamic_candidates[0]['next_dynamic']

        elif len(dynamic_candidates) > 1:
            print("** Global dynamic is more than one:")
            print("- measure number {}".format(self.measure_number))
            print(dynamic_candidates)
            self.is_new_dynamics = True
            self.dynamics[dynamic_candidates[0]['dynamic']] = 1
            self.next_dynamics = dynamic_candidates[0]['next_dynamic']            
            # raise AssertionError	

        # if no dynamic is assigned
        elif len(dynamic_candidates) == 0: 
            self.is_new_dynamics = False
            if self.prev_dynamics is not None: 
                if self.prev_next_dynamics is None:
                    self.dynamics = self.prev_dynamics
                # in case if "fp" kinds
                elif self.prev_next_dynamics is not None:
                    self.dynamics[self.prev_next_dynamics] = 1
            # This case can happen when no dynamics at start
            elif self.prev_dynamics is None: 
                self.dynamics['mp'] = 1

    def get_same_onset(self):
        '''
        see xml and find whether onsets are the same with previous, next notes
        '''
        if self.prev_note is None: # first note:
            self.same_onset['start'] = 1
        elif self.prev_note is not None:
            if self.is_grace_note == 0:
                if self.time_position > self.prev_time_position:
                    self.same_onset['start'] = 1
                elif self.time_position == self.prev_time_position:
                    self.same_onset['cont'] = 1
            elif self.is_grace_note == 1:
                if self.x_position != self.prev_x_position:
                    self.same_onset['start'] = 1
                elif self.x_position == self.prev_x_position:
                    self.same_onset['cont'] = 1

    def get_pitch(self):
        '''
        get pitch from note.pitch
        '''
        midi_num = self.note.pitch[1]
        self.pitch = midi_num 
        self.pitch_name = self.note.pitch[0]
        '''
        measure.key_signature.key --> based on fifths
        - -1(F), 0(C), 1(G), D(2), ...
        abs(fifths * 7) % 12 --> tonic
        '''
        # if self.measure.key_signature is not None:
        #     fifths_in_measure = self.measure.key_signature.key
        #     if fifths_in_measure < 0:
        #         key = ((fifths_in_measure * 7) % -12) + 12 # if Ab major, "-4"
        #     elif fifths_in_measure >= 0:
        #         key = (fifths_in_measure * 7) % 12

        #     self.mode = self.measure.key_signature.mode # 'major' / 'minor'
        #     if self.mode == "minor":
        #         self.key_final = (key - 3 + 12) % 12 # minor 3 below
        #     elif self.mode == "major":
        #         self.key_final = key
        
        # elif self.measure.key_signature is None:
        #     self.key_final = self.prev_key_final

        # self.pitch_norm = midi_num - self.key_final # normalize to C major/minor
        self.pitch_class = np.mod(midi_num, 12) # pitch class
        self.octave = int(midi_num / 12) - 1 # octave
        assert self.pitch_class != None
        assert self.octave != None

    def get_downbeat(self):
        '''
        get measure number of each note and see transition point 
        - notes in same onset group are considered as one event 
        '''
        if self.is_grace_note == 0:
            if self.prev_note is None:
                self.is_downbeat = 1
            elif self.prev_note is not None:
                # if in different onset group
                if self.prev_time_position != self.time_position:
                    # new measure (measure_num_base -> disregard grace note)
                    if self.measure_num_base != self.prev_measure_num_base: 
                        self.is_downbeat = 1
                    # same measure
                    elif self.measure_num_base == self.prev_measure_num_base: 
                        self.is_downbeat = 0
                # if in same onset group
                elif self.prev_time_position == self.time_position:
                    self.is_downbeat = self.prev_is_downbeat
        elif self.is_grace_note == 1:
            self.is_downbeat = 0 # if grace note, no downbeat

    def get_ornament(self):
        '''
        get ornaments from note.note_notations.*:
            - is_trill
            - is_mordent
            - wavy_line
        '''
        if self.note.note_notations.is_trill is True:
            self.ornament['trill'] = 1
        elif self.note.note_notations.is_mordent is True:
            self.ornament['mordent'] = 1
        elif self.note.note_notations.wavy_line is not None:
            self.ornament['wavy'] = 1
        else:
            self.ornament['none'] = 1

    def get_tuplet(self):
        '''
        get tuplets from note.note_notations.tuplet_*:
            - start
            - stop
        '''
        if self.note.note_notations.tuplet_start is True:
            self.tuplet['start'] = 1
        elif self.note.note_notations.tuplet_stop is True:
            self.tuplet['stop'] = 1
        else:
            self.tuplet['none'] = 1

    def get_tied(self):
        '''
        get tied from note.note_notations.tied_start
        * tied_stop notes are applied as longer duration
        '''
        if self.note.note_notations.tied_start is True:
            self.is_tied = 1

    def get_wedge(self):
        '''
        get cresc. attributes measure.directions.type 
        - for word, either cresc or dim is assigned (both cannot happen at the same time)
        - this function is only for parsing middle wedges
        - more complex version is in "note_seq_parsing2.py"
        '''

        # newly assign class if first note of the onset group
        self.new_wedge = list() # newly assigned wedge

        # collect new wedges in vertical position of 
        if len(self.current_directions) > 0:
            # assign class for all directions
            for direction in self.current_directions:
                staff = int(direction.staff)
                place = direction.placement 
                # only if middle
                if (staff == 1 and place == "below") or (staff == 2 and place == "above"):	
                    if direction.type["type"] == "words" and \
                        'cresc' in str(direction.type["content"]): 	
                        self.new_wedge.append(['cresc_word_start', staff, place])
                    elif direction.type["type"] == "words" and \
                        'dim' in str(direction.type["content"]): 
                        self.new_wedge.append(['dim_word_start', staff, place])
                    elif direction.type["type"] == "crescendo" and \
                        direction.type["content"] == "start":	
                        self.new_wedge.append(['cresc_wedge_start', staff, place])
                    elif direction.type["type"] == "diminuendo" and \
                        direction.type["content"] == "start":
                        self.new_wedge.append(['dim_wedge_start', staff, place])
                    elif direction.type["type"] == "none" and \
                        direction.type["content"] == "stop":
                        self.new_wedge.append(['wedge_stop', staff, place])
        
        print(self.new_wedge)

        cresc_start, cresc_end, dim_start, dim_end, none = 0, 0, 0, 0, 0
        # if only one wedge newly appears in current note's onset   
        if len(self.new_wedge) == 1:
            print("--> new wedge is 1")

            _content = self.new_wedge[0][0]
            _staff = self.new_wedge[0][1]
            _place = self.new_wedge[0][2]
            _pos = self.new_wedge[0][3]

            # make sure that wedge is in middle
            assert (_staff == 1 and _place == 'below') or (_staff == 2 and _place == 'above')
            self.wedge_in_middle = True

            # for cresc wedge
            if 'cresc' in _content:
                cresc_start = 1
                if _content == 'cresc_word_start':
                    self.cresc_in_word = True
                elif _content == 'cresc_wedge_start':
                    self.cresc_in_wedge = True

            # for dim wedge
            elif 'dim' in _content:
                dim_start = 1
                if _content == 'dim_word_start':
                    self.dim_in_word = True
                elif _content == 'dim_wedge_start':
                    self.dim_in_wedge = True

            # for none wedge
            elif _content == 'wedge_stop':
                if self.prev_wedge_in_middle == True:
                    if self.cresc_in_wedge == True:
                        cresc_end = 1
                        self.cresc_in_wedge = False
                    elif self.dim_in_wedge == True:
                        dim_end = 1
                        self.dim_in_wedge = False
                else:
                    print("no wedge to stop!")
                    raise AssertionError

        # if multiple wedges newly appears in current note's onset 
        elif len(self.new_wedge) > 1:
            print("--> new wedge is >1")

            multi_contents = [nw[0] for nw in self.new_wedge]
            cresc_content = 0
            dim_content = 0
            for c in multi_contents:
                if "cresc" in c:
                    cresc_content += 1
                elif "dim" in c:
                    dim_content += 1

            if cresc_content >= 1 and dim_content >= 1:
                print("Both cresc and dim appeared for new wedge!")
                raise AssertionError
            else:
                if cresc_content >= 1:
                    cresc_start = 1
                    if 'cresc_word_start' in multi_contents:
                        self.cresc_in_word = True
                    if 'cresc_wedge_start' in multi_contents:
                        self.cresc_in_wedge = True
                elif dim_content >= 1:
                    dim_start = 1
                    if 'dim_word_start' in multi_contents:
                        self.dim_in_word = True
                    if 'dim_wedge_start' in multi_contents:
                        self.dim_in_wedge = True
                                            
        # if no new wedge
        elif len(self.new_wedge) == 0:
            print("--> new wedge is 0")

            # if there is new dynamic (from dynamic class)
            if self.is_new_dynamics == True:
                if self.prev_wedge_in_middle == True:
                    if self.cresc_in_wedge == True:
                        cresc_end = 1
                        self.cresc_in_wedge = False
                    elif self.dim_in_wedge == True:
                        dim_end = 1
                        self.dim_in_wedge = False	
                else:
                    if self.cresc_in_word == True:
                        cresc_end = 1
                        self.cresc_in_word == False
                    elif self.dim_in_word == True:
                        dim_end = 1
                        self.dim_in_word == False
                    else:
                        none = 1
            else:
                none = 1

        # if there is absolutely no wedge
        if self.cresc_in_word == False and self.dim_in_word == False and \
            self.cresc_in_wedge == False and self.dim_in_wedge == False:
            self.wedge_in_middle = None

        # finally assign wedge information
        self.wedge['cresc_start'] = cresc_start
        self.wedge['cresc_end'] = cresc_end
        self.wedge['dim_start'] = dim_start
        self.wedge['dim_end'] = dim_end
        self.wedge['none'] = none

        print([cresc_start, cresc_end, dim_start, dim_end, none])
        assert sum([cresc_start, cresc_end, dim_start, dim_end, none]) == 1




