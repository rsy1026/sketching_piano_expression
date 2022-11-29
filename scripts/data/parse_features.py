import os
import numpy as np
from glob import glob
from pretty_midi import Note
import csv
import time
import copy
from decimal import Decimal, getcontext, ROUND_HALF_UP, InvalidOperation

from sketching_piano_expression.utils.match import XML_SCORE_PERFORM_MATCH as MATCH
from sketching_piano_expression.utils.parse_utils import *
from .make_batches import (
    make_onset_based_all,
    make_onset_based_pick,
)


dc = getcontext()
dc.prec = 48
dc.rounding = ROUND_HALF_UP


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
    * Function to search 3 kinds of files in 'dirname'
        - xml(score): 'xml_list'
        - midi(score): 'score_midi_list'
        - midi(perform): 'perform_midi_list'
    """
    # initialize lists 
    xml_list = dict()
    score_midi_list = dict()
    perform_midi_list = dict()

    # collect directories 
    categs = sorted(glob(os.path.join(dirname, "*/"))) # category ex) Ballade

    for c in categs:
        c_name = c.split('/')[-2] # get category name
        xml_list[c_name] = dict()
        score_midi_list[c_name] = dict()
        perform_midi_list[c_name] = dict()
        pieces = sorted(glob(os.path.join(c, "*/"))) # piece ex) (Ballade) No.1
        for p in pieces: 
            p_name = p.split('/')[-2] # get piece name
            players = sorted(glob(os.path.join(p, "*/"))) # player 3x) (Ballade No.1) player1
            # get each path of xml, score, performance files
            xml_path = os.path.join(p, "*.musicxml")
            score_path = os.path.join(p, "*score*.mid")
            # assign paths to corresponding piece category
            if os.path.exists(xml_path) is True:
                xml_list[c_name][p_name] = xml_path
                score_midi_list[c_name][p_name] = score_path
                perform_midi_list[c_name][p_name] = list()
                for pl in players:
                    pl_name = pl.split('/')[-2]
                    perform_path = glob(os.path.join(pl, '[!score]*.mid'))
                    perform_path += glob(os.path.join(pl, '[!score]*.MID'))
                    perform_path = [p for p in perform_path if os.path.basename(p).split(".")[1] != "cleaned"]
                    perform_midi_list[c_name][p_name].append(perform_path[0])
    
    return xml_list, score_midi_list, perform_midi_list

def save_matched_files(dirname):
    # parent directories
    program_dir = os.getcwd()
    # get directory lists
    xml_list, score_midi_list, perform_midi_list = search(dirname)
    match_files = MATCH(
        current_dir=os.getcwd(), 
        program_dir=program_dir)	
    
    # start matching
    for categ in sorted(perform_midi_list): 
        for piece in sorted(perform_midi_list[categ]):
            xml = xml_list[categ][piece]
            score = score_midi_list[categ][piece]
            # save_xml_to_midi(xml, score)
            performs = perform_midi_list[categ][piece]
            # save pairs: xml-score-perform
            pair_path = os.path.join(
                os.path.dirname(performs[0]), "xml_score_perform_pairs.npy")
            if os.path.exists(pair_path) is False:
                '''
                <examples>
                xml = "~/data_samples/raw_samples/musicxml_cleaned_plain.musicxml"
                score = "~/data_samples/raw_samples/score_plain.mid"
                performs = ["~/data_samples/raw_samples/01/Na06.mid"]
                '''
                _, _ = match_files(xml, score, performs, save_pairs=True, plot=True)	

            print("saved pairs for {}:{}".format(categ, piece))

def save_features_xml():
    # dirname = './data/raw_samples'
    categs = sorted(glob(os.path.join(dirname, "*/")))

    for categ in categs:
        c_name = categ.split('/')[-2]
        pieces = sorted(glob(os.path.join(categ, "*/")))
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
                    csv_list.append(['> key_signature: {} (COF: {})'.format(parsed_note.key_signature, parsed_note.num_acc)])
                    csv_list.append(['> onset: {:4f}'.format(parsed_note.time_position)])
                    csv_list.append(['> pitch: {} (pc: {} / octave: {})'.format(
                        parsed_note.pitch_name, parsed_note.pitch_class, parsed_note.octave)])
                    csv_list.append(["> dynamics: {}".format(parsed_note.dynamics)])
                    csv_list.append(["> is_new_dynamic: {}".format(parsed_note.is_new_dynamics)])
                    csv_list.append(["> next_dynamic: {}".format(parsed_note.next_dynamics)])
                    csv_list.append(["> beat: {}".format(parsed_note.beat)])
                    csv_list.append(["> is_downbeat: {}".format(parsed_note.is_downbeat)])
                    csv_list.append(["---------------------------------------------------------------------------"])
                    csv_list.append(["--> tempo input: {}".format(_input[:1])])
                    csv_list.append(["--> beat input: {}".format(_input[1:13])]) # 12
                    csv_list.append(["--> dynamic input: {}".format(_input[13:19])]) # 6
                    csv_list.append(["--> staff input: {}".format(_input[19:21])]) # 2
                    csv_list.append(["--> downbeat input: {}".format(_input[21:23])])
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

def save_features_midi(dirname):

    # dirname = './data/raw_samples'
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

                    input_list, output_list = parse_midi_features(
                        pair_path=pair_path,
                        same_onset_ind=[110,112], null_tempo=120)

                    # save outputs
                    np.save(os.path.join(player, 'inp.npy'), input_list)
                    np.save(os.path.join(player, 'oup.npy'), output_list) # ioi in note-level

                    print("parsed {}/{} output: player {}\n(inp len: {} / oup len: {})".format(
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

def interpolate_feature(
    output_list, input_list, note_list, f_type=None, same_onset_ind=None):
    '''
    f_type: [ioi1, ioi2, dur]
    '''
    if f_type == "ioi1":
        f_ind = 2 
    elif f_type == "ioi2":
        f_ind = 3 
    elif f_type == "dur":
        f_ind = 1

    prev_f = None
    non_f = list()
    non_f_list = list()
    target_list = copy.deepcopy(output_list)

    for n, o in enumerate(target_list):
        f = o[1][f_ind]
        is_same_onset = np.argmax(
            input_list[n][1][same_onset_ind[0]:same_onset_ind[1]], -1)
        if prev_f == f and f == 99:
            non_f.append([is_same_onset, n])
        elif prev_f != f and f == 99:
            if len(non_f) > 0:
                non_f_list.append(non_f)
            non_f = [[is_same_onset, n]]
        prev_f = f
    non_f_list.append(non_f)

    for f in non_f_list:
        # find entries for interpolation
        onsets = [i for i in f if i[0] == 0]
        try:
            prev_f = target_list[f[0][1]-1][1][f_ind]
        except IndexError:
            prev_f = 1 / note_list[0][1].tempo_ratio
        try:
            next_f = target_list[f[-1][1]+1][1][f_ind]
        except IndexError:
            next_f = prev_f
        try:
            next_is_same_onset = np.argmax(
                input_list[f[-1][1]+1][1][same_onset_ind[0]:same_onset_ind[1]], -1) 
        except IndexError:
            next_is_same_onset = 1
        
        if next_is_same_onset == 1:
            for each in f:
                target_list[each[1]][1][f_ind] = next_f
        elif next_is_same_onset == 0:
            interps = np.linspace(prev_f, next_f, len(onsets)+1, endpoint=False)[1:]
            n = -1
            for each in f:
                if each[0] == 0:
                    n += 1
                target_list[each[1]][1][f_ind] = interps[n]
    
    return target_list, non_f_list

def parse_midi_features(
    pairs_score=None, pair_path=None,
    tempo=None, null_tempo=None, same_onset_ind=None):

    '''
    * function for parsing MIDI features
        - always parse in order of SCORE MIDI 
    '''

    ## FOR DEBUGGING ## 
    # pair_path = './data/raw_samples/beethoven_piano_sonatas/8-2/01/xml_score_perform_pairs.npy'
    # xml = './data/raw_samples/beethoven_piano_sonatas/8-2/musicxml_cleaned_plain.musicxml'        
    # tempo, time_sig, key_sig = get_signature_from_xml(xml, measure_start=0)
    # pairs_score = None 
    # midi_notes = None
    # null_tempo = 120
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

        _input = parsed_note.get_input_features(
            base_onset_score, next_onset_score)
        _output = parsed_note.get_output_features(
            base_onset_perform, next_onset_perform)

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
    output_list_, non_durs = interpolate_feature(output_list, input_list, note_list, 
        f_type="dur", same_onset_ind=same_onset_ind)
    output_list_, non_ioi1s = interpolate_feature(output_list_, input_list, note_list,
        f_type="ioi1", same_onset_ind=same_onset_ind)
    output_list_, non_ioi2s = interpolate_feature(output_list_, input_list, note_list, 
        f_type="ioi2", same_onset_ind=same_onset_ind)

    # to numpy array
    input_list = np.array(input_list, dtype=object)
    output_list = np.array(output_list_, dtype=object)

    inp = np.asarray([i[1] for i in input_list])
    oup = np.asarray([o[1][0] for o in output_list])

    # rearrange mean onsets
    base_onsets = make_onset_based_pick(
        inp, np.asarray(base_onset_perform_list), same_onset_ind=same_onset_ind)
    mean_onsets = make_onset_based_pick(
        inp, np.asarray(mean_onset_perform_list), same_onset_ind=same_onset_ind)
    next_onsets = make_onset_based_pick(
        inp, np.asarray(next_onset_perform_list), same_onset_ind=same_onset_ind)

    assert np.array_equal(base_onsets[1:], mean_onsets[:-1])
    assert np.array_equal(mean_onsets[1:], next_onsets[:-1])

    return input_list, output_list

def parse_test_features_noY(
    xml, score, measures=None,
    tempo=None, null_tempo=None, same_onset_ind=None):

    '''
    * function for parsing MIDI features
        - always parse in order of SCORE MIDI 
    '''

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
    else:
        pairs_score = pairs_score_all

    # get first onsets
    pairs_score_onset = make_onset_pairs(pairs_score, fmt="xml")
    first_onset_score = np.min(
        [n['xml_note'][1].note_duration.time_position \
            for n in pairs_score_onset[0]])

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

        _input = parsed_note.get_input_features(
            base_onset_score, next_onset_score)

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

    # check the order of onsets
    base_onsets = make_onset_based_pick(
        inp, np.asarray(base_onset_list), same_onset_ind=same_onset_ind)
    mean_onsets = make_onset_based_pick(
        inp, np.asarray(mean_onset_list), same_onset_ind=same_onset_ind)
    next_onsets = make_onset_based_pick(
        inp, np.asarray(next_onset_list), same_onset_ind=same_onset_ind)

    assert np.array_equal(base_onsets[1:], mean_onsets[:-1])
    assert np.array_equal(mean_onsets[1:], next_onsets[:-1])


    return inp, pairs_score, note_ind

def parse_test_features(
    xml=None, score=None, perform=None, corresp=None,
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
        pairs_score=pairs_score,
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
        self, onset_for_ioi, next_onset_for_ioi):
        '''
        outputs based on performed attributes
        '''
        self.get_velocity() # velocity value v
        self.get_ioi_ratio(onset_for_ioi, next_onset_for_ioi) # ioi ratio
        self.get_duration_ratio() # duration ratio v
        
        self.output_to_vector()
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

    def output_to_vector(self):
        _velocity = self.velocity
        _dur_ratio = float(self.dur_ratio)
        _ioi_ratio1 = float(self.ioi_ratio1)
        _ioi_ratio2 = float(self.ioi_ratio2)
        
        self._output = np.stack(
            [_velocity, _dur_ratio, _ioi_ratio1, _ioi_ratio2], axis=-1) # note-level

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

            # if score onset same as previous note
            elif self.score_onset == self.prev_note.score_onset:
                self.is_same_onset = True           
        
        assert self.is_same_onset is not None
    
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

    def get_ioi_ratio(self, onset_for_ioi, next_onset_for_ioi):
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
        self.dynamics = {'pp': None, 'p': None, 'mp': None,
                         'ff': None, 'f': None, 'mf': None}
        self.beat = None
        self.is_downbeat = None
        self.is_grace_note = None
        self.time_num = None 
        self.time_denom = None
        self.same_onset = {'start': None, 'cont': None}
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
                    self.num_acc = self.prev_note.num_acc
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
        self.get_grace_note() # whether grace note (also for downbeat)
        self.get_dynamics() # global dynamics (no wedge)
        self.get_downbeat() # whether first beat of a measure
        self.get_same_onset() # whether in same onset group: start, cont 
        self.get_pitch() # pitch class and octave
        self.get_ioi() # xml IOI
        self.get_beat() # beat position (should be parsed last)

    def features_to_onehot(self):
        '''
        Make features into binary vectors
        '''
        # tempo
        _tempo = np.zeros([1,])
        _tempo[0] = self.tempo
        # staff
        _staff = np.zeros([2,])
        _staff[self.staff-1] = 1
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

        # check if onehot for each feature 
        assert np.sum(_staff) == 1
        if np.sum(_dynamics) != 1:
        	print(self.dynamics, _dynamics, self.note)
        	raise AssertionError
        assert np.sum(_downbeat) == 1
        assert np.sum(_beat) == 1
        assert np.sum(_num) == 1
        assert np.sum(_denom) == 1
        assert np.sum(_key) == 1


        # concatenate all features into one vector 
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





