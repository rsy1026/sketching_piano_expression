import numpy as np
import os
import pickle

import model.piano_model as model
from generate import *


def main(
    xml, score, perform=None, measures=None, save_dir=None, device=None):

    '''
    xml = "/workspace/Piano/gen_task/musicxml_cleaned_plain.musicxml"
    score = "/workspace/Piano/gen_task/score_plain.mid"
    perform = "/workspace/Piano/gen_task/Sun01_3.mid"
    measures = [1,10]
    save_dir = "./"
    device = None
    '''

    p_name = os.path.basename(score).split(".")[0]
    
    message = ">> Parsing features for files: {}".format(p_name)
    if measures is not None:
        assert len(measures) == 2
        message += " >> measures {}-{}".format(measures[0], measures[1])
    else:
        message += " >> all measures"
    print(message)
        
    ## LOAD DATA ##
    null_tempo = 120
    get_data = GetData(null_tempo=null_tempo, 
                        same_onset_ind=[110,112],
                        stat=np.mean)

    if perform is None:
        test_x, test_m, pairs, xml_notes, tempo, p_name = \
            get_data.file2data_noY(
                files=[xml, score], measures=measures)
        # get input data
        test_inputs_ = get_data.data2input_noY(
            test_x, test_m, device=device)
        test_x_, test_m_ = test_inputs_
        test_inputs_np = [t.cpu().data.numpy() for t in test_inputs_]
        # save data
        save_name = os.path.join(save_dir, "{}.features_noY.npy".format(p_name))
        pickle.dump(test_inputs_np, open(save_name, "wb"))
    
    elif perform is not None:
        test_x, test_y, test_m, pairs, xml_notes, perform_notes, tempo, p_name = \
            get_data.file2data(
                files=[xml, score, perform], measures=measures, 
                pair_path=None, save_mid=True)
        # get input data
        test_inputs_ = get_data.data2input(
            test_x, test_y, test_m, art=True, device=device)
        test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs_
        test_inputs_np = [t.cpu().data.numpy() for t in test_inputs_]
        # raw features (vel, art, ioi)
        # raw_notes, raw_chords = get_data.file2yfeature(
            # files=[xml, score, perform], measures=measures, pair_path=None)
        # save data
        save_name = os.path.join(save_dir, "{}.features.npy".format(p_name))
        pickle.dump(test_inputs_np, open(save_name, "wb"))
        


if __name__ == "__main__":
    import logging, sys, argparse
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--xml', type=str, default=None,
                        help='name of MusicXML file')
    parser.add_argument('--score', type=str, default=None,
                        help='name of score MIDI file')
    parser.add_argument('--perform', type=str, default=None,
                        help='name of performance MIDI file')
    parser.add_argument('--measures', type=int, nargs='+', default=None,
                        help='range by measure')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='output directory')
    parser.add_argument('--device', type=int, default=None,
                        help='cpu/gpu device')

    args = parser.parse_args()

    main(
        xml=args.xml, 
        score=args.score, 
        perform=args.perform, 
        measures=args.measures,
        save_dir=args.save_dir,
        device=args.device)
