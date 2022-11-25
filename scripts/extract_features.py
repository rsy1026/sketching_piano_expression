import numpy as np
import os

import models.piano_model as model
from generate import *


def main(
    xml, score, perform=None, save_dir=None):

    score = mid
    p_name = os.path.basename(mid).split(".")[0]
    save_name = os.path.join(save_dir, "{}.features.npy".format(p_name))

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
        test_inputs_ = get_data.data2input_noY(test_x, test_m, device=device)
        test_x_, test_m_ = test_inputs_

        np.save(save_name, test_inputs_, dtype=object)
    
    elif perform is not None:
        test_x, test_y, test_m, pairs, xml_notes, perform_notes, tempo, p_name = \
            get_data.file2data(
                files=[xml, score, perform], measures=measures, 
                pair_path=None, save_mid=True)
        # get input data
        test_inputs_ = get_data.data2input(test_x, test_y, test_m, 
            cond=ind, art=True, device=device)
        test_x_, test_y_, test_y2_, test_m_, test_clab_ = test_inputs_
        # raw features (vel, art, ioi)
        raw_notes, raw_chords = get_data.file2yfeature(
            files=[xml, score, perform], measures, pair_path=None)
        
        np.save(save_name, (test_inputs_, raw_notes, raw_chords), dtype=object)
        


if __name__ == "__main__"():
    import logging, sys, argparse
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--xml', type=str, default=None,
                        help='name of MusicXML file')
    parser.add_argument('--score', type=int, default=None,
                        help='name of score MIDI file')
    parser.add_argument('--perform', type=int, default=None,
                        help='name of performance MIDI file')
    parser.add_argument('--save_dir', type=str, default='./',
                        help='output directory')

    args = parser.parse_args()

    main(
        xml=args.xml, 
        score=args.score, 
        perform=args.perform, 
        save_dir=args.save_dir)
