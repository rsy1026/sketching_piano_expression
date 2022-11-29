import numpy as np
import os
import sys 
sys.path.append('~/utils')
from glob import glob
import h5py

from sketching_piano_expression.utils.parse_utils import *


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

def main(dirname, same_onset_ind=[110,112]):
    print("Saving batches...")

    # dirname = './features'
    groups = sorted(glob(os.path.join(dirname, "*/"))) # train/val/test
    maxlen, hop = 16, 4

    all_batches = list()
    all_num = 0
    for group in groups: # train/val/test
        datapath = os.path.join(group, 'raw')
        savepath = os.path.join(group, 'batch')
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        categs = sorted(glob(os.path.join(datapath, '*/')))

        for categ in categs:
            c_name = categ.split('/')[-2]
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
                
                for inp_path, oup_path in zip(inps, oups):

                    pl_name = os.path.basename(oup_path).split('.')[-2]
                    inp_ = np.load(inp_path, allow_pickle=True)
                    oup_ = np.load(oup_path, allow_pickle=True)
                    inp_ = sorted(inp_, key=lambda x: x[0])
                    oup_ = sorted(oup_, key=lambda x: x[0]) # same order with score midi
                    if not len(inp_) == len(oup_) == len(cond_):
                        print(c_name, p_name, pl_name)
                        raise AssertionError

                    # augmentation
                    '''
                    cond: tempo(1), beat(12), dynamics(6), staff(2), downbeat(2), 
                        time_sig(num)(12), time_sig(denom)(12), key_sig(12)
                    '''
                    cond = np.asarray([c[1] for c in cond_])
                    inp = np.asarray([i[1] for i in inp_])
                    t_aug = [1]
                    d_aug = [1]

                    print("{}: {}: {}".format(c_name, p_name, pl_name), end="\r")

                    for t in t_aug: # augment tempo
                        for d in d_aug: # augment dynamics
                            in_batch = inp
                            out_batch = np.asarray([
                                [o[1][0]*d, o[1][1]*t,
                                o[1][2]*t, o[1][3]*t] for o in oup_]) # , o[1][4]*t
                                                        
                            onset_ind = note_to_onset_ind(in_batch, same_onset_ind=same_onset_ind)
                            t_ind = ind2str(int(t*100), 3)
                            d_ind = ind2str(int(d*100), 3)

                            # save batch 
                            num = 1
                            for b in range(0, len(onset_ind)-maxlen, hop): # onset-based
                                
                                b_ind = ind2str(num, 4) 

                                # onset_based
                                onset_in_range = onset_ind[b:b+maxlen+1]
                                minind = onset_in_range[0]
                                maxind = onset_in_range[-1]

                                in_ = in_batch[minind:maxind]
                                out_ = out_batch[minind:maxind]
                                con_ = cond[minind:maxind]

                                in2_ = make_pianoroll_x(in_, same_onset_ind=same_onset_ind)
                                if len(in2_) < 4:
                                    continue
                                in3_ = make_align_matrix(in_, in2_, same_onset_ind=same_onset_ind) # note2chord
                                in4_ = make_notenum_onehot(in3_) # NumInChord
                                in5_ = get_vertical_position(in_, same_onset_ind=same_onset_ind) # PositionInChord

                                assert in_.shape[0] == in3_.shape[0] == in4_.shape[0]
                                assert in2_.shape[0] == in3_.shape[1]

                                in_ = np.concatenate([in_, in4_, in5_], axis=-1)

                                # save batch
                                savename_x = os.path.join(savepath, '{}.{}.{}.batch_x.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))
                                savename_m = os.path.join(savepath, '{}.{}.{}.batch_m.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))
                                savename_y = os.path.join(savepath, '{}.{}.{}.batch_y.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))  
                                savename_c = os.path.join(savepath, '{}.{}.{}.batch_cond.{}.t{}_d{}.npy'.format(
                                    c_name.lower(), p_name.lower(), pl_name, b_ind, t_ind, d_ind))  

                                assert len(in_) == len(out_) == len(con_)

                                np.save(savename_x, in_)
                                np.save(savename_m, in3_)
                                np.save(savename_y, out_)
                                np.save(savename_c, con_) 

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

def create_h5_dataset(dataset='train', savepath='./data/data_samples'): # save npy files into one hdf5 dataset
    batch_path = "./data/data_samples/features/{}/batch".format(dataset)
    # load filenames
    x_path = [np.string_(x) for x in sorted(glob(os.path.join(batch_path, "*.batch_x.*.t100_d100.npy")))]
    m_path = [np.string_(m) for m in sorted(glob(os.path.join(batch_path, "*.batch_m.*.t100_d100.npy")))]
    y_path = [np.string_(y) for y in sorted(glob(os.path.join(batch_path, "*.batch_y.*.t100_d100.npy")))]
    c_path = [np.string_(c) for c in sorted(glob(os.path.join(batch_path, "*.batch_cond.*.t100_d100.npy")))]

    # save h5py dataset
    f = h5py.File(os.path.join(savepath, "{}.h5".format(dataset)), "w")
    dt = h5py.special_dtype(vlen=str) # save data as string type
    f.create_dataset("x", data=x_path, dtype=dt)
    f.create_dataset("m", data=m_path, dtype=dt)
    f.create_dataset("y", data=y_path, dtype=dt)
    f.create_dataset("c", data=c_path, dtype=dt)
    f.close()






if __name__ == "__main__":
    import logging, sys, argparse
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_dir', type=str, default='./data/data_samples/features',
                        help='input directory (subdirectories are train/val/test)')

    args = parser.parse_args()

    main(dirname=args.input_dir)


