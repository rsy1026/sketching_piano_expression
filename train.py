import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import gc
import sys 
sys.path.append("./parse_utils")
import time
from glob import glob
import pretty_midi 
import h5py
from decimal import Decimal, getcontext, ROUND_HALF_UP
# from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dc = getcontext()
dc.prec = 6
dc.rounding = ROUND_HALF_UP

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Manager
# from torch.multiprocessing import Pool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

from model import PerformGenerator, Mask
from process_data \
    import make_align_matrix, make_note_based, corrupt_to_onset
from parse_utils \
    import extract_midi_notes, save_new_midi, moving_average, \
           make_midi_start_zero, make_pianoroll, ind2str, poly_predict
from piano_cvae_main2_torch_test \
    import features_by_condition_note


# LOAD DATA
class CustomDataset(Dataset):
    def __init__(self, 
                 x=None,
                 m=None,
                 y=None,
                 c=None,
                 same_onset_ind=None):
        manager = Manager()
        self.x = manager.list(x)
        self.m = manager.list(m)
        self.y = manager.list(y)
        self.c = manager.list(c)
        self.same_onset_ind=same_onset_ind

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
       
        x = np.load(self.x[idx]).astype(np.float32)
        m = np.load(self.m[idx]).astype(np.float32)
        c = np.load(self.c[idx]).astype(np.float32)
        y = np.load(self.y[idx]).astype(np.float32)

        # modify features
        vel = y[:,0]
        dur = np.log10(y[:,1])
        ioi1 = np.log10(y[:,2])
        ioi2 = np.log10(y[:,3])
        art = dur - ioi2 # log10(dur/ioi2)

        vel = np.interp(vel, [1, 127], [-1, 1]) # 0 means no change
        art = np.interp(art, [-0.6, 0.6], [-1, 1])
        ioi = np.interp(ioi1, [-0.9, 0.9], [-1, 1]) # IOI 1

        x_ = np.concatenate([x, c[:,19:23]], axis=-1)
        m_ = m
        y_ = np.stack([vel, art, ioi], axis=-1)
        y2_ = corrupt_to_onset(x, y_, same_onset_ind=self.same_onset_ind)
        clab = poly_predict(y2_, N=4)

        return x_, m_, y_, y2_, clab

class PadCollate(object):
    '''
    Ref: https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7
    '''

    def __init__(self, dim=0):
        self.dim = dim

    def pad_matrix(self, inp, pad1, pad2):
        padded = torch.zeros([pad1, pad2])
        padded[:inp.size(0), :inp.size(1)] = inp
        return padded

    def pad_collate(self, batch):
        '''
        x, m, y, y2, c = 0, 1, 2, 3, 4
        '''
        xs = [torch.from_numpy(b[0]) for b in batch]
        ms = [torch.from_numpy(b[1]) for b in batch]
        ys = [torch.from_numpy(b[2]) for b in batch]
        y2s = [torch.from_numpy(b[3]) for b in batch]
        cs = [torch.from_numpy(b[4]) for b in batch]

        # stack all
        xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
        ys = nn.utils.rnn.pad_sequence(ys, batch_first=True)
        y2s = nn.utils.rnn.pad_sequence(y2s, batch_first=True)
        ms = map(lambda x: 
            self.pad_matrix(x, pad1=ys.size(1), pad2=y2s.size(1)), ms)
        ms = torch.stack(list(ms), dim=0)
        cs = nn.utils.rnn.pad_sequence(cs, batch_first=True) 

        return xs, ms, ys, y2s, cs

    def __call__(self, batch):
        return self.pad_collate(batch)



def main():
    start_time = time.time()

    # attributes
    batch_size = 64
    batch_size_val = 16
    total_epoch = 100
    exp_num = '1971' 
    checkpoint_num = '000'
    same_onset_ind = [110,112]
    max_norm = 1.

    # LOAD DATA
    datapath = '/workspace/Piano/gen_task'
    result_path = os.path.join(datapath, 'result_cvae')
    model_path = os.path.join(datapath, 'model_cvae')
    train_data = os.path.join(datapath, 'chopin_cleaned_train_onset_16.h5')
    val_data = os.path.join(datapath, 'chopin_cleaned_val_onset_16.h5')

    with h5py.File(train_data, "r") as f:
        train_x = np.asarray(f["x"])
        train_m = np.asarray(f["m"])
        train_y = np.asarray(f["y"])
        train_c = np.asarray(f["c"])
    with h5py.File(val_data, "r") as f:
        val_x = np.asarray(f["x"])
        val_m = np.asarray(f["m"])
        val_y = np.asarray(f["y"])
        val_c = np.asarray(f["c"])

    train_len = len(train_x)
    val_len = len(val_x)
    step_size = int(np.ceil(train_len / batch_size))

    _time0 = time.time()
    load_data_time = np.round(_time0 - start_time, 3)
    print("LOADED DATA")
    print("__time spent for loading data: {} sec".format(load_data_time))

    # LOAD MODEL
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")

    # generator
    model = PerformGenerator(device=device)
    model.to(device)
    trainer = optim.Adam(model.parameters(), lr=1e-5)
    trainerD = optim.Adam(model.decoder.parameters(), lr=1e-5)

    # model_path_ = "./model_cvae/piano_cvae_ckpt_exp{}_{}".format(exp_num, checkpoint_num)
    # checkpoint = torch.load(model_path_)
    # model.load_state_dict(checkpoint['state_dict'])
    # trainer.load_state_dict(checkpoint['optimizer'])

    # scheduler_G = StepLR(trainer_G, step_size=10, gamma=0.9)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer=trainer, lr_lambda=lambda epoch: 0.95 ** epoch)
    schedulerD = optim.lr_scheduler.LambdaLR(
        optimizer=trainerD, lr_lambda=lambda epoch: 0.95 ** epoch)

    _time1 = time.time()
    load_graph_time = np.round(_time1 - _time0, 3)
    print("LOADED GRAPH")
    print("__time spent for loading graph: {} sec".format(load_graph_time))
    print() 
    print("Start training...")
    print("** step size: {}".format(step_size))
    print()
    bar = 'until next 20th steps: '
    rest = '                    |' 
 

    # TRAIN
    start_train_time = time.time()
    prev_epoch_time = start_train_time
    # model.train()

    loss_list = list()
    loss_list_val = list()
    # loss_list = checkpoint['loss'].tolist()
    # loss_list_val = checkpoint['loss_val'].tolist()

    for epoch in range(int(checkpoint_num), total_epoch):

        # load data loader
        train_dataset = CustomDataset(train_x, train_m, train_y, 
           train_c, same_onset_ind=same_onset_ind)
        val_dataset = CustomDataset(val_x, val_m, val_y, 
            val_c, same_onset_ind=same_onset_ind)

        generator = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=4, 
            collate_fn=PadCollate(), shuffle=True, drop_last=True, pin_memory=False)
        generator_val = DataLoader(
            val_dataset, batch_size=batch_size_val, num_workers=0, 
            collate_fn=PadCollate(), shuffle=False, pin_memory=False)

        epoch += 1
        model.train()
        
        for step, sample in enumerate(generator):

            # load batch
            x, m, y, y2, clab = sample
            # x, m, y, y2, clab = next(iter(generator))
            x = x.float().to(device)
            m = m.float().to(device)
            y = y.float().to(device)
            y2 = y2.float().to(device)
            clab = clab.float().to(device)

            step += 1         

            ## GENERATOR ## 
            trainer.zero_grad()

            # first forward
            s_note, s_group, p_note, \
                z_prior_moments, c_moments, z_moments, \
                c, z, recon_note, recon_group, \
                est_c, est_z, zlab = model(x, y, y2, m, clab)

            # sample (z only)
            # _, _, _, _, _, \
                # sampled_note = model.sample(x, m, c_=c.detach())

            # compute loss 
            mask = Mask(m=m)
            loss, recon_loss, kld_c, kld_z, \
            disc_c, disc_z, reg_loss = loss_fn(
                z_prior_moments, c_moments, z_moments, 
                recon_note, recon_group, y, y2,
                est_c, est_z, c, clab, zlab, mask)  

            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            trainer.step()

            loss_list.append(
                [loss.detach().item(), recon_loss.detach().item(), \
                kld_c.detach().item(), kld_z.detach().item(), \
                disc_c.detach().item(), disc_z.detach().item(), \
                reg_loss.detach().item()])  

            # remove unneeded graph
            loss = None

            # update decoder only
            trainerD.zero_grad()

            # sample (z only)
            sampled_note_ = model.sample_decoder(
                s_note, s_group, m, c_=c)
            new_c = model.sample_c_only(sampled_note_, m)
            est_c2 = model.predict_EP(new_c, m)

            disc_d = lossD_fn(est_c2, clab, mask) 

            disc_d.backward()
            nn.utils.clip_grad_norm_(model.decoder.parameters(), max_norm)
            trainerD.step()


            if step % 1 == 0:
                bar += '='
                rest = rest[1:]
                print(bar+'>'+rest, end='\r')

            if step % 20 == 0:
                # print losses 
                print()
                print("[EXP {} --> epoch: {} / step: {}]\n".format(exp_num, epoch, step) + \
                "   --GENERATOR LOSS--\n" + \
                "           recon_loss: {:06.4f} / kld_c: {:06.4f} / kld_z: {:06.4f}\n".format(
                    -recon_loss, kld_c, kld_z) + \
                "           disc_c: {:06.4f} / disc_z: {:06.4f}\n".format(-disc_c, -disc_z) + \
                "           disc_d: {:06.4f}\n".format(-disc_d) + \
                "           reg_loss: {:06.4f}\n".format(-reg_loss))
                print()
                bar = 'until next 20th steps: '
                rest = '                    |'

            if step % 100 == 0:
                
                roll = model.pianoroll(x, m)
                x = x.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                y2 = y2.cpu().detach().numpy()
                clab = clab.cpu().detach().numpy()
                m = m.cpu().detach().numpy()

                recon_note = recon_note.cpu().detach().numpy()
                recon_group = recon_group[0].cpu().detach().numpy()
                sampled_note = sampled_note_.cpu().detach().numpy()

                y_vel = y[0,:,0]
                y_art = y[0,:,1]
                y_ioi = y[0,:,-1]
                y_pred_vel = recon_note[0,:,0]
                y_pred_art = recon_note[0,:,1]
                y_pred_ioi = recon_note[0,:,-1]
                y_sampled_vel = sampled_note[0,:,0]
                y_sampled_art = sampled_note[0,:,1]
                y_sampled_ioi = sampled_note[0,:,-1]
                # recon_group = corrupt_to_onset(x[0], recon_note[0], same_onset_ind=same_onset_ind)

                s_note_ = s_note[0].cpu().detach().numpy()
                p_note_ = p_note[0].cpu().detach().numpy()
                z_mu = z_moments[0][0].cpu().detach().numpy()
                z_prior_mu = z_prior_moments[0][0].cpu().detach().numpy()
                # att_ = s_attn[0][0].cpu().detach().numpy()
                est_z_ = est_z[0].cpu().detach().numpy()
                inf_c_ = est_c2[0].cpu().detach().numpy()
                target_c_ = clab[0]
                zlab_ = zlab[0].cpu().detach().numpy()
                roll_ = roll[0].cpu().detach().numpy()

                # save results plot
                plt.figure(figsize=(10,12))
                gs = gridspec.GridSpec(nrows=7, ncols=2,
                       height_ratios=[1,1,1,1,1,1,1],
                       width_ratios=[1,1])
                plt.subplot(gs[0,0])
                plt.title("score note")
                plt.imshow(np.transpose(s_note_), aspect='auto')
                plt.colorbar()
                plt.subplot(gs[0,1])
                plt.title("perform note")
                plt.imshow(np.transpose(p_note_), aspect='auto')
                plt.colorbar()
                plt.subplot(gs[1,0])
                plt.title("inferred c")
                plt.imshow(np.transpose(inf_c_), aspect='auto')
                plt.colorbar()
                plt.subplot(gs[1,1])
                plt.title("target c")
                plt.imshow(np.transpose(target_c_), aspect='auto')
                plt.colorbar()
                # plt.imshow(att_, aspect='auto')
                # plt.colorbar()
                plt.subplot(gs[2,0])
                plt.title("est z")
                plt.imshow(np.transpose(est_z_), aspect='auto')
                plt.colorbar()
                plt.subplot(gs[2,1])
                plt.title("z label")
                plt.imshow(np.transpose(zlab_), aspect='auto')
                plt.colorbar()
                plt.subplot(gs[3,0])
                plt.title("recon group")
                plt.imshow(np.transpose(recon_group), aspect='auto')
                plt.colorbar()
                plt.subplot(gs[3,1])
                plt.title("score roll")
                plt.imshow(np.transpose(roll_), aspect='auto')
                plt.colorbar()
                plt.subplot(gs[4,0])
                plt.title("Z mu")
                plt.imshow(np.transpose(z_mu), aspect='auto')
                plt.colorbar()
                plt.subplot(gs[4,1])
                plt.title("Z prior mu")
                plt.imshow(np.transpose(z_prior_mu), aspect='auto')
                plt.colorbar()                  
                plt.subplot(gs[5,0])
                plt.title("Velocity")
                plt.plot(range(len(y_vel)), y_vel, label="GT")
                plt.plot(range(len(y_pred_vel)), y_pred_vel, label="Pred")
                plt.plot(range(len(y_pred_vel)), y_sampled_vel, label="Sampled", alpha=0.5)
                plt.legend()
                plt.subplot(gs[6,0])
                plt.title("Articulation")
                plt.plot(range(len(y_vel)), y_art, label="GT")
                plt.plot(range(len(y_pred_vel)), y_pred_art, label="Pred")
                plt.plot(range(len(y_pred_vel)), y_sampled_art, label="Sampled", alpha=0.5)
                plt.legend()
                plt.subplot(gs[6,1])
                plt.title("IOI")
                plt.plot(range(len(y_ioi)), y_ioi, label="GT")
                plt.plot(range(len(y_pred_ioi)), y_pred_ioi, label="Pred")
                plt.plot(range(len(y_pred_ioi)), y_sampled_ioi, label="Sampled", alpha=0.5)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(
                    result_path, "exp{}_train_epoch{}_step{}.png".format(exp_num, epoch, step)))
                plt.close()

            gc.collect()
 
        _time2 = time.time()
        epoch_time = np.round(_time2 - prev_epoch_time, 3)
        print()
        print()
        print("------------------EPOCH {} finished------------------".format(epoch))
        print()
        print("==> time spent for this epoch: {} sec".format(epoch_time))
        # print("==> loss: {:06.4f}".format(loss))  

        model.eval()
        with torch.no_grad():
            
            Xv, Mv, Yv, Yv2, Cv = next(iter(generator_val))
            Xv = Xv.float().to(device)
            Mv = Mv.float().to(device)
            Yv = Yv.float().to(device)
            Yv2 = Yv2.float().to(device)
            Cv = Cv.float().to(device)

            # forward
            s_note, s_group, p_note, \
                z_prior_moments, c_moments, z_moments, \
                c, z, recon_note, recon_group, \
                est_c, est_z, zlab = model(Xv, Yv, Yv2, Mv, Cv)

            # sample   
            # _, _, _, _, _, \
                # sampled_note = model.sample(Xv, Mv, c_=c.detach())

            # sample (z only)
            sampled_note_ = model.sample_decoder(
                s_note, s_group, Mv, c_=c)
            new_c = model.sample_c_only(sampled_note_, Mv)
            est_c2 = model.predict_EP(new_c, Mv)

            # LOSS
            mask = Mask(m=Mv)
            val_loss, recon_loss, kld_c, kld_z, \
            disc_c, disc_z, reg_loss = loss_fn(
                z_prior_moments, c_moments, z_moments, 
                recon_note, recon_group, Yv, Yv2,
                est_c, est_z, c, Cv, zlab, mask)  

            disc_d = lossD_fn(est_c2, Cv, mask)

            loss_list_val.append(
                [val_loss.detach().item(), recon_loss.detach().item(), \
                kld_c.detach().item(), kld_z.detach().item(), \
                disc_c.detach().item(), disc_z.detach().item(), \
                disc_d.detach().item(), reg_loss.detach().item()]) 

            # print losses
            print()
            print()
            print("==> [EXP {} --> epoch {}] Validation:\n".format(exp_num, epoch) + \
                    "   --GENERATOR LOSS--\n" + \
                    "           recon_loss: {:06.4f} / kld_c: {:06.4f} / kld_z: {:06.4f}\n".format(
                        -recon_loss, kld_c, kld_z) + \
                    "           disc_c: {:06.4f} / disc_z: {:06.4f}\n".format(-disc_c, -disc_z) + \
                    "           disc_d: {:06.4f}\n".format(-disc_d) + \
                    "           reg_loss: {:06.4f}\n".format(-reg_loss))    
            print()
            print("------------------EPOCH {} finished------------------".format(epoch))
            print()
            
            # inference
            Xv_ = Xv[0].unsqueeze(0)
            Mv_ = Mv[0].unsqueeze(0)
            Yv_ = Yv[0].unsqueeze(0)
            Yv2_ = Yv2[0].unsqueeze(0)

            test_path = '/data/chopin_cleaned/exp_data/val/batch_onset_16'
            piece_name = 'chopin_etude.25_2.01'
            batch_name = '0001.t100_d100'
            test_x_path = os.path.join(test_path, "{}.batch_x.{}.npy".format(piece_name, batch_name))
            test_y_path = os.path.join(test_path, "{}.batch_y.{}.npy".format(piece_name, batch_name))
            test_m_path = os.path.join(test_path, "{}.batch_m.{}.npy".format(piece_name, batch_name))
            test_c_path = os.path.join(test_path, "{}.batch_cond.{}.npy".format(piece_name, batch_name))

            _x = np.load(test_x_path)
            _y = np.load(test_y_path)
            _c = np.load(test_c_path)
            test_x = np.concatenate([_x, _c[:,19:23]], axis=-1)
            test_m = np.load(test_m_path)

            # modify features
            test_y_f = features_by_condition_note(_y, x=test_x, 
                cond="fast", art=True, same_onset_ind=same_onset_ind)
            test_y_s = features_by_condition_note(_y, x=test_x, 
                cond="slow", art=True, same_onset_ind=same_onset_ind)
            test_y_q = features_by_condition_note(_y, x=test_x, 
                cond="quiet", art=True, same_onset_ind=same_onset_ind)
            test_y_l = features_by_condition_note(_y, x=test_x, 
                cond="loud", art=True, same_onset_ind=same_onset_ind)    
                
            test_ys = [test_y_f, test_y_s, test_y_q, test_y_l]
            indices = ["fast", "slow", "quiet", "loud"]
            c_dict = dict()

            # get latent variables by conditions
            for y, ind in zip(test_ys, indices):

                vel_, art_, ioi_ = y[:,0], y[:,1], y[:,2]
                test_y = np.stack([vel_, art_, ioi_], axis=-1)
                test_y2 = corrupt_to_onset(test_x, test_y, same_onset_ind=same_onset_ind)
                clab_test = moving_average(test_y2, win_len=5, stat=np.mean, half=False)

                # convert to tensor
                test_x_ = torch.from_numpy(test_x.astype(np.float32)).to(device).unsqueeze(0)
                test_m_ = torch.from_numpy(test_m.astype(np.float32)).to(device).unsqueeze(0)
                test_y_ = torch.from_numpy(test_y.astype(np.float32)).to(device).unsqueeze(0)
                test_y2_ = torch.from_numpy(test_y2.astype(np.float32)).to(device).unsqueeze(0)
                test_clab_ = torch.from_numpy(clab_test.astype(np.float32)).to(device).unsqueeze(0)

                # sample Z prior
                _, _, _, \
                    _, c_moments, _, \
                    _, _, _, _, \
                    _, _, _ = model(test_x_, test_y_, test_y2_, test_m_, test_clab_)

                c_dict[ind] = c_moments[0]
            

            # inference   
            c_rand = torch.randn(1, test_y2_.size(1), 12).to(device)
            _, z_moments0, z0, \
                sampled0_note = model.sample(Xv_, Mv_, c_=c_rand)

            # sample by conditions
            c_seed1 = c_dict['fast']
            c_seed1[:,:,8] += 3
            _, _, _, \
                sampled1_note = model.sample(Xv_, Mv_, c_=c_seed1, z_=z0)
            c_seed2 = c_dict['slow']
            c_seed2[:,:,8] -= 3
            _, _, _, \
                sampled2_note = model.sample(Xv_, Mv_, c_=c_seed2, z_=z0)
            c_seed3 = c_dict['loud']
            c_seed3[:,:,0] += 3
            _, _, _, \
                sampled3_note = model.sample(Xv_, Mv_, c_=c_seed3, z_=z0)
            c_seed4 = c_dict['quiet']
            c_seed4[:,:,0] -= 3
            _, _, _, \
                sampled4_note = model.sample(Xv_, Mv_, c_=c_seed4, z_=z0)

            # save results plot  
            roll = model.pianoroll(Xv, Mv)
            Xv_ = Xv_.cpu().detach().numpy()
            Yv_ = Yv_.cpu().detach().numpy()
            Yv2_ = Yv2.cpu().detach().numpy()
            Mv_ = Mv.cpu().detach().numpy()

            recon_note = recon_note.cpu().detach().numpy()

            y_vel = Yv_[0,:,0]
            y_art = Yv_[0,:,1]
            y_ioi = Yv_[0,:,-1]
            y_pred_vel = recon_note[0,:,0]
            y_pred_art = recon_note[0,:,1]
            y_pred_ioi = recon_note[0,:,-1]
            y_sampled0 = sampled0_note[0].cpu().detach().numpy()
            y_sampled1 = sampled1_note[0].cpu().detach().numpy()
            y_sampled2 = sampled2_note[0].cpu().detach().numpy()
            y_sampled3 = sampled3_note[0].cpu().detach().numpy()
            y_sampled4 = sampled4_note[0].cpu().detach().numpy()
            s_note_ = s_note[0].cpu().detach().numpy()
            p_note_ = p_note[0].cpu().detach().numpy()
            # att_ = s_attn[0][0].cpu().detach().numpy()
            z_mu = z_moments0[0][0].cpu().detach().numpy()
            z_ = z0[0].cpu().detach().numpy()
            roll_ = roll[0].cpu().detach().numpy()

            
            # save results plot
            plt.figure(figsize=(10,15))
            gs = gridspec.GridSpec(nrows=5, ncols=2,
                height_ratios=[1,1,1,1,1],
                width_ratios=[1,1])
            plt.subplot(gs[0,0])
            plt.title("perform note")
            plt.imshow(np.transpose(p_note_), aspect='auto')
            plt.colorbar()
            # plt.subplot(gs[0,1])
            # plt.title("self-attention")
            # plt.imshow(np.transpose(att_), aspect='auto')
            # plt.colorbar()
            plt.subplot(gs[1,0])
            plt.title("score note")
            plt.imshow(np.transpose(s_note_), aspect='auto')
            plt.colorbar()
            plt.subplot(gs[1,1])
            plt.title("roll")
            plt.imshow(np.transpose(roll_), aspect='auto')
            plt.colorbar()
            plt.subplot(gs[2,0])
            plt.title("z mu")
            plt.imshow(np.transpose(z_mu), aspect='auto')
            plt.colorbar()
            plt.subplot(gs[2,1])
            plt.title("z sampled")
            plt.imshow(np.transpose(z_), aspect='auto')
            plt.colorbar()
            plt.subplot(gs[3,0])
            plt.title("Predicted/generated velocity")
            plt.plot(range(len(y_vel)), y_vel, label="GT")
            plt.plot(range(len(y_vel)), y_sampled0[:,0], label="z sampled")
            plt.plot(range(len(y_vel)), y_sampled1[:,0], label="c trans (1st -3)")
            plt.plot(range(len(y_vel)), y_sampled2[:,0], label="c trans (1st +3)")
            plt.plot(range(len(y_vel)), y_sampled3[:,0], label="c trans (4th -3)")
            plt.plot(range(len(y_vel)), y_sampled4[:,0], label="c trans (4th +3)")
            plt.plot(range(len(y_pred_vel)), y_pred_vel, label="Pred")
            plt.legend()
            plt.subplot(gs[4,0])
            plt.title("Predicted/generated articulation")
            plt.plot(range(len(y_vel)), y_art, label="GT")
            plt.plot(range(len(y_vel)), y_sampled0[:,1], label="z sampled")
            plt.plot(range(len(y_vel)), y_sampled1[:,1], label="c trans (1st -3)")
            plt.plot(range(len(y_vel)), y_sampled2[:,1], label="c trans (1st +3)")
            plt.plot(range(len(y_vel)), y_sampled3[:,1], label="c trans (4th -3)")
            plt.plot(range(len(y_vel)), y_sampled4[:,1], label="c trans (4th +3)")
            plt.plot(range(len(y_pred_vel)), y_pred_art, label="Pred")
            plt.legend()
            plt.subplot(gs[4,1])
            plt.title("Predicted/generated ioi")
            plt.plot(range(len(y_ioi)), y_ioi, label="GT")
            plt.plot(range(len(y_ioi)), y_sampled0[:,-1], label="z sampled")
            plt.plot(range(len(y_ioi)), y_sampled1[:,-1], label="c trans (1st -3)")
            plt.plot(range(len(y_ioi)), y_sampled2[:,-1], label="c trans (1st +3)")
            plt.plot(range(len(y_ioi)), y_sampled3[:,-1], label="c trans (4th -3)")
            plt.plot(range(len(y_ioi)), y_sampled4[:,-1], label="c trans (4th +3)")
            plt.plot(range(len(y_pred_ioi)), y_pred_ioi, label="Pred")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(result_path, "exp{}_val_epoch{}.png".format(exp_num, epoch)))
            plt.close()

            print()
            print("------------------TEST finished------------------".format(epoch))
            print()
            print()

        scheduler.step()
        schedulerD.step()
        
        

        # save checkpoint & loss
        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': trainer.state_dict(),
                'optimizerD': trainerD.state_dict(),
                'loss': np.asarray(loss_list),
                'loss_val': np.asarray(loss_list_val)},
                os.path.join(model_path, "piano_cvae_ckpt_exp{}".format(
                    exp_num)))

            _time3 = time.time()
            end_train_time = np.round(_time3 - start_train_time, 3)  
            print("__time spent for entire training: {} sec".format(end_train_time))

        prev_epoch_time = time.time()     

def lossD_fn(est_c, clab, mask):

    # sampled_c = torch.mean(sampled_note, dim=1)
    disc_d = torch.mean(mask((est_c - clab)**2))

    return disc_d

def loss_fn(
    z_prior_moments, c_moments, z_moments, 
    recon_note, recon_group, y_target, y_target2,
    est_c, est_z, c_, clab, zlab, mask):

    # VAE losses
    kld_c = torch.mean(mask(kld(*c_moments)))
    kld_z = torch.mean(mask(kld(*(z_moments + z_prior_moments))))
    recon_loss1 = -torch.mean(mask(torch.abs(recon_note - y_target))) # l2 loss
    recon_loss2 = -torch.mean(mask(torch.abs(recon_group - y_target2))) # l2 loss
    recon_loss = recon_loss1 + recon_loss2

    disc_c = -torch.mean(mask((est_c - clab)**2))
    # disc_z = -torch.mean(torch.sum(mask(torch.abs(est_z - zlab)), dim=-1))

    dist = (est_z - zlab)**2
    # dist_tri = torch.triu(dist)
    disc_z = -torch.mean(mask(dist))

    # sampled_c = torch.mean(sampled_note, dim=1)
    # disc_d = -torch.mean((inf_c - c_moments[0])**2)

    # regression loss
    M, t = c_.size(0), c_.size(1) # batch size
    c_ = mask(c_)
    s = torch.stack([c_[:,:,0], c_[:,:,4], c_[:,:,8]], dim=-1)
    s_l = mask(clab)
    s1 = s.unsqueeze(0).expand(M, M, t, -1).reshape(M, M, -1)
    s2 = s.unsqueeze(1).expand(M, M, t, -1).reshape(M, M, -1)
    s_l1 = s_l.unsqueeze(0).expand(M, M, t, -1).reshape(M, M, -1)
    s_l2 = s_l.unsqueeze(1).expand(M, M, t, -1).reshape(M, M, -1)
    s_D = s1 - s2 
    s_l_D = s_l1 - s_l2 
    reg_dist = (torch.tanh(s_D) - torch.sign(s_l_D))**2
    reg_loss = -torch.mean(reg_dist)

    # VAE ELBO
    elbo = recon_loss - kld_c - kld_z + 100*disc_z + 1000*disc_c + 10*reg_loss

    # total loss
    total_loss = -elbo # negative to minimize

    return total_loss, recon_loss, kld_c, kld_z, disc_c, disc_z, reg_loss

def kld(mu, logvar, q_mu=None, q_logvar=None):
    '''
    KL(q(z2|x)||p(z2|u2)))(expectation along q(u2))
        --> b/c p(z2) depends on p(u2) (p(z2|u2))
        --> -0.5 * (1 + logvar - q_logvar 
            - (exp(logvar) + (mu - q_mu)^2) / exp(q_logvar)) 
    '''
    if q_mu is None:
        q_mu = torch.zeros_like(mu)
    if q_logvar is None:
        q_logvar = torch.zeros_like(logvar)

    return -0.5 * (1 + logvar - q_logvar - \
        (torch.pow(mu - q_mu, 2) + torch.exp(logvar)) / torch.exp(q_logvar))





if __name__ == "__main__":
    main()



