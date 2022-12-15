import numpy as np
import os
import gc
import time
from glob import glob
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Manager

from model.piano_model import (
    PerformGenerator, 
    Mask
)
from data.make_batches import corrupt_to_onset
from data.parse_features import make_onset_based_all
from sketching_piano_expression.utils.parse_utils import poly_predict
from generate import features_by_condition



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
    total_epoch = 100
    exp_num = '000' 
    checkpoint_num = '000'
    same_onset_ind = [110,112]
    max_norm = 1.

    # LOAD DATA
    data_path = './data/data_samples'
    model_path = './model/model_ckpts'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train_data = os.path.join(data_path, 'train.h5')
    # val_data = os.path.join(data_path, 'val.h5')

    with h5py.File(train_data, "r") as f:
        train_x = np.asarray(f["x"])
        train_m = np.asarray(f["m"])
        train_y = np.asarray(f["y"])
        train_c = np.asarray(f["c"])
    # with h5py.File(val_data, "r") as f:
    #     val_x = np.asarray(f["x"])
    #     val_m = np.asarray(f["m"])
    #     val_y = np.asarray(f["y"])
    #     val_c = np.asarray(f["c"])

    train_len = len(train_x)
    # val_len = len(val_x)
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
    loss_list = list()
    loss_list_val = list()

    # load data loader
    train_dataset = CustomDataset(train_x, train_m, train_y, 
        train_c, same_onset_ind=same_onset_ind)
    # val_dataset = CustomDataset(val_x, val_m, val_y, 
        # val_c, same_onset_ind=same_onset_ind)

    generator = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, 
        collate_fn=PadCollate(), shuffle=True, drop_last=True, pin_memory=False)
    # generator_val = DataLoader(
        # val_dataset, batch_size=batch_size_val, num_workers=0, 
        # collate_fn=PadCollate(), shuffle=False, pin_memory=False)

    for epoch in range(int(checkpoint_num), total_epoch):

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

            gc.collect()
 
        _time2 = time.time()
        epoch_time = np.round(_time2 - prev_epoch_time, 3)
        print()
        print()
        print("------------------EPOCH {} finished------------------".format(epoch))
        print()
        print("==> time spent for this epoch: {} sec".format(epoch_time))
        # print("==> loss: {:06.4f}".format(loss))  

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


# LOSS FUNCTIONS
def lossD_fn(est_c, clab, mask):

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
    disc_z = -torch.mean(mask((est_z - zlab)**2))

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
    elbo = recon_loss - kld_c - kld_z + 1000*disc_c + 100*disc_z + 10*reg_loss

    # total loss
    total_loss = -elbo # negative to minimize

    return total_loss, recon_loss, kld_c, kld_z, disc_c, disc_z, reg_loss

def kld(mu, logvar, q_mu=None, q_logvar=None):
    '''
    KL(N(mu, var)||N(qmu, qvar))
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



