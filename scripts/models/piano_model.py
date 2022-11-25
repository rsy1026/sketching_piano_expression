from .modules import *



# MODELS
class ScoreEncoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super(ScoreEncoder, self).__init__()
        # layers
        self.linear1 = MaskedFCBlock_LN(in_dim, hidden)
        self.linear2 = MaskedFCBlock_LN(hidden, hidden)
        self.note2group = Note2Group()

    def forward(self, x, m, mask):
        # encoders
        note = self.linear1(x, mask)
        note = self.linear2(note, mask)
        group = self.note2group(note, m)

        return note, group

class PerformEncoder(nn.Module):
    def __init__(self, in_dim, hidden):
        super(PerformEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden = hidden 
        # layers
        self.conv1 = nn.Conv1d(in_dim, hidden, 1, 1, 0)
        self.conv2 = MaskedCausalConvBlock(hidden, hidden, 3, 1, 1, dropout=0.2)
        self.conv3 = MaskedCausalConvBlock(hidden, hidden, 3, 1, 1, dropout=0.2)
        self.note2group = Note2Group()
        self.linear = MaskedFCBlock(hidden, hidden, dropout=0.2)

    def forward(self, p, m, mask):
        # encoders
        note = self.conv1(p.transpose(1, 2))
        note = self.conv2(note, mask)
        note = self.conv3(note, mask)
        group = self.note2group(note.transpose(1, 2), m)
        group = self.linear(group, mask)
        return note, group 

class LatentCEncoder(nn.Module):
    def __init__(self, l_dim, hidden):
        super(LatentCEncoder, self).__init__()
        self.d = hidden
        # layers
        self.bigru = nn.GRU(hidden, hidden, 
            batch_first=True, bidirectional=True)
        self.enc = MaskedFCBlock(hidden*2, hidden)
        self.mu = nn.Linear(hidden, l_dim)
        self.logvar = nn.Linear(hidden, l_dim)
        self.reparam = Reparameterize()

    def forward(self, p, mask):
        out, _ = self.bigru(p)
        enc = self.enc(out, mask)
        mu = self.mu(enc)
        logvar = self.logvar(enc)
        s = self.reparam(mu, logvar)
        return [mu, logvar], s

class LatentZEncoder(nn.Module):
    def __init__(self, l_dim, hidden, device):
        super(LatentZEncoder, self).__init__()
        self.d_l = l_dim
        self.d_model = hidden
        self.device = device
        # layers
        self.gru = nn.GRU(hidden+hidden//2, hidden, batch_first=True)
        self.enc = MaskedFCBlock_LN(hidden, hidden)
        self.mu = nn.Linear(hidden, l_dim)
        self.logvar = nn.Linear(hidden, l_dim)
        self.phi_z = FCBlock(l_dim, hidden, batchnorm=False)
        self.prior_gru = nn.GRUCell(hidden+hidden//2, hidden)
        self.prior_enc = FCBlock_LN(hidden, hidden)
        self.prior_mu = nn.Linear(hidden, l_dim)
        self.prior_logvar = nn.Linear(hidden, l_dim)
        self.trunc = TruncatedNorm()
        self.reparam = Reparameterize(device=device)

    def forward(self, p, s, mask):
        inputs = torch.cat([p, s], dim=-1)
        gru, _ = self.gru(inputs)
        enc = self.enc(gru, mask)
        mu = self.mu(enc)
        logvar = self.logvar(enc)
        z = self.reparam(mu, logvar)
        return [mu, logvar], z

    def prior(self, s, trunc=False, threshold=None):
        # initialize
        n, t = s.size(0), s.size(1)
        z_t = torch.zeros(n, 1, self.d_l).to(self.device)
        h_t = torch.zeros(n, self.d_model).to(self.device)
        mus, logvars, zs = list(), list(), list()

        # vrnn loop for z
        for i in range(t):
            z_t = self.phi_z(z_t)
            x_t = torch.cat([z_t.squeeze(1), s[:,i]], dim=-1)
            # rnn
            h_t = self.prior_gru(x_t, h_t)
            # encode z prior
            enc = self.prior_enc(h_t.unsqueeze(1))
            mu = self.prior_mu(enc)
            logvar = self.prior_logvar(enc)         
            z_t = self.reparam(mu, logvar, trunc=trunc, threshold=threshold)
            # gather
            mus.append(mu)
            logvars.append(logvar)
            zs.append(z_t)

        mus = torch.cat(mus, dim=1)
        logvars = torch.cat(logvars, dim=1)
        z = torch.cat(zs, dim=1)
        return [mus, logvars], z

class Decoder(nn.Module):
    def __init__(self, p_dim, c_dim, z_dim, hidden, device):
        super(Decoder, self).__init__()
        h_hidden = hidden//2
        self.p_dim = p_dim 
        self.device = device
        # layers
        self.group_decoder = MaskedFCBlock(hidden*2+hidden//2, hidden, dropout=0.2)
        self.group_final = nn.Sequential(
            nn.Linear(hidden, p_dim),
            nn.Tanh())
        self.conv1 = MaskedCausalConvBlock(hidden+hidden//2+p_dim, hidden, 3, 1, 1, dropout=0.2)
        self.conv2 = MaskedCausalConvBlock(hidden, hidden, 3, 1, 1, dropout=0.2)
        self.conv3 = MaskedConvBlock(hidden, p_dim, 1, 1, 0, 
                batchnorm=False, nonlinearity=nn.Tanh())
        self.note2group = Note2Group()
        self.proj_c = MaskedFCBlock(c_dim, hidden)
        self.proj_z = MaskedFCBlock_LN(z_dim, hidden)

    def forward(self, s_note, s_group, p, c, z, m, mask=None):
        group = torch.cat([s_group, self.proj_c(c, mask), self.proj_z(z, mask)], dim=-1)
        group = self.group_decoder(group, mask)
        est_group = self.group_final(group)
        g_expand = self.note2group.reverse(group, m)

        p_train = torch.cat([torch.zeros_like(p[:,:1]), p[:,:-1]], dim=1)
        note_in = torch.cat([s_note, g_expand, p_train], dim=-1)
        note = self.conv1(note_in.transpose(1, 2), mask)
        note = self.conv2(note, mask)
        note = self.conv3(note, mask)
        est_note = note.transpose(1, 2)
        return est_note, est_group

    def test(self, s_note, s_group, c, z, m, mask=None):
        c_proj = self.proj_c(c, train=False)
        z_proj = self.proj_z(z, train=False)
        group = torch.cat([s_group, c_proj, z_proj], dim=-1)
        group = self.group_decoder(group, train=False)
        g_expand = self.note2group.reverse(group, m)

        n, t = s_note.size(0), s_note.size(1)
        init = torch.zeros(n, 1, self.p_dim).to(self.device)
        notes = list()
        for i in range(t):
            note_in = torch.cat([s_note[:,:i+1], g_expand[:,:i+1], init], dim=-1)
            note = self.conv1(note_in.transpose(1, 2), train=False)
            note = self.conv2(note, train=False)
            note = self.conv3(note, train=False)
            init = torch.cat([init, note[:,:,-1].unsqueeze(1)], dim=1)
            notes.append(note[:,:,-1])
        notes = torch.stack(notes, dim=1)
        return notes


class PredictC1(nn.Module):
    def __init__(self, c_dim, hidden, p_dim):
        super(PredictC1, self).__init__()
        # layers
        self.linear = MaskedFCBlock(c_dim, hidden)
        self.predict = nn.Sequential(
            nn.Linear(hidden, p_dim),
            nn.Tanh())

    def forward(self, c, mask):
        c = self.linear(c, mask)
        est_c = self.predict(c)
        return est_c

class PredictC2(nn.Module):
    def __init__(self, c_dim, hidden, p_dim):
        super(PredictC2, self).__init__()
        # layers
        self.linear = MaskedFCBlock(c_dim, hidden)
        self.predict = nn.Sequential(
            nn.Linear(hidden, p_dim),
            nn.Tanh())

    def forward(self, c, mask):
        c = self.linear(c, mask)
        est_c = self.predict(c)
        return est_c

class PredictC3(nn.Module):
    def __init__(self, c_dim, hidden, p_dim):
        super(PredictC3, self).__init__()
        # layers
        self.linear = MaskedFCBlock(c_dim, hidden)
        self.predict = nn.Sequential(
            nn.Linear(hidden, p_dim),
            nn.Tanh())

    def forward(self, c, mask):
        c = self.linear(c, mask)
        est_c = self.predict(c)
        return est_c

class PredictZ(nn.Module):
    def __init__(self, z_dim, hidden, p_dim):
        super(PredictZ, self).__init__()
        # layers
        self.linear = MaskedFCBlock_LN(z_dim, hidden)
        self.predict = nn.Sequential(
            nn.Linear(hidden, p_dim),
            nn.Tanh())
        self.label = StyleEstimator() 

    def forward(self, z, y, clab, m, mask):
        z = self.linear(z, mask)
        est_z = self.predict(z)
        zlab = self.label(y, clab)
        return zlab, est_z


class PerformGenerator(nn.Module):
    def __init__(self,
                 s_dim=138,
                 p_dim=3,
                 c_dim=12,
                 z_dim=64,
                 hidden=256,
                 a_hidden=128,
                 attn_heads=8,
                 device=None):
        super(PerformGenerator, self).__init__()
        self.c_dim = c_dim
        self.device = device 

        # layers
        self.score_encoder = ScoreEncoder(in_dim=s_dim, hidden=hidden//2)
        self.perform_encoder = PerformEncoder(in_dim=p_dim, hidden=hidden)
        self.c_encoder = LatentCEncoder(l_dim=c_dim, hidden=hidden)
        self.z_encoder = LatentZEncoder(l_dim=z_dim, hidden=hidden, device=device)
        self.decoder = Decoder(p_dim=p_dim, c_dim=c_dim, z_dim=z_dim, hidden=hidden, device=device)
        self.predict_c1 = PredictC1(c_dim=4, hidden=hidden, p_dim=1) 
        self.predict_c2 = PredictC2(c_dim=4, hidden=hidden, p_dim=1) 
        self.predict_c3 = PredictC3(c_dim=4, hidden=hidden, p_dim=1) 
        self.predict_z = PredictZ(z_dim=z_dim, hidden=hidden, p_dim=3) 
        self.split_score = SplitScore()

    def forward(self, s, p, p2, m, clab):
        # mask
        mask = Mask(m=m)
        ## Condition ##
        s = self.split_score(s)
        s_note, s_group = self.score_encoder(s, m, mask)
        ## Inference ##
        p_note, p_group = self.perform_encoder(p, m, mask)
        c_moments, c = self.c_encoder(p_group, mask)
        z_moments, z = self.z_encoder(p_group, s_group, mask)
        z_prior_moments, z_prior = self.z_encoder.prior(s_group)
        ## Decoder ##
        est_note, est_group = self.decoder(s_note, s_group, p, c, z, m, mask)

        ## Auxiliary tasks ##
        est_c1 = self.predict_c1(c[:,:,:4], mask)
        est_c2 = self.predict_c2(c[:,:,4:8], mask)
        est_c3 = self.predict_c3(c[:,:,8:12], mask)
        est_c = torch.cat([est_c1, est_c2, est_c3], dim=-1)
        zlab, est_z = self.predict_z(z, p2, clab, m, mask)

        return s_note, s_group, p_note, \
            z_prior_moments, c_moments, z_moments, \
            c, z, est_note, est_group, est_c, est_z, zlab

    def sample(self, s, m, c_=None, z_=None, trunc=False, threshold=2.):
        # mask
        mask = Mask(m=m)
        ## Condition ##
        s = self.split_score(s)
        s_note, s_group = self.score_encoder(s, m, mask)
        ## Sample ## 
        c = torch.randn(s.size(0), m.size(-1), self.c_dim).to(self.device)
        if trunc is True:
            trunc = TruncatedNorm()
            c = torch.FloatTensor(
                trunc([s.size(0), m.size(-1), self.c_dim], 
                threshold=threshold)).to(self.device)
        z_moments, z = self.z_encoder.prior(
            s_group, trunc=trunc, threshold=threshold)
        # transfer 
        if c_ is not None:
            c = c_
        if z_ is not None:
            z = z_
        ## Decoder ##
        est_note = self.decoder.test(s_note, s_group, c, z, m, mask)

        return s_note, z_moments, z, est_note

    def predict_EP(self, c, m):
        mask = Mask(m=m)
        est_c1 = self.predict_c1(c[:,:,:4], mask)
        est_c2 = self.predict_c2(c[:,:,4:8], mask)
        est_c3 = self.predict_c3(c[:,:,8:12], mask)
        est_c = torch.cat([est_c1, est_c2, est_c3], dim=-1)
        return est_c 

    def sample_z_only(self, s, p, m):
        # mask
        mask = Mask(m=m)
        ## Condition ##
        s = self.split_score(s)
        s_note, s_group = self.score_encoder(s, m, mask)
        ## Inference ##
        p_note, p_group = self.perform_encoder(p, m, mask)
        z_moments, z = self.z_encoder(p_group, s_group, mask)

        return z_moments, z

    def sample_decoder(self, s_note, s_group, m, c_=None):
        # mask
        mask = Mask(m=m)
        ## Sample ##
        z_moments, z = self.z_encoder.prior(s_group.detach())
        # z_roll = torch.roll(z, 1, 0)
        ## Decoder ##
        est_note = self.decoder.test(s_note.detach(), s_group.detach(), 
            c_.detach(), z.detach(), m, mask)

        return est_note

    def infer_c_only(self, p, m):
        mask = Mask(m=m)
        ## Inference ##
        p_note, p_group = self.perform_encoder(p, m, mask)
        c_moments, c = self.c_encoder(p_group, mask)

        return c_moments[0]

    def sample_c_only(self, p, m):
        mask = Mask(m=m)
        ## Inference ##
        p_note, p_group = self.perform_encoder(p, m, mask)
        c_moments, c = self.c_encoder(p_group, mask)

        return c
        
    def pianoroll(self, x, m):
        roll = torch.matmul(
            x.transpose(1, 2), m).transpose(1, 2)
        return roll


