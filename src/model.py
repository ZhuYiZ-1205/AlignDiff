import torch.nn as nn
import torch
import math
from diffurec import DiffuRec
import torch.nn.functional as F
import copy
import numpy as np
from step_sample import LossAwareSampler
import torch as th
from Energey_Function import DeeperSequenceEnergyFunction


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0

        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        Q = self.linear_q(queries)  
        K = self.linear_k(keys)  
        V = self.linear_v(keys)  
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0) 
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5 
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  
        diag_vals = torch.ones_like(matmul_output[0, :, :])  
        tril = torch.tril(diag_vals)  
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings,
                                       matmul_output_m1)  
        matmul_output_sm = self.softmax(matmul_output_m2)  
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  
        matmul_output_qm = matmul_output_sm * query_mask
        matmul_output_dropout = self.dropout(matmul_output_qm)
        output_ws = torch.bmm(matmul_output_dropout, V_)  
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  
        output_res = output + queries
        return output_res


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ResidualDeepPositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_in if i == 0 else d_hid),
                nn.Conv1d(d_in if i == 0 else d_hid, d_hid if i < num_layers - 1 else d_in, 1),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for i in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)  
        for layer in self.layers:
            output = layer(output)
        output = output.transpose(1, 2)  
        output = self.dropout(output)
        output = self.layer_norm(output + residual)  
        return output


class Att_Diffuse_model(nn.Module):
    def __init__(self, Align, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num+1
        self.item_embeddings = nn.Embedding(self.item_num, self.emb_dim)
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.Align = Align
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
        self.state_size = args.max_len
        self.device = args.device
        self.ln_1 = nn.LayerNorm(args.hidden_size)
        self.ln_2 = nn.LayerNorm(args.hidden_size)
        self.ln_3 = nn.LayerNorm(args.hidden_size)
        self. mh_attn = MultiHeadAttention(args.hidden_size,args.hidden_size,args.num_heads,args.dropout)
        self.feed_forward = PositionwiseFeedForward(args.hidden_size, args.hidden_size, args.dropout)
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=args.hidden_size,
        )
        self.p = args.dropout
        self.register_buffer('position_table', get_sinusoid_encoding_table(args.max_len, args.hidden_size))
        self.pos_combiner = nn.Linear(args.hidden_size * 2, args.hidden_size)
        self.gate_layer = nn.Linear(args.hidden_size, args.hidden_size) 
        self.gate_network = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.Sigmoid()
        )
        self.h_gate_network = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.Sigmoid()
        )

    def embed(self,sequence):
        Embeded = self.item_embeddings
        return Embeded

    def diffu_pre(self, item_rep, tag_emb, mask_seq, h, target_embeddings, target_mask, train_flag):
        seq_rep_diffu, item_rep_out, weights, t, dsm_loss  = self.diffu(item_rep, tag_emb, mask_seq, h, target_embeddings, target_mask, train_flag)
        return seq_rep_diffu, item_rep_out, weights, t, dsm_loss    
    def reverse(self, item_rep, noise_x_t, mask_seq, h, train_flag):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, noise_x_t, mask_seq, h, train_flag)
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))

    def loss_diffu(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        scores_pos = scores.gather(1 , labels)  
        scores_neg_mean = (torch.sum(scores, dim=-1).unsqueeze(-1)-scores_pos)/(scores.shape[1]-1)
        loss = torch.min(-torch.log(torch.mean(torch.sigmoid((scores_pos - scores_neg_mean).squeeze(-1)))), torch.tensor(1e8))
        return loss   

    def loss_diffu_ce(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        """
        ### norm scores
        item_emb_norm = F.normalize(self.item_embeddings.weight, dim=-1)
        rep_diffu_norm = F.normalize(rep_diffu, dim=-1)
        temperature = 0.07
        scores = torch.matmul(rep_diffu_norm, item_emb_norm.t())/temperature
        """
        return self.loss_ce(scores, labels.squeeze(-1))

    def diffu_rep_pre(self, rep_diffu):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return scores
    
    def loss_rmse(self, rep_diffu, labels):
        rep_gt = self.item_embeddings(labels).squeeze(1)
        return torch.sqrt(self.loss_mse(rep_gt, rep_diffu))
    
    def routing_rep_pre(self, rep_diffu):
        item_norm = (self.item_embeddings.weight**2).sum(-1).view(-1, 1)  ## N x 1
        rep_norm = (rep_diffu**2).sum(-1).view(-1, 1)  ## B x 1
        sim = torch.matmul(rep_diffu, self.item_embeddings.weight.t())  ## B x N
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)
        
        return -dist

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))




    def extract_axis_1(self, data, indices):
        res = []
        for i in range(data.shape[0]):
            res.append(data[i, indices[i], :])
        res = torch.stack(res, dim=0).unsqueeze(1)
        return res

    def cacu_h(self, states, len_states, p):
        inputs_emb = self.item_embeddings(states)
        positions = torch.arange(self.state_size, device=self.device)
        pos_learned = self.position_embeddings(positions)
        #pos_sin_cos1 = self.position_table[positions]
        #-------- Concatenate the two positional encodings and pass through a linear layer---#
        #combined_pos = torch.cat([pos_learned, pos_sin_cos1], dim=-1)
        #pos_sin_cos = self.pos_combiner(combined_pos)
        #inputs_emb += self.position_embeddings(torch.arange(self.state_size).to(self.device))
        #inputs_emb += pos_sin_cos
        inputs_emb += pos_learned
        seq = self.embed_dropout(inputs_emb)
        seq1 = self.LayerNorm(seq)
        mask = (states > 0).float().unsqueeze(-1).to(self.device)
        seq1 *= mask
        seq_normalized = self.ln_1(seq1)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = self.extract_axis_1(ff_out, len_states.squeeze() - 1)
        h = state_hidden.squeeze()
        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)
        # print(h.device, self.none_embedding(torch.tensor([0]).to(self.device)).device, mask.device)
        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)
        return h

    def predict(self, states, len_states):
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.position_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.embed_dropout(inputs_emb)
        seq1 = self.LayerNorm(seq)
        mask = (states > 0).float().unsqueeze(-1).to(self.device)
        seq1 *= mask
        seq_normalized = self.ln_1(seq1)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = self.extract_axis_1(ff_out, len_states.squeeze() - 1)
        h = state_hidden.squeeze()

        return h

    def forward(self, sequence, tag, len_seq, y_target, train_flag=True):
        seq_length = sequence.size(1)
        if train_flag:
            h = self.cacu_h(sequence, len_seq, self.p)
        else:
            h1 = self.predict(sequence, len_seq)
        item_embeddings = self.item_embeddings(sequence)
        item_embeddings = self.embed_dropout(item_embeddings) 
        item_embeddings = self.LayerNorm(item_embeddings)        
        mask_seq = (sequence>0).float()
        target_embeddings = self.item_embeddings(y_target)
        target_mask = (y_target>0).float()

        if train_flag:
            tag_emb = self.item_embeddings(tag.squeeze(-1))  ## B x H
            rep_diffu, rep_item, weights, t, dsm_loss = self.diffu_pre(item_embeddings, tag_emb, mask_seq, h, target_embeddings, target_mask, train_flag)

            
            item_rep_dis = None
            seq_rep_dis = None
        else:
            dsm_loss = None
            noise_x_t = th.randn_like(item_embeddings[:,-1,:])
            rep_diffu = self.reverse(item_embeddings, noise_x_t, mask_seq, h1, train_flag)
            weights, t, item_rep_dis, seq_rep_dis = None, None, None, None
        scores = None
        return scores, rep_diffu, weights, t, item_rep_dis, seq_rep_dis, dsm_loss   

def create_model_diffu(args):
    diffu_pre = DiffuRec(args)
    return diffu_pre


#sinusoid encoding
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / (10000 ** (2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = torch.tensor([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    # Apply sine to even indices and cosine to odd indices
    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # Zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return sinusoid_table