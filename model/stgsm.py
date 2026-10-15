import math

import torch

import utils.constants as Constants
from model.grau import GRAU
from model.time_encoder import *
from model.merger import *
from model.transformer import TransformerBlock, LongTermAttention
from model.encoder import GraphEncoder
from model.gru import CustomGRU, CustomGRAU

import torch
import torch.nn as nn
import torch.nn.functional as F

def rff_transform(embedding, w):
    D = w.size(1)
    out = torch.mm(embedding, w)
    d1 = torch.cos(out)
    d2 = torch.sin(out)
    return np.sqrt(1 / D) * torch.cat([d1, d2], dim=1)

def approx_infonce(h1, h2, temp, rff_dim, mode='rff'):
    z1 = F.normalize(h1, dim=-1)
    z2 = F.normalize(h2, dim=-1)

    z = torch.cat([z1, z2], dim = 0)

    w = torch.randn(z.size(1), rff_dim).to(z.device) / np.sqrt(temp)
    rff_out = rff_transform(z, w)


    rff_1, rff_2 = rff_out.chunk(2, dim = 0)

    neg_sum = torch.sum(rff_out, dim=0, keepdim=True)
    neg_score = np.exp(1 / temp) * (torch.sum(rff_1 * neg_sum, dim=1))
    neg_score2 = np.exp(1 / temp) * (torch.sum(rff_2 * neg_sum, dim=1))
    neg_score = torch.cat([neg_score, neg_score2],0)
    return neg_score

class InfoNCE_Linear(nn.Module):
    def __init__(self):
        super(InfoNCE_Linear, self).__init__()
        self.temperature = 0.8

    def forward(self, view1, view2):

        batch_size = view1.shape[0]
        view1 = F.normalize(view1, p=2, dim=1)
        view2 = F.normalize(view2, p=2, dim=1)
        sim_ij = torch.sum(view1*view2, 1)


        positives = torch.cat([sim_ij, sim_ij], dim=0)
        nominator = torch.exp(positives/self.temperature)

        denominator = approx_infonce(view1, view2, self.temperature, rff_dim = 8* 64, mode='sorf' )

        losses = -torch.log(nominator/denominator)
        loss = torch.sum(losses)/(2*batch_size)
        return loss

class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class STGSM(nn.Module):
    def __init__(self,opt, hypergraph_list, hypergraph_list_micro):
        super(STGSM, self).__init__()

        self.opt=opt
        if getattr(opt, 'num_negatives', 1) not in (1, 5, 10):
            raise ValueError('num_negatives must be 1, 5 or 10')
        self.generated_query_scale = getattr(opt, 'generated_query_scale', 1.0)
        if not 0.0 < self.generated_query_scale <= 2.0:
            raise ValueError('generated_query_scale must be finite and in (0, 2]')
        self.query_history_residual = getattr(opt, 'query_history_residual', 0.0)
        if not 0.0 <= self.query_history_residual <= 1.0:
            raise ValueError('query_history_residual must be finite and in [0, 1]')
        self.negative_sampling = getattr(opt, 'negative_sampling', 'batch_roll')
        if self.negative_sampling not in ('batch_roll', 'prefix_excluded'):
            raise ValueError('negative_sampling must be batch_roll or prefix_excluded')
        dropout=opt.dropout
        self.ntoken = opt.ntoken
        self.user_size = opt.user_size
        self.ninp = opt.d_word_vec
        self.transformer_dim = opt.transformer_dim
        self.pos_dim = opt.pos_dim
        self.__name__ = "GRU"


        self.graph_type = opt.graph_type
        self.time_encoder_type = opt.time_encoder_type


        self.dropout = nn.Dropout(dropout)


        self.user_encoder = GraphEncoder(opt, self.graph_type, hypergraph_list, hypergraph_list_micro)
        self.pos_dim = self.ninp
        self.linear_pos_embedding = nn.Embedding(1000, self.pos_dim)

        self.register_buffer("position_vec", torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / self.pos_dim) for i in range(self.pos_dim)]), persistent=False)


        self.att = nn.Parameter(torch.zeros(1, self.ninp))
        self.att_m = nn.Parameter(torch.zeros(self.ninp, self.ninp))


        n_heads = int(getattr(opt, 'heads', 1))
        self.stu_ca = TransformerBlock(d_q=self.ninp, d_k = self.ninp , d_v =self.ninp, n_heads=n_heads, is_layer_norm=True)
        self.linear_pos_ca = TransformerBlock(d_q=self.ninp, d_k=self.ninp, d_v =self.ninp, n_heads=n_heads, is_layer_norm=True)
        self.sin_cos_pos_ca = LongTermAttention(input_size = self.ninp)


        self.cas_encoder = nn.GRU(self.ninp, self.ninp, batch_first=True)
        self.cas_encoder2 = nn.GRU(self.ninp, self.ninp, batch_first=True)


        decoder_dropout = float(getattr(opt, 'diff_decoder_dropout', 0.2))
        self.decoder_diff = nn.TransformerDecoderLayer(
            d_model=self.ninp, nhead=2, dim_feedforward=self.ninp,
            dropout=decoder_dropout, activation="gelu")


        self.mask_fill_value = float(getattr(opt, 'mask_fill', -1e4)) if opt is not None else -1e4

        self.target_score_scale = float(getattr(opt, 'target_score_scale', 0.0))
        if not 0.0 <= self.target_score_scale <= 20.0:
            raise ValueError('target_score_scale must be finite and in [0, 20]')
        self.diff_objective = getattr(opt, 'diff_objective', 'legacy')
        if self.diff_objective not in ('legacy', 'x0'):
            raise ValueError('diff_objective must be legacy or x0')


        self.generator_mlp = bool(getattr(opt, 'generator_mlp', False))
        if self.generator_mlp:
            mlp_dropout = float(getattr(opt, 'gen_mlp_dropout', 0.1))
            self.gen_mlp = nn.Sequential(
                nn.Linear(self.ninp, self.ninp),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(self.ninp, self.ninp),
            )
            self.gen_mlp_norm = nn.LayerNorm(self.ninp)


        self.aux_query_space = bool(getattr(opt, 'aux_query_space', False))


        self.T = int(getattr(opt, 'corruption_steps', 40))
        self.beta_1 = float(getattr(opt, 'noise_beta_1', 1e-4))
        self.beta_T = float(getattr(opt, 'noise_beta_T', 0.002))
        if self.T < 1:
            raise ValueError('corruption_steps must be positive')
        if not 0.0 < self.beta_1 <= self.beta_T < 1.0:
            raise ValueError('require 0 < noise_beta_1 <= noise_beta_T < 1')


        self.chain_steps = 0 if self.diff_objective == 'x0' else self.T
        self.time_embeddings = nn.Embedding(max(self.T, 1), self.ninp)
        self.layernorm = nn.LayerNorm(self.ninp)

        device = torch.device("cpu")

        self.betas = torch.linspace(self.beta_1, self.beta_T, self.T).double().to(device)
        self.alphas = 1.0-self.betas.to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = torch.cat((torch.Tensor([1.0]).to(device), self.alphas_cumprod[:-1])).to(device)
        self.alphas_cumprod_next = torch.cat((self.alphas_cumprod[1:], torch.Tensor([0.0]).to(device))).to(device)


        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0-self.alphas_cumprod).to(device)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod).to(device)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0/self.alphas_cumprod).to(device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0/self.alphas_cumprod -1).to(device)


        self.posterior_variance = (
            self.betas * (1-self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)
        ).to(device)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat((torch.Tensor([self.posterior_variance[1]]).to(device), self.posterior_variance[1:]))
        ).to(device)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod)
        ).to(device)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas)/(1.0 - self.alphas_cumprod)
        ).to(device)

        self.clr_criterion = InfoNCE_Linear()


        self.layernorm2 = nn.LayerNorm(self.ninp)
        self.fus1 = Fusion(self.ninp, self.ninp, dropout=dropout)
        self.fus2 = Fusion(self.ninp, self.ninp, dropout=dropout)


        self.decoder = Decoder(input_size=self.ninp, user_size=self.user_size, opt=opt)
        self.init_weights()


    def init_weights(self):
        init.xavier_normal_(self.linear_pos_embedding.weight)
        init.xavier_normal_(self.att_m)
        init.xavier_normal_(self.att)

    def sin_cos_temporal_enc(self, time):


        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])

        return result


    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings


    def _decode_diffusion(self, outputs, time_emb, padding_mask):

        causal_mask = torch.triu(torch.ones(outputs.size(1), outputs.size(1),
                                           dtype=torch.bool, device=outputs.device), diagonal=1)
        key_padding_mask = padding_mask.clone()

        key_padding_mask[padding_mask.all(dim=1), 0] = False
        decoder_input = outputs.masked_fill(padding_mask.unsqueeze(-1), 0)
        decoded = self.decoder_diff(
            decoder_input.transpose(0, 1), time_emb.transpose(0, 1),
            tgt_mask=causal_mask, tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask)
        decoded = decoded.transpose(0, 1)
        if getattr(self, 'generator_mlp', False):


            decoded = self.gen_mlp_norm(decoded + self.gen_mlp(decoded))
        return decoded.masked_fill(padding_mask.unsqueeze(-1), 0).contiguous()

    def _query_projection(self, x_pred, linear_tgt_batch_t_emb, linear_seed_batch_t_emb, seed_emb, mask):


        rng_state = torch.get_rng_state()
        projected = self.linear_pos_ca(linear_tgt_batch_t_emb + x_pred,
                                       linear_seed_batch_t_emb + seed_emb, seed_emb, mask=mask)
        torch.set_rng_state(rng_state)
        return projected

    def _aligned_auxiliary_input(self, x_pred, linear_tgt_batch_t_emb, linear_seed_batch_t_emb,
                                 seed_emb, mask, target_len):


        if not self.aux_query_space:
            return x_pred
        projected = self._query_projection(x_pred, linear_tgt_batch_t_emb,
                                           linear_seed_batch_t_emb, seed_emb, mask)
        return projected[:, :target_len, :]


    def forward(self, input, input_timestamp, input_id, train=True):

        batch_size, max_len = input.shape


        seed_emb = self.user_encoder(input, input_timestamp, train)
        seed_emb = self.dropout(seed_emb)


        mask = (input == Constants.PAD)
        seed_batch_t = torch.arange(input.size(1), device=input.device).expand(input.size())
        tgt_batch_t = seed_batch_t + 1


        linear_seed_batch_t_emb = self.linear_pos_embedding(seed_batch_t)
        linear_tgt_batch_t_emb = self.linear_pos_embedding(tgt_batch_t)

        linear_zero = self.linear_pos_embedding(torch.tensor([[0]], device=input.device)).view(1,self.ninp).repeat(batch_size,1)
        pos0_emb = seed_emb[:,0]

        sincos_t_hidden = self.sin_cos_pos_ca(pos0_emb + linear_zero, linear_seed_batch_t_emb + seed_emb, seed_emb, mask = mask)
        stu_hidden = self.stu_ca(linear_seed_batch_t_emb + seed_emb, linear_seed_batch_t_emb + seed_emb, seed_emb, mask = mask)

        user_hidden = self.fus1(stu_hidden, sincos_t_hidden)

        user_hidden = self.layernorm2(user_hidden)
        outputs, _ = self.cas_encoder(user_hidden)
        target_training = getattr(self.opt, "target_training", "original")
        query_mode = getattr(self.opt, "query_mode", "sample")
        if target_training == "direct" or (not train and query_mode == "center"):

            step = 0 if target_training == "direct" else getattr(self.opt, "inference_step", 0)
            time_ids = torch.full(input.shape, step, dtype=torch.long, device=input.device)
            time_emb = self.time_embeddings(time_ids)
            x_pred = self._decode_diffusion(outputs, time_emb, mask)
            loss_diffusion = 0.0
            if train:
                target_emb = torch.zeros_like(seed_emb)
                target_emb[:, :-1, :] = seed_emb[:, 1:, :]
                target_neg_emb = self._sample_negative_targets(target_emb, input, train)
                if getattr(self.opt, 'corrupt_negative', False):


                    noise = torch.randn_like(target_emb) if getattr(
                        self.opt, 'share_negative_noise', False) else None
                    if noise is not None and target_neg_emb.dim() == 4:
                        noise = noise.unsqueeze(2)
                    target_neg_emb = self.q_sample(target_neg_emb, time_ids[:, 0], noise=noise)
                is_target = (input > 0).view(input.size(0) * max_len)
                aux_pred = self._aligned_auxiliary_input(
                    x_pred, linear_tgt_batch_t_emb, linear_seed_batch_t_emb, seed_emb,
                    mask, x_pred.size(1))
                loss_t, _ = self.cross_entropy(aux_pred, target_emb, target_neg_emb, is_target)

                loss_diffusion = loss_t * sum(1.0 / (i + 1) for i in range(self.T))
        else:

            x_pred = torch.randn_like(outputs)


            loss_diffusion = torch.zeros((), device=outputs.device)

            is_target = (input>0).view(input.size(0)*max_len)
            target_emb = torch.zeros_like(seed_emb)

            target_emb[:, :-1, :] = seed_emb[:, 1:, :]

            target_neg_emb = self._sample_negative_targets(target_emb, input, train)
            indices = list(range(40))[::-1]
            for i in indices:

                t = x_pred.new_ones([x_pred.shape[0], ], dtype=torch.long) * i
                time_ids = t.unsqueeze(1).expand(x_pred.shape[0], x_pred.shape[1])

                time_emb = self.time_embeddings(time_ids)


                model_output = self._decode_diffusion(outputs, time_emb, mask)


                if self.diff_objective == 'x0':


                    x_pred = self._predict_x0_from_output(model_output, t)
                else:
                    out = self.p_sample(model_output, x_pred, t)
                    x_pred = out["sample"]


                share_noise = getattr(self.opt, 'share_negative_noise', False)


                needs_shared = share_noise and target_training != "no_corruption"
                needs_shared = needs_shared or (share_noise and train and
                                                getattr(self.opt, 'corrupt_negative', False))


                noise = torch.randn_like(target_emb) if needs_shared else None
                neg_noise = noise
                if neg_noise is not None and target_neg_emb.dim() == 4:
                    neg_noise = neg_noise.unsqueeze(2)
                pos_emb = (target_emb if target_training == "no_corruption"
                               else self.q_sample(target_emb, t, noise=noise))
                if train and getattr(self.opt, 'corrupt_negative', False):

                    neg_emb = self.q_sample(target_neg_emb, t, noise=neg_noise)
                else:
                    neg_emb = target_neg_emb


                pos_emb = self._clean_target_when_x0(pos_emb, target_emb)
                neg_emb = self._clean_target_when_x0(neg_emb, target_neg_emb)


                if train:
                    aux_pred = self._aligned_auxiliary_input(
                        x_pred, linear_tgt_batch_t_emb, linear_seed_batch_t_emb, seed_emb,
                        mask, max_len)
                    loss_t, batch_auc = self.cross_entropy(aux_pred, pos_emb, neg_emb, is_target)
                    loss_diffusion += (loss_t)/(i+1)


        if self.generated_query_scale != 1.0:
            x_pred = self.generated_query_scale * x_pred
        if self.query_history_residual != 0.0:
            x_pred = x_pred + self.query_history_residual * outputs
        linear_t_hidden = self.linear_pos_ca(linear_tgt_batch_t_emb + x_pred, linear_seed_batch_t_emb + seed_emb, seed_emb, mask = mask)
        outputs = self.fus2(outputs, linear_t_hidden)
        outputs = self.layernorm(outputs)
        outputs, _ = self.cas_encoder2(outputs)
        outputs = outputs[:, :-1]


        pred = self.decoder(outputs)
        if self.target_score_scale != 0.0:


            pred = pred + self._target_similarity_scores(x_pred[:, :pred.size(1), :])
        input = input[:, :-1]
        mask = self.get_previous_user_mask(input, self.user_size)
        output = pred + mask
        user_predction = output.view(-1, output.size(-1))


        return user_predction, 0., loss_diffusion * getattr(self.opt, "aux_scale", 1.9)

    def _target_similarity_scores(self, generated):


        embedding = self.user_encoder.gnn_layer.embedding.weight
        users = embedding[2:self.user_size]


        generated_unit = F.normalize(generated, p=2, dim=-1)
        width = users.size(0)
        scores = generated.new_zeros(generated.size(0), generated.size(1), self.user_size)


        chunk = max(1, int(getattr(self.opt, 'score_chunk', 4096)))
        for start in range(0, width, chunk):
            stop = min(start + chunk, width)
            block = F.normalize(users[start:stop], p=2, dim=-1)
            scores[..., 2 + start:2 + stop] = torch.matmul(generated_unit, block.t())


        return self.target_score_scale * scores

    def get_previous_user_mask(self, seq, user_size):

        assert seq.dim() == 2
        prev_shape = (seq.size(0), seq.size(1), seq.size(1))
        seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
        previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
        previous_mask = torch.from_numpy(previous_mask).to(seq.device)

        masked_seq = previous_mask * seqs.data.float()


        PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1, device=seq.device)
        masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
        ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size, device=seq.device)


        masked_seq = ans_tmp.scatter_(2, masked_seq.long(),
                                      getattr(self, 'mask_fill_value', -1e4))
        masked_seq = Variable(masked_seq, requires_grad=False)
        return masked_seq
    def _predict_x0_from_output(self, model_output, t):


        return model_output

    def _clean_target_when_x0(self, corrupted, clean):


        return clean if self.diff_objective == 'x0' else corrupted

    def q_mean_variance(self, x_0, t):
        mean =  extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        variance = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, log_variance

    def q_sample(self, x_0, t, noise=None):


        if noise is None: noise = torch.randn_like(x_0)
        mean, variance, log_variance = self.q_mean_variance(x_0, t)
        x_t = mean + torch.exp(0.5*log_variance) * noise
        return x_t

    def q_posterior_mean_variance(self, x_0, x_t, t):


        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def _predict_x_start_from_xprev(self, x_t, t, xprev):
        return extract(1.0/self.posterior_mean_coef1, t, x_t.shape)*xprev - extract(self.posterior_mean_coef2/self.posterior_mean_coef1, t, x_t.shape) * x_t


    def p_mean_variance(self, model_output, x_t, t, clip_denoised=True, denoise_fn=None, model_kwargs=None):


        model_mean = model_output
        model_variance = extract(self.posterior_variance, t, x_t.shape)
        model_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        pred_xstart = self._predict_x_start_from_xprev(x_t=x_t, t=t, xprev=model_output)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance":  model_log_variance,
            "pred_xstart": pred_xstart
        }

    def p_sample(self, model_output, x_t, t):
        out = self.p_mean_variance(model_output, x_t, t)
        re = {}
        re['sample'] =  out['mean'] + torch.exp(0.5*out['log_variance']) * torch.randn_like(x_t)
        return re

    def _negative_targets(self, target_emb, train):
        count = min(getattr(self.opt, 'num_negatives', 1), max(target_emb.size(0) - 1, 1)) if train else 1
        if count == 1:
            return torch.roll(target_emb, shifts=-1, dims=0)

        return torch.stack([torch.roll(target_emb, shifts=-i, dims=0)
                            for i in range(1, count + 1)], dim=2)

    def _sample_negative_targets(self, target_emb, input, train):


        if not train:


            return torch.zeros_like(target_emb)
        if self.negative_sampling == 'batch_roll':
            return self._negative_targets(target_emb, train)
        count = min(getattr(self.opt, 'num_negatives', 1), max(input.size(0) - 1, 1)) if train else 1
        shape = (input.size(0), input.size(1)) if count == 1 else (
            input.size(0), input.size(1), count)


        if int(input.max()) >= self.user_size:
            raise ValueError(
                f'cascade contains user id {int(input.max())} >= user_size {self.user_size}; the '
                f'negative sampler indexes a flat (batch, user) table and cannot represent it')

        ids = torch.randint(1, self.user_size, shape, device=input.device)


        length = input.size(1)
        batch = input.size(0)


        offset = torch.arange(batch, device=input.device).view(batch, 1) * self.user_size
        flat_index = (offset + input).reshape(-1)
        first_seen = torch.full((batch * self.user_size,), length, dtype=torch.long,
                                device=input.device)
        first_seen = first_seen.index_reduce(
            0, flat_index, torch.arange(length, device=input.device)
            .view(1, length).expand(batch, length).reshape(-1), 'amin', include_self=True
        ).view(batch, self.user_size)


        drawn_first = first_seen.gather(1, ids.reshape(batch, -1)).reshape(ids.shape)
        position = torch.arange(length, device=input.device).view(1, length)
        if ids.dim() == 3:
            position = position.unsqueeze(-1)
        excluded = drawn_first <= position


        following = torch.cat([input[:, 1:], torch.zeros_like(input[:, :1])], dim=1)
        positive = following.view(batch, length, *([1] * (ids.dim() - 2)))
        excluded = excluded | (ids == positive)
        ids = ids.masked_fill(excluded, self._negative_holdout_id())
        return self.user_encoder.lookup_encoded_users(ids)

    def _negative_holdout_id(self):


        table = getattr(self.user_encoder, 'last_user_embedding_lookup', None)
        rows = self.ntoken if table is None else int(table.shape[0])
        if rows <= self.user_size:
            raise ValueError(
                f'no spare embedding row for the negative holdout: table has {rows} rows but '
                f'user_size is {self.user_size}, so every row is a real user or PAD/EOS')
        return self.user_size

    def cross_entropy(self, seq_out, pos_emb, neg_emb, istarget):


        pos = pos_emb.view(-1, pos_emb.size(2))
        multiple = neg_emb.dim() == 4
        neg = neg_emb.view(-1, neg_emb.size(2), self.ninp) if multiple else neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.ninp)
        if getattr(self.opt, 'normalize_cd', False):
            seq_emb = F.normalize(seq_emb, p=2, dim=-1)
            pos = F.normalize(pos, p=2, dim=-1)
            neg = F.normalize(neg, p=2, dim=-1)
        pos_logits = torch.sum(pos * seq_emb, -1)
        neg_logits = torch.sum(neg * (seq_emb.unsqueeze(1) if multiple else seq_emb), -1)
        neg_log = torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        comparison = (torch.sign((pos_logits.unsqueeze(1) if multiple else pos_logits) - neg_logits) + 1) / 2
        if multiple:
            neg_log = neg_log.mean(dim=1)
            comparison = comparison.mean(dim=1)

        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            neg_log * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            comparison * istarget
        ) / torch.sum(istarget)

        return loss, auc

    def loss_simclr(self, pred_xstart, x_start, x_start_neg, istarget):
        B, S, D = pred_xstart.shape
        pred_xstart = pred_xstart.view(-1, D)[istarget]
        x_start = x_start.view(-1,D)[istarget]
        x_start_neg = x_start_neg.view(-1,D)[istarget]
        loss = self.clr_criterion(pred_xstart, x_start)

        return loss

def extract(v, t, x_shape):


    target_device = t.device
    t = t.to(v.device)
    out = torch.gather(v, index=t, dim=0).float().to(target_device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
