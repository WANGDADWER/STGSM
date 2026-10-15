from utils.constants import Logger, Options
from model.stgsm import STGSM
from utils.optim import ScheduledOptim
from utils.metrics import Metrics
from utils.data import DataConstruct
import utils.constants as Constants
import torch.nn as nn
import torch
import argparse
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
import os
from torch_geometric.data import Data

def resolve_device(requested):
    mps = getattr(torch.backends, "mps", None)
    mps_available = mps is not None and mps.is_available()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if mps_available else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Use auto, cpu, mps, cuda or cuda:N") from error
    if device.type not in {"cpu", "mps", "cuda"} or (
            device.type != "cuda" and device.index is not None):
        raise argparse.ArgumentTypeError("Use auto, cpu, mps, cuda or cuda:N")
    if device.type == "mps" and not mps_available:
        raise argparse.ArgumentTypeError("MPS requested but unavailable in this process; check PyTorch and GPU access.")
    if device.type == "cuda" and (not torch.cuda.is_available() or (
            device.index is not None and device.index >= torch.cuda.device_count())):
        raise argparse.ArgumentTypeError("Requested CUDA device is unavailable.")
    return device

np.set_printoptions(threshold=np.inf)
def get_kl_beta(epoch, total_epochs, warmup_ratio=0.3, max_beta=1.0):
    warmup_epochs = int(total_epochs * warmup_ratio)
    if epoch < warmup_epochs:
        return max_beta * (epoch / warmup_epochs)
    else:
        return max_beta
def diffusion_weight_for_epoch(epoch, opt, total_epochs):


    initial_diff_weight = float(getattr(opt, 'diff_weight_start', 0.9))
    final_diff_weight = float(getattr(opt, 'diff_weight_end', 0.1))
    decay_start = int(getattr(opt, 'diff_decay_start', 50))
    decay_end = int(getattr(opt, 'diff_decay_end', total_epochs))
    mode = getattr(opt, 'diff_weight_mode', 'anneal')
    if mode == 'const' or decay_end <= decay_start or epoch < decay_start:
        return initial_diff_weight
    diff_weight = initial_diff_weight - (initial_diff_weight - final_diff_weight) * \
                  (epoch - decay_start) / (decay_end - decay_start)
    return max(diff_weight, final_diff_weight)


def best_result(scores, k_list=[10,50,100], weights=[1,1,1,1,1,1]):


    result = 0.0
    i = 0
    for k in k_list:
        result += weights[i]*scores["hits@"+str(k)]
        result += weights[i+1] * scores["map@" + str(k)]
        i += 2
    return result


def selection_weights():

    raw = getattr(opt, 'selection_weights', None)
    if raw is None:
        return [1, 1, 1, 1, 1, 1]
    parts = [piece for piece in str(raw).replace(';', ',').split(',') if piece.strip()]
    if len(parts) != 6:
        raise SystemExit('--selection_weights needs six comma separated numbers, got %d' % len(parts))
    return [float(piece) for piece in parts]


def set_seed(SEED):

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(False)
    print(f"Init_every_seed with {SEED}")


root_path = './'
parser = argparse.ArgumentParser(description='Train the original STGSM model or evaluate a checkpoint.')
parser.add_argument('--data', choices=['christianity', 'android', 'twitter', 'memetracker'],
                    default='christianity')
parser.add_argument('--device', type=resolve_device, default='auto', help='auto, cpu, mps, cuda or cuda:N')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--d_model', type=int, default=None, help='Twitter 128, other datasets 64')
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.15)
parser.add_argument('--lr', type=float, default=0.001, help='Initial Adam learning rate')
parser.add_argument('--lr_schedule', choices=['legacy', 'const'], default='legacy')
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--aux_scale', type=float, default=1.9)
parser.add_argument('--smooth', type=float, default=0.0)
parser.add_argument('--lightgcn_layers', type=int, default=None, help='Christianity 6, Android/Twitter 3, Memetracker 2')
parser.add_argument('--lightgcn_gamma', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--query_mode', choices=['center', 'sample'], default='center')
parser.add_argument('--save_path', help='Checkpoint output path, or input path with --eval_only')
parser.add_argument('--eval_only', action='store_true')
opt = parser.parse_args()
if opt.eval_only and opt.save_path is None:
    parser.error('--eval_only requires --save_path')
if opt.epoch < 1 or opt.batch_size < 1:
    parser.error('--epoch and --batch_size must be positive')

opt.generator_mode = 'legacy'
opt.target_training = 'original'
opt.inference_step = 0
opt.num_negatives = 1
opt.negative_sampling = 'batch_roll'
opt.normalize_cd = False
opt.corrupt_negative = False
opt.hypergraph_scope = 'train'
opt.hypergraph_step_split = 'legacy' if opt.eval_only else 'repaired'
opt.checkpoint_selection = 'val'
opt.corruption_steps = 40
opt.noise_beta_1 = 1e-4
opt.noise_beta_T = 0.002
opt.diff_weight_mode = 'anneal'
opt.diff_weight_start = 0.9
opt.diff_weight_end = 0.1
opt.diff_decay_start = 50
opt.diff_decay_end = opt.epoch
opt.n_warmup_steps = 1000
opt.lr_warmup_epochs = 5
opt.lr_min_ratio = 0.01
opt.use_plateau = opt.lr_schedule == 'legacy'
opt.adam_betas_tuple = (0.9, 0.98)
opt.adam_eps = 1e-9
opt.graph_type = 'social+diffusion+item'
opt.time_encoder_type = 'Interval'
opt.pos_dim = 8
opt.time_dim = 8
opt.norm = True
if opt.d_model is None:
    opt.d_model = 128 if opt.data == 'twitter' else 64
if opt.lightgcn_layers is None:
    opt.lightgcn_layers = {'christianity': 6, 'android': 3, 'twitter': 3, 'memetracker': 2}[opt.data]
opt.d_word_vec = opt.d_model
opt.transformer_dim = opt.d_model + opt.pos_dim
if opt.save_path is None:
    opt.save_path = f'checkpoints/STGSM_{opt.data}_{int(time.time())}.pt'
os.makedirs(os.path.dirname(os.path.abspath(opt.save_path)), exist_ok=True)
device = opt.device
print(opt)
set_seed(opt.seed)
metric = Metrics()
data_path = opt.data


def save_checkpoint(model, path):
    runtime = {'device', 'save_path', 'eval_only', 'd_word_vec', 'transformer_dim',
               'user_size', 'ntoken', 'data_path', 'norm'}
    config = {name: value for name, value in vars(opt).items() if name not in runtime}
    torch.save({'state_dict': model.state_dict(), 'config': config}, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'state_dict' in checkpoint:
        config = checkpoint['config']
        if config['data'] != opt.data:
            raise ValueError(f"Checkpoint dataset is {config['data']!r}; use --data {config['data']}")
        if config.get('generator_mode', 'legacy') != 'legacy':
            raise ValueError('This checkpoint uses the newer generator, not the original STGSM model.')
        runtime = {name: getattr(opt, name) for name in ('data', 'device', 'save_path', 'eval_only')}
        if opt.eval_only:
            runtime['hypergraph_step_split'] = opt.hypergraph_step_split
        vars(opt).update(config)
        vars(opt).update(runtime)
    else:
        checkpoint = {'state_dict': checkpoint}
        opt.d_model = checkpoint['state_dict']['user_encoder.gnn_layer.embedding.weight'].shape[1]
        opt.corruption_steps = checkpoint['state_dict']['time_embeddings.weight'].shape[0]
    opt.d_word_vec = opt.d_model
    opt.transformer_dim = opt.d_model + opt.pos_dim
    print(f'[checkpoint] {path} (d={opt.d_model}, graph layers={opt.lightgcn_layers}, batch size={opt.batch_size})')
    return checkpoint['state_dict']


def get_performance(crit, pred, gold):


    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)

    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

    true_set = set()
    for items in gold.cpu().numpy().tolist():
        true_set.add(items)
    pre_set = set()
    for item in pred.cpu().numpy().tolist():
        if item in true_set:
            pre_set.add(item)

    return loss, n_correct, len(pre_set), len(true_set)


def train_epoch(model, training_data, loss_func, optimizer, epoch, total_epochs):

    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0
    total_same_user = 0.0
    n_total_uniq_user = 0.0
    batch_num = 0.0


    diff_weight = diffusion_weight_for_epoch(epoch, opt, total_epochs)

    beta = get_kl_beta(epoch, total_epochs)

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):

        tgt, tgt_timestamp, tgt_id = batch
        tgt = tgt.to(device)
        tgt_timestamp = tgt_timestamp.to(device)
        user_gold = tgt[:, 1:]
        time_gold = tgt_timestamp[:, 1:]

        n_words = user_gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        batch_num += tgt.size(0)

        optimizer.zero_grad()


        user_pred, _, diff_loss = model(tgt, tgt_timestamp, tgt_id, train=True)


        loss, n_correct, same_user, input_users = get_performance(loss_func, user_pred, user_gold)


        total_batch_loss = loss + diff_weight * diff_loss

        total_batch_loss.backward()


        optimizer.step()
        optimizer.update_learning_rate(epoch)


        n_total_correct += n_correct
        total_loss += total_batch_loss.item()
        total_same_user += same_user
        n_total_uniq_user += input_users

    avg_loss = total_loss / n_total_words
    avg_acc = n_total_correct / n_total_words
    avg_user_same_ratio = total_same_user / n_total_uniq_user

    return avg_loss, avg_acc, avg_user_same_ratio


@torch.no_grad()
def evaluate(model, validation_data, k_list=[10, 50, 100]):

    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0
    for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):


        tgt, tgt_timestamp, tgt_id = batch
        tgt = tgt.to(device)
        tgt_timestamp = tgt_timestamp.to(device)


        user_gold = tgt[:, 1:].contiguous().view(-1).detach().cpu().numpy()

        user_pred, diff_loss,_= model(tgt, tgt_timestamp, tgt_id,train=False)

        user_pred = user_pred.detach().cpu().numpy()

        scores_batch, scores_len,MRR = metric.compute_metric(user_pred, user_gold, k_list)

        n_total_words += scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

    return scores, MRR


def build_cascade_hypergraph(cascades):

    hyper_edge_list = []
    hyper_edge_indice = []
    counter = 0
    for cascade in cascades:
        cascade = set(cascade)
        cascade.discard(0)
        if len(cascade) > 1:
            hyper_edge_list += list(cascade)
            hyper_edge_indice += [counter]*len(cascade)
            counter += 1

    hyper_edge = torch.tensor([hyper_edge_list, hyper_edge_indice])

    cascade_hypergraph = Data(edge_index= hyper_edge)
    return cascade_hypergraph

def build_dynamic_hypergraph(examples, examples_times, step_split=2):


    hypergraph_list = []
    time_sorted = []
    for time in examples_times:
        time_sorted += time[:-1]
    time_sorted = sorted(time_sorted)
    split_length = len(time_sorted) // step_split
    start_time = 0
    end_time = 0

    for x in range(split_length, split_length * step_split, split_length):


        start_time = end_time
        end_time = time_sorted[x]

        selected_examples = []
        for i in range(len(examples)):
            example = examples[i]
            example_times = examples_times[i]
            if isinstance(example, list):
                example = torch.tensor(example)
                example_times = torch.tensor(example_times, dtype=torch.float64)
            selected_example = torch.where((example_times < end_time) & (example_times > start_time), example, torch.zeros_like(example))
            selected_examples.append(selected_example.numpy().tolist())

        sub_hypergraph = build_cascade_hypergraph(selected_examples)

        hypergraph_list.append(sub_hypergraph)


    start_time = end_time
    selected_examples = []
    for i in range(len(examples)):
        example = examples[i]
        example_times = examples_times[i]
        if isinstance(example, list):
            example = torch.tensor(example)
            example_times = torch.tensor(example_times, dtype=torch.float64)
        selected_example = torch.where(example_times > start_time, example, torch.zeros_like(example))

        selected_examples.append(selected_example.numpy().tolist())
    hypergraph_list.append(build_cascade_hypergraph(selected_examples))
    print("超图数量：", len(hypergraph_list))

    return hypergraph_list

def build_dynamic_hypergraph_micro(examples, examples_times, step_split=8):


    hypergraph_list = []
    time_sorted = []
    for time in examples_times:
        time_sorted += time[:-1]
    time_sorted = sorted(time_sorted)
    split_length = len(time_sorted) // step_split
    start_time = 0
    end_time = 0

    for x in range(split_length, split_length * step_split, split_length):


        start_time = end_time
        end_time = time_sorted[x]

        selected_examples = []
        for i in range(len(examples)):
            example = examples[i]
            example_times = examples_times[i]
            if isinstance(example, list):
                example = torch.tensor(example)
                example_times = torch.tensor(example_times, dtype=torch.float64)
            selected_example = torch.where((example_times < end_time) & (example_times > start_time), example, torch.zeros_like(example))
            selected_examples.append(selected_example.numpy().tolist())

        sub_hypergraph = build_cascade_hypergraph(selected_examples)

        hypergraph_list.append(sub_hypergraph)

    start_time = end_time
    selected_examples = []
    for i in range(len(examples)):
        example = examples[i]
        example_times = examples_times[i]
        if isinstance(example, list):
            example = torch.tensor(example)
            example_times = torch.tensor(example_times, dtype=torch.float64)
        selected_example = torch.where(example_times > start_time, example, torch.zeros_like(example))

        selected_examples.append(selected_example.numpy().tolist())
    hypergraph_list.append(build_cascade_hypergraph(selected_examples))
    print("超图数量：", len(hypergraph_list))

    return hypergraph_list

def build_hypergraphs(train_cascades, train_time_stamps, valid_cascades, valid_time_stamps,
                      test_cascades, test_time_stamps):


    scope = getattr(opt, 'hypergraph_scope', 'train')
    split = getattr(opt, 'hypergraph_step_split', 'repaired')
    cascades, time_stamps = train_cascades, train_time_stamps
    if scope in ('trainval', 'all'):
        cascades = cascades + valid_cascades
        time_stamps = time_stamps + valid_time_stamps
    if scope == 'all':
        cascades = cascades + test_cascades
        time_stamps = time_stamps + test_time_stamps
    macro_split, micro_split = (5, 10) if split == 'legacy' else (1, 2)
    print(f"[graph] hypergraph_scope={scope} hypergraph_step_split={split} "
          f"macro_step_split={macro_split} micro_step_split={micro_split}", flush=True)
    hypergraph_list = build_dynamic_hypergraph(cascades, time_stamps, step_split=macro_split)
    hypergraph_list_micro = build_dynamic_hypergraph_micro(cascades, time_stamps, step_split=micro_split)
    return hypergraph_list, hypergraph_list_micro


def train(data_path):


    train_data = DataConstruct(data_path, data=0, load_dict=True, batch_size=opt.batch_size, cuda=False, seed=opt.seed)
    valid_data = DataConstruct(data_path, data=1, batch_size=opt.batch_size, cuda=False, seed=opt.seed)
    test_data = DataConstruct(data_path, data=2, batch_size=opt.batch_size, cuda=False, seed=opt.seed)

    train_cascades = train_data._train_cascades
    train_time_stamps  = train_data._train_cascades_timestamp

    hypergraph_list, hypergraph_list_micro = build_hypergraphs(
        train_cascades, train_time_stamps,
        valid_data._valid_cascades, valid_data._valid_cascades_timestamp,
        test_data._test_cascades, test_data._test_cascades_timestamp)


    opt.user_size = train_data.user_size
    opt.ntoken = train_data.ntoken


    opt.data_path = data_path
    opt.norm = train_data.need_norm
    model = STGSM(opt, hypergraph_list = hypergraph_list, hypergraph_list_micro = hypergraph_list_micro).to(device)


    params = filter(lambda p: p.requires_grad, model.parameters())


    _betas = getattr(opt, 'adam_betas_tuple', None)
    if _betas is None:
        _betas = tuple(float(_v) for _v in
                       str(getattr(opt, 'adam_betas', '0.9,0.98')).replace(' ', '').split(','))
    lr_schedule = getattr(opt, 'lr_schedule', 'legacy')
    optimizerAdam = torch.optim.Adam(params, lr=opt.lr, betas=_betas, eps=getattr(opt, 'adam_eps', 1e-09),
                                     weight_decay=opt.l2)
    optimizer = ScheduledOptim(
        optimizerAdam, opt.d_model, opt.n_warmup_steps, data_path,
        lr_schedule=lr_schedule, lr=getattr(opt, 'lr', 1e-3),
        total_epochs=getattr(opt, 'epoch', 50),
        warmup_epochs=getattr(opt, 'lr_warmup_epochs', 5),
        min_lr_ratio=getattr(opt, 'lr_min_ratio', 0.01))


    use_plateau = getattr(opt, 'use_plateau', lr_schedule == 'legacy')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerAdam, 'min', factor=0.4, patience=7) if use_plateau else None


    class_weight = None
    alpha = float(getattr(opt, 'class_weight_alpha', 0.0))
    if alpha > 0.0:
        flat = torch.tensor([u for cascade in train_data._train_cascades for u in cascade],
                            dtype=torch.long)
        counts = torch.bincount(flat, minlength=opt.user_size).float()
        counts = counts.clamp(min=1.0)
        weight = counts.pow(-alpha)
        weight[Constants.PAD] = 0.0
        weight = weight / weight[weight > 0].mean()
        class_weight = weight.to(device)
    loss_func = nn.CrossEntropyLoss(size_average=False, ignore_index=Constants.PAD,
                                    label_smoothing=float(getattr(opt, 'smooth', 0.0)),
                                    weight=class_weight)

    loss_func = loss_func.to(device)

    validation_history = 0.0


    patience = int(getattr(opt, 'early_stop_patience', 0) or 0)
    epochs_without_improvement = 0
    epochs_run = 0
    for epoch_i in range(opt.epoch):
        epochs_run = epoch_i + 1
        print('\n[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu, train_pred = train_epoch(model, train_data, loss_func, optimizer,epoch_i,opt.epoch)
        print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, predected:{pred:3.3f} %' \
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, pred=100 * train_pred,
            elapse=(time.time() - start) / 60), flush=True)

        if epoch_i >= 0:
            start = time.time()
            val_scores, val_MRR = evaluate(model, valid_data)
            val_score = best_result(val_scores, weights=selection_weights())
            print('  - ( Validation )) ')
            for metric in val_scores.keys():
                print(metric + ' ' + str(val_scores[metric]))
            print("Validation use time: ", (time.time() - start) / 60, "min")
            print(f"MRR: {val_MRR}")


            legacy_selection = opt.checkpoint_selection == 'test'
            if legacy_selection:
                print('  - (Test) ')
                test_scores, test_MRR = evaluate(model, test_data)
                for metric in test_scores.keys():
                    print(metric + ' ' + str(test_scores[metric]))
                monitored_score = best_result(test_scores, weights=selection_weights())
            else:
                monitored_score = val_score

            improved = validation_history <= monitored_score
            if improved:
                if legacy_selection:

                    print("Best at Epoch:{} selected_on_split:test test_score:{:.6f} "
                          "(legacy --checkpoint_selection test; not a validation score)".format(
                              epoch_i, monitored_score))
                else:
                    print("Best Validation at Epoch:{} val_score:{:.6f}".format(epoch_i, monitored_score))
                validation_history = monitored_score
                epochs_without_improvement = 0
                if not (getattr(opt, 'freeze_after_select_epoch', False)
                        and getattr(opt, 'select_epoch', -1) >= 0
                        and epoch_i > getattr(opt, 'select_epoch', -1)):
                    save_checkpoint(model, opt.save_path)
            _forced = getattr(opt, 'select_epoch', -1)
            if _forced is not None and _forced >= 0 and epoch_i == _forced and not improved:


                save_checkpoint(model, opt.save_path)
                print(f"Forced save at epoch {epoch_i} (--select_epoch)")
                if legacy_selection:
                    print(f"MRR: {test_MRR}")
            else:
                epochs_without_improvement += 1


            if not legacy_selection and (improved or getattr(opt, 'test_eval_every_epoch', False)):
                print('  - (Test) ')
                with torch.random.fork_rng(devices=[]):
                    test_scores, test_MRR = evaluate(model, test_data)
                for metric in test_scores.keys():
                    print(metric + ' ' + str(test_scores[metric]))
                print(f"MRR: {test_MRR}")
        if scheduler is not None:
            scheduler.step(validation_history)

        if patience and epochs_without_improvement >= patience:
            print(f"[early-stop] no validation improvement for {epochs_without_improvement} epochs; "
                  f"stopping after epoch {epoch_i} with the best checkpoint already saved")
            break

    print(f"[budget] epochs_run={epochs_run} of {opt.epoch}")


def evaluate_checkpoint(data_path):
    state_dict = load_checkpoint(opt.save_path)
    train_data = DataConstruct(data_path, data=0, load_dict=True, batch_size=opt.batch_size, cuda=False)
    valid_data = DataConstruct(data_path, data=1, batch_size=opt.batch_size, cuda=False)
    test_data = DataConstruct(data_path, data=2, batch_size=opt.batch_size, cuda=False)

    train_cascades = train_data._train_cascades
    train_time_stamps = train_data._train_cascades_timestamp
    hypergraph_list, hypergraph_list_micro = build_hypergraphs(
        train_cascades, train_time_stamps,
        valid_data._valid_cascades, valid_data._valid_cascades_timestamp,
        test_data._test_cascades, test_data._test_cascades_timestamp)

    opt.user_size = train_data.user_size
    opt.ntoken = train_data.ntoken


    opt.data_path = data_path
    opt.norm = train_data.need_norm
    model = STGSM(opt, hypergraph_list = hypergraph_list, hypergraph_list_micro = hypergraph_list_micro).to(device)
    opt.user_size = train_data.user_size
    opt.ntoken = train_data.ntoken
    opt.data_path = data_path
    opt.norm = train_data.need_norm

    model.load_state_dict(state_dict, strict=True)

    scores,MRR= evaluate(model, test_data)
    print('  - (Test) ')
    for metric in scores.keys():
        print(metric + ' ' + str(scores[metric]))
    print(f"MRR: {MRR}")
    return scores


if __name__ == "__main__":


    hits10 = []
    hits50 = []
    hits100 = []
    map10 = []
    map50 = []
    map100 = []

    for i in range(1):
        if not opt.eval_only:
            train(data_path)
        scores = evaluate_checkpoint(data_path)
        hits10.append(scores['hits@10'])
        hits50.append(scores['hits@50'])
        hits100.append(scores['hits@100'])
        map10.append(scores['map@10'])
        map50.append(scores['map@50'])
        map100.append(scores['map@100'])

    print('Hits10', np.mean(hits10), np.std(hits10))
    print('Hits50', np.mean(hits50), np.std(hits50))
    print('Hits100', np.mean(hits100), np.std(hits100))
    print('MAP10', np.mean(map10), np.std(map10))
    print('MAP50', np.mean(map50), np.std(map50))
    print('MAP100', np.mean(map100), np.std(map100))
