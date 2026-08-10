import math
import os
import subprocess

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import mlflow

import prep_OOD

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
import numpy as np
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from DAL_archi import WideResNet
from utils import utils
import prep
from utils.display_results import get_measures, print_measures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# IF CUDA RELATED PROBLEMS
# sudo rmmod nvidia_uvm
# sudo modprobe nvidia_uvm

parser = argparse.ArgumentParser()
# parser.add_argument('src', type=str)

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=53, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=10, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# DAL hyper parameters
parser.add_argument('--gamma', default=1, type=float) # increase: higher prevention of large shifts // tradeoff: too restricted, no robustness
parser.add_argument('--beta',  default=0.9, type=float) # higher separation between ID and OOD // forgetting primary target
parser.add_argument('--rho',   default=0.1, type=float) # higher size of OOD space // OOD overlap with ID
parser.add_argument('--strength', default=0.01, type=float) # pushes OOD torwards worst case boundary // exploding gradients, overshoot the boundary space
parser.add_argument('--warmup', type=int, default=0) # time to form class clusters and learn representation before OOD // leaves fewer epochs to learn OOD
parser.add_argument('--iter', default=10, type=int) # time to find worst case point within purturbation // computational runtime
# Others
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--seed', type=int, default=1)


args = parser.parse_args()
torch.manual_seed(1)
np.random.seed(args.seed)
torch.cuda.manual_seed(1)

cuda = True if torch.cuda.is_available() else False

print(args.gamma, args.beta, args.rho)

cudnn.benchmark = True  # fire on all cylinders

train_dataset = utils.CustomDataset(prep.X_train_sc.to_numpy(), prep.y_train.to_numpy())
test_dataset = utils.CustomDataset(prep.X_test_sc.to_numpy(), prep.y_test.to_numpy())

train_loader_in, train_loader_out = utils.dataset_function(train_dataset, X = prep_OOD.X_train_sc, batch_size = args.batch_size, batch_size_o=args.oe_batch_size, train=True)
test_loader_in, test_loader_out = utils.dataset_function(test_dataset, X = prep_OOD.X_test_sc, batch_size = args.batch_size, batch_size_o=args.oe_batch_size, train=False)


ood_num_examples = len(prep.X_test_sc) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(prep.X_test_sc))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

# def get_ood_scores(loader, in_dist=False):
#     _score = []
#     net.eval()
#     with torch.no_grad():
#         for batch_idx, (data, target) in enumerate(loader):
#             if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
#                 break
#             data, target = data.to(torch.float).cuda(), target.cuda()
#             output = net(data)
#             smax = to_np(F.softmax(output, dim=1))
#             _score.append(-np.max(smax, axis=1))
#     if in_dist:
#         return concat(_score).copy() # , concat(_right_score).copy(), concat(_wrong_score).copy()
#     else:
#         return concat(_score)[:ood_num_examples].copy()

def get_ood_scores_with_indices(loader):
    scores = []
    indices = []

    net.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs:
                break
            data = data.to(torch.float).cuda()
            output = net(data)
            smax = F.softmax(output, dim=1)
            batch_scores = -smax.max(dim=1).values
            scores.append(batch_scores.cpu().numpy())
            start = batch_idx * args.test_bs
            end = start + len(data)
            indices.extend(range(start, end))
    return (
        np.concatenate(scores)[:ood_num_examples],
        np.array(indices[:ood_num_examples])
    )

def get_and_print_results(ood_loader, in_score, num_to_avg=1):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score, _ = get_ood_scores_with_indices(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr, '')
    return fpr, auroc, aupr


def get_confusion_details(ood_loader, in_score, recall_level=0.95):

    net.eval()

    out_score, ood_indices = get_ood_scores_with_indices(ood_loader)

    y_true = np.concatenate([
        np.ones(len(out_score)),
        np.zeros(len(in_score))
    ])

    y_score = np.concatenate([
        out_score,
        in_score
    ])

    desc_score_indices = np.argsort(y_score)[::-1]

    y_score_sorted = y_score[desc_score_indices]
    y_true_sorted = y_true[desc_score_indices]

    tps = np.cumsum(y_true_sorted)
    recall = tps / tps[-1]

    cutoff_idx = np.argmin(
        np.abs(recall - recall_level)
    )

    threshold = y_score_sorted[cutoff_idx]

    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    ood_pred = y_pred[:len(out_score)]
    false_negative_mask = ood_pred == 0
    false_negative_indices = ood_indices[false_negative_mask]

    return (
        cm,
        threshold,
        out_score,
        false_negative_indices
    )


debug_stats = {}

def make_forward_hook(name):
    def hook(module, input, output):
        debug_stats[f'{name}_out_norm'] = output.norm(dim=-1).mean().item()
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"!!! NaN/Inf in {name} output !!!")
    return hook

def make_backward_hook(name):
    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            debug_stats[f'{name}_grad_norm'] = grad_output[0].norm().item()
            if torch.isnan(grad_output[0]).any():
                print(f"!!! NaN gradient at {name} !!!")
    return hook

def pre_hook(module, input):
    print(f"Input to {module.__class__.__name__}: shape={input[0].shape}, "
          f"range=[{input[0].min().item():.3f}, {input[0].max().item():.3f}]")


def train(epoch, gamma, debug_hooks=None):
    net.train()
    loss_avg = 0.0
    ce_avg, oe_avg, oe_old_avg = 0.0, 0.0, 0.0

    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):

        data, target = torch.cat((in_set[0], out_set[0]), 0), in_set[1]
        data, target = data.to(torch.float).cuda(), target.cuda()

        x, emb = net.pred_emb(data)
        l_ce = F.cross_entropy(x[:len(in_set[0])], target)
        l_oe_old = - (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        emb_oe = emb[len(in_set[0]):].detach()
        emb_bias = torch.rand_like(emb_oe) * 0.0001

        for _ in range(args.iter):
            emb_bias.requires_grad_()
            x_aug = net.fc_out(emb_bias + emb_oe)
            l_sur = - (x_aug.mean(1) - torch.logsumexp(x_aug, dim=1)).mean()
            r_sur = (emb_bias.abs()).mean(-1).mean()
            l_sur = l_sur - r_sur * gamma
            grads = torch.autograd.grad(l_sur, [emb_bias])[0]
            grads /= (grads ** 2).sum(-1).sqrt().unsqueeze(1)
            emb_bias = emb_bias.detach() + args.strength * grads.detach()
            optimizer.zero_grad()

        gamma -= args.beta * (args.rho - r_sur.detach())
        gamma = gamma.clamp(min=0.0, max=args.gamma)
        if epoch >= args.warmup:
            x_oe = net.fc_out(emb[len(in_set[0]):] + emb_bias)
        else:
            x_oe = net.fc_out(emb[len(in_set[0]):])

        l_oe = - (x_oe.mean(1) - torch.logsumexp(x_oe, dim=1)).mean()
        # print(x_oe.softmax(1).max(1)[0].mean())
        loss = l_ce + .5 * l_oe


        # ---- DEBUG BLOCK ----
        # if batch_idx % 20 == 0:
        #     print(f"\n[batch {batch_idx}] l_ce={l_ce.item():.4f}  l_oe={l_oe.item():.4f}  "
        #           f"l_oe_old={l_oe_old.item():.4f}  gamma={gamma.item():.4f}  r_sur={r_sur.item():.6f}")
        #     print(f"  x[:ID] logit range: [{x[:len(in_set[0])].min().item():.2f}, {x[:len(in_set[0])].max().item():.2f}]")
        #     print(f"  x[OOD] logit range: [{x[len(in_set[0]):].min().item():.2f}, {x[len(in_set[0]):].max().item():.2f}]")
        #     print(f"  emb norm: {emb.norm(dim=-1).mean().item():.4f}  emb_bias norm: {emb_bias.norm(dim=-1).mean().item():.4f}")
        #     if torch.isnan(loss) or torch.isinf(loss):
        #         print("  !!! NaN/Inf detected in loss !!!")
        # ---- END DEBUG ----

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx % 20 == 0 and debug_hooks:
        #     print(debug_stats)

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        ce_avg = ce_avg * 0.8 + float(l_ce) * 0.2
        oe_avg = oe_avg * 0.8 + float(l_oe) * 0.2

        sys.stdout.write('\r epoch %2d %d/%d loss %.2f (ce %.2f, oe %.2f)' %
                          (epoch, batch_idx + 1, len(train_loader_in), loss_avg, ce_avg, oe_avg))
        # print("confidence:", x_oe.softmax(1).max(1).values.mean().item())
        # print("probs:", x_oe.softmax(1).mean(0))
        # print(f"emb norm: {emb.norm(dim=-1).mean().item():.4f}")
        # relative = emb_bias.norm(dim=1) / emb_oe.norm(dim=1)
        # print(relative.mean().item())
        # print(f"r_sur: {r_sur.item():.6f}") #  bias: {emb_bias.norm(dim=1).mean().item():.4f}")
        # print(f"gamma: {gamma.item():.6f}")
        # print(f"l_oe={l_oe.item():.4f}  floor={math.log(3):.4f}  diff={l_oe.item() - math.log(3):.4f}")
        scheduler.step()

    return gamma, loss_avg, ce_avg, oe_avg

# def test():
#     net.eval()
#     correct = 0
#     y, c = [], []
#     with torch.no_grad():
#         for data, target in test_loader_in:
#             data, target = data.cuda(), target.cuda()
#             output = net(data)
#             pred = output.data.max(1)[1]
#             correct += pred.eq(target.data).sum().item()
#     return correct / len(test_loader_in.dataset) * 100


num_classes = 3
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()

# handles = []
# for h in handles:
#     h.remove()
# handles.append(net.block1.register_forward_hook(make_forward_hook('block1')))
# handles.append(net.block2.register_forward_hook(make_forward_hook('block2')))
# handles.append(net.block3.register_forward_hook(make_forward_hook('block3')))
# handles.append(net.block1.register_forward_pre_hook(pre_hook))
# handles.append(net.block2.register_forward_pre_hook(pre_hook))
# handles.append(net.block3.register_forward_pre_hook(pre_hook))
# handles.append(net.block1.register_full_backward_hook(make_backward_hook('block1')))
# handles.append(net.block2.register_full_backward_hook(make_backward_hook('block2')))
# handles.append(net.block3.register_full_backward_hook(make_backward_hook('block3')))



optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader_in), 1, 1e-6 / args.learning_rate))
# if args.src:
#     model_path = './models/wrn_pretrained_epoch_99.pt'
#     net.load_state_dict(torch.load(model_path))
# else:
#     pass


def plots():
    in_batch, _ = next(iter(train_loader_in))
    out_batch, _ = next(iter(train_loader_out))

    in_batch = in_batch.numpy()
    out_batch = out_batch.numpy()

    print("ID shape:", in_batch.shape, " OOD shape:", out_batch.shape)

    X = np.vstack([in_batch, out_batch])
    y = np.concatenate([np.zeros(len(in_batch)), np.ones(len(out_batch))])

    clf = RandomForestClassifier(n_estimators=200, max_depth=6)
    scores = cross_val_score(clf, X, y, cv=3, scoring='roc_auc')
    print(f"ID vs OOD separability (AUROC): {scores.mean():.3f}")
    combined = np.vstack([in_batch, out_batch])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)

    n_in = len(in_batch)
    plt.figure(figsize=(6, 6))
    plt.scatter(proj[:n_in, 0], proj[:n_in, 1], alpha=0.5, label='ID (train_loader_in)', s=10)
    plt.scatter(proj[n_in:, 0], proj[n_in:, 1], alpha=0.5, label='OOD (train_loader_out)', s=10)
    plt.legend()
    plt.title('PCA projection: ID vs OOD')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig("orginal_OOD.png")
    num_features_to_plot = min(6, in_batch.shape[1])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i in range(num_features_to_plot):
        axes[i].hist(in_batch[:, i], bins=30, alpha=0.5, label='ID', density=True)
        axes[i].hist(out_batch[:, i], bins=30, alpha=0.5, label='OOD', density=True)
        axes[i].set_title(f'Feature {i}')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("original_OOD_hist.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(np.corrcoef(in_batch.T), ax=axes[0], cmap='coolwarm', center=0, vmin=-1, vmax=1)
    axes[0].set_title('ID correlation matrix')
    sns.heatmap(np.corrcoef(out_batch.T), ax=axes[1], cmap='coolwarm', center=0, vmin=-1, vmax=1)
    axes[1].set_title('OOD correlation matrix')
    plt.tight_layout()
    plt.savefig("original_OOD_corr.png")



if __name__ == "__main__":
    # process = subprocess.Popen(["mlflow", "server", "--host", "127.0.0.1", "--port", "8080"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

        gamma = 0.01

    # mlflow.set_experiment("OOD")
    # mlflow.pytorch.autolog()
    #
    # with mlflow.start_run():
    #     mlflow.log_params({"epcohs": args.epochs, "learning_rate": args.learning_rate, "batch_size": args.batch_size,
    #                        "oe_batch": args.oe_batch_size})

        for epoch in range(args.epochs):
            gamma, loss_avg, ce_avg, oe_avg = train(epoch, gamma, debug_hooks=True)

            if epoch % 10 == 9:
                net.eval()
                in_score, _ = get_ood_scores_with_indices(test_loader_in)
                metric_ll = []
                metric_ll.append(get_and_print_results(test_loader_out, in_score))
                cm, threshold, out_score, fn_indices = get_confusion_details(
                    test_loader_out,
                    in_score
                )
                print('\n & %.2f & %.2f & %.2f' % tuple((100 * torch.Tensor(metric_ll).mean(0)).tolist()))
                print(cm)
                # print("Threshold:", threshold)
                # print("Indices:", fn_indices[:20])
                # false_negatives = prep_OOD.X_test_sc.iloc[fn_indices]
                # ood_correct = prep_OOD.X_test_sc.drop(prep_OOD.X_test_sc.index[fn_indices])
                #
                # print("False negatives:")
                # print(false_negatives.describe().to_csv("false.csv"))
                #
                # print("\nCorrect OOD:")
                # print(ood_correct.describe().to_csv("correct.csv"))

                torch.save(net.state_dict(), f"wr{ce_avg}.pt")

                records = []
                worst_rows, worst_indices, worst_scores = utils.get_worst_attacks(out_score, fn_indices,
                                                                            test_loader_out.dataset, n=10)

                for idx, score in zip(worst_indices, worst_scores):
                    row = test_loader_out[idx][0].numpy()
                    records.append({
                        'epoch': epoch,
                        'index': int(idx),
                        'ood_score': float(score),
                        **{f'feature_{i}': v for i, v in enumerate(row)}
                    })

                df = pd.DataFrame(records)
                df.to_csv(f"high_conf_attacks_epoch{epoch}.csv", index=False)

            #     mlflow.log_metric("in_score", in_score, step=epoch)
            #

    #         mlflow.log_metrics({"gamma": gamma, "loss_avg": loss_avg, "ce_avg": ce_avg, 'oe_avg': oe_avg}, step=epoch)
    # #
    # mlflow.pytorch.log_model(net, name="model", serialization_format="pickle")


# epoch  9 1916/1916 loss 1.45 (ce 0.85, oe 1.20)
#
#  & 0.88 & 99.70 & 99.72
# [[26194    69]
#  [ 1313 24950]]
#  epoch 19 1916/1916 loss 1.06 (ce 0.43, oe 1.25)
#
#  & 0.06 & 99.98 & 99.98
# [[26261     2]
#  [ 1313 24950]]
#  epoch 29 1916/1916 loss 0.93 (ce 0.30, oe 1.27)
#
#  & 0.00 & 99.69 & 99.84
# [[26180    83]
#  [ 1313 24950]]
#  epoch 39 1916/1916 loss 0.93 (ce 0.31, oe 1.24)
#
#  & 0.00 & 99.92 & 99.96
# [[26236    27]
#  [ 1312 24951]]
# epoch 49 1916/1916 loss 0.81 (ce 0.20, oe 1.23)
#
#  & 0.00 & 99.83 & 99.92
# [[26217    46]
#  [ 1312 24951]]

# rho : 0.1 -> 0.01
#  epoch  9 1916/1916 loss 1.77 (ce 1.14, oe 1.26)
#
#  & 0.22 & 99.96 & 99.96
# [[26261     2]
#  [ 1313 24950]]
#  epoch 19 1916/1916 loss 1.05 (ce 0.41, oe 1.29)
#
#  & 0.08 & 99.68 & 99.82
# [[26166    97]
#  [ 1313 24950]]
#  epoch 29 1916/1916 loss 1.13 (ce 0.49, oe 1.28)
#
#  & 0.00 & 99.85 & 99.92
# [[26215    48]
#  [ 1313 24950]]
#  epoch 39 1916/1916 loss 1.02 (ce 0.39, oe 1.26)
#
#  & 0.00 & 99.99 & 99.99
# [[26232    31]
#  [ 1313 24950]]
#  epoch 49 1916/1916 loss 0.82 (ce 0.20, oe 1.24)
#
#  & 0.00 & 100.00 & 100.00
# [[26263     0]
#  [ 1313 24950]]

# default
# epoch  9 1916/1916 loss 1.36 (ce 0.76, oe 1.20)& 0.92 & 99.08 & 99.17
#
#  & 0.92 & 99.08 & 99.17
# [[16571   197]
#  [ 1313 24950]]
#  epoch 19 1916/1916 loss 0.83 (ce 0.23, oe 1.20)& 0.05 & 99.07 & 99.34
#
#  & 0.05 & 99.07 & 99.34
# [[16553   215]
#  [ 1313 24950]]
#  epoch 29 1916/1916 loss 1.90 (ce 1.30, oe 1.20)& 1.48 & 98.99 & 97.30
#
#  & 1.48 & 98.99 & 97.30
# [[16537   231]
#  [ 1313 24950]]
#  epoch 39 1916/1916 loss 2.17 (ce 1.57, oe 1.20)& 3.19 & 98.76 & 97.30
#
#  & 3.19 & 98.76 & 97.30
# [[16533   235]
#  [ 1313 24950]]
#  epoch 49 1916/1916 loss 1.84 (ce 1.24, oe 1.20)& 0.06 & 99.90 & 99.90
#
#  & 0.06 & 99.90 & 99.90
# [[16740    28]
#  [ 1313 24950]]
#  epoch 59 1916/1916 loss 2.00 (ce 1.42, oe 1.16)& 0.01 & 98.86 & 99.25
#
#  & 0.01 & 98.86 & 99.25
# [[16554   214]
#  [ 1313 24950]]
#  epoch 69 1916/1916 loss 1.36 (ce 0.78, oe 1.17)& 0.57 & 99.43 & 99.45
#
#  & 0.57 & 99.43 & 99.45
# [[16498   270]
#  [ 1313 24950]]
#  epoch 79 1916/1916 loss 1.35 (ce 0.77, oe 1.14)& 0.00 & 100.00 & 99.99
#
#  & 0.00 & 100.00 & 99.99
# [[16766     2]
#  [ 1313 24950]]
#  epoch 89 1916/1916 loss 0.89 (ce 0.32, oe 1.15)& 0.00 & 100.00 & 100.00
#
#  & 0.00 & 100.00 & 100.00
# [[16768     0]
#  [ 1313 24950]]
#  epoch 99 1916/1916 loss 1.05 (ce 0.48, oe 1.14)& 0.00 & 99.99 & 99.99
#
#  & 0.00 & 99.99 & 99.99
# [[16766     2]
#  [ 1312 24951]]

# str 0.05
#  epoch  9 1916/1916 loss 1.78 (ce 1.18, oe 1.22)& 5.97 & 99.13 & 98.69
#
#  & 5.97 & 99.13 & 98.69
# [[15697  1071]
#  [ 1313 24950]]
#  epoch 19 1916/1916 loss 1.33 (ce 0.72, oe 1.22)& 0.07 & 99.73 & 99.76
#
#  & 0.07 & 99.73 & 99.76
# [[16697    71]
#  [ 1313 24950]]
#  epoch 29 1916/1916 loss 1.58 (ce 0.96, oe 1.23)& 0.21 & 99.56 & 99.64
#
#  & 0.21 & 99.56 & 99.64
# [[16691    77]
#  [ 1313 24950]]
#  epoch 39 1916/1916 loss 1.73 (ce 1.11, oe 1.24)& 0.16 & 99.95 & 99.93
#
#  & 0.16 & 99.95 & 99.93
# [[16764     4]
#  [ 1313 24950]]
#  epoch 49 1916/1916 loss 1.39 (ce 0.76, oe 1.26)& 0.72 & 99.83 & 99.73
#
#  & 0.72 & 99.83 & 99.73
# [[16748    20]
#  [ 1313 24950]]
#  epoch 59 1916/1916 loss 1.10 (ce 0.49, oe 1.21)& 0.00 & 99.87 & 99.89
#
#  & 0.00 & 99.87 & 99.89
# [[16724    44]
#  [ 1313 24950]]
#  epoch 69 1916/1916 loss 1.05 (ce 0.46, oe 1.19)& 0.00 & 99.96 & 99.97
#
#  & 0.00 & 99.96 & 99.97
# [[16760     8]
#  [ 1313 24950]]
#  epoch 79 1916/1916 loss 1.08 (ce 0.50, oe 1.16)& 0.00 & 100.00 & 100.00
#
#  & 0.00 & 100.00 & 100.00
# [[16768     0]
#  [ 1313 24950]]
#  epoch 89 1916/1916 loss 1.05 (ce 0.47, oe 1.15)& 0.00 & 99.99 & 99.99
#
#  & 0.00 & 99.99 & 99.99
# [[16765     3]
#  [ 1313 24950]]
#  epoch 99 1916/1916 loss 0.83 (ce 0.25, oe 1.15)& 0.00 & 99.99 & 99.99
#
#  & 0.00 & 99.99 & 99.99
# [[16766     2]
#  [ 1312 24951]]


# lr 0.01
# epoch  9 1916/1916 loss 1.56 (ce 0.98, oe 1.16)& 15.69 & 96.60 & 96.27
#
#  & 15.69 & 96.60 & 96.27
# [[14859  1909]
#  [ 1313 24950]]
#  epoch 19 1916/1916 loss 1.24 (ce 0.65, oe 1.18)& 1.51 & 99.72 & 99.59
#
#  & 1.51 & 99.72 & 99.59
# [[16640   128]
#  [ 1313 24950]]
#  epoch 29 1916/1916 loss 1.27 (ce 0.66, oe 1.21)& 0.15 & 99.96 & 99.93
#
#  & 0.15 & 99.96 & 99.93
# [[16763     5]
#  [ 1313 24950]]
#  epoch 39 1916/1916 loss 0.99 (ce 0.38, oe 1.21)& 0.05 & 99.98 & 99.98
#
#  & 0.05 & 99.98 & 99.98
# [[16767     1]
#  [ 1313 24950]]
#  epoch 49 1916/1916 loss 1.73 (ce 1.13, oe 1.21)& 0.05 & 99.99 & 99.98
#
#  & 0.05 & 99.99 & 99.98
# [[16767     1]
#  [ 1313 24950]]
#  epoch 59 1916/1916 loss 1.18 (ce 0.57, oe 1.22)& 0.00 & 100.00 & 99.99
#
#  & 0.00 & 100.00 & 99.99
# [[16768     0]
#  [ 1313 24950]]
#  epoch 69 1916/1916 loss 0.81 (ce 0.20, oe 1.23)& 0.00 & 99.99 & 99.99
#
#  & 0.00 & 99.99 & 99.99
# [[16765     3]
#  [ 1313 24950]]
#  epoch 79 1916/1916 loss 1.19 (ce 0.57, oe 1.24)& 0.00 & 100.00 & 100.00
#
#  & 0.00 & 100.00 & 100.00
# [[16768     0]
#  [ 1313 24950]]
#  epoch 89 1916/1916 loss 0.84 (ce 0.23, oe 1.23)& 0.00 & 100.00 & 100.00
#
#  & 0.00 & 100.00 & 100.00
# [[16767     1]
#  [ 1313 24950]]
#  epoch 99 1916/1916 loss 1.06 (ce 0.45, oe 1.22)& 0.00 & 99.99 & 99.99
#
#  & 0.00 & 99.99 & 99.99
# [[16766     2]
#  [ 1313 24950]]



# iter 20
#  epoch  9 1916/1916 loss 1.56 (ce 0.98, oe 1.15)& 9.68 & 97.53 & 96.76
#
#  & 9.68 & 97.53 & 96.76
# [[15112  1656]
#  [ 1313 24950]]
#  epoch 19 1916/1916 loss 1.11 (ce 0.53, oe 1.17)& 1.33 & 99.74 & 99.61
#
#  & 1.33 & 99.74 & 99.61
# [[16712    56]
#  [ 1313 24950]]
#  epoch 29 1916/1916 loss 1.51 (ce 0.90, oe 1.21)& 0.33 & 99.91 & 99.87
#
#  & 0.33 & 99.91 & 99.87
# [[16755    13]
#  [ 1313 24950]]
#  epoch 39 1916/1916 loss 1.59 (ce 0.99, oe 1.20)& 0.09 & 99.97 & 99.96
#
#  & 0.09 & 99.97 & 99.96
# [[16763     5]
#  [ 1313 24950]]
#  epoch 49 1916/1916 loss 1.89 (ce 1.29, oe 1.21)& 0.06 & 99.98 & 99.97
#
#  & 0.06 & 99.98 & 99.97
# [[16761     7]
#  [ 1313 24950]]
#  epoch 59 1916/1916 loss 1.13 (ce 0.52, oe 1.22)& 0.02 & 99.99 & 99.98
#
#  & 0.02 & 99.99 & 99.98
# [[16763     5]
#  [ 1313 24950]]
#  epoch 69 1916/1916 loss 1.21 (ce 0.60, oe 1.21)& 0.02 & 99.97 & 99.97
#
#  & 0.02 & 99.97 & 99.97
# [[16760     8]
#  [ 1313 24950]]
#  epoch 79 1916/1916 loss 1.04 (ce 0.43, oe 1.23)& 0.00 & 99.99 & 99.99
#
#  & 0.00 & 99.99 & 99.99
# [[16763     5]
#  [ 1313 24950]]
#  epoch 89 1916/1916 loss 0.87 (ce 0.26, oe 1.21)& 0.01 & 99.98 & 99.98
#
#  & 0.01 & 99.98 & 99.98
# [[16760     8]
#  [ 1313 24950]]
#  epoch 99 1916/1916 loss 1.12 (ce 0.52, oe 1.21)& 0.01 & 99.99 & 99.98
#
#  & 0.01 & 99.99 & 99.98
# [[16762     6]
#  [ 1313 24950]]


# rho 0.05
#  epoch  9 1916/1916 loss 2.05 (ce 1.45, oe 1.20)& 1.03 & 99.41 & 99.30
#
#  & 1.03 & 99.41 & 99.30
# [[16421   347]
#  [ 1313 24950]]
#  epoch 19 1916/1916 loss 1.71 (ce 1.12, oe 1.19)& 13.17 & 91.20 & 73.17
#
#  & 13.17 & 91.20 & 73.17
# [[ 1415 15353]
#  [ 1313 24950]]
#  epoch 29 1916/1916 loss 1.26 (ce 0.66, oe 1.20)& 0.08 & 99.43 & 99.57
#
#  & 0.08 & 99.43 & 99.57
# [[16641   127]
#  [ 1313 24950]]
#  epoch 39 1916/1916 loss 1.79 (ce 1.20, oe 1.18)& 0.04 & 99.17 & 99.43
#
#  & 0.04 & 99.17 & 99.43
# [[16588   180]
#  [ 1313 24950]]
#  epoch 49 1916/1916 loss 1.42 (ce 0.83, oe 1.19)& 0.08 & 99.95 & 99.93
#
#  & 0.08 & 99.95 & 99.93
# [[16748    20]
#  [ 1313 24950]]
#  epoch 59 1916/1916 loss 1.14 (ce 0.55, oe 1.18)& 0.21 & 99.88 & 99.84
#
#  & 0.21 & 99.88 & 99.84
# [[16684    84]
#  [ 1313 24950]]
#  epoch 69 1916/1916 loss 1.21 (ce 0.62, oe 1.17)& 0.00 & 99.54 & 99.70
#
#  & 0.00 & 99.54 & 99.70
# [[16676    92]
#  [ 1313 24950]]
#  epoch 79 1916/1916 loss 1.11 (ce 0.53, oe 1.15)& 0.00 & 99.65 & 99.77
#
#  & 0.00 & 99.65 & 99.77
# [[16698    70]
#  [ 1313 24950]]
#  epoch 89 1916/1916 loss 0.92 (ce 0.35, oe 1.15)& 0.00 & 99.69 & 99.80
#
#  & 0.00 & 99.69 & 99.80
# [[16703    65]
#  [ 1313 24950]]
#  epoch 99 1916/1916 loss 0.85 (ce 0.27, oe 1.15)& 0.00 & 99.67 & 99.79
#
#  & 0.00 & 99.67 & 99.79
# [[16707    61]
#  [ 1313 24950]]


# rho + str
# epoch  9 1916/1916 loss 1.60 (ce 0.99, oe 1.23)& 0.27 & 99.95 & 99.92
#
#  & 0.27 & 99.95 & 99.92
# [[16767     1]
#  [ 1313 24950]]
#  epoch 19 1916/1916 loss 1.02 (ce 0.40, oe 1.24)& 0.02 & 99.86 & 99.90
#
#  & 0.02 & 99.86 & 99.90
# [[16738    30]
#  [ 1313 24950]]
#  epoch 29 1916/1916 loss 1.04 (ce 0.41, oe 1.25)& 0.01 & 99.77 & 99.85
#
#  & 0.01 & 99.77 & 99.85
# [[16726    42]
#  [ 1313 24950]]
#  epoch 39 1916/1916 loss 0.96 (ce 0.33, oe 1.26)& 0.00 & 99.76 & 99.84
#
#  & 0.00 & 99.76 & 99.84
# [[16718    50]
#  [ 1313 24950]]
#  epoch 49 1916/1916 loss 0.93 (ce 0.31, oe 1.24)& 0.01 & 99.99 & 99.99
#
#  & 0.01 & 99.99 & 99.99
# [[16768     0]
#  [ 1313 24950]]
#  epoch 59 1916/1916 loss 1.20 (ce 0.57, oe 1.25)& 0.46 & 99.39 & 99.41
#
#  & 0.46 & 99.39 & 99.41
# [[16504   264]
#  [ 1313 24950]]
#  epoch 69 1916/1916 loss 0.85 (ce 0.23, oe 1.24)& 0.00 & 99.98 & 99.97
#
#  & 0.00 & 99.98 & 99.97
# [[16743    25]
#  [ 1313 24950]]
#  epoch 79 1916/1916 loss 0.95 (ce 0.34, oe 1.22)& 0.00 & 99.98 & 99.98
#
#  & 0.00 & 99.98 & 99.98
# [[16760     8]
#  [ 1313 24950]]
#  epoch 89 1916/1916 loss 0.85 (ce 0.24, oe 1.21)& 0.00 & 99.99 & 99.99
#
#  & 0.00 & 99.99 & 99.99
# [[16766     2]
#  [ 1313 24950]]
#  epoch 99 1916/1916 loss 0.87 (ce 0.26, oe 1.21)& 0.00 & 99.98 & 99.98
#
#  & 0.00 & 99.98 & 99.98
# [[16762     6]
#  [ 1313 24950]]



# beta 0.9
# epoch  9 1916/1916 loss 1.53 (ce 0.94, oe 1.18)& 8.34 & 97.42 & 97.75
#
#  & 8.34 & 97.42 & 97.75
# [[15859   909]
#  [ 1313 24950]]
#  epoch 19 1916/1916 loss 1.83 (ce 1.24, oe 1.18)& 0.53 & 99.66 & 99.57
#
#  & 0.53 & 99.66 & 99.57
# [[16668   100]
#  [ 1313 24950]]
#  epoch 29 1916/1916 loss 1.86 (ce 1.26, oe 1.21)& 0.27 & 99.03 & 99.21
#
#  & 0.27 & 99.03 & 99.21
# [[16553   215]
#  [ 1313 24950]]
#  epoch 39 1916/1916 loss 2.30 (ce 1.72, oe 1.17)& 79.76 & 92.47 & 94.32
#
#  & 79.76 & 92.47 & 94.32
# [[14973  1795]
#  [ 1313 24950]]
#  epoch 49 1916/1916 loss 1.60 (ce 1.02, oe 1.17)& 66.73 & 93.78 & 93.49
#
#  & 66.73 & 93.78 & 93.49
# [[15517  1251]
#  [ 1313 24950]]
#  epoch 59 1916/1916 loss 2.08 (ce 1.50, oe 1.16)& 0.14 & 97.66 & 98.29
#
#  & 0.14 & 97.66 & 98.29
# [[16208   560]
#  [ 1313 24950]]
#  epoch 69 1916/1916 loss 0.82 (ce 0.26, oe 1.13)& 0.43 & 96.75 & 97.81
#
#  & 0.43 & 96.75 & 97.81
# [[16072   696]
#  [ 1313 24950]]
#  epoch 79 1916/1916 loss 1.11 (ce 0.53, oe 1.15)& 0.12 & 97.35 & 98.23
#
#  & 0.12 & 97.35 & 98.23
# [[16256   512]
#  [ 1313 24950]]
#  epoch 89 1916/1916 loss 1.07 (ce 0.49, oe 1.15)& 0.13 & 97.61 & 98.41
#
#  & 0.13 & 97.61 & 98.41
# [[16290   478]
#  [ 1313 24950]]
#  epoch 99 1916/1916 loss 0.77 (ce 0.20, oe 1.13)& 0.05 & 98.35 & 98.93
#
#  & 0.05 & 98.35 & 98.93
# [[16446   322]
#  [ 1313 24950]]



# lr
#  epoch  9 1916/1916 loss 1.45 (ce 0.87, oe 1.15)& 20.24 & 96.17 & 94.88
#
#  & 20.24 & 96.17 & 94.88
# [[13526  3242]
#  [ 1313 24950]]
#  epoch 19 1916/1916 loss 1.10 (ce 0.52, oe 1.17)& 0.95 & 99.77 & 99.68
#
#  & 0.95 & 99.77 & 99.68
# [[16749    19]
#  [ 1313 24950]]
#  epoch 29 1916/1916 loss 1.37 (ce 0.77, oe 1.19)& 0.18 & 99.95 & 99.93
#
#  & 0.18 & 99.95 & 99.93
# [[16763     5]
#  [ 1313 24950]]
#  epoch 39 1916/1916 loss 1.41 (ce 0.81, oe 1.20)& 0.05 & 99.98 & 99.97
#
#  & 0.05 & 99.98 & 99.97
# [[16766     2]
#  [ 1313 24950]]
#  epoch 49 1916/1916 loss 1.33 (ce 0.72, oe 1.21)& 0.11 & 99.96 & 99.94
#
#  & 0.11 & 99.96 & 99.94
# [[16763     5]
#  [ 1313 24950]]
#  epoch 59 1916/1916 loss 1.84 (ce 1.25, oe 1.19)& 0.03 & 99.99 & 99.98
#
#  & 0.03 & 99.99 & 99.98
# [[16766     2]
#  [ 1313 24950]]
#  epoch 69 1916/1916 loss 0.92 (ce 0.32, oe 1.19)& 0.02 & 99.99 & 99.98
#
#  & 0.02 & 99.99 & 99.98
# [[16766     2]
#  [ 1313 24950]]
#  epoch 79 1916/1916 loss 1.22 (ce 0.62, oe 1.21)& 0.02 & 99.98 & 99.98
#
#  & 0.02 & 99.98 & 99.98
# [[16763     5]
#  [ 1313 24950]]
#  epoch 89 1916/1916 loss 1.00 (ce 0.39, oe 1.21)& 0.02 & 99.99 & 99.98
#
#  & 0.02 & 99.99 & 99.98
# [[16763     5]
#  [ 1313 24950]]
#  epoch 99 1916/1916 loss 1.51 (ce 0.91, oe 1.21)& 0.01 & 99.99 & 99.98
#
#  & 0.01 & 99.99 & 99.98
# [[16763     5]
#  [ 1313 24950]]