import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
import numpy as np
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from TGAN_archi_opt import WideResNet
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
parser.add_argument('--epochs', '-e', type=int, default=2, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=13, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=10, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# DAL hyper parameters
parser.add_argument('--gamma', default=1, type=float)
parser.add_argument('--beta',  default=0.5, type=float)
parser.add_argument('--rho',   default=0.01, type=float)
parser.add_argument('--strength', default=0.01, type=float)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--iter', default=10, type=int)
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

train_dataset = utils.CustomDataset(prep.X_train_disc.to_numpy(), prep.y_train.to_numpy())
test_dataset = utils.CustomDataset(prep.X_test_disc.to_numpy(), prep.y_test.to_numpy())

train_loader_in, train_loader_out = utils.dataset_function(train_dataset, X = prep.X_train_disc, batch_size_t = args.batch_size, batch_size_o=args.oe_batch_size, train=True)
test_loader_in, test_loader_out = utils.dataset_function(train_dataset, X = prep.X_test_disc, batch_size_t = args.batch_size, batch_size_o=args.oe_batch_size, train=False)
ood_num_examples = len(prep.X_test) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(prep.X_test))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def get_ood_scores(loader, in_dist=False):
    _score = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data, target = data.to(torch.float).cuda(), target.cuda()
            output = net(data)
            smax = to_np(F.softmax(output, dim=1))
            _score.append(-np.max(smax, axis=1))
    if in_dist:
        return concat(_score).copy() # , concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def get_and_print_results(ood_loader, in_score, num_to_avg=1):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr, '')
    return fpr, auroc, aupr

def train(epoch, gamma):

    net.train()

    loss_avg = 0.0
    train_loader_out.dataset._regenerate()
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
        loss = l_ce + .5 * l_oe

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        sys.stdout.write('\r epoch %2d %d/%d loss %.2f' % (epoch, batch_idx + 1, len(train_loader_in), loss_avg))
        scheduler.step()

    return gamma

def test():
    net.eval()
    correct = 0
    y, c = [], []
    with torch.no_grad():
        for data, target in test_loader_in:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    return correct / len(test_loader_in.dataset) * 100


num_classes = 3
net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()

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
    gamma = 0.01

    in_batch, _ = next(iter(train_loader_in))
    out_batch, _ = next(iter(train_loader_out))

    in_batch = in_batch.numpy()
    out_batch = out_batch.numpy()

    print("ID shape:", in_batch.shape, " OOD shape:", out_batch.shape)


    # X = np.vstack([in_batch, out_batch])
    # y = np.concatenate([np.zeros(len(in_batch)), np.ones(len(out_batch))])
    #
    # clf = RandomForestClassifier(n_estimators=200, max_depth=6)
    # scores = cross_val_score(clf, X, y, cv=3, scoring='roc_auc')
    # print(f"ID vs OOD separability (AUROC): {scores.mean():.3f}")
    for epoch in range(args.epochs):
        gamma = train(epoch, gamma)

        if epoch % 2 == 0:
            net.eval()
            in_score = get_ood_scores(test_loader_in, in_dist=True)
            metric_ll = []
            metric_ll.append(get_and_print_results(test_loader_out, in_score))
            print('\n & %.2f & %.2f & %.2f' % tuple((100 * torch.Tensor(metric_ll).mean(0)).tolist()))


