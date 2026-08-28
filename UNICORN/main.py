import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

from sklearn.metrics import confusion_matrix
from utils.display_results import get_measures, print_measures
import numpy as np
import torch
import torch.nn.functional as F

from DAL import prep, prep_OOD
from UNICORN import ForwConvTabular
from utils import utils

ID_train_dataset = utils.CustomDataset(prep.X_train_sc.to_numpy(), prep.y_train.to_numpy())
ID_test_dataset = utils.CustomDataset(prep.X_test_sc.to_numpy(), prep.y_test.to_numpy())

OOD_train_dataset = utils.CustomDataset(prep_OOD.X_train_sc.to_numpy(), prep_OOD.y_train.to_numpy())
OOD_test_dataset = utils.CustomDataset(prep_OOD.X_test_sc.to_numpy(), prep_OOD.y_test.to_numpy())

train_loader_in, train_loader_out = utils.dataset_function(ID_train_dataset, OOD=OOD_train_dataset,
                                                           batch_size=128,
                                                           batch_size_o=256, train=True)
test_loader_in, test_loader_out = utils.dataset_function(ID_test_dataset, OOD=OOD_test_dataset,
                                                         batch_size=128,
                                                         batch_size_o=256, train=False)

ood_num_examples = len(prep.X_test_sc) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(prep.X_test_sc))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

net = ForwConvTabular(input_dim=58, num_classes=3).cuda()
optim = torch.optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=0.0005,
                        nesterov=True)


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(optim,
                                              lr_lambda=lambda step: cosine_annealing(step, 50 * len(train_loader_in),
                                                                                      1, 1e-6 / 0.01))


def batch_hard_triplet_loss(embeddings, labels, margin=1.0, eps=1e-8):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    labels = labels.unsqueeze(0)
    same_class = labels == labels.T
    diff_class = ~same_class

    diag = torch.eye(same_class.size(0), dtype=torch.bool).cuda()
    same_class = same_class & ~diag

    pos_dist = pairwise_dist.clone()
    pos_dist[~same_class] = -1.0
    hardest_positive = pos_dist.max(dim=1).values

    neg_dist = pairwise_dist.clone()
    neg_dist[~diff_class] = float("inf")
    hardest_negative = neg_dist.min(dim=1).values

    valid = (hardest_positive >= 0) & torch.isfinite(hardest_negative)
    if valid.sum() == 0:
        return torch.tensor(0.0, requires_grad=True).cuda()

    losses = F.relu(hardest_positive[valid] - hardest_negative[valid] + margin)
    return losses.mean()

def train(margin=1.0):
    net.train()
    total_loss, total_triplet, total_ce, n_batches = 0.0, 0.0, 0.0, 0

    for batch_idx, in_set in enumerate(train_loader_in):

        data, target = in_set[0].to(torch.float).cuda(), in_set[1].cuda()

        logits, embedding = net(data)

        l1_triplet = batch_hard_triplet_loss(embedding, target, margin=margin)
        l2_ce = F.cross_entropy(logits[:len(in_set[0])], target)
        loss = l1_triplet + l2_ce

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        total_triplet += l1_triplet.item()
        total_ce += l2_ce.item()
        n_batches += 1
    loss = total_loss / n_batches
    triplet = total_triplet / n_batches
    ce = total_ce / n_batches
    sys.stdout.write('\r epoch %2d %d/%d loss %.2f (ce %f, oe %f)' %
                     (epoch, batch_idx + 1, len(train_loader_in), loss, ce, triplet))
    scheduler.step()

    return loss, triplet, ce


def build_id_clusters(num_classes=3):
    net.eval()
    sums = None
    counts = torch.zeros(num_classes)

    for data, target in train_loader_in:
        data = data.to(torch.float).cuda()
        target = target.cuda()

        _, feat = net(data)  # (logits, feature) -- adjust if needed
        feat = feat.detach().cpu()

        if sums is None:
            sums = torch.zeros(num_classes, feat.shape[1])

        for c in range(num_classes):
            mask = (target.cpu() == c)
            if mask.any():
                sums[c] += feat[mask].sum(dim=0)
                counts[c] += mask.sum()

    centers = sums / counts.unsqueeze(1).clamp(min=1)
    return centers

def get_ood_scores_with_indices(loader, test_bs=200):
    scores = []
    indices = []

    net.eval()
    centers = build_id_clusters()

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // test_bs:
            break
        data = data.to(torch.float).cuda()

        _, feat = net(data)

        dists = torch.cdist(feat.cuda(), centers.cuda())
        min_dist = dists.min(dim=1).values

        scores.append(min_dist.cpu().detach().numpy())

        start = batch_idx * test_bs
        end = start + len(data)
        indices.extend(range(start, end))

    return (
        np.concatenate(scores)[:ood_num_examples],
        np.array(indices[:ood_num_examples])
    )
def get_and_print_results(ood_loader, in_score, out_as_pos=True, num_to_avg=1):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score, _ = get_ood_scores_with_indices(ood_loader)
        if out_as_pos: # OE's defines out samples as positive
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


if __name__ == "__main__":

    for epoch in range(50):
        loss, triplet, ce = train()

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

            torch.save(net.state_dict(), f"wr{loss}.pt")
