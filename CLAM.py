import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from ABMIL import GatedAttentionBasedMILPooling, AttentionBasedMILPooling
from math import log

class SmoothTop1SVMLoss(nn.Module):
    def __init__(self, num_classes, tau=1.0, alpha=None):
        super().__init__()
        self.alpha = alpha if alpha is not None else 1
        self.register_buffer('labels', torch.arange(num_classes))
        self.num_classes = num_classes
        self.tau = tau
        self.thresh = 1e3

    def forward(self, x, y):
        # x: (num_samples, num_classes)
        # y: (num_samples)

        # determine whether to switch to hard SVM loss or not
        top = torch.topk(x, 2, dim=-1)[0]
        hard = torch.ge(top[:, 0] - top[:, 1], 1 * self.tau * log(self.thresh)).detach()
        smooth = torch.logical_not(hard)

        loss = 0.0
        if smooth.data.sum():
            x_s, y_s = x[smooth], y[smooth]

            # 0 if a label is a ground truth label y 
            # else 1
            delta = torch.ne(y_s[:, None], self.labels[None, :]).float() # (num_samples, num_classes)
            x_s = x_s + self.alpha * delta - torch.gather(x_s, dim=1, index=y_s[:, None])
            smooth_loss = self.tau * torch.logsumexp(x_s / self.tau, dim=1)
            loss += smooth_loss.sum() / x_s.size(0)

        if hard.data.sum():
            x_h, y_h = x[hard], y[hard]
            delta = torch.ne(y_h[:, None], self.labels[None, :]).float()
            max_ = (x_h + self.alpha * delta).max(dim=1)
            hard_loss = max_ - torch.gather(x_h, dim=1, index=y_h[:, None])
            loss += hard_loss.sum() / x_h.size(0)

        return loss

class CLAM_SB(nn.Module):
    def __init__(self, gated=True, dropout=0., num_samples=8, embed_dim=1024):
        super().__init__()
        num_classes = 2
        
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if gated:
            self.attention = GatedAttentionBasedMILPooling(512, 256, need_weights=True)
        else:
            self.attention = AttentionBasedMILPooling(512, 256, need_weights=True)

        self.bag_classifier = nn.Linear(512, num_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(512, 2) for i in range(num_classes)])
        self.num_samples = num_samples
        # self.cluster_loss_fn = nn.CrossEntropyLoss()
        self.cluster_loss_fn = SmoothTop1SVMLoss(num_classes=2)

    def cluster_for_true_class(self, attn_w, h, classifier):
        top_p_ids = torch.topk(attn_w, self.num_samples, dim=-1)[1].squeeze()
        top_n_ids = torch.topk(attn_w, self.num_samples, dim=-1, largest=False)[1].squeeze()
        top_p_instances = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_instances = torch.index_select(h, dim=0, index=top_n_ids)

        # generate pseudo labels
        p_targets = torch.ones(self.num_samples).long()
        n_targets = torch.zeros(self.num_samples).long()

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p_instances, top_n_instances], dim=0)
        logits = classifier(all_instances)
        # preds = torch.topk(logits, 1, dim=-1)[1].squeeze(-1)
        loss = self.cluster_loss_fn(logits, all_targets)

        return loss

    """
    def cluster_for_wrong_class(self, attn_w, h, classifier):
        top_p_ids = torch.topk(attn_w, self.num_samples, dim=-1)
        top_p_instances = torch.index_select(h, dim=0, index=top_p_ids)
        targets = torch.zeros(self.num_samples).float()

        logits = classifier(top_n_instances)
        # preds = torch.topk(logits, 1, dim=-1)[1].squeeze(-1)
        loss = self.cluster_loss_fn(logits, targets)

        return loss
    """

    def forward(self, x, labels, clustering=False):
        batch_size = torch.numel(labels)

        h = self.fc(x)
        bag_embed, attn_w = self.attention(h)

        cluster_loss = 0.0
        if clustering:
            for b in range(batch_size):
                label = int(labels[b].item())
                classifier = self.instance_classifiers[label]
                cluster_loss += self.cluster_for_true_class(attn_w[b], h[b], classifier)
            cluster_loss /= batch_size

        logits = self.bag_classifier(bag_embed)
        Y_hat = torch.topk(logits, 1, dim=-1)[1]
        Y_prob = F.softmax(logits, dim=-1)

        ret = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'cluster_loss': cluster_loss}
        return ret

if __name__ == "__main__":
    num_samples = 8
    embed_dim = 1024
    num_patches = 256
    num_batches = 4
    model = CLAM_SB(num_samples=num_samples, embed_dim=embed_dim)
    x = torch.randn(num_batches, num_patches, embed_dim)
    labels = torch.randint(low=0, high=2, size=(num_batches,))
    ret = model(x, labels, clustering=True)
    # print(ret['Y_prob'])
    print(ret['Y_hat'])
    print(ret['cluster_loss'])
