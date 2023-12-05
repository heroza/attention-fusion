from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, accuracy_score, recall_score, f1_score, precision_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import load_metric
from transformers import Trainer
from skimage.measure import find_contours
import cv2
import colorsys
import random

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def get_augmentation(aug):
    if aug == 0:
      the_aug = T.RandomRotation(degrees=(0, 0))
    elif aug == 1:
      the_aug = T.Compose(
        [
            T.RandomRotation(degrees=(0, 180)),
            T.RandomAffine(degrees = 0, translate = (0.1, 0.1), scale=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5)
        ]
      )
    elif aug == 2:
      the_aug = T.AutoAugment()
    elif aug == 3:
      the_aug = T.RandAugment()
    elif aug == 4:
      the_aug = T.AugMix()
    else:
      the_aug = T.TrivialAugmentWide()
    return the_aug

def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def get_the_metrics(y_score, y_true):
    y_pred = np.argmax(y_score, axis=-1)
    y_score_0 = y_score[:,1]
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score_0)
    ap = average_precision_score(y_true, y_score_0)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return (bal_acc, auc, ap, acc, precision, recall, f1)

def my_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
    label_smoothing: float = 0.5,
):
    # print('myloss parameters (alpha, gamma, label_smoothing): ', alpha, gamma, label_smoothing)
    p = torch.sigmoid(inputs)
    targets = (1 - label_smoothing) * targets + label_smoothing / 2
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # ce_loss = F.cross_entropy(inputs, targets, reduction="none", label_smoothing=label_smoothing)
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float().cuda()

    weights = torch.tensor(weights).float().cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

class CustomTrainer(Trainer):
    def __init__(
        self,
        model = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        preprocess_logits_for_metrics = None,
        output_dir = None,
        cls_num_list = None,
        alpha = 0,
        beta = 0,
        gamma = 0,
        label_smoothing = 0,
    ):
        super().__init__(model=model, 
                         args=args, 
                         data_collator=data_collator, 
                         train_dataset=train_dataset, 
                         eval_dataset=eval_dataset, 
                         tokenizer=tokenizer, 
                         model_init=model_init, 
                         compute_metrics=compute_metrics, 
                         callbacks=callbacks,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        self.output_dir = output_dir
        self.cls_num_list = cls_num_list
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False):
              labels = inputs.get("labels")
              outputs = model(**inputs)
              logits = outputs.get("logits")
              if self.output_dir == 'ldam':
                  loss_fct = LDAMLoss(cls_num_list=self.cls_num_list, max_m=0.5, s=30, weight= None)
                  loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
              elif self.beta != 0:
                no_of_classes = 2
                if self.alpha < 0.1:
                    loss_type = "focal"
                elif self.alpha < 1.1:
                    loss_type = "sigmoid"
                elif self.alpha < 2.1:
                    loss_type = "softmax"
                loss = CB_loss(labels, logits, self.cls_num_list, no_of_classes,loss_type, self.beta, self.gamma)
              else:
                labels_one_hot = F.one_hot(labels.view(-1), num_classes=self.model.config.num_labels).float()
                loss = my_loss(logits.view(-1, self.model.config.num_labels), labels_one_hot, alpha=self.alpha, gamma=self.gamma, reduction='mean', label_smoothing=self.label_smoothing)
              return (loss, outputs) if return_outputs else loss

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)