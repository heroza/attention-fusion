import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTConfig
import os
import skimage.io
from PIL import Image
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from incl import *
from self_attention import *

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-a", "--aug", default=3, type=int, help="Image augmentation method")
parser.add_argument("-m", "--model", default=0, type=int, help="Index model")
parser.add_argument("-l", "--lastlayers", default=1, type=int, help="Last-n layers to be replaced")
parser.add_argument("-t", "--threshold", default=None, type=float, help="Threshold to keep for mask")
parser.add_argument("-o", "--output_dir", default="output", type=str, help="Output directory")
args = vars(parser.parse_args())

model_name_or_path = 'google/vit-base-patch16-224'
aug = args["aug"]
idx_model = args["model"]
lastlayers = args["lastlayers"]
threshold = args["threshold"]
output_dir = args["output_dir"]
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
size = feature_extractor.size["height"]
the_aug = get_augmentation(aug)
_train_transforms = T.Compose(
    [
        T.Resize(size, interpolation=T.InterpolationMode.LANCZOS),
        the_aug
    ]
)

_val_transforms = T.Compose(
    [
        T.Resize(size, interpolation=T.InterpolationMode.LANCZOS),
        # T.CenterCrop(size)
    ]
)

def train_transform(example_batch):
    inputs = [_train_transforms(image) for image in example_batch["image"]]
    inputs = feature_extractor([x for x in inputs], return_tensors='pt')

    inputs['labels'] = example_batch['label']
    return inputs

def valid_transform(example_batch):
    inputs = [_val_transforms(image) for image in example_batch["image"]]
    inputs = feature_extractor([x for x in inputs], return_tensors='pt')

    inputs['labels'] = example_batch['label']
    return inputs

images = []
images.append(Image.open(r"/home/rh22708/data/isic2017_base_dir/combined/task-1/test/seb/ISIC_0014653.jpg"))  
inputs = [_val_transforms(image) for image in images]
inputs = feature_extractor([x for x in inputs], return_tensors='pt')

labels = ['combined', 'mel']
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True
)

config = ViTConfig.from_pretrained(model_name_or_path)
if idx_model==0:
    for x in range(lastlayers):
        model.vit.encoder.layer[x-lastlayers].attention.attention = ScaledDotProductAttention(config)
    model.load_state_dict(torch.load('/home/rh22708/pt/exp30/_att1_last3/model_weights_1_1.pth'))
elif idx_model==1:
    for x in range(lastlayers):
        model.vit.encoder.layer[x-lastlayers].attention.attention = MultiplicativeAttention(config)
    model.load_state_dict(torch.load('/home/rh22708/pt/exp30/_att2_last3/model_weights_2_1.pth'))
elif idx_model==2:
    for x in range(lastlayers):
        model.vit.encoder.layer[x-lastlayers].attention.attention = AdditiveAttention(config)
    model.load_state_dict(torch.load('/home/rh22708/pt/exp30/_att3_last3/model_weights_3_1.pth'))

# code based on https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
# forward pass
model.to(device)
with torch.no_grad():
    outputs = model(**inputs.to(device), output_attentions=True)
list_attentions = outputs.attentions[-1]
w_featmap = inputs.pixel_values.shape[-2] // config.patch_size
h_featmap = inputs.pixel_values.shape[-1] // config.patch_size
nh = list_attentions.shape[1] # number of head
for k in range(list_attentions.shape[0]):
    attentions = list_attentions[k, :, 0, 1:].reshape(nh, -1)
    if threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=config.patch_size, mode="nearest")[0].cpu().numpy()
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=config.patch_size, mode="nearest")[0].cpu().numpy()
    torchvision.utils.save_image(torchvision.utils.make_grid(inputs.pixel_values[k,:], normalize=True, scale_each=True), os.path.join(output_dir, str(k) + "-img.png"))
    for j in range(nh):
        fname = os.path.join(output_dir, "attn-head" + str(k) + '-' + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    if threshold is not None:
        image = skimage.io.imread(os.path.join(output_dir, str(k) + "-img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(output_dir, "mask_th" + str(threshold) + "_head" + str(k) + '-' + str(j) +".png"), blur=False)

print('finished')