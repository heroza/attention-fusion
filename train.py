import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, ViTConfig
from collections import Counter
import sys
from transformers.utils import logging
from incl import *
from self_attention import *

logging.disable_progress_bar()

output_dir = sys.argv[1]

att = int(sys.argv[2])
print('att: ', att)
nattlayer = int(sys.argv[3])
print('nattlayer: ', nattlayer)
alpha = 0.8
print('alpha: ', alpha)
beta = 0
print('beta: ', beta)
gamma = 0
print('gamma: ', gamma)
label_smoothing = 0.5
print('label_smoothing: ', label_smoothing)
weight_decay = 0
print('weight_decay: ', weight_decay)
aug = 3
print('aug: ', aug)

np.random.seed(42)
model_name_or_path = 'google/vit-base-patch16-224'
the_aug = get_augmentation(aug)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
size = feature_extractor.size["height"]

_train_transforms = T.Compose(
    [
        T.Resize(size, interpolation=T.InterpolationMode.LANCZOS),
        the_aug
    ]
)

_val_transforms = T.Compose(
    [
        T.Resize(size, interpolation=T.InterpolationMode.LANCZOS)
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

training_args = TrainingArguments(
  output_dir="./"+output_dir,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=20,
  fp16=True,
  save_steps=100,         
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-5,
  weight_decay=weight_decay,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  load_best_model_at_end=True,
)

base_dir = "/home/rh22708/data/isic2017_base_dir/separated/task-1"
print('task-1')
for idx_repeat in range(2):
    #prepare dataset
    ds = load_dataset("imagefolder", data_dir=base_dir, drop_labels=False)
    labels = ds['train'].features['label'].names
    counter = Counter(ds['train']['label'])
    cls_num_list = [value for key, value in counter.most_common()]
    train_ds = ds['train'].with_transform(train_transform)
    val_ds = ds["validation"].with_transform(valid_transform)
    test_ds = ds["test"].with_transform(valid_transform)

    #prepare model
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True
    )

    config = ViTConfig.from_pretrained(model_name_or_path)
    if att == 0:
        pass
    else:
        if att == 1: #scaleddotproduct attention
            for x in range(nattlayer):
                model.vit.encoder.layer[x-nattlayer].attention.attention = ScaledDotProductAttention(config)
        elif att == 2: #multiplicative attention
            for x in range(nattlayer):
                model.vit.encoder.layer[x-nattlayer].attention.attention = MultiplicativeAttention(config)
        elif att == 3: #additive attention
            for x in range(nattlayer):
                model.vit.encoder.layer[x-nattlayer].attention.attention = AdditiveAttention(config)
        elif att == 18: #linformer
            for x in range(nattlayer):
                model.vit.encoder.layer[x-nattlayer].attention.attention = LinformerSelfAttention(dim=768, seq_len = 197, num_heads=12, num_feats=32)
        elif att == 19: #performer
            for x in range(nattlayer):
                model.vit.encoder.layer[x-nattlayer].attention.attention = PerformerWrapper()
        elif att >= 4: #max fusion
            for x in range(nattlayer):
                model.vit.encoder.layer[x-nattlayer].attention.attention = CustomAttention(config,att)

    #prepare Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        output_dir = output_dir,
        cls_num_list = cls_num_list,
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        label_smoothing = label_smoothing,
    )

    # Training
    print('start training')
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    #Eval on training set
    predictions = trainer.predict(train_ds)
    metrics = get_the_metrics(predictions.predictions, predictions.label_ids)
    print('eval on train set: ', end='')
    print(metrics)

    #Eval on validation set
    predictions = trainer.predict(val_ds)
    metrics = get_the_metrics(predictions.predictions, predictions.label_ids)
    print('eval on validation set: ', end='')
    print(metrics)

    #Eval on test set
    predictions = trainer.predict(test_ds)
    metrics = get_the_metrics(predictions.predictions, predictions.label_ids)
    print('eval on test set: ', end='')
    print(metrics)

    # confusion matrix
    y_pred = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(predictions.label_ids, y_pred)
    print('confusion_matrix: ', cm)

    print('saving the model')
    torch.save(trainer.model.state_dict(), output_dir+'/model_weights_'+str(att)+'_'+str(idx_repeat)+'.pth')
    base_dir = "/home/rh22708/data/isic2017_base_dir/separated/task-2" 
    print('task-2')
print('finished')