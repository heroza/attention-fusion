import numpy as np
import argparse
import torch
import torchvision.transforms as T
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, ViTConfig, AutoImageProcessor, DeiTForImageClassification
from collections import Counter
import sys
from transformers.utils import logging
from incl import *
from self_attention import *

def train_transform(example_batch):
    _train_transforms = T.Compose(
        [
            T.Resize(size, interpolation=T.InterpolationMode.LANCZOS),
            imageAugmentation
        ]
    )
    inputs = [_train_transforms(image) for image in example_batch["image"]]
    inputs = feature_extractor([x for x in inputs], return_tensors='pt')

    inputs['labels'] = example_batch['label']
    return inputs

def valid_transform(example_batch):
    _val_transforms = T.Compose(
        [
            T.Resize(size, interpolation=T.InterpolationMode.LANCZOS)
        ]
    )
    inputs = [_val_transforms(image) for image in example_batch["image"]]
    inputs = feature_extractor([x for x in inputs], return_tensors='pt')

    inputs['labels'] = example_batch['label']
    return inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fusion Attention Model')
    parser.add_argument('--lr', default=0.00002, type=float, nargs='+', help='learning rate')
    parser.add_argument("--epochs", default=20, type=int, help="Epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")

    parser.add_argument("--pretrained_model", default="google/vit-base-patch16-224", type=str, help="Path of the pretrained model")
    parser.add_argument('--output_dir', default='.', help='Path where to save result.')
    parser.add_argument("--att", default=13, type=int, help="Attention fusion type")
    parser.add_argument("--nattlayer", default=3, type=int, help="Number of n-last fusion attention block")
    parser.add_argument("--alpha", default=0.8, type=float, help="Weight of cross entropy loss")
    parser.add_argument("--beta", default=0, type=float, help="Beta value of balanced-class loss")
    parser.add_argument("--gamma", default=0, type=float, help="Gamma value of focal loss")
    parser.add_argument("--label_smoothing", default=0.5, type=float, help="Label smoothing on cross entropy loss")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight decay")
    parser.add_argument("--aug", default=3, type=int, help="Augmentation type")
    args = parser.parse_args()

    logging.disable_progress_bar()
    np.random.seed(42)

    imageAugmentation = get_augmentation(args.aug)
    feature_extractor = ViTFeatureExtractor.from_pretrained(args.pretrained_model) # for ViT
    size = feature_extractor.size["height"]

    training_args = TrainingArguments(
      output_dir="./"+args.output_dir,
      per_device_train_batch_size=args.batch_size,
      evaluation_strategy="steps",
      num_train_epochs=args.epochs,
      fp16=True,
      save_steps=100,         
      eval_steps=100,
      logging_steps=10,
      learning_rate=args.lr,
      weight_decay=args.weight_decay,
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
        model = getFusionModel(args.pretrained_model, labels, args.att, args.nattlayer)

        #prepare Trainer
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            output_dir = args.output_dir,
            cls_num_list = cls_num_list,
            alpha = args.alpha,
            beta = args.beta,
            gamma = args.gamma,
            label_smoothing = args.label_smoothing,
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
        torch.save(trainer.model.state_dict(), args.output_dir+'/model_weights_'+str(args.att)+'_'+str(idx_repeat)+'.pth')
        base_dir = "/home/rh22708/data/isic2017_base_dir/separated/task-2" 
        print('task-2')
    print('finished')
