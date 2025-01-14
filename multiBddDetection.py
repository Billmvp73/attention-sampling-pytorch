import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
import numpy as np
from models.attention_model import AttentionModelBddDetection, AttentionModelMultiBddDetection
from models.feature_model import FeatureModelBddDetection
from models.classifier import ClassificationHead

from ats.core.ats_layer import ATSModel, MultiATSModel, MultiParallelATSModel, MultiAtsParallelATSModel, FixedNParallelATSModel
from ats.utils.regularizers import MultinomialEntropy
from ats.utils.logging import AttentionSaverMultiBddDetection, AttentionSaverMultiParallelBddDetection, AttentionSaverMultiBatchBddDetection

from dataset.bdd_detection_dataset import BddDetection
from dataset.multiBddDetectionDataset import MultiBddDetection
from train import trainMultiResBatches, evaluateMultiResBatches, train, evaluate, trainMultiRes, evaluateMultiRes, save_checkpoint, load_checkpoint

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def main(opts):
    if not os.path.exists(opts.output_dir):
      os.mkdir(opts.output_dir)
    if '/' not in opts.load_dir:
      opts.load_dir = os.path.join(opts.output_dir, opts.load_dir)
    print(opts.load_dir)
    if not os.path.exists(opts.load_dir):
      os.mkdir(opts.load_dir)
    if not opts.multiResBatch:
      train_dataset = MultiBddDetection('dataset/bdd_detection', split="train", scales = opts.scales)
      test_dataset = MultiBddDetection('dataset/bdd_detection', split='val', scales = opts.scales)
    else:
      train_dataset = BddDetection('dataset/bdd_detection', split="train")
      test_dataset = BddDetection('dataset/bdd_detection', split="val")
    print(len(train_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=opts.batch_size, num_workers=opts.num_workers)

    if not opts.multiResBatch:
      attention_model = AttentionModelBddDetection(squeeze_channels=True, softmax_smoothing=1e-4)
      feature_model = FeatureModelBddDetection(in_channels=3, strides=[1, 2, 2, 2], filters=[32, 32, 32, 32])
      classification_head = ClassificationHead(in_channels=32, num_classes=len(train_dataset.CLASSES))

      ats_model = None
      logger = None
      if opts.map_parallel:
          print("Run parallel model.")
          print("n patches for high res, and another n for low res.")
          if opts.parallel_models:
            print("Multiple attention models for multiple scales.")
            attention_models = [AttentionModelBddDetection(squeeze_channels=True, softmax_smoothing=1e-4).to(opts.device) for _ in opts.scales]
            if not opts.norm_resample:
              if opts.fixed_patches is None:
                ats_model = MultiAtsParallelATSModel(attention_models, feature_model, classification_head, n_patches=opts.n_patches, patch_size=opts.patch_size, scales=opts.scales)
              else:
                opts.fixed_patches = [32, 8, 2]
                ats_model = FixedNParallelATSModel(attention_models, feature_model, classification_head, opts.fixed_patches, patch_size=opts.patch_size, scales=opts.scales)
            else:
              # Normalize the probability of samples among all scales
              ats_model = MultiAtsParallelATSModel(attention_models, feature_model, classification_head, n_patches=opts.n_patches, patch_size=opts.patch_size, scales=opts.scales, norm_resample=True, norm_atts_weight=opts.norm_atts_weight)
          else:
            print("Single attention models for multiple scales.")
            ats_model = MultiParallelATSModel(attention_model, feature_model, classification_head, n_patches=opts.n_patches, patch_size=opts.patch_size, scales=opts.scales)
          ats_model = ats_model.to(opts.device)

          logger = AttentionSaverMultiParallelBddDetection(opts.output_dir, ats_model, test_dataset, opts)

      else:
          print("Run unparallel model.")
          attention_model = AttentionModelMultiBddDetection(squeeze_channels=True, softmax_smoothing=1e-4)
          if opts.area_norm:
            print("Merge before softmax with area normalization.")
            ats_model = MultiATSModel(attention_model, feature_model, classification_head, n_patches=opts.n_patches, patch_size=opts.patch_size, scales=opts.scales, area_norm=True)
          else:
            print("Merge before softmax without area normalization.")
            ats_model = MultiATSModel(attention_model, feature_model, classification_head, n_patches=opts.n_patches, patch_size=opts.patch_size, scales=opts.scales, area_norm=False)
          ats_model = ats_model.to(opts.device)

          logger = AttentionSaverMultiBddDetection(opts.output_dir, ats_model, test_dataset, opts)
    else:
        attention_model = AttentionModelBddDetection(squeeze_channels=True, softmax_smoothing=1e-4)
        feature_model = FeatureModelBddDetection(in_channels=3, strides=[1, 2, 2, 2], filters=[32, 32, 32, 32])
        classification_head = ClassificationHead(in_channels=32, num_classes=len(train_dataset.CLASSES))

        ats_model = ATSModel(attention_model, feature_model, classification_head, n_patches=opts.n_patches, patch_size=opts.patch_size, replace=True)
        ats_model = ats_model.to(opts.device)
        logger = AttentionSaverMultiBatchBddDetection(opts.output_dir, ats_model, test_dataset, opts)
    # ats_model = ats_model.to(opts.device)
    if not opts.parallel_models:
      optimizer = optim.Adam([{'params': ats_model.attention_model.part1.parameters(), 'weight_decay': 1e-5},
                            {'params': ats_model.attention_model.part2.parameters()},
                            {'params': ats_model.feature_model.parameters()},
                            {'params': ats_model.classifier.parameters()},
                            {'params': ats_model.sampler.parameters()},
                            {'params': ats_model.expectation.parameters()}
                            ], lr=opts.lr)
    else:
      if opts.fixed_patches is None:
        optimizer = optim.Adam([{'params': ats.part1.parameters(), 'weight_decay': 1e-5} for ats in ats_model.attention_models] + [{'params': ats.part2.parameters()} for ats in ats_model.attention_models] + [{'params': ats_model.feature_model.parameters()}, {'params': ats_model.classifier.parameters()}, {'params': ats_model.sampler.parameters()},{'params': ats_model.expectation.parameters()}], lr=opts.lr)
      else:
        optimizer = optim.Adam([{'params': ats.part1.parameters(), 'weight_decay': 1e-5} for ats in ats_model.attention_models] + [{'params': ats.part2.parameters()} for ats in ats_model.attention_models] + [{'params': ats_model.feature_model.parameters()}, {'params': ats_model.classifier.parameters()}, {'params': ats_model.expectation.parameters()}] + [{'params': sampler.parameters()} for sampler in ats_model.sampler_list], lr=opts.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.decrease_lr_at, gamma=0.1)

    
    class_weights = train_dataset.class_frequencies
    class_weights = torch.from_numpy((1. / len(class_weights)) / class_weights).to(opts.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    entropy_loss_func = MultinomialEntropy(opts.regularizer_strength)

    start_epoch = 0
    opts.checkpoint_path = os.path.join(opts.output_dir, "checkpoint")
    if not os.path.exists(opts.checkpoint_path):
       os.mkdir(opts.checkpoint_path)
    if opts.resume:
        # start_epoch = opts.load_epoch + 1
        ats_model, optimizer, start_epoch = load_checkpoint(ats_model, optimizer, os.path.join(opts.load_dir , "checkpoint{:02d}.pth".format(opts.load_epoch)))
        start_epoch += 1
        print("load %s successfully."%(os.path.join(opts.load_dir , "checkpoint{:02d}.pth".format(opts.load_epoch))))
    else:
      print("nothing to load.")

    for epoch in range(start_epoch, opts.epochs):
        print("Start epoch %d"%epoch)
        if not opts.visualize:
          if opts.multiResBatch:
            train_loss, train_metrics = trainMultiResBatches(ats_model, optimizer, train_loader, criterion, entropy_loss_func, opts)
          else:
            train_loss, train_metrics = trainMultiRes(ats_model, optimizer, train_loader, criterion, entropy_loss_func, opts)
          # if epoch % 2 == 0:
          save_checkpoint(ats_model, optimizer, os.path.join(opts.checkpoint_path, "checkpoint{:02d}.pth".format(epoch)), epoch)
          print("Save "+os.path.join(opts.checkpoint_path, "checkpoint{:02d}.pth".format(epoch))+" successfully.")
          if not opts.multiResBatch:
            print("Epoch {}, train loss: {:.3f}, train metrics: {:.3f}".format(epoch, train_loss, train_metrics["accuracy"]))
          else:
            scale_avg = [[], []]
            for i, s in enumerate(opts.scales):
              print("Epoch {}, scale {}, train loss: {:.3f}, train metrics: {:.3f}".format(epoch, s, train_loss[i], train_metrics[i]["accuracy"]))
              scale_avg[0].append(train_loss[i])
              scale_avg[1].append(train_metrics[i]['accuracy'])
            avg_train_loss = np.round(np.mean(scale_avg[0]), 4)
            avg_train_metrics = np.mean(scale_avg[1])
            print("Epoch {}, avg train loss: {:.3f}, train metrics: {:.3f}".format(epoch, avg_train_loss, avg_train_metrics))
        with torch.no_grad():
          if opts.multiResBatch:
            test_loss, test_metrics = evaluateMultiResBatches(ats_model, test_loader, criterion, entropy_loss_func, opts)
          else:
            test_loss, test_metrics = evaluateMultiRes(ats_model, test_loader, criterion, entropy_loss_func, opts)
        logger(epoch, (train_loss, test_loss), (train_metrics, test_metrics))
        if not opts.multiResBatch:
          print("Epoch {}, test loss: {:.3f}, test metrics: {:.3f}".format(epoch, test_loss, test_metrics["accuracy"]))
        else:
          scale_avg = [[], []]
          for i, s in enumerate(opts.scales):
            print("Epoch {}, scale {} test loss: {:.3f}, test metrics: {:.3f}".format(epoch, s, test_loss[i], test_metrics[i]["accuracy"]))
            scale_avg[0].append(test_loss[i])
            scale_avg[1].append(test_metrics[i]["accuracy"])
          avg_test_loss = np.round(np.mean(scale_avg[0]), 4)
          avg_test_metrics = np.mean(scale_avg[1])
          print("Epoch {}, avg test loss: {:.3f}, test metrics: {:.3f}".format(epoch, avg_test_loss, avg_test_metrics))
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularizer_strength", type=float, default=0.05,
                        help="How strong should the regularization be for the attention")
    parser.add_argument("--softmax_smoothing", type=float, default=1e-4,
                        help="Smoothing for calculating the attention map")
    parser.add_argument("--lr", type=float, default=0.001, help="Set the optimizer's learning rate")
    parser.add_argument("--n_patches", type=int, default=5, help="How many patches to sample")
    parser.add_argument("--patch_size", type=int, default=100, help="Patch size of a square patch")
    parser.add_argument("--scales", type=list, default=[1, 0.5, 0.25], help="Multi scales")
    parser.add_argument("--batch_size", type=int, default=32, help="Choose the batch size for SGD")
    parser.add_argument("--epochs", type=int, default=500, help="How many epochs to train for")
    parser.add_argument("--decrease_lr_at", type=float, default=250, help="Decrease the learning rate in this epoch")
    parser.add_argument("--clipnorm", type=float, default=1, help="Clip the norm of the gradients")
    parser.add_argument("--output_dir", type=str, help="An output directory", default='output/bdd_detection')
    # parser.add_argument("--checkpoint_path", type=str, help="An output checkpoint directory", default='output/bdd_detection/checkpoint')
    parser.add_argument("--map_parallel", type=bool, default=False)
    parser.add_argument("--parallel_models", type=bool, default=False)
    parser.add_argument("--norm_resample", type=bool, default=False),
    parser.add_argument("--fixed_patches", type=bool, default=None),
    parser.add_argument("--norm_atts_weight", type=bool, default=False)
    parser.add_argument("--area_norm", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--multiResBatch", type=bool, default=False, help="Flag to train multiresolution in separate batches")
    parser.add_argument("--visualize", type=bool, default=False)
    parser.add_argument("--load_dir", type=str, default="output/bdd_detection/checkpoint")
    parser.add_argument("--load_epoch", type=int, default=0)
    parser.add_argument('--run_name', type=str, default='run')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers to use for data loading')

    opts = parser.parse_args()
    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    opts.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(opts)