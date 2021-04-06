import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time

from models.attention_model import AttentionModelBddDetection, AttentionModelMultiBddDetection
from models.feature_model import FeatureModelBddDetection
from models.classifier import ClassificationHead

from ats.core.ats_layer import ATSModel, MultiATSModel, MultiParallelATSModel
from ats.utils.regularizers import MultinomialEntropy
from ats.utils.logging import AttentionSaverMultiBddDetection, AttentionSaverMultiParallelBddDetection

from dataset.bdd_detection_dataset import BddDetection
from dataset.multiBddDetectionDataset import MultiBddDetection
from train import train, evaluate, trainMultiRes, evaluateMultiRes, save_checkpoint, load_checkpoint

def main(opts):
    if not os.path.exists(opts.load_dir):
      os.mkdir(opts.load_dir)
    train_dataset = MultiBddDetection('dataset/bdd_detection', split="train", scales = [1, 0.5, 0.25])
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    test_dataset = MultiBddDetection('dataset/bdd_detection', split='val', scales = [1, 0.5, 0.25])
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=opts.batch_size, num_workers=opts.num_workers)

    attention_model = AttentionModelBddDetection(squeeze_channels=True, softmax_smoothing=1e-4)
    feature_model = FeatureModelBddDetection(in_channels=3, strides=[1, 2, 2, 2], filters=[32, 32, 32, 32])
    classification_head = ClassificationHead(in_channels=32, num_classes=len(train_dataset.CLASSES))

    ats_model = None
    logger = None
    if opts.map_parallel:
        print("Run parallel model.")
        print("n patches for high res, and another n for low res.")
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

    # ats_model = ats_model.to(opts.device)
    optimizer = optim.Adam([{'params': ats_model.attention_model.part1.parameters(), 'weight_decay': 1e-5},
                            {'params': ats_model.attention_model.part2.parameters()},
                            {'params': ats_model.feature_model.parameters()},
                            {'params': ats_model.classifier.parameters()},
                            {'params': ats_model.sampler.parameters()},
                            {'params': ats_model.expectation.parameters()}
                            ], lr=opts.lr)
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
        train_loss, train_metrics = trainMultiRes(ats_model, optimizer, train_loader, criterion, entropy_loss_func, opts)
        if epoch % 2 == 0:
          save_checkpoint(ats_model, optimizer, os.path.join(opts.checkpoint_path, "checkpoint{:02d}.pth".format(epoch)), epoch)
          print("Save "+os.path.join(opts.checkpoint_path, "checkpoint{:02d}.pth".format(epoch))+" successfully.")
        print("Epoch {}, train loss: {:.3f}, train metrics: {:.3f}".format(epoch, train_loss, train_metrics["accuracy"]))
        with torch.no_grad():
            test_loss, test_metrics = evaluateMultiRes(ats_model, test_loader, criterion, entropy_loss_func, opts)

        logger(epoch, (train_loss, test_loss), (train_metrics, test_metrics))
        print("Epoch {}, test loss: {:.3f}, test metrics: {:.3f}".format(epoch, test_loss, test_metrics["accuracy"]))
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
    parser.add_argument("--area_norm", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--load_dir", type=str, default="output/bdd_detection/checkpoint")
    parser.add_argument("--load_epoch", type=int, default=0)
    parser.add_argument('--run_name', type=str, default='run')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers to use for data loading')

    opts = parser.parse_args()
    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    opts.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(opts)