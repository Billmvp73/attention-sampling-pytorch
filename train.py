import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pdb

from utils import calc_cls_measures, move_to
from ats.utils import visualize, showPatch, patchGrid, mapGrid

def train(model, optimizer, train_loader, criterion, entropy_loss_func, opts):
    """ Train for a single epoch """

    y_probs = np.zeros((0, len(train_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []

    # Put model in training mode
    model.train()

    for i, (x_low, x_high, label) in enumerate(tqdm(train_loader)):
        x_low, x_high, label = move_to([x_low, x_high, label], opts.device)

        optimizer.zero_grad()
        y, attention_map, patches, x_low = model(x_low, x_high)

        entropy_loss = entropy_loss_func(attention_map)

        loss = criterion(y, label) - entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clipnorm)
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        y_prob = F.softmax(y, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    train_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)
    return train_loss_epoch, metrics


def evaluate(model, test_loader, criterion, entropy_loss_func, opts):
    """ Evaluate a single epoch """

    y_probs = np.zeros((0, len(test_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []

    # Put model in eval mode
    model.eval()

    for i, (x_low, x_high, label) in enumerate(tqdm(test_loader)):

        x_low, x_high, label = move_to([x_low, x_high, label], opts.device)

        y, attention_map, patches, x_low = model(x_low, x_high)

        entropy_loss = entropy_loss_func(attention_map)
        loss = criterion(y, label) - entropy_loss

        loss_value = loss.item()
        losses.append(loss_value)

        y_prob = F.softmax(y, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    test_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)
    return test_loss_epoch, metrics

#----- Train/Evaluate MultiResolution ------
def trainMultiRes(model, optimizer, train_loader, criterion, entropy_loss_func, opts):
    """ Train for a single epoch """

    y_probs = np.zeros((0, len(train_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []

    # Put model in training mode
    model.train()

    for i, (x_lows, x_highs, label) in enumerate(tqdm(train_loader)):
        x_lows, x_highs, label = move_to([x_lows, x_highs, label], opts.device)

        optimizer.zero_grad()
        y, attention_maps, patches, x_lows = model(x_lows, x_highs)
       
        if type(attention_maps) is list:
            # for attention_map in attention_maps:
            #     print(torch.max(attention_map))
            #     print(torch.min(attention_map))
            entropy_loss = torch.tensor([entropy_loss_func(attention_map) for attention_map in attention_maps]).sum() / len(opts.scales)

            loss = criterion(y, label) - entropy_loss
        else:
            entropy_loss = entropy_loss_func(attention_maps)
            loss = criterion(y, label) - entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clipnorm)
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        y_prob = F.softmax(y, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    train_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)
    return train_loss_epoch, metrics


def evaluateMultiRes(model, test_loader, criterion, entropy_loss_func, opts):
    """ Evaluate a single epoch """

    y_probs = np.zeros((0, len(test_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []

    # Put model in eval mode
    model.eval()

    for i, (x_lows, x_highs, label) in enumerate(tqdm(test_loader)):

        x_lows, x_highs, label = move_to([x_lows, x_highs, label], opts.device)

        y, attention_maps, patches, x_lows = model(x_lows, x_highs)

        ## visualize
        # for i, (scale, x_low) in  enumerate(zip(model.scales, x_lows)):
        #     if type(attention_maps) is list:
        #         ats_map = attention_maps[i]
        #         showPatch()
        if opts.visualize:
            for b in range(patches.shape[0]):
                batch_patches = patches[b]
                patchGrid(batch_patches, (3, 5))
                if type(attention_maps) is list:
                    for attention_map in attention_maps:
                        print(torch.max(attention_map))
                        print(torch.min(attention_map))
                    batch_maps = [attention_maps[i][b] for i in range(len(model.scales))]
                else:
                    batch_maps = [attention_maps[b] for i in range(len(model.scales))]
                batch_imgs = [x_lows[i][b] for i in range(len(model.scales))]
                mapGrid(batch_maps, batch_imgs, model.scales)

        if type(attention_maps) is list:
            
            entropy_loss = torch.tensor([entropy_loss_func(attention_map) for attention_map in attention_maps]).sum() / len(opts.scales)

            loss = criterion(y, label) - entropy_loss
        else:
            entropy_loss = entropy_loss_func(attention_maps)
            loss = criterion(y, label) - entropy_loss

        loss_value = loss.item()
        losses.append(loss_value)

        y_prob = F.softmax(y, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

    test_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)
    return test_loss_epoch, metrics

def trainMultiResBatches(model, optimizer, train_loader, criterion, entropy_loss_func, opts):
    """ Train for a single epoch """

    y_probs = np.zeros((0, len(train_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = [[] for s in opts.scales]
    
    metrics = []
    # Put model in training mode
    model.train()

    for i, (x_low, x_high, label) in enumerate(tqdm(train_loader)):
        loss_set = []
        # high res batch
        x_low, x_high, label = move_to([x_low, x_high, label], opts.device)

        optimizer.zero_grad()

        for i in range(0, len(opts.scales)):
            s = opts.scales[i]
            x_low_i = F.interpolate(x_low, scale_factor = s, mode='bilinear')
            x_high_i = F.interpolate(x_high, scale_factor = s, mode='bilinear')

            x_low_i, x_high_i = move_to([x_low_i, x_high_i], opts.device)

            # optimizer.zero_grad()
            y_i, attention_map, patches, x_low_i_out = model(x_low_i, x_high_i)

            entropy_loss = entropy_loss_func(attention_map)

            loss = criterion(y_i, label) - entropy_loss
            loss_set.append(loss)
            # loss_i.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clipnorm)

            # optimizer.step()

            loss_value = loss.item()

            losses[i].append(loss_value)

            y_prob = F.softmax(y_i, dim=1)
            y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
            y_trues = np.concatenate([y_trues, label.cpu().numpy()])

            metric = calc_cls_measures(y_probs, y_trues)
            metrics.append(metric)
        loss = None
        for i in loss_set:
            if loss is None:
                loss = i
            else:
                loss += i
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clipnorm)
        optimizer.step()
    train_loss_epoch = [np.round(np.mean(loss_s), 4) for loss_s in losses]
    # metrics = calc_cls_measures(y_probs, y_trues)
    return train_loss_epoch, metrics

def evaluateMultiResBatches(model, test_loader, criterion, entropy_loss_func, opts):
    """ Train for a single epoch """

    y_probs = np.zeros((0, len(test_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = [[] for s in opts.scales]
    metrics = []
    # Put model in eval mode
    model.eval()

    for i, (x_low, x_high, label) in enumerate(tqdm(test_loader)):
        # high res batch
        x_low, x_high, label = move_to([x_low, x_high, label], opts.device)

        y, attention_map, patches, x_low_out = model(x_low, x_high)

        entropy_loss = entropy_loss_func(attention_map)

        loss = criterion(y, label) - entropy_loss

        loss_value = loss.item()
        losses[0].append(loss_value)

        y_prob = F.softmax(y, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

        metric = calc_cls_measures(y_probs, y_trues)
        metrics.append(metric)

        # scale-2 low res batch
        for i in range(1, len(opts.scales)):
            s = opts.scales[i]
            x_low_i = F.interpolate(x_low, scale_factor = s, mode='bilinear')
            x_high_i = F.interpolate(x_high, scale_factor = s, mode='bilinear')

            x_low_i, x_high_i = move_to([x_low_i, x_high_i], opts.device)

            y, attention_map, patches, x_low_i_out = model(x_low_i, x_high_i)

            entropy_loss = entropy_loss_func(attention_map)

            loss = criterion(y, label) - entropy_loss

            loss_value = loss.item()

            losses[i].append(loss_value)

            y_prob = F.softmax(y, dim=1)
            y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
            y_trues = np.concatenate([y_trues, label.cpu().numpy()])

            metric = calc_cls_measures(y_probs, y_trues)
            metrics.append(metric)

    test_loss_epoch = [np.round(np.mean(loss_s), 4) for loss_s in losses]
    # metrics = calc_cls_measures(y_probs, y_trues)
    return test_loss_epoch, metrics

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch