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
    total_sampled_scales = np.zeros(len(opts.scales), dtype=np.int)
    # Put model in training mode
    model.train()

    for i, (x_lows, x_highs, label) in enumerate(tqdm(train_loader)):
        x_lows, x_highs, label = move_to([x_lows, x_highs, label], opts.device)

        optimizer.zero_grad()
        y, attention_maps, patches, x_lows, patch_features, sampled_scales = model(x_lows, x_highs)
       
        if type(attention_maps) is list:
            if sampled_scales is not None:
                freq_sampled_scales = np.bincount(sampled_scales.data.cpu().numpy().reshape(-1), minlength=len(opts.scales))
                entropy_loss = torch.tensor([freq_sampled_scales[i] * entropy_loss_func(attention_map) for i, attention_map in enumerate(attention_maps)]).sum() / freq_sampled_scales.sum()
            else:
                entropy_loss = torch.tensor([entropy_loss_func(attention_map) for attention_map in attention_maps]).sum() / len(opts.scales)
            # entropy_loss = torch.tensor([entropy_loss_func(attention_map * scale ** 2) for attention_map, scale in zip(attention_maps, opts.scales)]).sum() / len(opts.scales)

            loss = criterion(y, label) - entropy_loss
            if sampled_scales is not None:
                freq_sampled_scales = np.bincount(sampled_scales.data.cpu().numpy().reshape(-1), minlength=len(opts.scales))
                total_sampled_scales += freq_sampled_scales
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
    print("Sampled scale frequencies: ", total_sampled_scales)
    return train_loss_epoch, metrics


def evaluateMultiRes(model, test_loader, criterion, entropy_loss_func, opts):
    """ Evaluate a single epoch """

    y_probs = np.zeros((0, len(test_loader.dataset.CLASSES)), np.float)
    y_trues = np.zeros((0), np.int)
    losses = []
    total_sampled_scales = np.zeros(len(opts.scales), dtype=np.int)
    # Put model in eval mode
    model.eval()

    for i, (x_lows, x_highs, label) in enumerate(tqdm(test_loader)):

        x_lows, x_highs, label = move_to([x_lows, x_highs, label], opts.device)

        y, attention_maps, patches, x_lows, patch_features, sampled_scales = model(x_lows, x_highs)

        ## visualize
        # for i, (scale, x_low) in  enumerate(zip(model.scales, x_lows)):
        #     if type(attention_maps) is list:
        #         ats_map = attention_maps[i]
        #         showPatch()

        if type(attention_maps) is list:
            
            if sampled_scales is not None:
                freq_sampled_scales = np.bincount(sampled_scales.data.cpu().numpy().reshape(-1), minlength=len(opts.scales))
                entropy_loss = torch.tensor([freq_sampled_scales[i] * entropy_loss_func(attention_map) for i, attention_map in enumerate(attention_maps)]).sum() / freq_sampled_scales.sum()
            else:
                entropy_loss = torch.tensor([entropy_loss_func(attention_map) for attention_map in attention_maps]).sum() / len(opts.scales)

            loss = criterion(y, label) - entropy_loss
            if sampled_scales is not None:
                freq_sampled_scales = np.bincount(sampled_scales.data.cpu().numpy().reshape(-1), minlength=len(opts.scales))
                total_sampled_scales += freq_sampled_scales
        else:
            entropy_loss = entropy_loss_func(attention_maps)
            loss = criterion(y, label) - entropy_loss

        loss_value = loss.item()
        losses.append(loss_value)

        y_prob = F.softmax(y, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        y_trues = np.concatenate([y_trues, label.cpu().numpy()])

        if opts.visualize:
            for b in range(patches.shape[0]):
                print("expectation prediction: ", y_probs[b])
                print("label prediction: ", label[0])
                if sampled_scales is not None:
                    print("sampled patch scales: ", sampled_scales[b])
                batch_patches = patches[b]
                patch_feature = patch_features[b]
                y_patch = model.classifier(patch_feature)
                y_patch_prop = F.softmax(y_patch, dim=1)
                # print("patch %d: "%b, y_patch)
                # print("probability %d: " % b, y_patch_prop)
                predicted = []
                for prob in y_patch_prop:
                    if prob[0] >= prob[1]:
                        predicted.append(0)
                    else:
                        predicted.append(1)
                # patchGrid(batch_patches, (3, 5))
                if type(attention_maps) is list:
                    batch_maps = [attention_map[b].cpu().numpy() for attention_map in attention_maps]
                    # for attention_map in batch_maps:
                    #     print(np.max(attention_map))
                    #     print(np.min(attention_map))
                    # batch_maps = [attention_maps[i][b] for i in range(len(model.scales))]
                else:
                    # batch_maps = [attention_maps[b] for i in range(len(model.scales))]
                    batch_maps = [attention_maps[b].cpu().numpy()]
                batch_imgs = [x_lows[i][b] for i in range(len(model.scales))]
                # mapGrid(batch_maps, batch_imgs, model.scales)
                patchGrid(batch_patches, batch_maps, batch_imgs, (3, 5), predicted)

    test_loss_epoch = np.round(np.mean(losses), 4)
    metrics = calc_cls_measures(y_probs, y_trues)
    print("Sampled scale frequencies: ", total_sampled_scales)
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
        # high res batch
        x_low, x_high, label = move_to([x_low, x_high, label], opts.device)

        optimizer.zero_grad()
        y, attention_map, patches, x_low_out = model(x_low, x_high)

        entropy_loss = entropy_loss_func(attention_map)

        loss = criterion(y, label) - entropy_loss
        loss.backward()
        # for p in model.parameters():
        #     print(p.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clipnorm)
        optimizer.step()

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

            optimizer.zero_grad()
            y, attention_map, patches, x_low_i_out = model(x_low_i, x_high_i)

            entropy_loss = entropy_loss_func(attention_map)

            loss = criterion(y, label) - entropy_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clipnorm)

            optimizer.step()

            loss_value = loss.item()

            losses[i].append(loss_value)

            y_prob = F.softmax(y, dim=1)
            y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
            y_trues = np.concatenate([y_trues, label.cpu().numpy()])

            metric = calc_cls_measures(y_probs, y_trues)
            metrics.append(metric)

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

    all_patches = []
    all_maps = []
    all_x_low = []
    all_sampled_ats = []
    for i, (x_low, x_high, label) in enumerate(tqdm(test_loader)):
        # high res batch
        x_low, x_high, label = move_to([x_low, x_high, label], opts.device)

        y, attention_map, patches, x_low_out, sampled_attention = model(x_low, x_high)
        if opts.visualize:
            all_patches.append(patches)
            all_maps.append(attention_map)
            all_x_low.append(x_low_out)
            all_sampled_ats.append(sampled_attention)
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

            y, attention_map, patches, x_low_i_out, sampled_attention = model(x_low_i, x_high_i)

            if opts.visualize:
                all_patches.append(patches)
                all_maps.append(attention_map)
                all_x_low.append(x_low_i_out)
                all_sampled_ats.append(sampled_attention)
            entropy_loss = entropy_loss_func(attention_map)

            loss = criterion(y, label) - entropy_loss

            loss_value = loss.item()

            losses[i].append(loss_value)

            y_prob = F.softmax(y, dim=1)
            y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
            y_trues = np.concatenate([y_trues, label.cpu().numpy()])

            metric = calc_cls_measures(y_probs, y_trues)
            metrics.append(metric)

        if opts.visualize:
            all_patches_tensor = torch.cat(all_patches, dim=1)
            # all_maps_tensor = torch.stack(all_maps, dim=1)
            for b in range(patches.shape[0]):
                batch_patches = all_patches_tensor[b]
                batch_maps = [attention_map[b].cpu().numpy() for attention_map in all_maps]
                for ats in batch_maps:
                    print(ats)
                    # print(torch.min())
                batch_imgs = [x_low_i[b] for x_low_i in all_x_low]
                batch_sampled_ats = [sampled_attetion[b].cpu().numpy() for sampled_attetion in all_sampled_ats]
                print(batch_sampled_ats)
                patchGrid(batch_patches, batch_maps, batch_imgs, (3, 5))
                # mapGrid(batch_maps, batch_imgs, opts.scales)

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