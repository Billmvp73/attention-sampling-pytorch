import os
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


class AttentionSaverTrafficSigns:
    """Save the attention maps to monitor model evolution."""

    def __init__(self, output_directory, ats_model, training_set, opts):
        self.dir = output_directory
        os.makedirs(self.dir, exist_ok=True)
        self.ats_model = ats_model
        self.opts = opts

        idxs = training_set.strided(9)
        data = [training_set[i] for i in idxs]
        self.x_low = torch.stack([d[0] for d in data]).cpu()
        self.x_high = torch.stack([d[1] for d in data]).cpu()
        self.labels = torch.LongTensor([d[2] for d in data]).numpy()

        self.writer = SummaryWriter(os.path.join(self.dir, opts.run_name), flush_secs=5)
        self.on_train_begin()

    def on_train_begin(self):
        opts = self.opts
        with torch.no_grad():
            _, _, _, x_low = self.ats_model(self.x_low.to(opts.device), self.x_high.to(opts.device))
            x_low = x_low.cpu()
            image_list = [x for x in x_low]

        grid = torchvision.utils.make_grid(image_list, nrow=3, normalize=True, scale_each=True)

        self.writer.add_image('original_images', grid, global_step=0, dataformats='CHW')
        self.__call__(-1)

    def __call__(self, epoch, losses=None, metrics=None):
        opts = self.opts
        with torch.no_grad():
            _, att, _, x_low = self.ats_model(self.x_low.to(opts.device), self.x_high.to(opts.device))
            att = att.unsqueeze(1)
            att = F.interpolate(att, size=(x_low.shape[-2], x_low.shape[-1]))
            att = att.cpu()

        grid = torchvision.utils.make_grid(att, nrow=3, normalize=True, scale_each=True, pad_value=1.)
        self.writer.add_image('attention_map', grid, epoch, dataformats='CHW')

        if metrics is not None:
            train_metrics, test_metrics = metrics
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Test', test_metrics['accuracy'], epoch)

        if losses is not None:
            train_loss, test_loss = losses
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Test', test_loss, epoch)

    @staticmethod
    def imsave(filepath, x):
        if x.shape[-1] == 3:
            plt.imshow(x)
            plt.savefig(filepath)
        else:
            plt.imshow(x, cmap='viridis')
            plt.savefig(filepath)

    @staticmethod
    def reverse_transform(inp):
        """ Do a reverse transformation. inp should be a torch tensor of shape [3, H, W] """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = (inp * 255).astype(np.uint8)

        return inp

    @staticmethod
    def reverse_transform_torch(inp):
        """ Do a reverse transformation. inp should be a torch tensor of shape [3, H, W] """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = torch.from_numpy(inp).permute(2, 0, 1)

        return inp


class AttentionSaverMNIST:
    def __init__(self, output_directory, ats_model, dataset, opts):
        self.dir = output_directory
        os.makedirs(self.dir, exist_ok=True)
        self.ats_model = ats_model
        self.opts = opts

        idxs = [random.randrange(0, len(dataset)-1) for _ in range(9)]
        data = [dataset[i] for i in idxs]
        self.x_low = torch.stack([d[0] for d in data]).cpu()
        self.x_high = torch.stack([d[1] for d in data]).cpu()
        self.label = torch.LongTensor([d[2] for d in data]).numpy()

        self.writer = SummaryWriter(os.path.join(self.dir, opts.run_name), flush_secs=2)
        self.__call__(-1)

    def __call__(self, epoch, losses=None, metrics=None):
        opts = self.opts
        with torch.no_grad():
            _, att, patches, x_low = self.ats_model(self.x_low.to(opts.device), self.x_high.to(opts.device))
            att = att.unsqueeze(1)
            att = F.interpolate(att, size=(x_low.shape[-2], x_low.shape[-1]))
            att = att.cpu()

        grid = torchvision.utils.make_grid(att, nrow=3, normalize=True, scale_each=True, pad_value=1.)
        self.writer.add_image('attention_map', grid, epoch, dataformats='CHW')

        if metrics is not None:
            train_metrics, test_metrics = metrics
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Test', test_metrics['accuracy'], epoch)

        if losses is not None:
            train_loss, test_loss = losses
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Test', test_loss, epoch)


class AttentionSaverMultiBddDetection:
    """Save the attention maps to monitor model evolution."""

    def __init__(self, output_directory, ats_model, training_set, opts):
        self.dir = output_directory
        os.makedirs(self.dir, exist_ok=True)
        self.ats_model = ats_model
        self.opts = opts

        idxs = training_set.strided(9)
        data = [training_set[i] for i in idxs]
        x_low_array = [[] for i in range(len(opts.scales))]
        x_high_array = [[] for i in range(len(opts.scales))]
        for x_lows, x_highs, label in data:
            for i in range(len(opts.scales)):
                x_low_array[i].append(x_lows[i])
                x_high_array[i].append(x_highs[i])
        # self.x_low = [torch.stack([dd[i] for dd in d[0] for d in data]).cpu() for i in range(len(opts.scales))]
        self.x_lows = [torch.stack(x_low).cpu() for x_low in x_low_array]
        self.x_highs = [torch.stack(x_high).cpu() for x_high in x_high_array]
        self.labels = torch.LongTensor([d[2] for d in data]).numpy()

        self.writer = SummaryWriter(os.path.join(self.dir, opts.run_name), flush_secs=5)
        self.on_train_begin()

    def on_train_begin(self):
        opts = self.opts
        with torch.no_grad():
            lows = []
            highs = []
            for x_low, x_high in zip(self.x_lows, self.x_highs):
                lows.append(x_low.to(opts.device))
                highs.append(x_high.to(opts.device))
            _, _, _, x_low = self.ats_model(lows, highs)
            x_low = x_low.cpu()
            image_list = [x for x in x_low]

        grid = torchvision.utils.make_grid(image_list, nrow=3, normalize=True, scale_each=True)

        self.writer.add_image('original_images', grid, global_step=0, dataformats='CHW')
        self.__call__(-1)

    def __call__(self, epoch, losses=None, metrics=None):
        opts = self.opts
        with torch.no_grad():
            lows = []
            highs = []
            for x_low, x_high in zip(self.x_lows, self.x_highs):
                lows.append(x_low.to(opts.device))
                highs.append(x_high.to(opts.device))
            _, att, _, x_low = self.ats_model(lows, highs)
            att = att.unsqueeze(1)
            att = F.interpolate(att, size=(x_low.shape[-2], x_low.shape[-1]))
            att = att.cpu()

        grid = torchvision.utils.make_grid(att, nrow=3, normalize=True, scale_each=True, pad_value=1.)
        self.writer.add_image('attention_map', grid, epoch, dataformats='CHW')

        if metrics is not None:
            train_metrics, test_metrics = metrics
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Test', test_metrics['accuracy'], epoch)

        if losses is not None:
            train_loss, test_loss = losses
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Test', test_loss, epoch)

    @staticmethod
    def imsave(filepath, x):
        if x.shape[-1] == 3:
            plt.imshow(x)
            plt.savefig(filepath)
        else:
            plt.imshow(x, cmap='viridis')
            plt.savefig(filepath)

    @staticmethod
    def reverse_transform(inp):
        """ Do a reverse transformation. inp should be a torch tensor of shape [3, H, W] """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = (inp * 255).astype(np.uint8)

        return inp

    @staticmethod
    def reverse_transform_torch(inp):
        """ Do a reverse transformation. inp should be a torch tensor of shape [3, H, W] """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = torch.from_numpy(inp).permute(2, 0, 1)

        return inp

class AttentionSaverMultiParallelBddDetection:
    """Save the attention maps to monitor model evolution."""

    def __init__(self, output_directory, ats_model, training_set, opts):
        self.dir = output_directory
        os.makedirs(self.dir, exist_ok=True)
        self.ats_model = ats_model
        self.opts = opts

        idxs = training_set.strided(9)
        data = [training_set[i] for i in idxs]

        x_low_array = [[] for i in range(len(opts.scales))]
        x_high_array = [[] for i in range(len(opts.scales))]
        for x_lows, x_highs, label in data:
            for i in range(len(opts.scales)):
                x_low_array[i].append(x_lows[i])
                x_high_array[i].append(x_highs[i])
        # self.x_low = [torch.stack([dd[i] for dd in d[0] for d in data]).cpu() for i in range(len(opts.scales))]
        self.x_lows = [torch.stack(x_low).cpu() for x_low in x_low_array]
        self.x_highs = [torch.stack(x_high).cpu() for x_high in x_high_array]
        self.labels = torch.LongTensor([d[2] for d in data]).numpy()

        self.writer = SummaryWriter(os.path.join(self.dir, opts.run_name), flush_secs=5)
        self.on_train_begin()

    def on_train_begin(self):
        opts = self.opts
        with torch.no_grad():
            lows = []
            highs = []
            for x_low, x_high in zip(self.x_lows, self.x_highs):
                lows.append(x_low.to(opts.device))
                highs.append(x_high.to(opts.device))
            _, _, _, x_lows = self.ats_model(lows, highs)
            x_lows = [x_low.cpu() for x_low in x_lows]
            image_lists = [[x for x in x_low] for x_low in x_lows]
        for scale, image_list in zip(self.opts.scales, image_lists):
            grid = torchvision.utils.make_grid(image_list, nrow=3, normalize=True, scale_each=True)

            self.writer.add_image('original_images'+str(scale), grid, global_step=0, dataformats='CHW')
        self.__call__(-1)

    def __call__(self, epoch, losses=None, metrics=None):
        opts = self.opts
        with torch.no_grad():
            lows = []
            highs = []
            for x_low, x_high in zip(self.x_lows, self.x_highs):
                lows.append(x_low.to(opts.device))
                highs.append(x_high.to(opts.device))
            _, atts, _, x_lows = self.ats_model(lows, highs)
            atts = [att.unsqueeze(1) for att in atts]
            atts = [F.interpolate(att, size=(x_low.shape[-2], x_low.shape[-1])).cpu() for att, x_low in zip(atts, x_lows)]
            # att = att.cpu()
        for scale, att in zip(self.opts.scales, atts):
            grid = torchvision.utils.make_grid(att, nrow=3, normalize=True, scale_each=True, pad_value=1.)
            self.writer.add_image('attention_map'+str(scale), grid, epoch, dataformats='CHW')

        if metrics is not None:
            train_metrics, test_metrics = metrics
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Test', test_metrics['accuracy'], epoch)

        if losses is not None:
            train_loss, test_loss = losses
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Test', test_loss, epoch)

    @staticmethod
    def imsave(filepath, x):
        if x.shape[-1] == 3:
            plt.imshow(x)
            plt.savefig(filepath)
        else:
            plt.imshow(x, cmap='viridis')
            plt.savefig(filepath)

    @staticmethod
    def reverse_transform(inp):
        """ Do a reverse transformation. inp should be a torch tensor of shape [3, H, W] """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = (inp * 255).astype(np.uint8)

        return inp

    @staticmethod
    def reverse_transform_torch(inp):
        """ Do a reverse transformation. inp should be a torch tensor of shape [3, H, W] """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = torch.from_numpy(inp).permute(2, 0, 1)

        return inp

class AttentionSaverMultiBatchBddDetection:
    """Save the attention maps to monitor model evolution."""

    def __init__(self, output_directory, ats_model, training_set, opts):
        self.dir = output_directory
        os.makedirs(self.dir, exist_ok=True)
        self.ats_model = ats_model
        self.opts = opts

        idxs = training_set.strided(9)
        data = [training_set[i] for i in idxs]
        self.x_low = torch.stack([d[0] for d in data]).cpu()
        self.x_high = torch.stack([d[1] for d in data]).cpu()
        self.labels = torch.LongTensor([d[2] for d in data]).numpy()

        self.writer = SummaryWriter(os.path.join(self.dir, opts.run_name), flush_secs=5)
        self.on_train_begin()

    def on_train_begin(self):
        opts = self.opts
        with torch.no_grad():
            _, _, _, x_low = self.ats_model(self.x_low.to(opts.device), self.x_high.to(opts.device))
            x_low = x_low.cpu()
            image_list = [x for x in x_low]

        grid = torchvision.utils.make_grid(image_list, nrow=3, normalize=True, scale_each=True)

        self.writer.add_image('original_images', grid, global_step=0, dataformats='CHW')
        self.__call__(-1)

    def __call__(self, epoch, losses=None, metrics=None):
        opts = self.opts
        with torch.no_grad():
            _, att, _, x_low = self.ats_model(self.x_low.to(opts.device), self.x_high.to(opts.device))
            att = att.unsqueeze(1)
            att = F.interpolate(att, size=(x_low.shape[-2], x_low.shape[-1]))
            att = att.cpu()

        grid = torchvision.utils.make_grid(att, nrow=3, normalize=True, scale_each=True, pad_value=1.)
        self.writer.add_image('attention_map', grid, epoch, dataformats='CHW')

        if metrics is not None:
            train_metrics, test_metrics = metrics
            for i, s in enumerate(opts.scales):
                self.writer.add_scalar('%f-Accuracy/Train'%s, train_metrics[i]['accuracy'], epoch)
                self.writer.add_scalar('%f-Accuracy/Test'%s, test_metrics[i]['accuracy'], epoch)
            train_metrics_avg_scales = np.mean([train_metric['accuracy'] for train_metric in train_metrics])
            test_metrisc_avg_scales = np.mean([test_metric['accuracy'] for test_metric in test_metrics])
            self.writer.add_scalar('multiArgAccuracy/Train', train_metrics_avg_scales, epoch)
            self.writer.add_scalar('multiArgAccuracy/Test', test_metrisc_avg_scales, epoch)

        if losses is not None:
            train_loss, test_loss = losses
            for i, s in enumerate(opts.scales):
                self.writer.add_scalar('%f-Loss/Train'%s, train_loss[i], epoch)
                self.writer.add_scalar('%f-Loss/Test'%s, test_loss[i], epoch)
            train_loss_avg_scales = np.round(np.mean(train_loss), 4)
            test_loss_avg_scales = np.round(np.mean(test_loss), 4)
            self.write.add_scalar('multiAvgLoss/Train', train_loss_avg_scales, epoch)
            self.writer.add_scalar('multiAvgLoss/Test', test_loss_avg_scales, epoch)

    @staticmethod
    def imsave(filepath, x):
        if x.shape[-1] == 3:
            plt.imshow(x)
            plt.savefig(filepath)
        else:
            plt.imshow(x, cmap='viridis')
            plt.savefig(filepath)

    @staticmethod
    def reverse_transform(inp):
        """ Do a reverse transformation. inp should be a torch tensor of shape [3, H, W] """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = (inp * 255).astype(np.uint8)

        return inp

    @staticmethod
    def reverse_transform_torch(inp):
        """ Do a reverse transformation. inp should be a torch tensor of shape [3, H, W] """
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        inp = torch.from_numpy(inp).permute(2, 0, 1)

        return inp