import numpy as np


class ScheduledOptim(object):


    def __init__(self, optimizer, d_model, n_warmup_steps,data_path,
                 lr_schedule='legacy', lr=1e-3, total_epochs=50, warmup_epochs=5,
                 min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.data_path=data_path


        self.lr_schedule = lr_schedule
        self.lr = lr
        self.total_epochs = max(int(total_epochs), 1)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.min_lr_ratio = min_lr_ratio

    def step(self):

        self.optimizer.step()

    def zero_grad(self):

        self.optimizer.zero_grad()

    def update_learning_rate(self,epoch):

        if self.lr_schedule == 'legacy':
            self._update_learning_rate_legacy(epoch)
            return
        new_lr = self._controlled_learning_rate(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _controlled_learning_rate(self, epoch):


        if self.lr_schedule == 'const':
            return self.lr
        if self.lr_schedule == 'inv_sqrt':


            self.n_current_steps += 1
            return self.lr * np.power(self.d_model, -0.5) * np.min([
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])


        warmup = self.warmup_epochs
        if warmup > 0 and epoch < warmup:
            return self.lr * float(epoch + 1) / float(warmup)
        decay_epochs = max(self.total_epochs - warmup, 1)
        progress = min(max(epoch - warmup, 0), decay_epochs) / float(decay_epochs)
        if self.lr_schedule == 'cosine':
            factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        else:
            factor = 1.0 - progress
        return self.lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * factor)

    def _update_learning_rate_legacy(self,epoch):


        if self.data_path.find("twitter") != -1:
            self.n_current_steps += 1
            new_lr = np.power(self.d_model, -0.5) * np.min([
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr


        if self.data_path.find("memetracker") != -1:

            self.n_current_steps += 1
            new_lr = np.power(self.d_model, -0.5) * np.min([
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
