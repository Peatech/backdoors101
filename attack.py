import logging
from typing import Dict

import torch
from copy import deepcopy
import numpy as np
from models.model import Model
from models.nc_model import NCModel
from synthesizers.synthesizer import Synthesizer
from losses.loss_functions import compute_all_losses_and_grads
from utils.min_norm_solvers import MGDASolver
from utils.parameters import Params

logger = logging.getLogger('logger')


class Attack:
    params: Params
    synthesizer: Synthesizer
    nc_model: Model
    nc_optim: torch.optim.Optimizer
    loss_hist = list()
    # fixed_model: Model

    def __init__(self, params, synthesizer):
        self.params = params
        self.synthesizer = synthesizer

        # NC hyper params
        if 'neural_cleanse' in self.params.loss_tasks:
            self.nc_model = NCModel(params.input_shape[1]).to(params.device)
            self.nc_optim = torch.optim.Adam(self.nc_model.parameters(), 0.01)

    def compute_blind_loss(self, model, criterion, batch, attack):
        """

        :param model:
        :param criterion:
        :param batch:
        :param attack: Do not attack at all. Ignore all the parameters
        :return:
        """
        # 🛑 Ensure batch is valid
        if batch is None or batch.inputs is None or batch.labels is None:
            print("⚠️ WARNING: Invalid batch detected in `compute_blind_loss`.")
            return torch.tensor(0.0, requires_grad=True)
            
        batch = batch.clip(self.params.clip_batch)                                   # Optionally clips the batch to specified bounds for stability.
        loss_tasks = self.params.loss_tasks.copy() if attack else ['normal']         # If attack is True, includes additional loss tasks specified in self.params.loss_tasks (e.g., backdoor loss).
        batch_back = self.synthesizer.make_backdoor_batch(batch, attack=attack)      # Generates a batch with backdoor triggers if attack is True.
        scale = dict()

        # 🔍 Debug: Check if batch processing is correct
        print(f"🛠️ DEBUG: Running compute_blind_loss() | Attack: {attack} | Loss Tasks: {loss_tasks}")


        if 'neural_cleanse' in loss_tasks:
            self.neural_cleanse_part1(model, batch, batch_back)

        if self.params.loss_threshold and (np.mean(self.loss_hist) >= self.params.loss_threshold
                                           or len(self.loss_hist) < 1000):
            loss_tasks = ['normal']

        if len(loss_tasks) == 1:
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=False
            )
        elif self.params.loss_balance == 'MGDA':
            # Calculates the loss values for each task.
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=True)

            # Detect gradient explosion
            for task, grad in grads.items():
                if grad is not None:
                    grad_norm = torch.norm(torch.cat([g.view(-1) for g in grad if g is not None]))
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"⚠️ WARNING: Gradient for {task} exploded! Norm: {grad_norm}")

            
            if len(loss_tasks) > 1:
                scale = MGDASolver.get_scales(grads, loss_values,
                                              self.params.mgda_normalize,
                                              loss_tasks)
        elif self.params.loss_balance == 'fixed':
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=False)

            for t in loss_tasks:
                scale[t] = self.params.fixed_scales[t]
        else:
            raise ValueError(f'Please choose between `MGDA` and `fixed`.')
        # Determines scaling factors for each loss component.
        if len(loss_tasks) == 1:
            scale = {loss_tasks[0]: 1.0}

        # 🛠️ Debugging: Check for NaNs in loss values
        for task, loss_val in loss_values.items():
            if torch.isnan(loss_val).any():
                print(f"❌ ERROR: `NaN` detected in loss for task {task}. Loss Value: {loss_val}")
            if torch.isinf(loss_val).any():
                print(f"❌ ERROR: `Inf` detected in loss for task {task}. Loss Value: {loss_val}")


        # Append to loss history
        self.loss_hist.append(loss_values['normal'].item())
        self.loss_hist = self.loss_hist[-1000:]
        blind_loss = self.scale_losses(loss_tasks, loss_values, scale)    #Aggregates the scaled losses into a single loss value for optimization.

        if torch.isnan(blind_loss):
            print(f"❌ ERROR: Final computed loss is NaN! Loss Values: {loss_values}, Scales: {scale}")
    
        return blind_loss
    # Keeps track of individual and total losses for logging.
    def scale_losses(self, loss_tasks, loss_values, scale):
        blind_loss = 0
        for it, t in enumerate(loss_tasks):
            self.params.running_losses[t].append(loss_values[t].item())
            self.params.running_scales[t].append(scale[t])
            if it == 0:
                blind_loss = scale[t] * loss_values[t]
            else:
                blind_loss += scale[t] * loss_values[t]
        self.params.running_losses['total'].append(blind_loss.item())
        return blind_loss

    def neural_cleanse_part1(self, model, batch, batch_back):
        self.nc_model.zero_grad()
        model.zero_grad()

        self.nc_model.switch_grads(True)
        model.switch_grads(False)
        output = model(self.nc_model(batch.inputs))
        nc_tasks = ['neural_cleanse_part1', 'mask_norm']

        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        loss_values, grads = compute_all_losses_and_grads(nc_tasks,
                                                          self, model,
                                                          criterion, batch,
                                                          batch_back,
                                                          compute_grad=False
                                                          )
        # Using NC paper params
        logger.info(loss_values)
        loss = 0.999 * loss_values['neural_cleanse_part1'] + 0.001 * loss_values['mask_norm']
        loss.backward()
        self.nc_optim.step()

        self.nc_model.switch_grads(False)
        model.switch_grads(True)


    def fl_scale_update(self, local_update: Dict[str, torch.Tensor]):
        for name, value in local_update.items():
            value.mul_(self.params.fl_weight_scale)
