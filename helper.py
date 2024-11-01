"""
The Helper class manages configurations, training, attack injection, logging, and 
model saving. It sets up the environment, establishes reproducibility, and structures 
the training process, especially for federated learning and backdoor attacks. 
This design allows for easy modifications by adjusting the parameters in the configuration
file, enabling flexible experimentation with different tasks and attack types.
"""


import importlib
import logging
import os
import random
from collections import defaultdict
from copy import deepcopy
from shutil import copyfile
from typing import Union

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from attack import Attack
from synthesizers.synthesizer import Synthesizer
from tasks.fl.fl_task import FederatedLearningTask
from tasks.task import Task
from utils.parameters import Params
from utils.utils import create_logger, create_table

logger = logging.getLogger('logger')


class Helper:
    params: Params = None
    task: Union[Task, FederatedLearningTask] = None
    synthesizer: Synthesizer = None
    attack: Attack = None
    tb_writer: SummaryWriter = None

    def __init__(self, params):
        self.params = Params(**params)

        self.times = {'backward': list(), 'forward': list(), 'step': list(),
                      'scales': list(), 'total': list(), 'poison': list()}
        if self.params.random_seed is not None:
            self.fix_random(self.params.random_seed)

        self.make_folders()
        self.make_task()
        self.make_synthesizer()
        self.attack = Attack(self.params, self.synthesizer)

        if 'neural_cleanse' in self.params.loss_tasks:
            self.nc = True
        # if 'spectral_evasion' in self.params.loss_tasks:
        #     self.attack.fixed_model = deepcopy(self.task.model)

        self.best_acc = float(0)


    """
    Purpose: Sets up the machine learning task (federated learning or standard training) specified 
    in the configuration. Process:
    1. Dynamically imports the required task module using importlib.
    2. The script determines the correct task type (federated or not) based on the fl parameter in params.
    3. After importing, it initializes the task with self.params.
    """
    def make_task(self):
        name_lower = self.params.task.lower()
        name_cap = self.params.task
        if self.params.fl:
            module_name = f'tasks.fl.{name_lower}_task'
            path = f'tasks/fl/{name_lower}_task.py'
        else:
            module_name = f'tasks.{name_lower}_task'
            path = f'tasks/{name_lower}_task.py'
        try:
            task_module = importlib.import_module(module_name)
            task_class = getattr(task_module, f'{name_cap}Task')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(f'Your task: {self.params.task} should '
                                      f'be defined as a class '
                                      f'{name_cap}'
                                      f'Task in {path}')
        self.task = task_class(self.params)


    """
    Purpose: Sets up the backdoor synthesizer to manage malicious input generation.
    Process:
    1. Dynamically loads the specific synthesizer specified in params.
    2. Initializes the synthesizer class, passing it the task object (either standard or federated learning)
    """
    def make_synthesizer(self):
        name_lower = self.params.synthesizer.lower()
        name_cap = self.params.synthesizer
        module_name = f'synthesizers.{name_lower}_synthesizer'
        try:
            synthesizer_module = importlib.import_module(module_name)
            task_class = getattr(synthesizer_module, f'{name_cap}Synthesizer')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(
                f'The synthesizer: {self.params.synthesizer}'
                f' should be defined as a class '
                f'{name_cap}Synthesizer in '
                f'synthesizers/{name_lower}_synthesizer.py')
        self.synthesizer = task_class(self.task)



    """
    Purpose: Creates directories for logs and saves experiment information.
    Process:
    1. If params.log is enabled, it creates a folder specified by params.folder_path for saving logs and configurations.
    2. Logging Setup:
        Adds a logging file handler to save logs to a file.
        Writes a link to the current GitHub commit and timestamp in an HTML file.
        TensorBoard Writer:
    If params.tb (TensorBoard) is enabled, initializes SummaryWriter for visualizing model performance.
    """
    def make_folders(self):
        log = create_logger()
        if self.params.log:
            try:
                os.mkdir(self.params.folder_path)
            except FileExistsError:
                log.info('Folder already exists')

            with open('saved_models/runs.html', 'a') as f:
                f.writelines([f'<div><a href="https://github.com/ebagdasa/'
                              f'backdoors/tree/{self.params.commit}">GitHub'
                              f'</a>, <span> <a href="http://gpu/'
                              f'{self.params.folder_path}">{self.params.name}_'
                              f'{self.params.current_time}</a></div>'])

            fh = logging.FileHandler(
                filename=f'{self.params.folder_path}/log.txt')
            formatter = logging.Formatter('%(asctime)s - %(name)s '
                                          '- %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            log.addHandler(fh)

            log.warning(f'Logging to: {self.params.folder_path}')
            log.error(
                f'LINK: <a href="https://github.com/ebagdasa/backdoors/tree/'
                f'{self.params.commit}">https://github.com/ebagdasa/backdoors'
                f'/tree/{self.params.commit}</a>')

            with open(f'{self.params.folder_path}/params.yaml.txt', 'w') as f:
                yaml.dump(self.params, f)

        if self.params.tb:
            wr = SummaryWriter(log_dir=f'runs/{self.params.name}')
            self.tb_writer = wr
            params_dict = self.params.to_dict()
            table = create_table(params_dict)
            self.tb_writer.add_text('Model Params', table)


    """
    Purpose: Saves the model state at specified intervals or if it achieves a new best accuracy.
    Process:
    1. Creates a dictionary saved_dict to store model state and relevant parameters.
    2. Uses save_checkpoint to save this dictionary.
    3. Saves additional copies of the model on specific epochs if epoch is in params.save_on_epochs.
    If val_acc surpasses best_acc, saves the model as the best model and updates best_acc.
    """
    def save_model(self, model=None, epoch=0, val_acc=0):

        if self.params.save_model:
            logger.info(f"Saving model to {self.params.folder_path}.")
            model_name = '{0}/model_last.pt.tar'.format(self.params.folder_path)
            saved_dict = {'state_dict': model.state_dict(),
                          'epoch': epoch,
                          'lr': self.params.lr,
                          'params_dict': self.params.to_dict()}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params.save_on_epochs:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False,
                                     filename=f'{model_name}.epoch_{epoch}')
            if val_acc >= self.best_acc:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_acc = val_acc

    """
    Purpose: Saves the model checkpoint to a specified file and optionally marks it as the best model.
    Process:
    1. Saves the model state dictionary to a specified filename.
    2. If is_best is True, copies this checkpoint to a separate file for easy access to the best model.
    """
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params.save_model:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    """
    Ensures any TensorBoard data buffered in memory is saved to disk. 
    This helps prevent data loss if the program unexpectedly terminates.
    """
    def flush_writer(self):
        if self.tb_writer:
            self.tb_writer.flush()
            
    """
    Logs scalar values to TensorBoard.
    Process:
    Adds the scalar (e.g., loss, accuracy) with a tag name for easy identification in TensorBoard.
    """
    
    def plot(self, x, y, name):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag=name, scalar_value=y, global_step=x)
            self.flush_writer()
        else:
            return False

    """
     Logs training losses and scaling factors to TensorBoard and displays them in the console.
    Process:
    1. Uses logger to report losses and scales.
    2. Plots them to TensorBoard using plot() for visualization.
    3. Resets running_losses and running_scales for the next logging interval.
    """
    def report_training_losses_scales(self, batch_id, epoch):
        if not self.params.report_train_loss or \
                batch_id % self.params.log_interval != 0:
            return
        total_batches = len(self.task.train_loader)
        losses = [f'{x}: {np.mean(y):.2f}'
                  for x, y in self.params.running_losses.items()]
        scales = [f'{x}: {np.mean(y):.2f}'
                  for x, y in self.params.running_scales.items()]
        logger.info(
            f'Epoch: {epoch:3d}. '
            f'Batch: {batch_id:5d}/{total_batches}. '
            f' Losses: {losses}.'
            f' Scales: {scales}')
        for name, values in self.params.running_losses.items():
            self.plot(epoch * total_batches + batch_id, np.mean(values),
                      f'Train/Loss_{name}')
        for name, values in self.params.running_scales.items():
            self.plot(epoch * total_batches + batch_id, np.mean(values),
                      f'Train/Scale_{name}')

        self.params.running_losses = defaultdict(list)
        self.params.running_scales = defaultdict(list)


    """
    Purpose: Sets a fixed random seed to ensure results are reproducible.
    Process:
    1. Sets seeds for the random, torch, and numpy libraries.
    2. Configures torch.backends.cudnn for deterministic behavior.
    """
    @staticmethod
    def fix_random(seed=1):
        from torch.backends import cudnn

        logger.warning('Setting random_seed seed for reproducible results.')
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = False
        cudnn.enabled = True
        cudnn.benchmark = True
        np.random.seed(seed)

        return True
