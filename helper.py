"""
The Helper class manages configurations, training, attack injection, logging, and 
model saving. It sets up the environment, establishes reproducibility, and structures 
the training process, especially for federated learning and backdoor attacks. 
This design allows for easy modifications by adjusting the parameters in the configuration
file, enabling flexible experimentation with different tasks and attack types.
"""


import importlib                             # Allows dynamic importing of modules (loading modules during runtime).
import logging                               # Used for logging information, warnings, and errors.
import os                                    # Provides functions for interacting with the operating system (e.g., file paths).
import random                                # For generating random numbers.
from collections import defaultdict          # A dictionary subclass that calls a factory function to supply missing values.
from copy import deepcopy                    # Creates a deep copy of objects.
from shutil import copyfile                  # Copies files.
from typing import Union                     # Used for type hints, indicating that a variable can be one of multiple types.

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter             # For logging information to TensorBoard, a visualization tool.

from attack import Attack                                     # The module containing the Attack class.
from synthesizers.synthesizer import Synthesizer              # Base class for synthesizers used to create backdoor data.
from tasks.fl.fl_task import FederatedLearningTask            # Contains the FederatedLearningTask class for federated learning tasks.
from tasks.task import Task                                   # Contains the base Task class for standard tasks.
from utils.parameters import Params                           # Contains the Params dataclass that holds configuration parameters.
from utils.utils import create_logger, create_table

logger = logging.getLogger('logger')

"""
The Params class in parameters.py defines all configuration options for training and attack settings.
The Params class is a dataclass that serves as a container for all configuration settings used in the training and attack 
simulation. A dataclass in Python automatically generates an initializer (__init__) based on the defined attributes, 
making it convenient for organizing and managing configurations.
Each attribute in Params corresponds to a specific setting that can be loaded from a YAML file, as shown in 
your mnist_params.yaml, or manually set in code. This class provides default values for many attributes, which means that 
even if a parameter isn’t explicitly defined in the YAML file, it will take on its default value as specified in Params.
"""

# Defining the Helper Class
#######################################################################################################################
class Helper:
    """
    An instance of Params,task, synthesizer,and attack which holds all the configuration parameters is first created.
    Assigning None serves as a placeholder to indicate that the attribute will be assigned a 
    proper value later, typically during initialization in the __init__ method.
    """
    params: Params = None 
    task: Union[Task, FederatedLearningTask] = None
    synthesizer: Synthesizer = None
    attack: Attack = None
    tb_writer: SummaryWriter = None

    def __init__(self, params):
        """ 
        Params(**params) Converts the dictionary params into an instance of the Params dataclass.
        The ** operator unpacks the dictionary, passing its contents as keyword arguments to the Params constructor.
        *****************************
        """
        self.params = Params(**params) 

        """ 
        Initialization of Time Tracking Dictionaries. These lists will be used to record the 
        duration of various operations during training, such as forward pass, backward pass, etc.
        *****************************
        """ 
        self.times = {'backward': list(), 'forward': list(), 'step': list(),
                      'scales': list(), 'total': list(), 'poison': list()}

        """ 
        Setting Random Seed for Reproducibility: Ensures that results are reproducible by setting 
        the random seed if specified in the parameters.
        *****************************
        """
        if self.params.random_seed is not None:
            self.fix_random(self.params.random_seed) # Calls a method to set the random seed.

        # Creating Necessary Folders
        self.make_folders() # Sets up directories for logging and saving models.
       
        # Initializing the Task: Dynamically loads and initializes the task specified in the parameters (e.g., MNIST classification).
        self.make_task()

        # Initializing the Synthesizer: Sets up the synthesizer used for creating backdoor data based on the specified method.
        self.make_synthesizer()
        
        # Initializing the Attack: Creates an Attack instance that will manage attack logic during training.
        self.attack = Attack(self.params, self.synthesizer)

        # Checks if 'neural_cleanse' is among the specified loss tasks and sets a flag accordingly
        if 'neural_cleanse' in self.params.loss_tasks:
            self.nc = True
        # if 'spectral_evasion' in self.params.loss_tasks:
        #     self.attack.fixed_model = deepcopy(self.task.model)

        # Keeps track of the best validation accuracy achieved during training to save the best model.
        self.best_acc = float(0)


# Dynamically Loading the Task    
#######################################################################################################################
    """
    Purpose: Sets up the machine learning task (federated learning or standard training) specified 
    in the configuration. Process:
    1. Dynamically imports the required task module using importlib.
    2. The script determines the correct task type (federated or not) based on the fl parameter in params.
    3. After importing, it initializes the task with self.params.
    """
    def make_task(self):
        # Preparing Module and Class Names: Formats the task name to match module and class naming conventions.
        name_lower = self.params.task.lower()       # Converts the task name to lowercase (e.g., 'MNIST' becomes 'mnist').
        name_cap = self.params.task                 #  Keeps the original task name (e.g., 'MNIST').
        
        # Decides the module path based on whether federated learning (fl) is enabled.
        if self.params.fl:
            module_name = f'tasks.fl.{name_lower}_task'
            path = f'tasks/fl/{name_lower}_task.py'
        else:
            module_name = f'tasks.{name_lower}_task'
            path = f'tasks/{name_lower}_task.py'
        
        # Imports the module that contains the task class. Gets the task class from the module using the formatted class name. Provides a clear error message if the module or class is not found.
        try:
            task_module = importlib.import_module(module_name)
            task_class = getattr(task_module, f'{name_cap}Task')
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(f'Your task: {self.params.task} should '
                                      f'be defined as a class '
                                      f'{name_cap}'
                                      f'Task in {path}')
        self.task = task_class(self.params)      # Creates an instance of the task class, passing in the parameters.


# Dynamically Loading the Synthesizer   
#######################################################################################################################
    """
    Purpose: Sets up the backdoor synthesizer to manage malicious input generation.
    Process:
    1. Dynamically loads the specific synthesizer specified in params.
    2. Initializes the synthesizer class, passing it the task object (either standard or federated learning)
    """
    def make_synthesizer(self):
        # Formats the synthesizer name to match module and class naming conventions.
        name_lower = self.params.synthesizer.lower()
        name_cap = self.params.synthesizer
        module_name = f'synthesizers.{name_lower}_synthesizer'

        # Imports the module containing the synthesizer class.
        try:
            synthesizer_module = importlib.import_module(module_name)
            task_class = getattr(synthesizer_module, f'{name_cap}Synthesizer')  # Gets the synthesizer class from the module.
       
        # Provides an informative error message if the module or class is missing.
        except (ModuleNotFoundError, AttributeError):
            raise ModuleNotFoundError(
                f'The synthesizer: {self.params.synthesizer}'
                f' should be defined as a class '
                f'{name_cap}Synthesizer in '
                f'synthesizers/{name_lower}_synthesizer.py')
        
        # Creates an instance of the synthesizer class, passing in the task instance.
        self.synthesizer = task_class(self.task)

    
# Creating Necessary Folders
#######################################################################################################################
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
        log = create_logger()    # Initializes a logger to handle logging messages.
        if self.params.log:     # Only proceeds to create folders and set up logging if logging is enabled in the parameters.
            try:                # Attempts to create a directory specified by self.params.folder_path. If the folder already exists, it logs a message instead of crashing.
                os.mkdir(self.params.folder_path)
            except FileExistsError:
                log.info('Folder already exists')

            # Appends a new entry to runs.html with links to the GitHub commit and the experiment folder.
            with open('saved_models/runs.html', 'a') as f:        
                f.writelines([f'<div><a href="https://github.com/ebagdasa/'
                              f'backdoors/tree/{self.params.commit}">GitHub'
                              f'</a>, <span> <a href="http://gpu/'
                              f'{self.params.folder_path}">{self.params.name}_'
                              f'{self.params.current_time}</a></div>'])

            # Adds a file handler to the logger to write log messages to a file named log.txt in the experiment folder.
            fh = logging.FileHandler(               
                filename=f'{self.params.folder_path}/log.txt')
            formatter = logging.Formatter('%(asctime)s - %(name)s '
                                          '- %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            log.addHandler(fh)

            # Logs the path where logs are stored and provides a link to the GitHub commit for traceability.
            log.warning(f'Logging to: {self.params.folder_path}')   
            log.error(
                f'LINK: <a href="https://github.com/ebagdasa/backdoors/tree/'
                f'{self.params.commit}">https://github.com/ebagdasa/backdoors'
                f'/tree/{self.params.commit}</a>')

            # Saves the experiment's parameters to a file for future reference.
            with open(f'{self.params.folder_path}/params.yaml.txt', 'w') as f:
                yaml.dump(self.params, f)
      
        # Setting Up TensorBoard Logging
        if self.params.tb:
            wr = SummaryWriter(log_dir=f'runs/{self.params.name}')
            self.tb_writer = wr
            params_dict = self.params.to_dict()
            table = create_table(params_dict)
            self.tb_writer.add_text('Model Params', table)


# Model Saving Methods
#######################################################################################################################
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

# Setting Random Seed for Reproducibility
#######################################################################################################################
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
        random.seed(seed) # Sets the seed for Python's built-in random module.
        torch.manual_seed(seed) # torch.manual_seed(seed): 
        torch.cuda.manual_seed_all(seed) # Sets the seed for all CUDA devices (GPUs).
        cudnn.deterministic = False
        cudnn.enabled = True
        cudnn.benchmark = True
        np.random.seed(seed) # Sets the seed for NumPy's RNG.

        return True
