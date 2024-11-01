"""
This training.py script is designed to perform training, testing, and federated learning (FL) 
for a machine learning model, potentially under adversarial conditions (e.g., backdoor attacks). 
Key Elements:
Helper: A central class that orchestrates various tasks like model training, logging, and handling attacks.
Backdoor Attack Simulation: The attack argument in train() and backdoor argument in test() allow for simulating adversarial conditions (e.g., backdoor attacks) on the dataset.
Federated Learning: The code has a dedicated flow for FL, where local models are trained on user data and their updates are aggregated to form a global model.
Workflow:
Setup: Load configuration, initialize Helper, create directories for logs.
Training: Depending on whether federated learning is enabled, either standard training (run) or FL (fl_run) is performed.
Testing: After each training epoch, the model is evaluated on the test data, potentially simulating attacks.
Logging and Saving: The results are logged to TensorBoard and saved periodically. If interrupted, the user is given a chance to delete the generated files.
"""

import argparse
import shutil
from datetime import datetime

import yaml
from prompt_toolkit import prompt
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from utils.utils import *

logger = logging.getLogger('logger')


def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True):
    criterion = hlpr.task.criterion
    model.train()

    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack)
        loss.backward()
        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break

    return


def test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()

    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric


def run(hlpr):
    acc = test(hlpr, 0, backdoor=False)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        acc = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)
        hlpr.save_model(hlpr.task.model, epoch, acc)
        if hlpr.task.scheduler is not None:
            hlpr.task.scheduler.step(epoch)

def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, metric)


def run_fl_round(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=True)
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)

    hlpr.task.update_global_model(weight_accumulator, global_model)


############################################################################################################################
# The main code execution starts here by defining the argparse 
############################################################################################################################
"""
The argparse library is used to handle command-line arguments.
--params specifies the YAML file with training parameters.
--name is the dataset or experiment name.
--commit is a unique identifier (here it defaults to none).

The purpose of this code block is to set up and handle command-line arguments for training.py. Each line is essential for configuring and customizing the training run without needing to modify the code itself. Here’s a step-by-step breakdown of what each line does and why it’s important:

Line 1: parser = argparse.ArgumentParser(description='Backdoors')
Purpose: This line initializes an ArgumentParser object, parser, from Python's argparse library.
Function: ArgumentParser allows the program to accept parameters from the command line, which lets users control the script's behavior by passing specific arguments when they run the program.
Description: The string 'Backdoors' is provided as a description. This appears in the help message when a user runs the script with --help. It helps users understand the purpose of the script.
Line 2-4: parser.add_argument(...)
Each add_argument() line defines a command-line argument that the script can accept. Here’s what each does:

parser.add_argument('--params', dest='params', default='utils/params.yaml')

Argument: --params
Users can specify a configuration file using --params, which points to a .yaml file that stores all training parameters.
Destination: dest='params'
This argument will be accessible in the code as args.params, allowing the script to load and use the specified configuration file.
Default Value: default='utils/params.yaml'
If a user does not specify a --params argument, the script will default to utils/params.yaml.
Essence: This argument gives users flexibility to specify different configurations, making the script adaptable for various experiments without hardcoding parameters.
parser.add_argument('--name', dest='name', required=True)

Argument: --name
Users specify a name for the experiment with --name.
Destination: dest='name'
This argument is stored as args.name, allowing the script to access the name later in the code.
Required: required=True
This argument must be provided; otherwise, the script will raise an error and won’t run.
Essence: --name provides a unique identifier for the experiment, which is useful for logging, model saving, and tracking purposes. In this code, it is especially important since the experiment's results may be logged in multiple places.
parser.add_argument('--commit', dest='commit', default=get_current_git_hash())

Argument: --commit
Users can specify a unique commit identifier, which is particularly useful for version control.
Destination: dest='commit'
This argument is stored as args.commit.
Default Value: default=get_current_git_hash()
If a user doesn’t specify this argument, it defaults to the result of get_current_git_hash().
get_current_git_hash() is likely a function that retrieves the latest Git commit hash for the project, allowing users to record the exact version of the code used in an experiment.
Essence: By logging the Git commit hash, the script provides traceability, enabling users to reproduce results with the same code version, which is essential for experiment tracking.
Line 5: args = parser.parse_args()
Purpose: parse_args() processes the command-line arguments and returns them as an object, args, where each argument is accessible by its destination name (e.g., args.params, args.name, and args.commit).
Essence: This line captures the command-line arguments into the args object, making the argument values accessible throughout the rest of the script.

"""
############################################################################################################################
# CODE
############################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit',
                        default=get_current_git_hash())

    args = parser.parse_args()

############################################################################################################################
# After defining the argparse, the yaml configuration file is opened and the simulation configuration is loaded to params
############################################################################################################################
    """
            with open(args.params) as f:
    Purpose: Opens the YAML file specified by args.params (e.g., configs/mnist_params.yaml).
    Details:
    args.params holds the file path provided by the user through the command-line argument --params.
    Using with open(...) as f: ensures that the file is automatically closed after the block of code completes, which is 
    good practice to prevent resource leaks.
            params = yaml.load(f, Loader=yaml.FullLoader)
    Purpose: Reads the contents of the YAML file and converts it into a Python dictionary called params.
    Details:
    yaml.load(f, Loader=yaml.FullLoader) reads the YAML file content from f and loads it into the params variable.
            Why Loader=yaml.FullLoader:
    The FullLoader is a safer option compared to the default loader and ensures that only standard YAML types are loaded 
    (e.g., dictionaries, lists, strings, numbers), which mitigates security risks.
    The result is a dictionary structure (params) where each key-value pair corresponds to entries in the YAML file.
    The YAML file specified by --params is loaded into the params dictionary, which will configure the entire 
    training, testing, and attack procedures.
    The result is a dictionary structure (params) where each key-value pair corresponds to entries in the YAML file.
    For example, if configs/mnist_params.yaml contains:
        task: MNIST
        batch_size: 64
        lr: 0.01
        epochs: 10
    Then params would be:
        {
            "task": "MNIST",
            "batch_size": 64,
            "lr": 0.01,
            "epochs": 10
        }
    """
############################################################################################################################
# CODE 
############################################################################################################################

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    
    """
    Adds additional information to params, including the current timestamp, commit identifier, and experiment name.
    """
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name

############################################################################################################################
# A helper class instance is created for managing data data loading, logging, attack simulation, model saving, and federated learning tasks
############################################################################################################################

    """
    Helper Initialization: Helper is a central class that manages data loading, logging, attack simulation, model 
    saving, and federated learning tasks. helper is initialized with params, which configures all the settings for 
    the training session. When Helper is initialized, it stores params as self.params, making it accessible as helper.params.
    Log Parameter Table: This line logs a table of the parameters, helping track the configuration for this specific 
    run. 
    """
    helper = Helper(params)
    logger.warning(create_table(params))
############################################################################################################################
# The code checks to see if fl is set to true so that it runs federated Learning, else it runs centralized learning
############################################################################################################################

    """
    If fl is set to True in the YAML file or programmatically (using the Params dataclass contained in parameters.py, the federated learning workflow (fl_run(helper)) is initiated. 
    If it’s False, the code runs standard training with run(helper).
    """
    try:
        if helper.params.fl:
            fl_run(helper)
        else:
            run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
