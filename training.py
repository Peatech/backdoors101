
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

# Training Function (train)    
#######################################################################################################################
def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True):
    criterion = hlpr.task.criterion
    model.train() # Puts the model into training mode
    """
    tqdm: A progress bar library to visualize the loop progress in the console.
    ** enumerate(train_loader): Provides both the batch index i and the batch data data.
    ** batch = hlpr.task.get_batch(i, data): Retrieves and processes the batch data.
    ** model.zero_grad() : Clears the gradients of all optimized tensors to prevent accumulation from previous iterations.
    ** loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack) : Calculates the loss for the current batch.
    ** hlpr.attack.compute_blind_loss: Handles Attack Logic: Incorporates backdoor attack mechanisms if attack is True.
    ** hlpr.report_training_losses_scales(i, epoch) : Logs training losses and scaling factors for analysis and debugging.
    ** loss.backward() : Computes the gradient of the loss with respect to the model parameters
    ** optimizer.step() : Updates the model parameters based on the computed gradients.
    ** hlpr.report_training_losses_scales(i, epoch) : Logs training losses and scaling factors for analysis and debugging.
    ** if i == hlpr.params.max_batch_id break:  Allows for early stopping after a certain number of batches, as defined by max_batch_id in the parameters.
            Use Case: Useful for debugging or when you don't want to process the entire dataset in each epoch.
    """
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

# Testing Function (test)
#######################################################################################################################
# The test function evaluates the model's performance on the test dataset.
"""
Key Points
Model Evaluation Mode: model.eval() sets the model to evaluation mode, which affects layers like dropout and batch normalization.
No Gradient Computation: with torch.no_grad() disables gradient calculation, reducing memory consumption and computational overhead during evaluation.
Backdoor Testing: If backdoor=True, the test data is modified to include backdoor triggers.
Metric Accumulation: hlpr.task.accumulate_metrics collects performance metrics for reporting.
Metric Reporting: hlpr.task.report_metrics logs the metrics to the console and TensorBoard.
"""
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


# Standard Training Loop (run function)  
#######################################################################################################################

def run(hlpr):
    # Initial Testing Before Training. Evaluates the model's performance on the test dataset before any training has occurred. This provides a baseline accuracy.
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


# Federated Learning Execution (fl_run function)
#######################################################################################################################

def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch) # Executes a single round of federated learning
        metric = test(hlpr, epoch, backdoor=False) # Evaluates the updated global model on clean and backdoor data.
        test(hlpr, epoch, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, metric) # Saves the global model checkpoint.

def run_fl_round(hlpr, epoch):
    global_model = hlpr.task.model        # The shared model that is updated each round.
    local_model = hlpr.task.local_model   # A copy of the global model used for local training on each client.

    round_participants = hlpr.task.sample_users_for_round(epoch)  # Selects a subset of clients (users) to participate in the current round.
    weight_accumulator = hlpr.task.get_empty_accumulator()        #  An empty data structure used to accumulate local model updates from participants.

    # Loops over each selected participant to perform local training and collect updates.
    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)        # Ensures that each participant starts with the latest global model parameters.
        optimizer = hlpr.task.make_optimizer(local_model)       # Sets up an optimizer for the local model on the participant's device.
        
        # Trains the local model on the participant's data for a specified number of local epochs.
        # If the user is compromised, the train function is called with attack=True, simulating backdoor attacks during local training.
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised:                        
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=True)
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False)
                
        # Calculates the difference between the updated local model and the global model. The fuction get_fl_update is contained in fl_task.py in tasks/fl folder 
        local_update = hlpr.task.get_fl_update(local_model, global_model)
       
        # Modifies the local update from compromised users to amplify the impact of the attack.
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)     # Aggregates the local update to the weight accumulator.

    hlpr.task.update_global_model(weight_accumulator, global_model)       # Updates the global model by applying the aggregated updates from all participants.


############################################################################################################################
# The main code execution starts here by defining the argparse 
############################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit',
                        default=get_current_git_hash())

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    
    """
    Adds additional information to params, including the current timestamp, commit identifier, and experiment name.
    """
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name

    helper = Helper(params)
    # After loading the configuration, the script logs all the parameters in the params dictionary using create_table() and prints it out in human readable form
    logger.warning(create_table(params))

############################################################################################################################
# Running the Training Loop. The code checks to see if fl is set to true so that it runs federated Learning, else it runs centralized learning
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
