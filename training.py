import argparse
import shutil
from datetime import datetime
from defences.cka import FedAvgCKA
import yaml
from prompt_toolkit import prompt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import random
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
    # Build our CKA helper in one line—no Task changes needed
    cka_helper = FedAvgCKA(
        model_template = hlpr.task.model,
        root_dataset   = hlpr.task.train_loader.dataset,
        ref_size       = 32,              # size of your reference set
        layer     = 'layer4',           # penultimate layer
        device         = hlpr.params.device,
        discard_ratio  = 0.5
    )

    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch, cka_helper)
        metric = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, metric)

def run_fl_round(hlpr, epoch, cka_helper):
    global_model = hlpr.task.model
    local_model  = hlpr.task.local_model
    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    locals_w, locals_uid = [], []

    # ─── Per‐client local training ───────────────────────────
    for user in tqdm(round_participants):
        # copy global → local
        hlpr.task.copy_params(global_model, local_model)

        # create optimizer for local_model
        optimizer = hlpr.task.make_optimizer(local_model)

        # run local epochs
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised:
                train(hlpr,
                      local_epoch,
                      local_model,
                      optimizer,
                      user.train_loader,
                      attack=True)
            else:
                train(hlpr,
                      local_epoch,
                      local_model,
                      optimizer,
                      user.train_loader,
                      attack=False)

        # get the update (weights or delta)
        local_update = hlpr.task.get_fl_update(local_model, global_model)

        # scale it if this user is adversarial
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)

        # collect for defence
        locals_w.append(local_update)
        locals_uid.append(user.user_id)
    # ─────────────────────────────────────────────────────────

    # ─── CKA defence & heat-map (after collecting all updates) ─
    keep_idx, _, _, sim = cka_helper.filter_and_aggregate(locals_w)

    import seaborn as sns, matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    sns.heatmap(sim,
                vmin=0, vmax=1,
                cmap='viridis',
                xticklabels=locals_uid,
                yticklabels=locals_uid,
                square=True)
    plt.title(f'CKA similarities – round {epoch}')
    plt.show()

    # filter out the low-similarity updates
    locals_w = [locals_w[i] for i in keep_idx]
    # ─────────────────────────────────────────────────────────

    # ─── Aggregate survivors exactly as before ───────────────
    for w in locals_w:
        hlpr.task.accumulate_weights(weight_accumulator, w)

    # update the global model in place
    hlpr.task.update_global_model(weight_accumulator, global_model)
    # ─────────────────────────────────────────────────────────



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit',
                        default=get_current_git_hash())

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name

    helper = Helper(params)
    logger.warning(create_table(params))

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
