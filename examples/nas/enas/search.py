# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from argparse import ArgumentParser
import json
import torch
import torch.nn as nn

import datasets
from macro import GeneralNetwork
from micro import MicroNetwork
from nni.algorithms.nas.pytorch import enas
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint, ModelCheckpoint,
                                       LRSchedulerCallback)
from utils import accuracy, reward_accuracy

logger = logging.getLogger('nni')


if __name__ == "__main__":
    parser = ArgumentParser("enas")
    # parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "custom_classification"])
    # parser.add_argument("--search-for", choices=["macro", "micro"], default="macro")
    # parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (default: macro 310, micro 150)")
    parser.add_argument("--visualization", default=True, action="store_true")
    parser.add_argument("--train-data-dir", default="/home/savan/Documents/train_data", help="train dataset for classification")
    parser.add_argument("--valid-data-dir", default="/home/savan/Documents/test_data", help="validation dataset for classification")
    parser.add_argument("--config", default="batch-size=128 \n search-for=macro \n epochs=30")
    args = parser.parse_args()

    extras = args.config.split("\n")
    print("nas extras", extras)
    extras_processed = [i.split("#")[0].replace(" ","") for i in extras if i]
    print("nas extra processed", extras_processed)
    config = {i.split('=')[0]:i.split('=')[1] for i in extras_processed}
    print("nas config", config)
    config.update(vars(args))
    args = config

    dataset_train, dataset_valid = datasets.get_dataset(args['dataset'], train_dir=args['train_data_dir'], valid_data=args['valid_data_dir'])
    if args['search_for'] == "macro":
        model = GeneralNetwork(num_classes=int(args['num_classes']))
        num_epochs = int(args['epochs']) or 310
        mutator = None
    elif args['search_for'] == "micro":
        model = MicroNetwork(num_layers=6, out_channels=20, num_nodes=5, dropout_rate=0.1, num_classes=int(args['num_classes']), use_aux_heads=True)
        num_epochs = int(args['epochs']) or 150
        mutator = enas.EnasMutator(model, tanh_constant=1.1, cell_exit_extra_step=True)
    else:
        raise AssertionError

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=1.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    trainer = enas.EnasTrainer(model,
                               loss=criterion,
                               metrics=accuracy,
                               reward_function=reward_accuracy,
                               optimizer=optimizer,
                               callbacks=[LRSchedulerCallback(lr_scheduler), ArchitectureCheckpoint("/mnt/output"), ModelCheckpoint("/mnt/output")],
                               batch_size=int(args['batch_size']),
                               num_epochs=num_epochs,
                               dataset_train=dataset_train,
                               dataset_valid=dataset_valid,
                               log_frequency=args['log_frequency'],
                               mutator=mutator)
    if args['visualization']:
        trainer.enable_visualization()
    trainer.train()
    metrics = [{'name':'accuracy', 'value':trainer.val_model_summary['acc1'].avg}, {'name':'loss', 'value':trainer.val_model_summary['loss'].avg}]
    with open('/tmp/sys-metrics.json', 'w') as f:
        json.dump(metrics, f)