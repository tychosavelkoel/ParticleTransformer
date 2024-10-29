import json

#DSVDD packages
import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.plots import plot_distance, plot_loss
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset


#ParT packages
import os
import sys
import shutil
import glob
import argparse
import functools
from torch.utils.data import DataLoader
from weaver.utils.logger import _logger, _configLogger
from weaver.utils.dataset import SimpleIterDataset
from weaver.utils.import_tools import import_module
from weaver.utils.data.tools import _concat

"""
python main.py mnist mnist_LeNet ../log/mnist_test ../data --objective one-class --lr 0.0001
--lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 50 
--ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3;
"""

################################################################################
# Settings
################################################################################

parser = argparse.ArgumentParser()
#Arguments DSVDD
parser.add_argument('dataset_name', default = 'mnist', type=str, help = "name of the dataset choose between 'mnist', 'cifar10'")
parser.add_argument('net_name', default = 'mnist_LeNet', type=str, help = "name of the neural network choose between 'mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'ParT'")
parser.add_argument('xp_path', default = '../log/mnist_test', type=str, help = "export path")
parser.add_argument('data_path', default = '../data',  type=str, help = 'path to data')
parser.add_argument('--load_config', default=None, type=str,
              help='Config JSON-file path (default: None).')
parser.add_argument('--load_model',  default=None, type=str,
              help='Model file path (default: None).')
parser.add_argument('--objective', default='one-class', type=str,
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
parser.add_argument('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
parser.add_argument('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
parser.add_argument('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
parser.add_argument('--optimizer_name', default='adam', type=str,
              help='Name of the optimizer to use for Deep SVDD network training.')
parser.add_argument('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
parser.add_argument('--lr_milestone', nargs = '+', type=int, default=0,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Cannot yet be multiple and must be increasing.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
parser.add_argument('--warm_up_n_epochs', type=int, default=0,
              help='epoch at which training is going to update radius')
parser.add_argument('--delta', type=float, default=1e-3,
              help='thresshold to set radius')
parser.add_argument('--epsilon', type=float, default=1e-5,
              help='thresshold to finish learning')
parser.add_argument('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
parser.add_argument('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
#for pretrain:
parser.add_argument('--pretrain', type=bool, default=False,
              help='Pretrain neural network parameters via autoencoder.')
parser.add_argument('--ae_optimizer_name', default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
parser.add_argument('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
parser.add_argument('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
parser.add_argument('--ae_lr_milestone',nargs = '+', type=int, default=0,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Cannot yet be multiple and must be increasing.')
parser.add_argument('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
parser.add_argument('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')

#Arguments particle transformer
parser.add_argument('--regression-mode', action='store_true', default=False,
                    help='run in regression mode if this flag is set; otherwise run in classification mode')
parser.add_argument('-c', '--data-config', type=str,
                    help='data config YAML file')
parser.add_argument('--extra-selection', type=str, default=None,
                    help='Additional selection requirement, will modify `selection` to `(selection) & (extra)` on-the-fly')
parser.add_argument('--extra-test-selection', type=str, default=None,
                    help='Additional test-time selection requirement, will modify `test_time_selection` to `(test_time_selection) & (extra)` on-the-fly')
parser.add_argument('-i', '--data-train', nargs='*', default=[],
                    help='training files; supported syntax:'
                         ' (a) plain list, `--data-train /path/to/a/* /path/to/b/*`;'
                         ' (b) (named) groups [Recommended], `--data-train a:/path/to/a/* b:/path/to/b/*`,'
                         ' the file splitting (for each dataloader worker) will be performed per group,'
                         ' and then mixed together, to ensure a uniform mixing from all groups for each worker.')
parser.add_argument('-l', '--data-val', nargs='*', default=[],
                    help='validation files; when not set, will use training files and split by `--train-val-split`')
parser.add_argument('-t', '--data-test', nargs='*', default=[],
                    help='testing files; supported syntax:'
                         ' (a) plain list, `--data-test /path/to/a/* /path/to/b/*`;'
                         ' (b) keyword-based, `--data-test a:/path/to/a/* b:/path/to/b/*`, will produce output_a, output_b;'
                         ' (c) split output per N input files, `--data-test a%%10:/path/to/a/*`, will split per 10 input files')
parser.add_argument('--data-fraction', type=float, default=1,
                    help='fraction of events to load from each file; for training, the events are randomly selected for each epoch')
parser.add_argument('--file-fraction', type=float, default=1,
                    help='fraction of files to load; for training, the files are randomly selected for each epoch')
parser.add_argument('--fetch-by-files', action='store_true', default=False,
                    help='When enabled, will load all events from a small number (set by ``--fetch-step``) of files for each data fetching. '
                         'Otherwise (default), load a small fraction of events from all files each time, which helps reduce variations in the sample composition.')
parser.add_argument('--fetch-step', type=float, default=0.01,
                    help='fraction of events to load each time from every file (when ``--fetch-by-files`` is disabled); '
                         'Or: number of files to load each time (when ``--fetch-by-files`` is enabled). Shuffling & sampling is done within these events, so set a large enough value.')
parser.add_argument('--in-memory', action='store_true', default=False,
                    help='load the whole dataset (and perform the preprocessing) only once and keep it in memory for the entire run')
parser.add_argument('--train-val-split', type=float, default=0.8,
                    help='training/validation split fraction')
parser.add_argument('--no-remake-weights', action='store_true', default=False,
                    help='do not remake weights for sampling (reweighting), use existing ones in the previous auto-generated data config YAML file')
parser.add_argument('--demo', action='store_true', default=False,
                    help='quickly test the setup by running over only a small number of events')
parser.add_argument('--lr-finder', type=str, default=None,
                    help='run learning rate finder instead of the actual training; format: ``start_lr, end_lr, num_iters``')
parser.add_argument('--tensorboard', type=str, default=None,
                    help='create a tensorboard summary writer with the given comment')
parser.add_argument('--tensorboard-custom-fn', type=str, default=None,
                    help='the path of the python script containing a user-specified function `get_tensorboard_custom_fn`, '
                         'to display custom information per mini-batch or per epoch, during the training, validation or test.')
parser.add_argument('-n', '--network-config', type=str,
                    help='network architecture configuration file; the path must be relative to the current dir')
parser.add_argument('-o', '--network-option', nargs=2, action='append', default=[],
                    help='options to pass to the model class constructor, e.g., `--network-option use_counts False`')
parser.add_argument('-m', '--model-prefix', type=str, default='models/{auto}/network',
                    help='path to save or load the model; for training, this will be used as a prefix, so model snapshots '
                         'will saved to `{model_prefix}_epoch-%%d_state.pt` after each epoch, and the one with the best '
                         'validation metric to `{model_prefix}_best_epoch_state.pt`; for testing, this should be the full path '
                         'including the suffix, otherwise the one with the best validation metric will be used; '
                         'for training, `{auto}` can be used as part of the path to auto-generate a name, '
                         'based on the timestamp and network configuration')
parser.add_argument('--load-model-weights', type=str, default=None,
                    help='initialize model with pre-trained weights')
parser.add_argument('--exclude-model-weights', type=str, default=None,
                    help='comma-separated regex to exclude matched weights from being loaded, e.g., `a.fc..+,b.fc..+`')
parser.add_argument('--max-epochs', type=int, default=20,
                    help='maximum number of epochs')
parser.add_argument('--min-epochs', type=int, default=5,
                    help='minimum number of epochs')
parser.add_argument('--steps-per-epoch', type=int, default=None,
                    help='number of steps (iterations) per epochs; '
                         'if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples')
parser.add_argument('--steps-per-epoch-val', type=int, default=None,
                    help='number of steps (iterations) per epochs for validation; '
                         'if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples')
parser.add_argument('--samples-per-epoch', type=int, default=None,
                    help='number of samples per epochs; '
                         'if neither of `--steps-per-epoch` or `--samples-per-epoch` is set, each epoch will run over all loaded samples')
parser.add_argument('--samples-per-epoch-val', type=int, default=None,
                    help='number of samples per epochs for validation; '
                         'if neither of `--steps-per-epoch-val` or `--samples-per-epoch-val` is set, each epoch will run over all loaded samples')
parser.add_argument('--optimizer', type=str, default='ranger', choices=['adam', 'adamW', 'radam', 'ranger'],  # TODO: add more
                    help='optimizer for the training')
parser.add_argument('--optimizer-option', nargs=2, action='append', default=[],
                    help='options to pass to the optimizer class constructor, e.g., `--optimizer-option weight_decay 1e-4`')
parser.add_argument('--lr-scheduler', type=str, default='flat+decay',
                    choices=['none', 'steps', 'flat+decay', 'flat+linear', 'flat+cos', 'one-cycle'],
                    help='learning rate scheduler')
parser.add_argument('--warmup-steps', type=int, default=0,
                    help='number of warm-up steps, only valid for `flat+linear` and `flat+cos` lr schedulers')
parser.add_argument('--load-epoch', type=int, default=None,
                    help='used to resume interrupted training, load model and optimizer state saved in the `epoch-%%d_state.pt` and `epoch-%%d_optimizer.pt` files')
parser.add_argument('--start-lr', type=float, default=5e-3,
                    help='start learning rate')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--use-amp', action='store_true', default=False,
                    help='use mixed precision training (fp16)')
parser.add_argument('--gpus', type=str, default='0',
                    help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`')
parser.add_argument('--predict-gpus', type=str, default=None,
                    help='device for the testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`; if not set, use the same as `--gpus`')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with this numbers')
parser.add_argument('--predict', action='store_true', default=False,
                    help='run prediction instead of training')
parser.add_argument('--predict-output', type=str,
                    help='path to save the prediction output, support `.root` and `.parquet` format')
parser.add_argument('--export-onnx', type=str, default=None,
                    help='export the PyTorch model to ONNX model and save it at the given path (path must ends w/ .onnx); '
                         'needs to set `--data-config`, `--network-config`, and `--model-prefix` (requires the full model path)')
parser.add_argument('--onnx-opset', type=int, default=15,
                    help='ONNX opset version.')
parser.add_argument('--io-test', action='store_true', default=False,
                    help='test throughput of the dataloader')
parser.add_argument('--copy-inputs', action='store_true', default=False,
                    help='copy input files to the current dir (can help to speed up dataloading when running over remote files, e.g., from EOS)')
parser.add_argument('--log', type=str, default='',
                    help='path to the log file; `{auto}` can be used as part of the path to auto-generate a name, based on the timestamp and network configuration')
parser.add_argument('--print', action='store_true', default=False,
                    help='do not run training/prediction but only print model information, e.g., FLOPs and number of parameters of a model')
parser.add_argument('--profile', action='store_true', default=False,
                    help='run the profiler')
parser.add_argument('--backend', type=str, choices=['gloo', 'nccl', 'mpi'], default=None,
                    help='backend for distributed training')
parser.add_argument('--cross-validation', type=str, default=None,
                    help='enable k-fold cross validation; input format: `variable_name%%k`')


def to_filelist(args, mode='train'):
    if mode == 'train':
        flist = args.data_train
    elif mode == 'val':
        flist = args.data_val
    else:
        raise NotImplementedError('Invalid mode %s' % mode)

    # keyword-based: 'a:/path/to/a b:/path/to/b'
    file_dict = {}
    for f in flist:
        if ':' in f:
            name, fp = f.split(':')
        else:
            name, fp = '_', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    if args.local_rank is not None:
        if mode == 'train':
            local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[args.local_rank::local_world_size]
                assert(len(new_files) > 0)
                np.random.shuffle(new_files)
                new_file_dict[name] = new_files
            file_dict = new_file_dict

    if args.copy_inputs:
        import tempfile
        tmpdir = tempfile.mkdtemp()
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        new_file_dict = {name: [] for name in file_dict}
        for name, files in file_dict.items():
            for src in files:
                dest = os.path.join(tmpdir, src.lstrip('/'))
                if not os.path.exists(os.path.dirname(dest)):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                _logger.info('Copied file %s to %s' % (src, dest))
                new_file_dict[name].append(dest)
            if len(files) != len(new_file_dict[name]):
                _logger.error('Only %d/%d files copied for %s file group %s',
                              len(new_file_dict[name]), len(files), mode, name)
        file_dict = new_file_dict

    filelist = sum(file_dict.values(), [])
    assert(len(filelist) == len(set(filelist)))
    return file_dict, filelist

def train_load(args): #From ParT
    """
    Loads the training data.
    :param args:
    :return: train_loader, val_loader, data_config, train_inputs
    """

    train_file_dict, train_files = to_filelist(args, 'train')
    if args.data_val:
        val_file_dict, val_files = to_filelist(args, 'val')
        train_range = val_range = (0, 1)
    else:
        val_file_dict, val_files = train_file_dict, train_files
        train_range = (0, args.train_val_split)
        val_range = (args.train_val_split, 1)
    _logger.info('Using %d files for training, range: %s' % (len(train_files), str(train_range)))
    _logger.info('Using %d files for validation, range: %s' % (len(val_files), str(val_range)))

    if args.demo:
        train_files = train_files[:20]
        val_files = val_files[:20]
        train_file_dict = {'_': train_files}
        val_file_dict = {'_': val_files}
        _logger.info(train_files)
        _logger.info(val_files)
        args.data_fraction = 0.1
        args.fetch_step = 0.002

    if args.in_memory and (args.steps_per_epoch is None or args.steps_per_epoch_val is None):
        raise RuntimeError('Must set --steps-per-epoch when using --in-memory!')
    train_data = SimpleIterDataset(train_file_dict, args.data_config, for_training=True,
                                   extra_selection=args.extra_selection,
                                   remake_weights=not args.no_remake_weights,
                                   load_range_and_fraction=(train_range, args.data_fraction),
                                   file_fraction=args.file_fraction,
                                   fetch_by_files=args.fetch_by_files,
                                   fetch_step=args.fetch_step,
                                   infinity_mode=args.steps_per_epoch is not None,
                                   in_memory=args.in_memory,
                                   name='train' + ('' if args.local_rank is None else '_rank%d' % args.local_rank))
    val_data = SimpleIterDataset(val_file_dict, args.data_config, for_training=True,
                                 extra_selection=args.extra_selection,
                                 load_range_and_fraction=(val_range, args.data_fraction),
                                 file_fraction=args.file_fraction,
                                 fetch_by_files=args.fetch_by_files,
                                 fetch_step=args.fetch_step,
                                 infinity_mode=args.steps_per_epoch_val is not None,
                                 in_memory=args.in_memory,
                                 name='val' + ('' if args.local_rank is None else '_rank%d' % args.local_rank))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, pin_memory=True,
                              num_workers=min(args.num_workers, int(len(train_files) * args.file_fraction)),
                              persistent_workers=args.num_workers == 0 and args.steps_per_epoch is not None)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, pin_memory=True,
                            num_workers=min(args.num_workers, int(len(val_files) * args.file_fraction)),
                            persistent_workers=args.num_workers == 0 and args.steps_per_epoch_val is not None)
    data_config = train_data.config
    train_input_names = train_data.config.input_names
    train_label_names = train_data.config.label_names

    return train_loader, val_loader, data_config, train_input_names, train_label_names


def test_load(args): #From ParT
    """
    Loads the test data.
    :param args:
    :return: test_loaders, data_config
    """
    # keyword-based --data-test: 'a:/path/to/a b:/path/to/b'
    # split --data-test: 'a%10:/path/to/a/*'
    file_dict = {}
    split_dict = {}
    for f in args.data_test:
        if ':' in f:
            name, fp = f.split(':')
            if '%' in name:
                name, split = name.split('%')
                split_dict[name] = int(split)
        else:
            name, fp = '', f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files

    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    # apply splitting
    for name, split in split_dict.items():
        files = file_dict.pop(name)
        for i in range((len(files) + split - 1) // split):
            file_dict[f'{name}_{i}'] = files[i * split:(i + 1) * split]

    def get_test_loader(name):
        filelist = file_dict[name]
        _logger.info('Running on test file group %s with %d files:\n...%s', name, len(filelist), '\n...'.join(filelist))
        num_workers = min(args.num_workers, len(filelist))
        test_data = SimpleIterDataset({name: filelist}, args.data_config, for_training=False,
                                      extra_selection=args.extra_test_selection,
                                      load_range_and_fraction=((0, 1), args.data_fraction),
                                      fetch_by_files=True, fetch_step=1,
                                      name='test_' + name)
        test_loader = DataLoader(test_data, num_workers=num_workers, batch_size=args.batch_size, drop_last=False,
                                 pin_memory=True)
        return test_loader

    test_loaders = {name: functools.partial(get_test_loader, name) for name in file_dict}
    data_config = SimpleIterDataset({}, args.data_config, for_training=False).config
    return test_loaders, data_config


def model_setup(args, data_config, device='cpu'): #From ParT
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config, name='_network_module')
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    _logger.info('Network options: %s' % str(network_options))
    if args.use_amp:
        network_options['use_amp'] = True
    model, model_info = network_module.get_model(data_config, **network_options)
    if args.load_model_weights:
        model_state = torch.load(args.load_model_weights, map_location='cpu')
        if args.exclude_model_weights:
            import re
            exclude_patterns = args.exclude_model_weights.split(',')
            _logger.info('The following weights will not be loaded: %s' % str(exclude_patterns))
            key_state = {}
            for k in model_state.keys():
                key_state[k] = True
                for pattern in exclude_patterns:
                    if re.match(pattern, k):
                        key_state[k] = False
                        break
            model_state = {k: v for k, v in model_state.items() if key_state[k]}
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _logger.info('Model initialized with weights from %s\n ... Missing: %s\n ... Unexpected: %s' %
                     (args.load_model_weights, missing_keys, unexpected_keys))
    # _logger.info(model)
    flops(model, model_info, device=device)
    
    return model, model_info

def save_root(args, output_path, data_config, scores, observers):
    """
    Saves as .root
    :param data_config:
    :param scores:
    :param observers
    :return:
    """
    import awkward as ak
    from weaver.utils.data.fileio import _write_root
    output = {}
    output['Squared_distance'] = scores
    output.update(observers)
    
    try:
        _write_root(output_path, ak.Array(output))
        _logger.info('Written output to %s' % output_path, color='bold')
    except Exception as e:
        _logger.error('Error when writing output ROOT file: \n' + str(e))

def _main(args):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """
    #Arguments DSVDD
    dataset_name = args.dataset_name 
    net_name =  args.net_name
    xp_path = args.xp_path
    data_path = args.data_path
    load_config = args.load_config
    load_model = args.load_model
    objective = args.objective
    nu = args.nu 
    device = args.device
    seed = args.seed
    optimizer_name = args.optimizer_name
    lr = args.lr
    lr_milestone = args.lr_milestone
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    pretrain = args.pretrain 
    ae_optimizer_name = args.ae_optimizer_name
    ae_lr = args.ae_lr
    ae_n_epochs = args.ae_n_epochs
    ae_lr_milestone = args.ae_lr_milestone
    ae_batch_size = args.ae_batch_size 
    ae_weight_decay = args.ae_weight_decay
    n_jobs_dataloader = args.n_jobs_dataloader 
    normal_class = args.normal_class
    warm_up_n_epochs = args.warm_up_n_epochs

    ### From ParT ###
    

    training_mode = not args.predict

    # device
    if args.gpus:
        # distributed training
        if args.backend is not None:
            local_rank = args.local_rank
            torch.cuda.set_device(local_rank)
            gpus = [local_rank]
            dev = torch.device(local_rank)
            torch.distributed.init_process_group(backend=args.backend)
            _logger.info(f'Using distributed PyTorch with {args.backend} backend')
        else:
            gpus = [int(i) for i in args.gpus.split(',')]
            dev = torch.device(gpus[0])
    else:
        gpus = None
        dev = torch.device('cpu')
        try:
            if torch.backends.mps.is_available():
                dev = torch.device('mps')
        except AttributeError:
            pass
    
    

    # load data
    if training_mode:
        train_loader, val_loader, train_data_config, train_input_names, train_label_names = train_load(args)
    else:
        test_loaders, test_data_config = test_load(args)


    if args.print:
        return

    if args.tensorboard:
        from weaver.utils.nn.tools import TensorboardHelper
        tb = TensorboardHelper(tb_comment=args.tensorboard, tb_custom_fn=args.tensorboard_custom_fn)
    else:
        tb = None

    ### From DSVDD ###
    # Get configuration
    cfg = Config(locals().copy())
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % args.objective)
    logger.info('Nu-paramerter: %.2f' % args.nu)

    # Set seed
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.info('Set seed to %d.' %args.seed)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    #dataset = load_dataset(dataset_name, data_path, normal_class)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(args.objective, args.nu)
    deep_SVDD.set_network(net_name, train_data_config)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    #Not yet in model
    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deep_SVDD.train(optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader,
                    train_loader = train_loader,
                    val_loader =  val_loader,
                    data_config = train_data_config,
                    steps_per_epoch = args.steps_per_epoch,
                    steps_per_epoch_val = args.steps_per_epoch_val,
                    min_epochs = args.min_epochs,
                    max_epochs = args.max_epochs,
                    warm_up_n_epochs = warm_up_n_epochs,
                    epsilon = args.epsilon,
                    delta = args.delta)
    
    train_scores = deep_SVDD.results['train_scores']
    val_scores = deep_SVDD.results['val_scores']
    loss_train = deep_SVDD.results['loss']
    loss_val = deep_SVDD.results['loss_val']
    loss_condition_train = deep_SVDD.results['loss_condition']
    loss_condition_val = deep_SVDD.results['loss_condition_val']
    train_observers = deep_SVDD.Train_observers
    train_observers = {k: _concat(v) for k, v in train_observers.items()}
    

    test_loaders, test_data_config = test_load(args)
    for name, get_test_loader in test_loaders.items():
            test_loader = get_test_loader()


    # Test model
    deep_SVDD.test(device=device, 
                    n_jobs_dataloader=n_jobs_dataloader,
                    test_loader = test_loader,
                    data_config = test_data_config,
                    steps_per_epoch = args.steps_per_epoch)

    
    test_scores = deep_SVDD.results['test_scores']
    test_observers = deep_SVDD.Test_observers
    test_observers = {k: _concat(v) for k, v in test_observers.items()}

    if args.predict_output:
        predict_output = os.path.join(
            os.path.dirname(args.model_prefix),
            'predict_output', args.predict_output)       
        os.makedirs(os.path.dirname(predict_output), exist_ok=True)
        output_path2 = os.path.join(
            os.path.dirname(args.model_prefix),
            'predict_output', "train_results.root")
        if name == '':
            output_path = predict_output
        else:
            base, ext = os.path.splitext(predict_output)
            output_path = base + '_' + name + ext
        if output_path.endswith('.root'):
            save_root(args, output_path2, train_data_config, train_scores, train_observers)
            save_root(args, output_path, test_data_config, test_scores, test_observers)

    #idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score
    if dataset_name == "ParT":
        radius = deep_SVDD.R
        plot_distance(radius**2, train_scores, title = os.path.join(os.path.dirname(args.model_prefix),
                            'predict_output', "distance_train"))
        plot_distance(radius**2, val_scores, title = os.path.join(os.path.dirname(args.model_prefix),
                            'predict_output', "distance_val"))
        plot_loss(loss_train, loss_val, title = os.path.join(os.path.dirname(args.model_prefix),
                            'predict_output', "loss"))
        plot_loss(loss_condition_train, loss_condition_val, ylabel = 'Loss condition', title = os.path.join(os.path.dirname(args.model_prefix),
                            'predict_output', "loss_condition"))
        plot_distance(radius**2, test_scores, title = os.path.join(os.path.dirname(args.model_prefix),
                            'predict_output', "distance_test"))
        
        # Save results, model, and configuration
        deep_SVDD.save_results(export_json= os.path.join(os.path.dirname(args.model_prefix),
                            'predict_output', "results.json"))
        deep_SVDD.save_model(export_model= os.path.join(os.path.dirname(args.model_prefix),
                            'predict_output', "model.tar"), save_ae=False)
        #cfg.save_config(export_json=xp_path + '/config.json')
        parsed = parser.parse_args()

        with open(os.path.join(os.path.dirname(args.model_prefix),
                            'predict_output', "config.json"), 'w') as fp:
                json.dump(parsed.__dict__, fp)

    '''
    # Plot most anomalous and most normal (within-class) test samples
    if dataset_name in ('mnist', 'cifar10'):

        if dataset_name == 'mnist':
            X_normals = dataset.test_set.test_data[idx_sorted[:32], ...].unsqueeze(1)
            X_outliers = dataset.test_set.test_data[idx_sorted[-32:], ...].unsqueeze(1)

        if dataset_name == 'cifar10':
            X_normals = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[:32], ...], (0, 3, 1, 2)))
            X_outliers = torch.tensor(np.transpose(dataset.test_set.test_data[idx_sorted[-32:], ...], (0, 3, 1, 2)))

        plot_images_grid(X_normals, export_img=xp_path + '/normals', title='Most normal examples', padding=2)
        plot_images_grid(X_outliers, export_img=xp_path + '/outliers', title='Most anomalous examples', padding=2)
    '''


    

def main():
    args = parser.parse_args()
    if args.samples_per_epoch is not None:
        if args.steps_per_epoch is None:
            args.steps_per_epoch = args.samples_per_epoch // args.batch_size
        else:
            raise RuntimeError('Please use either `--steps-per-epoch` or `--samples-per-epoch`, but not both!')

    if args.samples_per_epoch_val is not None:
        if args.steps_per_epoch_val is None:
            args.steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size
        else:
            raise RuntimeError('Please use either `--steps-per-epoch-val` or `--samples-per-epoch-val`, but not both!')

    if args.steps_per_epoch_val is None and args.steps_per_epoch is not None:
        args.steps_per_epoch_val = round(args.steps_per_epoch * (1 - args.train_val_split) / args.train_val_split)
    if args.steps_per_epoch_val is not None and args.steps_per_epoch_val < 0:
        args.steps_per_epoch_val = None

    if '{auto}' in args.model_prefix or '{auto}' in args.log:
        import hashlib
        import time
        model_name = time.strftime('%H%M%S')
        if len(args.network_option):
            model_name = model_name + "_" + hashlib.md5(str(args.network_option).encode('utf-8')).hexdigest()
        model_name += '_{optim}_lr{lr}_batch{batch}'.format(lr=args.start_lr,
                                                            optim=args.optimizer, batch=args.batch_size)
        args._auto_model_name = model_name
        args.model_prefix = args.model_prefix.replace('{auto}', model_name)
        args.log = args.log.replace('{auto}', model_name)
        print('Using auto-generated model prefix %s' % args.model_prefix)
    if '{datum}' in args.model_prefix or '{auto}' in args.log:
        model_date = time.strftime("%Y%m%d")
        args.model_prefix = args.model_prefix.replace('{datum}', model_date)
        args.log = args.log.replace('{datum}', model_date)

    if args.predict_gpus is None:
        args.predict_gpus = args.gpus

    args.local_rank = None if args.backend is None else int(os.environ.get("LOCAL_RANK", "0"))

    stdout = sys.stdout
    if args.local_rank is not None:
        args.log += '.%03d' % args.local_rank
        if args.local_rank != 0:
            stdout = None
    _configLogger('weaver', stdout=stdout, filename=args.log)

    if args.cross_validation:
        model_dir, model_fn = os.path.split(args.model_prefix)
        if args.predict_output:
            predict_output_base, predict_output_ext = os.path.splitext(args.predict_output)
        load_model = args.load_model_weights or None
        var_name, kfold = args.cross_validation.split('%')
        kfold = int(kfold)
        for i in range(kfold):
            _logger.info(f'\n=== Running cross validation, fold {i} of {kfold} ===')
            opts = copy.deepcopy(args)
            opts.model_prefix = os.path.join(f'{model_dir}_fold{i}', model_fn)
            if args.predict_output:
                opts.predict_output = f'{predict_output_base}_fold{i}' + predict_output_ext
            opts.extra_selection = f'{var_name}%{kfold}!={i}'
            opts.extra_test_selection = f'{var_name}%{kfold}=={i}'
            if load_model and '{fold}' in load_model:
                opts.load_model_weights = load_model.replace('{fold}', f'fold{i}')

            _main(opts)
    else:
        _main(args)


if __name__ == '__main__':
    main()
