# FedL2P: Federated Learning to Personalize

## Abstract
> Federated learning (FL) research has made progress in developing algorithms for distributed learning of global models, as well as algorithms for local personalization of those common models to the specifics of each client’s local data distribution. However, different FL problems may require different personalization strategies, and it may not even be possible to define an effective one-size-fits-all personalization strategy for all clients: depending on how similar each client’s optimal predictor is to that of the global model, different personalization strategies may be preferred. In this paper, we consider the federated meta-learning problem of learning personalization strategies. Specifically, we consider meta-nets that induce the batch-norm and learning rate parameters for each client given local data statistics. By learning these meta-nets through FL, we allow the whole FL network to collaborate in learning a customized personalization strategy for each client. Empirical results show that this framework improves on a range of standard hand-crafted personalization baselines in both label and feature shift situations.

## Directory Structure \& Main Files

```
main.py
config.py # Contains utility functions to parse configs
configs # Contains YAML files
|   default.yaml # default configurations
│
src
│   log.py # Checkpoint and logging functions to local directory & wandb
│   simulation.py # Distributed simulation functions 
│   ...
│
└───apps # Application-specific functions and main pipeline
│   │   app.py # Generic application class
│   │   ...
│   │
│   └───clients # Local training/finetuning and evaluation
│       │   ...
│   
└───data - Downloads and partitions dataset
│   │   fl_dataset.py # Generic federated dataset class
│   │   ...
│
└───models - Model definitions
│   │   ...
│
└───server
   │   ...
   │
   └───client_managers # Client sampling
   │   │   ...
   │   │
   └───strategies # Server aggregation strategies (FedAvg, FedAdam, etc)
       │   ...
       │   
       └───valuations # Measuring utility of client updates
           │   ...
```

## Datasets

All datasets are to be placed in `data.args.path_to_data`. Data is automatically partitioned in `data.args.dataset_fl_root`.

- Cifar10 - automatically downloads to `data.args.path_to_data`

- Cifar10-C - download the official [dataset](https://zenodo.org/record/2535967), unzip, and place in `data.args.path_to_data`

- SpeechCommands - automatically downloads to `data.args.path_to_data`

- DomainNet & Office-Caltech10 - download the pre-processed datasets in [FedBN](https://github.com/med-air/FedBN), unzip, and place datasets in `data.args.path_to_data`

## Usage & Examples

### General Usage

```
python main.py {path_to_yaml_file} # you can pass multiple yaml files and arguments. Later yaml/arguments will take precedence.
python main.py ./wandb/{wandb_local_run_folder}/files/user_config.yaml # resume a previous run (only if sync with wandb)
```

Set the maximum GPU memory allocated for each client by overwriting argument `vram`. Add `wandb_args.mode=disabled` to disable wandb or specify your own wandb entity `wandb_args.entity={your entity}`.

### CIFAR10 and CIFAR10-C Examples

```
# Cifar10 FedAvg. Pretrain from scratch with 1000 IID clients ($\alpha=1000$)
python main.py configs/cifar10/cifar10.yaml configs/cifar10/cifar_fedavg.yaml data.args.lda_alpha=\{1000:1000\} 

# include additional commands: training with 10 non-IID clients for 10 rounds, sampling 1 client per round.
python main.py configs/cifar10/cifar10.yaml configs/cifar10/cifar_fedavg.yaml data.args.lda_alpha=\{0.1:10\} simulation.num_clients=10 app.run.num_rounds=10 server.strategy.args.min_fit_clients=1

# Cifar10 FedL2P using pretrained model on 1000 IID clients
python main.py configs/cifar10/cifar10.yaml configs/cifar10/cifar_fedl2p.yaml data.args.lda_alpha=\{1000:1000\} app.args.load_model={local_path_to_model OR wandb_path_to_model}

# Cifar10-C FedL2P using pretrained model on 250 non-IID clients ($\alpha=1.0$)
python main.py configs/cifar10/cifar10.yaml configs/cifar10/cifar_fedl2p.yaml configs/cifar10/cifar10C.yaml data.args.lda_alpha=\{1.0:250\} app.args.load_model={local_path_to_model OR wandb_path_to_model}
```

### Speech Commands Examples
```
# SpeechCommands V2 FedAvg. Pretrain from scratch with default 250 clients
python main.py configs/commands/commands.yaml configs/commands/commands_fedavg.yaml 

# SpeechCommands V2 FedL2P using pretrained model 
python main.py configs/commands/commands.yaml configs/commands/commands_fedl2p.yaml app.args.load_model={local_path_to_model OR wandb_path_to_model}
```

### Office-Caltech10 & DomainNet Examples

In the paper, we used the pretrained [model](https://download.pytorch.org/models/resnet18-f37072fd.pth) provided by torchvision.

```
# Office FedL2P using pretrained model on 4 clients
python main.py configs/office/office.yaml configs/office/office_fedl2p.yaml app.args.load_model={local_path_to_model} data.args.lda_alpha=\{1000:4\}

# DomainNet FedL2P using pretrained model on 150 non-IID clients ($\alpha=1.0$) 
python main.py configs/domainnet/domainnet.yaml configs/domainnet/domainnet_fedl2p.yaml app.args.load_model={local_path_to_model} data.args.lda_alpha=\{1.0:150\}
``` 