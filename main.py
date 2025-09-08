## Imports ##
import os
import torch
import random
from torch import nn
import torch.utils.data
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import models.spikformer.model
from data import load_data
from args import parse_args
from train import train_one_epoch, evaluate

## Set Seed ##
# _seed_ = 2021
# random.seed(2021)
# torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
# torch.cuda.manual_seed_all(_seed_)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

root_path = os.path.abspath(__file__)

def format(num):
  whole, decimal = str(num).split(".")
  if len(whole) < 2: whole = "0"+whole
  if len(decimal) < 2: decimal = decimal + "0"
  return f'{whole}.{decimal}'
  

## Args ##
args = parse_args()

## Load Data ##
print("Loading Data...")
train_loader, test_loader = load_data(args.data_path, args.batch_size, args.T)

## Load Model ##
print("Loading Model...")
if args.model == 'spikformer':
  from timm.models import create_model
  model = create_model('spikformer').to(args.device)
else:
  from models.spikevit.config import get_config
  config = get_config(args.model)
  from models.spikevit.spikevit import create_spike_vit
  model = create_spike_vit(config, args).to(args.device)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number of params: {n_parameters}")
with open(args.file, 'w') as f: 
  f.write(f'Parameters: {n_parameters:,}\n')
  f.write(f'         \t|     Train     |      Test     |\n')
  f.write(f'         \t| Top-1 | Top-3 | Top-1 | Top-3 |\n')

## Training
optimizer = create_optimizer(args, model)
criterion = nn.CrossEntropyLoss().to(args.device)
lr_scheduler, num_epochs = create_scheduler(args, optimizer)
print(f'         \t|     Train     |      Test     |')
print(f'         \t| Top-1 | Top-3 | Top-1 | Top-3 |')
for epoch in range(args.start_epoch, num_epochs):
  train_acc1, train_acc3 = train_one_epoch(model, criterion, optimizer, train_loader, args.device)
  lr_scheduler.step(epoch + 1)
  test_acc1, test_acc3 = evaluate(model, test_loader, args.device)
  print(f'[{epoch+1}/{num_epochs}] \t| {format(round(train_acc1,2))} | {format(round(train_acc3,2))} | {format(round(test_acc1,2))} | {format(round(test_acc3,2))} |')
  with open(args.file, 'a') as f:
    f.write(f'[{epoch+1}/{num_epochs}] \t| {format(round(train_acc1,2))} | {format(round(train_acc3,2))} | {format(round(test_acc1,2))} | {format(round(test_acc3,2))} |\n')






















































