import torch
from torchvision import transforms
import autoaugment
from spikingjelly.clock_driven import functional
import utils    

aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
train_aug = autoaugment.SNNAugmentWide()
def train_one_epoch(model, criterion, optimizer, data_loader, device, mixup_fn=None):
  model.train()
  global_acc1 = 0.0
  global_acc3 = 0.0
  for i, (images, labels) in enumerate(data_loader):

    # Prepare Data  
    images, labels = images.to(device), labels.to(device)
    images = images.float() # B T C H W
    B = images.shape[0]

    # Image Augmentation
    images = torch.stack([(aug(images[i])) for i in range(B)])
    images = torch.stack([(train_aug(images[i])) for i in range(B)])

    if mixup_fn is not None:
        images, labels = mixup_fn(images, labels)
        target_for_compu_acc = labels.argmax(dim=-1)


    output = model(images)
    loss = criterion(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    functional.reset_net(model)
    
    if mixup_fn is not None:
        acc1, acc3 = utils.accuracy(output, target_for_compu_acc, topk=(1, 3))
    else:
        acc1, acc3 = utils.accuracy(output, labels, topk=(1, 3))
    

    global_acc1 += acc1.item()
    global_acc3 += acc3.item()

  # gather the stats from all processes
  return global_acc1 / (i+1), global_acc3 / (i+1)

def evaluate(model, data_loader, device):
  # model.eval()
  global_acc1 = 0.0
  global_acc3 = 0.0
  with torch.no_grad():
    for i, (images, labels) in enumerate(data_loader):
      images = images.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)
      images = images.float()
      
      output = model(images)

      functional.reset_net(model)

      acc1, acc3 = utils.accuracy(output, labels, topk=(1, 3))

      global_acc1 += acc1.item()
      global_acc3 += acc3.item()

  return global_acc1 / (i+1), global_acc3 / (i+1)