import torch
import torch.nn as nn

from .spiking import SpikeConv, SpikeBottleneck
# from .baseconv import SpikeBottleNeck
# from .bridge import ToFormerBridge, ToMobileBridge
from .block import SpikeFormer, SpikeMobile, ToMobileBridge, ToFormerBridge


class GAP(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, x):
    T, B, C, H, W = x.shape

    k = torch.ones(size=(T, B, C, W, 1)).to(x.device)
    v = torch.ones(size=(T, B, C, 1, H)).to(x.device)

    x_k = torch.matmul(x, k) # T B C H 1
    x_k_v = torch.matmul(v, x_k).reshape(T, B, C) # T B C

    return x_k_v / (H*W)

class Head(nn.Module):
  def __init__(self, in_channels, out_channels, embed_dim, tokens):
    super().__init__()
    
    self.pool = GAP()
    self.head = nn.Linear(in_channels+embed_dim, out_channels)

  def forward(self, x, tokens):
    
    # Global Average Pooling
    x = self.pool(x) # T B C

    # Cls Token
    token = tokens[:, :, :, 0] # T B D

    # Merge
    output = torch.cat([x, token], dim=-1).mean(0) # B D+C

    # Classify
    return self.head(output)
  
class ChannelConv(nn.Module): 
  def __init__(self, in_channels, out_channels):
    super().__init__()

    self.conv = SpikeConv(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1, stride=1, padding=0)
  
  def forward(self, x):
    return self.conv(x)
  
class SpikeBlock(nn.Module):
  def __init__(self, args, in_channels, hidden_channels, out_channels, embed_dim, 
               mlp_ratio=2, stride=1, kernel_size=3, padding=1):
    super().__init__()

    self.to_former = ToFormerBridge(in_channels=in_channels,embed_dim=embed_dim)
    self.former = SpikeFormer(embed_dim=embed_dim, mlp_ratio=mlp_ratio, heads=args.heads)
    self.mobile = SpikeMobile(in_channels=in_channels,
                              hidden_channels=hidden_channels,
                              out_channels=out_channels,
                              stride=stride,
                              kernel_size=kernel_size,
                              padding=padding)
    self.to_mobile = ToMobileBridge(in_channels=out_channels, embed_dim=embed_dim)
  
  def forward(self, x, tokens):

    # Former
    tokens = self.to_former(x, tokens)
    tokens =self.former(tokens)

    # Mobile
    x = self.mobile(x)
    x = self.to_mobile(x, tokens)

    return x, tokens

class Stem(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels, stride, kernel_size, padding):
    super().__init__()

    self.conv = SpikeConv(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding)

    self.bneck = SpikeBottleneck(in_channels=out_channels,
                                 hidden_channels=hidden_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 padding=padding)
  
  def forward(self, x):
    return self.bneck(self.conv(x))

class SpikeViT(nn.Module):
  def __init__(self, 
               config,
               args, 
               in_channels):
    super().__init__()

    # Set Values
    T = args.T   
    M, D = config.TOKENS

    # Initialize Tokens
    self._token(M,D,T)

    # Stem
    self.stem = Stem(in_channels=in_channels,
                     hidden_channels=config.STEM.HIDDEN_CHANNELS,
                     out_channels=config.STEM.OUT_CHANNELS,
                     stride=config.STEM.STRIDE,
                     kernel_size=config.STEM.KERNEL_SIZE,
                     padding=config.STEM.PADDING)

    # Blocks
    self.blocks = nn.ModuleList([])
    for i,_ in enumerate(config.BLOCK.IN_CHANNELS):
      self.blocks.append(
        SpikeBlock(args=args,
                   in_channels=config.BLOCK.IN_CHANNELS[i],
                   hidden_channels=config.BLOCK.HIDDEN_CHANNELS[i],
                   out_channels=config.BLOCK.OUT_CHANNELS[i],
                   embed_dim=D,
                   mlp_ratio=config.BLOCK.MLP_RATIO[i],
                   stride=config.BLOCK.STRIDE[i],
                   kernel_size=config.BLOCK.KERNEL_SIZE[i],
                   padding=config.BLOCK.PADDING[i])
      )

    self.end_former_bridge = ToFormerBridge(in_channels=config.BLOCK.OUT_CHANNELS[-1],embed_dim=D)

    self.channel_conv = ChannelConv(in_channels=config.BLOCK.OUT_CHANNELS[-1],
                                    out_channels=config.CHANNEL_CONV.OUT_CHANNELS)
    
    self.head = Head(in_channels=config.CHANNEL_CONV.OUT_CHANNELS,
                     out_channels=args.classes,
                     embed_dim=D,
                     tokens=M)
    
  def _token(self, M, D, T):

    # Token Shape
    shape = [T, 1, D, M]

    # Random Initialization [0,1]
    self.tokens = torch.rand(size=shape, dtype=torch.float32)
    
    # Binarize
    self.tokens = torch.round(self.tokens)

    # Parameterize
    self.tokens = nn.Parameter(self.tokens)

  def _to_batch_tokens(self, B):
    return torch.cat([self.tokens]*B, dim=1)
  
  def forward(self, x):

    # Permute Input
    x = x.permute(1, 0, 2, 3, 4)
    T, B, C, H, W = x.shape

    # Batch Tokens
    tokens = self._to_batch_tokens(B)
    T, B, D, M = tokens.shape

    # Stem
    x = self.stem(x)

    # Blocks
    for block in self.blocks:
      x, tokens = block(x, tokens)
    
    tokens = self.end_former_bridge(x, tokens)
    x = self.channel_conv(x)

    return self.head(x, tokens)
    



def create_spike_vit(config, args):
  return SpikeViT(config, args, in_channels=2)