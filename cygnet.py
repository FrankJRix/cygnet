import numpy as np
import uproot
import matplotlib.pyplot as plt
import os, gc, time
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image as im

picmin = 198
picmax = 220

class RootInterface():
    def __init__(self, path, is_binary=True):
        self.load_file(path)
        self.is_binary = is_binary
    
    def load_file(self, path):
        self.file = uproot.open(path)
        tree = self.file["event_info"]
        self.evnum = len(tree["eventnumber"].arrays(library="np")["eventnumber"])
        self.ixs = tree["redpix_ix"].arrays(library="np")["redpix_ix"]
        self.iys = tree["redpix_iy"].arrays(library="np")["redpix_iy"]
        self.izs = tree["redpix_iz"].arrays(library="np")["redpix_iz"]
        self.p_shape = self.file[f"pic_run1_ev{0}"].to_numpy()[0].shape

    def get_num_of_events(self):
        return self.evnum

    def get_noisy_image(self, idx):
        return self.file[f"pic_run1_ev{idx}"].to_numpy()[0].T

    def get_clean_image(self, idx):
        ix_event = self.ixs[idx]
        iy_event = self.iys[idx]
        iz_event = self.izs[idx]
        image = np.zeros(self.p_shape, dtype=np.uint16)
        image[ix_event, iy_event] = 1 if self.is_binary else iz_event

        return image.T

    def plot_noisy_image(self, idx, num_title = -1):
        if num_title == -1:
            num_title = idx
        img = self.get_noisy_image(idx)
        plt.figure(figsize = (8,8))
        plt.title(f"EVENT #{num_title}")
        plt.imshow(img, vmin = picmin, vmax = picmax)
        plt.show()
    
    def plot_clean_image(self, idx, num_title = -1):
        if num_title == -1:
            num_title = idx
        img = self.get_clean_image(idx)
        plt.figure(figsize = (8,8))
        plt.title(f"redpix of EVENT #{num_title}")
        plt.imshow(img)
        plt.show()

    def plot_event(self, idx, num_title = -1):
        if num_title == -1:
            num_title = idx
        noisy = self.get_noisy_image(idx)
        clean = self.get_clean_image(idx)

        fig, axs = plt.subplots(1, 2, figsize=(15,15))
        axs[0].imshow(noisy, vmin = picmin, vmax = picmax)
        axs[0].set_title(f"EVENT #{num_title}")
        axs[1].imshow(clean)
        axs[1].set_title(f"redpix of EVENT #{num_title}")

class RootManager():
    def __init__(self, dir):
        self.data_dir = dir
        self.files = sorted(os.listdir(self.data_dir))

        self.events_per_file_dict = {}
        self.rootfile_starts_dict = {}
        sum = 0
        for file in self.files:
            path = os.path.join(self.data_dir, file)
            ifc = RootInterface(path)
            tmp = ifc.get_num_of_events()

            self.rootfile_starts_dict[file] = sum
            self.events_per_file_dict[file] = tmp
            sum += tmp
        self.total_events = sum
        self.rootfile_starts_dict["end"] = sum
        #display(self.events_per_file_dict)
        #display(self.rootfile_starts_dict)

        self.load_file(self.files[0])
    
    def load_file(self, file):
        path = os.path.join(self.data_dir, file)
        self.ifc = RootInterface(path)
        self.current_file = file
    
    def update_file_pointer(self, idx):
        if idx >= self.total_events or idx < 0:
            print(f"{idx} is OUT OF BOUNDS!")
            return

        file = ""
        offset = 0
        for key, val in self.rootfile_starts_dict.items():
            #print(key, val)
            if idx >= val:
                file = key
                offset = val
            else:
                break
        #print(f"{idx} sta in {file}")

        if file == "":
            print("SOMETHING WENT WRONG")
            return
        elif file != self.current_file:
            self.load_file(file)
            #print(f"{file} loaded")
        
        #self.ifc.plot_event(idx - offset, idx)
    
    def get_noisy(self, idx):
        self.update_file_pointer(idx)
        offset = self.rootfile_starts_dict[self.current_file]
        return self.ifc.get_noisy_image(idx - offset)
    
    def get_clean(self, idx):
        self.update_file_pointer(idx)
        offset = self.rootfile_starts_dict[self.current_file]
        return self.ifc.get_clean_image(idx - offset)
    
    def get_pair(self, idx):
        return self.get_noisy(idx), self.get_clean(idx)

class ImageDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.rmngr = RootManager(dir)
        self.transform = transform

    def __len__(self):
        return self.rmngr.total_events

    def __getitem__(self, idx):
        img, target = self.rmngr.get_pair(idx)
        if self.transform:
            img = self.transform(img)
            target = self.transform(target)
        
        return img, target


# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print(f"Total execution time = {(end_time - start_time):.3f} sec")
    print(f"Max memory used by tensors = {torch.cuda.max_memory_allocated() / 1024**3:.3f} Gbytes")


# Network

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Net(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        # Encoder
        self.down = nn.AvgPool2d(2)
        self.enc1 = conv_block(1,      base)       # 16
        self.enc2 = conv_block(base,   base*2)     # 32
        self.enc3 = conv_block(base*2, base*4)     # 64
        self.enc4 = conv_block(base*4, base*8)     # 128
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Bottleneck
        self.bottleneck = conv_block(base*8, base*16)  # 256

        # Decoder (lightweight — just upsample + one conv_block)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = conv_block(base*16, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block(base*8,  base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4,  base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2,  base)

        # Single output head
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        # Encode
        e1 = self.enc1(self.down(x))
        #e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decode with skips
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.upsample(self.head(d1))  # raw logits, (N, 1, H, W)

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    # target: binary float mask from redpixels > 0
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt  = torch.exp(-bce)
    return (alpha * (1 - pt)**gamma * bce).mean()

def kl_loss(y_actual, x_pred, eps=1e-8):
    xx = x_pred.flatten()
    yy = y_actual.flatten()
    return ((yy - xx) * torch.log(yy / (xx + eps))).sum() / xx.numel()