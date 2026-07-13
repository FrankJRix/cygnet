import cygnet
import imageio.v3 as iio
import torch
import numpy as np
from matplotlib import pyplot as plt

raw_data_path = "data/"
raw_data = cygnet.RootDataset(raw_data_path, 4)

last = 0

for i, pair in enumerate(raw_data):
    print(i)
    ev, gt = pair
    #print(ev, gt)
    # iio.imwrite(f"data_processed/{i}_i_scaled.png", ev.squeeze()*200)
    # iio.imwrite(f"data_processed/{i}_gt_scaled.png", gt.squeeze()*(2**16-1))
    if i >= last:
        break


vettore = np.zeros((16,16))
tensore = torch.tensor(vettore).unsqueeze(0)

print(tensore.shape)

maxed = cygnet.maxpooler(tensore)
print(maxed.shape)

avger = cygnet.AvgTransform(4)
avged = avger(tensore)
print(avged.shape)

imaged = cygnet.downscaler_input(vettore)
print(imaged.shape)