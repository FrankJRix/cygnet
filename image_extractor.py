import cygnet
import imageio.v3 as iio

raw_data_path = "data/"
raw_data = cygnet.ImageDataset(raw_data_path)

last = 0

for i, pair in enumerate(raw_data):
    print(i)
    ev, gt = pair
    #print(ev, gt)
    # iio.imwrite(f"data_processed/{i}_i_scaled.png", ev.squeeze()*200)
    # iio.imwrite(f"data_processed/{i}_gt_scaled.png", gt.squeeze()*(2**16-1))
    if i >= last:
        break