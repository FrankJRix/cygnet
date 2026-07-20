import numpy as np
import uproot
import matplotlib.pyplot as plt
import os, gc, time
import imageio.v3 as iio
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset
import midas.file_reader

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
		image = np.zeros(self.p_shape, dtype=np.uint8 if self.is_binary else np.uint16)
		image[ix_event, iy_event] = 255 if self.is_binary else iz_event

		gc.collect()

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
		gc.collect()
	
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



def print_through(x):
	print("\n")
	print(x)
	print("\n")
	return x


class AvgTransform(nn.Module):
	def __init__(self, k_size):
		super().__init__()
		self.pooling = torch.nn.AvgPool2d(kernel_size=k_size)

	def forward(self, x):
		return self.pooling(x)

class MaxTransform(nn.Module):
	def __init__(self, k_size):
		super().__init__()
		self.pooling = torch.nn.MaxPool2d(kernel_size=k_size)

	def forward(self, x):
		return self.pooling(x)

class InflateTransform(nn.Module):
	def __init__(self, radius):
		super().__init__()
		
		k_size = 2 * radius + 1
		self.pooling = torch.nn.MaxPool2d(kernel_size=k_size, padding=radius, stride=1)

	def forward(self, x):
		return self.pooling(x)

#avgpooler = torch.nn.AvgPool2d(kernel_size=2)
#maxpooler = torch.nn.MaxPool2d(kernel_size=2)

def input_transform_builder(pool_k_size):
	downscaler = transforms.Compose([transforms.ToImage(), 
										transforms.ToDtype(torch.float32, scale=False), 
										AvgTransform(pool_k_size),
										torch.round, 
										transforms.ToDtype(torch.uint16, scale=False), 
										])
	return downscaler

def target_transform_builder(pool_k_size):
	downscaler = transforms.Compose([transforms.ToImage(), 
											transforms.ToDtype(torch.float32, scale=False), 
											MaxTransform(pool_k_size), 
											transforms.ToDtype(torch.uint8, scale=False)])
	return downscaler

def inflate_transform_builder(radius):
	downscaler = transforms.Compose([transforms.ToImage(), 
											transforms.ToDtype(torch.float32, scale=False), 
											InflateTransform(radius), 
											transforms.ToDtype(torch.uint8, scale=False)])
	return downscaler

class RootDataset(Dataset):
	def __init__(self, dir, pool_k_size, raw=False):
		self.rmngr = RootManager(dir)
		self.transform_input = input_transform_builder(pool_k_size)
		self.transform_target = target_transform_builder(pool_k_size)
		self.raw = raw

	def __len__(self):
		return self.rmngr.total_events

	def __getitem__(self, idx):
		img, target = self.rmngr.get_pair(idx)
		if not self.raw:
			img = self.transform_input(img)
			target = self.transform_target(target)
		
		return img, target

itrtr = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
trtr = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])

class CygnoSet(Dataset):
	def __init__(self, input_dir, target_dir, transform_input=itrtr, transform_target=trtr):
		self.input_dir = input_dir
		self.input_images = os.listdir(input_dir)
		self.target_dir = target_dir
		self.target_images = os.listdir(target_dir)

		self.mask_prefix = self.target_images[0].split("_")[0]
		#print(self.mask_prefix)

		self.transform_input = transform_input
		self.transform_target = transform_target

		self.stats = {}

	def __len__(self):
		return len(self.input_images)
	
	def get_raw_item(self, idx):
		input_image_path = os.path.join(self.input_dir, f"i_{idx}.png")
		input = iio.imread(input_image_path)
		target_image_path = os.path.join(self.target_dir, f"{self.mask_prefix}_{idx}.png")
		target = iio.imread(target_image_path)

		return input, target

	def __getitem__(self, idx):
		input, target = self.get_raw_item(idx)
		target = tv_tensors.Mask(target).unsqueeze(0)

		#print(input_image_path, target_image_path)
		
		if self.transform_input:
			input = self.transform_input(input)
		if self.transform_target:
			target = self.transform_target(target)
		
		return input, target
	
	def print_stats(self):
		print("=== dataset stats: ===")
		for key, item in self.stats.items():
			print(f"{key}: {item}")
		print()

	def compute_stats(self, raw=True):
		#if self.stats: print
		
		if raw:
			getter = self.get_raw_item
		else:
			getter = self.__getitem__

		i_means = []
		i_sqmeans = []

		m_means = []
		m_sqmeans = []

		for ii in range(len(self)):
			image, mask = getter(ii)
			i_means.append(image.mean())
			m_means.append(mask.mean())
		
		i_mean = np.array(i_means).mean()
		m_mean = np.array(m_means).mean()

		self.stats["input_mean"] = i_mean
		self.stats["mask_mean"] = m_mean

		self.print_stats()

def debug_plot(noisy, mask, alpha = 0.1, vmin = picmin, vmax = picmax):
	plt.figure(figsize = (12,12))
	plt.title(f"MASKED")
	plt.imshow(noisy, vmin = vmin, vmax = vmax)
	plt.imshow(mask, alpha=alpha)
	plt.show()

def open_mid(path):
	if os.path.exists(path):
		f = midas.file_reader.MidasFile(path)
	else:
		raise(RuntimeError())

	return f

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

class DoubleDown2(nn.Module):
	def __init__(self, chin, chout):
		super().__init__()
		self.seq = nn.Sequential(
			nn.Conv2d(chin, chout, 3, padding=1, bias=False),
			nn.BatchNorm2d(chout),
			nn.ReLU(),
			nn.Conv2d(chout, chout, 3, padding=1, bias=False),
			nn.BatchNorm2d(chout),
			nn.ReLU(),
		)
		self.mp = nn.MaxPool2d(2, return_indices=True)

	def forward(self, x):
		y = self.seq(x)
		pool_shape = y.shape
		y, indices = self.mp(y)
		return y, indices, pool_shape

class DoubleDown3(nn.Module):
	def __init__(self, chin, chout):
		super().__init__()
		self.seq = nn.Sequential(
			nn.Conv2d(chin, chout, 3, padding=1, bias=False),
			nn.BatchNorm2d(chout),
			nn.ReLU(),
			nn.Conv2d(chout, chout, 3, padding=1, bias=False),
			nn.BatchNorm2d(chout),
			nn.ReLU(),
			nn.Conv2d(chout, chout, 3, padding=1, bias=False),
			nn.BatchNorm2d(chout),
			nn.ReLU(),
		)
		self.mp = nn.MaxPool2d(2, return_indices=True)

	def forward(self, x):
		y = self.seq(x)
		pool_shape = y.shape
		y, indices = self.mp(y)
		return y, indices, pool_shape

class DoubleUp2(nn.Module):
	def __init__(self, chin, chout):
		super().__init__()
		self.seq = nn.Sequential(
			nn.Conv2d(chin, chin, 3, padding=1, bias=False),
			nn.BatchNorm2d(chin),
			nn.ReLU(),
			nn.Conv2d(chin, chout, 3, padding=1, bias=False),
			nn.BatchNorm2d(chout),
			nn.ReLU(),
		)
		self.mup = nn.MaxUnpool2d(2)
		
	def forward(self, x, indices, output_size):
		y = self.mup(x, indices, output_size=output_size)
		y = self.seq(y)
		return y

class DoubleUp3(nn.Module):
	def __init__(self, chin, chout):
		super().__init__()
		self.seq = nn.Sequential(
			nn.Conv2d(chin, chin, 3, padding=1, bias=False),
			nn.BatchNorm2d(chin),
			nn.ReLU(),
			nn.Conv2d(chin, chin, 3, padding=1, bias=False),
			nn.BatchNorm2d(chin),
			nn.ReLU(),
			nn.Conv2d(chin, chout, 3, padding=1, bias=False),
			nn.BatchNorm2d(chout),
			nn.ReLU(),
		)
		self.mup = nn.MaxUnpool2d(2)
		
	def forward(self, x, indices, output_size):
		y = self.mup(x, indices, output_size=output_size)
		y = self.seq(y)
		return y

class DoubleUp2Out(nn.Module):
	def __init__(self, chin, chout):
		super().__init__()
		self.seq = nn.Sequential(
			nn.Conv2d(chin, chin, 3, padding=1, bias=False),
			nn.BatchNorm2d(chin),
			nn.ReLU(),
			nn.Conv2d(chin, chout, 3, padding=1, bias=False),
			#nn.BatchNorm2d(chout),
			#nn.ReLU(),
		)
		self.mup = nn.MaxUnpool2d(2)
		
	def forward(self, x, indices, output_size):
		y = self.mup(x, indices, output_size=output_size)
		y = self.seq(y)
		return y

class Net(nn.Module):
	def __init__(self, base):
		super().__init__()
		self.bn_input = nn.BatchNorm2d(1)
		
		self.dc1 = DoubleDown2(1, base)
		self.dc2 = DoubleDown2(base, base*2)
		self.dc3 = DoubleDown3(base*2, base*4)
		self.dc4 = DoubleDown3(base*4, base*8)
		self.dc5 = DoubleDown3(base*8, base*8)
		
		self.uc5 = DoubleUp3(base*8, base*8)
		self.uc4 = DoubleUp3(base*8, base*4)
		self.uc3 = DoubleUp3(base*4, base*2)
		self.uc2 = DoubleUp2(base*2, base)
		self.uc1 = DoubleUp2Out(base, 1)

	def forward(self, batch: torch.Tensor):
		x = self.bn_input(batch)

		x, mp1_indices, shape1 = self.dc1(x)
		x, mp2_indices, shape2 = self.dc2(x)
		x, mp3_indices, shape3 = self.dc3(x)
		#x, mp4_indices, shape4 = self.dc4(x)
		#x, mp5_indices, shape5 = self.dc5(x)

		#x = self.uc5(x, mp5_indices, output_size=shape5)
		#x = self.uc4(x, mp4_indices, output_size=shape4)
		x = self.uc3(x, mp3_indices, output_size=shape3)
		x = self.uc2(x, mp2_indices, output_size=shape2)
		x = self.uc1(x, mp1_indices, output_size=shape1)
		
		return x

class SkipNet(nn.Module):
	def __init__(self, base):
		super().__init__()
		self.bn_input = nn.BatchNorm2d(1)
		
		self.dc1 = DoubleDown2(1, base)
		self.dc2 = DoubleDown2(base, base*2)
		self.dc3 = DoubleDown3(base*2, base*4)
		self.dc4 = DoubleDown3(base*4, base*8)
		self.dc5 = DoubleDown3(base*8, base*8)
		
		self.uc5 = DoubleUp3(base*8, base*8)
		self.uc4 = DoubleUp3(base*8, base*4)
		self.uc3 = DoubleUp3(base*4, base*2)
		self.uc2 = DoubleUp2(base*2, base)
		self.uc1 = DoubleUp2Out(base, 1)

	def forward(self, batch: torch.Tensor):
		x = self.bn_input(batch)

		x1, mp1_indices, shape1 = self.dc1(x)
		x2, mp2_indices, shape2 = self.dc2(x1)
		x3, mp3_indices, shape3 = self.dc3(x2)
		x4, mp4_indices, shape4 = self.dc4(x3)
		#x, mp5_indices, shape5 = self.dc5(x)

		#x = self.uc5(x, mp5_indices, output_size=shape5)
		x = self.uc4(x4, mp4_indices, output_size=shape4)
		x = self.uc3(x3 + x, mp3_indices, output_size=shape3)
		x = self.uc2(x + x2, mp2_indices, output_size=shape2)
		x = self.uc1(x + x1, mp1_indices, output_size=shape1)
		
		return x

# Metrics and Losses

def focal_loss(pred, target, alpha=0.3, gamma=2.0):
	bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
	pt  = torch.exp(-bce)
	return (alpha * (1 - pt)**gamma * bce).mean()

def binary_focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
	""" Focal loss for binary classification. """
	probs = torch.sigmoid(inputs)
	targets = targets.float()

	# Compute binary cross entropy
	bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

	# Compute focal weight
	p_t = probs * targets + (1 - probs) * (1 - targets)
	focal_weight = (1 - p_t) ** gamma

	alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
	bce_loss = alpha_t * bce_loss

	# Apply focal loss weighting
	loss = focal_weight * bce_loss

	return loss.mean()

def IoUMetric(pred, gt, softmax=False):
	# Run softmax if input is logits.
	if softmax is True:
		pred = nn.Softmax(dim=1)(pred)
	# end if
	
	# Add the one-hot encoded masks for all 3 output channels
	# (for all the classes) to a tensor named 'gt' (ground truth).
	gt = torch.cat([ (gt == i) for i in range(3) ], dim=1)
	# print(f"[2] Pred shape: {pred.shape}, gt shape: {gt.shape}")

	intersection = gt * pred
	union = gt + pred - intersection

	# Compute the sum over all the dimensions except for the batch dimension.
	iou = (intersection.sum(dim=(1, 2, 3)) + 0.001) / (union.sum(dim=(1, 2, 3)) + 0.001)
	
	# Compute the mean over the batch dimension.
	return iou.mean()

class IoULoss(nn.Module):
	def __init__(self, softmax=False):
		super().__init__()
		self.softmax = softmax
	
	# pred => Predictions (logits, B, 3, H, W)
	# gt => Ground Truth Labales (B, 1, H, W)
	def forward(self, pred, gt):
		# return 1.0 - IoUMetric(pred, gt, self.softmax)
		# Compute the negative log loss for stable training.
		return -(IoUMetric(pred, gt, self.softmax).log())

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
	# Average of Dice coefficient for all batches, or for a single mask
	assert input.size() == target.size()
	assert input.dim() == 3 or not reduce_batch_first

	sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

	inter = 2 * (input * target).sum(dim=sum_dim)
	sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
	sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

	dice = (inter + epsilon) / (sets_sum + epsilon)
	return dice.mean()

def dice_loss(input: Tensor, target: Tensor):
	# Dice loss (objective to minimize) between 0 and 1
	return 1 - dice_coeff(torch.sigmoid(input), target, reduce_batch_first=False)