import torch
import torch.nn.functional as F
import torch.nn as nn
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		
		#self.feature = feature_extractor
		self.fc1 = nn.Linear(512, 120)
		self.fc2 = nn.Linear(120, 1)
	def forward(self, x):
		#x = self.feature(x)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return torch.cat([x, -x], dim=1)
