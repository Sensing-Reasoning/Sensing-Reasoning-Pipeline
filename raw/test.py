import numpy as np
import torch 
from model import NEURAL
from dataset import DataMain
import time


#device = torch.device('cuda:0'  if torch.cuda.is_available() else 'cpu')
lr_rate = 0.01
batch_size = 400
n_iters = 50000

def test(data, model, label_mapping, noise_sd):

	STOP_true_positive = 0
	STOP_false_positive = 0
	STOP_num = 0
	
	correct = 0
	tot = 0
	c0 = 0
	X,GT = data.sequential_test_batch()
	while X is not None:

		X = X.cuda()#to(device)
		X = X + torch.randn_like(X).cuda() * noise_sd
		GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
		#GT = GT.cuda()#to(device)
			
		Y = model(X)
		Y = torch.argmax(Y,dim=1)

		this_batch_size = len(Y)
		
		for i in range(this_batch_size):
			tot+=1

			if GT[i] == Y[i]:
				correct+=1

			if GT[i] == 0:
				STOP_num+=1
				if (GT[i] == Y[i]): c0 += 1
				if Y[i]==0:
					STOP_true_positive+=1
			elif Y[i]==0:
				STOP_false_positive+=1
			
		X,GT = data.sequential_test_batch()
	#print('acc = %d / %d = %.2f' % (correct,tot, 100*correct / tot))
	#print('TN = %d / %d = %.2f' % (c0, STOP_num, 100*c0 / STOP_num))
	#print('TP = %d / %d = %.2f' % (correct - c0, tot - STOP_num, 100. * (correct - c0) / (tot - STOP_num)))

	recall = STOP_true_positive/STOP_num
	if STOP_true_positive+STOP_false_positive == 0:
		precision = 0
	else:
		precision = STOP_true_positive/(STOP_true_positive+STOP_false_positive)
	#print('stop sign : recall = %d/%d = %f, precision = %d/%d = %f' % \
	#	(STOP_true_positive,STOP_num, recall, \
	#		STOP_true_positive,STOP_true_positive+STOP_false_positive, precision) )
	print(tot - STOP_num)
	return (correct - c0) / (tot - STOP_num), c0 / STOP_num





print('[Data] Preparing .... ')
data = DataMain(batch_size=batch_size)
data.data_set_up(istrain=True)
data.greeting()
print('[Data] Done .... ')


for sigma in [0.12, 0.25, 0.50]:
	print("sigma = %.2f" % sigma)
	print("Hierarchy => Main")

	mappings = np.array([
		[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]).astype(np.int64)

	for i in range(12):
		model = NEURAL(n_class=1,n_channel=3) 
		model.load_state_dict(torch.load("hier/model_%d_%.2f.pt" % (i, sigma)))
		model = model.cuda()#to(device)
		model.eval()

		label_mapping = mappings[i]
		tp, tn = test(data, model, label_mapping, sigma)
		print("%.3f %.3f" % (tp, tn))

	mappings = np.array([
		[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
		[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		[1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
		[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
		[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]).astype(np.int64)

	print("Main => Attribute")

	for i in range(13):
		model = NEURAL(n_class=1,n_channel=3) 
		model.load_state_dict(torch.load("attr/model_%d_%.2f.pt" % (i, sigma)))
		model = model.cuda()#to(device)
		model.eval()

		label_mapping = mappings[i]
		tp, tn = test(data, model, label_mapping, sigma)

		print("%.3f %.3f" % (tp, tn))
		#print("%d\t%.3f\t%.3f" % (i, tp, tn))
