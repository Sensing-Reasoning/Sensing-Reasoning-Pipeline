import numpy as np
import torch 
from model import NEURAL
from dataset import DataMain
import time


#device = torch.device('cuda:0'  if torch.cuda.is_available() else 'cpu')
lr_rate = 0.01
batch_size = 200
n_iters = 50000
noise_sd = 0.50

def test(data, model, label_mapping):

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
	print('acc = %d / %d = %.2f' % (correct,tot, 100*correct / tot))
	print('TN = %d / %d = %.2f' % (c0, STOP_num, 100*c0 / STOP_num))
	print('TP = %d / %d = %.2f' % (correct - c0, tot - STOP_num, 100. * (correct - c0) / (tot - STOP_num)))

	recall = STOP_true_positive/STOP_num
	if STOP_true_positive+STOP_false_positive == 0:
		precision = 0
	else:
		precision = STOP_true_positive/(STOP_true_positive+STOP_false_positive)
	print('stop sign : recall = %d/%d = %f, precision = %d/%d = %f' % \
		(STOP_true_positive,STOP_num, recall, \
			STOP_true_positive,STOP_true_positive+STOP_false_positive, precision) )
	
	return (correct - c0) / (tot - STOP_num), correct / tot





print('[Data] Preparing .... ')
data = DataMain(batch_size=batch_size)
data.data_set_up(istrain=True)
data.greeting()
print('[Data] Done .... ')


print('[Model] Preparing .... ')
model = NEURAL(n_class=1,n_channel=3) 
model = model.cuda()#to(device)
print('[Model] Done .... ')

loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, weight_decay = 1e-4)

st = time.time()

model.train()
max_acc = 0
cand_tn = 0
save_n = 0
stable_iter = 0

from matplotlib import pyplot as plt

sensor_id = 7

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


label_mapping = mappings[sensor_id]

from torch.nn import CrossEntropyLoss

W = 10
print('[Training] Starting ...')
for i in range(n_iters):
	X,GT = data.random_train_batch()
	X = X.cuda()#to(device)
	#cnt = np.zeros(12)
	#for j in range(X.size(0)):
	#	cnt[GT[j].item()] += 1
	#if (np.min(cnt) >= 1):
	#	cnt = np.zeros(12)
	#	for j in range(X.size(0)):
	#		cnt[GT[j].item()] += 1
	#		if (cnt[GT[j].item()] == 1):
	#			print(np.min(X[j].cpu().numpy()), np.max(X[j].cpu().numpy()))
	#			plt.imshow(torch.permute(X[j] + 0.5, (1, 2, 0)).cpu().numpy())
	#			plt.savefig("img%d.pdf" % (GT[j].item()))
	#	break

	X = X + torch.randn_like(X).cuda() * noise_sd
	
	GT = torch.from_numpy(label_mapping[GT.numpy()]).cuda()
	#GT = GT.cuda()#to(device)
	#print(X.shape)
	Y = model(X)
	weight = torch.ones(GT.shape)
	weight[GT == 1] = W
	weight = weight.cuda()

	loss = torch.mean(CrossEntropyLoss(reduce=False)(Y,GT) * weight)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	now = (time.time() - st) / 60.0

	if (i+1) % 1000 == 0 :
	#if (i+1) % 10 == 0 :
		print(' ### Eval ###')
		print('Time = %f minutes, Iter = %d/%d, Loss = %f' % (now,i+1,n_iters,loss))
		model.eval()
		
		tp, tn = test(data,model,label_mapping)

		if (tp>max_acc+1e-7 or (tp >= max_acc - 1e-7 and tn > cand_tn + 1e-7)):
			print('[save]..')
			max_acc = tp
			cand_tn = tn
			stable_iter = 0
			print(tp, tn)
			torch.save(model.state_dict(), "./tpenhance/model_%d_%.2f_%d.pt" % (sensor_id, noise_sd, W))# + str(save_n) + "_acc=%f.ckpt"%(score))
			#save_n+=1
		else:
			stable_iter += 1
			if stable_iter == 10:
				print('Stable ... Training END ..')
				break
		model.train()

print(max_acc, cand_tn)
print('[Training] Done ...')
