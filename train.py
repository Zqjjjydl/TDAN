import parameter
import torch
from dataLoader import labeledDataset
from torch.utils.data import Dataset, DataLoader
from model import TDAN,GRL,Discriminator
import torch.optim as optim
from parameter import domain_class,domain_class_name
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

args=parameter.parse_args()
dictionary=torch.load("./processedData/dictionary")

net_arch = args
net_arch.num_vocub=len(dictionary["token2id"])
net_arch.paddingIdx=dictionary["token2id"]["<padding>"]
wordvec_matrix=dictionary['idx2vec']
source_domain=domain_class.index(args.source_domain)
target_domain=domain_class.index(args.target_domain)
source_domain_name=domain_class_name[source_domain]
target_domain_name=domain_class_name[target_domain]
bow=torch.load("./processedData/TAN_input_"+source_domain_name+target_domain_name)

source_labeled_dataDealer = labeledDataset(source_domain,bow["specific_bow_source"])
target_labeled_dataDealer = labeledDataset(target_domain,bow["specific_bow_target"])

idx1=random.sample(range(3000),500)
idx2=random.sample(range(3000,6000),500)
idx3=[i for i in range(0,3000) if i not in idx1]
idx4=[i for i in range(3000,6000) if i not in idx2]
target_labeled_dataDealer_eval = labeledDataset(target_domain,\
    bow["specific_bow_target"],1,idx1,idx2)
target_labeled_dataDealer_test = labeledDataset(target_domain,\
    bow["specific_bow_target"],2,idx3,idx4)

source_labeled_loader = DataLoader(dataset=source_labeled_dataDealer,
                          batch_size=args.batch_size,
                          shuffle=True)
target_labeled_loader = DataLoader(dataset=target_labeled_dataDealer,
                          batch_size=args.batch_size,
                          shuffle=True)
target_labeled_loader_eval = DataLoader(dataset=target_labeled_dataDealer_eval,
                          batch_size=args.batch_size,
                          shuffle=True)
target_labeled_loader_test = DataLoader(dataset=target_labeled_dataDealer_test,
                          batch_size=args.batch_size,
                          shuffle=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

net=TDAN(net_arch)
net.copyWordEmbed(wordvec_matrix)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  net = nn.DataParallel(net)
net=net.to(device)

discriminator=Discriminator(net_arch)
discriminator=discriminator.to(device)

grl=GRL()

optimizer = optim.Adam(net.parameters(), args.lr1, weight_decay=args.wd)
optimizer_dis = optim.Adam(discriminator.parameters(), args.lr2, weight_decay=args.wd)



crossEntropyLoss=nn.CrossEntropyLoss()
loss_fn_kl = F.kl_div 

best_preci_eval=0
best_preci_eval_ever=0
best_idx=-1

eval_results=[]

for epoch in range(args.num_epoch):
    source_labeled_iter = iter(source_labeled_loader)
    target_labeled_iter = iter(target_labeled_loader)
    step_len=len(source_labeled_iter)
    loss1=0
    loss2=0
    adaptation_rate=min(net_arch.adaptation_rate,2/(1+np.exp(-10*((epoch+1)/args.num_epoch))))
    net.train()
    for step in range(step_len):
        #train
        optimizer.zero_grad()#set all gradient to zero
        optimizer_dis.zero_grad()

        source_labeled_specific_bow,source_labeled_idx, \
            source_labeled_label=source_labeled_iter.next()
        target_labeled_specific_bow, \
            target_labeled_idx,_=target_labeled_iter.next()

        #put to gpu
        source_labeled_specific_bow\
            ,source_labeled_idx, \
                source_labeled_label\
                =source_labeled_specific_bow.to(device)\
                    ,source_labeled_idx.to(device), source_labeled_label.to(device)

        target_labeled_specific_bow\
            ,target_labeled_idx\
                =target_labeled_specific_bow.to(device)\
                    ,target_labeled_idx.to(device)
        source_labeled_len=source_labeled_idx.shape[0]
        target_labeled_len=target_labeled_idx.shape[0]
        source_len=source_labeled_len
        target_len=target_labeled_len

        bow_specific=torch.cat([source_labeled_specific_bow,target_labeled_specific_bow],dim=0)
        sent=torch.cat([source_labeled_idx,target_labeled_idx],dim=0)

        class_pre,feature_vec=net(sent,bow_specific)

        class_pre=class_pre[0:source_labeled_len]
        label=source_labeled_label
        label_=torch.unsqueeze(label, 1)
        one_hot_label = torch.FloatTensor(label.shape[0], net_arch.num_class).zero_().to(device)
        one_hot_label.scatter_(1, label_, 1)
        cls_loss=loss_fn_kl(class_pre.log(),one_hot_label, reduction='batchmean')

        
        feature_vec=grl.apply(feature_vec,adaptation_rate)
        domain_pred=discriminator(feature_vec)
        zero_tensor=torch.zeros((source_len,),dtype=torch.long)
        one_tensor=torch.full((target_len,),1,dtype=torch.long)
        domain_label=torch.cat([zero_tensor,one_tensor],dim=0).to(device)
        dis_loss=crossEntropyLoss(domain_pred,domain_label)

        loss=cls_loss+args.lambda1*dis_loss
        loss.backward() 
        optimizer.step()
        loss1+=cls_loss.item()
        loss2+=args.lambda1*dis_loss.item()

        optimizer_dis.step()



    loss1/=step_len
    loss2/=step_len
    print("L_CLS:",loss1,"L_DIS",loss2)
    torch.cuda.empty_cache()
    net.eval()
    correct_rate_eval=0
    length=0
    for data in target_labeled_loader_eval:
        target_labeled_specific_bow\
            ,target_labeled_idx,target_labeled_label=data

        target_labeled_specific_bow\
            ,target_labeled_idx,\
                target_labeled_label\
                =target_labeled_specific_bow.to(device)\
                    ,target_labeled_idx.to(device),\
                        target_labeled_label.to(device)
        
        
        pred,_=net(target_labeled_idx\
            ,target_labeled_specific_bow)
        pred=torch.argmax(pred,dim=1)
        torch.sum(pred==target_labeled_label)
        correct_rate_eval+=(torch.sum(pred==target_labeled_label)).item()
    correct_rate_eval/=len(target_labeled_dataDealer_eval)

    correct_rate_test=0
    for data in target_labeled_loader_test:
        target_labeled_specific_bow\
            ,target_labeled_idx,target_labeled_label=data

        target_labeled_specific_bow\
            ,target_labeled_idx,\
                target_labeled_label\
                =target_labeled_specific_bow.to(device)\
                    ,target_labeled_idx.to(device),\
                        target_labeled_label.to(device)
        
        
        pred,_=net(target_labeled_idx\
            ,target_labeled_specific_bow)
        pred=torch.argmax(pred,dim=1)
        torch.sum(pred==target_labeled_label)
        correct_rate_test+=(torch.sum(pred==target_labeled_label)).item()
    correct_rate_test/=len(target_labeled_dataDealer_test)

    #perform early stop if neccesary
    eval_results.append(correct_rate_eval)
    if(len(eval_results)>8):
        latest_eval_result=eval_results[-8:]
        flag=True
        for i in range(len(latest_eval_result)-1):
            if latest_eval_result[i]<latest_eval_result[i+1]:
                flag=False
                break
        if flag:
            break

    print("eval correctness rate: ",correct_rate_eval,"test correctness rate: ",correct_rate_test)
    if correct_rate_eval>best_preci_eval:
        best_preci_eval=correct_rate_eval
        best_preci_test=correct_rate_test
        best_idx=epoch
        torch.save(net.state_dict(), './model/model'+args.source_domain+args.target_domain+'.pkl')

    if correct_rate_eval>best_preci_eval_ever:
        best_preci_eval_ever=correct_rate_eval
with open("./result/log_"+args.source_domain+args.target_domain,"w") as f:
    print("at epoch ",best_idx," precision is ",best_preci_test,file=f)


 