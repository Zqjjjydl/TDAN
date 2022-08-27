from pickle import FALSE
from torch.utils.data import Dataset
import torch
from parameter import domain_class,domain_class_name
import random


class labeledDataset(Dataset): 
    def __init__(self,domain,specific_bow,isTest=0,idx1=None,idx2=None):
        super(labeledDataset,self).__init__()
        self.specific_bow=torch.tensor(specific_bow,dtype=torch.int)
        dataidx=torch.load('./processedData/dataInIdx')
        self.dataInIdx=torch.tensor(dataidx['labeled_text_inIdx'][domain])#[documentNum,maxDocumentLen]
        t=torch.load('./processedData/label')
        self.target=torch.tensor(t['label'][domain],dtype=torch.int64)
        
        if isTest==1:#eval set
            self.specific_bow=torch.cat([self.specific_bow[idx1],self.specific_bow[idx2]],dim=0)
            self.dataInIdx=torch.cat([self.dataInIdx[idx1],self.dataInIdx[idx2]],dim=0)
            self.target=torch.cat([self.target[idx1],self.target[idx2]],dim=0)
        elif isTest==2:#test set
            self.specific_bow=torch.cat([self.specific_bow[idx1],self.specific_bow[idx2]],dim=0)
            self.dataInIdx=torch.cat([self.dataInIdx[idx1],self.dataInIdx[idx2]],dim=0)
            self.target=torch.cat([self.target[idx1],self.target[idx2]],dim=0)
    def __getitem__(self, index):
        return self.specific_bow[index],\
            self.dataInIdx[index],self.target[index]
    def __len__(self): 
        return len(self.dataInIdx)
