import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)
        
    # pos_seq =  pos_seq = torch.arange(seq_len-1, -1, -1.0)
    def forward(self, pos_seq, batch_size=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[None,:,:].expand(batch_size,-1, -1)
        else:
            return pos_emb[:,None,:]


# module for self-attention, layernorm and linear transformation
class transformerEncoder(nn.Module):
    def __init__(self,net_arch,is_specific_net):
        super(transformerEncoder, self).__init__()
        self.net_arch=net_arch
        self.Q=nn.Linear(net_arch.h_dim, net_arch.h_dim)
        self.K=nn.Linear(net_arch.h_dim, net_arch.h_dim)
        self.V=nn.Linear(net_arch.h_dim, net_arch.h_dim)
        self.c_drop = nn.Dropout(net_arch.drop_rate)

        #feed forward layer
        self.ff1=nn.Linear(net_arch.h_dim, net_arch.ff_dim)
        self.ff2=nn.Linear(net_arch.ff_dim, net_arch.h_dim)
        self.f_drop = nn.Dropout(net_arch.drop_rate)
        self.relu=nn.ReLU()
        if is_specific_net:
            self.layerNorm1=nn.LayerNorm([net_arch.sent_length_bow,net_arch.h_dim])
            self.layerNorm2=nn.LayerNorm([net_arch.sent_length_bow,net_arch.h_dim])
        else:
            self.layerNorm1=nn.LayerNorm([net_arch.sent_length_idx,net_arch.h_dim])
            self.layerNorm2=nn.LayerNorm([net_arch.sent_length_idx,net_arch.h_dim])
        self.l1_drop = nn.Dropout(net_arch.drop_rate)
        self.l2_drop = nn.Dropout(net_arch.drop_rate)


        # initial
        nn.init.xavier_normal_(self.Q.weight, 1)
        nn.init.xavier_normal_(self.K.weight, 1)
        nn.init.xavier_normal_(self.V.weight, 1)
        nn.init.xavier_normal_(self.ff1.weight, 1)
        nn.init.xavier_normal_(self.ff2.weight, 1)

        self.multihead_attn=nn.MultiheadAttention(net_arch.h_dim,net_arch.head_num,net_arch.drop_rate)

    def forward(self,list_para):
        h_vec=list_para[0]
        mask=list_para[1]
        #attention
        Q=self.Q(h_vec).permute(1,0,2)# shape=[B, L, H]
        K=self.K(h_vec).permute(1,0,2)# shape=[B,H,L]
        V=self.V(h_vec).permute(1,0,2)# shape=[B, L, H]
        c_vec,_ = self.multihead_attn(Q, K, V,key_padding_mask=(mask))
        c_vec=c_vec.permute(1,0,2)
        c_vec = self.c_drop(c_vec)

        c_vec=self.layerNorm1(c_vec+h_vec)
        c_vec=self.l1_drop(c_vec)

        #fead forward
        f_vec = self.ff1(c_vec)
        f_vec = self.relu(f_vec)
        f_vec = self.f_drop(f_vec)
        f_vec = self.ff2(f_vec)

        f_vec=self.layerNorm2(f_vec+c_vec)
        f_vec=self.l2_drop(f_vec)

        return [f_vec,mask]

#MLP attention network
class MLP_attention(nn.Module):
    def __init__(self, net_arch):
        super(MLP_attention, self).__init__()
        self.q=nn.Parameter(torch.zeros(1,net_arch.h_dim))
        self.W1=nn.Linear(net_arch.h_dim,net_arch.h_dim)
        self.W2=nn.Linear(net_arch.h_dim,net_arch.h_dim)
        self.W=nn.Linear(net_arch.h_dim,1)
        self.f_drop = nn.Dropout(net_arch.drop_rate)


        # initial
        nn.init.xavier_normal_(self.W1.weight, 1)
        nn.init.xavier_normal_(self.W2.weight, 1)
        nn.init.xavier_normal_(self.W.weight, 1)
        nn.init.xavier_normal_(self.q, 1)
    def forward(self,c_vec,mask_1D):
        #c_vec:[B,L,H] mask_1D:[B,L,1]
        temp1=self.W1(c_vec)#[B,L,H]
        temp2=self.W2(self.q).unsqueeze(0)#[1,1,H]
        temp=self.W(F.tanh(temp1+temp2))#[B,L,1]
        a=F.softmax(temp+mask_1D,dim=1)##[B,L,1]
        fea_vec=torch.sum(torch.mul(c_vec,a),dim=1)
        fea_vec=self.f_drop(fea_vec)
        return fea_vec

class HAN(nn.Module):
    '''
    hierarchy attention network
    several self-attention + MLP-attention
    '''
    def __init__(self, net_arch,is_specific_net):
        super(HAN, self).__init__()
        self.net_arch=net_arch
        if is_specific_net:
            self.san=nn.Sequential(*[transformerEncoder(net_arch,is_specific_net) for i in range(net_arch.DSPWAN_layer_num)])
        else:
            self.san=nn.Sequential(*[transformerEncoder(net_arch,is_specific_net) for i in range(net_arch.SAN_layer_num)])
        self.embed2Hidden=nn.Linear(net_arch.in_dim,net_arch.h_dim)
        self.h_drop     = nn.Dropout(net_arch.drop_rate)
        nn.init.xavier_normal_(self.embed2Hidden.weight, 1)

    def forward(self, bow_vec,mask):
        hidden_vec=self.h_drop(self.embed2Hidden(bow_vec))
        c_vec,_=self.san([hidden_vec,mask])#[B,L,H]
        return c_vec

class TDAN(nn.Module):
    '''
    topic driven attention network
    can encode information from two field
    '''
    def __init__(self, net_arch):
        super(TDAN, self).__init__()
        self.net_arch=net_arch
        self.generator_semantics=HAN(net_arch,False)
        self.generator_specific=HAN(net_arch,True)
        self.mlp_attention_pooling1=MLP_attention(net_arch)
        self.mlp_attention_pooling2=MLP_attention(net_arch)
        self.mlp_attention1=MLP_attention(net_arch)
        self.mlp_attention2=MLP_attention(net_arch)
        
        self.classifier=nn.Linear(net_arch.h_dim, net_arch.num_class)
        self.topic2hidden1=nn.Linear(net_arch.num_topic, net_arch.h_dim)
        self.topic2hidden2=nn.Linear(net_arch.num_topic, net_arch.h_dim)
        self.concFeature2feature=nn.Linear(net_arch.h_dim*net_arch.module_num, net_arch.h_dim)

        #embedding
        self.embed=nn.Embedding(net_arch.num_vocub,net_arch.in_dim)
        self.embed.weight.requires_grad = False
        self.posiEmbed=PositionalEmbedding(net_arch.in_dim).to(device)
        pos_seq = torch.arange(net_arch.sent_length_idx-1, -1, -1.0).to(device)
        self.posi_embed=nn.Parameter(self.posiEmbed(pos_seq,batch_size=100)*0.1)
        self.posi_embed.requires_grad=False

        #dropout
        self.topic_drop1 = nn.Dropout(net_arch.drop_rate)
        self.topic_drop2 = nn.Dropout(net_arch.drop_rate)
        self.feature_drop = nn.Dropout(net_arch.drop_rate)
        self.embed_drop1 = nn.Dropout(net_arch.drop_rate)
        self.embed_drop2 = nn.Dropout(net_arch.drop_rate)
        self.embed_drop3 = nn.Dropout(net_arch.drop_rate)


        nn.init.xavier_normal_(self.classifier.weight, 1)
        nn.init.xavier_normal_(self.topic2hidden1.weight, 1)
        nn.init.xavier_normal_(self.topic2hidden2.weight, 1)
        nn.init.xavier_normal_(self.concFeature2feature.weight, 1)
        
        

    def forward(self, x,bow_specific):
        
        mask_1D=self.make_mask_1D(x,self.net_arch).to(x.device)
        mask_1D_long=self.make_mask_1D(x,self.net_arch,True).to(x.device)
        mask_for_san=(x==self.net_arch.paddingIdx)#mask for self-attention network

        mask_1D_s_bow=self.make_mask_1D(bow_specific,self.net_arch).to(bow_specific.device)
        mask_1D_s_bow_long=self.make_mask_1D(bow_specific,self.net_arch,True).to(bow_specific.device)
        mask_for_san_s_bow=(bow_specific==self.net_arch.paddingIdx)#mask for self-attention network

        x_embed=self.embed(x)
        posi_embed=self.posi_embed[:x.shape[0],:x.shape[1],:]
        embed=x_embed+posi_embed# shape=[B, L, H]
        embed=self.embed_drop1(embed)


        bow_specific_embed=self.embed_drop3(self.embed(bow_specific))

        c_vec1=self.generator_semantics(embed,mask_for_san)#[B,L,H]
        c_vec2=self.generator_specific(bow_specific_embed,mask_for_san_s_bow)

        pooling_vec1=self.mlp_attention_pooling1(c_vec1,mask_1D).unsqueeze(1)#[B,1,H]
        pooling_vec2=self.mlp_attention_pooling2(c_vec2,mask_1D_s_bow).unsqueeze(1)#[B,1,H]

        c_vec1=torch.cat([pooling_vec2,c_vec1],dim=1)
        c_vec2=torch.cat([pooling_vec1,c_vec2],dim=1)

        fea_vec1=self.mlp_attention1(c_vec1,mask_1D_long)
        fea_vec2=self.mlp_attention2(c_vec2,mask_1D_s_bow_long)

        fea_vec=torch.cat([fea_vec1,fea_vec2],dim=-1)
        fea_vec=self.feature_drop(self.concFeature2feature(fea_vec))
        classifier_pre_vec=F.softmax(self.classifier(fea_vec), dim=1)
        return classifier_pre_vec,fea_vec

    def make_mask_1D(self,x,net_arch,is_long_mask=False):
        mask = torch.full((x.shape[0], x.shape[1], 1), 0, dtype=torch.float32)
        mask[x==net_arch.paddingIdx]=float('-inf')
        if is_long_mask:
            add_dim=torch.zeros((x.shape[0], 1, 1), dtype=torch.float32)
            mask=torch.cat([add_dim,mask],dim=1)
        return mask
    def copyWordEmbed(self,idx2vecmatrix):
        self.embed.weight.data.copy_(idx2vecmatrix)
        pass

class Discriminator(nn.Module):
    def __init__(self,net_arch):
        super(Discriminator,self).__init__()
        self.w=nn.Linear(net_arch.h_dim,2)
        nn.init.xavier_normal_(self.w.weight, 1)
    def forward(self,x):
        output=self.w(x)
        output=F.softmax(output,dim=1)
        return output

class GRL(torch.autograd.Function):


    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None