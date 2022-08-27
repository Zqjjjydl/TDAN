# -*- coding: utf-8 -*-
import argparse
domain_class=["b","d","e","k","y"]
domain_class_name=["books","dvd","electronics","kitchen","yelp"]
def parse_args():
    parser = argparse.ArgumentParser(description='Topic Driven Adaptive Network for Cross-Domain Sentiment Classification')

    # dataset
    parser.add_argument('--num_class',  default=2,     type=int,   help='number of classes')
    parser.add_argument('--sent_length_idx', default=500, type=int, help='idx sengtence length')
    parser.add_argument('--sent_length_bow', default=176, type=int, help='bow sengtence length')

    # model arguments
    parser.add_argument('--in_dim',     default=300,   type=int,   help='size of input word vector')
    parser.add_argument('--module_num',     default=2,   type=int,   help='module num')
    parser.add_argument('--h_dim',      default=304,   type=int,   help='size of hidden unit')
    parser.add_argument('--ff_dim',      default=300,   type=int,   help='size of ff layer in transformer')
    parser.add_argument('--num_topic',  default=50,    type=int,   help='number of topics')
    parser.add_argument('--drop_rate',   default=0.25, type=float, help='dropout rate')
    parser.add_argument('--head_num',   default=4, type=int, help='head num')
    parser.add_argument('--DSPWAN_layer_num',   default=3, type=int, help='DSPWAN layer num')
    parser.add_argument('--SAN_layer_num',   default=6, type=int, help='SAN layer num')
    
    

    # training arguments
    parser.add_argument('--lambda1',          default=1,  type=float, help='the weight for discriminator loss')
    parser.add_argument('--num_epoch',  default=50,    type=int,   help='number of total epochs to run')
    parser.add_argument('--batch_size', default=20,    type=int,   help='batchsize for optimizer updates')
    parser.add_argument('--lr1',         default=0.00002, type=float, help='the generator learning rate')
    parser.add_argument('--lr2',         default=0.00002, type=float, help='the discriminator learning rate')
    parser.add_argument('--wd',         default=5e-5,  type=float, help='weight decay')
    parser.add_argument('--adaptation_rate',   default=0.1,  type=float)

    # domain setting
    parser.add_argument('--source_domain',  default='y',    type=str,   help='source domain')
    parser.add_argument('--target_domain', default='b',    type=str,   help='target domain')

    args = parser.parse_args()
    return args
