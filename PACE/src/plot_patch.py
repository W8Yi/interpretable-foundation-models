from cProfile import label
from transformers import pipeline
from transformers import BertTokenizer, BertModel
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import load_dataset
import pickle
import os
from math import pi
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
from numpy import random
import scipy.sparse as sp 
from scipy.special import gammaln
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys, re, time, string
from scipy.special import gammaln, psi
from numpy.linalg import *
import math
import pandas as pd
from config import parser
import torchvision.transforms as transforms
import torchvision
from utils import accuracy_score, dirichlet_expectation, read_tsv_file, compute_metrics, Adam, posterior_mu, posterior_mu_sigma, vis, kmeans_init
from utils import  run_kmeans, ImageNetDataset, Cub2011, plot_topics
from model import PACE, ViTClassify
# from torchviz import make_dot
from utils import load_train_data, load_val_data, softmax, dirichlet_expectation
from torchvision.transforms.functional import InterpolationMode
# from captum.attr import Lime, LimeBase
from augment import relevance, faithfulness, contrastive_learning, contrative_transform, image_augment
# import wandb
from transformers import TrainingArguments, Trainer, AutoFeatureExtractor
from utils import MyImageDataset
from datasets import load_dataset
from evaluate_utils import stability, faithfulness, get_topics, coherence, diversity
import shap, lime
from utils import StanfordCars, MyImageDatasetFromStanfordCars, build_transform

args = parser.parse_args() 
args.save_path = os.path.join(args.save_path, args.name)
sample_path = os.path.join('../sample', args.name)

np.random.seed(args.seed)   
torch.manual_seed(args.seed)
random.seed(args.seed)

if args.task == 'flower102':
    dataset_name = "nelorth/oxford-flowers"
    dataset = load_dataset(dataset_name)

    extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    train_inputs = extractor(dataset['train']['image'], return_tensors="pt")
    test_inputs = extractor(dataset['test']['image'], return_tensors="pt")
    train_dataset = MyImageDataset(train_inputs['pixel_values'], dataset['train']['label'])
    test_dataset = MyImageDataset(test_inputs['pixel_values'], dataset['test']['label'])
    val_dataset = test_dataset
    args.out_dim = 102
elif args.task == 'cub2011':
    transform = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    train_dataset = Cub2011(args.data_path, train=True, transform=transform, download=False)
    val_dataset = Cub2011(args.data_path, train=False, transform=transform, download=False)
    test_dataset = val_dataset
    args.out_dim = 200


model = ViTClassify(in_dim = args.b_dim, out_dim=args.out_dim,hid_dim=args.c_dim, layer=args.layer)
#print(model)
#model.linear.load_state_dict(torch.load(load_path+'linear.pt'))
#model.classify.load_state_dict(torch.load(load_path+'classify.pt'))
model = model.cuda()
#model = nn.DataParallel(model, device_ids=[0,1,2,3])
#x = train_dataset['encodings'][0]
#x = torch.zeros((args.train_batch_size, 3, 224,224)).cuda()
#y = model(x)
#make_dot(y, params=dict(list(model.named_parameters()))).render("vit_torchviz", format="png")


if 'PACE' in args.name:
    PACE = PACE(d=args.c_dim,K=args.K,D=args.D,N=args.N,alpha=args.alpha,C = args.out_dim)
else:
    PACE = None

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=args.num_epoches,      # total number of training epochs
    per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.eval_batch_size,   # batch size for evaluation
    warmup_steps=0,                # number of warmup steps for learning rate scheduler     change steps from 100 to 0
    weight_decay=args.weight_decay,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    seed = args.seed,
    load_best_model_at_end=True,
    metric_for_best_model=args.metric, # 'eval_matthews_correlation' for cola, etc.
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate = args.lr,
    # report_to="wandb",
    #resume_from_checkpoint=True,
   # eval_steps=100,
)



test_set = DataLoader(val_dataset,batch_size=args.eval_batch_size,shuffle=False)

print('train size', len(train_dataset))
print('eval size', len(val_dataset))

#X0 = np.load(os.path.join(args.save_path, 'X-L-2.npy'))
#PACE._mus = PACE._mu0 = run_kmeans(X0, args.K)

print('evaluating')
    #model.load_state_dict(torch.load('../ckpt/bert-base' +'/' + args.task + '_' +'epoch10'+'_lr-3e-5.pt'))
model.load_state_dict(torch.load(args.save_path +'/' + args.task + '_' +'epoch'+str(args.num_epoches)+ '_L'+ str(args.layer)+'-MLP.pt'))
    #score = trainer.evaluate()
    #print('score', score)
#torch.save(model.linear.state_dict(), args.save_path +'/' + args.task + '_' +'linear-epoch'+str(args.num_epoches)+'.pt')
#torch.save(model.classify.state_dict(), args.save_path+'/'+ args.task + '_' +'classify-epoch'+str(args.num_epoches)+'.pt')

if PACE is not None:
    PACE._mus = np.load(args.save_path+'/' + args.task + '_'+'mus-epoch'+str(args.num_epoches)+ '_L'+ str(args.layer)+'-MLP.npy')
    PACE._sigmas = np.load(args.save_path+'/' + args.task + '_' +'sigmas-epoch'+str(args.num_epoches)+ '_L'+ str(args.layer)+'-MLP.npy')
    PACE._eta = np.load(args.save_path+'/' + args.task + '_'+'eta-epoch'+str(args.num_epoches)+ '_L'+ str(args.layer)+'-MLP.npy')





# temperarilly CPU bounded, instead of GPU-bounded, needs multi-thread if multiple run at the same time
# numpy matrix manipulation test

x = None
pos = []
topic = []
name = []
patch_img = []
full_img = []
word_embed = {}
word_cnt = {}
top_words = [{} for _ in range(args.K)] # maintain a priority queue of prob for tokens in each topic
pred_label = []
tok = []
font = []
topic_cnt = {}

for idx in range(args.K):  # args.K
        if PACE is None:
            continue
        name.append(0)
        topic.append('T_'+str(idx))
        #font.append(1) # np.exp(det(PACE._sigmas[idx]))
        if x is None:
            x = PACE._mus[idx].reshape(-1,args.c_dim)
        else:
            x = np.concatenate([x,PACE._mus[idx].reshape(-1,args.c_dim)],axis=0)
        patch_img.append(np.ones((224//16,224//16,3))) # patch_img[-1].shape
        full_img.append(np.ones((224,224,3)))  # full_img[-1].shape
        pos.append((-1,-1))

x = torch.Tensor(x).cuda()

topic_cnt = dict(sorted(topic_cnt.items(), key=lambda item: item[1],reverse=True))
print(topic_cnt)
concepts = [[] for _ in range(args.K)]
#top_topics = list(topic_cnt)[1:6]
top_topics = {}
tt_cp = [x for x in top_topics]
tw = {}


#top_topics = list(topic_cnt)[5:10]


topic_se = 24
class_1 = 10
class_2 = 20

sample_num = 2000
avg_corr = 0
batch_cnt = 0

# test metrics for LIME model
# ref https://captum.ai/api/lime.html


# interprete classifier from embedding inputs

concept_all = []
label_all = []

model.eval()
concept_test = []
concept_aug = []
pred_test = []
embeds = []
corpus = []
patches = []
attentions = []






with torch.no_grad():
    cnt = 0
    for id, inputs in enumerate(test_set):
        if len(name)>sample_num:
            continue
        test_encodings = inputs['encodings'].cuda()
        test_labels = inputs['labels'].cuda()
        #test_path = inputs['path']
        
        #test_mask = inputs['attention_mask'].cuda()
        #print(test_encodings.size())
        logits, states, att = model(test_encodings)

        # get augmented outputs
        image_trans = image_augment(inputs['encodings'])
        logits_trans, states_trans, att_trans = model(image_trans) 

        preds = logits.argmax(-1)
        #logits = logits.detach().cpu().numpy()
        
        for i in preds:
            pred_label.append(i)
        if PACE is None:
            continue

        
        gamma, phi = PACE.do_e_step(states, att[args.layer + 1]) # inference w/o learning, so e step instead of em step.
        gamma_trans, phi_trans = PACE.do_e_step(states_trans, att_trans[args.layer + 1])
        
        if True:
            num_samples = 100
            samples_idx = np.random.choice(states.shape[1], num_samples)
            for j in range(num_samples):
                embeds.append(states[0,samples_idx[j]].detach().cpu().numpy())
                px = 224//16*((samples_idx[j]+1)//16)
                py = 224//16*((samples_idx[j]+1)%16)
                pp = test_encodings[0,:,px:px + 224//16,py:py+224//16].permute(1,2,0).detach().cpu().numpy()
                patches.append(pp)
                #print('jj',samples_idx[j], att[args.layer+1].shape)
                attentions.append(att[args.layer + 1].mean(1)[0,0,samples_idx[j]].detach().cpu().numpy())
            if len(corpus)<25:
                num_samples = 25
                samples_idx = np.random.choice(states.shape[1], num_samples)
                clist = []
                for j in range(num_samples):
                    clist.append(states[0,samples_idx[j]].detach().cpu().numpy())
                corpus.append(clist)    

        print('corpus', len(corpus), len(corpus[0]))
        #pred_y = softmax(logits[0])
        #phi_mean = PACE._phi[0].mean(0)
        #log_p = PACE.log_p_y(logits)
        #print('pred_y', pred_y)
        #print('log_p', log_p)
        
        #corr = relevance(gamma, preds) # concept and tilde y
        #print(test_encodings.size(), preds.shape)
        # ablation, last few layers as inputs to lime
        
        # compute and print stability/faithfulness 
        
        
        #print(torch.linspace(0, N).cuda().long().view(1,N,1))
        #X= embed.view(B,-1)
        #feature_mask = torch.linspace(0, N, N).cuda().long().view(1,N,1).repeat(B,1,D).view(B,-1)
        #print(X.size(), feature_mask.size())
        #attr = lime.attribute(X, target=preds, feature_mask=feature_mask)
        #e = shap.GradientExplainer((model, embed), test_encodings)
        #e = shap.GradientExplainer(model, test_encodings)
       # shap_values,indexes = e.shap_values(test_encodings, ranked_outputs=2, nsamples=200)
        #print(shap_values)
        #corr = relevance(attr, preds) # concept and tilde y . different dimensions of atrr and gamma?
        #print('corr shape: ',corr.shape,'corr: ', corr.mean())
        #avg_corr += corr.mean()
        batch_cnt += 1
        concept_all.append(gamma)
        label_all.append(preds)
        
        # each seq 5 tokens 1:6, ten seq
# use gamma, phi (w mask) to collect topic words with prob
        wmflag = False
        for idm in range(preds.shape[0]):

            if idm>0:
                continue
            ttf = False
            # get patch concepts and represent them as images
            #seq_topic = np.argmax(phi[idm,:,:],-1).reshape(-1)
            seq_w = []
        #print(stop_topic.shape)
        #print('stop topic',stop_topic)
            anc = None
            gamma1 = np.zeros(args.K)
            gamma2 = np.zeros(args.K)
            full_seq = ''
            #t_select = torch.zeros(stop_x.size()[0],dtype=torch.int64)
           # for idx in range(50):
           #     full_seq += vocab[test_ids[idm,:].flatten()[idx]] + ' '
            
            for idx in range(states[idm].size()[0]):
                #if (idm+idx) % 100   != 5:
                #    continue
                #tt = phi[idm,stop_idx[idx],:].argsort()[-3:][::-1]
               # ww = vocab[test_ids[idm,:].flatten()[stop_idx[idx]]]
                tt = phi.argmax(-1)[idm,idx]
                if tt in [0]:
                    continue
                #tt=1
                ww = test_labels[idm].item()
                #print(test_encodings.size())
                px = 224//16*((idx+1)//16)
                py = 224//16*((idx+1)%16)
                pp = test_encodings[idm,:,px:px + 224//16,py:py+224//16].permute(1,2,0).detach().cpu().numpy()
                #print('patch',pp)
                #if tt != topic_se: # or ww not in [class_1, class_2]:
                #    continue
                if tt or PACE is None:
                    pos.append(((idx+1)//16,(idx+1)%16))
                    #topic.append(tt)
                    name.append(ww)
                    patch_img.append(pp)
                    full_img.append(test_encodings[idm,:,:,:].permute(1,2,0).detach().cpu().numpy())
                    if x is None:
                        x = states[idm,idx].reshape(-1,args.c_dim)
                    else:
                        x = torch.cat([x,states[idm,idx].reshape(-1,args.c_dim)],axis=0)

               # seq_w.append(ww)
            # plot the  image, patch with label, GT, attention weight
            # log image id in the jpg name
            #imgplot = plt.imshow(full_img[-1])
            #new_path = os.path.join(sample_path, test_path[idm].split('/')[0])
            #if not os.path.exists(new_path):
            #    os.makedirs(new_path)
            #plt.savefig(os.path.join(sample_path, '{}_{}_{}.png'.format(test_path[idm][:-4], name[-1],topic[-1])))
            # connot show on the remote server, use savefig instead
            #plt.show()
            
#print(tok) 
#print('topic',len(topic))

'''
concept_test = np.concatenate(concept_test,axis=0)
concept_aug = np.concatenate(concept_aug,axis=0)
pred_test = np.concatenate(pred_test,axis=0)
fscore = faithfulness(concept_test, pred_test, concept_test, pred_test)
sscore = stability(concept_test, concept_aug)
print('faithfulness', fscore)
print('stability', sscore)
'''
test_id = [idx for idx in range(len(pred_label))] 

topics, select_patches, select_attentions = get_topics(embeds, PACE._mus, patches, attentions)

'''
# remove patches with attention weights lower than 0.0015
for idx in range(len(select_patches)):
    select_patches[idx] = np.array(select_patches[idx])
    select_attentions[idx] = np.array(select_attentions[idx])
    topics[idx] = topics[idx][select_attentions[idx]>0.0015]
    select_patches[idx] = select_patches[idx][select_attentions[idx]>0.0015]
    select_attentions[idx] = select_attentions[idx][select_attentions[idx]>0.0015]
'''
            
div = diversity(topics)
print('div', div)

coh, indiv_coh = coherence(topics, corpus)
print('coh', coh)
print('indiv_coh', indiv_coh)

# top 4 topics in terms of indiv_coh, get original topic idex from indiv_coh
# trans list to dict
#indiv_coh = dict(zip(range(len(indiv_coh)),indiv_coh))
#top_topics = sorted(indiv_coh.items(), key=lambda item: item[1],reverse=True)[:4]

num_top = 10 # 4
top_topics = np.array(indiv_coh).argsort()[::-1][:num_top]
print('top_topics', top_topics)
print(np.array(indiv_coh).argsort()[::-1])



# get top 25 patches in top 25 topics by coherence
topic_patch = []
topic_attetion = []
print('topics', len(topics), len(topics[0]), topics[0][0].shape)
for tt in top_topics:
    topic_patch.append(select_patches[tt][:10])
    topic_attetion.append(select_attentions[tt][:10])
topic_patch = np.array(topic_patch)
print('topic_patch', topic_patch.shape)
plot_topics(topic_patch, topic_attetion)


name = name[:sample_num]
topic = topic[:sample_num]
patch_img = patch_img[:sample_num]
full_img = full_img[:sample_num]
pos = pos[:sample_num]
x = x[:sample_num,:].cpu()
#np.save(os.path.join(args.save_path, 'X.npy'),x)

for idx in range(len(name)):
    # tt is index where topic is nearest to x[idx]
    tt = np.argmin([np.linalg.norm(PACE._mus[i]-x[idx].detach().cpu().numpy()) for i in range(args.K)])
    topic.append(tt)

if PACE is not None:

    sigmas = [det(PACE._sigmas[i]) for i in range(args.K)]
    mus = [PACE._mus[i].mean() for i in range(args.K)]
    print('sigmas', sigmas)
    print('mus', mus)
    vis(x,name,topic,patch_img, pos, full_img, sigmas, div=div, coh=coh, top_topics=top_topics, indiv_coh=indiv_coh)
    for word in word_embed:
        word_embed[word] /= word_cnt[word]
        word_embed[word] = word_embed[word].cpu().detach().numpy()
        
    for idx in list(topic_cnt):
        top_words[idx] = sorted(word_embed.items(), key=lambda item: np.linalg.norm(PACE._mus[idx]-item[1]))
        print(idx,[x[0] for x in list(top_words[idx])[:15]])
# write predictions into tsv file, for submitting
print('pred', pred_label[:10])
if args.task == "rte":
    pred_label = [('entailment' if ll == 0 else 'not_entailment') for ll in pred_label]
elif args.task == 'stsb':
    pred_label = [np.round(ll*5,3) for ll in pred_label]
print('pred', pred_label[:10])


# sort concepts by distance to PACE mus
concepts = {k: sorted(v, key=lambda item: np.linalg.norm(PACE._mus[k]-item)) for k,v in enumerate(concepts)}
# save top 10 patches as images for each topic, in a dir named concepts

'''
for idx in range(10): # args.K
    new_path = os.path.join(sample_path, 'concepts', str(idx))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for idm in range(min(10,len(concepts[idx]))):

        # add coherence/diversity score to the image title, including avg scores on selected topics
        plt.imshow(concepts[idx][idm]) # .reshape(224,224,3)
        plt.savefig(os.path.join(new_path, '{}_{}.png'.format(idx,idm)))
        plt.show()
'''

fig = plt.figure(figsize=(20, 20))  # Adjust the size as needed
fig.suptitle(f'div {div}, coh {coh}', fontsize=20)  # Add your title here

for idx in range(10):  # args.K
    new_path = os.path.join(sample_path, 'concepts', str(idx))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for idm in range(min(10, len(concepts[idx]))):
        # add coherence/diversity score to the image title, including avg scores on selected topics
        ax = fig.add_subplot(10, 10, idx * 10 + idm + 1)  # Creates a subplot in the correct position
        ax.imshow(concepts[idx][idm])  # .reshape(224,224,3)
        ax.axis('off')  # Removes axis labels
        fig.savefig(os.path.join(new_path, '{}_{}.png'.format(idx, idm)))

plt.show()
