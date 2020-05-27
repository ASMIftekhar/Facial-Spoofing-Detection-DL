import argparse
import ouluLoader as dt
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from network import cnn_lstm as eff_nt

from tqdm import tqdm



csv_file_tr='OULU_train.csv'
csv_file_te='OULU_test.csv'
root_dir_tr='/media/data/spoof_data/Train_files'
root_dir_te='/media/data/spoof_data/Test_files'


#Creating the results directory##########

try:
  os.mkdir('results/')
except OSError as exc:
  pass
####################################################################



pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
### Inputs ####
parser=argparse.ArgumentParser()
parser.add_argument('-e','--number_of_epochs',type=int,required=False,default=100,help='Number_of_Epochs')
parser.add_argument('-l','--learning_rate',type=float,required=False,default=1e-4,help='Learning_Rate')
parser.add_argument('-ba','--batch_size',type=int,required=False,default=4,help='Batch_Size')
parser.add_argument('-fps','--frames_per_video',type=int,required=False,default=10,help='Frames_per_video')
parser.add_argument('-sh','--shape', nargs='+',default=[200,200],help='Input shape to the network use  W,H as two spced integers like 128 128. Use only even numbers')
parser.add_argument('-c','--Check_point',nargs='+',required=False,default='best',help='CheckPointName')
parser.add_argument('-fw','--first_word',type=str,required=False,default='you_dont_put_a_name',help='Name_of_the_folder_you_want_to_save')
parser.add_argument('-lw','--load_word',type=str,required=False,default='best',help='Name_of_the_folder_you_want_to_load')
parser.add_argument('-r','--resume_model',type=str,required=False,default='f',help='Resume The Model')
parser.add_argument('-i','--only_inference',type=str,required=False,default='f',help='For doing only inference no training')
parser.add_argument('-nw','--number_of_workers',type=int,required=False,default=4,help='Number_of_workers')
parser.add_argument('-h_l','--hyper_load',type=str,required=False,default='f',help='If this flag is t then the model will load stored hyper parameters')
#################
args=parser.parse_args()
batch_size=args.batch_size
frames_ps=args.frames_per_video
shape=(int(args.shape[0]),int(args.shape[1]))
n_wor=args.number_of_workers
learning_rate=args.learning_rate
resume=args.resume_model
inf=args.only_inference
epoch_num=args.number_of_epochs
folder_name='results'+'/'+args.first_word 
hyp=args.hyper_load
#import pdb;pdb.set_trace()

###### Fixing Seed #################
seed=10
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
def _init_fn(worker_id):
    np.random.seed(int(seed))
##############################





#import pdb;pdb.set_trace()
##############################
frozen_endpoints=(
#    #### EfficientNet#####
        'pretrain._conv_stemweight',
        'pretrain._bn0weight',
        'pretrain._bn0bias',
        'pretrain._blocks0',
        'pretrain._blocks1',
        'pretrain._blocks2',
        'pretrain._blocks3',
        'pretrain._blocks4',
        'pretrain._blocks5',
        'pretrain._blocks6',
        'pretrain._blocks7',
        'pretrain._blocks8',
        'pretrain._blocks9',
        'pretrain._blocks10',
        'pretrain._blocks11',
        'pretrain._blocks12',
        'pretrain._blocks13',
        'pretrain_blocks14',
)
###########################

#Creating the folder where the results would be stored##########

try:
  os.mkdir(folder_name)
except OSError as exc:
  pass
file_name_result=folder_name+'/'+'best.json'

####################################################################


####Initilizing the network ####
#######Network################################
net=eff_nt()

log_loss_best=100
#import pdb;pdb.set_trace()
    
net=nn.DataParallel(net)
  
#################################

#######Freezing Some Parameters########
trainables=[]
not_trainables=[]
for name, p in net.named_parameters():
    
    #import pdb;pdb.set_trace()
    #print(name)
    #freeze=name.split('.')[1]+'.'+name.split('.')[2]
    try:
        freeze=name.split('.')[1]+'.'+name.split('.')[2]+name.split('.')[3]
    except:
        pass
    if freeze in frozen_endpoints:
      #import pdb;pdb.set_trace()
        p.requires_grad=False  
        not_trainables.append(p)
    else:
        #print(name)
        trainables.append(p)

#import pdb;pdb.set_trace()
###################################################
######Optimizer#######
optimizer = optim.Adam([{"params":trainables,"lr":learning_rate}],betas=(0.9, 0.999), eps=1e-07, weight_decay=0, amsgrad=False)
########################

#rnn=nn.DataParallel(rnn)
net.cuda()
### Scheuduler ######
lambda1 = lambda epoch: 1.0 if epoch < 2 else (0.01 if epoch < 2 else 0.001)  
scheduler=optim.lr_scheduler.LambdaLR(optimizer,[lambda1])
###############################3


#### Saving CheckPoint##########
def save_checkpoint(state,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
###################################

##### Dataloader ###############
dataloader=dt.run(csv_file_tr,root_dir_tr,frames_ps,csv_file_te,root_dir_te,batch_size,n_wor)
#dataloader=dt.excution(crop)

## Defining the Loss ####
loss = nn.BCEWithLogitsLoss(reduction='mean')
sigmoid=nn.Sigmoid()
present_epoch=0


########Loading the Model###############

#import pdb;pdb.set_trace()
if resume=='t':
    try:
        print("Loading Our pretrained Model")
        checkpoint=torch.load(folder_name+'/'+args.load_word+'checkpoint.pth.tar')
        net.load_state_dict(checkpoint['state_dict'],strict=True)
        loss_pre_test=checkpoint['loss_best_test']
        present_epoch=checkpoint['epoch']
        print("Loading Done,Best Loss:{}, Current Epoch:{}".format(loss_pre_test,present_epoch))
    except:
        print("Cant Load the model")
        import pdb;pdb.set_trace()
#import pdb;pdb.set_trace()
#######################

########Loading Hyperparameters###############
if hyp=='t':
    try:
        print('Loading previous Hyperparameters')
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    except:
        print('Failed to load previous Hyperparameters')

################################################
if inf!='f':
    print("only doing inference from a preloaded model")
    epoch_num=present_epoch+1
else:
    epoch_num=present_epoch+epoch_num





def run():
    try:
        loss_best_test=loss_pre_test
        loss_te_mean=loss_pre_test
    except:
        loss_best_test=100
        loss_te_mean=100

    best_epoch_test=present_epoch
    best_epoch_train=0
  
    loss_best_train=100
    loss_tr_mean=100

    plot_train=[]
    plot_test=[]
    
    for epoch in range(present_epoch,epoch_num):
        loss_tr=0
        N_tr=0
        optimizer.zero_grad()
        if inf=='f':
            for iteration,i in enumerate(tqdm(dataloader['train'])):
                phase='train'          
                net.train()
                input=i['image'].cuda().float()
                labels=i['label'].cuda().float()
          
                #####Input and Output######  
                with torch.set_grad_enabled(phase=='train'):
                    #input=i['data'].cuda()
                    N_tr+=len(input)
                    output=net(input,frames_ps)
                    logloss=loss(output,labels)
                    lossf=logloss
                    lossf.backward()
                    loss_tr+=lossf.item()     
                    if (iteration+1)%1==0:
                        #if (iteration+1)%5==0:
                        optimizer.step()           
                        optimizer.zero_grad()

                loss_tr_mean=(loss_tr)/(iteration+1)
            
                data=[('Loss Train',loss_tr_mean),('Loss Test',loss_te_mean),("Cureent Loss",lossf.item())]
                data_best=[("Best Loss Train",loss_best_train),("Best Loss Test",loss_best_test),("Best Epoch(For Loss in Test)",best_epoch_test),("Best Epoch (For Loss in Train)",best_epoch_train)]
                info_sum=[('Phase',phase),('Epoch',epoch+1),('Iteration',iteration+1),('Predictions',np.around(sigmoid(output).data.cpu().numpy(),6)),('GD',labels.data.cpu().numpy())]
            
                res_sum=pd.DataFrame(data,columns=['Name', "Value"])
                res_best=pd.DataFrame(data_best,columns=['Name', "Value"])
                res_info=pd.DataFrame(info_sum,columns=['Name', "Value"])
                combine=pd.concat([res_sum,res_best],axis=1)
                print('\n',res_info,'\n',combine)
            #import pdb;pdb.set_trace()
            
            plot_train.append(loss_tr_mean)  
            if loss_tr_mean< loss_best_train:
                loss_best_train= (loss_tr)/(iteration+1)
                best_epoch_train=epoch+1
            #########################


        preds=[]
        gd=[]
        loss_te=0
        for iteration,i in enumerate(tqdm(dataloader['test'])):
            phase='test'
            net.eval()
        
            input=i['image'].cuda().float()
            labels=i['label'].cuda().float()
            all_gds=labels.data.cpu().numpy()
            #####Input and Output######  
            with torch.no_grad():
                #N_te+=len(input)
                output=net(input,frames_ps)
                for inn,val in enumerate(sigmoid(output).data.cpu().numpy()):
                    preds.append(val)
                    gd.append(all_gds[inn])

                logloss=loss(output,labels)
            
                lossf=logloss
                loss_te+=lossf.item()
                loss_te_mean=(loss_te)/(iteration+1)


            data=[('Loss Train',loss_tr_mean),('Loss Test',loss_te_mean),("Cureent Loss",lossf.item())]
            data_best=[("Best Loss Train",loss_best_train),("Best Loss Test",loss_best_test),("Best Epoch(For Loss in Test)",best_epoch_test),("Best Epoch (For Loss in Train)",best_epoch_train)]
            info_sum=[('Phase',phase),('Epoch',epoch+1),('Iteration',iteration+1),('Predictions',np.around(sigmoid(output).data.cpu().numpy(),6)),('GD',labels.data.cpu().numpy())]
            
            res_sum=pd.DataFrame(data,columns=['Name', "Value"])
            res_best=pd.DataFrame(data_best,columns=['Name', "Value"])
            res_info=pd.DataFrame(info_sum,columns=['Name', "Value"])
            combine=pd.concat([res_sum,res_best],axis=1)
            print('\n',res_info,'\n',combine)
      
      #log_loss_best  
        plot_test.append(loss_te_mean)
        if loss_te_mean< loss_best_test and inf=='f':
            loss_best_test=loss_te_mean
        
            best_epoch_test=epoch+1
            save_checkpoint({'epoch': epoch + 1,'state_dict': net.state_dict(),
                                'loss_best_test': loss_best_test,
                                'optimizer' : optimizer.state_dict(),
                                'scheduler':scheduler.state_dict()
                        
                        },filename=folder_name+'/'+'bestcheckpoint.pth.tar')
          
        if inf=='f':
            with open(folder_name+'/'+'plot.json','w') as fp:
                json.dump([plot_train,plot_test,learning_rate],fp)
           
        elif inf=='t':
            preds_bin=np.where(np.array(preds)>0.5,1,0)
            accr=accuracy_score(gd, preds_bin)
            ap=average_precision_score(gd,preds)
            
            print("Accuracy: {},AP: {}".format(accr,ap))
            #import pdb;pdb.set_trace()
            gd_s=[k.item() for k in gd]
            preds_s=[k.item() for k in preds]
            with open(folder_name+'/'+'predictions.json','w') as fp:
                json.dump([gd_s,preds_s],fp)

    #  submission_df = pd.DataFrame({"Ground_Truth":gd, "label": preds})
    #  submission_df.to_csv("submission.csv", index=False)
    scheduler.step()


if __name__=='__main__':
  run()
