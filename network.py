import torch
import torch.nn as nn
import time
import torchvision.models as models
import os
import numpy as np
from efficientnet_pytorch import EfficientNet



print("loading backbone network")
start=time.time()
model=EfficientNet.from_pretrained("efficientnet-b0")
print("loading done")





class Flatten(nn.Module):
   def __init__(self):
        super(Flatten,self).__init__()
 
   def forward(self, x):
        return x.view(x.size()[0], -1)



class cnn_lstm(nn.Module):
  def __init__(self):
    super(cnn_lstm,self).__init__()
    self.pretrain =model
    self.prelstm=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)) ,Flatten(),)
    self.lstm=nn.LSTM(1280,512,3,batch_first=True)

    self.classification=nn.Sequential(
                            nn.Dropout(0.9),
                            #nn.Linear(1024,512),
                            #nn.Sigmoid(), 
                            nn.Linear(512,1),
                            )    
  def forward(self,input,frames_per_video):
    stacked_input=[]
    lstms=[]
    start=0
    select_f=16
    for inn,frames in enumerate(input):
      stacked_input.append(frames)  
    features=self.pretrain.extract_features((torch.cat(stacked_input)))
    features=self.prelstm(features)
        
    self.lstm.flatten_parameters()
    for inn,sin_feat in enumerate(input):
      try:  
        all_output,hid=self.lstm(features[start:start+frames_per_video].unsqueeze(0))
      except:
        import pdb;pdb.set_trace()

      #print(all_output[-1].shape)

      #lstms.append(all_output[:,-1,:])
      lstms.append(hid[0][-1])
      start+=frames_per_video
    
    

    output=self.classification(torch.cat(lstms)).squeeze()
    return output





if __name__=='__main__':  
    
  network=rnn_paper(1024,1024).cuda()
  inputs=torch.rand(1,10,3,224,224).float().cuda()
  import pdb;pdb.set_trace()
  
  out_vgg=network(inputs,np.array([[10,10]])) 
  







    
