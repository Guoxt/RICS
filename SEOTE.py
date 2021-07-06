#!/usr/bin/env python
#coding:utf8
import nibabel as nib
import os
import numpy as np
import random
import torch


###############
###parameter### 
###############
patch = np.zeros((4,128,128,128))
s = 2
n = 1
sita_x = 32
sita_y = 32
sita_z = 32

###
class dataread(data.Dataset):
    def __init__(self, root, transforms = None, train = True, test = False, val = False):
        self.test = test
        self.train = train
        self.val = val

        if self.train:
            self.root = self.train
            self.folderlist = os.listdir(os.path.join(self.root))
        elif self.val:
            self.root = self.val
            self.folderlist = os.listdir(os.path.join(self.root))
        elif self.test:
            self.root = self.test
            self.folderlist = os.listdir(os.path.join(self.root))

    def __getitem__(self,index):
          
        if self.train: 
        
            ###
            out = np.zeros((patch.shape))
    
            flag = 'error'
            while flag == 'error':
                flag = '!error'
                location = np.zeros((n,3))
                for i in range(n):
                    location[i,:] = [random.randint(sita_x,patch.shape[1]-sita_x),random.randint(sita_y,patch.shape[2]-sita_y),random.randint(sita_z,patch.shape[3]-sita_z)]
        
                x_list = []
                y_list = []
                z_list = []
                for i in range(n):    
                    x_list.append(location[i,0].astype(int))  
                    y_list.append(location[i,1].astype(int)) 
                    z_list.append(location[i,2].astype(int))
                #print(x_list,y_list,z_list)
            
                x_list.sort()
                y_list.sort()
                z_list.sort()
    
                for i in range(n):
                    if i == 0:
                        if x_list[i] - 0 < sita_x:
                            flag = 'error'
                        if y_list[i] - 0 < sita_y:
                            flag = 'error'
                        if z_list[i] - 0 < sita_z:
                            flag = 'error'
                    if i == n-1:
                        if patch.shape[1] - x_list[i] < sita_x:
                            flag = 'error'
                        if patch.shape[2] - y_list[i] < sita_y:
                            flag = 'error'
                        if patch.shape[3] - z_list[i] < sita_z:
                            flag = 'error'          
                    if i > 0:
                        if x_list[i] - x_list[i-1] < sita_x:
                            flag = 'error'
                        if y_list[i] - y_list[i-1] < sita_y:
                            flag = 'error'
                        if z_list[i] - z_list[i-1] < sita_z:
                            flag = 'error'
                        
            x_list.insert(0,0)
            y_list.insert(0,0)
            z_list.insert(0,0)
            x_list.append(patch.shape[1])
            y_list.append(patch.shape[2])
            z_list.append(patch.shape[3])
                
            if flag != 'error':
    
                for i in range(len(x_list)-1):
                    for ii in range(len(y_list)-1):
                        for iii in range(len(z_list)-1):
                            if i == len(x_list)-2: 
                                x1 = x_list[i] 
                                x2 = x_list[i]+x_list[i+1]-x_list[i] 
                            else:
                                x1 = x_list[i] 
                                x2 = x_list[i]+x_list[i+1]-x_list[i] - s
                            
                            if ii == len(y_list)-2: 
                                y1 = y_list[ii] 
                                y2 = y_list[ii]+y_list[ii+1]-y_list[ii] 
                            else:
                                y1 = y_list[ii] 
                                y2 = y_list[ii]+y_list[ii+1]-y_list[ii] - s  
                                
                            if iii == len(z_list)-2: 
                                z1 = z_list[iii] 
                                z2 = z_list[iii]+z_list[iii+1]-z_list[iii]
                            else:
                                z1 = z_list[iii] 
                                z2 = z_list[iii]+z_list[iii+1]-z_list[iii] - s 
                                
                            fflag = random.random()
                            if fflag < 0.2:
                                img_temp = np.load(self.root+self.folderlist[random.randint(0,len(self.folderlist)-1)])
                                #print(img_temp.shape)
                                lx = random.randint(0, img_temp.shape[1]-(x2-x1))
                                ly = random.randint(0, img_temp.shape[2]-(y2-y1))
                                lz = random.randint(0, img_temp.shape[3]-(z2-z1)) 
                                while(np.sum(img_temp[3,lx:lx+(x2-x1),ly:ly+(y2-y1),lz:lz+(z2-z1)]==1)==0):
                                    lx = random.randint(0, img_temp.shape[1]-(x2-x1))
                                    ly = random.randint(0, img_temp.shape[2]-(y2-y1))
                                    lz = random.randint(0, img_temp.shape[3]-(z2-z1)) 
                                out[:,x1:x2,y1:y2,z1:z2] = img_temp[:,lx:lx+(x2-x1),ly:ly+(y2-y1),lz:lz+(z2-z1)] 
                            if 0.2 <= fflag < 0.4:
                                img_temp = np.load(self.root+self.folderlist[random.randint(0,len(self.folderlist)-1)])
                                #print(img_temp.shape)
                                lx = random.randint(0, img_temp.shape[1]-(x2-x1))
                                ly = random.randint(0, img_temp.shape[2]-(y2-y1))
                                lz = random.randint(0, img_temp.shape[3]-(z2-z1)) 
                                while(np.sum(img_temp[3,lx:lx+(x2-x1),ly:ly+(y2-y1),lz:lz+(z2-z1)]==2)==0):
                                    lx = random.randint(0, img_temp.shape[1]-(x2-x1))
                                    ly = random.randint(0, img_temp.shape[2]-(y2-y1))
                                    lz = random.randint(0, img_temp.shape[3]-(z2-z1)) 
                                out[:,x1:x2,y1:y2,z1:z2] = img_temp[:,lx:lx+(x2-x1),ly:ly+(y2-y1),lz:lz+(z2-z1)] 
                            if 0.4<= fflag < 0.6:
                                img_temp = np.load(self.root+self.folderlist[random.randint(0,len(self.folderlist)-1)])
                                #print(img_temp.shape)
                                lx = random.randint(0, img_temp.shape[1]-(x2-x1))
                                ly = random.randint(0, img_temp.shape[2]-(y2-y1))
                                lz = random.randint(0, img_temp.shape[3]-(z2-z1)) 
                                while(np.sum(img_temp[3,lx:lx+(x2-x1),ly:ly+(y2-y1),lz:lz+(z2-z1)]==3)==0):
                                    lx = random.randint(0, img_temp.shape[1]-(x2-x1))
                                    ly = random.randint(0, img_temp.shape[2]-(y2-y1))
                                    lz = random.randint(0, img_temp.shape[3]-(z2-z1)) 
                                out[:,x1:x2,y1:y2,z1:z2] = img_temp[:,lx:lx+(x2-x1),ly:ly+(y2-y1),lz:lz+(z2-z1)] 
                            if 0.6<= fflag < 0.8:
                                img_temp = np.load(self.root+self.folderlist[random.randint(0,len(self.folderlist)-1)])
                                #print(img_temp.shape)
                                lx = random.randint(0, img_temp.shape[1]-(x2-x1))
                                ly = random.randint(0, img_temp.shape[2]-(y2-y1))
                                lz = random.randint(0, img_temp.shape[3]-(z2-z1)) 
                                while(np.sum(img_temp[3,lx:lx+(x2-x1),ly:ly+(y2-y1),lz:lz+(z2-z1)]==4)==0):
                                    lx = random.randint(0, img_temp.shape[1]-(x2-x1))
                                    ly = random.randint(0, img_temp.shape[2]-(y2-y1))
                                    lz = random.randint(0, img_temp.shape[3]-(z2-z1)) 
                                out[:,x1:x2,y1:y2,z1:z2] = img_temp[:,lx:lx+(x2-x1),ly:ly+(y2-y1),lz:lz+(z2-z1)] 
                            if 0.8<= fflag:
                                img_temp = np.load(self.root+self.folderlist[random.randint(0,len(self.folderlist)-1)])
                                #print(img_temp.shape)
                                lx = random.randint(0, img_temp.shape[1]-(x2-x1))
                                ly = random.randint(0, img_temp.shape[2]-(y2-y1))
                                lz = random.randint(0, img_temp.shape[3]-(z2-z1)) 
                                while(np.sum(img_temp[3,lx:lx+(x2-x1),ly:ly+(y2-y1),lz:lz+(z2-z1)]==5)==0):
                                    lx = random.randint(0, img_temp.shape[1]-(x2-x1))
                                    ly = random.randint(0, img_temp.shape[2]-(y2-y1))
                                    lz = random.randint(0, img_temp.shape[3]-(z2-z1)) 
                                out[:,x1:x2,y1:y2,z1:z2] = img_temp[:,lx:lx+(x2-x1),ly:ly+(y2-y1),lz:lz+(z2-z1)] 
                                                
            img_out = out[0:3,:,:,:].astype(float)
            label_out = out[3,:,:,:].astype(float)
            #print(out.shape)
            img = torch.from_numpy(img_out).float()        
            label = torch.from_numpy(label_out).long()
                
        elif self.val:
            path = self.root
            img = np.load(os.path.join(path,self.folderlist[index]))
            img = np.asarray(img)
            img_out = img[0:3,:,:,:].astype(float)
            label_out = img[3,:,:,:].astype(float)
            #print(img.shape)
            img = torch.from_numpy(img_out).float()     
            label = torch.from_numpy(label_out).long()
        else:
            print('###$$$$$$$$$$$$$$$$$$$^^^^^^^^^^^^^')     

        return img, label

    def __len__(self):
        return len(self.folderlist)







