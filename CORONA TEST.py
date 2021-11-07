#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import os


# In[32]:


# this function downloads photos according to their label 
# BU KOD datasetdan train(mashq qilish) va tekshirish(test)  papkalari orqali rasmlarni yuklab oladi 

import glob
training_data_dir = r'C:/Users/USER/datasets2021/chest_xray/training/'
test_data_dir = r'C:/Users/USER/datasets2021/chest_xray/test/'
def load_dataset(data_dir):
    photo_list=[]
    photo_type=['NORMAL','PNEUMONIA']
    print(photo_type)
    for i in photo_type:
        for file in glob.glob(os.path.join(data_dir,i,'*')): #  chest_xray/training(test)/NORMAL(PNEUMONIA)
            photo = cv2.imread(file)
            if photo is not None:
                photo_list.append((photo,i))
    return photo_list
images = load_dataset(training_data_dir)


# In[33]:


# NOMLARNI kamroq joy egallashi uchun raqamga almashtiradi
# changes label of files to numbers(to save memory)
def encode(label):
    if label=='PNEUMONIA':
        return 1
    else:
        return 0


# In[34]:


#resizes an image
# rasmni shaklini bir xil qilib beradi
def standartlash(photo,size):
    photo = cv2.resize(photo,size) 
    return photo


# In[35]:


# rasmni standart shaklga keltiradi
# returns a list of standart images
def standartlangan_list(photo_list):
    photostd_list=[]
    for i in photo_list:
        photostd_list.append((standartlash(i[0],(200,200)),encode(i[1])))
    return photostd_list
images = standartlangan_list(images)


# In[84]:


# returns an avarage arithmetic of an image
# orta arifmetigini topadi
def average(photo):
    return np.sum(photo[:,:,2]) / (photo.shape[0]*photo.shape[1])


# In[85]:


# rasmni orta arifmetigini solishtiradi
def choose(photo):
    aver=average(photo)
    if aver>50:
        return 1
    else:
        return 0


# In[86]:


test_photos = load_dataset(test_data_dir)
standart_test = standartlangan_list(test_photos)


# In[87]:


np.random.shuffle(standart_test)
def accuracy(photos):
    error = []
    for i in photos:
        photo = i[0]
        label = i[1]
        
        predict = choose(photo)
        if predict != label:
            error.append((photo,label,predict))
    return error


# In[89]:


error = accuracy(standart_test)
overall = len(standart_test)

accur = overall - len(error)

print("Accuracy rate = ", accur/overall)
print(len(error))

plt.figure(figsize=(20,16))
for i in range(len(error)):
    plt.subplot(9,4,i+1)
    plt.title(str(error[i][1]) + str(error[i][2]))
    plt.imshow(error[i][0])
plt.show()


# In[ ]:





# In[ ]:




