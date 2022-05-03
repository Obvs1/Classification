import os
import numpy as np
import shutil
import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-r", "--root",dest="data", help="path to model ")
args = parser.parse_args()
root_dir = args.model
classes_dir = os.listdir(root_dir)

train_ratio = 0.7


for cls in classes_dir:
    os.makedirs(root_dir +'train/' + cls, exist_ok=True)
    os.makedirs(root_dir +'test/' + cls, exist_ok=True)
    #os.makedirs(input_destination +'val_ds/' + cls, exist_ok=True)
    
    # for each class, let's counts its elements
    src = root_dir + cls
    allFileNames = os.listdir(src)

    # shuffle it and split into train/test
    np.random.shuffle(allFileNames)
    train_FileNames, test_FileNames= np.split(np.array(allFileNames),[int(train_ratio * len(allFileNames))])
    
    # save their initial path
    train_FileNames = [src+'/'+ name  for name in train_FileNames.tolist()]
    test_FileNames  = [src+'/' + name for name in test_FileNames.tolist()]
    
    print("\n *****************************",
          "\n Total images: ",cls, len(allFileNames),
          '\n Training: ', len(train_FileNames),
          '\n Testing: ', len(test_FileNames),
          '\n *****************************')
    
    # copy files from the initial path to the final folders
    for name in train_FileNames:
      shutil.move(name, root_dir +'train/' + cls)
    for name in test_FileNames:
      shutil.move(name, root_dir +'test/' + cls)
   


# checking everything was fine
paths = ['train/', 'test/']
for p in paths:
  for dir,subdir,files in os.walk(root_dir + p):
    print(dir,' ', p, str(len(files)))