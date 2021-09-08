import glob
import os
import shutil

path_train_m='./RACE/train/middle'
path_train_h='./RACE/train/high'
path_dev_m='./RACE/dev/middle'
path_dev_h='./RACE/dev/high'
path_test_m='./RACE/test/middle'
path_test_h='./RACE/test/high'

paths=[path_train_m,path_train_h,path_dev_m,path_dev_h,path_test_m,path_test_h]

for path in paths:
    filelists=os.listdir(path)
    print(path," --datasets-- ",len(filelists))
    num=0
    to_dir=list(path)
    to_dir.insert(2,'RE_SHAPE_')
    to_dir=''.join(to_dir)
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)

    # half=os.listdir(to_dir)
    # print("half datasets:  ",len(half))
    cut=0
    for f in filelists:
        num+=1
        if num%2==0:
            shutil.copy(os.path.join(path,f),os.path.join(to_dir,f))
            cut+=1

    print(to_dir," --cut_datasets-- ",cut)

# for path in [path_train_h,path_train_m]:
#     filenames=glob.glob(path+"/*txt")
