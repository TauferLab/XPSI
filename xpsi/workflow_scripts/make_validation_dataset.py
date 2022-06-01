import configparser as cp
import numpy as np
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split
import sys
import utils


args = sys.argv[1:]

conf = cp.ConfigParser()
if len(args) > 0:
    conf.read(args[0])
else:
    conf.read("configs/global.ini")

indices = range(int(conf['fullData']['dataSize']))
train_ind, valid_ind = train_test_split(indices, train_size=int(conf['data']['dataSize']), random_state=42)


euler_angles = np.loadtxt(conf['fullData']['eulerAngleFilePath'])


def copy_angles(path, angles):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as angle_file:
        for a1_1, a1_2, i1 in angles:
            angle_file.write(str(a1_1)+'\t'+str(a1_2)+'\t'+str(int(i1))+'\n')


copy_angles(conf['data']['eulerAngleFilePath'], euler_angles[train_ind, :])
copy_angles(conf['validationData']['eulerAngleFilePath'], euler_angles[valid_ind, :])

def move_files(conf, ids, dst_dir):
    src_dir = conf['fullData']['dataPath']
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    for i,j in zip(ids, range(len(ids))):
        name = utils.id_to_filename(i, conf)
        dst = dst_dir + utils.id_to_filename(j, conf)
        src = src_dir + utils.id_to_filename(i, conf)
        copyfile(src, dst)

move_files(conf, train_ind, conf['data']['dataPath'])
move_files(conf, valid_ind, conf['validationData']['dataPath'])


