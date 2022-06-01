import configparser as cp
import glob
import itertools
import math
import os
import re
import sys
import tarfile
import tempfile
import utils

#This function extracts the angles from tar and prints them in a temp file 
def angles_to_temp(tar_obj, temp):

    with tar.extractfile(tar_obj) as reader:

        # ignore header
        for i in range(15):
            reader.readline()

        while True:
            line = reader.readline().decode("utf-8")
            if len(line) == 0:
                return
            temp.write('{} {} {}\n'.format(line[38:51].strip(), line[52:64].strip(), line[65:75].strip()))

#Reads the 3 angles form the temp files and creates a python list for each 
def read_angles_temp(tempfile):
    tempfile.seek(0)

    angles_1 = [] # phi (azimuth)
    angles_2 = [] # theta (elevation)
    angles_3 = [] # psi (rotation) 
    while True:
        line = tempfile.readline()
        if len(line) == 0:
            break
        a1,a2,a3 = line.split(' ')
        angles_1.append(float(a1))
        angles_2.append(float(a2))
        angles_3.append(float(a3))
    return (angles_1, angles_2, angles_3)

#Takes the list and creates a new file with the name given by the config with the processed angles
#If the flag of differences is on, it will compute the differences and allocate them on a different file 
def process_angles(conf, angles, conformation, ptype, compute_differences=False):
    idLength = conf['data']['fileIdLength']
    angles_file = conf['data']['eulerAngleFilePath']
    diff_file = conf['data']['degreeAngleFilePath']

    os.makedirs(os.path.dirname(angles_file), exist_ok=True)
    os.makedirs(os.path.dirname(diff_file), exist_ok=True)

    print(' reading raw angle file')
    raw_angles_1,raw_angles_2,raw_angles_3 = angles
    
    print(' writing processed angle file')
    with open(angles_file, 'a+') as angles:
        for a1_1, a1_2, a1_3, i1 in zip(raw_angles_1, raw_angles_2, raw_angles_3, itertools.count(0)):
            angles.write(str(a1_1)+'\t'+str(a1_2)+'\t'+str(a1_3)+'\t'+str(conformation)+'\t'+str(ptype)+'\t'+str(i1)+'\n')

    if compute_differences:
        print(' computing angle differences')

        angles_1 = [math.radians(a) for a in raw_angles_1]
        angles_2 = [math.radians(a) for a in raw_angles_2]
        sin_angle_2 = [math.sin(a) for a in angles_2]
        cos_angle_2 = [math.cos(a) for a in angles_2]

        print(' computing angle differences')
        with open(diff_file, 'w') as diffs:
            for r1_1, i1, sin_r1_2, cos_r1_2 in zip(angles_1, itertools.count(0), sin_angle_2, cos_angle_2):
                for r2_1, i2, sin_r2_2, cos_r2_2 in zip(angles_1, itertools.count(0), sin_angle_2, cos_angle_2):

                    # angular difference between spherical coordinates
                    cos_d = cos_r1_2*cos_r2_2*math.cos(r1_1 - r2_1) + sin_r1_2*sin_r2_2
                    if cos_d > 1:
                        cos_d = 1
                    elif cos_d < -1:
                        cos_d = -1
                    d = math.degrees(math.acos(cos_d))
                    diffs.write(('{:0'+str(idLength)+'d}\t{:0'+str(idLength)+'d}\t{:f}\n').format(i1, i2, d))


#Each image is saved in another directory with the correspondant name/ID, so there's consistency/link between the image and the angles 
def write_file(conf, dest, tar_obj, conformation, ptype):
    match = re.search(r'''ptm(\d+).tiff''', member.name)
    id = int(match.group(1))
    filename = utils.id_to_filename(id, conf, conformation, ptype)
    with tar.extractfile(member) as reader:
        content = reader.read()
        with open(dest+filename, 'wb') as f:
            f.write(content)


if __name__ == '__main__':

    if len(sys.argv) < 2 or sys.argv[1] == '--help' or sys.argv[1] == '-h':
        print('Usage: '+sys.argv[0]+' config_file')
        print('raw file specified by data > rawData field of config file')
        print('raw data should be a .tar.bz2 with the following files')
        print('  angle_files/*.doc - the XMIPP_STAR_1 output specifying the orienations')
        print('  <intensity>/*/ptm*.tiff - diffraction image files', flush=True)
        exit(1)

    confpath = sys.argv[1]
    conf = cp.ConfigParser()
    conf.read(confpath)

    compute_differences = '--compute-differences' in sys.argv[2:]

    raw_file = conf['data']['rawData'].split(';')
    for i, val in enumerate (raw_file):
        raw_file[i]=val.split(',')
    p_type = conf['data']['type'].split(',')
    conformations = conf['data']['conformations'].split(';')
    for i,val in enumerate(conformations):
        conformations[i]=val.split(',')
    intensity = conf['data']['intensity'].split(';')
    image_dest = conf['data']['dataPath']
    #Delete file with angles and conformations if there's one
    #os.remove(conf['data']['eulerAngleFilePath'])
    os.makedirs(image_dest, exist_ok=True)
    print('raw_file', raw_file)
    
    #Now we have different types of protein
    for j, ptype in enumerate(p_type):
        print('**** Extracting data for protein type ', ptype)
        #Multiple conformations from a same protein need to be processed
        #Per each file in the list we will:
        for i in range(0,len(conformations[j])):
            #Extract data from the tar to a temp file
            print('***Extracting data for conformation ', conformations[j][i])
            with tempfile.TemporaryFile(mode='w+') as angles_temp:
 
                print('Processing tar file', raw_file[j][i],flush=True)
                with tarfile.open(raw_file[j][i], mode='r:bz2') as tar:
                    print('tar file opened', flush=True)
                    for member in tar.getmembers():
                        if member.name.startswith(intensity[j]+'/'):
                            write_file(conf, image_dest, member, conformations[j][i], ptype)
                        elif member.name.startswith('angle_list/') and member.name.endswith('.doc'):
                            print('Found angle file '+member.name, flush=True)
                            angles_to_temp(member, angles_temp)
        
                angles = read_angles_temp(angles_temp)
                print('Processing angles', flush=True)
                process_angles(conf, angles, conformations[j][i], ptype, compute_differences)
    
