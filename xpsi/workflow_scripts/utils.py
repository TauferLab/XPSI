#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

def id_to_filename(id, conf, conformation, ptype):
    """Computes the filename for the input file with the given id"""
    file_prefix = conf['data']['filePrefix']
    file_extension = conf['data']['fileExtension']
    file_id_length = int(conf['data']['fileIdLength'])
    padding_length = file_id_length - len(str(id))
    return str(ptype)+'_'+str(conformation)+'_'+file_prefix +("0" * padding_length) + str(id) + file_extension


# Returns the angle in radians between euclidian angle pairs 'a1' and 'a2'
# euclidian angle pairs are of the form (phi, theta) and measured in degrees
#When we predict the third angle, we compute the difference
def angular_difference(a1, a2):
    phi1 = math.radians(a1[0])
    theta1 = math.radians(a1[1])
    phi2 = math.radians(a2[0])
    theta2 = math.radians(a2[1])
    dphi = phi1 - phi2
    dtheta = theta1 - theta2
    hav_diff = math.sin(dtheta/2)**2 + math.sin(theta1) * math.sin(theta2) * math.sin(dphi/2)**2
    if (len(a2)==2):
       return 2*math.degrees(math.asin(math.sqrt(hav_diff))),0
    else:
       return 2*math.degrees(math.asin(math.sqrt(hav_diff))), abs(a1[2]-a2[2]) 
