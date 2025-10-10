#!/usr/bin/env python3

import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
# import time


def align_molecule(mobile,target):
    target_origin=np.average(target,axis=0)
    mobile_origin=np.average(mobile,axis=0)
    temp_target=np.copy(target)-target_origin
    temp_mobile=np.copy(mobile)-mobile_origin
    rot, rssd = R.align_vectors(temp_target, temp_mobile, return_sensitivity=False)
    newcoords=rot.apply(temp_mobile)+target_origin
    return newcoords


def read_pdb(filename,deletewater=False):

    def generate_residue_labels(data):
        N=np.size(data["chainID"])
        labels=data["chainID"].astype(object)  + \
               np.full(shape=N,fill_value="_") + \
               data["resname"].astype(object)  + \
               np.full(shape=N,fill_value="_") + \
               data["resid"].astype(str)
        return labels

    with open(filename) as inpfile:
        lines=inpfile.readlines()
    lines=[line for line in lines if (line.startswith("ATOM") or line.startswith("HETATM"))]
    if deletewater:
        lines=[line for line in lines if "HOH" not in line]
    data={}
    data["chainID"] = np.array([line[21] for line in lines])
    data["resname"] = np.array([line[17:21].strip() for line in lines])
    data["coord"]   = np.array([[line[30:38].strip(),line[38:46].strip(),line[46:54].strip()] for line in lines]).astype(np.double)
    data["resid"]   = np.array([line[6:11].strip() for line in lines]).astype(np.int32)
    data["name"]    = np.array([line[13:17].strip() for line in lines])
    data["labels"]  = generate_residue_labels(data)
    return data

def test_order(data1,data2):
    N = np.size(data1.keys())
    are_the_same=True
    keys=list(data1.keys())
    cnt=0
    while cnt<N and are_the_same==True:
        key=keys[cnt]
        if len(np.shape(data1[key]))==1:
            if ~(data1[key]==data1[key]).all():
                are_the_same=False
        cnt+=1
    return are_the_same


def get_mask(pdbdata,selection="all"):

    def find_protein(resnames):
        aminoacids=["Ala", "Arg", "Asn", "Asp", "Cys",
                    "Gln", "Glu", "Gly", "His", "Ile",
                    "Leu", "Lys", "Met", "Phe", "Pro",
                    "Ser", "Thr", "Trp", "Tyr", "Val"]
        aminoacids=[AA.upper() for AA in aminoacids]
        mask=np.full(shape=(len(resnames)),fill_value=False).astype(bool)
        for name in aminoacids:
            mask[resnames==name]=True
        return mask

    def mask_by_name(all_names,allowed_names):
        mask=np.full(shape=(len(all_names)),fill_value=False).astype(bool)
        for name in allowed_names:
            mask[all_names==name]=True
        return mask

    if selection=="all":
        mask=np.full(shape=(len(pdbdata["name"])),fill_value=True).astype(bool)
    elif selection=="protein":
        mask = find_protein(pdbdata["resname"])
    elif selection=="ligand":
        mask = np.invert(find_protein(pdbdata["resname"]))
    elif selection=="backbone":
        mask_name    = mask_by_name(pdbdata["name"],["C","CA","N","O"])
        mask_protein = find_protein(pdbdata["resname"])
        mask = mask_name & mask_protein
    else:
        print("Unrecognized selection")
        quit()
    return mask


def calculate_RMSD(coords1,coords2):
    return np.sqrt(np.average(np.sum(np.power(coords1-coords2,2),axis=1)))


def calculate_RMSD_per_residue(reference_geom,analyzed_geom):

    unique_labels=np.unique(reference_geom["labels"])
    RMSDs=np.zeros_like(unique_labels)
    weights=np.zeros_like(unique_labels).astype(np.int16)

    for index,label in enumerate(unique_labels):
        coords1 = reference_geom["coord"][reference_geom["labels"]==label]
        coords2 = analyzed_geom["coord"][analyzed_geom["labels"]==label]
        RMSDs[index]=calculate_RMSD(coords1,coords2)
        weights[index]=np.size(coords1,axis=0)

    return RMSDs, weights

if __name__=="__main__":

    if len(sys.argv)<3:
        print("Usage: {} [reference_filename] [analyzed_filename]".format(sys.argv[0]))
        quit()
    reference_filename = sys.argv[1]
    analyzed_filename  = sys.argv[2]

    reference_geom = read_pdb(reference_filename)
    analyzed_geom  = read_pdb(analyzed_filename)

    if test_order(reference_geom,analyzed_geom)!=True:
        print("The structures are not in the same order")
        quit()

    selection_mask = get_mask(reference_geom,selection="backbone")

    aligned_coords=align_molecule(analyzed_geom["coord"][selection_mask],
                                  reference_geom["coord"][selection_mask])
    RMSD_global=calculate_RMSD(reference_geom["coord"][selection_mask],aligned_coords)
    print(RMSD_global)
    


