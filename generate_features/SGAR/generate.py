import numpy as np
import sys,os, gc
import csv, glob
import os.path as path
import torch, pickle
from models import *
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from Bio.PDB import *
import argparse

def get_finetune_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, help='PDB PATH')
    parser.add_argument('--save_path', type=str,
                        help='save file')
    parser.add_argument('--csv', type=str, help='used csv file')
    parser.add_argument('--workdir', type=str, help='foldx out path')
    
    args = parser.parse_args()
    return args

def gen_graph_data(pdbfile, mutinfo, interfile,  cutoff, if_info=None):
    max_dis = 12
    pdbfile = open(pdbfile)
    lines = pdbfile.read().splitlines()
    chainid = [x.split('_')[0] for x in mutinfo]
    interface_res = read_inter_result(interfile,if_info, chainid)
    if len(interface_res)==0: print('Warning: We do not find any interface residues between the two parts: {}. Please double check your inputs. Thank you!'.format(if_info))
    sample = build_graph(lines, interface_res,mutinfo, cutoff,max_dis)
    return sample

def read_inter_result(path, if_info=None, chainid=None, old2new=None):
    if if_info is not None:
        info1 = if_info.split('_')
        pA = info1[0]
        pB = info1[1]
        mappings = {}
        for a in pA:
            for b in pB:
                if a not in mappings:
                    mappings[a] = [b]
                else:
                    mappings[a] += [b]
                if b not in mappings:
                    mappings[b] = [a]
                else:
                    mappings[b] += [a]


        target_chains = []
        for chainidx in chainid:
            if chainidx in mappings:
                target_chains += mappings[chainidx]

        target_inters = []
        for chainidx in chainid:
            target_inters += ['{}_{}'.format(chainidx,y) for y in target_chains]+\
                    ['{}_{}'.format(y,chainidx) for y in target_chains]

        target_inters =list(set(target_inters))
    else:
        target_inters = None

    inter = open(path)
    interlines = inter.read().splitlines()
    interface_res = []
    for line in interlines:
        iden = line[:3]
        if target_inters is None:
            if iden.split('_')[0] not in chainid and iden.split('_')[1] not in chainid:
                continue
        else:
            if iden not in target_inters:
                continue
        infor = line[4:].strip().split('_')#  chainid, resid
        assert len(infor)==2
        interface_res.append('_'.join(infor))

    if old2new is not None:
        mapps = {x[:-4] :y[:-4]  for x,y in old2new.items()}
        interface_res = [mapps[x] for x in interface_res if x in mapps]

    return interface_res

def build_graph(lines, interface_res, mutinfo, cutoff=3, max_dis=12, noisedict = None):
    atomnames = ['C','N','O','S']
    residues = ['ARG','MET','VAL','ASN','PRO','THR','PHE','ASP','ILE',\
            'ALA','GLY','GLU','LEU','SER','LYS','TYR','CYS','HIS','GLN','TRP']
    res_code = ['R','M','V','N','P','T','F','D','I',\
            'A','G','E','L','S','K','Y','C','H','Q','W']
    res2code ={x:idxx for x,idxx in zip(residues, res_code)}

    atomdict = {x:i for i,x in enumerate(atomnames)}
    resdict = {x:i for i,x in enumerate(residues)}
    V_atom = len(atomnames)
    V_res = len(residues)

    # build chain2id
    chain2id = []
    interface_coordinates = []
    line_list= []
    mutant_coords = []
    for line in lines:
        if line[0:4] == 'ATOM':
            atomname = line[12:16].strip()
            elemname = list(filter(lambda x: x.isalpha(), atomname))[0]
            resname  = line[16:21].strip()
            chainid =  line[21]
            res_idx = line[22:28].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            if elemname not in atomdict:
                continue

            coords = torch.tensor([x,y,z])
            atomid = atomdict[elemname]
            if resname not in resdict:
                resname = resname[1:]
            if resname not in resdict:
                continue

            if chainid not in chain2id:
                chain2id.append(chainid)
 
            line_token = '{}_{}_{}_{}'.format(atomname,resname,chainid, res_idx)
            if line_token not in line_list:
                line_list.append(line_token)
            else:
                continue

            resid  = resdict[resname]
            cr_token = '{}_{}'.format(chainid, res_idx)
            float_cd  = [float(x) for x in coords]
            cd_tensor = torch.tensor(float_cd)
            if cr_token in interface_res:
                interface_coordinates.append(cd_tensor)
            if mutinfo is not None and cr_token in mutinfo:
                interface_coordinates.append(cd_tensor)
                interface_res.append(cr_token)
                mutant_coords.append(cd_tensor)

    inter_coors_matrix = torch.stack(interface_coordinates)
    chain2id = {x:i for i,x in enumerate(chain2id)}
    global_resid2noise = {}

    n_features = V_atom+V_res+1+1+3 +1 +1+1+1
    line_list= []
    atoms = []
    flag_mut = False
    res_index_set = {}
    for line in lines:
        if line[0:4] == 'ATOM':
            features = [0]*n_features
            atomname = line[12:16].strip()
            elemname = list(filter(lambda x: x.isalpha(), atomname))[0]
            resname  = line[16:21].strip()
            chainid =  line[21]
            res_idx = line[22:28].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            if elemname not in atomdict:
                continue

            coords = torch.tensor([x,y,z])
            atomid = atomdict[elemname]
            if resname not in resdict:
                resname = resname[1:]
            if resname not in resdict:
                continue
            line_token = '{}_{}_{}_{}'.format(atomname,resname,chainid, res_idx)
            if line_token not in line_list:
                line_list.append(line_token)
            else:
                continue

            resid  = resdict[resname]
            features[atomid] = 1
            features[V_atom+resid] = 1

            cr_token = '{}_{}'.format(chainid, res_idx)
            float_cd  = [float(x) for x in coords]
            cd_tensor = torch.tensor(float_cd)
            #24
            if cr_token in interface_res:
                features[V_atom+V_res] = 1
            
            if mutinfo is not None:
                for inforrr in mutinfo:
                    mut_chainid = inforrr.split('_')[0]
                    if chainid==mut_chainid:
                        #25
                        features[V_atom+V_res+1] = 1
            #26
            features[V_atom+V_res+2] = chain2id[chainid]

            #27
            if cr_token not in res_index_set:
                res_index_set[cr_token] = len(res_index_set)+1

            features[V_atom+V_res+3] = res_index_set[cr_token]

            #28
            if atomname=='CA':
                features[V_atom+V_res+4] = res_index_set[cr_token]
                if noisedict is not None and cr_token in noisedict:
                    global_resid2noise[res_index_set[cr_token]] = noisedict[cr_token]

            flag = False
            dissss = torch.norm(cd_tensor-inter_coors_matrix,dim=1)
            flag = (dissss<max_dis).any()


            #29-31
            features[V_atom+V_res+5:V_atom+V_res+8] = float_cd

            res_iden_token = '{}_{}_{}'.format(chainid, res_idx, resname).upper()
            if  mutinfo is not None and cr_token in mutinfo:
                #32
                features[V_atom+V_res+8]=1
                flag_mut = True
                flag = True
            
            if flag:
                atoms.append(features)

    if mutinfo is not None and len(interface_res)>0:
        assert flag_mut==True
    
    if len(atoms)<5:
        return None
    atoms = torch.tensor(atoms, dtype=torch.float)
    N = atoms.size(0)
    atoms_type = torch.argmax(atoms[:,:4],1)
    atoms_type = atoms_type.unsqueeze(1).repeat(1,N)
    edge_type = atoms_type*4+atoms_type.t()

    pos  = atoms[:,-4:-1] #N,3
    row = pos[:,None,:].repeat(1,N,1)
    col = pos[None,:,:].repeat(N,1,1)
    direction = row-col
    del row, col
    distance = torch.sqrt(torch.sum(direction**2,2))+1e-10
    distance1 = (1.0/distance)*(distance<float(cutoff)).float()
    del distance
    diag = torch.diag(torch.ones(N))
    dist = diag+ (1-diag)*distance1
    del distance1, diag
    flag = (dist>0).float()
    direction = direction*flag.unsqueeze(2)
    del direction, dist
    edge_sparse = torch.nonzero(flag) #K,2
    edge_attr_sp = edge_type[edge_sparse[:,0],edge_sparse[:,1]] #K,4
    if noisedict is None:
        savefilecont = [ atoms, edge_sparse, edge_attr_sp]
    else:
        savefilecont = [ atoms, edge_sparse, edge_attr_sp, global_resid2noise]
    return savefilecont

def main(PDB, MUT, INTER, DDD,workdir):
    gnnfile = 'GeoEnc.tor'
    gbtfile = 'gbt-s4169.pkl'
    idxfile = 'sortidx.npy'
    pdb_path=args.pdb_path
    pdbfile = pdb_path+PDB+'.pdb'

    mutationinfo = MUT
    if_info = INTER

    try:
        sorted_idx = np.load(idxfile)
    except:
        print('File reading error: Please redownload the file {} from the GitHub website again!'.format(idxfile))

    cutoff = 3

    graph_mutinfo = []
    flag = False
    info = mutationinfo
    wildname = info[0]
    chainid = info[1]
    resid = info[2:-1]
    mutname = info[-1]
    if wildname == mutname: flag = True
    graph_mutinfo.append('{}_{}'.format(chainid, resid))

    wildtypefile = '{}/WT_{}_1.pdb'.format(workdir,PDB)
    mutantfile = '{}/{}_1.pdb'.format(workdir,PDB)

    os.system('python gen_interface.py {} {} {} > {}/pymol.log'.format(pdbfile, if_info, workdir, workdir))
    interfacefile = '{}/interface.txt'.format(workdir)

    try:
        A, E, _ = gen_graph_data(wildtypefile, graph_mutinfo, interfacefile, cutoff, if_info)
        A_m, E_m, _ = gen_graph_data(mutantfile, graph_mutinfo, interfacefile, cutoff, if_info)
    except:
        print(
            'Data processing error: Please double check your inputs is correct! Such as the pdb file path, mutation information and binding partners. You might find more error details at {}/foldx.log'.format(
                workdir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GeometricEncoder(256)
    try:
        model.load_state_dict(torch.load(gnnfile, map_location='cpu'))
    except:
        print('File reading error: Please redownload the file {} from the GitHub website again!'.format(gnnfile))

    model.to(device)
    model.eval()
    A = A.to(device)
    E = E.to(device)
    A_m = A_m.to(device)
    E_m = E_m.to(device)

    try:
        with open(gbtfile, 'rb') as pickle_file:
            forest = pickle.load(pickle_file)
    except:
        print('File reading error: Please redownload the file {} via the following command: \
                    wget https://media.githubusercontent.com/media/Liuxg16/largefiles/8167d5c365c92d08a81dffceff364f72d765805c/gbt-s4169.pkl -P trainedmodels/'.format(
            gbtfile))

    wt_fea,mt_fea=GeoPPIfeature(A, E, A_m, E_m, model, sorted_idx, flag)

    Result = pd.DataFrame(wt_fea)

    Result["pdbname"] = PDB
    Result['DDG'] = DDD
    Result["mutation"] = MUT
    Resultt = pd.DataFrame(mt_fea)
    Resulttt=pd.concat([Result,Resultt],axis=1)
    fold_path = args.save_path
    Resulttt.to_csv(fold_path, mode='a', index=False, header=False)
    print(PDB + "-" + MUT + "-" + str(DDD), "predicted finished")

import multiprocessing
import pandas as pd
if __name__ == "__main__":
    args = get_finetune_config()
    file = pd.read_csv(args.csv,header=None,index_col=None)
    #cases=[]
    num = file.shape[0]
    for i in range(num):
        pdbname = file.iloc[i, 1]
        if pdbname=='1.00E+96':
            pdbname='1E96'
        if pdbname == '1.00E+50':
            pdbname = '1E50'
        inter1 = file.iloc[i, 2]
        inter2 = file.iloc[i, 3]
        if_info = inter1 + "_" + inter2
        mutation = file.iloc[i, 4]
        wildname = mutation[0]
        chainid = mutation[1]
        resid_d = mutation[2:-1]
        resid = ''.join(list(filter(str.isdigit, resid_d)))
        mutname = mutation[-1]
        mutationinfo = wildname + chainid + resid + mutname

        DDD = file.iloc[i, 5]
        workdir = args.workdir+pdbname + str(mutationinfo)

        main(pdbname, mutationinfo, if_info, DDD,workdir)
        # cases.append([PDB,MUT,INTER,DDD,workdir])
    # pool = multiprocessing.Pool(processes=min(65, multiprocessing.cpu_count()))
    # for args in cases:
    #     pool.apply_async(main, args=args)
    # pool.close()
    # pool.join()

 
