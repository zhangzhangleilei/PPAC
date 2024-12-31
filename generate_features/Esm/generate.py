import esm
# import matplotlib
import pandas as pd
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str,help='save file')
parser.add_argument('--csv', type=str, help='used csv file')    
args = parser.parse_args()

model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_convert = alphabet.get_batch_converter()
model.eval()

file=pd.read_csv(args.csv,header=0,index_col=None)

data=[]
ttt=[]
def get_fea():

    batch_labels, batch_strs, batch_tokens = batch_convert(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations = []

    for i, tokens_len in enumerate(batch_lens):
        ten = token_representations[i, 1: tokens_len - 1].mean(0)
        ten_ = ten.numpy()
        ttt.append(ten_)

for index,row in file.iterrows():
    mutation = row["mutation"]
    DDG = row["ddg"]
    pdbname = row["pdbname"]
    wt_seq = str(row["chainA"]) + str(row["chainB"])
    mt_seq = str(row["chainA_mt"]) + str(row["chainB_mt"])

    data.append((pdbname+"_wt",wt_seq))

    # get fea:
    get_fea()
    data.clear()
    data.append((pdbname+"_mt",mt_seq))
    get_fea()
    data.clear()
    nnn=[x.tolist() for x in ttt]
    re = pd.DataFrame(nnn[0]).T
    # p = batch_labels[i]
    re["pdb"]=pdbname
    re["ddg"] = DDG
    re["mutation"] = mutation

    re_ =pd.DataFrame(nnn[1]).T
    re__=pd.concat([re,re_],axis=1)
    # 5123:2560:wt pdb mutation ddg 2560:mt
    re__.to_csv(args.save_path, mode="a", header=False,
            index=False)

    ttt.clear()

    print(index,pdbname+"-"+mutation,"save finished")

