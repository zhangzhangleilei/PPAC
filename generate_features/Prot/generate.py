import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import pandas as pd
from transformers import BertForMaskedLM,BertTokenizer,pipelines
from transformers import T5Tokenizer,T5EncoderModel
from transformers import AutoModel,AutoTokenizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str,help='save file')
parser.add_argument('--csv', type=str, help='used csv file')    
args = parser.parse_args()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5EncoderModel.from_pretrained("prot_t5_xl_uniref50/model/")
tokenizer = T5Tokenizer.from_pretrained("prot_t5_xl_uniref50/model/")
model.eval()

wt_result=[]
mt_result=[]
p=[]
d=[]
m=[]

file=pd.read_csv(args.csv
,header=0,index_col=None)
sequence_examples=[]
for index,row in file.iterrows():

    mutation=row["mutation"]
    DDG=row["ddg"]
    pdbname=row["pdbname"]

    wt_seq = str(row["chainA"]) + str(row["chainB"])
    mt_seq = str(row["chainA_mt"]) + str(row["chainB_mt"])

    m.append(mutation)
    p.append(pdbname)
    d.append(DDG)

    sequence_examples.append(wt_seq)
    sequence_examples.append(mt_seq)
    processed_seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequence_examples]
    ids = tokenizer.batch_encode_plus(processed_seqs, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids'])
    attention_mask = torch.tensor(ids['attention_mask'])
    with torch.no_grad():

    emb_wt = embedding_repr.last_hidden_state[0, :]
    emb_wt_per_protein = emb_wt.mean(dim=0) 
    emb_mt = embedding_repr.last_hidden_state[1, :] 
    emb_mt_per_protein = emb_mt.mean(dim=0)
    wt_result.append(emb_wt_per_protein.numpy())
    mt_result.append(emb_mt_per_protein.numpy())
    sequence_examples.clear()

    print(index, pdbname + "-" + mutation + "-" + str(DDG), "save finished")

    wt_result_=[x.tolist() for x in wt_result]
    ww=pd.DataFrame(wt_result_)

    mt_result_=[x.tolist() for x in mt_result]
    mm=pd.DataFrame(mt_result_)

    r={"pdbname":p,"ddg":d,"mutation":m}
    r_=pd.DataFrame(r)

    result=pd.concat([ww,r_,mm],axis=1)

    result.to_csv(args.save_path,mode="a",header=False,index=False)
    m.clear()
    p.clear()
    d.clear()
    c.clear()
    wt_result.clear()
    mt_result.clear()
