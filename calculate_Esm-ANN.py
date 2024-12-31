import numpy as np
from ann_model_20240519 import ProteinNet
import pandas as pd
import os
from sklearn.metrics import accuracy_score,mean_squared_error
import torch
from scipy.stats import pearsonr,kendalltau
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,help='model path')
parser.add_argument('--save_path', type=str,help='save file')
parser.add_argument('--test_csv', type=list, help='used csv file')
args = parser.parse_args()

folder=args.model_path
for file in args.test_csv:
    data=pd.read_csv(file,header=None, index_col=None)
    for filename in os.listdir(folder):
        filepath=os.path.join(folder,filename)
        if filepath.endswith('.tor'):

            model = ProteinNet(fea_dim=2560, hidden_dim1=128, hidden_dim2=256, num_layers=5)
            try:
                model.load_state_dict(torch.load(filepath, map_location='cpu'))
            except:
                print('please check your model')

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            true_d=[]
            pdb=[]
            mut=[]
            pre_d=[]
            for i in range(data.shape[0]):
                wt_features = data.iloc[i, :2560]
                mt_features = data.iloc[i, 2563:]
                wt_features = wt_features.reset_index(drop=True)
                mt_features = mt_features.reset_index(drop=True)
                mwfeatures = mt_features - wt_features
                mw_features = torch.tensor(mwfeatures, dtype=torch.float32)

                true_ddg = data.iloc[i, 2561]
                pdbname = data.iloc[i, 2560]
                mutation = data.iloc[i, 2562]

                pred_ddg = model(mw_features)
                pred_ddg = float(pred_ddg.detach().cpu().numpy())
                # pred_ddg = np.negative(pred_ddg)
                true_d.append(true_ddg)
                pdb.append(pdbname)
                mut.append(mutation)
                pre_d.append(pred_ddg)

            result={"pdbname":pdb,"mutation":mut,"true-ddg":true_d,"pre_ddg":pre_d}
            resultf=pd.DataFrame(result)
            resultf.to_csv(folder + "{}_pred{}.csv".format(os.path.basename(file).split(".")[0].split("_")[0],
                                                           os.path.basename(filepath)), index=False)

            rmsee = []
            pcc1 = []
            pcc2 = []
            modell = []
            ff=[]
            rmse = np.sqrt(mean_squared_error(true_d, pre_d))
            rp = pearsonr(true_d, pre_d)
            rank = kendalltau(true_d, pre_d)
            rmsee.append(rmse)
            pcc1.append(rp[0])
            pcc2.append(rank[0])
            modell.append(os.path.basename(filepath))
            ff.append(os.path.basename(file).split(".")[0].split("_")[0])
            ress = {"file_name": ff, "modelname": modell, "rmse": rmsee, "pcc1": pcc1, "pcc2": pcc2}
            reeee = pd.DataFrame(ress)
            reeee.to_csv(folder + "rmse_pcc.csv", mode="a", index=False, header=False)
            print(os.path.basename(file).split(".")[0].split("_")[0],
                  os.path.basename(filepath),
                  rmse, rp)

# if __name__ == '__main__':
#     main()