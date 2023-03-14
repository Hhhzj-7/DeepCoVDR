
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE
from get_data import GetData

class DataEncoding:
    def __init__(self,vocab_dir):
        self.vocab_dir = vocab_dir
        self.Getdata = GetData()

    def _drug2emb_encoder(self,smile):
 
        vocab_path = "ESPF/drug_codes_chembl_freq_1500.txt"
        sub_csv = pd.read_csv("ESPF/subword_units_map_chembl_freq_1500.csv")

        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        idx2word_d = sub_csv['index'].values
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

        max_d = 50
        t1 = dbpe.process_line(smile).split()  # split
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])
        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)

    def encode(self,traindata,testdata):
        drug_smiles = self.Getdata.getDrug()
        drugid2smile = dict(zip(drug_smiles['drug_id'],drug_smiles['smiles']))
        smile_encode = pd.Series(drug_smiles['smiles'].unique()).apply(self._drug2emb_encoder)
        uniq_smile_dict = dict(zip(drug_smiles['smiles'].unique(),smile_encode))

        traindata['smiles'] = [drugid2smile[i] for i in traindata['DRUG_ID']]
        testdata['smiles'] = [drugid2smile[i] for i in testdata['DRUG_ID']]
        traindata['drug_encoding'] = [uniq_smile_dict[i] for i in traindata['smiles']]
        testdata['drug_encoding'] = [uniq_smile_dict[i] for i in testdata['smiles']]
        traindata = traindata.reset_index()
        traindata['Label'] = traindata['LN_IC50']
        testdata = testdata.reset_index()
        testdata['Label'] = testdata['LN_IC50']


        train_rnadata, test_rnadata = self.Getdata.getRna(
            traindata=traindata,
            testdata=testdata)
        train_rnadata = (train_rnadata - train_rnadata.mean())/train_rnadata.std()
        test_rnadata = (test_rnadata - test_rnadata.mean())/test_rnadata.std()

        
        train_rnadata = train_rnadata.T
        test_rnadata = test_rnadata.T
        train_rnadata.index = range(train_rnadata.shape[0])
        test_rnadata.index = range(test_rnadata.shape[0])
        return traindata, train_rnadata, testdata, test_rnadata
