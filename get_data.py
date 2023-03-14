
import sys
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

class GetData():
    def __init__(self):
        PATH = 'GDSC_data/'

        rnafile = PATH + '/Cell_line_RMA_proc_basalExp.txt'
        smilefile = PATH + '/smile_inchi.csv'
        pairfile = PATH + '/GDSC2_fitted_dose_response_25Feb20.xlsx'
        drug_infofile = PATH + "/Drug_listTue_Aug10_2021.csv"
        drug_thred = PATH + '/IC50_thred.txt'
        self.pairfile = pairfile
        self.drugfile = drug_infofile
        self.rnafile = rnafile
        self.smilefile = smilefile
        self.drug_thred = drug_thred

    def getDrug(self):
        drugdata = pd.read_csv(self.smilefile,index_col=0)
        return drugdata

    def _filter_pair(self,drug_cell_df):
        # ['DATA.908134', 'DATA.1789883', 'DATA.908120', 'DATA.908442'] not in index
        not_index = ['908134', '1789883', '908120', '908442']
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[~drug_cell_df['COSMIC_ID'].isin(not_index)]
        print(drug_cell_df.shape)

        pub_df = pd.read_csv(self.drugfile)
        pub_df = pub_df.dropna(subset=['PubCHEM'])
        pub_df = pub_df[(pub_df['PubCHEM'] != 'none') & (pub_df['PubCHEM'] != 'several')]
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[drug_cell_df['DRUG_ID'].isin(pub_df['drug_id'])]
        print(drug_cell_df.shape)
        return drug_cell_df

    def _split(self,df,col,ratio,random_seed):

        col_list = df[col].value_counts().index
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for instatnce in col_list:
            sub_df = df[df[col] == instatnce]
            sub_df = sub_df[['DRUG_ID', 'COSMIC_ID','TCGA_DESC', 'LN_IC50']]
            sub_train, sub_test = train_test_split(sub_df, test_size=ratio,random_state=random_seed)
            if train_data.shape[0] == 0:
                train_data = sub_train
                test_data = sub_test
            else:
                train_data = train_data.append(sub_train)
                test_data = test_data.append(sub_test)

        return train_data,test_data

    def ByCancer(self,random_seed):

        drug_cell_df = pd.read_excel(self.pairfile)
        drug_cell_df = self._filter_pair(drug_cell_df)

        drug_cell_df = drug_cell_df.head(10000)
        print(drug_cell_df['TCGA_DESC'].value_counts())

        train_data, test_data = self._split(df=drug_cell_df, col='TCGA_DESC',
                                            ratio=0.2,random_seed=random_seed)

        return train_data, test_data

    def Drug_Thred(self):
        thred_data = pd.read_csv(self.drug_thred,sep='\t')
        thred_df = thred_data.T
        thred_df['drug_name'] =thred_df.index
        thred_df['threds'] = thred_df[0]
        thred_df = thred_df.drop(0,axis=1)
        thred_df.loc['VX-680','drug_name'] = 'Tozasertib'
        thred_df.loc['Mitomycin C','drug_name'] = 'Mitomycin-C'
        thred_df.loc['HG-6-64-1', 'drug_name'] = 'HG6-64-1'
        thred_df.loc['BAY 61-3606', 'drug_name'] = 'BAY-61-3606'
        thred_df.loc['Zibotentan, ZD4054', 'drug_name'] = 'Zibotentan'
        thred_df.loc['PXD101, Belinostat', 'drug_name'] = 'Belinostat'
        thred_df.loc['NU-7441', 'drug_name'] = 'NU7441'
        thred_df.loc['BIRB 0796', 'drug_name'] = 'BIRB-796'
        thred_df.loc['Nutlin-3a', 'drug_name'] = 'Nutlin-3a (-)'
        thred_df.loc['AZD6482.1', 'drug_name'] = 'AZD6482'
        thred_df.loc['BMS-708163.1', 'drug_name'] = 'BMS-708163'
        thred_df.loc['BMS-536924.1', 'drug_name'] = 'BMS-536924'
        thred_df.loc['GSK269962A.1', 'drug_name'] = 'GSK269962A'
        thred_df.loc['SB-505124', 'drug_name'] = 'SB505124'
        thred_df.loc['JQ1.1', 'drug_name'] = 'JQ1'
        thred_df.loc['UNC0638.1', 'drug_name'] = 'UNC0638'
        thred_df.loc['CHIR-99021.1', 'drug_name'] = 'CHIR-99021'
        thred_df.loc['piperlongumine', 'drug_name'] = 'Piperlongumine'
        thred_df.loc['PLX4720 (rescreen)', 'drug_name'] = 'PLX4720'
        thred_df.loc['Afatinib (rescreen)', 'drug_name'] = 'Afatinib'
        thred_df.loc['Olaparib.1', 'drug_name'] = 'Olaparib'
        thred_df.loc['AZD6244.1', 'drug_name'] = 'AZD6244'
        thred_df.loc['Bicalutamide.1', 'drug_name'] = 'Bicalutamide'
        thred_df.loc['RDEA119 (rescreen)', 'drug_name'] = 'RDEA119'
        thred_df.loc['GDC0941 (rescreen)', 'drug_name'] = 'GDC0941'
        thred_df.loc['MLN4924 ', 'drug_name'] = 'MLN4924'

        drug_info = pd.read_csv(self.drugfile)
        drugname2drugid = {}
        drugid2pubchemid = {}
        for idx,row in drug_info.iterrows():
            name = row['Name']
            drug_id = row['drug_id']
            pub_id = row['PubCHEM']
            drugname2drugid[name] = drug_id
            drugid2pubchemid[drug_id] = pub_id

        drug_info_filter_name = drug_info.dropna(subset=['Synonyms'])
        for idx,row in drug_info_filter_name.iterrows():
            name = row['Name']
            pub_id = row['PubCHEM']
            drug_id = row['drug_id']
            drugname2drugid[name] = drug_id
            Synonyms_list = row['Synonyms'].split(', ')
            for drug in Synonyms_list:
                drugname2drugid[drug] = drug_id

        drugid2thred = {}
        for idx,row in thred_df.iterrows():
            name = row['drug_name']
            thred = row['threds']
            if name in drugname2drugid:
                drugid2thred[drugname2drugid[name]] = thred

        id_li = []
        PubChem_li =[]
        thred_li =[]
        for i in drugid2thred:
            id_li.append(i)
            PubChem_li.append(drugid2pubchemid[i])
            thred_li.append(drugid2thred[i])

        drug_list = [drugname2drugid[i] for i in list(thred_df['drug_name']) if i in drugname2drugid]

        return drug_list,drugid2thred

    def getRna(self,traindata,testdata):

        train_rnaid = list(traindata['COSMIC_ID'])
        test_rnaid = list(testdata['COSMIC_ID'])
        train_rnaid = ['DATA.'+str(i) for i in train_rnaid]
        test_rnaid = ['DATA.' +str(i) for i in test_rnaid ]


        cell_line = pd.read_excel('SarsCov2_data/Vero6.xlsx')
        rnadata =  pd.read_csv('GDSC_data/Cell_line_RMA_proc_basalExp.txt',sep='\t')
        gene2value = dict(zip(cell_line['Gene'],cell_line['use']))
        gene_name = [x for x in list(cell_line['Gene']) if x in list(rnadata['GENE_SYMBOLS']) and gene2value[x] != 0]
        cancer_cell_gene = rnadata[rnadata['GENE_SYMBOLS'].isin(gene_name)]
        train_rnadata = cancer_cell_gene[train_rnaid]
        test_rnadata = cancer_cell_gene[test_rnaid]        
        print(train_rnadata)


        return train_rnadata,test_rnadata

