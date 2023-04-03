import pandas as pd
from sklearn.model_selection import train_test_split

# get cancer data
class GetData():
    def __init__(self):
        PATH = 'GDSC_data/'

        genefile = PATH + '/Cell_line_RMA_proc_basalExp.txt'
        smilefile = PATH + '/smile_inchi.csv'
        pairfile = PATH + '/GDSC2_fitted_dose_response_25Feb20.xlsx'
        drug_infofile = PATH + "/Drug_listTue_Aug10_2021.csv"
        self.pairfile = pairfile
        self.drugfile = drug_infofile
        self.genefile = genefile
        self.smilefile = smilefile

    # get drug data
    def getDrug(self):
        drugdata = pd.read_csv(self.smilefile,index_col=0)
        return drugdata
    
    # filt vacancy information
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
    
    # split function
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

        return train_data, test_data
    
    # split train and test data for Cancer dataset
    def ByCancer(self,random_seed):

        drug_cell_df = pd.read_excel(self.pairfile)
        drug_cell_df = self._filter_pair(drug_cell_df)

        drug_cell_df = drug_cell_df.head(10000)
        print(drug_cell_df['TCGA_DESC'].value_counts())

        train_data, test_data = self._split(df=drug_cell_df, col='TCGA_DESC',
                                            ratio=0.2, random_seed=random_seed)

        return train_data, test_data
    
    # get train and test data of cell line for Cancer dataset
    def getGene(self,traindata,testdata):

        train_geneid = list(traindata['COSMIC_ID'])
        test_geneid = list(testdata['COSMIC_ID'])
        train_geneid = ['DATA.'+str(i) for i in train_geneid]
        test_geneid = ['DATA.' +str(i) for i in test_geneid ]


        cell_line = pd.read_excel('SarsCov2_data/Vero6.xlsx')
        genedata =  pd.read_csv('GDSC_data/Cell_line_RMA_proc_basalExp.txt',sep='\t')
        gene2value = dict(zip(cell_line['Gene'],cell_line['use']))
        gene_name = [x for x in list(cell_line['Gene']) if x in list(genedata['GENE_SYMBOLS']) and gene2value[x] != 0]
        cancer_cell_gene = genedata[genedata['GENE_SYMBOLS'].isin(gene_name)]
        train_genedata = cancer_cell_gene[train_geneid]
        test_genedata = cancer_cell_gene[test_geneid]        

        return train_genedata,test_genedata

