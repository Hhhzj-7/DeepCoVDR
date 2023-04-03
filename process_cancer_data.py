from get_data import GetData

# get train and test data for cancer dataset
class Data_for_cancer:
    def __init__(self):
        self.Getdata = GetData()

    def get_train_and_test_for_cancer(self,traindata,testdata):
        drug_smiles = self.Getdata.getDrug()
        drugid2smile = dict(zip(drug_smiles['drug_id'],drug_smiles['smiles']))

        # get smiles
        traindata['smiles'] = [drugid2smile[i] for i in traindata['DRUG_ID']]
        testdata['smiles'] = [drugid2smile[i] for i in testdata['DRUG_ID']]
        
        traindata = traindata.reset_index()
        
        # get LN(IC50)
        traindata['Label'] = traindata['LN_IC50']
        testdata = testdata.reset_index()
        testdata['Label'] = testdata['LN_IC50']

        # get cell line
        train_genedata, test_genedata = self.Getdata.getGene(
            traindata=traindata,
            testdata=testdata
        )
        train_genedata = (train_genedata - train_genedata.mean())/train_genedata.std()
        test_genedata = (test_genedata - test_genedata.mean())/test_genedata.std()
  
        train_genedata = train_genedata.T
        test_genedata = test_genedata.T
        train_genedata.index = range(train_genedata.shape[0])
        test_genedata.index = range(test_genedata.shape[0])
        return traindata, train_genedata, testdata, test_genedata
