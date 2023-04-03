# encoding: utf-8
import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import gc

from mpn_models_chemprop.data.data import MoleculeDataset, MoleculeDatapoint
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
from sklearn.model_selection import KFold

import copy
from argparse import ArgumentParser

from enceoder import Encoder_MultipleLayers, Embeddings
from mpn_models_chemprop import MoleculeModel

# dataloader for Cancer dataset
class data_process_loader(data.Dataset):
	def __init__(self, list_IDs, labels, drug_df,rna_df):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.drug_df = drug_df
		self.rna_df = rna_df

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		index = self.list_IDs[index]
		v_d = self.drug_df.iloc[index]['smiles']
		v_p = np.array(self.rna_df.iloc[index])
		y = self.labels[index]

		return v_d, v_p, y

# dataloader for Sars-CoV-2 dataset
class data_process_loader_covid(data.Dataset):
	def __init__(self, list_IDs, labels, drug_df):
		'Initialization'
		self.labels = labels
		self.list_IDs = list_IDs
		self.drug_df = drug_df

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.list_IDs)

	def __getitem__(self, index):
		'Generates one sample of data'
		index = self.list_IDs[index]
		v_d = self.drug_df.iloc[index]['Smiles']
		y = self.labels[index]

		return v_d, y


# transformer for drug feature and cross-attention module
class Transformer(nn.Sequential):
    def __init__(self, dmpnn_encoder):
        super(Transformer, self).__init__()
        input_dim_drug = 2586
        transformer_emb_size_drug = 128
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1

        self.dmpnn_encoder = dmpnn_encoder       # dmpnn

        self.emb = Embeddings(input_dim_drug,
                         transformer_emb_size_drug,
                         50,
                         transformer_dropout_rate)

        self.encoder = Encoder_MultipleLayers(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
        self.position_embeddings = nn.Embedding(300, 300) # 300

        self.cell_linear = torch.nn.Linear(384, 128)

    def forward(self, v, v_P, fusion):    # fusion == True, used for cross-attention ; fusion == False, used for graphtransformer
        # datapoint
        smile_data = MoleculeDataset([              
            MoleculeDatapoint( 
                line=line,
            ) for i, line in enumerate(v)
        ])

        h_node = self.dmpnn_encoder(smile_data.smiles())[0]
        degree = self.dmpnn_encoder(smile_data.smiles())[1]
        mask = []
        # Unify dimension
        for i in range(len(h_node)):
            mask.append([] + h_node[i].size()[0] * [0] + (300-h_node[i].size()[0]) * [-10000])
            temp = torch.full([(300-h_node[i].size()[0]), 300], 0).to(device)
            degree_temp = torch.zeros(300-h_node[i].size()[0]).to(device)
            h_node[i] = torch.cat((h_node[i], temp),0)
            degree[i] = torch.cat((torch.Tensor(degree[i]).to(device), degree_temp), 0)
        # use degree encoding instead of position encoding
        h_node = torch.stack(h_node) + self.position_embeddings(torch.stack(degree).long())  
        # dmpnn
        h_node = self.dmpnn_encoder.ffn(h_node)

        mask = torch.tensor(mask, dtype=torch.float)
        mask =  mask.to(device).unsqueeze(1).unsqueeze(2)
        # cross-attention
        if fusion:        
            encoded_layers = self.encoder([h_node.float(),v_P.unsqueeze(1).expand(h_node.size()[0],300,128).float()], mask, fusion)
            return encoded_layers[:,:, 0]
        # graphtransformer
        else:            
            encoded_layers = self.encoder(h_node.float(), mask, fusion)
            return encoded_layers[:, 0]

# DNN for cell line feature
class DNN(nn.Sequential):
    def __init__(self):
        super(DNN, self).__init__()
        input_dim_gene = 11794
        hidden_dim_gene = 128
        dnn_hidden_dims_gene = [1024, 512, 256]
        layer_size = len(dnn_hidden_dims_gene) + 1
        dims = [input_dim_gene] + dnn_hidden_dims_gene + [hidden_dim_gene]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        v = v.float().to(device)
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v

# fuse representation of drug and cell line, then predict 
class Predict(nn.Sequential):
    def __init__(self, model_drug, model_gene, model_drug_nol):
        super(Predict, self).__init__()
        self.input_dim_drug = 128
        self.input_dim_gene = 0
        self.Dn = model_drug_nol
        self.model_drug = model_drug
        self.model_gene = model_gene
        self.dropout = nn.Dropout(0.1)
        self.hidden_dims = [256]
        layer_size = len(self.hidden_dims) + 1
        dims = [256] + self.hidden_dims + [1]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v_D, v_P):
        # first cell line representation 
        v_P = self.model_gene(v_P)
        # first drug representation 
        v_D_nol = self.Dn(v_D, v_P,False)
        # output of cross-attention module, including second drug representation and second cell line representation 
        v_D = self.model_drug(v_D, v_P, True)
        # fusion representation
        v_d_f = (v_D[0] + v_D_nol) / 2
        v_p_f = (v_D[1] + v_P) / 2
        v_f = torch.cat((v_d_f, v_p_f), 1)

        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))

        return v_f

# train and test
class DeepCoVDR:
    def __init__(self,modeldir, args, best_state=None):
        self.device = torch.device('cuda:0')
        # dmpnn
        self.dmpnn_encoder = MoleculeModel(classification=False, multiclass=False)
        self.dmpnn_encoder.create_encoder(args)
        self.dmpnn_encoder.ffn = nn.Sequential( 
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.dmpnn_output_size)
        )
       
        # transformer for cross-attention module
        model_drug = Transformer(self.dmpnn_encoder)
        
        # transformer for first drug representation
        model_drug_nol = Transformer(self.dmpnn_encoder)
        self.model_drug_nol = model_drug
        
        # DNN for cell line feature
        model_gene = DNN()

        # fuse and predict
        self.model = Predict(model_drug, model_gene, model_drug_nol)

        self.modeldir = modeldir
        self.best_state = {}
        self.covid_best_state = {}
        if best_state != None:
            self.load_state = best_state
        else: self.load_state = None
    
    # test for Cancer dataset
    def test(self,datagenerator, model):
        y_label = []
        y_pred = []
        model.eval()
        for i,(v_drug,v_gene,label) in enumerate(datagenerator):
            score = model(v_drug,v_gene)
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1)
            loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(self.device))
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        model.train()

        return y_label, y_pred, \
               mean_squared_error(y_label, y_pred), \
               np.sqrt(mean_squared_error(y_label, y_pred)), \
               pearsonr(y_label, y_pred)[0], \
               pearsonr(y_label, y_pred)[1], \
               spearmanr(y_label, y_pred)[0], \
               spearmanr(y_label, y_pred)[1], \
               concordance_index(y_label, y_pred), \
               loss, \

    # test for SARS-CoV-2 dataset
    def test_covid(self,datagenerator,model,rna):
        y_label = []
        y_pred = []
        model.eval()
        for i,(v_drug,label) in enumerate(datagenerator):
            label1 = Variable(torch.from_numpy(np.array(label))).float().to(self.device)
            v_p = torch.tensor(rna).unsqueeze(0).repeat(label1.size()[0], 1)
            score = model(v_drug,v_p)
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1)
            loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(self.device))
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
        
        model.train()

        return y_label, y_pred, \
               mean_squared_error(y_label, y_pred), \
               np.sqrt(mean_squared_error(y_label, y_pred)), \
               pearsonr(y_label, y_pred)[0], \
               pearsonr(y_label, y_pred)[1], \
               spearmanr(y_label, y_pred)[0], \
               spearmanr(y_label, y_pred)[1], \
               concordance_index(y_label, y_pred), \
               loss\


    # train for Cancer dataset
    def train(self, train_drug, train_gene, val_drug, val_rna):
        lr = 1e-4
        decay = 0
        BATCH_SIZE = 32
        train_epoch = 10
        self.model = self.model.to(self.device) 
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay, eps=1e-08)
        if self.load_state != None:
            self.model.load_state_dict(self.load_state['model'])
            opt.load_state_dict(self.load_state['optimizer'])

        loss_history = []
        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0,
                  'drop_last': False}
        training_generator = data.DataLoader(data_process_loader(
            train_drug.index.values, train_drug.Label.values, train_drug, train_gene), **params)  
        validation_generator = data.DataLoader(data_process_loader(
            val_drug.index.values, val_drug.Label.values, val_drug, val_rna), **params)

        max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []

        float2str = lambda x: '%0.4f' % x
        print('--- Go for Training(cancer) ---')
        iteration_loss = 0
        for epo in range(train_epoch):
            for i, (v_d, v_p, label) in enumerate(training_generator):
                score = self.model(v_d, v_p)
                label = Variable(torch.from_numpy(np.array(label))).float().to(self.device)

                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)
                loss_history.append(loss.item())
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()


            with torch.set_grad_enabled(False):

                y_true,y_pred, mse, rmse, \
                person, p_val, \
                spearman, s_p_val, CI,\
                loss_val = self.test(validation_generator, self.model)
                lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, person, p_val, spearman,
                                                                   s_p_val, CI]))
                valid_metric_record.append(lst)

                
                print('Training at Epoch ' + str(epo + 1) +
                        ' iteration ' + str(i) + \
                        ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                        'Pearson ' + str(person)+\
                        'Spearman ' + str(spearman))

                if mse < max_MSE:
                    model_max = copy.deepcopy(self.model)
                    state = {'model': model_max.state_dict(), 'optimizer': opt.state_dict()}
                    max_MSE = mse
                    print('Validation at Epoch ' + str(epo + 1) +
                          ' with loss:' + str(loss_val.item())[:7] +
                          ', MSE: ' + str(mse)[:7] +
                          ' , Pearson Correlation: ' + str(person)[:7] +
                          ' with p-value: ' + str(p_val)[:7] +
                          ' Spearman Correlation: ' + str(spearman)[:7] +
                          ' with p_value: ' + str(s_p_val)[:7]
                    )
        self.model = model_max
        self.best_state = state

        print('--- Finished 1---')



    # train for SARS-CoV-2 dataset
    def train_covid(self, train_drug, val_drug, rna, emb_for_vis_pre, emb_for_vis_label):
        lr = 1e-4
        decay = 0
        BATCH_SIZE = 32
        train_epoch = 12000
        self.model = self.model.to(self.device) 
        opt = torch.optim.Adam(filter(lambda p:p.requires_grad, self.model.parameters()), lr=lr, weight_decay=decay, eps=1e-08)
        if self.load_state != None:
            self.model.load_state_dict(self.load_state['model'])
            opt.load_state_dict(self.load_state['optimizer'])

        loss_history = []

        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': 0,
                  'drop_last': False}
        training_generator = data.DataLoader(data_process_loader_covid(
            train_drug.index.values, train_drug['Standard Value'].values, train_drug), **params)  
        validation_generator = data.DataLoader(data_process_loader_covid(
            val_drug.index.values, val_drug['Standard Value'].values, val_drug), **params)

        max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []

        float2str = lambda x: '%0.4f' % x
        print('--- Go for Training(sars-covid-2) ---')
        iteration_loss = 0

        for epo in range(train_epoch):
            for i, (v_d, label) in enumerate(training_generator):

                label = Variable(torch.from_numpy(np.array(label))).float().to(self.device)
                v_p = torch.tensor(rna).unsqueeze(0).repeat(label.size()[0], 1)

                score = self.model(v_d, v_p)

                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)
                loss_history.append(loss.item())
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.set_grad_enabled(False):

                y_true,y_pred, mse, rmse, \
                person, p_val, \
                spearman, s_p_val, CI,\
                loss_val = self.test_covid(validation_generator, self.model, rna)
                lst = ["epoch " + str(epo)] + list(map(float2str, [mse, rmse, person, p_val, spearman,
                                                                   s_p_val, CI]))
                valid_metric_record.append(lst)

                if i % 500 == 0:
                    print('Training at Epoch ' + str(epo + 1) +
                            ' iteration ' + str(i) + \
                            ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                            'Pearson ' + str(person)+\
                            'Spearman ' + str(spearman))

                if mse < max_MSE:
                    model_max = copy.deepcopy(self.model)
                    covid_state = {'model': model_max.state_dict(), 'optimizer': opt.state_dict()}
                    max_MSE = mse
                    print('Validation at Epoch ' + str(epo + 1) +
                          ' with loss:' + str(loss_val.item())[:7] +
                          ', MSE: ' + str(mse)[:7] +
                          ' , Pearson Correlation: ' + str(person)[:7] +
                          ' with p-value: ' + str(p_val)[:7] +
                          ' Spearman Correlation: ' + str(spearman)[:7] +
                          ' with p_value: ' + str(s_p_val)[:7] +
                          ' , Concordance Index: ' + str(CI)[:7])

        emb_for_vis_pre.extend(y_pred)
        emb_for_vis_label.extend(y_true)
        self.model = model_max

        self.covid_best_state = covid_state

        print('--- Finished 2---')

    def save_model(self, num):
        torch.save(self.model, 'save_model/model'+str(num)+'.pt') # path

    def load_model(self):
        return self.best_state

    def load_covid_model(self):
        return self.covid_best_state
   


if __name__ == '__main__':

    parser = ArgumentParser()
    # argument for dmpnn
    parser.add_argument('--single_lambda', type=float, default=0.1)
    parser.add_argument('--combo_lambda', type=float, default=1)
    parser.add_argument('--dti_lambda', type=float, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dmpnn_output_size', type=int, default=128)
    parser.add_argument('--num_hiv_targets', type=int, default=7)
    parser.add_argument('--num_covid_targets', type=int, default=35)
    parser.add_argument('--trans_size', type=int, default=100)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--dim_model', type=int, default=256)
    parser.add_argument('--graph_pooling', type=str, default='cls')
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to skip training and only test the model')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')                   
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--same_val_test', action='store_true', default=False,
                        help='use the same data for validation and testing')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression', 'multiclass'],
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')
    parser.add_argument('--multiclass_num_classes', type=int, default=3,
                        help='Number of classes when running multiclass classification')
    parser.add_argument('--separate_val_path', type=str,
                        help='Path to separate val set, optional')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*',
                        help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*',
                        help='Path to file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds when performing cross validation')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--crossval_index_dir', type=str, 
                        help='Directory in which to find cross validation index files')
    parser.add_argument('--crossval_index_file', type=str, 
                        help='Indices of files to use as train/val/test'
                             'Overrides --num_folds and --seed.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--config_path', type=str,
                        help='Path to a .json file containing arguments. Any arguments present in the config'
                             'file will override arguments specified via the command line or by the defaults.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')                     
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')
    parser.add_argument('--attention', action='store_true', default=False,
                        help='Use attention in message aggregation')
    parser.add_argument('--cuda', default=True)

    args = parser.parse_args()



    from process_cancer_data import Data_for_cancer

    # for cancer
    obj = Data_for_cancer()

    traindata, testdata = obj.Getdata.ByCancer(random_seed=1)

    # Cancer drug and cell line
    traindata, train_genedata, testdata, test_rnadata = obj.get_train_and_test_for_cancer( 
        traindata=traindata, 
        testdata=testdata
    )
    modeldir = 'Model'
    modelfile = modeldir + '/model.pt'
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)

    # pre-train by cancer
    net = DeepCoVDR(modeldir=modeldir, args=args) 
    net.train(train_drug=traindata, train_gene=train_genedata,
              val_drug=testdata, val_rna=test_rnadata)

    best_state = net.load_model()


    # for covid19

    # SARS-CoV-2 drug
    ic50 = pd.read_csv('SarsCov2_data/sars_ic50.tsv', sep='\t')
    sub_df = ic50[['Molecule ChEMBL ID', 'Smiles', 'Standard Value']]
    sub_df['Standard Value'] = (sub_df['Standard Value'] / 1000).apply(np.log)
    print(sub_df['Standard Value'])
 
    # SARS-CoV-2 cell line 
    cell_line = pd.read_excel('SarsCov2_data/Vero6.xlsx')
    rnadata =  pd.read_csv('GDSC_data/Cell_line_RMA_proc_basalExp.txt',sep='\t')
    gene2value = dict(zip(cell_line['Gene'],cell_line['use']))
    gene_name = [x for x in list(cell_line['Gene']) if x in list(rnadata['GENE_SYMBOLS']) and gene2value[x] != 0]
    record_name = [x for x in cell_line['Gene'] if x in gene_name]
    cell_line = cell_line[cell_line['Gene'].isin(gene_name)] 
    cell_line.sort_values(by=['Gene'],ascending=False, inplace=True)
    cell_line['use'] = (cell_line['use'] - cell_line['use'].mean()) / cell_line['use'].std()
    cell_line_list = cell_line['use'].tolist()


    # 5-fold on SARS-CoV-2 dataset
    kf = KFold(n_splits=5, random_state=5, shuffle=True)
    foldnum = 0 
    emb_for_vis_pre = []
    emb_for_vis_label = []
    for X_train,X_test in kf.split(sub_df):
        foldnum += 1
        new_model = DeepCoVDR(best_state=best_state,modeldir='', args=args)
        print('___________'+str(foldnum)+' _fold______________')
        sub_train = sub_df.loc[X_train]
        sub_test = sub_df.loc[X_test]
        sub_train = sub_train.reset_index()
        sub_test = sub_test.reset_index()
        print(sub_train, sub_test)
        new_model.train_covid(train_drug=sub_train, val_drug=sub_test, rna=cell_line_list,emb_for_vis_pre=emb_for_vis_pre,emb_for_vis_label=emb_for_vis_label)
        del new_model
        torch.cuda.empty_cache()
        gc.collect()

    



