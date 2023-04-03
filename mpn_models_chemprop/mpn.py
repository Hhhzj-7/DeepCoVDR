import os
from argparse import Namespace
from typing import List, Union

import torch
import torch.nn as nn
import numpy as np

from .features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from .nn_utils import index_select_ND, get_activation_function


class MPNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """Initializes the MPNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.attention = args.attention
        self.features_only = args.features_only
        self.use_input_features = False
        self.args = args

        self.max_atom_num = 0

        if self.features_only:
            return

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        #self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        self.cached_zero_vector = torch.zeros(self.hidden_size).cuda()

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size * 2 + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # attention weights
        if self.attention:
            self.A = nn.Sequential(
                    nn.Linear(self.hidden_size + self.bond_fdim, self.hidden_size),
                    nn.LeakyReLU(0.2),
                    nn.Linear(self.hidden_size, 1),
                    nn.Softmax(dim=1)
            )

    def forward(self,
                mol_graph: BatchMolGraph,
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()

            if self.args.cuda:
                features_batch = features_batch.cuda()

            if self.features_only:
                return features_batch

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()

        if self.atom_messages:
            a2a = mol_graph.get_a2a()

        if self.args.cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), a2b.cuda(), b2a.cuda(), b2revb.cuda()

            if self.atom_messages:
                a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size

        message = self.act_func(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2  # 一开始假定的是有方向，atoms会对同一条bond的不同方向消息传递，此时如果我们认为无方向的话，需要平均取平均

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x (hidden + bond_fdim)
                if self.attention:
                    att_scores = self.A(nei_message)  # num_atoms x max_num_bonds x 1
                    nei_message = torch.sum(att_scores * nei_message, dim=1)  # num_atoms x (hidden + bond_fdim)
                else:
                    nei_message = nei_message.sum(dim=1)  # num_atoms x (hidden + bond_fdim)
                message = torch.cat((message, nei_message), dim=-1)
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden  a2b是每一个原子对应的获得消息的bonds 这里获得的向量是a2b里的每一个原子对应的bonds的特征表示
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden  为每个原子聚合所有键的信息，得到每个原子的表示向量
                rev_message = message[b2revb]  # num_bonds x hidden  message中每个bonds对应的revb bonds的特征表示
                # （只记录每个键的起始原子是因为这里认为消息传递是有顺序，有方向所以每个带方向的键的起始原子只能有一个）
                #  最终得到的这个message就是每个键除去自己发出去消息的那一个原子的’反方向‘键（其实就是自己）来源的最终的消息
                message = a_message[b2a] - rev_message  # num_bonds x hidden  每个键的起始原子的表示向量（b2a的size等于bonds数量） - 相应的键的revb的表示向量（这里代表的是从该起始原子放出去的消息）

            message = self.W_h(message)
            message = self.act_func(input + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:     # 没有原子
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)   # 当前分子的所有原子的表示向量
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                if self.max_atom_num < mol_vec.size()[0]:
                    self.max_atom_num = mol_vec.size()[0]
                    print('max:',self.max_atom_num)
                    print(mol_vec.size())

                # mol_vec = mol_vec.sum(dim=0) / a_size    # 获得每个分子的表示向量
                mol_vecs.append(mol_vec)

        # mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)  stack：沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状
        
        if self.use_input_features:
            features_batch = features_batch.to(mol_vecs)
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1,features_batch.shape[0]])
            mol_vecs = torch.cat([mol_vecs, features_batch], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x num_atoms x hidden


class MPN(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self,
                 args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 graph_input: bool = False):
        """
        Initializes the MPN.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param graph_input: If true, expects BatchMolGraph as input. Otherwise expects a list of smiles strings as input.
        """
        super(MPN, self).__init__()
        self.args = args
        self.atom_fdim = atom_fdim or get_atom_fdim(args)
        self.bond_fdim = bond_fdim or get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim
        self.graph_input = graph_input
        self.encoder = MPNEncoder(self.args, self.atom_fdim, self.bond_fdim)

    def forward(self,
                batch: Union[List[str], BatchMolGraph],
                features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular SMILES strings.

        :param batch: A list of SMILES strings or a BatchMolGraph (if self.graph_input is True).
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """
        if not self.graph_input:  # if features only, batch won't even be used
            batch = mol2graph(batch, self.args)

        output = self.encoder.forward(batch, features_batch)
        degree = batch.degree

        return [output, degree]
