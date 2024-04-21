from abc import ABC, abstractmethod
import time

import numpy as np
import networkx as nx
import torch_geometric.utils
from sklearn.manifold import spectral_embedding
from gensim.models import Word2Vec
import torch
import tqdm
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric as tg
import node2vec as n2v


class PredictionModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        pass

    @abstractmethod
    def predict(self, test_edges):
        pass



class TransEModel(torch.nn.Module):
    def __init__(self, num1, num2, output_dim, gamma):
        super(TransEModel, self).__init__()
        self.emb_ent_real = torch.nn.Embedding(num1, output_dim)  # real
        # Real embeddings of relations.
        self.emb_rel_real = torch.nn.Embedding(num2, output_dim)
        self.gamma = torch.nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )
        self.init()
        self.loss_f = torch.nn.BCELoss()

    def forward(self, x, rel=None):
        
        # Check if any indices are out of bounds
        if x[:, 1].max() >= self.emb_rel_real.weight.size(0) or x[:, 1].min() < 0:

            raise ValueError("Index out of bounds in self.emb_ent_real")

        emb_head = self.emb_ent_real(x[:, 0])
        if rel is None:
            emb_rel = self.emb_rel_real(x[:, 1])
        else:
            emb_rel = self.emb_rel_real(torch.tensor([rel]*x.size(0)).long())
        emb_tail = self.emb_ent_real(x[:, 1])
        distance = torch.norm((emb_head + emb_rel) - emb_tail, p=1, dim=1)
        score = self.gamma.item() - distance
        return torch.sigmoid(score)

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_ent_real.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel_real.weight.data)

    def loss(self, score, target):
        return self.loss_f(score, target)


class TransE(PredictionModel):
    """
    TransE trained with binary cross entropy
    """

    def __init__(self):
        super(TransE, self).__init__()
        self.time = 0
        self.dim = 128
        self.epochs = 180
        self.batch_size = 128
        self.seed = 18
        torch.manual_seed(self.seed)

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        x = torch.cat([torch.tensor(pos_edges), torch.tensor(neg_edges)])
        y = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))]).float()
        trainset = torch.utils.data.TensorDataset(x, y)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        st = time.time()
        self.num_rel = max(x[:, 1].max().item() + 1, classes[1])
        # Set num1 to be equal to the maximum value in x[:, 0] plus one
        self.num_ent = max(x.max().item() + 1, classes[0])
        self.model = TransEModel(self.num_ent, self.num_rel, self.dim, 0.0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=9 * 10 ** -4)
        for epoch in range(self.epochs):
            total_loss = 0
            counter = 0
            with tqdm.tqdm(total=len(trainloader.dataset), desc="Epoch: {}".format(epoch), unit='chunks') as prog_bar:
                for i in range(len(trainloader)):
                    data = next(iter(trainloader))
                    x = self.model(data[0])
                    counter += 1
                    loss = torch.clamp(self.model.loss(x, data[1]), min=0., max=50000.).double()
                    total_loss += loss
                    prog_bar.set_postfix(**{'run:': "TransE",
                                            'loss': loss.item()})
                    prog_bar.update(self.batch_size)
            optimizer.zero_grad()
            total_loss /= counter
            # print(total_loss.item())
            total_loss.backward()
            optimizer.step()
        self.time = time.time() - st

    def predict(self, test_edges):
        x_test = torch.tensor(test_edges)
        scores = [self.model(x_test, rel=i) for i in range(self.num_rel)]
        preds = torch.max(torch.stack(scores), dim=0).values.detach().numpy()

        return preds


class RotatEModel(torch.nn.Module):
    def __init__(self, num1, num2, output_dim, gamma):
        super(RotatEModel, self).__init__()
        self.emb_ent_real = torch.nn.Embedding(num1, output_dim)
        self.emb_ent_img = torch.nn.Embedding(num1, output_dim)
        self.emb_rel = torch.nn.Embedding(num2, output_dim)
        self.gamma = torch.nn.Parameter(
            torch.tensor([gamma]),
            requires_grad=False
        )
        self.embedding_range = torch.nn.Parameter(
            torch.tensor([(gamma + 2.0) / output_dim]),
            requires_grad=False
        )
        self.phase = torch.nn.Parameter(self.embedding_range / torch.tensor(np.pi).float(), requires_grad=False)
        self.init()
        self.loss_f = torch.nn.BCELoss()

    def init(self):
        torch.nn.init.xavier_normal_(self.emb_ent_real.weight.data)
        torch.nn.init.xavier_normal_(self.emb_ent_img.weight.data)
        torch.nn.init.xavier_normal_(self.emb_rel.weight.data)

    def forward(self, x, rel=None):
        head_real = self.emb_ent_real(x[:, 0])
        tail_real = self.emb_ent_real(x[:, 1])
        head_img = self.emb_ent_img(x[:, 0])
        tail_img = self.emb_ent_img(x[:, 1])
        if rel is None:
            emb_rel = self.emb_rel(x[:, 1])
        else:
            emb_rel = self.emb_rel(torch.tensor([rel]*x.size(0)).long())

        phase_relation = emb_rel / self.phase
        rel_real = torch.cos(phase_relation)
        rel_img = torch.sin(phase_relation)

        real_score = (head_real * rel_real - head_img * rel_img) - tail_real
        img_score = (head_real * rel_img + head_img * rel_real) - tail_img

        score = torch.stack([real_score, img_score], dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=1)
        return torch.sigmoid(score)

    def loss(self, score, target):
        return self.loss_f(score, target)


class RotatE(PredictionModel):
    """
    TransE trained with binary cross entropy
    """

    def __init__(self):
        super(RotatE, self).__init__()
        self.time = 0
        self.dim = 128
        self.epochs = 180
        self.batch_size = 128
        self.seed = 18
        torch.manual_seed(self.seed)

    def train(self, adj, pos_edges, neg_edges, val_pos, val_neg, mgraph, classes):
        x = torch.cat([torch.tensor(pos_edges), torch.tensor(neg_edges)])
        y = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))]).float()
        trainset = torch.utils.data.TensorDataset(x, y)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        st = time.time()
        self.num_rel = max(x[:, 1].max().item() + 1, classes[1])
        # Set num1 to be equal to the maximum value in x[:, 0] plus one
        self.num_ent = max(x.max().item() + 1, classes[0])
        self.model = RotatEModel(self.num_ent, self.num_rel, self.dim, 0.0)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=9 * 10 ** -4)
        for epoch in range(self.epochs):
            total_loss = 0
            counter = 0
            with tqdm.tqdm(total=len(trainloader.dataset), desc="Epoch: {}".format(epoch), unit='chunks') as prog_bar:
                for i in range(len(trainloader)):
                    data = next(iter(trainloader))
                    x = self.model(data[0])
                    counter += 1
                    loss = torch.clamp(self.model.loss(x, data[1]), min=0., max=50000.).double()
                    total_loss += loss
                    prog_bar.set_postfix(**{'run:': "RotatE",
                                            'loss': loss.item()})
                    prog_bar.update(self.batch_size)
            optimizer.zero_grad()
            total_loss /= counter
            # print(total_loss.item())
            total_loss.backward()
            optimizer.step()
        self.time = time.time() - st

    def predict(self, test_edges):
        x_test = torch.tensor(test_edges)
        scores = [self.model(x_test, rel=i) for i in range(self.num_rel)]
        preds = torch.max(torch.stack(scores), dim=0).values.detach().numpy()
        return preds
