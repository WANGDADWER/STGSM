import torch
import torch.nn as nn
import math


class GRAU(nn.Module):

    def __init__(self, input_size, hidden_size, opt):
        super(GRAU, self).__init__()
        if opt == True:
            self.grau_cell = GRAUCellNormed(input_size = input_size, hidden_size = hidden_size)
        else:
            self.grau_cell = GRAUCell(input_size = input_size, hidden_size = hidden_size)


    def forward(self, hid, user_emb, user_hidden):


        return self.grau_cell(hid, user_emb, user_hidden)


class GRAUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRAUCell, self).__init__()


        lb, ub = -math.sqrt(1/hidden_size), math.sqrt(1/hidden_size)
        self.in2hid_w = nn.ParameterList([self.__init(lb, ub, input_size, hidden_size) for _ in range(2)])
        self.hid2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(3)])
        self.in2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(2)])
        self.hid2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])


    @staticmethod
    def __init(low, upper, dim1, dim2=None):
        if dim2 is None:
            return nn.Parameter(torch.rand(dim1) * (upper - low) + low)
        else:
            return nn.Parameter(torch.rand(dim1, dim2) * (upper - low) + low)


    def forward(self, hid, user_emb, user_hidden):

        r = torch.sigmoid(torch.mm(user_emb, self.in2hid_w[0]) + self.in2hid_b[0] +

                          torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0])
        z = torch.sigmoid(torch.mm(user_emb, self.in2hid_w[1]) + self.in2hid_b[1] +

                          torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1])


        n = (torch.tanh((user_hidden
                       + torch.mul(r, (torch.mm(hid, self.hid2hid_w[2]) + self.hid2hid_b[2])))))
        next_hidden = ((torch.mul((1 - z), n) + torch.mul(z, hid)))
        return next_hidden


class GRAUCellNormed(nn.Module):


    def __init__(self, input_size, hidden_size):
        super(GRAUCellNormed, self).__init__()


        lb, ub = -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size)
        self.in2hid_w = nn.ParameterList([self.__init(lb, ub, input_size, hidden_size) for _ in range(2)])
        self.hid2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(3)])
        self.in2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(2)])
        self.hid2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])

        self.norm1 = nn.LayerNorm(normalized_shape=hidden_size)
        self.norm2 = nn.LayerNorm(normalized_shape=hidden_size)
        self.norm3 = nn.LayerNorm(normalized_shape=hidden_size)

    @staticmethod
    def __init(low, upper, dim1, dim2=None):
        if dim2 is None:
            return nn.Parameter(torch.rand(dim1) * (upper - low) + low)
        else:
            return nn.Parameter(torch.rand(dim1, dim2) * (upper - low) + low)

    def forward(self, hid, user_emb, user_hidden):

        r = torch.sigmoid(torch.mm(user_emb, self.in2hid_w[0]) + self.in2hid_b[0] +

                          torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0])
        z = torch.sigmoid(torch.mm(user_emb, self.in2hid_w[1]) + self.in2hid_b[1] +

                          torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1])

        n = self.norm1(torch.tanh((user_hidden
                         + torch.mul(r, self.norm2(torch.mm(hid, self.hid2hid_w[2]) + self.hid2hid_b[2])))))
        next_hidden = self.norm3((torch.mul((1 - z), n) + torch.mul(z, hid)))
        return next_hidden
