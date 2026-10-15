import torch
import torch.nn as nn

class LayerNormGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.Wx = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        self.Wa = nn.Linear(input_dim, 3 * hidden_dim, bias=False)
        self.Wh = nn.Linear(hidden_dim, 3 * hidden_dim)

        self.ln = nn.LayerNorm(3 * hidden_dim)
        self.att = nn.Parameter(torch.zeros(1, hidden_dim))
        self.att_m = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))

    def init_weights(self):

        nn.init.xavier_normal_(self.att_m)
        nn.init.xavier_normal_(self.att)

    def channel_attention(self, *channel_embeddings):
        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(
                    torch.multiply(self.att, torch.matmul(embedding, self.att_m)),
                    1))
        embs = torch.stack(weights, dim=0)
        score = torch.nn.functional.softmax(embs.t(), dim = -1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings

    def forward(self, x, a, h_prev):

        gates = self.ln(self.Wx(x) + self.Wa(a) + self.Wh(h_prev))
        r, z, n = gates.chunk(3, 1)

        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(r * n)
        h_new = (1 - z) * n + z * h_prev
        return h_new


class CustomGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = LayerNormGRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
    def forward(self, x, a):

        outputs = []
        h = torch.zeros(x.size(1), self.hidden_dim).to(x.device)
        for t in range(x.size(0)):
            h = self.cell(x[t], a[t], h)
            outputs.append(h)
        return torch.stack(outputs)


class GRAUCellNormed(nn.Module):


    def __init__(self, input_size, hidden_size):
        super(GRAUCellNormed, self).__init__()


        self.x_to_rz = nn.Linear(input_size, 2 * hidden_size)
        self.hidden_linear = nn.Linear(input_size, 3 * hidden_size)
        self.a_to_hidden = nn.Linear(input_size, 3 * hidden_size)

        self.norm1 = nn.LayerNorm(normalized_shape=hidden_size)
        self.norm2 = nn.LayerNorm(normalized_shape=hidden_size)
        self.norm3 = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(self, hid, user_emb, user_hidden):

        user_r_z = self.x_to_rz(user_emb)
        hid_r_z_n = self.hidden_linear(hid)

        user_r, user_z = user_r_z.chunk(2, 1)
        hid_r, hid_z, hid_n = hid_r_z_n.chunk(3, 1)


        r = torch.sigmoid(user_r + hid_r)
        z = torch.sigmoid(user_z + hid_z)
        n = self.norm1(torch.tanh(user_hidden + r * self.norm2(hid_n)))
        n = self.norm3((torch.mul((1 - z), n) + torch.mul(z, hid)))
        return n


class CustomGRAU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = GRAUCellNormed(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
    def forward(self, x, a):

        outputs = []
        h = torch.zeros(x.size(1), self.hidden_dim).to(x.device)
        for t in range(x.size(0)):
            h = self.cell(h, x[t], a[t])
            outputs.append(h)
        return torch.stack(outputs)


if __name__ == '__main__':

    x = torch.randn(3, 64, 64)

    model = CustomGRU(64, 64)

    out = model(x, x)
    print(out.shape)
