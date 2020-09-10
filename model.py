import torch


class Retriever(torch.nn.Module):
    def __init__(self, in_dim=768, hid_dim=1024, out_dim=768):
        super(Retriever, self).__init__()
        self.input = torch.nn.Linear(in_dim, hid_dim)
        self.linear = torch.nn.Linear(hid_dim, hid_dim)
        self.output = torch.nn.Linear(hid_dim, out_dim)

    def forward(self, X):
        
        X = self.input(X).view(X.size(0), -1)
        X = self.linear(X).view(X.size(0), -1)
        return self.output(X).view(X.size(0), -1)
