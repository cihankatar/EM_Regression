import torch.nn as nn 

class MLP_Regressor(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(MLP_Regressor,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size

        self.linear1 = nn.Linear(input_size,hidden_size)
        self.bnorm1    = nn.BatchNorm1d(hidden_size)

        self.linear2 = nn.Linear(hidden_size,2*hidden_size)
        self.bnorm2    = nn.BatchNorm1d(2*hidden_size)
        
        self.linear3 = nn.Linear(2*hidden_size,hidden_size)
        self.bnorm3    = nn.BatchNorm1d(hidden_size)
        
        self.linear4 = nn.Linear(hidden_size,2)
        
        self.relu    = nn.ReLU()

    def forward (self,X):

        X = self.relu(self.bnorm1(self.linear1(X)))
        X = self.relu(self.bnorm2(self.linear2(X)))
        X = self.relu(self.bnorm3(self.linear3(X)))
        X = self.linear4(X)
        return X 
    


"""
   def forward (self,X):

        X = self.relu(self.linear1(X))
        X = self.relu(self.linear2(X))
        X = self.relu(self.linear3(X))
        X = self.linear4(X)
        return X 
"""