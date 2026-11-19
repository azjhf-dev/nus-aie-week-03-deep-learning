"""Train three small neural architectures on deterministic synthetic tasks, CPU only.
No pretrained weights, external datasets, API calls or billing.
"""
import torch
from torch import nn
import numpy as np

torch.set_num_threads(1)


def run_architecture(kind='CNN', epochs=40, seed=12):
    torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed)
    n=320
    if kind=='CNN':
        y=torch.arange(n)%2
        X=torch.randn(n,1,8,8,generator=generator)*.2
        X[y==0,0,2:6,3:5]+=1
        X[y==1,0,3:5,2:6]+=1
        net=nn.Sequential(nn.Conv2d(1,6,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(6*8*8,2))
    elif kind=='RNN':
        X=torch.randn(n,7,1,generator=generator)
        y=(X.sum(dim=(1,2))>0).long()
        class SumRNN(nn.Module):
            def __init__(self):
                super().__init__();self.rnn=nn.RNN(1,12,batch_first=True);self.head=nn.Linear(12,2)
            def forward(self,x):
                out,_=self.rnn(x);return self.head(out[:,-1,:])
        net=SumRNN()
    else:
        X=torch.randn(n,6,3,generator=generator)
        y=(X[:,0,0]+.5*X[:,1,1]>0).long()
        class TinyTransformer(nn.Module):
            def __init__(self):
                super().__init__();self.embed=nn.Linear(3,12);self.position=nn.Parameter(torch.randn(1,6,12)*.05)
                layer=nn.TransformerEncoderLayer(12,3,24,dropout=0,batch_first=True)
                self.encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False);self.head=nn.Linear(12,2)
            def forward(self,x):
                h=self.encoder(self.embed(x)+self.position);return self.head(h[:,0,:])
        net=TinyTransformer()
    # Split before fitting; the seed is fixed and the final 80 rows are held out.
    train_x,test_x=X[:240],X[240:];train_y,test_y=y[:240],y[240:]
    optimizer=torch.optim.Adam(net.parameters(),lr=.015)
    losses=[]
    for _ in range(epochs):
        net.train();optimizer.zero_grad();logits=net(train_x)
        loss=nn.functional.cross_entropy(logits,train_y);loss.backward();optimizer.step();losses.append(float(loss.detach()))
    net.eval()
    with torch.no_grad():
        predicted=net(test_x).argmax(1);accuracy=float((predicted==test_y).float().mean())
        matrix=torch.zeros(2,2,dtype=torch.int64)
        for truth,pred in zip(test_y,predicted):matrix[truth,pred]+=1
    return {'architecture':kind,'epochs':epochs,'loss':losses,'held_out_accuracy':accuracy,'confusion':matrix.numpy().tolist(),'parameters':sum(p.numel() for p in net.parameters()),'test_n':len(test_y),'seed':seed}

if __name__=='__main__':
    import json
    print(json.dumps([run_architecture(k) for k in ['CNN','RNN','Transformer']],indent=2))
