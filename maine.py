import json
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
import time
import tqdm 
from sklearn.neighbors import BallTree
import numpy as tf


LR=2e-5
K_OUT = 20
FREEZE = False
CHECK_P = 'trained.tar'
original = 'data/persona_self_original.json'
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PersonaData(Dataset):
    def __init__(self, data, entokener):
        self.data = data
        self.tokenizer = entokener

    def __getitem__(self, index):
        return self.data[index]['history'], self.data[index]['candidates']

    def tokize(self, sample):
        return self.tokenizer(sample, padding=True, truncation=True, return_tensors='pt')

    def __len__(self):
        return len(self.data)

def prepere(data, len_fun):
    blou = []
    dialogs = []
    for dialog in data:
        for utt in dialog['utterances']:
            if not len_fun(utt):
                continue
            utt['history'] = ['' if u == '__ SILENCE __' else u 
                    for u in utt['history']]
            blou.append(utt['candidates'][-1])
            dialogs.append(utt)
    return dialogs, blou


def embed(data, encodyr, tokenizer):
    vectyr = torch.cat([encodyr(**tok(u, tokenizer)) for u in data])
    return vectyr.detach().cpu().numpy()


def infer(model, context, space, tokenizer):
    inputs = tok(context, tokenizer)
    outputs = model(**inputs)[1].detach().cpu().numpy().reshape(1, -1)
    print(outputs)
    distons, indoks = space.query(outputs, k=3)
    return distons[0], indoks[0]


def tok(data, tokenizer):
    return tokenizer(data, padding=True, truncation=True, return_tensors='pt')


def validate(model, val_data, utter_space, tokenizer):
    for ix, data in enumerate(val_data):
        context = data['history']
        target = data['candidates'][-1]
        response = infer(model, context, utter_space, tokenizer)
        target = model(**inputs)[1].detach().cpu().numpy().reshape(1, -1)


def save_checkpoint(path, model, optimizer, loss, epoch):
    torch.save({
        'epoch': epoch + 1, 
        'model_state_dict': model.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'out_dim': model.out_dim
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = BertRetrieval(checkpoint['out_dim'], FREEZE)
    optimizer = torch.optim.Adam(lr=2e-5, params=model.parameters())
    loss = checkpoint['loss']
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    return model, optimizer, loss, epoch


def u_map(blou, model, tokenizer):
    with torch.no_grad():
        vectors = embed(blou, model, tokenizer)
    utter_space = BallTree(vectors)
    return utter_space 


class BertRetrieval(torch.nn.Module):
    def __init__(self, out_dim, freeze_bert=True):
        super(BertRetrieval, self).__init__()
        self.out_dim = out_dim
        self.ber = BertModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for param in self.ber.parameters():
                param.requires_grad=False
        self.dense = torch.nn.Linear(768, out_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, sense = self.ber(input_ids, attention_mask, token_type_ids)
        condensed = self.dense(sense)
        return condensed

def trayn(model, train_loader, val_data, epoch_start, epoch_end):
    model.train()
    for e in range(epoch_start, epoch_end):
        for i, data in enumerate(tqdm.tqdm(train_loader)):
            contexts, targets = data
            targets = targets[-1]
            contexts = [c for c in contexts]
            contexts, targets = tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')
            
            outputs = model(**contexts)
            with torch.no_grad():
                targets = model(**targets)

            optimizer.zero_grad()

            loss = torch.mean(torch.cdist(outputs, targets))
            
            loss.backward()
            optimizer.step()
            if i % 1000 == 1:
                print(f'E={e}, Loss={loss}')

    save_checkpoint(CHECK_P, model, optimizer, loss, e)

if __name__ == '__main__':

    writer = SummaryWriter()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    with open(original, 'r') as j_in:
        P = json.load(j_in)
    
    train_data, BIG_LIST_OF_U = prepere(P['train'], 
                                        lambda u: len(u['history']) == 5)
    train_loader = DataLoader(
        PersonaData(train_data, tokenizer),
        batch_size=32, 
        shuffle=True,
        drop_last=True)
    
    val_loader = DataLoader(P['valid'])
    
    try:
        model, optimizer, loss, epoch = load_checkpoint(CHECK_P)
        print(f'Loaded model on epoch={epoch}')
    except:
        import traceback
        traceback.print_exc()
        print('Creating a new model')
        model = BertRetrieval(K_OUT, FREEZE)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
        epoch = 0
    
    model.to(device)
    
    if epoch < 1:
        t_start = time.time()
        print('Train start')
        model.to(device)
        trayn(model, train_loader, val_loader, epoch, epoch+1)
        print(f'Train end {t_start-time.time()}s')
    
    try:
        import pickle
        with open('utter_map.json', 'rb') as u_in:
            utter_space = pickle.load(u_in)
        print('Loaded prebuilt utterance vectors')
    except:
        utter_space = u_map(BIG_LIST_OF_U, model, tokenizer)
        with open('utter_map.json', 'wb') as u_out:
            pickle.dump(utter_space, u_out, protocol=-1)
        print(f'Saved {len(BIG_LIST_OF_U)} utterances')
    
    history = ['']
    while True:
        request = input()
        
        if request == '/reset':
            history = ['']
            continue
        
        if request == '/stop':
            break
    
        history.append(request)
    
        distance, index = infer(model, history, utter_space, tokenizer)
        responses = [BIG_LIST_OF_U[i] for i in index]
        history.append(responses[0])
        for i in 0, 1, 2:
            seo = '\t'*i
            print(f'{seo} - {responses[i]} {distance[i]}')
    
