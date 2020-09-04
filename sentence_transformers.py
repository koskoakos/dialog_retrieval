import json
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
import time
import tqdm 
from sklearn.neighbors import BallTree
import numpy as tf
from sentence_transformers import SentenceTransformer, util
import pickle 


LR=2e-5
K_OUT = 20
FREEZE = False
CHECK_P = 'trained.tar'
original = 'data/persona_self_original.json'
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PersonaData(Dataset):
    def __init__(self, data, encoder, **enc_kw):
        self.data = data
        self.encoder = encoder
        self.encoder_args = enc_kw

    def __getitem__(self, index):
        context = self.data[index]['history']
        target = self.data[index]['candidates'][-1]
        negative = self.data[index]['candidates'][:-1]
        return tuple(map(self.encode, (context, target, negative)))

    def encode(self, sample):
        return self.encoder.encode(sample, **self.encoder_args)

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
    distons, indoks = space.query(outputs, k=3)
    return distons[0], indoks[0]


def tok(data, tokenizer):
    return tokenizer(data, padding=True, truncation=True, return_tensors='pt').to(device)


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

def train(model, train_loader, val_data, epoch_start, epoch_end):
    model.train()
    for e in range(epoch_start, epoch_end):

        for i, data in enumerate(tqdm.tqdm(train_loader)):
            contexts, targets = data

            contexts = tokenizer(contexts, padding=True, truncation=True, return_tensors='pt').to(device)
            targets = tokenizer(targets, padding=True, truncation=True, return_tensors='pt').to(device)
            
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

class Sequencer(torch.nn.Module):
    def __init__(self, embedder, input_size=768, out_len=768, hidden_size=100, batch_size=32):
        super(Sequencer, self).__init__()
        self.embedder = embedder
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size * 5, out_len)
        self.hidden_cell = (torch.zeros(1, batch_size, hidden_size), 
                            torch.zeros(1, batch_size, hidden_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        linear_out = self.linear(lstm_out.reshape(lstm_out.size(0), -1))
        return linear_out


if __name__ == '__main__':

    writer = SummaryWriter()
    
    with open(original, 'r') as j_in:
        P = json.load(j_in)
    
    ember = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    
    train_data, BIG_LIST_OF_U = prepere(P['train'], 
                                        lambda u: len(u['history']) == 5)
    pdata = PersonaData(train_data, ember, convert_to_tensor=True)
    train_loader = DataLoader(
        pdata,
        batch_size=32, 
        shuffle=True,
        drop_last=True)

    try:
        with open('embedded_corpus.pkl', 'rb') as pickled_corpus:
            corpus = pickle.load(pickled_corpus)
    except:
        import traceback
        traceback.print_exc()
        corpus = ember.encode(BIG_LIST_OF_U, convert_to_tensor=True, show_progress_bar=True)

        with open('embedded_corpus.pkl', 'wb') as pickled_out:
            pickle.dump(corpus, pickled_out, protocol=-1)

    model = Sequencer(ember)
    optimizer = torch.optim.Adam(lr=LR, params=model.parameters())
    EPOCHS = 10
    loss_fn = torch.nn.MSELoss()

    for e_ix in range(EPOCHS):
        running_loss = 0
        with tqdm.tqdm(total=len(train_loader.dataset), desc=f'[Epoch {e_ix+1:2d}/{EPOCHS}]') as progress_bar:
            for batch_ix, (x, y, not_y) in enumerate(train_loader):
                optimizer.zero_grad()

                model.hidden_cell = (torch.zeros(1, 32, model.hidden_size),
                                     torch.zeros(1, 32, model.hidden_size))

                yhat = model(x.to(device))
                
                loss = loss_fn(y.to(device), yhat)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                progress_bar.set_postfix({'loss': running_loss/(batch_ix+1)})
                progress_bar.update(x.shape[0])
            
            train_loss = running_loss/len(train_loader)

            progress_bar.set_postfix({'loss': train_loss})


    history = [''] * 5
    while True:
        request = ember.encode(input(), convert_to_tensor=True)
        
        if request == '/reset':
            history = ['']
            continue
        
        if request == '/stop':
            break

        context = history[-5:]

        predicted_embedding = model(context)
        
        similar_cos = util.pytorch_cos_sim(predicted_embedding, corpus)[0].cpu()
        best = tf.argpartition(-similar_cos, range(5))[:5]
        for idx in best:
            print(BIG_LIST_OF_U[idx].strip(), "(Score: %.4f)" % (similar_cos[idx]))
    
