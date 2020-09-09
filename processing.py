import json
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util


def load_data(path):
    with open(path) as j_in:
        return json.load(j_in)


def prepare(data, len_fun):
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
 

class PersonaData(Dataset):
    def __init__(self, data, encoder=None, encoder_kw={}):
        self.data = data
        self.encoder = encoder
        self.encoder_kw = encoder_kw

    def __getitem__(self, index):
        context = self.data[index]['history']
        target = self.data[index]['candidates'][-1]
        negatives = self.data[index]['candidates'][:-1]
        return context, negatives, target

    def encode(self, sample):
        return sample if self.encoder is None else self.encoder(sample, **self.encoder_kw)
        
    def __len__(self):
        return len(self.data)


def get_loaders(data, encoder, train_batch_size, val_batch_size, predicate=(lambda u: len(u['history']) == 1)):
    train_data, train_utts = prepare(data['train'], predicate)
    val_data, val_utts = prepare(data['valid'], predicate)
    train_loader = DataLoader(
        PersonaData(train_data),
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True
        )
    val_loader = DataLoader(
        PersonaData(val_data),
        batch_size=val_batch_size,
        shuffle=True,
        drop_last=True
    )
    return train_loader, train_utts, val_loader, val_utts
