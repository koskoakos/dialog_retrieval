import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, util


def load_data(path):
    with open(path) as j_in:
        return json.load(j_in)


def prepare(data, encoder, min_length):
    blou = []
    dialogs = []
    for dialog in data:
        blou.extend(dialog['utterances'][-1]['history'])
        blou.append(dialog['utterances'][-1]['candidates'][-1])
        for utt in dialog['utterances']:
            if len(utt['history']) < min_length:
                utt['history'][:min_length] = [''] * (min_length - len(utt['history'])) + utt['history']

            utt['history'][:] = encoder.encode(['' if u == '__ SILENCE __' else u 
                    for u in utt['history']], convert_to_tensor=True)

            utt['candidates'][:] = encoder.encode(utt['candidates'], convert_to_tensor=True)

            dialogs.append(utt)
    u_map = {tuple(encoder.encode(u)):u for u in blou}
    return dialogs, u_map
 

class PersonaData(Dataset):
    def __init__(self, data, context_length=1, encoder=None, encoder_kw={}):
        self.data = data
        self.context_length = context_length
        self.encoder = encoder
        self.encoder_kw = encoder_kw

    def __getitem__(self, index):
        context = self.data[index]['history'][-self.context_length:]
        target = self.data[index]['candidates'][-1]
        negatives = self.data[index]['candidates'][0:1]
        return context, negatives, target

    def encode(self, sample):
        return sample if self.encoder is None else self.encoder(sample, **self.encoder_kw)
        
    def __len__(self):
        return len(self.data)


def encode_plus(batch, encoder, device):
    x, not_ys, y = batch
    x = encoder.encode(x[0], convert_to_tensor=True).to(device)
    not_ys = encoder.encode(not_ys[0], convert_to_tensor=True).unsqueeze(0).to(device)
    y = encoder.encode(y, convert_to_tensor=True).to(device)
    return x, not_ys, y


def get_loaders(data, encoder, batch_size, context_length=1):
    train_data, train_utts = prepare(data['train'], encoder, context_length)
    val_data, val_utts = prepare(data['valid'], encoder, context_length)
    train_loader = DataLoader(
        PersonaData(train_data, context_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
        )
    val_loader = DataLoader(
        PersonaData(val_data, context_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    return train_loader, train_utts, val_loader, val_utts


def query(model, contexts, space, u_map, k=1):
    """
    Parameters:
        request: array-like
            Embedded dialog histories
        space: BallTree (or k-dTree)
            Space of all embedded dialog utterances
        k: int
            Amount of nearest vectors to return
    Returns:
        (indices, distances): ([int], [float])
            Indices and distances (sorted) to the nearest vectors
    """
    outputs = model(contexts).detach().cpu().numpy().reshape(1, -1)
    distances, indices = space.query(outputs, k)
    entries = space.get_arrays()[0][indices[0]]
    return entries, distances
