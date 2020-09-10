import torch
from model import Retriever
from processing import prepare, get_loaders, load_data
from ignite.metrics import Loss, Recall
from ignite.utils import setup_logger
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Engine
from sentence_transformers import SentenceTransformer


CUDA = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


retrieval_model = Retriever()

retrieval_model.to(CUDA)
optimizer = torch.optim.Adam(retrieval_model.parameters(), lr=0.0001)
optimizer
loss_fn = torch.nn.L1Loss().to(CUDA)

encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
data = load_data('data/persona_self_original.json')
train_loader, train_utts, val_loader, val_utts = get_loaders(data, encoder, 32, 64)

def train(model, encoder, optimizer, loss_fn, train_loader, max_epochs=5):
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, not_y, y = (encoder.encode(i[0], convert_to_tensor=True).to(CUDA) for i in batch)
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    trainer = Engine(train_step)
    trainer.logger = setup_logger('trainer')
    trainer.run(train_loader, max_epochs=max_epochs)

    return model


def evaluate(model, encoder, metrics, val_loader):
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, not_y, y = (encoder.encode(i[0], covert_to_tensor=True).to(CUDA) for i in batch)
            yhat = model(x)
            return yhat, y
        
        
    evaluator = Engine(eval_step)
    evaluator.logger = setup_logger('evaluator')
    evaluator.run(val_loader)


val_metrics = {'l1 loss': Loss(loss_fn),
               'r1': Recall(average=False)
              }

trained = train(retrieval_model, encoder, optimizer, loss_fn, train_loader)
evaluate(trained, encoder, val_metrics, val_loader)

