import torch
from model import Retriever
from processing import prepare, get_loaders, load_data
from ignite.metrics import Loss, Recall
from ignite.utils import setup_logger
from ignite.engine import Engine, Events
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

def run(model):
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, not_y, y = (encoder.encode(i[0], convert_to_tensor=True).to(CUDA) for i in batch)
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, not_y, y = (encoder.encode(i[0], convert_to_tensor=True).to(CUDA) for i in batch)
            yhat = model(x)
            return yhat, y
    
    trainer = Engine(train_step)
    trainer.logger = setup_logger('trainer')

    evaluator = Engine(eval_step)
    evaluator.logger = setup_logger('evaluator')
    
    l1 = Loss(loss_fn)
    r1 = Recall(average=False)
    
    
    l1.attach(evaluator, 'l1')
    r1.attach(evaluator, 'r1')
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        los = metrics['l1']
        avg_rec = metrics['r1']
        print(f"Training Results - Epoch: {engine.state.epoch} " 
              f"L1: {metrics['l1']:.2f} R1: {metrics['r1']:.2f}")
        
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch: {engine.state.epoch} "
              f"L1: {metrics['l1']:.2f} R1: {metrics['r1']:.2f}")



    
    trainer.run(train_loader, max_epochs=10)


        
run(retrieval_model)
