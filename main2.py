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
loss_fn = torch.nn.L1Loss(reduction='sum').to(CUDA)

encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

data = load_data('data/persona_dev.json')
train_loader, train_utts, val_loader, val_utts = get_loaders(data, encoder, 64, 64)

def run(model):
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, not_ys, y = batch
        x = encoder.encode(x[0], convert_to_tensor=True).to(CUDA)
        not_ys = [encoder.encode(not_y, convert_to_tensor=True).to(CUDA) for not_y in not_ys]
        y = encoder.encode(y, convert_to_tensor=True).to(CUDA)
        yhat = model(x)
        loss = loss_fn(yhat, y)
        gains = torch.sum(torch.stack([loss_fn(not_y, y) for not_y in not_ys])) / 20.0
        loss /= gains
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, not_ys, y = batch
            x = encoder.encode(x[0], convert_to_tensor=True).to(CUDA)
            not_ys = [encoder.encode(not_y, convert_to_tensor=True).to(CUDA) for not_y in not_ys]
            y = encoder.encode(y, convert_to_tensor=True).to(CUDA)
            yhat = model(x)
            return yhat, y
    
    trainer = Engine(train_step)
    trainer.logger = setup_logger('trainer')

    evaluator = Engine(eval_step)
    evaluator.logger = setup_logger('evaluator')
    
    l1 = Loss(loss_fn)
    
    l1.attach(evaluator, 'l1')
    
    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def log_training(engine):
        batch_loss = engine.state.output
        lr = optimizer.param_groups[0]['lr']
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration
        print("Epoch {}/{} : {} - batch loss: {}, lr: {}".format(e, n, i, batch_loss, lr))

    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print(f"Training Results - Epoch: {engine.state.epoch} " 
              f"L1: {metrics['l1']:.2f}")
        
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch: {engine.state.epoch} "
              f"L1: {metrics['l1']:.2f}")

    trainer.run(train_loader, max_epochs=10)
        
run(retrieval_model)
