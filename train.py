import torch
from argparse import ArgumentParser
from model import Retriever
from config import RetrieverConfig as Config
from processing import prepare, get_loaders, load_data, encode_plus
from ignite.metrics import Loss, Recall
from ignite.utils import setup_logger
from ignite.engine import Engine, Events
from sentence_transformers import SentenceTransformer
from torch.utils.tensorboard import SummaryWriter
from metrics import RecallAt
from sklearn.neighbors import BallTree


def run():
    writer = SummaryWriter()

    CUDA = Config.device
    model = Retriever()

    model.to(CUDA)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    loss_fn = torch.nn.L1Loss().to(CUDA)

    encoder = SentenceTransformer(Config.sentence_transformer)

    data = load_data(Config.data_source)
    train_loader, train_utts, val_loader, val_utts = get_loaders(data, encoder, Config.batch_size)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, not_ys, y = encode_plus(batch, encoder, CUDA)

        yhat = model(x)
        loss = loss_fn(yhat, y)
        gains = loss_fn(not_ys, torch.stack([y] * not_ys.size(0))) * Config.negative_weight
        loss -= gains

        loss.backward()
        optimizer.step()
        return loss.item()
    
    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, _, y = encode_plus(batch, encoder, CUDA)
            yhat = model(x)
            return yhat, y
    
    trainer = Engine(train_step)
    trainer.logger = setup_logger('trainer')

    evaluator = Engine(eval_step)
    evaluator.logger = setup_logger('evaluator')
    
    utterance_map = BallTree(encoder.encode(train_utts))

    l1 = Loss(loss_fn)

    recall = RecallAt(utterance_map)

    recall.attach(evaluator, 'recall')
    l1.attach(evaluator, 'l1')
    
    @trainer.on(Events.ITERATION_COMPLETED(every=5))
    def log_training(engine):
        batch_loss = engine.state.output
        lr = optimizer.param_groups[0]['lr']
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration
        print("Epoch {}/{} : {} - batch loss: {}, lr: {}".format(e, n, i, batch_loss, lr))
        writer.add_scalar('Training/loss', batch_loss, i)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        recall = metrics['recall']
        print(f"Training Results - Epoch: {engine.state.epoch} " 
              f" L1: {metrics['l1']:.2f} "
              f" R@1: {recall['r1']:.2f} "
              f" R@3: {recall['r3']:.2f} ")
        writer.add_scalar('Training/Avg loss', metrics['l1'], engine.state.epoch)
        writer.add_scalar('Training/R@1', recall['r1'], engine.state.epoch)
        
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Validation Results - Epoch: {engine.state.epoch} "
              f"L1: {metrics['l1']:.2f}")
        writer.add_scalar('Validation/Avg loss', metrics['l1'], engine.state.epoch)

    trainer.run(train_loader, max_epochs=Config.max_epochs)


if __name__ == '__main__':
    paper = ArgumentParser()
    for element in dir(Config):
        if element.startswith('__'):
            continue
        paper.add_argument(f'--{element}', default=getattr(Config, element))
    paper.add_argument('--checkpoint')
    args = paper.parse_args()
    for arg, value in vars(args).items():
        setattr(Config, arg, value)
    run()

