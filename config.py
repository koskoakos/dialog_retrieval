
class RetrieverConfig:
    device = 'cuda'
    data_source = 'data/personachat_self_original.json'
    checkpoint = 'checkpoint.pth'

    context_length = 1
    negative_samples = 1
    negative_weight = 0.5

    batch_size = 32

    LR = 2e-5
    max_epochs = 10

    sentence_transformer = 'distilbert-base-nli-stsb-mean-tokens'

    def update(self, other):
        for k, v in other.items():
            setattr(self, k, v)

    def __iter__(self):
        return 
