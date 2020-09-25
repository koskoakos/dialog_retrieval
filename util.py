from config import RetrieverConfig as Config
from processing import query

def interact(model, encoder, latent_space, utterance_map, context_length=1):
    history = ['']
    while True:
        request = input()
        if request == '/stop':
            return
        if request == '/reset':
            history = ['']
        if request.startswith('/set context_length'):
            context_length = int(request.partition(' ')[2])
        
        history.append(request)
        request = encoder.encode(history[-context_length:], convert_to_tensor=True)
        vectors, distances = query(model, request, latent_space, utterance_map, k=3)
        for vec, _ in zip(vectors, distances):
            best = utterance_map[tuple(vec)]
            history.append(best)
            print(best)
