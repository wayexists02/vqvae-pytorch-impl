import torch


def save(context, path):
    state_dict = dict()

    for key in context.keys():
        state_dict[key] = context[key].state_dict()
    
    torch.save(state_dict, path)

    
def load(context, path):
    state_dict = torch.load(path)
    
    for key in state_dict.keys():
        context[key].load_state_dict(state_dict[key])
