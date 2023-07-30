import torch

models = {}
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

# decorator 修饰器的作用 register_model的作用
def get_model(name, num_cls=10, **args):
    net = models[name](num_cls=num_cls, **args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net


