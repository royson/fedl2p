from importlib import import_module

def get_func_from_config(class_type: str):
    ''' The YAML config allows certain elements to specify types of
    pytorch objects or functions. This method reads a `class` config
    element and returns it. (e.g. reads {'class': "torch.optim.SGD"},
    will return a torch.optim.SGD function) '''

    module_name, class_name = class_type["class"].rsplit(".", 1)
    return getattr(import_module(module_name), class_name)
