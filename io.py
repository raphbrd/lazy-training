import pickle


def to_dict(a):
    if not isinstance(a, dict):
        return a.__dict__
    return a


def load_file(f):
    with open(f, "rb") as rb:
        yield to_dict(pickle.load(rb))
        yield pickle.load(rb)
