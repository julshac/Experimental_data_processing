

def normalize(f, s):
    return ((f - f.min()) / f.max() - f.min()) * s
