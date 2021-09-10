from argparse import Namespace


def get_prefixed_args(args, prefix):
    """Extracts and yields arguments starting with `prefix`; deletes the prefix."""
    new_args = dict()
    for k, v in vars(args).items():
        if k.startswith(f"{prefix}_"):
            new_args[k.replace(f"{prefix}_", "")] = v
    new_args = Namespace(**new_args)
    return new_args
