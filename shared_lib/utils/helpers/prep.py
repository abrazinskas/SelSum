def partition(data, train_part=0.8, val_part=0.1, test_part=0.1):
    """Splits groups into training, validation, and test partitions.

    Args:
        data (list): list of units (e.g. dicts).
        train_part (float): proportion in [0, 1] of units for training.
        val_part (float): self-explanatory.
        test_part (float): self-explanatory.
    """
    assert train_part + val_part + test_part == 1.

    total_size = len(data)
    train_part_end = int(total_size * train_part)
    val_part_end = train_part_end + int(total_size * val_part)

    train_groups = data[:train_part_end]
    val_groups = data[train_part_end:val_part_end]
    if test_part == 0.:
        val_groups += data[val_part_end:]
        test_groups = []
    else:
        test_groups = data[val_part_end:]

    return train_groups, val_groups, test_groups
