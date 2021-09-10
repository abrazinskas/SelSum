def get_child_attr(obj, hier_attr_name):
    """Recursively traverses `par_obj`; returns the sub-child attribute.

    Args
        par_obj:
        hier_attr_name (str): e.g., attr_1.attr_2.attr_3

    Return:
        hierarchical child attribute value.
    """
    if "." not in hier_attr_name:
        return getattr(obj, hier_attr_name)
    attrs = hier_attr_name.split(".")
    curr_attr_val = getattr(obj, attrs[0])
    return get_child_attr(curr_attr_val, ".".join(attrs[1:]))
