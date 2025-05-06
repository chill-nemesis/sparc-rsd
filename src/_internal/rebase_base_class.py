def REBASE_BASE_CLASS(cls, new_base):
    """
    Returns a new base class tuple replacing the original base
    with `new_base`, but preserving methods and attributes.
    """
    # Inherit all class attributes except __dict__ internals
    attrs = {k: v for k, v in cls.__dict__.items() if not (k.startswith("__") and k not in ("__doc__", "__module__"))}
    # Create a new base class with the same name and body
    return type(cls.__name__, (new_base,), attrs)
