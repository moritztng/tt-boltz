import torch


def get_dropout_mask(
    dropout: float,
    z: torch.Tensor,
    training: bool,
    columnwise: bool = False,
) -> torch.Tensor:
    """Get the dropout mask.

    Parameters
    ----------
    dropout : float
        The dropout rate
    z : torch.Tensor
        The tensor to apply dropout to
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        Whether to apply dropout columnwise

    Returns
    -------
    Tensor
        The dropout mask

    """
    dropout = dropout * training
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    d = torch.rand_like(v) > dropout
    d = d * 1.0 / (1.0 - dropout)
    return d


def get_dropout_mask_columnwise(
    dropout: float,
    z: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """Get the dropout mask.

    Parameters
    ----------
    dropout : float
        The dropout rate
    z : torch.Tensor
        The tensor to apply dropout to
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        Whether to apply dropout columnwise

    Returns
    -------
    Tensor
        The dropout mask

    """
    dropout = dropout * training
    v = z[:, 0:1, :, 0:1]
    d = torch.rand_like(v) > dropout
    d = d * 1.0 / (1.0 - dropout)
    return d


def get_dropout_mask_rowise(
    dropout: float,
    z: torch.Tensor,
    training: bool,
) -> torch.Tensor:
    """Get the dropout mask.

    Parameters
    ----------
    dropout : float
        The dropout rate
    z : torch.Tensor
        The tensor to apply dropout to
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        Whether to apply dropout columnwise

    Returns
    -------
    Tensor
        The dropout mask

    """
    dropout = dropout * training
    v = z[:, :, 0:1, 0:1]
    d = torch.rand_like(v) > dropout
    d = d * 1.0 / (1.0 - dropout)
    return d
