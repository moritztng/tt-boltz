from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from tt_boltz.boltzgen.data.data import Tokenized


class Cropper(ABC):
    """Abstract base class for cropper."""

    @abstractmethod
    def crop(
        self,
        data: Tokenized,
        max_tokens: int,
        random: np.random.Generator,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.Generator
            The random state for reproducibility.
        max_atoms : Optional[int]
            The maximum number of atoms to consider.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        raise NotImplementedError
