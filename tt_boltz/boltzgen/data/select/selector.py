from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from tt_boltz.boltzgen.data.data import Tokenized


class Selector(ABC):
    """Abstract base class for cropper."""

    @abstractmethod
    def select(
        self,
        data: Tokenized,
        random: np.random.Generator,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
    ) -> Tokenized:
        """Select which part of tokenized should be designed

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        random : np.random.Generator
            The random state for reproducibility.
        
        Returns
        -------
        Tokenized
            The data with updated design entry.

        """
        raise NotImplementedError
