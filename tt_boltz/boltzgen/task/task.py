from abc import ABC, abstractmethod

from omegaconf import OmegaConf


class Task(ABC):
    """A task to be executed."""

    @abstractmethod
    def run(self, config: OmegaConf) -> None:
        """Run the task.

        Parameters
        ----------
        config : OmegaConf
            The configuration for the task.

        """
        raise NotImplementedError
