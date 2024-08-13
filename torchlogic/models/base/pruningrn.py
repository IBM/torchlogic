import logging
from torchlogic.models.base import ReasoningNetworkModel


class PruningReasoningNetworkModel(ReasoningNetworkModel):

    def __init__(self):
        """
        Initialize a PruningReasoningNetworkModel
        """
        super(PruningReasoningNetworkModel, self).__init__()
        self.best_state = {}
        self.target_names = None
        self.rn = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def perform_prune(self) -> None:
        """
        Update the nodes in the PruningLogicalNetwork by pruning and growing.
        """
        raise Warning("Abstract method, must be implemented!")


__all__ = [PruningReasoningNetworkModel]
