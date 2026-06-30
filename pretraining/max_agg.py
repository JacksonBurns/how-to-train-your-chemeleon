import torch
from torch import Tensor
from chemprop.nn import Aggregation, AggregationRegistry


@AggregationRegistry("max")
class MaxAggregation(Aggregation):
    """
    Constructs a graph-level representation by taking the element-wise maximum 
    of the node-level representations across each graph in the batch.
    
    This is the pooling strategy explicitly recommended by the MiniMol paper 
    for extracting the final molecular fingerprint from their GINE backbone.
    """
    def forward(self, H: Tensor, batch: Tensor) -> Tensor:
        """
        Aggregate the graph-level representations of a batch of graphs.
        
        Parameters
        ----------
        H : Tensor
            A tensor of shape `V x d` containing the batched node-level representations
        batch : Tensor
            A tensor of shape `V` containing the index of the graph a given vertex corresponds to.
            
        Returns
        -------
        Tensor
            A tensor of shape `B x d` containing the graph-level representations.
        """
        # Handle the edge case of an empty batch
        if batch.numel() == 0:
            return torch.zeros(0, H.shape[1], dtype=H.dtype, device=H.device)
            
        # Calculate the number of graphs in the batch
        dim_size = int(batch.max().item()) + 1
        
        # Initialize output tensor.
        # Empty graphs (0 nodes) will retain these initial 0 values.
        out = torch.zeros(dim_size, H.shape[1], dtype=H.dtype, device=H.device)
        
        # Expand batch indices to match hidden dimensions: [V, d]
        index = batch.unsqueeze(1).expand_as(H)
        
        # Use scatter_reduce_ with reduce="amax". 
        # Setting include_self=False is critical: it ensures that even if all node 
        # features in a graph are negative, the true maximum is retained rather 
        # than falsely defaulting to the initialized 0.
        out.scatter_reduce_(self.dim, index, H, reduce="amax", include_self=False)
        
        return out
