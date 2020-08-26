# imageSeg-2.5_topo

This is another implementation of topological loss, containing both 2D and 2.5D version. 


# Tips for train with topological loss

The computation of persistent homology is run on CPU and kind of slow. So in order to make it work, actually I adopted lots of engineering tricks during training:

1) Use pretrained models: pretrain the model without topo loss until obtaining reasonable likelihood maps and then finetune with topo loss to fix specific positions.

2) Small patches: apply topo loss on small patches instead of whole images to reduce computation cost.

3) Reduce unnecessary computation: during the finetune step, skip the topo loss computation if the likelihood map is already acceptable (checking by betti number).
