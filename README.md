[//]: # (![Coverage Report]&#40;./assets/coverage.svg&#41;)

![](./static/torchlogic_logo.png)

# <span style="color:#0E6FFF">TORCHLOGIC</span> DOCUMENTATION

_torchlogic_ is a pytorch framework for developing Neuro-Symbolic AI systems
based on [Weighted Lukasiewicz Logic](https://arxiv.org/abs/2006.13155) that we denote as _Neural Reasoning Networks_.  
The design principles of the _torchlogic_ provide computational efficiency for Neuro-Symbolic AI through
GPU scaling.

### Design Principles

- _Neural == Symbolic_: Symbolic operations should not deviate computationally from Neural operations and leverage PyTorch directly
- _Masked Tensors_: Reasoning Networks use tensors and masking to represent any logical structure _and_ leverage GPU optimized computations
- _Neural -> Symbolic Extension_: Symbolic operations in _torchlogic_ are PyTorch Modules and can therefore integrate with existing Deep Neural Networks seamlessly

With these principles, _torchlogic_ and Neural Reasoning Networks are able
to extend and integrate with our current state-of-the-art technologies that leverage advances in 
Deep Learning.  Neural Reasoning Networks developed with _torchlogic_ can scale with
multi-GPU support.  Finally, those familiar with PyTorch development principles will have only a small step
in skill building to develop with _torchlogic_.

### Documentation

The API reference and additional documentation for torchlogic are available
on through the </link/to/documentation> site.
The current code is in an Alpha state so there may be bugs and the functionality
is expanding quickly.  We'll do our best to keep the documentation up to date
with the latest changes to our API reflected there.

### Tutorial

There are several tutorials demonstrating how to use the R-NRN algorithm
in multiple use cases.

[Tutorial Source](./tutorials/brrn.md)

### Data Science

To understand the basic of the torchlogic framework and Neural Reasoning Networks
check out the [Data Science](./ds/rn.md) section, which gives an introduction to some of the
models developed so far using torchlogic.

