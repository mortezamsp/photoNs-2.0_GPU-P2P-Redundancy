This is the application of P2P-Redundancy in PhotoNs, a cosmological simulation that utilizes the Fast Multipole Method (FMM) for short-range interactions. 

The main repository for PhotoNs can be found at "https://github.com/nullike/photoNs-2.0," and the related paper, titled "A Hybrid Fast Multipole Method for Cosmological N-body Simulations," is available at "https://arxiv.org/pdf/2006.14952."

P2P-Redundancy is a under-developing technique initially outlined in the paper "Optimizing Near Field Computation in the MLFMA Algorithm with Data Redundancy and Performance Modeling on a Single GPU," with the PDF version accessible at "https://arxiv.org/abs/2403.01596."

This technique specifically accelerates P2P interactions on the GPU through redundancy. 
The basic implementation is referred to as "Indexing," which represents a straightforward GPU approach, while the enhanced version is called "Redundant," which involves duplicating data in global memory for threads. 
Since PhotoNs employs a dual buffering technique to overlap tree walks with computations, there are two variations of both the indexing and redundant methods: one that utilizes dual buffering and another that executes tree walks and computations serially for clearer analytical modeling.
