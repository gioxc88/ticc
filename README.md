# Toeplitz Inverse Covariance Clustering (TICC)
A `sklearn` compatible implementation of the TICC algorithm.  
All credits to go the author. You can find the original repo [here](https://github.com/davidhallac/TICC).   
TICC is a python solver for efficiently segmenting and clustering a multivariate time series. 
It takes as input a T-by-n data matrix, a regularization parameter `lambda` and smoothness parameter `beta`, 
the window size `w` and the number of clusters `k`.  
TICC breaks the T timestamps into segments where each segment belongs to one of the `k` clusters. 
The total number of segments is affected by the smoothness parameter `beta`. 
It does so by running an EM algorithm where TICC alternately assigns points to cluster using a dynamic programming algorithm and updates the cluster parameters by solving a Toeplitz Inverse Covariance Estimation problem. 

For details about the method and implementation see the paper [1].

## References
[1] D. Hallac, S. Vare, S. Boyd, and J. Leskovec [Toeplitz Inverse Covariance-Based Clustering of
Multivariate Time Series Data](http://stanford.edu/~hallac/TICC.pdf) Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 215--223
