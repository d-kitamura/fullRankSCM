# Full-Rank Spatial Covariance Model

## About
Sample MATLAB script for full-rank spatial covariance model (full-rank SCM) and its application to multichannel audio source separation.

## Contents
- bss_eval [dir]:           open source (GPLv3 license) for evaluating audio source separation performance
- input [dir]:              includes test audio signals (reverberation time is around 300 ms)
- bss_fullRankSCM.m:        apply pre- and post-processing for multichannel source separation (STFT, initialization, full-rank SCM estimation, permutation solver, multichannel Wiener filtering, and ISTFT)
- fullRankSCM.m:            estimation of full-rank SCM based on expectation-maximization algorithm
- fullRankSCM_permSolver.m: permutation solver
- fullRankSCM_readable.m:   estimation of full-rank SCM based on expectation-maximization algorithm (slow but somewhat readable implementation)
- ISTFT.m:			        inverse short-time Fourier transform
- main.m:			        main script with parameter settings
- STFT.m:       			short-time Fourier transform

## Usage Note
Implementation is quite heuristic and not readable. This is because the direct implementation of full-rank SCM estimation is very slow. In particular, for loop should be avoided as much as possible in MATLAB. One example is to calculate time-frequency-wise inverse matrix. Straightforward implementation is just looping inv(A) function, but it is quite slow. To speed up this kind of calculation, in this script, the inverse matrix is directly described using a well-known determinant formula. Stupid, but it's fast.

## Original paper
Full-rank SCM model was proposed in 
* N. Q. K. Duong, E. Vincent, and R. Gribonval, "Underdetermined reverberant audio source separation using a fullrank spatial covariance model," IEEE Transactions on Audio, Speech, and Language Processing, vol. 18, no. 7, pp. 1830-1840, May 2010.

The permutation solver used in this script was proposed in
* H. Sawada et.al., "Grouping separated frequency Components by estimating propagation model parameters in frequency-domain blind source separation", IEEE Transactions on Audio, Speech, and Language Processing, vol.. 15, no. 5, pp 1592-1604, 2007.

## See Also
* HP: http://d-kitamura.net