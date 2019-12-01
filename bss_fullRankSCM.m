function [sep,cost] = bss_fullRankSCM(mix,ns,fftSize,shiftSize,it,fs,micSpacing,soundSpeed,refMic,drawConv)
%
% bss_fullRankSCM: Multichannel source separation based on full-rank spatial covariance model
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% # Original paper
% N. Q. K. Duong, E. Vincent, and R. Gribonval, "Underdetermined
% reverberant audio source separation using a fullrank spatial covariance
% model," IEEE Transactions on Audio, Speech, and Language Processing,
% vol. 18, no. 7, pp. 1830-1840, May 2010.
%
% see also
% http://d-kitamura.net
%
% [syntax]
%   [sep,cost] = bss_fullRankSCM(mix,ns,fftSize,shiftSize,it,fs,micSpacing,soundSpeed,refMic,drawConv)
%
% [inputs]
%          mix: observed mixture (len x mic)
%           ns: number of sources (scalar)
%      fftSize: window length in STFT (scalar)
%    shiftSize: shift length in STFT (scalar)
%           it: number of iterations (scalar)
%           fs: sampling frequency ([Hz], scalar)
%   micSpacing: distance between microphones ([m], scalar)
%   soundSpeed: speed of sound ([m/s], scalar)
%       refMic: refarence microphone for output
%     drawConv: plot cost function values in each iteration or not (true or false)
%
% [outputs]
%          sep: estimated signals (length x channel x ns)
%         cost: convergence behavior of cost function in full-rank spatial covariance model (it+1 x 1)
%

% Short-time Fourier transform
[X, window] = STFT(mix,fftSize,shiftSize,'hamming');
[I,J,M] = size(X); % fftSize/2+1 x time frames x mics

% Full-rank spatial covariance model
[~,v,R,cost] = fullRankSCM(X,ns,it,drawConv);
% [~,v,R,cost] = fullRankSCM_readable(X,ns,it,drawConv);

% Solve permutation problem
[~,perm] = fullRankSCM_permSolver( R, fs, micSpacing, soundSpeed, drawConv );
tmpR = zeros(size(R));
tmpv = zeros(size(v));
for i = 1:I
    tmpR(:,:,i,:) = R(:,:,i,perm(:,i));
    tmpv(i,:,:) = v(i,:,perm(:,i));
end
R = tmpR;
v = tmpv;

% Multichannel Wienner filtering
Se = zeros(M, I, J, ns);
for i = 1:I
    for j = 1:J
        vR = zeros(M,M);
        for n = 1:ns
            vR = vR + v(i,j,n) * R(:,:,i,n);
        end
        for n = 1:ns
            Se(:,i,j,n) = v(i,j,n) * R(:,:,i,n) * (vR^(-1)) * reshape(X(i,j,:),size(X,3),1);
        end
    end
end
Y = squeeze(Se(refMic,:,:,:)); % I x J x N

% Inverse STFT for each source
sep = ISTFT(Y, shiftSize, window, size(mix,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%