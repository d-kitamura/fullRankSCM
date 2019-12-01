%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample program for multichannel source separation using full-rank       %
% spatial covariance model                                                %
%                                                                         %
% Coded by D. Kitamura (d-kitamura@ieee.org)                              %
%                                                                         %
% # Original paper                                                        %
% N. Q. K. Duong, E. Vincent, and R. Gribonval, "Underdetermined          %
% reverberant audio source separation using a fullrank spatial covariance %
% model," IEEE Transactions on Audio, Speech, and Language Processing,    %
% vol. 18, no. 7, pp. 1830-1840, May 2010.                                %
%                                                                         %
% See also:                                                               %
% http://d-kitamura.net                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;
addpath('./bss_eval'); % BSS eval is shared under GPLv3 license

% Setting parameters
seed = 1; % pseudo random seed
refMic = 1; % reference mic for performance evaluation using bss_eval_sources
fsResample = 16000; % resampling frequency [Hz]
ns = 2; % number of sources
fftSize = 1024; % window length in STFT [points]
shiftSize = 512; % shift length in STFT [points]
micSpacing = 0.0556; % microhpone spacing [m]
soundSpeed = 344.82; % speed of sound [m/s]
it = 100; % number of iterations (define by checking convergence behavior with drawConv=true)
drawConv = true; % true or false (true: plot cost function values in each iteration and show convergence behavior, false: faster and do not plot cost function values)

% Fix random seed
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed))

% Input data and resample
[sig(:,:,1), fs] = audioread('./input/drums.wav'); % signal x channel x source
[sig(:,:,2), fs] = audioread('./input/piano.wav'); % signal x channel x source
sig_resample(:,:,1) = resample(sig(:,:,1), fsResample, fs, 100); % resampling for reducing computational cost
sig_resample(:,:,2) = resample(sig(:,:,2), fsResample, fs, 100); % resampling for reducing computational cost

% Mixing source images in each channel to produce observed signal
mix(:,1) = sig_resample(:,1,1) + sig_resample(:,1,2);
mix(:,2) = sig_resample(:,2,1) + sig_resample(:,2,2);
if abs(max(max(mix))) > 1.00 % check clipping
    error('Cliping detected.\n');
end

% Reference signals for performance evaluation using bss_eval_sources
src(:,1) = sig_resample(:,refMic,1);
src(:,2) = sig_resample(:,refMic,2);

% Calculate input SDRs and SIRs
inputSDRSIR(1,1) = 10.*log10( sum(sum(squeeze(sig_resample(:,1,refMic)).^2)) ./ sum(sum(squeeze(sig_resample(:,2,refMic)).^2)) );
inputSDRSIR(2,1) = 10.*log10( sum(sum(squeeze(sig_resample(:,2,refMic)).^2)) ./ sum(sum(squeeze(sig_resample(:,1,refMic)).^2)) );

% Multichannel source separation based on full-rank spatial covariance model
[sep,cost] = bss_fullRankSCM(mix,ns,fftSize,shiftSize,it,fsResample,micSpacing,soundSpeed,refMic,drawConv);

% Performance evaluation using bss_eval_sources
[SDR,SIR,SAR] = bss_eval_sources(sep.',src.');
SDRimp = SDR - inputSDRSIR
SIRimp = SIR - inputSDRSIR
SAR

% Output separated signals
outputDir = sprintf('./output');
if ~isdir( outputDir )
    mkdir( outputDir );
end
audiowrite(sprintf('%s/observedMixture.wav', outputDir), mix, fsResample); % observed signal
audiowrite(sprintf('%s/originalSource1.wav', outputDir), src(:,1), fsResample); % source signal 1
audiowrite(sprintf('%s/originalSource2.wav', outputDir), src(:,2), fsResample); % source signal 2
audiowrite(sprintf('%s/estimatedSignal1.wav', outputDir), sep(:,1), fsResample); % estimated signal 1
audiowrite(sprintf('%s/estimatedSignal2.wav', outputDir), sep(:,2), fsResample); % estimated signal 2

fprintf('The files are saved in "./output".\n');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%