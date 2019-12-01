function [A,perm] = fullRankSCM_permSolver(R,fs,micSpacing,soundSpeed,drawConv)
%
% fullRankSCM_permSolver: Permutation solver for multichannel source 
% separation based on full-rank spatial covariance model
%
% Coded by D. Kitamura (d-kitamura@ieee.org)
%
% # Original paper
% N. Q. K. Duong, E. Vincent, and R. Gribonval, "Underdetermined
% reverberant audio source separation using a fullrank spatial covariance
% model," IEEE Transactions on Audio, Speech, and Language Processing,
% vol. 18, no. 7, pp. 1830-1840, May 2010.
% H. Sawada et.al., "Grouping separated frequency cComponents by estimating
% propagation model parameters in frequency-domain blind source separation",
% IEEE Transactions on Audio, Speech, and Language Processing, vol.. 15, 
% no. 5, pp 1592-1604, 2007.
%
% see also
% http://d-kitamura.net
%
% [syntax]
%   [A,perm] = fullRankSCM_permSolver(R,fs,micSpacing,soundSpeed,drawConv)
%
% [inputs]
%            R: estimated sourcewise spatial covariance (channels x channels x freq. x sources)
%           fs: sampling frequency ([Hz], scalar)
%   micSpacing: microphone distance ([m], scalar)
%   soundSpeed: sound velocity ([m/s], scalar)
%     drawConv: plot cost function values in each iteration or not (true or false)
%
% [outputs]
%            A: estimated mixing matrix (sources x channels x freq.)
%         perm: aligned order of sources among all frequency bins (channels x freq.)
%

[M,M,I,N] = size(R); % channel x channel x frequency bin x source

% Apply principal component analysis to Rj
A = zeros(M,N,I);
for i = 1:I
    for n = 1:N
        [E,D] = eig(R(:,:,i,n));
        [~,ind] = max(diag(abs(D)));
        A(:,n,i) = E(:,ind);
    end
end

Iall = 2*(I-1);
A1 = zeros(M,N,I);
A2 = zeros(M,N,I);
refMic = 1;  % reference mic for normalization
micSpacing = micSpacing*(M - refMic); % maximum distance between microphones
fL = floor(soundSpeed/(2*micSpacing));
KL = floor((fL/fs)*Iall); % low-frequency bin without spatial aliasing
permList = perms(1:N);

% Normalization
for i = 1:I
    for n = 1:N
        A1(:,n,i) = A(:,n,i)*exp(-1i*angle(A(refMic,n,i))) / norm(A(:,n,i));
    end
    A2(:,:,i) = abs(A1(:,:,i)).*exp( 1i*angle(A1(:,:,i)) / (4*(i*fs/Iall)*micSpacing/soundSpeed) );
end

% Take the farest mic to maximize angle difference
for n = 1:N
    arg_org(n,:) = angle(A1(M,n,:));
end

% For low frequencies without spatial aliasing
% Initialize h
sumh = zeros(M,N);
for n = 1:N
    for i = 1:min(KL,I)
        sumh(:,n) = sumh(:,n) + A2(:,n,i);
    end
    h(:,n) = sumh(:,n) / norm(sumh(:,n));
end

% k-means algorithm
It = 5;
for it = 1:It
    sumh = zeros(M,N);
    for i = 1:min(KL,I)
        for m = 1:length(permList)
            order = permList(m,:);
            JM(m) = 0;
            % check all the permutation of a2
            for n = 1:N
                JM(m) = JM(m) + norm(A2(:,order(n),i)-h(:,n));
            end
        end
        [~,index] = min(JM);
        trueOrder = permList(index,:);
        % exchange columns according to true order
        for n = 1:N
            sumh(:,n) = sumh(:,n) + A2(:,trueOrder(n),i);
        end
        if (it == It)
            tmp1 = A1(:,trueOrder,i);
            A1(:,:,i) = tmp1;
            tmp = A(:,trueOrder,i);
            A(:,:,i) = tmp;
            perm(:,i) = trueOrder;
        end
    end
    for n = 1:N
        h(:,n) = sumh(:,n) / norm(sumh(:,n));
    end
end

% Compute average value for lamda & tau [see reference paper]
lamda = abs(h);
tau = -2*micSpacing*angle(h)/(pi*soundSpeed);

% For high frequencies with spatial aliasing
if KL<I  % if aliasing exist
    mu = 10^-5;
    for i = (KL+1):I
        h1 = lamda .* exp( -1i * 2 * pi * (i*fs/Iall) * tau );
        for m = 1:length(permList)
            order = permList(m,:);  JM(m) = 0;
            % check all the permutation
            for n=1:N
                JM(m) = JM(m) + norm(A1(:,order(n),i)-h1(:,n));
                %JM(m) = JM(m) + sum(abs(angle(A1(:,order(j),k)-h1(:,j))));
            end
        end
        [~,index] = min(JM);
        trueOrder = permList(index,:);
        tmp1 = A1(:,trueOrder,i);
        A1(:,:,i) = tmp1;
        tmp = A(:,trueOrder,i);
        A(:,:,i) = tmp;
        perm(:,i) = trueOrder;
        % update lamda(i,l) and tau(i,l)
        sum1 = ((i*fs/Iall)/fs) * imag(A1(:,:,i).*exp( 1i * 2 * pi * (i*fs/Iall) * tau));
        sum2 = lamda - real(A1(:,:,i) .* exp( 1i * 2 * pi * (i*fs/Iall) * tau ));
        tau = tau - mu * lamda .* sum1;
        lamda = lamda - mu*sum2;
    end
end

if drawConv
    % plot angle BEFORE permutation alignment
    figure;
    subplot(2,1,1);
    for n=1:N
        color(n,:) = rand(1,3); plot(arg_org(n,:),'.','Color',color(n,:));
        hold on;
    end
    title('Angle of A_{Ij} before the permutation');
    xlabel('Frequency bin'); ylabel('Argument');
    axis([1 I -pi pi]);
    
    % plot angle AFTER permutation alignment
    for n=1:N
        arg_per(n,:) = angle(A1(M,n,:));
    end
    subplot(2,1,2);
    for n=1:N
        plot(arg_per(n,:),'.','Color',color(n,:));
        hold on;
    end
    title('Angle of A_{Ij} after the permutation');
    xlabel('Frequency bin'); ylabel('Argument');
    axis([1 I -pi pi]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%