function [chat,v,R,cost] = fullRankSCM_readable(X,N,maxIt,drawConv,v,R)
%
% fullRankSCM: Multichannel source separation based on full-rank spatial covariance model
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
%   [chat,R,cost] = fullRankSCM(X,N)
%   [chat,R,cost] = fullRankSCM(X,N,maxIt)
%   [chat,R,cost] = fullRankSCM(X,N,maxIt,drawConv)
%   [chat,R,cost] = fullRankSCM(X,N,maxIt,drawConv,v)
%   [chat,R,cost] = fullRankSCM(X,N,maxIt,drawConv,v,R)
%
% [inputs]
%         X: input multichannel signals in time-frequency domain (freq. x frames x channels)
%         N: number of sources
%     maxIt: number of iterations (default: 100)
%  drawConv: plot cost function values in each iteration or not (true or false)
%         v: initial sourcewise variance (freq. x frames x sources, optional)
%         R: initial spatial covariance (channels x channels x freq. x sources, optional)
%
% [outputs]
%      chat: estimated source images (channels x freq. x frames x sources)
%         v: estimated sourcewise variance (freq. x frames x sources)
%         R: estimated sourcewise spatial covariance (channels x channels x freq. x sources)
%      cost: convergence behavior of cost function in full-rank spatial covariance model (maxIt+1 x 1)
%

% Check errors and set default values
[I,J,M] = size(X); % Frequency bin x time frame x channel
eyeM = eye(M);
if (M == 1)
    error('The input sepctrogram must be a multichannel format.\n');
end
if (M > I)
    error('The input sepctrogram might be wrong. The size of it must be (freq x frame x ch).\n');
end
if (nargin < 3)
    maxIt = 100;
end
if (nargin < 4)
    drawConv = false;
end

% Initialization
if (nargin < 5)
    fprintf('Initializing spatial covariance...');
    v = ones(I,J,N); % sourcewise time-varying variance (power spectrogram)
    R = local_covarianceInit( X, N, 10*N ); % initialization based on hierarchiccal clustering
    fprintf('\n');
elseif (nargin < 6)
    R = repmat( eyeM, [1,1,I,N] );
end

% Memory allocation
x = permute(X, [3,1,2]); % M x I x J
R_c = zeros(M,M,I,J,N); % sourcewise time-variant spatial covariance (Vn*Rn)
R_x = zeros(M,M,I,J); % observed covariance
chat = zeros(M,I,J,N); % estimated source images
Rhat_c = zeros(M,M,J); % estimated source covariance (conditional covariance)
cost = zeros(maxIt+1,1);
eyeM = eye(M);

% Covariance calculation
for n = 1:N
    for i = 1:I
        RJ = repmat( R(:,:,i,n), [1,1,J] ); % M x M x J
        R_c(:,:,i,:,n) = permute( permute( v(i,:,n), [1,3,2] ) .* RJ, [1,2,4,3,5] ); % Eq. (4)
        R_x(:,:,i,:) = sum( R_c(:,:,i,:,:), 5 ); % Eq. (5)
    end
end
if drawConv
    cost(1,1) = local_costFunction( x, R_x, I, J );
end

% EM algorithm
fprintf('Iteration:    ');
for it = 1:maxIt
    fprintf('\b\b\b\b%4d', it);
    for i = 1:I
        for n = 1:N
            %%%%% E-step %%%%%
            for j = 1:J
                W = R_c(:,:,i,j,n) / R_x(:,:,i,j); % sourcewise Wiener filter (M x M), Eq.(32)
                chat(:,i,j,n) = W * x(:,i,j); % Eq. (33)
                Rhat_c(:,:,j) = chat(:,i,j,n) * chat(:,i,j,n)' + (eyeM - W) * R_c(:,:,i,j,n); % Eq. (34)
            end
            %%%%% M-step %%%%%
            invR = inv(R(:,:,i,n)); % calculate here because invR is common over j
            wRhat_c = zeros(2,2);
            for j = 1:J
                v(i,j,n) = max((1/M) * real(trace((invR * Rhat_c(:,:,j)))), eps); % Eq. (35)
                wRhat_c = wRhat_c + Rhat_c(:,:,j) / v(i,j,n);  % Eq. (36)
            end
            R(:,:,i,n) = (1/J) * wRhat_c; % Eq. (36)
        end
        %%%% Covariance calculation %%%%
        for j = 1:J
            for n = 1:N
                R_c(:,:,i,j,n) = v(i,j,n) * R(:,:,i,n); % Eq. (4)
            end
            R_x(:,:,i,j) = sum(R_c(:,:,i,j,:), 5); % Eq. (5)
        end
    end
    if drawConv
        cost(it+1,1) = local_costFunction( x, R_x, I, J );
    end
end
fprintf(' Full-rank spatial covariance model estimation done.\n');

if drawConv
    figure;
    plot( (0:it), cost );
    set(gca,'FontName','Times','FontSize',16);
    xlabel('Iteration','FontName','Arial','FontSize',16);
    ylabel('Value of cost function','FontName','Arial','FontSize',16);
end
end

%% Local functions
%%% Cost function %%%
function [ cost ] = local_costFunction( x, R_x, I, J )
cost = 0;
for i = 1:I
    for j = 1:J
        cost = cost + log( real( det( pi * R_x(:,:,i,j) ) ) ) + real( x(:,i,j)' / R_x(:,:,i,j) * x(:,i,j) );
    end
end
end

%%% Covariance initialization %%%
function [ R ] = local_covarianceInit( X, N, K )
[I,J,M] = size(X); % I: Frequency, J: Time, M: Channel
XX = permute(X,[3,2,1]); % M x J x I
X0 = XX;
% Normalization
for i = 1:I
    for j = 1:J
        XX(:,j,i) = XX(:,j,i) * exp( -1i * angle(XX(1,j,i))) / norm(XX(:,j,i) );
        X0(:,j,i) = X0(:,j,i) * exp( -1i * angle(X0(1,j,i)) );
    end
end
XX = permute(XX,[2,1,3]); % J x M x I
X0 = permute(X0,[2,1,3]);  % J x M x I
% Hierachical clustering   
R = zeros(M,M,I,N);
Rt = zeros(M,M,I,J,N);
for i = 1:I
    ind = 0;
    for j = 1:J-1
        for jj = j+1:J
            ind = ind+1;
            Y(ind) = norm( XX(j,:,i) - XX(jj,:,i) );
        end
    end
    Z = linkage(Y,'average'); 
    T = cluster(Z,'maxclust',K);
    
    C = zeros(K,1);
    for k = 1:K
        C(k) = length(find(T==k));
    end
    for n = 1:N
        [maxj,ind] = max(C);
        C(ind) = 0;
        Gj = find(T==ind);
        tmp = zeros(M,M);
        for k = 1:maxj 
            tmp = tmp + reshape(X0(Gj(k),:,i),M,1) * reshape(X0(Gj(k),:,i),M,1)'; 
            Rt(:,:,i,k,n) = reshape(X0(Gj(k),:,i),M,1) * reshape(X0(Gj(k),:,i),M,1)';
        end
        R(:,:,i,n) = tmp / maxj;
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%