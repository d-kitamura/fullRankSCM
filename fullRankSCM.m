function [chat,v,R,cost] = fullRankSCM(X,N,maxIt,drawConv,v,R)
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
cost = zeros(maxIt+1,1);
eyeMJ = repmat(eyeM,[1,1,J]); % M x M x J

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
            R_x_i = squeeze( R_x(:,:,i,:) ); % M x M x J
            R_c_in = squeeze( R_c(:,:,i,:,n) ); % M x M x J
            invR_x_i = local_inverse( R_x_i, J, M ); % M x M x J
            W = local_multiplicationSquare( R_c_in, invR_x_i, J, M ); % sourcewise and frequency-wise Wiener filter, M x M x J, Eq.(32)
            chat_in = local_multiplication( W, x(:,i,:) ); % M x 1 x J, Eq. (33)
            chat(:,i,:,n) = chat_in;
            chat_inH = permute( conj(chat_in), [2,1,3] ); % Harmitian transpose
            ccH = local_multiplication( chat_in, chat_inH ); % M x M x J
            R_c_in = squeeze( R_c(:,:,i,:,n) ); % M x M x J
            Rhat_c = ccH + local_multiplicationSquare( (eyeMJ - W), R_c_in, J, M ); % M x M x J, Eq. (34)
            %%%%% M-step %%%%%
            invR = repmat( inv(R(:,:,i,n)), [1,1,J] ); % M x M x J
            invRRhat_c = local_multiplicationSquare( invR, Rhat_c, J, M ); % M x M x J
            trinvRRhat_c = local_trace( real( invRRhat_c ), J, M ); % J x 1
            v(i,:,n) = max( (1/M) * trinvRRhat_c, eps ).'; % 1 x J, Eq. (35)
            R(:,:,i,n) = (1/J) * sum( Rhat_c ./ permute( v(i,:,n), [1,3,2] ), 3 ); % Eq. (36)
        end
        %%%% Covariance calculation %%%%
        for n = 1:N
            RJ = repmat( R(:,:,i,n), [1,1,J] ); % M x M x J
            R_c(:,:,i,:,n) = permute( permute( v(i,:,n), [1,3,2] ) .* RJ, [1,2,4,3,5] ); % Eq. (4)
            R_x(:,:,i,:) = sum( R_c(:,:,i,:,:), 5 ); % Eq. (5)
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

%%% Multiplication %%%
function [ XY ] = local_multiplication( X, Y )
[A, ~, I] = size(X);
[~, B, I] = size(Y);
XY = zeros( A, B, I );
for i = 1:I
    XY(:,:,i) = X(:,:,i)*Y(:,:,i);
end
end

%%% Multiplication %%%
function [ XY ] = local_multiplicationSquare( X, Y, I, M )
if M == 2
    XY = zeros( M, M, I );
    XY(1,1,:) = X(1,1,:).*Y(1,1,:) + X(1,2,:).*Y(2,1,:);
    XY(1,2,:) = X(1,1,:).*Y(1,2,:) + X(1,2,:).*Y(2,2,:);
    XY(2,1,:) = X(2,1,:).*Y(1,1,:) + X(2,2,:).*Y(2,1,:);
    XY(2,2,:) = X(2,1,:).*Y(1,2,:) + X(2,2,:).*Y(2,2,:);
elseif M == 3
    XY = zeros( M, M, I );
    XY(1,1,:) = X(1,1,:).*Y(1,1,:) + X(1,2,:).*Y(2,1,:) + X(1,3,:).*Y(3,1,:);
    XY(1,2,:) = X(1,1,:).*Y(1,2,:) + X(1,2,:).*Y(2,2,:) + X(1,3,:).*Y(3,2,:);
    XY(1,3,:) = X(1,1,:).*Y(1,3,:) + X(1,2,:).*Y(2,3,:) + X(1,3,:).*Y(3,3,:);
    XY(2,1,:) = X(2,1,:).*Y(1,1,:) + X(2,2,:).*Y(2,1,:) + X(2,3,:).*Y(3,1,:);
    XY(2,2,:) = X(2,1,:).*Y(1,2,:) + X(2,2,:).*Y(2,2,:) + X(2,3,:).*Y(3,2,:);
    XY(2,3,:) = X(2,1,:).*Y(1,3,:) + X(2,2,:).*Y(2,3,:) + X(2,3,:).*Y(3,3,:);
    XY(3,1,:) = X(3,1,:).*Y(1,1,:) + X(3,2,:).*Y(2,1,:) + X(3,3,:).*Y(3,1,:);
    XY(3,2,:) = X(3,1,:).*Y(1,2,:) + X(3,2,:).*Y(2,2,:) + X(3,3,:).*Y(3,2,:);
    XY(3,3,:) = X(3,1,:).*Y(1,3,:) + X(3,2,:).*Y(2,3,:) + X(3,3,:).*Y(3,3,:);
elseif M == 4
    XY = zeros( M, M, I );
    XY(1,1,:) = X(1,1,:).*Y(1,1,:) + X(1,2,:).*Y(2,1,:) + X(1,3,:).*Y(3,1,:) + X(1,4,:).*Y(4,1,:);
    XY(1,2,:) = X(1,1,:).*Y(1,2,:) + X(1,2,:).*Y(2,2,:) + X(1,3,:).*Y(3,2,:) + X(1,4,:).*Y(4,2,:);
    XY(1,3,:) = X(1,1,:).*Y(1,3,:) + X(1,2,:).*Y(2,3,:) + X(1,3,:).*Y(3,3,:) + X(1,4,:).*Y(4,3,:);
    XY(1,4,:) = X(1,1,:).*Y(1,4,:) + X(1,2,:).*Y(2,4,:) + X(1,3,:).*Y(3,4,:) + X(1,4,:).*Y(4,4,:);
    XY(2,1,:) = X(2,1,:).*Y(1,1,:) + X(2,2,:).*Y(2,1,:) + X(2,3,:).*Y(3,1,:) + X(2,4,:).*Y(4,1,:);
    XY(2,2,:) = X(2,1,:).*Y(1,2,:) + X(2,2,:).*Y(2,2,:) + X(2,3,:).*Y(3,2,:) + X(2,4,:).*Y(4,2,:);
    XY(2,3,:) = X(2,1,:).*Y(1,3,:) + X(2,2,:).*Y(2,3,:) + X(2,3,:).*Y(3,3,:) + X(2,4,:).*Y(4,3,:);
    XY(2,4,:) = X(2,1,:).*Y(1,4,:) + X(2,2,:).*Y(2,4,:) + X(2,3,:).*Y(3,4,:) + X(2,4,:).*Y(4,4,:);
    XY(3,1,:) = X(3,1,:).*Y(1,1,:) + X(3,2,:).*Y(2,1,:) + X(3,3,:).*Y(3,1,:) + X(3,4,:).*Y(4,1,:);
    XY(3,2,:) = X(3,1,:).*Y(1,2,:) + X(3,2,:).*Y(2,2,:) + X(3,3,:).*Y(3,2,:) + X(3,4,:).*Y(4,2,:);
    XY(3,3,:) = X(3,1,:).*Y(1,3,:) + X(3,2,:).*Y(2,3,:) + X(3,3,:).*Y(3,3,:) + X(3,4,:).*Y(4,3,:);
    XY(3,4,:) = X(3,1,:).*Y(1,4,:) + X(3,2,:).*Y(2,4,:) + X(3,3,:).*Y(3,4,:) + X(3,4,:).*Y(4,4,:);
    XY(4,1,:) = X(4,1,:).*Y(1,1,:) + X(4,2,:).*Y(2,1,:) + X(4,3,:).*Y(3,1,:) + X(4,4,:).*Y(4,1,:);
    XY(4,2,:) = X(4,1,:).*Y(1,2,:) + X(4,2,:).*Y(2,2,:) + X(4,3,:).*Y(3,2,:) + X(4,4,:).*Y(4,2,:);
    XY(4,3,:) = X(4,1,:).*Y(1,3,:) + X(4,2,:).*Y(2,3,:) + X(4,3,:).*Y(3,3,:) + X(4,4,:).*Y(4,3,:);
    XY(4,4,:) = X(4,1,:).*Y(1,4,:) + X(4,2,:).*Y(2,4,:) + X(4,3,:).*Y(3,4,:) + X(4,4,:).*Y(4,4,:);
else % slow
    XY = zeros( M, M, I );
    for i = 1:I
        XY(:,:,i) = X(:,:,i)*Y(:,:,i);
    end
end
end

%%% Inverse %%%
function [ invX ] = local_inverse( X, I, M )
if M == 2
    invX = zeros(M,M,I);
    detX = max(X(1,1,:).*X(2,2,:) - X(1,2,:).*X(2,1,:), eps);
    invX(1,1,:) = X(2,2,:);
    invX(1,2,:) = -1*X(1,2,:);
    invX(2,1,:) = -1*X(2,1,:);
    invX(2,2,:) = X(1,1,:);
    invX = bsxfun(@rdivide, invX, detX); % This can be rewritten as "invX = invX./detX;" using implicit expansion for later R2016b
elseif M == 3
    invX = zeros(M,M,I);
    detX = max(X(1,1,:).*X(2,2,:).*X(3,3,:) + X(2,1,:).*X(3,2,:).*X(1,3,:) + X(3,1,:).*X(1,2,:).*X(2,3,:) - X(1,1,:).*X(3,2,:).*X(2,3,:) - X(3,1,:).*X(2,2,:).*X(1,3,:) - X(2,1,:).*X(1,2,:).*X(3,3,:), eps);
    invX(1,1,:) = X(2,2,:).*X(3,3,:) - X(2,3,:).*X(3,2,:);
    invX(1,2,:) = X(1,3,:).*X(3,2,:) - X(1,2,:).*X(3,3,:);
    invX(1,3,:) = X(1,2,:).*X(2,3,:) - X(1,3,:).*X(2,2,:);
    invX(2,1,:) = X(2,3,:).*X(3,1,:) - X(2,1,:).*X(3,3,:);
    invX(2,2,:) = X(1,1,:).*X(3,3,:) - X(1,3,:).*X(3,1,:);
    invX(2,3,:) = X(1,3,:).*X(2,1,:) - X(1,1,:).*X(2,3,:);
    invX(3,1,:) = X(2,1,:).*X(3,2,:) - X(2,2,:).*X(3,1,:);
    invX(3,2,:) = X(1,2,:).*X(3,1,:) - X(1,1,:).*X(3,2,:);
    invX(3,3,:) = X(1,1,:).*X(2,2,:) - X(1,2,:).*X(2,1,:);
    invX = bsxfun(@rdivide, invX, detX); % This can be rewritten as "invX = invX./detX;" using implicit expansion for later R2016b
elseif M == 4
    invX = zeros(M,M,I);
    detX = max(X(1,1,:).*X(2,2,:).*X(3,3,:).*X(4,4,:) + X(1,1,:).*X(2,3,:).*X(3,4,:).*X(4,2,:) + X(1,1,:).*X(2,4,:).*X(3,2,:).*X(4,3,:) + X(1,2,:).*X(2,1,:).*X(3,4,:).*X(4,3,:) + X(1,2,:).*X(2,3,:).*X(3,1,:).*X(4,4,:) + X(1,2,:).*X(2,4,:).*X(3,3,:).*X(4,1,:) + X(1,3,:).*X(2,1,:).*X(3,2,:).*X(4,4,:) + X(1,3,:).*X(2,2,:).*X(3,4,:).*X(4,1,:) + X(1,3,:).*X(2,4,:).*X(3,1,:).*X(4,2,:) + X(1,4,:).*X(2,1,:).*X(3,3,:).*X(4,2,:) + X(1,4,:).*X(2,2,:).*X(3,1,:).*X(4,3,:) + X(1,4,:).*X(2,3,:).*X(3,2,:).*X(4,1,:) - X(1,1,:).*X(2,2,:).*X(3,4,:).*X(4,3,:) - X(1,1,:).*X(2,3,:).*X(3,2,:).*X(4,4,:) - X(1,1,:).*X(2,4,:).*X(3,3,:).*X(4,2,:) - X(1,2,:).*X(2,1,:).*X(3,3,:).*X(4,4,:) - X(1,2,:).*X(2,3,:).*X(3,4,:).*X(4,1,:) - X(1,2,:).*X(2,4,:).*X(3,1,:).*X(4,3,:) - X(1,3,:).*X(2,1,:).*X(3,4,:).*X(4,2,:) - X(1,3,:).*X(2,2,:).*X(3,1,:).*X(4,4,:) - X(1,3,:).*X(2,4,:).*X(3,2,:).*X(4,1,:) - X(1,4,:).*X(2,1,:).*X(3,2,:).*X(4,3,:) - X(1,4,:).*X(2,2,:).*X(3,3,:).*X(4,1,:) - X(1,4,:).*X(2,3,:).*X(3,1,:).*X(4,2,:), eps);
    invX(1,1,:) = X(2,2,:).*X(3,3,:).*X(4,4,:) + X(2,3,:).*X(3,4,:).*X(4,2,:) + X(2,4,:).*X(3,2,:).*X(4,3,:) - X(2,2,:).*X(3,4,:).*X(4,3,:) - X(2,3,:).*X(3,2,:).*X(4,4,:) - X(2,4,:).*X(3,3,:).*X(4,2,:);
    invX(1,2,:) = X(1,2,:).*X(3,4,:).*X(4,3,:) + X(1,3,:).*X(3,2,:).*X(4,4,:) + X(1,4,:).*X(3,3,:).*X(4,2,:) - X(1,2,:).*X(3,3,:).*X(4,4,:) - X(1,3,:).*X(3,4,:).*X(4,2,:) - X(1,4,:).*X(3,2,:).*X(4,3,:);
    invX(1,3,:) = X(1,2,:).*X(2,3,:).*X(4,4,:) + X(1,3,:).*X(2,4,:).*X(4,2,:) + X(1,4,:).*X(2,2,:).*X(4,3,:) - X(1,2,:).*X(2,4,:).*X(4,3,:) - X(1,3,:).*X(2,2,:).*X(4,4,:) - X(1,4,:).*X(2,3,:).*X(4,2,:);
    invX(1,4,:) = X(1,2,:).*X(2,4,:).*X(3,3,:) + X(1,3,:).*X(2,2,:).*X(3,4,:) + X(1,4,:).*X(2,3,:).*X(3,2,:) - X(1,2,:).*X(2,3,:).*X(3,4,:) - X(1,3,:).*X(2,4,:).*X(3,2,:) - X(1,4,:).*X(2,2,:).*X(3,3,:);
    invX(2,1,:) = X(2,1,:).*X(3,4,:).*X(4,3,:) + X(2,3,:).*X(3,1,:).*X(4,4,:) + X(2,4,:).*X(3,3,:).*X(4,1,:) - X(2,1,:).*X(3,3,:).*X(4,4,:) - X(2,3,:).*X(3,4,:).*X(4,1,:) - X(2,4,:).*X(3,1,:).*X(4,3,:);
    invX(2,2,:) = X(1,1,:).*X(3,3,:).*X(4,4,:) + X(1,3,:).*X(3,4,:).*X(4,1,:) + X(1,4,:).*X(3,1,:).*X(4,3,:) - X(1,1,:).*X(3,4,:).*X(4,3,:) - X(1,3,:).*X(3,1,:).*X(4,4,:) - X(1,4,:).*X(3,3,:).*X(4,1,:);
    invX(2,3,:) = X(1,1,:).*X(2,4,:).*X(4,3,:) + X(1,3,:).*X(2,1,:).*X(4,4,:) + X(1,4,:).*X(2,3,:).*X(4,1,:) - X(1,1,:).*X(2,3,:).*X(4,4,:) - X(1,3,:).*X(2,4,:).*X(4,1,:) - X(1,4,:).*X(2,1,:).*X(4,3,:);
    invX(2,4,:) = X(1,1,:).*X(2,3,:).*X(3,4,:) + X(1,3,:).*X(2,4,:).*X(3,1,:) + X(1,4,:).*X(2,1,:).*X(3,3,:) - X(1,1,:).*X(2,4,:).*X(3,3,:) - X(1,3,:).*X(2,1,:).*X(3,4,:) - X(1,4,:).*X(2,3,:).*X(3,1,:);
    invX(3,1,:) = X(2,1,:).*X(3,2,:).*X(4,4,:) + X(2,2,:).*X(3,4,:).*X(4,1,:) + X(2,4,:).*X(3,1,:).*X(4,2,:) - X(2,1,:).*X(3,4,:).*X(4,2,:) - X(2,2,:).*X(3,1,:).*X(4,4,:) - X(2,4,:).*X(3,2,:).*X(4,1,:);
    invX(3,2,:) = X(1,1,:).*X(3,4,:).*X(4,2,:) + X(1,2,:).*X(3,1,:).*X(4,4,:) + X(1,4,:).*X(3,2,:).*X(4,1,:) - X(1,1,:).*X(3,2,:).*X(4,4,:) - X(1,2,:).*X(3,4,:).*X(4,1,:) - X(1,4,:).*X(3,1,:).*X(4,2,:);
    invX(3,3,:) = X(1,1,:).*X(2,2,:).*X(4,4,:) + X(1,2,:).*X(2,4,:).*X(4,1,:) + X(1,4,:).*X(2,1,:).*X(4,2,:) - X(1,1,:).*X(2,4,:).*X(4,2,:) - X(1,2,:).*X(2,1,:).*X(4,4,:) - X(1,4,:).*X(2,2,:).*X(4,1,:);
    invX(3,4,:) = X(1,1,:).*X(2,4,:).*X(3,2,:) + X(1,2,:).*X(2,1,:).*X(3,4,:) + X(1,4,:).*X(2,2,:).*X(3,1,:) - X(1,1,:).*X(2,2,:).*X(3,4,:) - X(1,2,:).*X(2,4,:).*X(3,1,:) - X(1,4,:).*X(2,1,:).*X(3,2,:);
    invX(4,1,:) = X(2,1,:).*X(3,3,:).*X(4,2,:) + X(2,2,:).*X(3,1,:).*X(4,3,:) + X(2,3,:).*X(3,2,:).*X(4,1,:) - X(2,1,:).*X(3,2,:).*X(4,3,:) - X(2,2,:).*X(3,3,:).*X(4,1,:) - X(2,3,:).*X(3,1,:).*X(4,2,:);
    invX(4,2,:) = X(1,1,:).*X(3,2,:).*X(4,3,:) + X(1,2,:).*X(3,3,:).*X(4,1,:) + X(1,3,:).*X(3,1,:).*X(4,2,:) - X(1,1,:).*X(3,3,:).*X(4,2,:) - X(1,2,:).*X(3,1,:).*X(4,3,:) - X(1,3,:).*X(3,2,:).*X(4,1,:);
    invX(4,3,:) = X(1,1,:).*X(2,3,:).*X(4,2,:) + X(1,2,:).*X(2,1,:).*X(4,3,:) + X(1,3,:).*X(2,2,:).*X(4,1,:) - X(1,1,:).*X(2,2,:).*X(4,3,:) - X(1,2,:).*X(2,3,:).*X(4,1,:) - X(1,3,:).*X(2,1,:).*X(4,2,:);
    invX(4,4,:) = X(1,1,:).*X(2,2,:).*X(3,3,:) + X(1,2,:).*X(2,3,:).*X(3,1,:) + X(1,3,:).*X(2,1,:).*X(3,2,:) - X(1,1,:).*X(2,3,:).*X(3,2,:) - X(1,2,:).*X(2,1,:).*X(3,3,:) - X(1,3,:).*X(2,2,:).*X(3,1,:);
    invX = bsxfun(@rdivide, invX, detX); % This can be rewritten as "invX = invX./detX;" using implicit expansion for later R2016b
else % slow
    invX = zeros(M,M,I);
    for i = 1:I
        invX(:,:,i) = X(:,:,i)\eye(M);
    end
end
end

%%% Trace %%%
function [ trX ] = local_trace( X, I, M )
if M == 2
    X = permute(X, [3,1,2]); % I x M x M
    trX = X(:,1,1) + X(:,2,2);
elseif M == 3
    X = permute(X, [3,1,2]); % I x M x M
    trX = X(:,1,1) + X(:,2,2) + X(:,3,3);
elseif M == 4
    X = permute(X, [3,1,2]); % I x M x M
    trX = X(:,1,1) + X(:,2,2) + X(:,3,3) + X(:,4,4);
else % slow
    trX = zeros(I,1);
    parfor i = 1:I
        trX(i) = trace(X(:,:,i));
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%