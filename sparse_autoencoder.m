%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

visibleSize = 8*8;   % number of input units 
hiddenSize = 25;     % number of hidden units 
sparsityParam = 0.01;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, 
				   	 % which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.0001;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
%  After implementing sampleIMAGES, the display_network command should
%  display a random sample of 200 patches from the dataset
patches = sampleIMAGES;
display_network(patches(:,randi(size(patches,2),200,1)),8);


% Obtain random parameters theta, 参数初始化为均值为0，方差为 sigma 的很小的数，
% 若取得相同的值，会使得网络隐层每个单元有相同的输入,即a^(2)的每个分量相同
%  a^(2)=W^(1)*x^(1)
theta = initializeParameters(hiddenSize, visibleSize);

%%======================================================================
%% STEP 2: Implement sparseAutoencoderCost
%
%  You can implement all of the components (squared error cost, weight decay term,
%  sparsity penalty) in the cost function at once, but it may be easier to do 
%  it step-by-step and run gradient checking (see STEP 3) after each step.  We 
%  suggest implementing the sparseAutoencoderCost function using the following steps:
%
%  (a) Implement forward propagation in your neural network, and implement the 
%      squared error term of the cost function.  Implement backpropagation to 
%      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking 
%      to verify that the calculations corresponding to the squared error cost 
%      term are correct.
%
%  (b) Add in the weight decay term (in both the cost function and the derivative
%      calculations), then re-run Gradient Checking to verify correctness. 
%
%  (c) Add in the sparsity penalty term, then re-run Gradient Checking to 
%      verify correctness.
%
%  Feel free to change the training settings when debugging your
%  code.  (For example, reducing the training set size or 
%  number of hidden units may make your code run faster; and setting beta 
%  and/or lambda to zero may be helpful for debugging.)  However, in your 
%  final submission of the visualized weights, please use parameters we 
%  gave in Step 0 above.

[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
									lambda,sparsityParam, beta, patches);

%%======================================================================
%% STEP 3: Gradient Checking
%
% Hint: If you are debugging your code, performing gradient checking on smaller models 
% and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
% units) may speed things up.

% First, lets make sure your numerical gradient computation is correct for a
% simple function.  After you have implemented computeNumericalGradient.m,
% run the following: 
checkNumericalGradient();

% Now we can use it to check your cost function and derivative calculations
% for the sparse autoencoder.  
numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
						hiddenSize, lambda,sparsityParam, beta, patches), theta);

% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.
            % When you got this working, Congratulations!!! 

%%======================================================================
%% STEP 4: After verifying that your implementation of
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p,visibleSize, hiddenSize, ...
							lambda, sparsityParam, beta, patches),theta, options);
%%======================================================================
%% STEP 5: Visualization 

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 对应step1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%三个函数（sampleIMAGES）（normalizeData）（initializeParameters）%%%%
function patches = sampleIMAGES()
load IMAGES;    % 加载初始的10张512*512大图片 

patchsize = 8;  % 采样大小
numpatches = 10000;

%  初始化该矩阵为0，该矩阵为 64*10000维每一列为一张图片. 
patches = zeros(patchsize*patchsize, numpatches);
  
%  IMAGES 为一个包含10 张images的三维数组，IMAGES(:,:,6) 是一个第六张图片的 512x512 的二维数组,
%  命令 "imagesc(IMAGES(:,:,6)), colormap gray;" 可以把第六张图可视化.
% 这几张图是经过whiteing预处理的？
%  IMAGES(21:30,21:30,1) 就是从第一张图采样得到的(21,21) to (30,30) 的小patchs

%在每张图片中随机选取1000个patch，共10000个patch
for imageNum = 1:10
    [rowNum colNum] = size(IMAGES(:,:,imageNum));
	%实现每张图片选取1000个patch
    for patchNum = 1:1000
		%得到左上角的两个点
        xPos = randi([1,rowNum-patchsize+1]);
        yPos = randi([1, colNum-patchsize+1]);
		%填充到矩阵里
        patches(:,(imageNum-1)*1000+patchNum) = ...
			reshape(IMAGES(xPos:xPos+7,yPos:yPos+7,imageNum),64,1);
    end
end
%由于autoencoder的激励函数是sigmod函数，输出值限定在[0,1],故为了达到H W,b（x）= x，x作为输入，
%也要限定在0-1之间，故需要进行正则化
patches = normalizeData(patches);
end

% 正则化的函数，不太明白s-sigma法则？
function patches = normalizeData(patches)
% 减去均值 
patches = bsxfun(@minus, patches, mean(patches));
% s = std(X)，此处X是一个矢量，该函数返回标准偏差（注意其分母为n-1，而不是n） 。
% 结果s是一个X各样本偏差无偏估计的平方根(X包含独立的、同分布样本)。
% 如果X是一个矩阵，该函数返回一个行矢量，它包含了X每列元素的标准偏差。
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;
% 重新压缩 从[-1,1] 到 [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;
end

%首先初始化参数
function theta = initializeParameters(hiddenSize, visibleSize)
% Initialize parameters randomly based on layer sizes.
 % we'll choose weights uniformly from the interval [-r, r]
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1); 
%rand(a,b)产生均匀分布的随机矩阵维度为a*b，元素取值范围0.0 ～1.0。
W1 = rand(hiddenSize, visibleSize) * 2 * r - r; 
%rand(a,b)*2*r即取值范围为（0-2r）， rand(a,b)*2*r -r即取值范围为（-r - r）
W2 = rand(visibleSize, hiddenSize) * 2 * r - r;
b1 = zeros(hiddenSize, 1); %连接到hidden unit的偏置单元
b2 = zeros(visibleSize, 1); %链接到output layer的偏置单元
%  将矩阵合并为一个向量
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
%初始化参数结束
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 对应step 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%返回稀疏损失函数的值与梯度值%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
										lambda, sparsityParam, beta, data)
% visibleSize: 输入层单元数
% hiddenSize: 隐藏单元数 
% lambda: 正则项
% sparsityParam: （p）指定的平均激活度p
% beta: 稀疏权重项B
% data: 64x10000 的矩阵为training data,data(:,i)  是第i个训练样例.   
% 把参数拼接为一个向量，因为采用L-BFGS优化，L-BFGS要求的就是向量. 
% 将长向量转换成每一层的权值矩阵和偏置向量值
% theta向量的的 1->hiddenSize*visibleSize，W1共hiddenSize*visibleSize 个元素，重新作为矩阵
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);

%类似以上一直往后放
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% 参数对应的梯度矩阵 ;
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

Jcost = 0;  %直接误差
Jweight = 0;%权值惩罚
Jsparse = 0;%稀疏性惩罚
[n m] = size(data); %m为样本的个数，n为样本的特征数

%前向算法计算各神经网络节点的线性组合值和active值
%W1为 hiddenSize*visibleSize的矩阵
%data为 visibleSize* trainexampleNum的矩阵
%remat(b1,1,m)把向量b1复制扩展为hiddenSize*m列
% 根据公式 Z^(l) = z^(l-1)*W^(l-1)+b^(l-1)
%z2保存的是10000个样本下隐藏层的输入，为hiddenSize*m维的矩阵，每一列代表一次输入
z2= W1*data + remat(b1,1,m)；%第二层的输入
a2 = sigmoid(z2); %对z2取sigmod 即得到a2，即隐藏层的输出
%Z3为 inputsize*m的矩阵
z3 = W2*a2+repmat(b2,1,m); %output layer 的输入
a3 = sigmoid(z3); %output 层的输出

% 计算预测产生的误差
%对应J(W,b), 外边的sum是对所有样本求和，里边的sum是对输出层的所有分量求和
Jcost = (0.5/m)*sum(sum((a3-data).^2));
%计算权值惩罚项 
Jweight = (1/2)*(sum(sum(W1.^2))+sum(sum(W2.^2)));
%计算稀疏性规则项 sum(matrix,2)是进行按行求和运算，即所有样本在隐层的输出累加求均值
% rho为一个hiddenSize*1 维的向量，平均激活度
rho = (1/m).*sum(a2,2);%求出隐含层输出aj的平均值向量 rho为hiddenSize维的
%求稀疏项的损失
Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+(1-sparsityParam).*log((1-sparsityParam)./(1-rho)));
%损失函数的总表达式 损失项 + 正则化项 + 稀疏项
cost = Jcost + lambda*Jweight + beta*Jsparse;
%计算l = 3 即 output-layer层的误差dleta3，因为在autoencoder中输入等于输出h(W,b)=x
delta3 = -(data-a3).*sigmoidInv(z3);
%因为加入了稀疏规则项，所以计算偏导时需要引入该项，sterm为稀疏项，为hiddenSize维的向量
sterm = beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho))
% W2 为64*25的矩阵，d3为第三层的输出为64*10000的矩阵，每一列为每个样本x^(i)的输出，W2'为W2的转置
% repmat(sterm,1,m)会把函数复制扩展为m列的矩阵，每一列都为sterm向量。
% d2为hiddenSize*10000的矩阵
delta2 = (W2'*delta3+repmat(sterm,1,m)).*sigmoidInv(z2); 

%计算W1grad 
% data'为10000*64的矩阵 d2*data' 位25*64的矩阵
W1grad = W1grad+delta2*data';
W1grad = (1/m)*W1grad+lambda*W1;

%计算W2grad  
W2grad = W2grad+delta3*a2';
W2grad = (1/m).*W2grad+lambda*W2;

%计算b1grad 
b1grad = b1grad+sum(delta2,2);
b1grad = (1/m)*b1grad;%注意b的偏导是一个向量，所以这里应该把每一行的值累加起来

%计算b2grad 
b2grad = b2grad+sum(delta3,2);
b2grad = (1/m)*b2grad;
%计算完成重新转为向量
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

%sigmoid函数的导函数
function sigmInv = sigmoidInv(x)
    sigmInv = sigmoid(x).*(1-sigmoid(x));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 对应step 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%三个函数：（checkNumericalGradient）（simpleQuadraticFunction）（computeNumericalGradient）
function [] = checkNumericalGradient()
x = [4; 10];
%当前简单函数实际的值与实际的导函数
[value, grad] = simpleQuadraticFunction(x);
% 在点 x 处计算简单函数的梯度，("@simpleQuadraticFunction" denotes a pointer to a function.)
numgrad = computeNumericalGradient(@simpleQuadraticFunction, x);
% disp()等价于 print()
disp([numgrad grad]);
fprintf('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');
% norm 等价于 sqrt(sum(X.^2)); 如果实现正确，设置 EPSILON = 0.0001，误差应该为2.1452e-12 
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); 
fprintf('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n');
end

 %这个简单函数用来检验写的computeNumericalGradient函数的正确性
function [value,grad] = simpleQuadraticFunction(x)
% this function accepts a 2D vector as input. 
% Its outputs are:
%   value: h(x1, x2) = x1^2 + 3*x1*x2
%   grad: A 2x1 vector that gives the partial derivatives of h with respect to x1 and x2 
% Note that when we pass @simpleQuadraticFunction(x) to computeNumericalGradients, we're assuming
% that computeNumericalGradients will use only the first returned value of this function.
value = x(1)^2 + 3*x(1)*x(2);
grad = zeros(2, 1);
grad(1)  = 2*x(1) + 3*x(2);
grad(2)  = 3*x(1);
end

%梯度检验的函数
function numgrad = computeNumericalGradient(J, theta)
% theta: 参数，向量或者实数均可
% J: 输出值为实数的函数. 调用y = J(theta)将会返回函数在theta处的值

% numgrad初始化为0,与theta维度相同
numgrad = zeros(size(theta));
EPSILON = 1e-4;
% theta是一个行向量，size(theta,1)是求行数
n = size(theta,1);
%产生一个维度为n的单位矩阵
E = eye(n);
for i = 1:n
	% (n,:)代表第n行，所有的列
	% (:,n)代表所有行，第n列
	% 由于E是单位矩阵，所以只有第i行第i列的元素变为EPSILON
    delta = E(:,i)*EPSILON;
	%向量第i维度的值
    numgrad(i) = (J(theta+delta)-J(theta-delta))/(EPSILON*2.0);
end
%% ---------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 对应step 5 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%关于函数的展示%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h, array] = display_network(A, opt_normalize, opt_graycolor, cols, opt_colmajor)
% This function visualizes filters in matrix A. Each column of A is a
% filter. We will reshape each column into a square image and visualizes
% on each cell of the visualization panel. 
% All other parameters are optional, usually you do not need to worry
% about it.
% opt_normalize: whether we need to normalize the filter so that all of
% them can have similar contrast. Default value is true.
% opt_graycolor: whether we use gray as the heat map. Default is true.
% cols: how many columns are there in the display. Default value is the
% squareroot of the number of columns in A.
% opt_colmajor: you can switch convention to row major for A. In that
% case, each row of A is a filter. Default value is false.
warning off all

if ~exist('opt_normalize', 'var') || isempty(opt_normalize)
    opt_normalize= true;
end

if ~exist('opt_graycolor', 'var') || isempty(opt_graycolor)
    opt_graycolor= true;
end

if ~exist('opt_colmajor', 'var') || isempty(opt_colmajor)
    opt_colmajor = false;
end

% rescale
A = A - mean(A(:));

if opt_graycolor, colormap(gray); end

% compute rows, cols
[L M]=size(A);
sz=sqrt(L);
buf=1;
if ~exist('cols', 'var')
    if floor(sqrt(M))^2 ~= M
        n=ceil(sqrt(M));
        while mod(M, n)~=0 && n<1.2*sqrt(M), n=n+1; end
        m=ceil(M/n);
    else
        n=sqrt(M);
        m=n;
    end
else
    n = cols;
    m = ceil(M/n);
end

array=-ones(buf+m*(sz+buf),buf+n*(sz+buf));

if ~opt_graycolor
    array = 0.1.* array;
end


if ~opt_colmajor
    k=1;
    for i=1:m
        for j=1:n
            if k>M, 
                continue; 
            end
            clim=max(abs(A(:,k)));
            if opt_normalize
                array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz)/clim;
            else
                array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz)/max(abs(A(:)));
            end
            k=k+1;
        end
    end
else
    k=1;
    for j=1:n
        for i=1:m
            if k>M, 
                continue; 
            end
            clim=max(abs(A(:,k)));
            if opt_normalize
                array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz)/clim;
            else
                array(buf+(i-1)*(sz+buf)+(1:sz),buf+(j-1)*(sz+buf)+(1:sz))=reshape(A(:,k),sz,sz);
            end
            k=k+1;
        end
    end
end

if opt_graycolor
    h=imagesc(array,'EraseMode','none',[-1 1]);
else
    h=imagesc(array,'EraseMode','none',[-1 1]);
end
axis image off

drawnow;

warning on all
