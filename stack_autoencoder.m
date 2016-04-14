%% CS294A/CS294W Stacked Autoencoder Exercise

%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

DISPLAY = true;
inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.


%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta
%直接用L-BFGS求得最优的参数
addpath minFunc/;
options = struct;
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
%训练出第一层网络的参数
[sae1OptTheta, cost] =  minFunc(@(p)sparseAutoencoderCost(p,...
    inputSize,hiddenSizeL1,lambda,sparsityParam,beta,trainData),sae1Theta,options);
save('saves/step2.mat', 'sae1OptTheta');

%W1为400 * 28^2的矩阵，对W1的转置 W1' 进行展示便可得到结果
if DISPLAY
  W1 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
  display_network(W1');
end


%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.
% 计算出第一层的最有参数W^(1,1)后，用train data计算隐层输出a^(2)
% z^(2) = W^(1,1)*a(1)+b(1,1) (a^(1)即为输入x)
% sae1Features = a^(2) = f(z^(2))
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta

[sae2OptTheta, cost] =  minFunc(@(p)sparseAutoencoderCost(p,...
    hiddenSizeL1,hiddenSizeL2,lambda,sparsityParam,beta,sae1Features),sae2Theta,options);%训练出第一层网络的参数
save('saves/step3.mat', 'sae2OptTheta');

figure;
if DISPLAY
  W11 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
  W12 = reshape(sae2OptTheta(1:hiddenSizeL2 * hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
  % TODO(zellyn): figure out how to display a 2-level network
%  display_network(log(W11' ./ (1-W11')) * W12');
%   W12_temp = W12(1:196,1:196);
%   display_network(W12_temp');
%   figure;
%   display_network(W12_temp');
end

%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.
%  像上边那样计算出a^(3)
%  z^(3) = W^(1,2)*a(2)+b(1,2) (a^(2)即为上层stack autoencoder的输出)
%  sae1Features = a^(3) = f(z^(3))
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);

%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);
% 用sae2Features 来训练softmaxmodel
softmaxLambda = 1e-4;
numClasses = 10;
softoptions = struct;
softoptions.maxIter = 400;
softmaxModel = softmaxTrain(hiddenSizeL2,numClasses,softmaxLambda,...
                            sae2Features,trainLabels,softoptions);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

save('saves/step4.mat', 'saeSoftmaxOptTheta');

%%======================================================================
%% STEP 5: Finetune softmax model 对模型进行微调

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);%创建一个两行一列的cell数组
% 为stack的第一个cell赋值
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);

% 为stack的第二个cell赋值
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
% 是将stack层次的网络参数（可能是多个参数）转换成一个向量params，这样有利用使用各种优化算法来
% 进行优化操作。Netconfig中保存的是该网络的相关信息，其中netconfig.inputsize表示的是网络的输入
% 层节点的个数。netconfig.layersizes中的元素分别表示每一个隐含层对应节点的个数。 
[stackparams, netconfig] = stack2params(stack);

%把网络中的所有元素保存到stackedAETheta这个向量中
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".

%对整个网络的参数进行微调
[stackedAEOptTheta, cost] =  minFunc(@(p)stackedAECost(p,inputSize,hiddenSizeL2,...
                         numClasses, netconfig,lambda, trainData, trainLabels),...
                        stackedAETheta,options);
save('saves/step5.mat', 'stackedAEOptTheta');
%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);



% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)

%初始化网络的权值
function theta = initializeParameters(hiddenSize, visibleSize)

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
W2 = rand(visibleSize, hiddenSize) * 2 * r - r;

b1 = zeros(hiddenSize, 1);
b2 = zeros(visibleSize, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

%%%计算稀疏自编码器的损失和梯度
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
z3 = W2*a2+repmat(b2,1,m); %output layer 的输入
a3 = sigmoid(z3); %output 层的输出

% 计算预测产生的误差
%对应J(W,b), 外边的sum是对所有样本求和，里边的sum是对输出层的所有分量求和
Jcost = (0.5/m)*sum(sum((a3-data).^2));
%计算权值惩罚项 正则化项，并没有带正则项参数
Jweight = (1/2)*(sum(sum(W1.^2))+sum(sum(W2.^2)));
%计算稀疏性规则项 sum(matrix,2)是进行按行求和运算，即所有样本在隐层的输出累加求均值
% rho为一个hiddenSize*1 维的向量

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


function [activation] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, data)

% theta: trained weights from the autoencoder
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
  
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

%  Instructions: Compute the activation of the hidden layer for the Sparse Autoencoder.
%计算隐层输出a^(2)
activation  = sigmoid(W1*data+repmat(b1,[1,size(data,2)]));

%-------------------------------------------------------------------

end


function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options)
%softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
%
% inputSize: the size of an input vector x^(i)
% numClasses: the number of classes 
% lambda: weight decay parameter
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, i) is the ith input
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input
% options (optional): options
%   options.maxIter: number of iterations to train for

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 400;
end

% initialize parameters，randn(M,1)产生均值为0，方差为1长度为M的数组
theta = 0.005 * randn(numClasses * inputSize, 1);

% Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % softmaxCost.m satisfies this.
minFuncOptions.display = 'on';

[softmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, ...
                                   numClasses, inputSize, lambda, ...
                                   inputData, labels), ...                                   
                              theta, options);

% Fold softmaxOptTheta into a nicer format
softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;
                          
end


function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)
% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data

theta = reshape(theta, numClasses, inputSize);% 转化为k*n的参数矩阵

numCases = size(data, 2);%或者data矩阵的列数，即样本数
% M = sparse(r, c, v) creates a sparse matrix such that M(r(i), c(i)) = v(i) for all i.
% That is, the vectors r and c give the position of the elements whose values we wish 
% to set, and v the corresponding values of the elements
% labels = (1,3,4,10 ...)^T 
% 1:numCases=(1,2,3,4...M)^T
% sparse(labels, 1:numCases, 1) 会产生
% 一个行列为下标的稀疏矩阵
% (1,1)  1
% (3,2)  1
% (4,3)  1
% (10,4) 1
%这样改矩阵填满后会变成每一列只有一个元素为1，该元素的行即为其lable k
%1 0 0 ... 
%0 0 0 ...
%0 1 0 ...
%0 0 1 ...
%0 0 0 ...
%. . .
%上矩阵为10*M的 ，即 groundTruth 矩阵
groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;
% 每个参数的偏导数矩阵
thetagrad = zeros(numClasses, inputSize);

% theta(k*n) data(n*m)
%theta * data = k*m , 第j行第i列为theta_j^T * x^(i)
%max(M)产生一个行向量，每个元素为该列中的最大值，即对上述k*m的矩阵找出m列中每列的最大值

M = bsxfun(@minus,theta*data,max(theta*data, [], 1)); % 每列元素均减去该列的最大值，见图-
M = exp(M); %求指数
p = bsxfun(@rdivide, M, sum(M)); %sum(M),对M中的元素按列求和
cost = -1/numCases * groundTruth(:)' * log(p(:)) + lambda/2 * sum(theta(:) .^ 2);%损失函数值
%groundTruth 为k*m ，data'为m*n，即theta为k*n的矩阵，n代表输入的维度，k代表类别，即没有隐层的
%输入为n，输出为k的神经网络
thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta; %梯度，为 k * 

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end


function [params, netconfig] = stack2params(stack)

% Converts a "stack" structure into a flattened parameter vector and also
% stores the network configuration. This is useful when working with
% optimization toolboxes such as minFunc.
%
% [params, netconfig] = stack2params(stack)
%
% stack - the stack structure, where stack{1}.w = weights of first layer
%                                    stack{1}.b = weights of first layer
%                                    stack{2}.w = weights of second layer
%                                    stack{2}.b = weights of second layer
%                                    ... etc.


% Setup the compressed param vector
params = [];
for d = 1:numel(stack)
    
    % This can be optimized. But since our stacks are relatively short, it
    % is okay
    params = [params ; stack{d}.w(:) ; stack{d}.b(:) ];
    
    % Check that stack is of the correct form
    assert(size(stack{d}.w, 1) == size(stack{d}.b, 1), ...
        ['The bias should be a *column* vector of ' ...
         int2str(size(stack{d}.w, 1)) 'x1']);
    if d < numel(stack)
        assert(size(stack{d}.w, 1) == size(stack{d+1}.w, 2), ...
            ['The adjacent layers L' int2str(d) ' and L' int2str(d+1) ...
             ' should have matching sizes.']);
    end
    
end

if nargout > 1
    % Setup netconfig
    if numel(stack) == 0
        netconfig.inputsize = 0;
        netconfig.layersizes = {};
    else
        netconfig.inputsize = size(stack{1}.w, 2);
        netconfig.layersizes = {};
        for d = 1:numel(stack)
            netconfig.layersizes = [netconfig.layersizes ; size(stack{d}.w,1)];
        end
    end
end

end


function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.

%以下是求softmax层的梯度
% numel 返回stack数组中的元素个数
depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1, 1);
a{1} = data;

for layer = (1:depth)
  z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
  a{layer+1} = sigmoid(z{layer+1});
end

M = softmaxTheta * a{depth+1};
M = bsxfun(@minus, M, max(M));
p = bsxfun(@rdivide, exp(M), sum(exp(M)));

cost = -1/numClasses * groundTruth(:)' * log(p(:)) + lambda/2 * sum(softmaxTheta(:) .^ 2);
softmaxThetaGrad = -1/numClasses * (groundTruth - p) * a{depth+1}' + lambda * softmaxTheta;

%这是残差项，接下来是残差反向传播
d = cell(depth+1);
%这是a^(3)  的 残差项，由softmax的误差反向传递计算得到
d{depth+1} = -(softmaxTheta' * (groundTruth - p)) .* a{depth+1} .* (1-a{depth+1});

for layer = (depth:-1:2)
  d{layer} = (stack{layer}.w' * d{layer+1}) .* a{layer} .* (1-a{layer});
end

for layer = (depth:-1:1)
  stackgrad{layer}.w = (1/numClasses) * d{layer+1} * a{layer}';
  stackgrad{layer}.b = (1/numClasses) * sum(d{layer+1}, 2);
end

grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
depth = numel(stack);
z = cell(depth+1,1);
a = cell(depth+1, 1);
a{1} = data;

for layer = (1:depth)
  z{layer+1} = stack{layer}.w * a{layer} + repmat(stack{layer}.b, [1, size(a{layer},2)]);
  a{layer+1} = sigmoid(z{layer+1});
end
% -----------------------------------------------------------
end


function [] = checkStackedAECost()

% Check the gradients for the stacked autoencoder
%
% In general, we recommend that the creation of such files for checking
% gradients when you write new cost functions.
%

%% Setup random data / small model
inputSize = 4;
hiddenSize = 5;
lambda = 0.01;
data   = randn(inputSize, 5);
labels = [ 1 2 1 2 1 ];
numClasses = 2;

stack = cell(2,1);
stack{1}.w = 0.1 * randn(3, inputSize);
stack{1}.b = zeros(3, 1);
stack{2}.w = 0.1 * randn(hiddenSize, 3);
stack{2}.b = zeros(hiddenSize, 1);
softmaxTheta = 0.005 * randn(hiddenSize * numClasses, 1);

[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ softmaxTheta ; stackparams ];


[cost, grad] = stackedAECost(stackedAETheta, inputSize, hiddenSize, ...
                             numClasses, netconfig, ...
                             lambda, data, labels);

% Check that the numerical and analytic gradients are the same
numgrad = computeNumericalGradient( @(x) stackedAECost(x, inputSize, ...
                                        hiddenSize, numClasses, netconfig, ...
                                        lambda, data, labels), ...
                                        stackedAETheta);

% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
disp('Norm between numerical and analytical gradient (should be less than 1e-9)');
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

            % When you got this working, Congratulations!!! 
            
end          
