clear
clc
close all

% now make a real experiment

[ trainingSet, testSet, y1, y2 ] = getIrisData();

save trainingSet trainingSet
save testSet testSet
save y1 y1
save y2 y2
% train data
input = trainingSet;
% train labels
groundTruth = y1;

% Learning coefficient
coeff = 0.1;
% Number of learning iterations
iterations = 10000;

% Calculate weights randomly using seed.
rand('state',sum(100*clock));
%weights = -1 +2.*rand(3,3);

inputLength=size(input,1);
hiddenN=5; %  nodes of hidden layer1
outputN=size(groundTruth,2);
num_layers=3;
tol=0.1;

%initialize parameters
stack = cell(hiddenN+1,1); 
for k=1:num_layers-1   
    if k==1
        stack{k}.w = rand(hiddenN,inputLength);
        stack{k}.b = rand(hiddenN,1);    
    elseif k==num_layers-1
        stack{k}.w = rand(outputN,hiddenN);
        stack{k}.b = rand(outputN,1);
    else
        stack{k}.w = rand(hiddenN,hiddenN);
        stack{k}.b = rand(hiddenN,1);
    end
end
 
outputStack = cell(num_layers,1); %keep outputs
gradStack = cell(num_layers,1); %keep gradients

fig=figure; 
hold on;
% plot(1,1, 'b*');
% uiwait(fig,1)

for i = 1:iterations
     
    err=0;
    
    for j = 1:size(y1,1)       
      inputs=input(:,j);      
      outputStack{1}=inputs;
      % forward propagation
      for k=1:num_layers-1          
          outputStack{k+1} = stack{k}.w* outputStack{k}+stack{k}.b*-1;
          outputStack{k+1} = sigm(outputStack{k+1});
      end
    
      % backward propagation
      p = outputStack{end};
      
      epsilon = nnloss(groundTruth(j,:)', p, 1);
      cost = nnloss(groundTruth(j,:)', p, 0);
      err = err+cost;
      
     
      
      for k=num_layers:-1:2          
          gradStack{k-1}.epsilon = outputStack{k}.*(1-outputStack{k}).*epsilon;
          epsilon = stack{k-1}.w'*gradStack{k-1}.epsilon;         
      end

      % Add weight changes to original weights 
      % And use the new weights to repeat process.
      % delta = coeff*x*epsilon      
      for k=1:num_layers-1
          stack{k}.w = stack{k}.w + coeff*gradStack{k}.epsilon*outputStack{k}';
          stack{k}.b = stack{k}.b + coeff*(-1).*gradStack{k}.epsilon;          
      end

    end    
  
    if mod(i,100)==0
 
        hold on;
        plot(i,err, 'b*-');
        uiwait(fig,1)
    end
    
    if err<tol
        break;
    end
   
end

save stack stack

load stack stack
load testSet testSet
load y2 y2
%test the code
input = testSet;
tol=0.1;
groundTruth = y2;
out = zeros(size(y2,1),size(y2,2));
outputStack = cell(num_layers,1); 
count=0;
for j = 1:size(y2,1)
    inputs=input(:,j);
    outputStack{1}=inputs;
    % forward propagation
    for k=1:num_layers-1
      outputStack{k+1} = stack{k}.w* outputStack{k}+stack{k}.b*-1;
      outputStack{k+1} = sigm(outputStack{k+1});
    end
    out(j,:)=outputStack{end}';
    epsilon = (groundTruth(j,:) - out(j,:));
    err=sum(epsilon.^2);
    if err<tol
        count=count+1;
    end
end
   
disp('accuracy of system:')
acc = (count/size(out,1))*100
   