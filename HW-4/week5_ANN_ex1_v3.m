clear
clc


% XOR input for x1 and x2
input = [0 0; 0 1; 1 0; 1 1];
input=input';
% Desired output of XOR
groundTruth = [0;1;1;0];
% Initialize the bias
bias = [-1 -1 -1];
% Learning coefficient
coeff = 0.7;
% Number of learning iterations
iterations = 10000;
% Calculate weights randomly using seed.
rand('state',sum(100*clock));
%weights = -1 +2.*rand(3,3);

inputLength=2;
hiddenN=2;
outputN=1;

w1=ones(hiddenN,inputLength+1);%add bias term as +1
w1(1,:)=0.1;
w1(2,:)=0.2;

w2=ones(outputN,hiddenN+1);%add bias term as +1
w2(1,:)=0.3;

for i = 1:iterations
   out = zeros(4,1);

   for j = 1:4
       
      inputs=input(:,j);
      % Hidden layer1      
      HL1 = w1*[-1; inputs]; 
      %Be carefull: look at th place of bias
      %since the w11 is reserved for bias, thus the bias must be the first
      %input 

      % Send data through sigmoid function 1/1+e^-x
      % Note that sigma is a different m file 
      % that I created to run this operation
      HiddenLayerOutput1 = sigm(HL1);
  
      % Output layer
      x3_1 = w2*[-1;HiddenLayerOutput1];
    
             
      out(j) = sigm(x3_1);
      
      % Adjust delta values of weights
      % For output layer:
      % delta(wi) = xi*delta,
      % delta = f'(x3_1)*epsilon
      %epsilon=(desired output - actual output) 
      delta3 = sigmDerivative(x3_1)*(groundTruth(j)-out(j));
      
      % Propagate the delta backwards into hidden layers
      % we have two inner nodes, thus there are two propagation
      delta2 = sigmDerivative(HL1).*w2(:,2:end)'*delta3;
       
      
      % Add weight changes to original weights 
      % And use the new weights to repeat process.
      % delta weight = coeff*x*delta
      w2=w2+coeff*[-1; HiddenLayerOutput1]'.*delta3;
      w1=w1+coeff*delta2*([-1;inputs])';
      
      
   end   
end

%test the code

input = [0 0; 0 1; 1 0; 1 1];
input=input';
% Desired output of XOR
groundTruth = [0;1;1;0];
out = zeros(4,1);

   for j = 1:4
     inputs=input(:,j);
      % Hidden layer1      
      HL1 = w1*[-1; inputs]; 
      %Be carefull: look at th place of bias
      %since the w11 is reserved for bias, thus the bias must be the first
      %input 

      % Send data through sigmoid function 1/1+e^-x
      % Note that sigma is a different m file 
      % that I created to run this operation
      HiddenLayerOutput1 = sigm(HL1);
  
      % Output layer
      x3_1 = w2*[-1;HiddenLayerOutput1];
    
             
      out(j) = sigm(x3_1);
      
   end