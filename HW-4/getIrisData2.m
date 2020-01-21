function [ trainingSet, testSet, y1, y2 ] = getIrisData()

load('irisData')

trainingSet = zeros(5,120);
testSet = zeros(5,30);
y1 = zeros(120,1);
y2 = zeros(30,1);

trainingSet(1:4,1:40) = X(1:40,:)';
trainingSet(1:4,41:80) = X(51:90,:)';
trainingSet(1:4,81:120) = X(101:140,:)';
trainingSet(5,:) = 1;

testSet(1:4,1:10) = X(41:50,:)';
testSet(1:4,11:20) = X(91:100,:)';
testSet(1:4,21:30) = X(141:150,:)';
testSet(5,:) = 1;

for i=1:40
    y1(i,:) = [0.1];
    y1(i+40,:) = [0.5];
    y1(i+80,:) = [1];
end

for i=1:10
    
    y2(i,:) = [0.1];
    y2(i+10,:) = [0.5];
    y2(i+20,:) = [1];
    
end

end

