%-------------- Multiple-Layer Echo State Network--------------------------
clear
% load the data
data = load('MackeyGlass_t17.txt');
% plot some of it
tic
figure(10);
plot(data(1:1000));
title('A sample of data');
% generate the ESN reservoir
inSize = 1; outSize = 1;
resSize = 400;
a = 0.5; % leaking rate
Win = (rand(resSize,1+inSize)-0.5) .* 1;
W = rand(resSize,resSize)-0.5;
WL1 = (rand(resSize,resSize)-0.5);
WL2 = (rand(resSize,resSize)-0.5);
WL3 = (rand(resSize,resSize)-0.5);
WL4 = (rand(resSize,resSize)-0.5);
WL5 = (rand(resSize,resSize)-0.5);
WL6 = (rand(resSize,resSize)-0.5);
Wr  = (rand(resSize,1+inSize+resSize)-0.5);
% Option 1 - direct scaling (quick&dirty, reservoir-specific):
% W = W .* 0.13;
% Option 2 - normalizing and setting spectral radius (correct, slower):
disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
rhoW1 = abs(eigs(WL1,1,'LM',opt));
rhoW2 = abs(eigs(WL2,1,'LM',opt));
rhoW3 = abs(eigs(WL3,1,'LM',opt));
rhoW4 = abs(eigs(WL4,1,'LM',opt));
rhoW5 = abs(eigs(WL5,1,'LM',opt));
rhoW6 = abs(eigs(WL6,1,'LM',opt));
disp 'done.'
W = W .* ( 0.99/rhoW);
Wr = Wr .* ( 0.99/rhoW);
WL1 = WL1 .* ( 0.99 /rhoW1);
WL2 = WL2 .* ( 0.99 /rhoW2);
WL3 = WL3 .* ( 0.99 /rhoW3);
WL4 = WL4 .* ( 0.99 /rhoW4);
WL5 = WL5 .* ( 0.99 /rhoW5);
WL6 = WL6 .* ( 0.99 /rhoW6);
error = [];
meanerror = [];
for k=1:1
timehorizon=[5]%,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150];
for l=1:50

trainLen = 6000+l;
testLen = 250+l;
timestep=timehorizon(k)
initLen = 100;
rand( 'seed', 100 );
% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = data(initLen+2:trainLen+1)';
% run the reservoir with the data and collect X
x = zeros(resSize,1);

for t = 1:trainLen
	u = data(t);
	  x = a*x + (1-a)* tanh( Win*[1;u] + W*x );    %Layer 1
    x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL1*x));  %Layer 2
     % uncomment the below code and add layers one by one
    %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL2*x));  %Layer 3
    %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL3*x));  %Layer 4
    %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL4*x));  %Layer 5
    %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL5*x));  %Layer 6
    %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL6*x));  %Layer 7
	if t > initLen
		X(:,t-initLen) = [1;u;x];
	end
end
% train the output
reg = 1e-2;  % regularization coefficient
X_T = X';
Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
y1 = data(initLen+2:200+1) * Wout;
figure(1);
plot(y1)
hold on
plot(data(initLen+2:200+1))
%Wout = Yt*pinv(X);
% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen+timestep);
u = data(trainLen+1);
   
    for t = 1:testLen+timestep 
        x = a*x + (1-a)*tanh(Win*[1;u] + W*x);       %Layer 1
        % uncomment the below code and add layers one by one
       % x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL1*x));  %Layer 2
        %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL2*x));  %Layer 3
        %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL3*x));  %Layer 4
        %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL4*x));  %Layer 5
        %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL5*x));  %Layer 6
        %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL6*x));  %Layer 7
    	y = Wout*[1;u;x];
       	Y(:,t) = y;
    	% generative mode:
    	u = y;
    	% this would be a predictive mode:
    	%u = data(trainLen+t+1);
    end
errorLen = 100;
mse = sum((data(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen;
disp( ['MSE = ', num2str( mse )] );
[ err , all_errs] = mnae(Y(1,initLen:initLen+timestep),data(trainLen+initLen+1:trainLen+initLen+timestep+1)');
% plot some signals
error(l) = err
end
e = mean(error);
meanerror(k) = e
error = []
end
fileID = fopen('output(Ridge_Regression_ML_ESN_2).txt','w');
fprintf(fileID,'%f\r\n',meanerror);
fclose(fileID);
figure(1);
plot(data(trainLen+initLen+1:trainLen+initLen+timestep+1), 'color', [0,0.75,0] );
hold on;
plot( Y(1,initLen:initLen+timestep)', 'b' );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');
figure(2);
plot( X(1:20,1:200)' );
title('Some reservoir activations x(n)');
figure(3);
bar( Wout' )
title('Output weights W^{out}');
figure(4);
plot(meanerror)
timeelapsed = toc