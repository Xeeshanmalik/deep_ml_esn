function ESN = pattern()
clear
trainLen = 500;
testLen = 305;

data = load('CalproProg_un.csv');
uu1 = data;
data = data(:,1:56);

dim = size(data,2);
inSize = dim; outSize=1;
resSize = 100;
a = 0.5;
rand('seed', 42);
Win=(rand(resSize,1+inSize)-0.8) .* 1;
W  = rand(resSize,resSize)-0.5;
WL1 = (rand(resSize,resSize)-0.5);
WL2 = (rand(resSize,resSize)-0.5);
Wr  = (rand(resSize,1+inSize+resSize)-0.3) ;


disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
rhoW1 = abs(eigs(WL1,1,'LM',opt));
rhoW2 = abs(eigs(WL2,1,'LM',opt));
disp 'done.'
W = W .* (1.25/rhoW); %optimizing the weights of the reservoir to their
%Wr = Wr .* (1.25/rhoW);
WL1 = WL1 .* ( 1.25 /rhoW1);
WL2 = WL1 .* ( 1.25 /rhoW2);

%spectral radius
X = zeros(1+inSize+resSize,trainLen);
Yt= uu1(1:trainLen,58);
x = zeros(resSize,1);
for t = 1:trainLen
    u = data(t,:)';
    x = (1-a)*x + a* tanh(Win*[1;u] + W*x);
    x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL1*x));  %Layer 2
    %x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL2*x));  %Layer 2
    X(:,t) = [1;u;x];
end
reg = 1e-8; %regularization co-efficient
X_T = X';
%size(X)
%size(Yt)
%size(reg*eye(dim+inSize+resSize))
%Wout = Yt' * X_T * inv(X*X_T + reg*eye(dim+inSize+resSize));
Wout = Yt'*pinv(X);
%size(Wout)
%size([1;u;x])
Y = Wout * X_T'

E = sum(Y' - Yt).^2/trainLen
%E = ((E))
disp( ['MSE = ', num2str(E)] );


Y = zeros(outSize,testLen);
u = data(trainLen+1,:)';
Yt1= uu1(501:805,58);
Yt1
pause;
for t = 1:testLen 
	x = (1-a)*x + a*tanh(Win*[1;u] + W*x );
  x = a*x + (1-a)*(tanh(Wr*[1;u;x] + WL1*x));  %Layer 2
	y = Wout* [1;u;x];
	Y(:,t) = y;
	% generative mode:
% 	u = y;
	% this would be a predictive mode:
     if t < testLen
 	u = data(trainLen+t+1,:)';
     end
end
R = round(Y)
size(Y)
size(Yt1)

mse = sum((Y'-Yt1).^2)./testLen;
disp( ['MSE = ', num2str( mse )] );

disp( ['MSE = ', num2str(mse)] );

% plot some signals
figure(1);
plot(Yt1(1:50),'rs','LineWidth',2,...
                'MarkerEdgeColor','k',...
                'MarkerFaceColor','b',...
                'MarkerSize',10)
%plot( Yt1, 'color', 'b','LineWidth',2);
hold on;
%plot( abs(R), 'color', 'r');
plot(R(1:50),'rs','LineWidth',2,...
                'MarkerEdgeColor','g',...
                'MarkerSize',10)
hold off;
axis tight;
title('Target and predicted signals for the whole Testing Data');
legend('Target signal', 'predicted signal');

figure(2);
plot(X(1:20,1:200)');
title('Some reservoir activations x(n)');

figure(3);
bar( Wout' )
title('Output weights W^{out}');

figure(4);
 [X,Y] = perfcurve(Yt1,Y',1);
 plot(X,Y);
%plotroc(Yt1,Y');
end

function vars = sbs(x,y,p,mtype)
 
% function [vars] = sbs (x,y,p,mtype)
% 
% Stepwise backward selection of variables based, at each step,
% on building a model of type 'mtype'. 
% 
% Starting with all the variables, sbs removes the variable which
% leads to the smallest increase in prediction error.
%
% Variable removal stops
% when the new candidate model significantly increases
% the prediction error.
%
% Significance is measured by a partial F-test:
% see p.128 Kleinbaum or p.229 in Numerical Recipes 1996.
%
% If you wish to add your own model type you will need a function
% which returns SSEXP = SSY-SSE and SSE=sum((y-ypred).^2) where
% SSY = sum((y-mean(y)).^2) and ypred is the prediction of that model.
%
% x		inputs
% y		vector of targets
% p		significance level; DEFAULT=0.05
% mtype		model type 'lin'or 'mlp'; DEFAULT='lin'
%
% vars		selected variables 
  
if nargin < 2, error('Error in sfs: at least two arguments required'); end
if nargin < 3 | isempty(p), p=0.05; end
if nargin < 4 | isempty(mtype), mtype='lin'; end

nvars=size(x,2);
N=size(x,1);

% Assign variable list, v, to include all variables
v=[1:1:nvars];
[w,old_ssexp,old_sse] = sfslin (x(:,v),y);

for i=2:nvars,
  nm=length(v);
  % Get next nm models by removing each of selected variables in turn
  for j=1:nm,
	vars=remove_var (j,v);
	switch mtype
	 case 'lin',
	  %[model(j).w,model(j).ssexp,model(j).sse] = sfslin (x(:,vars),y);
	  [w,ssexp,sse] = sfslin (x(:,vars),y);
	  model(j).w=w;
	  model(j).ssexp=ssexp;
	  model(j).sse=sse;
	  model(j).vars=vars;
	 case 'mlp',
	  disp('Error in sbs: mlp model not yet implemented');
	  vars=[];
	  return
	 otherwise,
	  disp('Error in sbs: unknown model type');
	  vars=[];
	  return
	end
  end
  % Find best new model - this suggests removing variable 'worst'
  [tmp,worst]=max([model(1:nm).ssexp]);
  
  % Get p-value for new model
  k=length(v)-1;  

  [tmp,pval] = partialf(old_ssexp,old_sse,model(worst).ssexp,N,k);

  % Remove feature if it doesn't make a significant difference
  if pval > p
    %disp(sprintf('Removing variable %d, p=%1.4f',v(worst),pval));
    v=model(worst).vars;
    old_ssexp=model(worst).ssexp;
    old_sse=model(worst).sse;
  else
    vars=v;
    break
  end
end

end

function [w,ssexp,sse] = sfslin (x,y)

% function [w,ssexp,sse] = sfslin (x,y)
% Linear regression model
% x		inputs
% y		targets
%
% w		weight vector
% ssexp         sum of squares explained by model (ssy - sse)
% sse           sum of squared errors from model

n=size(x,1);
xx=[x, ones(n,1)];
y=y(:);

% w(n+1) will be bias term
w=pinv(xx)*y;

w=w(:);
xx=[x,ones(n,1)];
ypred=xx*w;

my=mean(y);
ssy=sum((y-my).^2);
sse=sum((y-ypred).^2);
ssexp=ssy-sse;
end

function [vars] = remove_var (j,v)

% function [vars] = remove_var (j,v)
% Remove jth entry from list
% j		variable to remove
% v		list of variables 
  
nv=length(v);

vars=[];
for i=1:nv,
	if ~(i==j)
	  vars=[vars, v(i)];
	end
end
end

function [f,p] = partialf (ssexp,sse,ssexpold,n,k)

% function [f,p] = partialf (ssexp,sse,ssexpold,n,k)
% Calculate partial f statistic 
% see p.128 Kleinbaum and p.229 Press
% ssexp		sum of squares explained by model (ssy - sse)
% sse		sum of squared errors from model
% ssexpold	sum of squares explained by old model (ssy-sseold)
% n		number of data points
% k		number of variables in old model
%
% f		partial f statistic
% p		significance of partial f statistic
  
if ((ssexp-ssexpold)==0 | sse==0)
        f=0;
	p=1;
else
        f = ((ssexp-ssexpold)*(n-k-2))/sse;
	p=ppartialf(f,n,k);
end

end

function [p] = ppartialf (f,n,k)

% function [p] = ppartialf (f,n,k)
% Calculate significance of partial f statistic 
% see p.128 Kleinbaum and p.229 Press
% f		partial f statistic
% n		number of data points
% k		number of variables in old model
%
% p		p value
  
if (f < 0.0)
      % Negative f will create x>1.0 thus causing 
      % an error in the betai routine 
	p=1;
else 
        v1=1;
        v2=n-k-2;
        x=v2/(v2+f*v1);
        if (x<0.0 | x>1.0)
              disp('Error in ppartialf: x must be between 0 and 1');
	end      
        p=betainc(x,v2/2,v1/2);
end


end