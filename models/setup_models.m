%% Brownian motion model

rng(2)
T = 100; 
theta = [1 1.2 log(1.5) log(1)]; 
m = logGBM_model(T, 'theta', theta);
figure; plot(m.ty, m.y, 'o-')
% save('lgbm_model.mat', 'm')

%% Stochastic volatility model

rng(2)
T = 200;
theta = [log(4) log(4) log(0.5) 5 0];
m = SV_model(T, 'theta', theta);
figure; plot(m.ty, m.y, '-');
% save('sv_model.mat', 'm');

%% Theta-logistic model

load('data_nutria.mat', 'y')
y = y(1:100);
T = length(y);
m = ThetaLogistic_model(T, 'y', log(y));
figure; plot(m.ty, m.y, '-');
% save('tl_model.mat', 'm');

%% Noisy Ricker model

rng(6)
T = 700; 
x0 = 1; 
theta = [log(10) 3.8 log(0.6)];
pr_bounds = [1.61 3; 2 5; -1.8 1]; 
m = Ricker_model(T, x0, 'theta', theta, 'pr_bounds', pr_bounds);
figure;plot(m.ty, m.y, 'o-');
% save('ricker2_model.mat', 'm')




