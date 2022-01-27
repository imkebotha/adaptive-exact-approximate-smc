
addpath(genpath('.'));

%% load model

mnum = 1;

switch mnum
    case 1 % Brownian motion model
        load('lgbm_model.mat', 'm');
        Nx_gold_standard = 240; 
        Sig = [2.9312  -0.0015   0.0089  -0.0065
              -0.0015   0.1643   0.0576  -0.0565
               0.0089   0.0576   0.0242  -0.0226
              -0.0065  -0.0565  -0.0226   0.0443];
          
    case 2 % stochastic volatility model
        load('sv_model.mat', 'm');
        Nx_gold_standard = 1650; 
        Sig = [0.0451   0.0783  -0.0105  -0.1249  -0.0147
               0.0783   0.1708  -0.0224  -0.2445   0.1110
              -0.0105  -0.0224   0.0256   0.0263   0.0650
              -0.1249  -0.2445   0.0263   0.4299  -0.1659
              -0.0147   0.1110   0.0650  -0.1659   1.4907];
          
    case 3 % theta-logistic model
        load('tl_model.mat', 'm');
        Nx_gold_standard = 4600; 
          Sig = [0.5181    0.0058   -0.0031   -0.0137    0.0867   -0.0237   -0.0829
                 0.0058    0.0029   -0.0102   -0.0141    0.0013   -0.0040   -0.0008
                -0.0031   -0.0102    0.7003    0.0928   -0.0031   -0.0340    0.0002
                -0.0137   -0.0141    0.0928    0.3041   -0.0029    0.0341    0.0012
                 0.0867    0.0013   -0.0031   -0.0029    0.0223   -0.0272   -0.0144
                -0.0237   -0.0040   -0.0340    0.0341   -0.0272    0.4590    0.0034
                -0.0829   -0.0008    0.0002    0.0012   -0.0144    0.0034    0.0139];
        
    case 4 % noisy Ricker model
        load('ricker_model.mat', 'm');
        Nx_gold_standard = 90000; 
        Sig = [0.0003  -0.0012  -0.0002
              -0.0012   0.0052   0.0009
              -0.0002   0.0009   0.0029];
end


%% run particle marginal Metropolis-Hastings

rng(2)
Nt = 10000; % length of chain

pmmhobj = MCMC_sampler(m, m.theta, Sig);
pmmhobj.sample(Nt, Nx_gold_standard);

plot_samples(m, pmmhobj.tsamples, 'TracePlots', true)


%% run exact-approximate SMC

rng(2)

Nt = 1000; % number of parameter particles
Nx = 10; % initial number of state particles
R = 10; % initial number of MCMC iterations
Nx_max = Nx_gold_standard * 5; % maximum number of state particles

method = "DensityTempering";
% method = "DataAnnealing";

adaptation = ["novel_esjd", "replace"];
% adaptation = [A, B]
% A = "double", "rescale_var", "rescale_std", "novel_var", "novel_esjd", "none"
% B = "reweight", "reinit", "replace", "none"

easmcobj = EASMC(m, R, Nx_max, method, 'adaptNx', adaptation, 'ESJD_target', 6, 'targetESS', 0.6);
easmcobj.sample(Nt, Nx);

plot_samples(m, easmcobj.tsamples, 'Weights', exp(easmcobj.logNWt))



