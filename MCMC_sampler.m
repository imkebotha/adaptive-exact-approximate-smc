classdef MCMC_sampler < handle
    
    properties
        Nt;             % length of chain
        Nx;             % number of state particles
        m;              % state-space model
        LL;             % log-likelihood estimates
        tsamples;       % transformed parameter (theta) samples
        Sig;            % covariance matrix for random walk
        ini;            % initial value of chain         
        ct;             % computation time
    end
    
    methods
       % constructor
       function obj = MCMC_sampler(m, ini, varargin) 
           if nargin == 3
               obj.Sig = varargin{:};
           else
               obj.Sig = 0.1*eye(m.np);
           end
           
           obj.m = m;
           obj.ini = ini;
       end
       
       %% particle marginal Metropolis-Hastings sampler
       function sample(o, Nt, Nx)
           o.Nt = Nt;
           o.Nx = Nx;
           
           % initialise
           o.LL = zeros(Nt, 1);
           o.tsamples = zeros(Nt, o.m.np);
           acc = 0;
           
           tic; % start timer
           o.tsamples(1, :) = o.ini;

           o.LL(1) = sum(ParticleFilter.standard(o.m, o.tsamples(1, :), Nx, o.m.nty), 2);
           log_posterior = o.LL(1) + sum(o.m.prior_lpdf(o.tsamples(1, :)));
           
           for i = 1:Nt
               % print and save progress
               if mod(i, round(Nt/10)) == 0
                   sprintf('Completed: %d%%', round(i/Nt*100))
% 				   save('temp.mat', 'o');
               end

               % proposal
               theta_new = mvnrnd(o.tsamples(i, :), o.Sig);
               loglike_new = sum(ParticleFilter.standard(o.m, theta_new, Nx, o.m.nty), 2);
               log_posterior_new = loglike_new + sum(o.m.prior_lpdf(theta_new));

               % Metropolis-Hastings ratio
               MHRatio = exp(log_posterior_new - log_posterior);

               % accept/reject 
               if (rand < MHRatio) 
                   o.tsamples(i+1, :) = theta_new;
                   o.LL(i+1) = loglike_new;
                   log_posterior = log_posterior_new;
                   acc = acc + 1;
               else
                   o.tsamples(i+1, :) = o.tsamples(i, :);
                   o.LL(i+1) = o.LL(i);
               end
           end
            o.ct = toc;
       end % sample
       
    end % methods
    
end % classdef