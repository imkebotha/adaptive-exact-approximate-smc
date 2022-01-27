classdef EASMC < handle
    %% Visible properties
    properties
        % constants -------------------------------------------------------
        m;                  % state space model
        dataAnnealing;      % true if using data annealing, false if using density tempering
        Nt;                 % number of parameter (theta) particles
        Nx_max;             % maximum allowable number of state particles
        adaptNx;            % [stage2 Option, stage3 Option]
        ESJD_target;        % target expected squared jumping distance 
        targetESS;          % target ESS (as ratio)
        k;                  % number of components for mixture model if reinit scheme is used
        iniSig;             % initial covariance matrix
        verbose;            % print progress
        
        % results ---------------------------------------------------------
        R_history;          % history of R values
        Nx_history;         % history of the number of state particles
        ct;                 % computation time
        TLL;                % number of log-likelihood calculations * number of observations
        ar;                 % mean acceptance rate
        ESJD;               % expected squared jumping distance
        post_mean;          % posterior mean
        tsamples;           % parameter samples
        temperatures;       % temperatures for density tempering
		logNWt;             % parameter normalised log weights
        update_t;
        ESS;
    end
     
    %% Hidden properties

    properties (Hidden)
        R;                  % current number of MCMC repeats
        LL;                 % incremental log-likelihood estimates
        tSig;               % covariance matrix of current sample
        logWt;              % parameter log weights
        Nx;                 % number of x samples (number of particles)
        xsamples;           % state particles
        logWx;              % state particle log weights
        GMModel;            % mixture model for reinitialisation
        adaptNxCode;
        
        reinit;             % whether the algorithm has been reinitialised
        d;                  % current density
        t;                  % current observation (data annealing)
        g;                  % current temperature (density tempering)
    end
    
    properties (Hidden, Constant)
        % options to increase the number of state particles (stage 2)
        S2_double = 1;
        S2_rescale_var = 2;
        S2_rescale_std = 3; 
        S2_novel_var = 4; 
        S2_novel_esjd = 5; 
        S2_none = 6; 
		
        % options to extend the particle set (stage 3)
        S3_reweight = 1; 
        S3_reinit = 2; 
        S3_replace = 3; 
        S3_none = 4;
    end
    
    %% Visible methods
    methods
        % -----------------------------------------------------------------
        % Constructor
        % -----------------------------------------------------------------
        function o = EASMC(m, R, Nx_max, method, varargin)
            % parse optional input
            p = inputParser;
            addParameter(p, 'tSig', 0.1*eye(m.np));
            addParameter(p, 'k', 3);
            addParameter(p, 'adaptNx', ["none", "none"]);
            addParameter(p, 'ESJD_target', 6);
            addParameter(p, 'verbose', true);
            addParameter(p, 'targetESS', 0.6);
            parse(p, varargin{:});
            
            % set general values
            o.adaptNx = p.Results.adaptNx;
            o.adaptNxCode(1) = find(["double" "rescale_var" "rescale_std" "novel_var" "novel_esjd" "none"] == o.adaptNx(1));
            o.adaptNxCode(2) = find(["reweight" "reinit" "replace" "none"] == o.adaptNx(2));
            
            o.verbose = p.Results.verbose;
            o.tSig = p.Results.tSig;
            o.k = p.Results.k;
            o.ESJD_target = p.Results.ESJD_target;
            o.iniSig = p.Results.tSig;
            o.targetESS = p.Results.targetESS;
            
            o.m = m;
            o.R_history = max(R, 2);
            o.Nx_max = Nx_max;
            
            o.dataAnnealing = method == "DataAnnealing";
        end % constructor
        
         
        % -----------------------------------------------------------------
        % Sampler
        % -----------------------------------------------------------------
        function sample(o, Nt, Nx)
            o.resetProperties();
            o.Nt = Nt; o.Nx = Nx;
            o.Nx_history = Nx;
            
            tic();
            
            % initialisation
            o.initialise();
            past_final = goToNextDistribution(o);
            
            while ~past_final
           
                % udpate x-particles (data annealing only)
                if o.dataAnnealing
                    if o.t == 1
                        [o.LL(:, 1), o.xsamples, o.logWx] = ParticleFilter.initialise(o.m, o.tsamples, o.Nx);
                    else
                        [o.LL(:, o.t), o.xsamples, o.logWx] = ...
                        ParticleFilter.iteration(o.m, o.tsamples, o.Nx, o.t, o.xsamples, o.logWx);
                        o.TLL = o.TLL + o.Nt*o.Nx*1;
                    end
                end
                
                % update theta-particle weights
                o.logWt = o.logNWt + o.incremental_logweights();
                o.logNWt = o.logWt - logsumexp(o.logWt); 
                      
                % re-sample, adapt and mutate
                resample_move = ~o.dataAnnealing || exp(-logsumexp(2*o.logNWt)) < o.targetESS*o.Nt;
                if resample_move
                    o.ESS = [o.ESS exp(-logsumexp(2*o.logNWt))];
					if o.dataAnnealing
                        o.d = o.d + 1; 
                        o.update_t = [o.update_t o.t];
                    end
                    o.resample();
                    adaptive_mutation_step(o);
                    
                    % record values used
                    o.R_history = [o.R_history o.R];
                    o.Nx_history = [o.Nx_history o.Nx];
                end
                
                past_final = goToNextDistribution(o);
            end
            
            % results
            o.ct = [o.ct toc];
            o.post_mean = mean(o.tsamples);

            % clear temporary properties
            o.R = []; o.Nx = []; o.LL = []; 
            o.logWt = []; o.xsamples = [];
            o.logWx = []; o.GMModel = [];
        end % sampler
    end
    
    
    methods (Hidden)
        
        function resetProperties(o)
            % reset
            if ~o.dataAnnealing 
                o.temperatures = 0;
            end
            o.R = o.R_history(1);
            o.tSig = o.iniSig;
            o.reinit = false;
            o.TLL = 0;
            
            % clear
            o.Nx_history = [];	o.logWx = [];       o.logWt = [];  
            o.post_mean = [];	o.GMModel = [];     o.ESS = []; 
            o.update_t = [];    o.xsamples = [];    o.ESJD = []; 
            o.ar = [];          o.LL = [];      
        end
        
    
        %% Log-likelihood, incremental weights and log-posterior
        
        %------------------------------------------------------------------
        % Estimate the log-likelihood
        %------------------------------------------------------------------
        function [ll, x, logwx] = loglikelihood(o, theta, nx)
            [ll, x, logwx] = ParticleFilter.standard(o.m, theta, nx, o.t);
            if ~o.dataAnnealing; logwx = []; end
            o.TLL = o.TLL + size(theta, 1)*nx*o.t;
        end

        
        %------------------------------------------------------------------
        % Estimate the standard deviation of the log-likelihood.
        %------------------------------------------------------------------
        function [stdLL, varLL] = std_loglike(o, nx, nsamples)
            theta_mean = repmat(mean(o.tsamples), nsamples, 1);
            LL_est = loglikelihood(o, theta_mean, nx); 
            LL_est = sum(LL_est, 2);
            stdLL = std(LL_est);
			varLL = var(LL_est);
        end
        
        
        %------------------------------------------------------------------
        % Calculate the incremental log weights
        %------------------------------------------------------------------
        function f = incremental_logweights(o)
            if o.dataAnnealing; f = o.LL(:, o.t); 
            else
                dg = (o.temperatures(o.d) - o.temperatures(o.d-1));
                f = dg*(sum(o.LL, 2) + sum(o.m.prior_lpdf(o.tsamples), 2) - o.Q_lpdf(o.tsamples));
            end
        end % incremental_logweights
        
        
        %------------------------------------------------------------------
        % Calculate the log-posterior
        %------------------------------------------------------------------
        function f = logposterior(o, LL, tsamples) 
            f = sum(LL, 2) + sum(o.m.prior_lpdf(tsamples), 2);
            if o.dataAnnealing 
                if o.t == 2; f = f - o.Q_lpdf(tsamples); end
            else
                if o.g < 1; f = o.g*f + (1-o.g)*o.Q_lpdf(tsamples); end
            end
        end  % logposterior
        
        
        %% Initialisation
        
        % -----------------------------------------------------------------
        % Functions to draw from the initial distribution and evaluate the
        % log pdf. Default initial distribution is the prior.
        %------------------------------------------------------------------
        function draws = Q_rnd(o)
            if ~o.reinit; draws = o.m.prior_rnd(o.Nt);
            else; draws = random(o.GMModel, o.Nt); end
        end 
        
        function f = Q_lpdf(o, tsamples)
            if ~o.reinit; f = sum(o.m.prior_lpdf(tsamples), 2);
            else; f = loggmmpdf(o.GMModel, tsamples); end
        end 
        
        
        % -----------------------------------------------------------------
        % Initialise the parameter particles, state particles and weights
        % -----------------------------------------------------------------
        function initialise(o)
            o.d = 1;
            o.tsamples = o.Q_rnd(); 
                
            if o.dataAnnealing 
                o.t = 0; 
                if o.verbose; fprintf('Completing iteration: 0 of %d\n', o.m.nty); end
            else 
                o.t = o.m.nty;
                o.g = o.temperatures(o.d); 
                if o.verbose; fprintf('Current temperature: %0.4f\n', o.g); end
                [o.LL, o.xsamples] = o.loglikelihood(o.tsamples, o.Nx);
            end
            o.logNWt = -log(o.Nt)*ones(o.Nt, 1); 
        end
        
        %% Transition to the next distribution
        
        % -----------------------------------------------------------------
        % If using density tempering, adapt the temperature using the
        % bisection method
        % -----------------------------------------------------------------
        function setNewTemperature(o)

            busy = true;
            a = o.g; b = 1; % initial bounds

            ESSa_diff = exp(-logsumexp(2*o.logNWt)) - o.targetESS*o.Nt;
            const = sum(o.m.prior_lpdf(o.tsamples), 2) ...
                - o.Q_lpdf(o.tsamples);

            if ESSa_diff < 0
                busy = false;
                p = min(b, a + 0.005);
            end

            while (busy)

                % calculate ESS for temperature p
                p = (a + b)/2;
                logw = o.logNWt + (p - o.g)*(sum(o.LL, 2) + const);
                lognw = logw - logsumexp(logw); 
                ESSp_diff = exp(-logsumexp(2*lognw)) - o.targetESS*o.Nt;

                % update bounds
                if ESSa_diff*ESSp_diff < 0
                    b = p;
                else
                    a = p;
                    ESSa_diff = ESSp_diff;
                end

                % set new values
                busy = abs(ESSp_diff) > 1e-2;
                if (b-a) <= 1e-4
                    p = b;
                    busy = false;
                end
            end
            o.g = p;
            o.temperatures = [o.temperatures p];
        end % setNewTemperature
        
        % -----------------------------------------------------------------
        % Move to the next distribution in the sequence
        % -----------------------------------------------------------------
        function past_final = goToNextDistribution(o)
            past_final = false;
            
            if o.dataAnnealing && (o.t + 1) <= o.m.nty
                o.t = o.t + 1;
            elseif ~o.dataAnnealing && (o.g < 1)
				o.d = o.d + 1;
                o.setNewTemperature();
            else
                past_final = true;
            end
            
            if o.verbose && ~past_final
                if o.dataAnnealing 
                    fprintf('Completing iteration: %d of %d\n', o.t, o.m.nty);
                else 
                    fprintf('Current temperature: %0.4f\n', o.g);
                end
            end
        end
        
        
        %% Mutate particles
        
        % -----------------------------------------------------------------
        % Resample the particles
        % -----------------------------------------------------------------
        function resample(o)
            I = randsample([1:o.Nt]', o.Nt, true, exp(o.logNWt));
            
            o.tsamples = o.tsamples(I, :);
            o.logWt = zeros(o.Nt, 1);
            o.logNWt = -log(o.Nt)*ones(o.Nt, 1);
            o.LL = o.LL(I, :);
            
            if o.m.Dx == 1
                o.xsamples = o.xsamples(I, :, :);
            else
                for i = 1:o.m.Dx
                    o.xsamples{i} = o.xsamples{i}(I, :, :);
                end
            end
            
            if o.dataAnnealing; o.logWx = o.logWx(I, :); end
        end
        

        % -----------------------------------------------------------------
        % Particle marginal Metropolis Hastings mutation kernel
        % -----------------------------------------------------------------
        function [esjd, I] = PMMH_mutation_kernel(o, varargin)
            
            if isempty(varargin)
				nx = o.Nx;
			else
				nx = varargin{:};
            end
            
            % current posterior
            log_posterior = o.logposterior(o.LL, o.tsamples);
            
            % proposal
            theta_new = mvnrnd(o.tsamples, o.tSig); 
            [LL_new, Xm_new, logW_new] = o.loglikelihood(theta_new, nx);
            log_posterior_new = o.logposterior(LL_new, theta_new);

            % Metropolis-Hastings ratio
            MHRatio = exp(log_posterior_new - log_posterior);
            I = rand(o.Nt, 1) < MHRatio;
            
            % expected squared jumping distance
            dtheta = (o.tsamples - theta_new); 
            distj = diag(dtheta/(o.tSig)*dtheta');
            esjd = distj.*min(1, MHRatio);

            % update values
            o.tsamples(I, :) = theta_new(I, :);
            o.LL(I, :) = LL_new(I, :);
            
            if o.m.Dx == 1
                o.xsamples(I, :, :) = Xm_new(I, :, :);
            else
                for i = 1:o.m.Dx
                    o.xsamples{i}(I, :, :) = Xm_new{i}(I, :, :);
                end
            end
            
            if o.dataAnnealing; o.logWx(I, :) = logW_new(I, :); end
        end
        

        % -----------------------------------------------------------------
        % Adaptive move step 
        % -----------------------------------------------------------------
        function adaptive_mutation_step(o)
           oNx = o.Nx; oR = o.R; adapt = false;
           
           % trigger adaptation of Nx
           if ~isempty(o.ESJD)
               adapt = o.ESJD(end) < o.ESJD_target;
               if (o.adaptNxCode(1) ~= o.S2_double) && (o.adaptNxCode(1) ~= o.S2_none) && (o.adaptNxCode(2) ~= o.S3_reinit)
                   adapt = adapt || (length(o.ESJD) > 1 && o.ESJD(end) > 2*(o.ESJD_target));
               end
           end
           
           % do adaptation
           if adapt && (o.adaptNxCode(2) ~= o.S3_reinit)
               if o.adaptNxCode(1) == o.S2_novel_esjd
                   [esjd, accepted] = o.increaseNx();
               else
                   if (o.adaptNxCode(1) ~= o.S2_none)
                       o.increaseNx();
                       if o.Nx ~= oNx; updateParticleSet(o); end
                   end
                   % initial mutation step and adaptation of R
                   [esjd, accepted] = PMMH_mutation_kernel(o);
                   o.R = max(1, ceil(o.ESJD_target./mean(esjd)));
               end
           else
               % initial mutation step
               [esjd, accepted] = PMMH_mutation_kernel(o);
           end

            % remaining mutations 
            for r = 2:o.R
               [esjdj, I] = o.PMMH_mutation_kernel();
               esjd = esjd + esjdj; accepted = accepted | I;
            end

            o.tSig = cov(o.tsamples);
            o.ar = [o.ar mean(accepted)]; 
            o.ESJD = [o.ESJD mean(esjd)];
            
            % print results
            if oNx ~= o.Nx; fprintf('**Changing Nx from %d to %d**\n', oNx, o.Nx); end
            if oR ~= o.R; fprintf('**Changing R from %d to %d**\n', oR, o.R); end
            
            % adaptation for reinit
            if (o.adaptNxCode(2) == o.S3_reinit) && (o.ESJD(end) < o.ESJD_target)
                o.increaseNx();
                if o.Nx ~= oNx
                    updateParticleSet(o); 
                    o.R = max(o.R, ceil(o.ESJD_target./o.ESJD(end)));
                    fprintf('**Changing Nx from %d to %d**\n', oNx, o.Nx)
                    fprintf('**Changing R from %d to %d**\n', oR, o.R);
                    return
                end
            end
            
        end % adaptive_mutation_step
        
        %% Helper functions for adaptation

        % -----------------------------------------------------------------
        % Select a new number of state particles
        % -----------------------------------------------------------------
        function [esjd, I] = increaseNx(o)
            esjd = zeros(o.Nt, 1); 
            I = zeros(o.Nt, 1);

            switch o.adaptNxCode(1)
                case o.S2_double 
                    o.Nx = min(2*o.Nx, o.Nx_max);
                    
                case o.S2_rescale_var 
                    [~, varLL] = o.std_loglike(o.Nx, 100);
                    nx = min(ceil(o.Nx*varLL), o.Nx_max);
                    if o.adaptNxCode(2) == o.S3_reinit
                        o.Nx = max(o.Nx, nx);
                    else
                        o.Nx = nx;
                    end
                                
                case o.S2_rescale_std 
                    stdLL = o.std_loglike(o.Nx, 100);
                    nx = min(ceil(o.Nx*stdLL), o.Nx_max);
                    if o.adaptNxCode(2) == o.S3_reinit
                        o.Nx = max(o.Nx, nx);
                    else
                        o.Nx = nx;
                    end
					
                case o.S2_novel_var
                    
                    s = 1; 
                    if ~o.dataAnnealing
						s = max(o.g^2, 0.36);
                    end
                    
                    target_var = 1^2 / s;
					min_var = 0.95^2 / s; 
					max_var = 1.05^2 / s; 
                    
                    % estimate variance of log-likelihood estimator
                    theta_vals = repmat(mean(o.tsamples), 100, 1); 
					LL_est = ParticleFilter.standard(o.m, theta_vals, o.Nx, o.t);
					o.TLL = o.TLL + size(theta_vals, 1)*o.Nx*o.t;
                    LL_est = sum(LL_est, 2);
                    varLL = var(LL_est);

					
					if varLL > max_var || varLL < min_var

						v = (varLL/target_var).^((2:4)'/4);
						Nxj = ceil(o.Nx*v/10)*10; % round up to the nearest 10
						Nxj(Nxj > o.Nx_max) = o.Nx_max;
						Nxj = unique(Nxj); % sort in ascending order
						Nxj = Nxj(Nxj ~= o.Nx); 
						n = length(Nxj);
						
						if n == 1
							o.Nx = Nxj;
						elseif n > 1

							vs = zeros(n, 1);
							m_star = n;
							for i = 1:n-1 
								LL_est = loglikelihood(o, theta_vals, Nxj(i)); 
								LL_est = sum(LL_est, 2);
								vs(i) = var(LL_est);
								if vs(i) < max_var
									m_star = i;
									break;
								end
							end 
							o.Nx = Nxj(m_star);
						end
					end
                    
                case o.S2_novel_esjd
                    [stdLL, varLL] = o.std_loglike(o.Nx, 100);
                    
                    if o.dataAnnealing
						v = [1; 2; stdLL; varLL]; 
                    else
                        v = [stdLL; varLL].*max(o.g^2, 0.6^2);
                        v = [1; 2; v]; 
                    end
					
					Nxj = ceil(o.Nx*v/10)*10; % round up to the nearest 10
					Nxj(Nxj > o.Nx_max) = o.Nx_max;
					Nxj = unique(Nxj); % sort in ascending order
					n = length(Nxj);
					
                    score = zeros(n, 1);
                    r = zeros(n, 1);

                    % mutation at current Nx
                    ic = find(Nxj == o.Nx);
                    currentNx = o.Nx; % current Nx
                    [esjd, accepted] = o.PMMH_mutation_kernel();
                    r(ic) = max(1, ceil(o.ESJD_target./mean(esjd)));
                    score(ic) = 1/(Nxj(ic) * r(ic)); 
						
                    m_star = n;
                    for i = 1:n
                        if Nxj(i) ~= o.Nx
                            currentNx = Nxj(i);
                            [o.LL, o.xsamples, o.logWx] = o.loglikelihood(o.tsamples, currentNx);
                            [esjdj, I] = o.PMMH_mutation_kernel(currentNx);
                            esjd = esjd + esjdj; accepted = accepted | I;
                            r(i) = max(1, ceil(o.ESJD_target./mean(esjdj)));
                            score(i) = 1/(currentNx * r(i)); 
                        end

                        if i > 1
                            relative_score = score(i)/score(i-1);
                            if relative_score < 0.95
                                m_star = i-1;
                                break;
                            elseif relative_score < 1.05
                                m_star = i;
                                break;
                            end
                        end
                    end 
                    if currentNx ~= Nxj(m_star)
                        [o.LL, o.xsamples, o.logWx] = o.loglikelihood(o.tsamples, Nxj(m_star));
                    end
                    o.Nx = Nxj(m_star);
                    o.R = r(m_star);
            end

        end % increaseNx
        

        % -----------------------------------------------------------------
        % Update the particle set for the new number of state particles
        % -----------------------------------------------------------------
        function updateParticleSet(o)
            switch o.adaptNxCode(2)
                case o.S3_reweight 
                    [LL_new, o.xsamples, o.logWx] = o.loglikelihood(o.tsamples, o.Nx);
                    
                    % reweight
                    ll = sum(LL_new, 2) - sum(o.LL, 2);
                    if ~o.dataAnnealing; ll = o.temperatures(o.d)*ll; end
                    o.logWt = o.logNWt + ll;
                    o.logNWt = o.logWt - logsumexp(o.logWt);
                    o.LL = LL_new;

                case o.S3_replace 
                    [o.LL, o.xsamples, o.logWx] = o.loglikelihood(o.tsamples, o.Nx);

                case o.S3_reinit % re-initialise
                    % fit mixture of Gaussians to current particle set
                    o.reinit = true;
                    o.GMModel = fitgmdist(o.tsamples, o.k, 'RegularizationValue', 1e-6);

                    % reset/reinitialise all properties
                    o.d = 1; o.LL = []; o.xsamples = []; o.logWx = []; 
                    o.initialise();
            end % switch
        end % updateParticleSet

    end % methods (Hidden)
    
  
end
