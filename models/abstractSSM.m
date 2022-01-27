classdef abstractSSM < handle
    properties
        y;          % observations
        
        x;          % latent states (true) if known
        Dx = 1;     % dimension of latent states
        theta;      % parameters (true) if known or initial values otherwise
        
        ty;         % observation times
        tx;         % latent state time points
        
        ity;        % position of observation times within tx
        dtx;        % discretised time steps
        
        nty;        % number of observations
        ntx;        % number of latent state points

        np;         % number of parameters
        tnames;     % names of transformed parameters
        names;      % names of untransformed parameters
        
        prior_mu;   % prior means
        prior_std;  % prior standard deviations
    end
    
    methods 
        function genTimePoints(ssm, obs_times, dt, varargin) 
            
            p = inputParser;
            addParameter(p, 'knowX0', true);
            parse(p, varargin{:});
            
            if p.Results.knowX0; txIni = 0;
            else; txIni = 1; end
            
            if obs_times(1) == 0 
                obs_times = obs_times + 1; 
            end
            
            
            % Given the observation times, find the time points for the
            % latent states using dt
            if length(obs_times) > 1
                dx = diff(obs_times)./ceil(diff(obs_times)/dt); % discretisation of latent states
                a = arrayfun(@(a, b, c) a:b:c, obs_times(1:end-1), dx, obs_times(2:end), 'UniformOutput', false);
                ssm.tx = unique([a{:}]);
                ssm.tx = [0:dt:ssm.tx(1)-dt ssm.tx];
                ssm.ty = obs_times;
            else
                ssm.tx = txIni:dt:obs_times;
                if ssm.tx(end) ~= obs_times
                   ssm.tx = [ssm.tx obs_times]; 
                end
                ssm.ty = 1:obs_times; 
            end
            
            [~, ssm.ity] = ismember(ssm.ty, ssm.tx);
            
            ssm.dtx = diff(ssm.tx);
            ssm.ntx = length(ssm.tx);
            ssm.nty = length(ssm.ty);
        end
        
    end
    
    methods (Abstract)
        % obervation density
        y_lpdf(o);
        y_rnd(o);
        
        % transition density
        initial_x_rnd(o);
        x_rnd(o);
        
        % prior
        prior_lpdf(o);
        prior_rnd(o);
    end
end