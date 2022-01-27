classdef logGBM_model < handle & abstractSSM %abstractHMM
    
    properties
        % constants
        X0 = 1; BETA = 2; LGAMMA = 3; LSIGMA = 4;
    end
    
    methods
        %% constructor
        function o = logGBM_model(ty, varargin)
            
            % parse optional input
            p = inputParser;
            addParameter(p, 'theta', nan, @(x) length(x) == 4 || isnan(x));
            addParameter(p, 'y', nan); 
            addParameter(p, 'x', nan);
            parse(p, varargin{:});
            
            genTimePoints(o, ty, 1); 
            
            o.theta = p.Results.theta;
            if isnan(p.Results.y) 
                if any(isnan(o.theta))
                    o.theta = [3.5 2 0 -0.7]; 
                end 
                o.x = o.simulate_trajectory(o.theta);
                o.y = y_rnd(o, o.x, o.theta); 
            else
                o.x = p.Results.x;
                o.y = p.Results.y;
            end
            
            % default values
            o.prior_mu = [3 2 sqrt(2/pi) sqrt(2/pi)];
            o.prior_std = [5 5 2 2];
            
            % fixed values
            o.np = 4;
            o.tnames = {'X0';'BETA';'LOG(GAMMA)';'LOG(SIGMA)'};
            o.names = {'X0';'BETA';'GAMMA';'SIGMA'};
        end
        
        %% variable transformation
        function theta = transform(o, theta, back)
            params = [o.LGAMMA o.LSIGMA];
            if back; theta(:, params) = exp(theta(:, params));
            else; theta(:, params) = log(theta(:, params));
            end
        end
        
        %% drift and diffusion
        function a = drift(o, theta)
            a = theta(:, o.BETA) - 0.5*exp(2*theta(:, o.LGAMMA));
        end
        function b = diffusion(o, theta)
            b = exp(theta(:, o.LGAMMA));
        end
        
        %% observation density - evaluate
        function s = y_lpdf(o, y, x, theta)  
            sig = exp(theta(:, o.LSIGMA));
            s = norm_lpdf(y, x, sig);
        end
        
        %% observation density - simulate
        function y = y_rnd(o, x, theta)  
            y = x + exp(theta(:, o.LSIGMA))*randn(size(x));
        end
        
        %% transition density - evaluate
        function f = x_lpdf(o, x_current, x_prev, theta)
           a = theta(:, o.BETA) - 0.5*exp(2*theta(:, o.LGAMMA));
           gam = exp(theta(:, o.LGAMMA));
           f =  norm_lpdf(x_current, x_prev + a, gam);
        end
        
        %% transition density - simulate
        
        function x1 = initial_x_rnd(o, theta, varargin)
            x0 = theta(:, o.X0); 
            if nargin == 3
                N = varargin{:};
                x0 = repmat(x0, 1, N);
            end
            x1 = o.x_rnd(x0, theta, 0, 1); 
        end
        
        % simulate full trajectory - output will include x0
        function [x_reduced, x] = simulate_trajectory(o, theta)
            b = theta(:, o.BETA);
            g = exp(theta(:, o.LGAMMA));
            gsq = exp(2*theta(:, o.LGAMMA));

            % simulate states
            x = zeros(o.ntx, 1);
            x(1) = initial_x_rnd(o, theta, 1); 
            for i = 2:o.ntx
                x(i) = x(i-1) + b - 0.5.*gsq + g.*randn(); 
            end
            x_reduced = x(o.ity);
        end
        
        % simulate from observation time m to n starting with x_ini 
        function [x_reduced, x] = x_rnd(o, x_ini, theta, m, n, varargin)
            
            numPoints = length(m:n);
            [Nr, Nc] = size(x_ini); 

            b = theta(:, o.BETA);
            g = exp(theta(:, o.LGAMMA));
            gsq = exp(2*theta(:, o.LGAMMA));
            
            x = zeros(Nr, Nc, numPoints);
            x(:, :, 1) = x_ini;
            for i = 2:numPoints % i = i + 1
                x(:, :, i) = x(:, :, i-1) + b - 0.5*gsq + g.*randn(Nr, Nc); 
            end

            x = x(:, :, 2:end); % remove x0
            x_reduced = x(:, :, end);
        end
        
        %% prior
        
        function f = prior_lpdf(o, theta)
            f = [norm_lpdf(theta(:, o.X0), o.prior_mu(o.X0), o.prior_std(o.X0))...
                 norm_lpdf(theta(:, o.BETA), o.prior_mu(o.BETA), o.prior_std(o.BETA))...
                 logHN_lpdf(theta(:, o.LGAMMA), o.prior_std(o.LGAMMA))...
                 logHN_lpdf(theta(:, o.LSIGMA), o.prior_std(o.LSIGMA))];
        end

        function thetas = prior_rnd(o, N)
            a = o.prior_mu;
            a([o.LGAMMA o.LSIGMA]) = [0 0];
            thetas = mvnrnd(a, diag(o.prior_std.^2), N);
            thetas(:, [o.LGAMMA o.LSIGMA]) = log(abs(thetas(:, [o.LGAMMA o.LSIGMA])));
        end

        
    end % methods
    
    
    
end

%% Helper functions
% log pdf of the normal distribution
function y = norm_lpdf(x, m, s)
    y = -0.5*log(2*pi) - log(s) - 0.5*(x - m).^2./exp(2*log(s)); 
end

% log pdf of the log half-normal distribution
function y = logHN_lpdf(logx, s)
    y = 0.5*log(2/pi)-log(s)-exp(2*logx)/(2*s^2) + logx;
end
