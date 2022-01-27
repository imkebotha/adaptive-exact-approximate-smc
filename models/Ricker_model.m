classdef Ricker_model < handle & abstractSSM 

    properties
        % constants
        LPHI = 1; LR = 2; LSIGMA = 3;
        
        x0; % initial population
        pr_bounds;
    end
    
    methods
        %% constructor
        function o = Ricker_model(ty, x0, varargin)
            
            % parse optional input
            p = inputParser;
            addParameter(p, 'theta', nan, @(x) length(x) == 3 || isnan(x));
            addParameter(p, 'y', nan); 
            addParameter(p, 'x', nan);
            addParameter(p, 'pr_bounds', nan);
            addParameter(p, 'dt', 1);
            parse(p, varargin{:});
            
            genTimePoints(o, ty, p.Results.dt); 
            
            o.x0 = x0;
            
            % For now assume that nty = ntx
            o.theta = p.Results.theta;

            if isnan(p.Results.y) 
                if any(isnan(o.theta))
                     o.theta = [log(10) 3.8 log(0.3)]; 
                end
           
                o.x = o.simulate_trajectory(o.theta);
                o.y = y_rnd(o, o.x, o.theta); 
            else
                o.x = p.Results.x;
                o.y = p.Results.y;
            end
            
            % priors
            o.pr_bounds=p.Results.pr_bounds;
            if isnan(o.pr_bounds)
                if all(~isnan(o.theta))
                    o.pr_bounds = [(o.theta' - [1;0.5;0.5]) (o.theta'+[1;1;1])];
                else
                    o.pr_bounds = [1.61 3; 2 5; -3 -0.22];
                end
            end
                
            o.prior_mu = 0.5*sum(o.pr_bounds, 2);
            o.prior_std = 1/12*diff(o.pr_bounds, 1, 2);
            
            % fixed values
            o.np = 3;
            o.tnames = {'LOG(PHI)';'LOG(R)';'LOG(SIGMA)'};
            o.names = {'PHI';'R';'SIGMA'};  
        end
        
        %% variable transformation
        function theta = transform(~, theta, back)
            if back
                theta = exp(theta);
            else
                theta = log(theta);
            end
        end
        
        %% observation density - evaluate
        function lp = y_lpdf(o, y, x, theta)
            llambda = theta(:, o.LPHI) + log(x);
            % assuming 0^0 = 1 for Poisson distribution, so 0*log(0) = 0
            a = y*llambda;
            if y == 0 
                a(isnan(a)) = 0;
            end
            lp = -gammaln(y+1) - exp(llambda) + a; 
        end
        
        %% observation density - simulate
        function y = y_rnd(o, x, theta)
            lambda = exp(theta(:, o.LPHI) + log(x));
            y = poissrnd(lambda, size(x));
        end
        

        %% transition density - simulate
        function [x_reduced, x] = x_rnd(o, x_ini, theta, m, n, varargin)
            
            numPoints = length(m:n);
            [Nr, Nc] = size(x_ini); 
            
            % simulate states (population)
            x = zeros(Nr, Nc, numPoints);
            x(:, :, 1) = x_ini;
            
            logr = theta(:, o.LR); sig = exp(theta(:, o.LSIGMA));
            for i = 2:numPoints % i = i + 1
                x(:, :, i) = exp(logr + log(x(:, :, i-1)) - x(:, :, i-1) + sig.*randn(Nr, Nc)); %z(:, :, i-1)
            end
 
            x = x(:, :, 2:end); % remove x_ini
            x_reduced = x(:, :, end);
        end
        
        
        function x1 = initial_x_rnd(o, theta, varargin)
            if nargin == 3; N = varargin{:};
            else; N = 1; end
            
            X0 = repmat(o.x0, size(theta, 1), N);
            x1 = o.x_rnd(X0, theta, 0, 1); 
        end
        
        
        function [x_reduced, x, z] = simulate_trajectory(o, theta)
            z = normrnd(0, exp(theta(o.LSIGMA)), o.ntx, 1);
            
            % simulate states
            x = zeros(o.ntx, 1);
            x(1) = initial_x_rnd(o, theta, 1); 
            
            r = exp(theta(o.LR));
            
            for i = 2:o.ntx % i = i + 1
                x(i) = r*x(i-1)*exp(-x(i-1) + z(i-1));
            end

            x_reduced = x(o.ity);
        end
        %% prior
        
        function setPriorBounds(o, pr_bounds)
            o.pr_bounds = pr_bounds;
        end
        
        function f = prior_lpdf(o, theta)
            
            f = (1./diff(o.pr_bounds, 1, 2)');
            theta_transpose = theta';
            
            if size(theta, 1) > o.np
                N = size(theta, 1);
                lb = repmat(o.pr_bounds(:, 1), 1, N);
                ub = repmat(o.pr_bounds(:, 2), 1, N);
            else
                lb = o.pr_bounds(:, 1);
                ub = o.pr_bounds(:, 2);
            end
            
            f = f'.*(theta_transpose >= lb).*(theta_transpose <= ub);
            f = log(f');

        end

        function thetas = prior_rnd(o, N)
            thetas = zeros(N, o.np);
            thetas(:, o.LPHI) = unifrnd(...
                o.pr_bounds(o.LPHI, 1), o.pr_bounds(o.LPHI, 2), N, 1);
            thetas(:, o.LR) = unifrnd(...
                o.pr_bounds(o.LR, 1), o.pr_bounds(o.LR, 2), N, 1);
            thetas(:, o.LSIGMA) = unifrnd(...
                o.pr_bounds(o.LSIGMA, 1), o.pr_bounds(o.LSIGMA, 2), N, 1);
        end
        
    end
    
end