classdef ThetaLogistic_model < handle & abstractSSM 
    properties
        % constants
        LX0 = 1; BETA0 = 2; BETA1 = 3; BETA2 = 4; 
        LGAMMA = 5; LSIGMA = 6; A = 7;
    end
    
    methods
        %% constructor
        function o = ThetaLogistic_model(ty, varargin)
            
            % parse optional input
            p = inputParser;
            addParameter(p, 'theta', nan, @(x) length(x) == 7 || isnan(x));
            addParameter(p, 'y', nan); 
            addParameter(p, 'x', nan);
            parse(p, varargin{:});
            
            genTimePoints(o, ty, 1); 
            
            o.theta = p.Results.theta;
            if isnan(p.Results.y) 
                if any(isnan(o.theta))
                    o.theta = [6.1 0 1 -0.5 -2.3 -4 1];
                end 
                o.x = o.simulate_trajectory(o.theta);
                o.y = y_rnd(o, o.x, o.theta); 
            else
                o.x = p.Results.x;
                o.y = p.Results.y;
            end
                        
            % fixed values
            o.np = 7;
            o.tnames = {'LOG(X0)';'BETA0';'BETA1';'BETA2';'LOG(GAMMA)';'LOG(SIGMA)'; 'A'};
            o.names = {'LOG(X0)';'BETA0';'BETA1';'BETA2';'GAMMA';'SIGMA'; 'A'};
        end
        
        %% variable transformation
        function theta = transform(o, theta, back)
            params = [o.LGAMMA o.LSIGMA];
            if back; theta(:, params) = exp(theta(:, params));
            else; theta(:, params) = log(theta(:, params));
            end
        end
        
        %% observation density - evaluate
        function s = y_lpdf(o, y, logx, theta)  
            sig = exp(theta(:, o.LSIGMA));
			a = theta(:, o.A);
            s = norm_lpdf(y, a.*logx, sig);
        end
        
        %% observation density - simulate
        function y = y_rnd(o, logx, theta) 
            sig = exp(theta(:, o.LSIGMA));
			a = theta(:, o.A);
            y = a*logx + sig*randn(size(logx));
        end
        
        %% transition density - evaluate
        function f = x_lpdf(o, logx_current, logx_prev, theta)
           mu = logx_prev + theta(:, o.BETA0) + theta(:, o.BETA1).*exp(theta(:, o.BETA2).*logx_prev);
           gam = exp(theta(:, o.LGAMMA));
           f =  norm_lpdf(logx_current, mu, gam);
        end
        
        %% transition density - simulate
        
        function logx1 = initial_x_rnd(o, theta, varargin)
            logx0 = theta(:, o.LX0); 
            if nargin == 3
                N = varargin{:};
                logx0 = repmat(logx0, 1, N);
            end
            logx1 = o.x_rnd(logx0, theta, 0, 1); 
        end
        
        % simulate full trajectory - output will include x0
        function [x_reduced, x] = simulate_trajectory(o, theta)
            
            b0 = theta(:, o.BETA0); 
            b1 = theta(:, o.BETA1);
            b2 = theta(:, o.BETA2);
            g = exp(theta(:, o.LGAMMA));
            
            % simulate states
            x = zeros(o.ntx, 1);
            x(1) = initial_x_rnd(o, theta, 1); 
            for i = 2:o.ntx
                x(i) = x(i-1) + b0 + b1.*exp(b2.*x(i-1)) + g.*randn(); 
            end
            x_reduced = x(o.ity);
        end
        
        % simulate from observation time m to n starting with x_ini 
        function [logx_reduced, logx] = x_rnd(o, logx_ini, theta, m, n, varargin)
            
            numPoints = length(m:n);
            [Nr, Nc] = size(logx_ini); 

            b0 = theta(:, o.BETA0); 
            b1 = theta(:, o.BETA1);
            b2 = theta(:, o.BETA2);
            g = exp(theta(:, o.LGAMMA));
            
            logx = zeros(Nr, Nc, numPoints);
            logx(:, :, 1) = logx_ini;
            for i = 2:numPoints % i = i + 1
                logx(:, :, i) = logx(:, :, i-1) + b0 + b1.*exp(b2.*logx(:, :, i-1)) + g.*randn(Nr, Nc); 
            end

            logx = logx(:, :, 2:end); % remove x0
            logx_reduced = logx(:, :, end);
        end
        
        %% prior
        
        function logf = prior_lpdf(o, theta)
            logf = [logHN_lpdf(theta(:, o.LX0), 1000) ...
                 norm_lpdf(theta(:, o.BETA0), 0, 1) ...
                 norm_lpdf(theta(:, o.BETA1), 0, 1) ...
                 norm_lpdf(theta(:, o.BETA2), 0, 1) ...
                 logExp_lpdf(theta(:, o.LGAMMA), 1) ...
                 logExp_lpdf(theta(:, o.LSIGMA), 1) ...
                 norm_lpdf(theta(:, o.A), 1, 0.5)];
        end

        function draws = prior_rnd(o, N)
            draws = zeros(N, o.np);
            draws(:, o.LX0) = log(abs(normrnd(0, 1000, N, 1)));
            draws(:, [o.BETA0 o.BETA1 o.BETA2]) = normrnd(0,1, N, 3);
            draws(:, [o.LGAMMA o.LSIGMA]) = log(exprnd(1, N, 2));
            draws(:, o.A) = normrnd(1, 0.5, N, 1);
        end

        
    end
    
end

%% Helper functions
% log pdf of the normal distribution
function y = norm_lpdf(x, m, s)
    y = -0.5*log(2*pi) - log(s) - 0.5*(x - m).^2./exp(2*log(s)); 
end

% log pdf of the log-transformed half-normal distribution
function y = logHN_lpdf(logx, s)
    y = 0.5*log(2/pi)-log(s)-exp(2*logx)/(2*s^2) + logx;
end

% log pdf of the log-transformed exponential distribution
function y = logExp_lpdf(logx, g)
    y = log(g) - g*exp(logx) + logx;
end
