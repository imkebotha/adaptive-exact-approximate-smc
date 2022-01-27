classdef SV_model < handle & abstractSSM
    
    properties
        LXI = 1; LOMEGA2 = 2; LLAMBDA = 3; BETA = 4; MU = 5;
        V = 1; Z = 2;
    end
    
    methods
        %% Constructor
        function o = SV_model(ty, varargin)
            
            % parse optional input
            p = inputParser;
            addParameter(p, 'theta', nan, @(x) length(x) == 5 || isnan(x));
            addParameter(p, 'y', nan); 
            addParameter(p, 'x', nan);
            addParameter(p, 'dt', 1);
            parse(p, varargin{:});
            
            genTimePoints(o, ty, p.Results.dt); 
            o.Dx = 2;
            o.theta = p.Results.theta;
            
            % if no data is supplied, simulate
            if isnan(p.Results.y) 
                if any(isnan(o.theta))
                    % theta = {log(xi), log(omega^2), log(lambda), beta, mu}
                     o.theta = [log(0.5) log(0.0625) log(0.01) 0 0]; 
                end
           
                o.x = o.simulate_trajectory(o.theta);
                o.y = y_rnd(o, o.x, o.theta); 
            else
                o.x = p.Results.x;
                o.y = p.Results.y;
            end
            
            % fixed values
            o.np = 5;
            o.tnames = {'LOG(XI)';'LOG(OMEGA2)';'LOG(LAMBDA)';'BETA';'MU'};
            o.names = {'XI';'OMEGA2';'LAMBDA';'BETA';'MU'};
        end
        
        %% variable transformation
        function theta = transform(o, theta, back)
            cnum = size(theta, 1);
            if cnum == o.np
                if back
                    theta([o.LXI o.LOMEGA2 o.LLAMBDA], :) = exp(theta([o.LXI o.LOMEGA2 o.LLAMBDA], :));
                else
                    theta([o.LXI o.LOMEGA2 o.LLAMBDA], :) = log(theta([o.LXI o.LOMEGA2 o.LLAMBDA], :));
                end
            else
                if back
                    theta(:, [o.LXI o.LOMEGA2 o.LLAMBDA]) = exp(theta(:, [o.LXI o.LOMEGA2 o.LLAMBDA]));
                else
                    theta(:, [o.LXI o.LOMEGA2 o.LLAMBDA]) = log(theta(:, [o.LXI o.LOMEGA2 o.LLAMBDA]));
                end
            end
        end
        
        %% observation density 
        function lp = y_lpdf(o, y, x, theta)
            v = x{o.V};
            mu = theta(:, o.MU) + theta(:, o.BETA).*v;
            lp = norm_lpdf(y, mu, sqrt(v) + 10^(-5));
        end
        
        function y = y_rnd(o, x, theta)
            v = x{o.V};
            y = theta(:, o.MU) + theta(:, o.BETA).*v + (sqrt(v)).*randn(size(v));
        end
        
        %% transition density
        
        % simulate x_ini forward from time m to time n
        function x = x_rnd(o, x_ini, theta, m, n, varargin)
            z_ini = x_ini{o.Z};
            [Nt, Nx] = size(z_ini);

            % extract all time points between m and n
            if m == 0
                slice_range = 1:o.ity(n);
            else
                slice_range = o.ity(m):o.ity(n);
            end
            t_slice = o.tx(slice_range);
            npoints = length(t_slice);

            % get untransformed parameters
            lambda = exp(theta(:, o.LLAMBDA));
            
            % k ~ Poisson(lambda * xi^2 / omega^2)
            k_parameter = exp(theta(:, o.LLAMBDA) ...
                    + 2*theta(:, o.LXI) - theta(:, o.LOMEGA2));
                
            % e_{1:k} ~ Exp(xi / omega^2) (rate, mu = omega^2/xi)
            e_parameter = exp(theta(:, o.LOMEGA2) - theta(:, o.LXI)); 

            for t = 2:npoints
             
                K = poissrnd(repmat(k_parameter, 1, Nx));
                K(K > 10^4) = 10^4; % truncated poisson
                
                % values if k = 0
                z = exp(-lambda).*z_ini; 
                v = (z_ini - z)./lambda;

                for i = 1:Nt
                    C = unifrnd(t_slice(t-1), t_slice(t), sum(K(i, :), 'all'), 1); indc = 1; % U(t-1, t)
                    E = exprnd(e_parameter(i), sum(K(i, :)), 1); inde = 1; %w_sq(i)./xi(i)
                    for n = 1:Nx
                        k = K(i, n);

                        if k > 0
                            c = C(indc:indc + k - 1); indc = indc + k;
                            e = E(inde:inde + k - 1); inde = inde + k;
                            
                            % z_t = exp(-lambda) * z_t-1 + sum_j=1^k exp(-lambda * (t - c_j)) * e_j
                            z(i, n) = exp(-lambda(i)).*z_ini(i, n) + sum(exp(-lambda(i)*(t_slice(t)-c) + log(e))); 
                            
                            % v_t = (z_t-1 - z_t + sum_j=1^k e_j)/lambda
                            v(i, n) = (z_ini(i, n) - z(i, n) + sum(e))./lambda(i); 
                        end
                        
                    end % Nx
                end % Nt
                z_ini = z;
                
            end % npoints

            x = {v, z}; 
        end
        

        function [x1, z0] = initial_x_rnd(o, theta, varargin)
            % z_0 ~ Gamma(xi^2 / omega^2, xi / omega^2) (shape, rate)
            % scale = omega^2/xi
            A = exp(2*theta(:, o.LXI) - theta(:, o.LOMEGA2));
            B = exp(theta(:, o.LOMEGA2) - theta(:, o.LXI));
            
            if nargin == 3
                Nx = varargin{:};
                A = repmat(A, 1, Nx);
                B = repmat(B, 1, Nx);
            end
            
            z0 = gamrnd(A, B); 
            x1 = o.x_rnd({0, z0}, theta, 0, 1);
        end
        
        function x = simulate_trajectory(o, theta)
            x = cell(o.nty, 1);
            x{1} = initial_x_rnd(o, theta); 
      
            for i = 2:o.nty % i = i + 1
                x{i} = x_rnd(o, x{i-1}, theta, i-1, i);
            end
            x = colCellMatCat(x); 
        end
        

        %% prior - theta = {log(xi), log(omega^2), log(lambda), beta, mu}
        
        function f = prior_lpdf(o, theta)     
            f = [lexp_lpdf(theta(:, o.LXI), 0.2) ...
                lexp_lpdf(theta(:, o.LOMEGA2), 0.2) ...
                lexp_lpdf(theta(:, o.LLAMBDA), 1) ...
                norm_lpdf(theta(:, o.BETA), 0, sqrt(2)) ...
                norm_lpdf(theta(:, o.MU), 0, sqrt(2))];
        end

        function thetas = prior_rnd(o, Nt)
            thetas = zeros(Nt, o.np);
            thetas(:, o.LXI) = log(exprnd(5, Nt, 1)); % rate = 0.2
            thetas(:, o.LOMEGA2) = log(exprnd(5, Nt, 1)); % rate = 0.2
            thetas(:, o.LLAMBDA) = log(exprnd(1, Nt, 1)); % rate = 1
            thetas(:, o.BETA) = normrnd(0, sqrt(2), Nt, 1);
            thetas(:, o.MU) = normrnd(0, sqrt(2), Nt, 1);          
        end
    end
    
end

%% Helper functions
% log pdf of the normal distribution
function y = norm_lpdf(x, m, s)
    y = -0.5*log(2*pi) - log(s) - 0.5*(x - m).^2./exp(2*log(s)); 
end

% log pdf of the log-exponential distribution
function y = lexp_lpdf(log_x, rate)
    y = log(rate) - rate*exp(log_x) + log_x;
end
