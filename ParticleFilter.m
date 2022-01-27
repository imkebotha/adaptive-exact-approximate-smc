classdef ParticleFilter

    methods(Static)
        
        function [LL, X, logW] = standard(m, theta, Nx, T)
            if m.Dx == 1
                [LL, X, logW] = ParticleFilter.PF_batch(m, theta, Nx, T);
            else
                [LL, X, logW] = ParticleFilter.PF_batch_multivariate(m, theta, Nx, T);
            end 
        end % standard
        
        %% initialisation of the standard particle filter
        function [LL1, X1, logW1] = initialise(m, theta, Nx)
            X1 = m.initial_x_rnd(theta, Nx);
            logW1 = m.y_lpdf(m.y(1), X1, theta) - log(Nx); 
            LL1 = logsumexp(logW1, 2); 
        end % initialise
        
        %% single iteration of the standard particle filter
        function [LLt, Xt, logWt] = iteration(m, theta, Nx, t, Xt, logWt)
            logNWt = logWt - logsumexp(logWt, 2);
            
            % adaptive resampling
            deficientESS = exp(-logsumexp(2*logNWt, 2)) < Nx/2;
            if (any(deficientESS))
                for p = find(deficientESS')
                    I = randsample(1:Nx, Nx, true, exp(logNWt(p, :)));
                    if m.Dx == 1
                        Xt(p, :) = Xt(p, I);
                    else
                        for i = 1:m.Dx
                            Xt{i}(p, :) = Xt{i}(p, I);
                        end
                    end
                    logNWt(p, :) = -log(Nx);
                end
            end

            % simulate model and update weights
            Xt = m.x_rnd(Xt, theta, t-1, t);
            logWt = m.y_lpdf(m.y(t), Xt, theta) + logNWt;

            % incremental log-likelihood estimate
            LLt = logsumexp(logWt, 2);
       
        end % iteration

    end % methods
    
    methods(Static, Hidden)
        function [LL, Xt, logWt] = PF_batch(m, theta, Nx, tk)
            % initialise
            Nt = size(theta, 1);
            LL = zeros(Nt, tk);

            % at t = 1
            Xt = m.initial_x_rnd(theta, Nx); 
            logWt = m.y_lpdf(m.y(1), Xt, theta) - log(Nx); 
            logNWt = logWt - logsumexp(logWt, 2);
            LL(:, 1) = logsumexp(logWt, 2);

            for t = 2:tk % t = t + 1

                % adaptive resampling
                deficientESS = exp(-logsumexp(2*logNWt, 2)) < Nx/2;
                if (any(deficientESS))
                    for p = find(deficientESS')
                        Xt(p, :) = randsample(Xt(p, :), Nx, true, exp(logNWt(p, :)));
                        logNWt(p, :) = -log(Nx);
                    end
                end

                % simulate model and update weights
                Xt = m.x_rnd(Xt, theta, t-1, t); 
                logWt = m.y_lpdf(m.y(t), Xt, theta) + logNWt;
                logNWt = logWt - logsumexp(logWt, 2);
                
                % update likelihood estimate
                LL(:, t) = logsumexp(logWt, 2);
            end
            I = isnan(LL); LL(I) = -inf;
        end % PF_batch


        function [LL, Xt, logWt] = PF_batch_multivariate(m, theta, Nx, tk)
            % initialise
            Nt = size(theta, 1);
            LL = zeros(Nt, tk);
            
            % at t = 1
            Xt = m.initial_x_rnd(theta, Nx); 
            logWt = m.y_lpdf(m.y(1), Xt, theta) - log(Nx); 
            logNWt = logWt - logsumexp(logWt, 2);
            LL(:, 1) = logsumexp(logWt, 2);

            for t = 2:tk 

                % adaptive resampling
                deficientESS = exp(-logsumexp(2*logNWt, 2)) < Nx/2;
                if any(deficientESS)
                    for p = find(deficientESS')
                        I = randsample(1:Nx, Nx, true, exp(logNWt(p, :)));
                        for i = 1:m.Dx
                            Xt{i}(p, :) = Xt{i}(p, I);
                        end
                        logNWt(p, :) = -log(Nx);
                    end
                end

                % simulate model and update weights
                Xt = m.x_rnd(Xt, theta, t-1, t); 
                logWt = m.y_lpdf(m.y(t), Xt, theta) + logNWt;
                logNWt = logWt - logsumexp(logWt, 2);

                % update likelihood estimate
                LL(:, t) = logsumexp(logWt, 2);
            end
        end % PF_batch_multivariate

    end % methods
    
end % ParticleFilter
