function logy = loggmmpdf(gmm, x)
    % returns the log pdf of the Gaussian mixture model gmm, evaluated at x
%     gmm = s.GMModel; % remove
%     x = s.tsamples;
    
    props = gmm.ComponentProportion;
    means = gmm.mu;
    covars = gmm.Sigma;
    K = gmm.NumComponents;
    
    % preallocate
    logy = zeros(size(x, 1), K);
    
    for k = 1:K
        logy(:, k) = log(props(k)) + logmvnpdf(x, means(k, :), covars(:, :, k));
    end
    logy = logsumexp(logy, 2);
    %logy = logy - logsumexp(logy);
    
    %logpz = gammaln(factorial(size(x, 1)) + 1)
    
    
    
    
%     mnpdf

end