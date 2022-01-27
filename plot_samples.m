function plot_samples(m, samples, varargin)

% parse input 
p = inputParser;
addParameter(p, 'TracePlots', false);
addParameter(p, 'Weights', ones(size(samples, 1), 1));
parse(p, varargin{:});  

W = p.Results.Weights./sum(p.Results.Weights);


% transform samples back
samples = m.transform(samples, true);


% create figure and subplots
subplots = cell(m.np,1);
figure
for i = 1:m.np
    subplots{i} = subplot(2, round(m.np/2), i);
    title(m.names(i), 'FontSize', 14)
end

% plot density
for i = 1:m.np 
    hold(subplots{i}, 'on')
    [y, x] = ksdensity(samples(:, i), 'Weights', W); 
    plot(subplots{i}, x, y, 'LineWidth', 1)
    x_min = round(min(x)-0.1, 1); 
    x_max = round(max(x)+0.1, 1);
    xlim(subplots{i}, [x_min x_max]);
    hold off
end

% plot prior
prior_samples = m.transform(m.prior_rnd(10000), true);
tv = m.transform(m.theta, true);
have_tv = all(~isnan(tv));

for i = 1:m.np % i = i + 1
    hold(subplots{i}, 'on')
    if have_tv; xline(subplots{i}, tv(i)); end
    ksdensity(subplots{i}, prior_samples(:, i));
    hold off
end

%% trace plots

if p.Results.TracePlots
    subplots = cell(m.np,1);
    figure
    for i = 1:m.np
        subplots{i} = subplot(2, round(m.np/2), i);
        plot(subplots{i}, samples(:, i));
        title(m.tnames(i), 'FontSize', 14)
    end
end
end