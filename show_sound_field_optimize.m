function field =  show_sound_field_optimize(speaker)

    grid_size = speaker.lambda / 20;
    x = 0:grid_size:100;
    y = -50:grid_size:50;
    field = zeros(length(x), length(y));
    for xi = 1 : length(x)
        for yi = 1 : length(y)
            for ni = 1 : speaker.num
                target = [x(xi), y(yi), 0];
                d = compute_distance(target, speaker.locs(ni,:));
                field(xi, yi) = field(xi, yi) + exp(-j*2*pi*speaker.fc / speaker.c * d) * speaker.weights_out(ni);
            end
        end
    end
    
    % figure(1)
    % clf
    % imagesc(abs(field))
    % pbaspect([1,1,1])
    % saveas(gcf, 'figs/matlab_test.png')
end