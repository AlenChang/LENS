function im = process_comsol_plot_data(data, target_direction)

x = data(:,1);
y = data(:,2);
z = data(:,3);

new_data = data(:, 4:end);
% save_name = strcat('fig_results/', string(target_direction), target_plane,'.eps');


ang = 0:5:90;
len = length(x);
im = reshape(new_data(:, ang == target_direction), sqrt(len), []);

im = flipud(im.');
im = abs(im);
im = interp2(im, 5, 'linear');


end
% center = [100, 200];


% for ti = 1:size(im, 1)
%     for ni = 1:size(im, 2)
%         d = compute_distance(center, [ti, ni]) / 1000;
%         im(ti, ni) = im(ti, ni) * d;
%     end
% end

% thre = prctile(im(:), 99.5);

% thre = 1e5;
% im(im > thre) = thre;

% compare_phase_array_lens