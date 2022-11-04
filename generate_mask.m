c = 34300;
fc = 20e3;
lambda = c / fc;
d = lambda / 2;
N = 16;

theta_0 = 0.886*lambda / N / d / pi * 180;
ang = -60:1:60;
theta = 20 ./ (cos(ang / 180 * pi));
% theta = round(theta);
mask = ones(length(ang), length(ang));

figure(1)
clf
plot(theta, 'linewidth', 3)
axis([0, length(mask), 0, 30])
saveas(gcf, 'figs/matlab_test.png')

for ti = 1 : length(ang)
    width = floor(theta(ti) / 2);
    % width = 20;
    lower = ti - width;
    upper = ti + width;
    if(lower < 1) lower = 1; end
    if(upper > length(ang)) upper = length(ang); end
    mask(ti, lower:upper) = 0;
end

figure(1)
clf
image(mask * 255)
pbaspect([1,1,1])
% axis([0, length(mask), 0, 30])
saveas(gcf, 'figs/matlab_test.png')

save mask.mat mask
