
rng(1)

len = 15;
a = rand(1,len);
a(4) = 3;
a(5) = 2.5;

x1 = 1:len;
x2 = linspace(1, len, 180);

b = interp1(x1, a, x2, 'spline');
b = b - min(b);
b = b / max(b);

figure(2)
clf
plot(b, 'linewidth', 2)
set(gca, 'linewidth', 2, 'fontsize', 20, 'xtick', 0:30:180, 'xticklabel', -90:30:90, 'ytick', [0, 1])
xlabel("Angles", 'fontsize', 30)
ylabel("Normalized Power", 'fontsize', 30)
axis([0, 180, 0, 1.2])
pbaspect([6,3.8,1])
saveas(gcf, 'figs/matlab_test.png')
print('-depsc', 'fig_results/non_diagonal_pks.pdf')
