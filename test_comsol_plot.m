data = load('comsol_data/test.txt');

aa = data(:,4);
im = reshape(aa, 41, []);
im = flipud(abs(im)');
im(40:41,:) = [];


im = interp2(im, 10, 'spline');
thre = prctile(im(:), 99);
im(im > thre) = thre;

figure(1)
clf

imagesc(abs(im))    
pbaspect([1,1,1])
saveas(gca, 'figs/matlab_test.png')