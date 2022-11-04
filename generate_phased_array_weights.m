addpath('./functions/')
clear
% load optimal.mat
% load parameters.mat
load optimal_non_uniform_2.mat
load parameters_non_uniform_2.mat

speaker_weights = ones(181, 6);

fc = 20e3;
c = 34300;

for ti = 0 : 180
    for mi = 1 : 5
        d = abs(speaker_locs(mi+1) - speaker_locs(1));
        deq = d * cos(ti / 180 * pi);
        speaker_weights(ti+1, mi+1) = exp(1j*2*pi*fc/c*deq);
    end
end

figure(1)
clf
subplot(221)
imagesc(angle(speaker_weights))
pbaspect([1,1,1])

subplot(222)
imagesc(angle(speaker_w .* conj(speaker_w(:, 1))))
pbaspect([1,1,1])



beams = zeros(181, 181);

for ti = 1 : 181
    for ni = 1 : 181
        for mi = 1 : 6
            d = abs(speaker_locs(mi) - speaker_locs(1));
            deq = d * cos(ni / 180 * pi);
            beams(ti, ni) = beams(ti, ni) + speaker_weights(ti, mi) * exp(-1j*2*pi*fc/c*deq);
        end
    end
end

beams2 = zeros(121, 121);

for ti = 30 : 150
    for ni = 30 : 150
        for mi = 1 : 6
            d = abs(speaker_locs(mi) - speaker_locs(1));
            deq = d * cos(ni / 180 * pi);
            beams2(ti-29, ni-29) = beams2(ti-29, ni-29) + speaker_w(ti-29, mi) * exp(-1j*2*pi*fc/c*deq);
        end
    end
end

subplot(223)
imagesc(abs(beams))
pbaspect([1,1,1])

subplot(224)
imagesc(abs(beams2))
pbaspect([1,1,1])

saveas(gcf, 'figs/matlab_test.png')

save non_uniform_phased_array.mat speaker_weights speaker_locs