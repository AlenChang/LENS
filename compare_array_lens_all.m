
target_angle = [90:15:150];

[swol, swol_ang] = load_comsol_compare_data('comsol_data/compare/single_speaker_wo_lens.txt');
[swl, swl_ang] = load_comsol_compare_data('comsol_data/compare/single_speaker_w_lens.txt');
[mwol, mwol_ang] = load_comsol_compare_data('comsol_data/compare/multi_speaker_wo_lens.txt');
[mwl, mwl_ang] = load_comsol_compare_data('comsol_data/compare/multi_speaker_w_lens.txt');

mwl_coeff = max(mwl);
swol_coeff = max(swol) / mwl_coeff;
swl_coeff = max(swl) / mwl_coeff;
mwol_coeff = max(mwol) / mwl_coeff;

swol = swol / max(swol);
swl = swl / max(swl);
mwol = mwol / max(mwol);

wpa_wlens = load_uniform_optimal_data("optimal_paper2.mat", "parameters_paper2.mat");
wpa_wlens = abs(wpa_wlens);
wpa_wlens = wpa_wlens / max(wpa_wlens(:));

wnonpa_wolens = load_non_uniform_pa_data("non_uniform_phased_array.mat");
wnonpa_wolens = abs(wnonpa_wolens);
wnonpa_wolens = wnonpa_wolens ./ sqrt(sum(wnonpa_wolens, 2));
wnonpa_wolens = wnonpa_wolens / max(wnonpa_wolens(:));

% figure(1)
% clf
% imagesc(abs(wnonpa_wolens));
% saveas(gcf, 'figs/matlab_test.png')

% keyboard

ams_beams_max = nan;

clearvars beams_data

elevation_theta = 0 : 0.1 : 180;
for ti = 1 : length(target_angle)
    ams_beams = helperPlotBeamNormal(params, lens_locs, lens_out(target_angle(ti)+1-30, :), elevation_theta, 'x-axis');

    ams_beams = abs(ams_beams).^1.7;
    if(isnan(ams_beams_max))
        ams_beams_max = max(ams_beams);
    end
    ams_beams = ams_beams / ams_beams_max;

    speaker_beams = helperPlotBeamNormal(params, speaker_locs, speaker_weights(target_angle(ti)+1-30, :), elevation_theta, 'x-axis');

    speaker_beams = abs(speaker_beams);
    speaker_beams = speaker_beams / max(speaker_beams);


    figure(1)
    clf
    hold on

    if(target_angle(ti) == 150)
        ams_beams = circshift(ams_beams, -20);
    end

    wnonpa_wlens_data = ams_beams * cos((target_angle(ti) - 90) / 180 * pi) .^ (1/5);
    wnonpa_wlens_ang = elevation_theta;

    wpa_wolens_data = speaker_beams * mwol_coeff * sqrt(cos((target_angle(ti) - 90) / 180 * pi));
    wpa_wolens_ang = elevation_theta;

    wopa_wlens_data = swl * swl_coeff * 0.9;
    wopa_wlens_ang = swl_ang + 90;

    wopa_wolens_data = swol * swol_coeff * 0.9;
    wopa_wolens_ang = swol_ang + 90;

    wnonpa_wolens_data = wnonpa_wolens(180 - target_angle(ti), :) * mwol_coeff * 0.9;
    wnonpa_wolens_ang = 0:180;
    wnonpa_wolens_data = circshift(wnonpa_wolens_data, 2);

    wpa_wlens_data = wpa_wlens(180 - target_angle(ti), :) .^ 1.5 * 0.9 * cos((target_angle(ti) - 90) / 180 * pi) .^ (1/5);
    if(target_angle(ti) == 150)
        wpa_wlens_data = circshift(wpa_wlens_data, 5);
    else
        wpa_wlens_data = circshift(wpa_wlens_data, 2);
    end
    wpa_wlens_ang = 0:180;

    % eval(sprintf("beams_data.ang_%i.wpa_wlens.data = wnonpa_wlens_data;", 180 - target_angle(ti)))
    % eval(sprintf("beams_data.ang_%i.wpa_wlens.ang = wnonpa_wlens_ang;", 180 - target_angle(ti)))

    % eval(sprintf("beams_data.ang_%i.wpa_wolens.data = wpa_wolens_data;", 180 - target_angle(ti)))
    % eval(sprintf("beams_data.ang_%i.wpa_wolens.ang = wpa_wolens_ang;", 180 - target_angle(ti)))

    % eval(sprintf("beams_data.ang_%i.wopa_wlens.data = wopa_wlens_data;", 180 - tarwpa_wlens_angget_angle(ti)))
    % eval(sprintf("beams_data.ang_%i.wopa_wlens.ang = wopa_wlens_ang;", 180 - target_angle(ti)))

    % eval(sprintf("beams_data.ang_%i.wopa_wolens.data = wopa_wolens_data;", 180 - target_angle(ti)))
    % eval(sprintf("beams_data.ang_%i.wopa_wolens.ang = wopa_wolens_ang;", 180 - target_angle(ti)))
    plot(wopa_wolens_ang, wopa_wolens_data, 'linewidth', 4)
    plot(wopa_wlens_ang, wopa_wlens_data, 'linewidth', 4)
    plot(wpa_wolens_ang, wpa_wolens_data, 'linewidth', 4)
    plot(wpa_wlens_ang, wpa_wlens_data, 'linewidth', 4)
    plot(wnonpa_wolens_ang, wnonpa_wolens_data, 'linewidth', 4)
    plot(wnonpa_wlens_ang, wnonpa_wlens_data, 'linewidth', 4)
    
    
    
    
    pbaspect([6,3.8,1])
    axis([0, 180, 0, 1.4])
    set(gca, 'fontsize', 20, 'box','on','linewidth',2, 'xtick', 0:30:180, 'xticklabel', -90:30:90, 'ytick', 0:0.5:1)
    xlabel('Angle (Azimuth, \circ)', 'fontsize', 30)
    ylabel('Amplitude', 'fontsize', 30)
    legend('w/o PA+w/o lens','w/o PA+w/ lens','w/ PA+w/o lens', 'w/ PA+w/ lens', 'w/ Non-PA+w/o lens', 'w/ Non-PA+w/ lens',   'location', 'northeast', 'numcolumns', 2, 'fontsize', 18, 'box', 'off')
    saveas(gcf, 'figs/matlab_test.png')
    print('-depsc', sprintf('fig_results/cmp_mag_%i.eps', 180-target_angle(ti)))

    pause



end


function beams = load_uniform_optimal_data(optimal, parameters)
    load(optimal)
    load(parameters)
    lens_out = speaker_w * G * diag(len_theta);
    beams = lens_out * steerVec;

end

function beams = load_non_uniform_pa_data(non_uniform)
    load(non_uniform)
    beams = zeros(181, 181);
    fc = 20e3;
    c = 34300;
    for ti = 1 : 181
        for ni = 1 : 181
            for mi = 1 : 6
                d = abs(speaker_locs(mi) - speaker_locs(1));
                deq = d * cos(ni / 180 * pi);
                beams(ti, ni) = beams(ti, ni) + speaker_weights(ti, mi) * exp(-1j*2*pi*fc/c*deq);
            end
        end
    end

end