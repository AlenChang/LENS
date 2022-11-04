
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

    if(target_angle(ti) == 30)
        ams_beams = circshift(ams_beams, 20);
    end

    wpa_wlens_data = ams_beams * cos((target_angle(ti) - 90) / 180 * pi) .^ (1/5);
    wpa_wlens_ang = elevation_theta;

    wpa_wolens_data = speaker_beams * mwol_coeff * sqrt(cos((target_angle(ti) - 90) / 180 * pi));
    wpa_wolens_ang = elevation_theta;

    wopa_wlens_data = swl * swl_coeff * 0.9;
    wopa_wlens_ang = swl_ang + 90;

    wopa_wolens_data = swol * swol_coeff * 0.9;
    wopa_wolens_ang = swol_ang + 90;

    eval(sprintf("beams_data.ang_%i.wpa_wlens.data = wpa_wlens_data;", 180 - target_angle(ti)))
    eval(sprintf("beams_data.ang_%i.wpa_wlens.ang = wpa_wlens_ang;", 180 - target_angle(ti)))

    eval(sprintf("beams_data.ang_%i.wpa_wolens.data = wpa_wolens_data;", 180 - target_angle(ti)))
    eval(sprintf("beams_data.ang_%i.wpa_wolens.ang = wpa_wolens_ang;", 180 - target_angle(ti)))

    eval(sprintf("beams_data.ang_%i.wopa_wlens.data = wopa_wlens_data;", 180 - target_angle(ti)))
    eval(sprintf("beams_data.ang_%i.wopa_wlens.ang = wopa_wlens_ang;", 180 - target_angle(ti)))

    eval(sprintf("beams_data.ang_%i.wopa_wolens.data = wopa_wolens_data;", 180 - target_angle(ti)))
    eval(sprintf("beams_data.ang_%i.wopa_wolens.ang = wopa_wolens_ang;", 180 - target_angle(ti)))

    plot(wpa_wlens_ang, wpa_wlens_data, 'linewidth', 4)
    plot(wpa_wolens_ang, wpa_wolens_data, 'linewidth', 4)
    plot(wopa_wlens_ang, wopa_wlens_data, 'linewidth', 4)
    plot(wopa_wolens_ang, wopa_wolens_data, 'linewidth', 4)
    pbaspect([4,2.8,1])
    axis([0, 180, 0, 1.2])
    set(gca, 'fontsize', 20, 'box','on','linewidth',2, 'xtick', 0:30:180, 'xticklabel', -90:30:90)
    xlabel('Angle (Azimuth, \circ)', 'fontsize', 30)
    ylabel('Magnitude (Normalized)', 'fontsize', 30)
    legend('w/ PA w/ lens', 'w/ PA w/o lens', 'w/o PA w/ lens', 'w/o PA w/o lens', 'location', 'north', 'numcolumns', 2, 'fontsize', 20, 'box', 'off')
    saveas(gcf, 'figs/matlab_test.png')
    print('-depsc', sprintf('fig_results/cmp_mag_%i.eps', target_angle(ti)))

    pause



end