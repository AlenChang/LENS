%% ======================================
%% Yi-Chao Chen@MSRA
%% ======================================
%function validate_bela7()
    clear 
    clc
    addpath('./functions');

    fontSize = 28;
    colors   = {[228 26 28]/255, [55 126 184]/255, [77,175,74]/255, [152,78,163]/255, [255,127,0]/255, [255,255,51]/255, [166,86,40]/255, [247,129,191]/255, [153,153,153]/255};
    lines    = {'-', '--', '-.', ':'};
    markers  = {'+', 'o', '*', '.', 'x', 's', 'd', '^', '>', '<', 'p', 'h'};


    %% ============================
    %% Configuration
    %% ============================
    %% Tasks
    PLOT_TIME_DOMAIN  = 0;
    PLOT_SPECTRUM     = 1;
    PLOT_MIX_FFT      = 1;
    HIGHPASS_FLTR     = 1;
    cal_error         = 0;
    METHOD_CANCEL     = 'NO_CANCEL';  %% 'NO_CANCEL' 'NO_AGC', 'REGRESSION', 'MEDIAN'
    
    %% File
    filename = 'data/exp7-repeatTunetable-spkNotRotate.txt';
    filename = 'E:\test-3.txt';
    filename = 'E:\tunetable-estimation-lens.txt';
    filename = 'E:\tunetable-estimation-nolens (1).txt';
    filename = 'E:\tunetable-estimation-nolens2.txt';
    filename = 'E:\tunetable-estimation-lens-10-80cm.txt';
    filename = 'E:\tunetable-estimation-lens-10-160cm.txt';    
    filename = 'E:/tunetable-estimation-lens-suddenlychange.txt';
    
    filename = 'E:/tests_suddenly.txt'; % lens, 40-70cm repeat 3 times,interval:10cm
    %filename = 'E:/tests_suddenly-nolens.txt'; % no lens, 20-120cm repeat 3 times, interval:20cm
    %filename = 'E:/tests_suddenly-withlens.txt'; % no lens, 30-40-50-70-10-110-130-150-170cm repeat 3 times
    
    %filename = 'E:\tunetable-estimation-nolens2.txt';
    
    % 0712
    filename = 'E:\both_with_withoutlens.txt';
    filename = 'E:\both_with_withoutlens-1.txt';
    
    filename = 'E:\0712_with_lens.txt';
    filename = 'E:\0712_without_lens.txt';
    
    filename = 'E:\0712_with_lens-fix.txt';
    filename = 'E:\0712_without_lens-fix.txt';
    
    % 0713-TurnTable
    filename = 'E:\0713_turntable-bg.txt';
   
    filename = 'E:\0713_formal1.txt';
    
    filename = './real_trace/tests_suddenly.txt';
    
    
    %% Ground Truth
    distance_lists = containers.Map;
    %distance_lists('E:\tunetable-estimation-nolens2.txt') = [0,20,30,40,50,60];
    distance_lists('E:/tests_suddenly.txt') = [0,40,50,60,70,80,90,80,70,60,50,40,50,60,70,80,90,0];
    distance_lists('E:\both_with_withoutlens.txt') = [0,30,30,40,40,50,50,80,80,110,110,140,140,170,170,140,140,110,110,80,80,50,50,40,40,30,30,0];
    distance_lists('E:\both_with_withoutlens-1.txt') = [0,30,30,40,40,50,50,80,80,110,110,140,140,170,170,0];
    distance_lists('E:\0712_with_lens.txt') = [0,30,40,60,80,100,120,140,160,140,120,100,80,60,50,40,30,0];
    distance_lists('E:\0712_without_lens.txt') = [0,30,40,60,80,100,120,140,160,140,120,100,80,60,50,40,30,0];
    distance_lists('E:\0712_with_lens-fix.txt') = [0,30,40,50,60,70,80,100,120,140,160,0];
    distance_lists('E:\0712_without_lens-fix.txt') = [0,30,40,50,60,70,80,100,120,140,160,0];
    distance_lists('E:\0713_turntable-bg.txt') = [0,0,0,0,0,0,0,0,0,0,0];
    
    distance_duration_time = 10; % duration time of each distance: 10s
    
    %% FMCW
    vc         = 343;
    Fs         = 44100;
    chirpLenS  = 0.04;
    chirpLen   = chirpLenS * Fs;
    Bw         = 4000;
    fmin       = 16000;
    t          = [0:chirpLen-1]/Fs;
    t0         = exp(1j*2*pi*(fmin*t+1/2*Bw*t.^2/chirpLenS));
    syncOffset = 200;

    %% Cancellation
    cancelTimeS  = 11; %% use the first few seconds to learn the reference signal for cancellation
    cancelTime   = cancelTimeS * Fs;
    nCancelChirp = floor(cancelTime / chirpLen);
    cancelTime   = nCancelChirp * chirpLen;
    cancelTimeS  = cancelTime / Fs;
    startTimeS   = 11;
    startTime    = startTimeS * Fs;


    %% ============================
    %% Load Trace
    %% ============================
    fprintf('- Load Trace\n');

    data = load(filename);
        
    fprintf('  size = %dx%d (%.1fs)\n', size(data), size(data, 1)/Fs);

    %% skip the first-second data
    data = data(1*Fs+1:end, :);
    dataLen = size(data, 1);
    nCh     = size(data, 2);

    %% Plot to timeseries
    if PLOT_TIME_DOMAIN
        figure(1); clf; 
        plot([0:dataLen-1]/Fs, data(:,1), '-b');
        xlabel('Time (s)');
        ylabel('Volume');
        set(gca, 'FontSize', fontSize);
        return;
    end


    %% ============================
    %% High-pass Filter
    %% ============================
    if HIGHPASS_FLTR
        fprintf('- High-pass Filter\n');
        for ci = 1:nCh
            data(:, ci) = filterFFT(data(:, ci).', Fs, fmin, fmin+Bw, 100).';
        end
    end


    %% ============================
    %% Plot Spectrum
    %% ============================
    if PLOT_SPECTRUM
        fprintf('- Plot Spectrum\n');

        win      = floor(Fs/512);
        noverlap = floor(win/4); % 75% overlap
        Nfft     = Fs;
        %% P matrix contains the power spectral density of each segment of the STFT
        % [S,F,T,P] = spectrogram(data(1:1*Fs, 1), win, noverlap, Nfft, Fs);
        
        fh = figure(11); clf;
        subplot(2,1,1); 
        [S,F,T,P] = spectrogram(data(1:1*Fs, 1), win, noverlap, Nfft, Fs);
        imagesc(T, F, 10*log10(P));
        colorbar;
        ax = gca; ax.YAxis.Exponent = 0;
        set(gca,'YDir','normal') 
        % ylim([17500 18500]);
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        title('Time-Frequency plot of a Audio signal (dB/Hz)');
        set(gca, 'FontSize', fontSize);

        subplot(2,1,2); 
        [S,F,T,P] = spectrogram(data(15*Fs:16*Fs, 1), win, noverlap, Nfft, Fs);
        imagesc(T, F, 10*log10(P));
        colorbar;
        ax = gca; ax.YAxis.Exponent = 0;
        set(gca,'YDir','normal') 
        % ylim([17500 18500]);
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        title('Time-Frequency plot of a Audio signal (dB/Hz)');
        set(gca, 'FontSize', fontSize);
        pause
    end


    %% ============================
    %% Cancellation
    %% ============================
    fprintf('- Calcellation\n');
    %% Learn Reference Signal
    ref = zeros(chirpLen, nCh);
    for ci = 1:nCh
        seq = data(1:cancelTime, ci);
        seq = reshape(seq, chirpLen, []);
        ref(:, ci) = median(seq, 2);
    end

    nChirps = floor(size(data, 1) / chirpLen);
    data2 = zeros(nChirps*chirpLen, nCh);
    if strcmpi(METHOD_CANCEL, 'REGRESSION') | strcmpi(METHOD_CANCEL, 'MEDIAN')
        fprintf('- Calcellation with Automated Gain Control\n');

        for ci = 1:nCh
            for si = 1:nChirps
                idx = (si-1)*chirpLen+1:si*chirpLen;
                seq = data(idx, ci);
                if strcmpi(METHOD_CANCEL, 'REGRESSION')
                    r = abs(ref(:,ci))\abs(seq);
                else
                    r = median(abs(seq)) / median(abs(ref(:,ci)));
                end

                % if ci == 1 & mod(si, 100) == 0
                %     figure(1); clf; hold on;
                %     plot(seq, '-ro');
                %     plot(ref(:,ci), '-bo')
                %     figure(2); clf; hold on;
                %     plot(seq, '-ro');
                %     plot(ref(:,ci)*r, '-bo');
                %     title(sprintf('r=%f', r));
                %     figure(3); clf; hold on;
                %     plot(seq, '-ro');
                %     plot(ref(:,ci)*r2, '-bo');
                %     title(sprintf('r=%f', r2));
                %     pause
                % end

                data2(idx, ci) = seq - ref(:,ci) * r;
            end
        end
    elseif strcmpi(METHOD_CANCEL, 'NO_AGC')
        data2 = data(1:nChirps*chirpLen, :) - repmat(ref, nChirps, 1);
    elseif strcmpi(METHOD_CANCEL, 'NO_CANCEL')
        data2 = data(1:nChirps*chirpLen, :);
    else 
        error('Wrong Calcellation Method');
    end
    datatmp = data;
    data = data2(startTime:end, :);


    %% ============================
    %% Plot Spectrum After Cancellation
    %% ============================
    if PLOT_SPECTRUM
        fprintf('- Plot Spectrum After Cancellation\n');

        win      = floor(Fs/512);
        noverlap = floor(win/4); % 75% overlap
        Nfft     = Fs;
        %% P matrix contains the power spectral density of each segment of the STFT
        % [S,F,T,P] = spectrogram(data2(1:1*Fs, 1), win, noverlap, Nfft, Fs);
        
        fh = figure(12); clf;
        subplot(2,1,1); 
        [S,F,T,P] = spectrogram(data2(1:1*Fs, 1), win, noverlap, Nfft, Fs);
        imagesc(T, F, 10*log10(P));
        colorbar;
        ax = gca; ax.YAxis.Exponent = 0;
        set(gca,'YDir','normal') 
        % ylim([17500 18500]);
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        title('Time-Frequency plot of a Audio signal (dB/Hz)');
        set(gca, 'FontSize', fontSize);

        subplot(2,1,2); 
        [S,F,T,P] = spectrogram(data2(15*Fs:16*Fs, 1), win, noverlap, Nfft, Fs);
        imagesc(T, F, 10*log10(P));
        colorbar;
        ax = gca; ax.YAxis.Exponent = 0;
        set(gca,'YDir','normal') 
        % ylim([17500 18500]);
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        title('Time-Frequency plot of a Audio signal (dB/Hz)');
        set(gca, 'FontSize', fontSize);
        pause
    end



    %% ============================
    %% Synchronization
    %% ============================
    fprintf('- Synchronize\n');

    dataSync = data(1:chirpLen*3, 1).';
    cr       = zeros(1, length(dataSync)-chirpLen+1);
    
    for ti = 1:length(cr)
        cr(ti) = abs(sum(dataSync(ti:ti+chirpLen-1) .* t0));
    end

    [v,stdIdx] = max(cr);
    if stdIdx < syncOffset
        stdIdx = stdIdx + chirpLen;
    end
    data = data(stdIdx-syncOffset:end, :);

    figure(2); clf; hold on;
    plot(cr, '-bo');
    plot(stdIdx, v, 'rx');
    xlabel('Time (samples)');
    ylabel('Corr');
    title('Synchronization');
    set(gca, 'FontSize', fontSize);


    %% ============================
    %% Estimate Distance
    %% ============================
    fprintf('- Estimate Distance\n');

    nSeg = floor(size(data,1) / chirpLen);
    %% Bug reoport here
    %% Change from `estDists = zeros(nSeg, nCh);` to `estDist = zeros(nSeg, nCh);`
    estDist = zeros(nSeg, nCh);
    images_mix = zeros(nSeg, nCh, Fs/2);

    for ci = 1:nCh % Mic nums
        for si = 1:nSeg % Chirp Nums
            dataSeg = data((si-1)*chirpLen+1:si*chirpLen, ci).';

            mix = t0 .* conj(dataSeg);
            Ym = fft(mix, Fs);
            Ym = Ym(1:Fs/2);
            Ym = abs(Ym);
            Xm = [0:Fs/2-1];
            XmDist = Xm * vc * chirpLenS / Bw;
            [~, idx] = max(Ym);
            estDist(si, ci) = XmDist(idx);
            
           %% change for better representation
            
            %             if PLOT_MIX_FFT
            %                 figure(1); clf; hold on;
            %                 plot(XmDist, Ym, '-b.');
            %                 xlabel('Distance (m)');
            %                 ylabel('Magnitude');
            %                 set(gca, 'FontSize', fontSize);
            %             end
            
            if PLOT_MIX_FFT
                images_mix(si,ci,:) = Ym; %
            end
            
        end
    end

    fprintf('- Plotting profiles\n');
        
    figure(3); clf; hold on;
    lhs = zeros(1, nCh);
    legends = {};
    for ci = 1:nCh
        lhs(ci) = plot([0:size(estDist,1)-1]*chirpLenS, estDist(:, ci), '-bo');
        set(lhs(ci), 'Color', colors{mod(ci-1,length(colors))+1});
        set(lhs(ci), 'marker', markers{mod(ci-1,length(markers))+1});
        legends{ci} = sprintf('mic%d', ci);
    end
    legend(lhs, legends);
    xlabel('Time (s)');
    ylabel('Distance (m)');
    title('Estimated Distances');
    set(gca, 'FontSize', fontSize);

    if PLOT_MIX_FFT
        figure(4); clf; hold on;    
        size(images_mix)
        size(XmDist)
        aaa = images_mix(:,1,:);
        bbb = squeeze(aaa);
        imagesc(bbb');
    end
    
    %% ============================
    %% Error Calculation
    %% ============================
    fprintf('- Error Calculation\n');
    %
    % Generate ground truth distances
    %
    if cal_error
        
        num_chirp_at_each_dist = distance_duration_time / chirpLenS;
        total_chirps = length(distance_lists(filename)) * num_chirp_at_each_dist;
        GroundTruths = ones(total_chirps,1);
        counter = 0;
        for a_distance = distance_lists(filename)
            GroundTruths(num_chirp_at_each_dist*counter+1:num_chirp_at_each_dist*(counter+1)) = ones(num_chirp_at_each_dist,1)*a_distance;
            counter = counter+1;    
        end
        size(GroundTruths)
        %
        % Normalize the estimated dist to Ground truth distances
        % Now just by observation for a,b to c,d
        %
        figure(6);
        %plot(estDist(:,1))
        %
        
        use_mic = 3;
        to_norm_dists = estDist(:,use_mic); % the first mic
        
        
        a=1.30; b=2.08;
        a = 1.52; b = 4.18;
        a = 1.41; b = 3.19;
        a = 1.48; b = 3.99;

        a = 1.83; b = 4.31;

        c = 50; d = 90;
        c = 30; d = 160;
        %
        tmp = (to_norm_dists - a) / (b - a);
        CorrDists = tmp * (d-c) + c;
        %plot(CorrDists)
        %
        % Align, and Cal Errors
        %
        GT = GroundTruths(startTimeS/chirpLenS:end);
        errors =  abs( GT - CorrDists(1:length(GT)) );
        xticks = (0+chirpLenS):chirpLenS:length(GT)*chirpLenS;
        plot(xticks,errors)

        xlabel('Time (s)');
        ylabel('Distance Errors(cm)');
        xline(0:20:max(GT)-30);
        title('Estimated Distances');
        set(gca, 'FontSize', fontSize);

    end
%end