%% evaluate the correctness of the model.
fc = 20e3;
c = 34300;
lambda = c / fc;

field_len = 20;

gridsize = lambda / 50;

fieldx = -field_len / 2:gridsize:field_len / 2;
fieldy = -field_len / 2:gridsize:field_len / 2;

field = zeros(length(fieldx), length(fieldy));
field_locs = zeros(length(fieldx), length(fieldy), 2);

for ti = 1:length(fieldx)

    for mi = 1:length(fieldy)
        field_locs(ti, mi, 1) = fieldx(ti);
        field_locs(ti, mi, 2) = fieldy(mi);
    end

end

% field_locs = field_locs;

% speaker_locs = [0 0] * lambda;
num_speakers = 9;
% speaker_locs = [0 -2; 0 -1; 0 0; 0 1; 0 2; ];
speaker_locs = zeros(num_speakers, 2);

for ti = 1:num_speakers
    speaker_locs(ti, 2) =- (num_speakers + 1) / 2 + ti;
end

speaker_locs = speaker_locs * lambda / 2;

num_target =120;
target_locs = zeros(num_target, 2);
target_coordinates = "surface";

if (strcmp(target_coordinates, "plane"))
    targetx = linspace(fieldx(1) + 1, fieldx(end) - 1, num_target);

    for ti = 1:num_target
        target_locs(ti, 1) = -8;
        target_locs(ti, 2) = targetx(ti);
    end

    b = ones(num_target, 1);

elseif (strcmp(target_coordinates, "polar"))

    target_direct = [pi/6, 3*pi/6];
    target_locs = zeros(num_target * length(target_direct), 2);
    b = [];

    for ni = 1:length(target_direct)
        r_len = linspace(4, 9, num_target);
        % if(target_direct(ni) > 0)
        %     phi_test = pi / 2;
        % else
        %     phi_test = -pi / 2;
        % end

        for ti = 1:num_target
            target_locs((ni - 1) * num_target + ti, 1) = r_len(ti) * sin(target_direct(ni));
            target_locs((ni - 1) * num_target + ti, 2) = r_len(ti) * cos(target_direct(ni));
        end

        b = [b; (exp(1j * 2 * pi * fc / c * r_len) ./ (2*pi*r_len)).'];
    end

    num_target = num_target * length(target_direct);

elseif(strcmp(target_coordinates, "surface"))
    phi = linspace(0, pi, num_target);
    % r_len = 6*(abs(sinc(linspace(-0.5, 0.5, num_target))))';
    r_len = 6 * ones(num_target, 1);
    for ti = 1 : num_target
        target_locs(ti, 1) = r_len(ti) * sin(phi(ti));
        target_locs(ti, 2) = r_len(ti) * cos(phi(ti));
    end
    % b = (exp(1j * 2 * pi * fc / c * r_len) ./ (2*pi*r_len)).';
    % b = (1 ./ (2*pi*r_len)).';
    b = abs(sinc(linspace(-4.5, 4.5, num_target)+3))' .* exp(1j*2*pi) + abs(sinc(linspace(-4.5, 4.5, num_target)))' .* exp(1j*pi / 2) + abs(sinc(linspace(-4.5, 4.5, num_target)-3))' .* exp(1j*pi);
end

% target_locs = target_locs * gridsize;

R = zeros(num_target, num_speakers);
dist = zeros(size(R));

for ti = 1:num_target

    for ni = 1:num_speakers
        d = target_locs(ti, :) - speaker_locs(ni, :);
        d = sqrt(d(1)^2 + d(2)^2);
        R(ti, ni) = exp(-1j * 2 * pi * fc * d / c) ./ (2 * pi * d);
    end

end

if (rank(R) == size(R, 2))
    a = pinv(R' * R) * R' * b / (b'*b);
elseif (rank(R) < size(R, 2))
    %     format long
    Z = null(R);

    at = pinv(R' * R) * R' * b / (b'*b);
    at = at / sqrt(at' * at);
    %     syms k1
    %     x = k1*Z+a;
    %     pretty(x)
    %     k = 2;
    %     a = Z*k'+a;
    k = max((1 - a) ./ Z);
    %     k = 1+2j;
    %     a = Z * 2 + at;
    a = Z * k.' +at;
    a = at;
else
    a = pinv(R' * R) * R' * b / (b'*b);
end

% a = a ./ max(abs(a));

% phi = rand(1, num_speakers) * 2 * pi;
% amp = rand(1, num_speakers);

phi = angle(conj(a));
amp = abs(a);

sound_field = zeros(size(field));

for ti = 1:num_speakers
    locs = speaker_locs(ti, :);

    for mi = 1:size(field_locs, 1)

        for ni = 1:size(field_locs, 2)
            dx = field_locs(mi, ni, 1) - locs(1);
            dy = field_locs(mi, ni, 2) - locs(2);
            d = sqrt(dx.^2 + dy.^2);
            field(mi, ni) = field(mi, ni) + amp(ti) * exp(-1j * (2 * pi * fc * d / c + phi(ti)));
        end

    end

    %     locs = reshape(locs, [1,1,2]);
    %     d = field_locs - locs;
    %     d = imag(sqrt(d(:, :, 1) .^ 2 + d(:, :, 2) .^ 2));
    %     field = field + cos(2*pi*fc*d / c);
end

figure(1)
clf
% subplot(221)
imagesc(fieldx, fieldy, real(field))
hold on
plot(target_locs(:, 2), target_locs(:, 1), 'ro')
plot(speaker_locs(:, 2), speaker_locs(:, 1), 'r*')

% circ = getCircle() * 10 / gridsize;
% plot(real(circ) - fieldx(1) / gridsize, imag(circ) - fieldy(1) / gridsize, 'r', 'LineWidth', 2)
% set(gca, 'XTick', [], 'YTick', [])
pbaspect([1, 1, 1])

out = zeros(size(target_locs, 1), 1);

for ti = 1:num_target
    out(ti) = field(round((target_locs(ti, 1) - fieldx(1)) / gridsize), round((target_locs(ti, 2) - fieldy(1)) / gridsize));
end

out
