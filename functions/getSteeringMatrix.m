function [steerVec, theta] = getSteeringMatrix(lens)
    theta = -90:1:90;
    d = lens.speaker_spacing * sin(theta / 180 * pi);
    steerVec = zeros(lens.num, length(theta));

    for ti = 1:lens.num
        steerVec(ti, :) = exp(1j * 2 * pi * lens.fc / lens.c * d * (ti - 1));
    end

end
