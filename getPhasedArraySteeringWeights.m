function speaker_weights = getPhasedArraySteeringWeights(params, num_speaker)
    speaker_weights = zeros(181, num_speaker);
    for ti = 1:181
    
        for mi = 1:num_speaker
            speaker_weights(ti, mi) = exp(-1j * 2 * pi * params.fc / params.c * (num_speaker - mi) * params.lambda / 2 * cos((ti - 1) / 180 * pi));
        end
    
    end
end