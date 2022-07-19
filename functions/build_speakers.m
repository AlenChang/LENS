function speaker = build_speakers(num_speakers, speaker_center, speaker_spacing)
speaker.num = num_speakers;
speaker.center = reshape(speaker_center, 1, 2);
speaker.locs = zeros(speaker.num, 2);
speaker.fc = 20e3;
speaker.c = 34300;
speaker.lambda = speaker.c / speaker.fc;
speaker.speaker_spacing = speaker.lambda * speaker_spacing;

for ti = 1:speaker.num
    speaker.locs(ti, 2) = (- (speaker.num - 1) / 2 + ti - 1) * speaker.speaker_spacing;
end

speaker.locs = speaker.locs  + speaker.center;


fprintf("================\n")
fprintf("Speaker config:\n")
fprintf("        Number of Speakers: %i\n", speaker.num)
fprintf("        Central frequency: %i kHz\n", speaker.fc / 1000)
fprintf("        Wave length: %f cm\n", speaker.lambda)
end