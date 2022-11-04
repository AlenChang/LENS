function [data, tx_packet] = lili_send_test(seed)

%--------------------------------------------------------------------------
% 0) Initialize some global constants
%--------------------------------------------------------------------------

% initialize random seed
rng(seed);

% use 96KHz as the sampling frequency for the WAV file
% later may want to switch to 44.1KHz
Fs = 48e3;

% consider [32KHz,44KHz] instead of [20KHz, 48KHz] to leave enough guard band
% on both sides.  BW = f_max - f_min = 12000 = Fs/8.  For simplicity, we 
% only consider Fs/2^k, later should be able to extend to more general case
f_min = 17000;   
f_max = 20000;

% use 128 samples so that eacy OFDM symbol is not too narrow
Nfft = 128;
Nfft_useful = Nfft * (f_max - f_min) / Fs;
% keyboard
% 1/4 symbols for cyclic prefix
Ncp = Nfft / 4;
Ntx = 1;
Mod = 4;
state = 4831;

% modulated symbols need to be a multiple of Nfft_useful!
data_size = 256;

% keyboard

%--------------------------------------------------------------------------
% 0) Generate CDMA code.  It is important to use a primitive generator polynomial.
% Here are a few good ones according to
%   http://www.weizmann.ac.il/matlab/toolbox/commblks/ref/pnsequencegenerator.html
%
% [2 1 0], [3 2 0], [4 3 0], [5 3 0], [6 5 0], [7 6 0], [8 6 5 4 0], [9 5 0]
%
% set chip_seq_length to 1 to disable CDMA
%--------------------------------------------------------------------------
chip_seq_length = 1; %16;
pns = comm.PNSequence('Polynomial',[6 5 0], 'SamplesPerFrame', chip_seq_length, ...
                      'InitialConditions',[0 0 0 0 0 1]);
chipping_seq = step(pns);
chipping_seq = step(pns);
chipping_seq = reshape(chipping_seq,1,[]);

%--------------------------------------------------------------------------
% 1) generate raw data
%--------------------------------------------------------------------------
data = randi([0 1], 1, data_size);

%--------------------------------------------------------------------------
% 2) FEC encoding
%--------------------------------------------------------------------------
hConEnc1by2 = comm.ConvolutionalEncoder(poly2trellis(7, [171 133]));
encodedData = step(hConEnc1by2, data.');
encodedData_size = length(encodedData);
aaa = data(1:10);
yyy = encodedData(1:10);

%--------------------------------------------------------------------------
% 3) random interleave
%--------------------------------------------------------------------------
stream = RandStream('mt19937ar', 'Seed', state);
interleaved_data = randintrlv(encodedData, stream).';
zzz =		   interleaved_data(1:10);

%--------------------------------------------------------------------------
% 4) asynchronous CDMA: xor with chipping sequence
%--------------------------------------------------------------------------
cdma_data = reshape(repmat(interleaved_data, chip_seq_length, 1), 1, chip_seq_length* ...
                    encodedData_size);
chipping_seq_all = repmat(chipping_seq, 1, encodedData_size);
cdma_data = xor(cdma_data, chipping_seq_all);

%--------------------------------------------------------------------------
% 5) modulate digital data to analog signal using QAM
%--------------------------------------------------------------------------
hModulator4 = comm.RectangularQAMModulator(Mod,...
             'BitInput',true, 'NormalizationMethod', 'Average power');
modSymbols = step(hModulator4, cdma_data');

%--------------------------------------------------------------------------
% 6) add preamble for synchronization                        
%--------------------------------------------------------------------------
preamble_length = Nfft_useful * 4;
% keyboard
pns = comm.PNSequence('Polynomial',[8 6 5 4 0], 'SamplesPerFrame', preamble_length, ...
                      'InitialConditions',[1 0 0 0 1 0 0 1]);
% keyboard
preamble = step(pns);
preamble = step(pns);
modPreamble = step(comm.BPSKModulator,preamble);
modSymbols = vertcat(modPreamble,modSymbols);

preamble_num_ofdm = preamble_length/Nfft_useful;

assert(rem(length(modSymbols),Nfft_useful)==0,...
       'modulated symbols should be interger multiple of Nfft');

%--------------------------------------------------------------------------
% 7) Stripe the data onto Nfft_useful subcarriers.  Only make use of channels
% corresponding to frequency range [f_min, f_max].
%--------------------------------------------------------------------------
r = length(modSymbols)/Nfft_useful;
c = Nfft;
data_shaped = zeros(r,c);
c_start = floor(f_min / Fs * Nfft) + 1;
c_end   = c_start + Nfft_useful - 1;
data_shaped(:,c_start:c_end) = reshape(modSymbols,Nfft_useful,length(modSymbols)/Nfft_useful).';

tx_packet = zeros(1,r*(c+Ncp));
Nofdm = Nfft+Ncp;
subs_index = 0:Nfft-1;

for f=1:size(data_shaped,1)

  fft_data = data_shaped(f,:);
  
  ifft_data = ifft(fft_data,Nfft);

  %---------------------------------------------------------------------------
  % XXX: Commented out.  Up-convert for some reason introduce
  % a huge low-frequency component.  making the noise really loud
  %---------------------------------------------------------------------------
  % 7.5) up convert to the right frequency by multiplying by exp(i*2*pi*f_min*t)
  %ifft_data = ifft_data.*exp(i*2*pi*f_min*(subs_index+(f-1)*Nfft));
  cdd_ifft_data = ifft_data;

  %---------------------------------------------------------------------------
  % 8) add cyclic prefix
  %---------------------------------------------------------------------------
  if (Ncp > 0)
    cp = cdd_ifft_data(end-Ncp+1:end);
  else
    cp = [];
  end
  ofdm_symbol = horzcat(cp,cdd_ifft_data);

  %---------------------------------------------------------------------------
  % 9) transmit
  %---------------------------------------------------------------------------
  tx_packet_tmp = ofdm_symbol;

  % add white gaussin noise
  tx_packet(1+Nofdm*(f-1):Nofdm*f) = tx_packet_tmp; % awgn(tx_packet_tmp,SNR);

  % post process tx_packet so that it can be saved as a wav file
  offset = Nofdm*(f-1)*2;
  for idx = 1:length(tx_packet_tmp)
    wav_packet(offset+1:offset+Nofdm) = real(tx_packet_tmp);
    wav_packet(offset+Nofdm+1:offset+2*Nofdm) = imag(tx_packet_tmp);  
  end
end

%---------------------------------------------------------------------------
% 10) perform amplitude normalization to make it louder
%---------------------------------------------------------------------------
wav_packet = wav_packet * Nfft / Nfft_useful;

%---------------------------------------------------------------------------
% 11) add random noise to test synchronization
%---------------------------------------------------------------------------
%wav_packet = [(rand(1,30)*2-1) wav_packet];

filename = sprintf('sent_packet%d.wav',seed);      
% keyboard
audioout = repmat(wav_packet, 1, round(5 * Fs / length(wav_packet)));
audiowrite(filename, audioout,Fs);
audioout = audioout';
audioout = [audioout zeros(size(audioout))];
audiowrite('tx.wav', audioout,Fs);
size(wav_packet)

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%% end of file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fs = 96000;
% 
% % machine1:
% [wav_packet,Fs,nbits] = audioread('sent_packet5');
% sound(wav_packet,Fs);
% 
% % machine2:
% r = audiorecorder(Fs,16,1);
% recordblocking(r,5);
% rcv_packet = getaudiodata(r,'int16');
% plot(rcv_packet);
% wavwrite(rcv_packet,Fs,'rcv_packet5');

% return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot the power spectrum
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
