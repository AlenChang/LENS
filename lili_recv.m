function [decodedData,reshaped_pkt] = yin_recv(seed)

%--------------------------------------------------------------------------
% 0) Initialize some global constants
%--------------------------------------------------------------------------

% initialize random seed
rng(seed);

% use 96KHz as the sampling frequency for the WAV file
% later may want to switch to 44.1KHz
Fs = 96000;

% consider [32KHz,44KHz] instead of [20KHz, 48KHz] to leave enough guard band
% on both sides.  BW = f_max - f_min = 12000 = Fs/8.  For simplicity, we 
% only consider Fs/2^k, later should be able to extend to more general case
f_min = 32000;   
f_max = 44000;

% use 128 samples so that eacy OFDM symbol is not too narrow
Nfft = 128;
Nfft_useful = Nfft * (f_max - f_min) / Fs;

c_start = floor(f_min / Fs * Nfft) + 1;
c_end   = c_start + Nfft_useful - 1;

% 1/4 symbols for cyclic prefix
Ncp = Nfft / 4;
Nrx = 1;
Mod = 4;
state = 4831;

decodedData = [];

data_size = 256;

subs_index = [0:Nfft-1];
Nofdm = Nfft + Ncp;

alpha = 1/8;
beta = 1/4;
p_ewma = -10000;
p_dev_ewma = -10000;

% generate CDMA code
chip_seq_length = 1; %16;
pns = comm.PNSequence('Polynomial',[6 5 0], 'SamplesPerFrame', chip_seq_length, ...
                      'InitialConditions',[0 0 0 0 0 1]);
chipping_seq = step(pns);
chipping_seq = step(pns);
chipping_seq = reshape(chipping_seq,1,[]);

% generate FEC decoder
hDec1by2 = comm.ViterbiDecoder('InputFormat','Hard',...
                               'TerminationMethod','Truncated');

% generate demodulator for the data             
hDeModulator4 = comm.RectangularQAMDemodulator(Mod,...
                                               'BitOutput',true, 'NormalizationMethod', 'Average power');

% prepare modulated preamble
preamble_length = Nfft_useful * 4;
pns = comm.PNSequence('Polynomial',[8 6 5 4 0], 'SamplesPerFrame', preamble_length, ...
                      'InitialConditions',[1 0 0 0 1 0 0 1]);
preamble = step(pns);
preamble = step(pns);
modPreamble = step(comm.BPSKModulator,preamble);
modPreamble_conj_normalized = conj(modPreamble)./norm(modPreamble,'fro');;

% process the wav file
filename=sprintf('32KHz-44KHz/recv/rcv_packet%d.wav',seed);
[wav_packet_orig,Fs,nbits] = wavread(filename);
filename=sprintf('sent_packet%d.wav',seed);
[sent_pkt_orig,Fs,nbits] = wavread(filename);

% convert from [0, 1] to [-1, 1]
wav_packet_orig = wav_packet_orig*Nfft_useful/Nfft;
sent_pkt_orig = sent_pkt_orig*Nfft_useful/Nfft;

ttt = wav_packet_orig(1:10)

% synchronization using preamble
preamble_offset = -1;

fd=fopen('p.txt','w');
p_lst = [];
preamble_num_ofdm = preamble_length/Nfft_useful;

% rcv_packet1: 197031
% 2: 192854

p_max = -1;

if 1
for try_offset = [1:length(wav_packet_orig)]

  % get the first few ofdm symbols
  first_ofdm = [];
  for r=1:preamble_num_ofdm
    for offset=1:Nofdm
      idx=(r-1)*Nofdm+offset;
      wav_idx=try_offset+(r-1)*2*Nofdm+offset-1;
      % if try_offset==301 && r==1 && offset==Nofdm
      %    [idx wav_idx wav_idx+Nofdm]
      % end
      if (wav_idx+Nofdm>length(wav_packet_orig))
        break;
      end
      first_ofdm(idx) = wav_packet_orig(wav_idx) + i*wav_packet_orig(wav_idx+Nofdm);
    end
  end

  if (length(first_ofdm) < Nofdm*preamble_num_ofdm)
     break;
  end
  first_ofdm = reshape(first_ofdm,Nofdm,preamble_num_ofdm).';
 
if try_offset== 1
   foobar = first_ofdm(1,1:10)
end
  % remove cp
  first_ofdm_matrix = zeros(preamble_num_ofdm, Nfft);
  for r=1:preamble_num_ofdm
      first_ofdm_matrix(r,:) = first_ofdm(r,Ncp+1:Nofdm);
  end


  %----------------------------------------------------------------------------------
  % XXX: No longer need to down convert
  %----------------------------------------------------------------------------------
  % down convert
  %for r=1:preamble_num_ofdm
  %   first_ofdm_matrix(r,:) = first_ofdm_matrix(r,:) .* exp(-i*2*pi*f_min*(subs_index));
  %end

  % FFT
  % fft_symbol = (1./sqrt(Nfft)).*fft(first_ofdm,Nfft);
  for r=1:preamble_num_ofdm
      fft_symbol(r,:) = fft(first_ofdm_matrix(r,:),Nfft).';
  end

  fft_symbol_vector = [];
  for r=1:preamble_num_ofdm
      fft_symbol_vector = [fft_symbol_vector fft_symbol(r,c_start:c_end)];
  end
  fft_symbol_vector = fft_symbol_vector.';
xxx = fft_symbol_vector(1:10);
  % preamble correlation
  % C = cov(modPreamble,fft_symbol);
  % p = C(2)/std(modPreamble)*std(fft_symbol); 
  p = abs(sum(modPreamble_conj_normalized .* fft_symbol_vector)./norm(fft_symbol_vector,'fro'));

  %----------------------------------------------------------------------
  % XXX: stopping critera: (1) p needs to exeed 0.5; (2) there is no larger
  % p in [preamble_offset + 1, preamble_offset + Ncp]
  %----------------------------------------------------------------------  
  if (p_max > 0) && (p < p_max) && (try_offset >= preamble_offset + Ncp)
    break;
  end

  % [try_offset p]
  % As long as we have a larger p, update preamble_offset and Hfft
  if (p > 0.5) && (p >= p_max)
    p_max = p;
    preamble_offset = try_offset;
    Hfft = fft_symbol_vector(1:Nfft_useful)./modPreamble(1:Nfft_useful);
    Hfft = real(Hfft)'+i*imag(Hfft)';
  end
end

p_max
preamble_offset

if preamble_offset==-1
   fprintf('Unable to find preamble\n');
   return
end
end

preamble_offset

geee = length(wav_packet_orig)/Nofdm
% 0) prepare complex numbers
% remove preamble
wav_packet_orig = wav_packet_orig(preamble_offset+Nofdm*preamble_num_ofdm*2:end);
sent_pkt_orig = sent_pkt_orig(Nofdm*preamble_num_ofdm*2:end);
ppp=size(wav_packet_orig)
preamble_offset+Nofdm*preamble_num_ofdm*2
num_ofdm = data_size*chip_seq_length*2/2/Nfft_useful; % xxx: todo

for r=1:num_ofdm
    for offset=1:Nofdm
      idx=(r-1)*Nofdm+offset; 
      wav_idx=(r-1)*2*Nofdm+offset;                            
      reshaped_pkt(idx) = wav_packet_orig(wav_idx) + i*wav_packet_orig(wav_idx+Nofdm);
      sent_pkt(idx) = sent_pkt_orig(wav_idx) + i*sent_pkt_orig(wav_idx+Nofdm);
    end
end

num_ofdm = ceil(length(reshaped_pkt)/Nofdm);
reshaped_pkt = reshape(reshaped_pkt,Nofdm,num_ofdm).'; 
reshaped_pkt(1,1:16) % ok
sent_pkt = reshape(sent_pkt,Nofdm,num_ofdm);

% 1) remove cp 
nfft_pkt = reshaped_pkt(:,Ncp+1:end);

% go through all the OFDM data symbols
for f = 1:size(nfft_pkt,1)

  % 2) down convert to the baseband
  %nfft_pkt(f,:) = nfft_pkt(f,:) .* exp(-i*2*pi*f_min*(subs_index+f*Nfft));

  % 3) FFT
  % fft_symbol = (1./sqrt(Nfft)).*fft(nfft_pkt(f,:),Nfft);
  fft_symbol = fft(nfft_pkt(f,:),Nfft);

  % 4) channel compensation
  equalized_symbol = fft_symbol(c_start:c_end)./Hfft;
  equalized_pkts(1+(f-1)*Nfft_useful:f*Nfft_useful) = equalized_symbol;
end
equalized_pkts = equalized_pkts.';

% 4) demodulate signals to data
demodSignal = step(hDeModulator4, equalized_pkts).';

% 5) extract the original encoded signals from CDMA encoded data: checked
encodedData_size = length(demodSignal)/length(chipping_seq)
chipping_seq_all = repmat(chipping_seq, 1, encodedData_size);
interleaved_data_rcv = xor(demodSignal, chipping_seq_all)+0;
interleaved_data_rcv = reshape(interleaved_data_rcv, chip_seq_length, encodedData_size);
interleaved_data_rcv = sum(interleaved_data_rcv, 1) >= ceil(chip_seq_length/2); 


yyy=interleaved_data_rcv(1:10)
% 6) de-interleave: checked
stream = RandStream('mt19937ar', 'Seed', state);
encodedData = randdeintrlv(interleaved_data_rcv+0, stream);

encodedData(1:10)
% 7) FEC decoding
decodedData = step(hDec1by2, encodedData.'+0.0).';

decodedData(1:10)


% 8) extract the data 
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% end of file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% correlation between two complex numbers
fd=fopen('p.txt','r');
x=fscanf(fd,'%f');
plot(x);

fd=fopen('p-try1.txt','r');
x=fscanf(fd,'%f');
plot(x)

figure
fd=fopen('p-try2.txt','r'); 
y=fscanf(fd,'%f'); 
plot(y)
                                                                         
     
