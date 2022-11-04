
[data, tx_packet] = lili_send_test(0);
[decodedData,reshaped_pkt] = lili_recv(0);
BER = sum(data ~= decodedData)/length(data);
disp(sprintf("BER is %.4f", BER))
% figure(1)
% clf
% plot(tx_packet)
% hold on
% plot(reshaped_pkt)