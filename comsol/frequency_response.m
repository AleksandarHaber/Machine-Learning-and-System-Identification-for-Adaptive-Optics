% this file is used to generate raw plots in Fig. 1b in the paper

% load the data
frequency_response_data

% we need to convert the response to dB
p1_db=20*log10(p1) %because the input amplitude is 1, we do not need to explicitly divide
p2_db=20*log10(p2) %because the input amplitude is 1, we do not need to explicitly divide

figure(1)
hold on
plot(freq,p1_db)
set(gca, 'XScale', 'log')

figure(2)
hold on
plot(freq,p2_db)
set(gca, 'XScale', 'log')