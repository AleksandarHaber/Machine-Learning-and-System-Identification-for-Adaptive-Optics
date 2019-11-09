% first load the data from "frequency_response_data.m" and "frequency_response_damped_rayleigh.m"

% we need to convert the response to dB
p1_db_undamped=20*log10(p1_undamped) %because the input amplitude is 1, we do not need to explicitly divide
p1_db_damped=20*log10(P1amplitude)

% we need to convert the response to dB
p2_db_undamped=20*log10(p2) %because the input amplitude is 1, we do not need to explicitly divide
p2_db_damped=20*log10(P2amplitude)

figure(1)
hold on
plot(freq,p1_db_undamped,'r')
hold on 
plot(freq,p1_db_damped)
set(gca, 'XScale', 'log')


figure(2)
hold on
plot(freq,p2_db_undamped)
hold on 
plot(freq,p2_db_damped)
set(gca, 'XScale', 'log')

