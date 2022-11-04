array = phased.ULA(6);
t = unigrid(0,0.001,0.01,'[)');
x = cos(2*pi*100*t)';
y = collectPlaneWave(array,x,[-90;0],1e9,physconst('LightSpeed'));
steervec = phased.SteeringVector('SensorArray',array);
sv = steervec(1e9,[-90;0]);
y1 = x*sv.';