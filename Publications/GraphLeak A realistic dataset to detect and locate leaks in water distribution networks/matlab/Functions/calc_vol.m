function output = calc_vol(len,tr_med,flow,tmin)
% == Function to calculate volume of water on the measurement nodes==
% len (int) = number of lines of the data set
% tr_med (array) = ID of links that measure flow
% flow (array) = flow data array
% tmin (double) = reporting time interval (minutes)

tmin = double(tmin);   
vol = zeros(len,length(tr_med)); 
for j = 1:length(tr_med)
for i = 1:len
vol(i,j) = tmin*flow(i,tr_med(j));
end
end

output = vol;
end
