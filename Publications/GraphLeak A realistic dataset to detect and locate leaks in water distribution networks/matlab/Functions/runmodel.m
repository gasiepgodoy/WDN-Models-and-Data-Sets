function output = runmodel(epanet,tv,tmin,cpm,cpv,leak_array,lktype,lktime,nos_med,tot_nos_vaz,tr_med,len,wkday)
% == Function that runs EPANET model and creates the data set ==
% epanet (epanet model) = EPANET model
% tv (double) = time vector for the data set
% tmin (double) = time interval (minutes)
% cpm (array) = coordinates of measurement nodes
% cpv (array) = coordinates of leakage nodes
% leak_array (array) = indicates which leakage node is active
% lktype (string) = indicates whether leakage is constant ('c'), intermitent
% ('i') or none ('n')
% lktime (int) = Starting time of leakage (only matters for intermitent type)
% nos_med (array) = IDs of measurement nodes
% tot_nos_vaz (int) = number of leakage points in the model
% tr_med (array) = IDs of measurement links (for flow measurement)
% len (int) = number of lines in the data set
% wkday (int) = weekday (e.g. 1 = weekday, 2 = weekend/holyday etc.)

g = epanet;
g.runsCompleteSimulation;
g.BinUpdateClass;
flow = g.getBinComputedLinkFlow;        
pres = g.getBinComputedNodePressure;    
vol = calc_vol(len,tr_med,flow,tmin); 

% Obtains coordinates of the leakage node
ivz_cpva = lk_map_v2(tot_nos_vaz,tv,leak_array,lktype,lktime,cpv,len);
% creates the dataset
output = [tv flow(:,tr_med) pres(:,nos_med) vol cpm ivz_cpva wkday ];

end


