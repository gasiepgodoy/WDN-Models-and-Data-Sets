%% Dataset generation - Case 1

close all
clear
clc
gen_csv = 1;  % Choose whether CSV files will be exported or not (1= true, 0=false)

% Basic setup
nc = 1;   % Scene ID number (for identification)
seeds = [11 28 135 248 505 998 1050 1334 2210 3151]; % Random number generator seeds
tas = [10 30 60 180 360  720];  % report intervals (minutes)
mults = [1 5 10 50]; % multiplication factors for pipe length

for (v_seed=1:length(seeds))
seed = seeds(v_seed);
for (v_mults = 1:length(mults));
mult = mults(v_mults);
for (v_tas = 1:length(tas));
ta = tas(v_tas);

% Columns in the dataset to check for negative pressure
first = 10;     
last = 17;  

% Variables setup
rng(seed)       % selects a specific seed for the random number generator
dsct = 0;       % dataset counter
log_pc = [];    % log of the drawn consumption patterns
fprintf('\n *-*-*-*-*-*-*-*-* \n')
fprintf(datestr(now,'dd-mm-yyyy HH:MM:SS'))
fprintf('Seed: %d, MultFact: %d, ReportTime: %d\n',seed,mult,ta);

%Load EPANET model
d = epanet('case1.inp'); % INP file generated on EPANET
d.solveCompleteHydraulics;


%% Pump adjustment (if needed)
d.setCurve(1,[75,40])   % [flow, load]
d.runsCompleteSimulation;
d.BinUpdateClass;
d.getBinCurvesInfo;

%% Report time adjustment (automatic)

d.setBinTimeReportingStep(ta*60); 
d.setBinTimeHydraulicStep(ta*60); 
d.runsCompleteSimulation;
d.BinUpdateClass;

%% Pipe length adjustment
n_pipes = 10;    % Insert the number of pipes (links) in the main network
for q = 1:n_pipes
tmp = d.getLinkLength(q);
d.setLinkLength(q,tmp*mult);
end

%% Check number of elements
nodes = d.getNodeCount;     % Number of nodes -> pressure measurement
links = d.getLinkCount;     % Number of links (pipes) -> flow measurement
tmp = d.getBinComputedLinkFlow;
len = size(tmp,1);
clear tmp

%% Scene data (Config manually)
num_res = 1;                        % Number of fixed-level reservoirs 
meas_nodes = [2 3 4 5 6 7 8 9];     % Measurement nodes IDs (joint nodes + consumption units)
meas_links = [1 6 8 11 13 3 9 4];   % Pipes with flow measurement
leak_nodes = [10 11 12 13 14];      % Nodes for leak simulation
tot_leak_nodes = length(leak_nodes);% total leakage nodes

%% Node coordinates

elev = d.getNodeElevations;   % Elevation of nodes
for i = 1:(nodes-num_res)
coord = d.getNodeCoordinates(i); 
coords{i} = [coord(1) coord(2) elev(i)];
end
clear i coord

%% Node mapping
cpm = node_map(meas_nodes,coords,len);
cpv = node_map(leak_nodes,coords,len);

%% Adjacency matrix (insert manually)
MCJ = [
    0	10	0	0	0	0	0	0;
    10	0	3	3	0	0	0	0;
    0	3	0	0	0	0	6	6;
    0	3	0	0	6	6	0	0;
    0	0	0	6	0	0	0	0;
    0	0	0	6	0	0	0	0;
    0	0	6	0	0	0	0	0;
    0	0	6	0	0	0	0	0
];

%% Timestamp creation
ttime = d.getTimeSimulationDuration;
step = d.getTimeReportingStep;
tv = double([0:step:ttime]');
tmin = double(step/60);
tot_hours = double(ttime/3600);
% To format as HH:MM:SS
hour = linspace(0,tot_hours,1+(ttime/step));
hour = hours(hour);
hour.Format = 'hh:mm';

%% Including 8 extra consumption patterns to enrich the model
% Follow the patterns available in the EPANET model
% In this case, there were 8 time points and 8 corresponding demand values

hrs = [1 5 8 12 15 18 21 24];
pts = [0.1 0.4 0.8 2 2.2 1.3 1.9 0.5];
[pstr, pv] = pat_interp(d, hrs, pts);
d.addPattern(pstr,pv); 

hrs = [1 5 8 12 15 18 21 24];
pts = [0.4 0.1 0.5 1.3 2 1 0.9 0.3];
[pstr, pv] = pat_interp(d, hrs, pts);
d.addPattern(pstr,pv);

hrs = [1 8 15 19 21 24];
pts = [0.2 0.7 3 2 1.1 0.3];
[pstr, pv] = pat_interp(d, hrs, pts);
d.addPattern(pstr,pv);

hrs = [1 8 15 17 19 21 24];
pts = [0.2 1 2 1.3 2 1.6 0.2];
[pstr, pv] = pat_interp(d, hrs, pts);
d.addPattern(pstr,pv);

hrs = [1 3 5 8 12 15 19 21 24];
pts = [0.2 0.5 0.8 2 1.1 0.7 1.5 2 0.2];
[pstr, pv] = pat_interp(d, hrs, pts);
d.addPattern(pstr,pv);

hrs = [1 5 9 12 15 19 21 24];
pts = [0.5 0.3 0.3 2 1.5 2.7 0.8 0.6];
[pstr, pv] = pat_interp(d, hrs, pts);
d.addPattern(pstr,pv);

hrs = [1 5 9 12 15 19 21 24];
pts = [0.5 0.3 0.3 2 1.5 2.7 1 0.6];
[pstr, pv] = pat_interp(d, hrs, pts);
d.addPattern(pstr,pv);

hrs = [1 5 9 12 15 19 21 24];
pts = [0.3 0.5 0.8 1.8 1.5 1.3 0.8 0.6];
[pstr, pv] = pat_interp(d, hrs, pts);
d.addPattern(pstr,pv); 

%% Additional demand patterns to simulate external consumption (P15 a P25)
% This loop generates randomized patterns within a predefined range [min,
% max] for demands
min = 0;
max = 3.5;

hrs = [1 4 7 10 13 16 19 21 24];
for i = 1:11
    pts = min + (max-min) .* rand(1,9);
    [pstr, pv] = pat_interp(d, hrs, pts);
    d.addPattern(pstr,pv);
end

%% Parameters to randomize base demands on external consumption nodes
bdmin = 1;  % minimal base demand
bdmax = 2;  % maximum base demand

%% In case you want to plot the available patterns:
%pat = d.getPattern;
%figure
%plot(pat(3,:))  % to plot pattern P3
%hold on
%plot(pat(4,:))  % to plot pattern P4
% etc... 

%% Simulation scheduler
% Without leakage - days 1 to 100
day(1:100) = 1;              % weekday (1 = normal day; 2 = weekend/holyday)
lk_tp(1:100) = 'n';          % leak type (c=constant; i=intermitent; n=none)
lk_tm(1:100) = 0;            % Starting time of leakage 
lk_arr(1:100) = {'no'};      % label to create the leakage array

% Leakage on node 10
day(101:200) = 1;
lk_tp(101:200) = 'c';
lk_tm(101:200) = 0;
lk_arr(101:200) = {'no10-di'}; 

% Leakage on node 11
day(201:300) = 1;
lk_tp(201:300) = 'c';
lk_tm(201:300) = 0;
lk_arr(201:300) = {'no11-di'}; 

% Leakage on node 12 
day(301:400) = 1;
lk_tp(301:400) = 'c';
lk_tm(301:400) = 0;
lk_arr(301:400) = {'no12-di'}; 

% Leakage on node 13
day(401:500) = 1;
lk_tp(401:500) = 'c';
lk_tm(401:500) = 0;
lk_arr(401:500) = {'no13-di'}; 

% Leakage on node 14
day(501:600) = 1;
lk_tp(501:600) = 'c';
lk_tm(501:600) = 0;
lk_arr(501:600) = {'no14-di'}; 

sz = length(lk_tm);

%% Running the model
% use this clause if you want to run some specific days for 
% debugging or configuration checks
% for c = [1, 220, 280, 330, 400, 450] 

%Use this line to run the whole schedule
for c = 1:sz 
     
    if (rem(c,50) == 0)   % Print some checkpoints on the command window
        fprintf('Day %d \n',c)
    end
    
    wkday = repmat(day(c),len,1);
    lktype = lk_tp(c);
    lktime = lk_tm(c);
    lk_id = char(lk_arr(c));
    
    switch lk_id 
        % Creates the 'leak-array' for supervised learning
        % Changes the model to create the desired leakage condition
        % Please set the labels for each case 
        case 'no'
            leak_array = [zeros(1,5)];
        case 'no10-di'
            leak_array = [1 zeros(1,4)]; 
            d.setNodeEmitterCoeff(10,0.3); %Sets the node emitter coefficient 
            % to 0.3 to produce leakage on node N10

        case 'no11-di'
            leak_array = [0 1 0 0 0]; 
            d.setNodeEmitterCoeff(11,0.3);

        case 'no12-di'
            leak_array = [0 0 1 0 0]; 
            d.setNodeEmitterCoeff(12,0.3);

        case 'no13-di'
            leak_array = [0 0 0 1 0]; 
            d.setNodeEmitterCoeff(13,0.3);

        case 'no14-di'
            leak_array = [0 0 0 0 1]; 
            d.setNodeEmitterCoeff(14,0.3);

        otherwise
            fprintf('ERRO')
    end
    
    % Randomize demand patterns to the consumption nodes
    % In this model, normal consumers are nodes N3-N6
    % The designed patterns for consumption nodes are P3-P6 (from the
    % EPANET model)and P7-P14 (Generated on lines 108-150)
    for j = 3:6    %Nodes N3-N6
        d.setNodeDemandPatternIndex(j,randi([3,14]));  % Patterns P3-P14
    end
    
    % Randomize demand patterns to the external consumer nodes
    % In this model, the outside consumers are nodes N15-N44
    % The designed patterns for outside consumers are P15-P20 (generated on
    % lines 152-163)
    for i = 15:44
        d.setNodeDemandPatternIndex(i,randi([15,20]));
        val = bdmin + (bdmax-bdmin) .* rand;
        d.setNodeBaseDemands(i,val);
    end

    % Execute the model
    dsct = dsct +1;             % increment the dataset counter
    Data{dsct,1} = runmodel(d,tv,tmin,cpm,cpv,leak_array,lktype,lktime,meas_nodes,tot_leak_nodes,meas_links,len,wkday);
        
    % Check for negative pressure (invalid condition)
    t = Data{dsct,1}(:,first:last);
    if any(t(:)<0)
        fprintf('ALERT: Negative pressure on dataset %d \n', dsct)
    end
    
    % Check average pressure on the inlet water meter
    pmi = d.getBinComputedNodePressure;
    avgPress(dsct,1) = mean(pmi(:,2));
    
    switch lk_id % reset leakage nodes
        case 'no'
        case 'no10-di'
            d.setNodeEmitterCoeff(10,0); % disables the leakage on node N10
        case 'no11-di'
            d.setNodeEmitterCoeff(11,0);
        case 'no12-di'
            d.setNodeEmitterCoeff(12,0);
        case 'no13-di'
            d.setNodeEmitterCoeff(13,0);
        case 'no14-di'
            d.setNodeEmitterCoeff(14,0); 
 
        otherwise
            fprintf('ERRO')   
    end

end



%% CSV export
if (gen_csv == 1)
    ddata = cell2mat(Data);
    
    cd C:\Users\Aluno\Documents\MATLAB\output   % choose the output folder
    
    name = sprintf('csv-cena%d-dist-x%d-seed%d-%dd-%dmin.csv',nc,mult,seed,sz,ta);
    csvwrite(name,ddata)
    
    % If you want, print a log file of the consumption patterns used in
    % the simulation program
    % name2 = sprintf('log-pad-cons-cena%d-dist-x%d-seed%d-%dd-%dmin.csv',nc,mult,seed,sz,ta);
    % csvwrite(name2,log_pc);

    cd C:\Users\Aluno\Documents\MATLAB\  % go back to your MATLAB default folder
    
end

beep

end
end
end

if (gen_csv == 1)
    csvwrite('Mat-adj-com-junc.csv',MCJ) % adjacency matrix
end

fprintf('Avg pressure on inlet water meter: %.3f mca \n', mean(avgPress))
disp('End')