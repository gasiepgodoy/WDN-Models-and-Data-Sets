%% Geração de dados - rede condomínio 4 - v2 (Quali - Cena 2)
close all
clear
clc
gen_csv = 1;

% Basic setup
nc = 2;                  % Scene ID number (for identification)
seeds = [ 11 33 157 269 535 981 1235 2685 3561 4004 ];  % seeds for random number generator
tas = [10 30 60 180 360 720];     % report intervals
mults = [1 5 10 25 50 100 200];    % multiplication factors for pipe length
diams = [50.80 ];  %   pipe diameter (50.8  = 2 pol)

for (v_seed=1:length(seeds))
seed = seeds(v_seed);
for (v_mults = 1:length(mults));
mult = mults(v_mults);
for (v_tas = 1:length(tas));
ta = tas(v_tas);
for (v_diams = 1:length(diams));
diam = diams(v_diams);

% Columns in the dataset to check for negative pressure
first = 21;     
last = 39;      

%Variables setup
rng(seed)       % selects a specific seed for the random number generator
dsct = 0;       % dataset counter
log_pc = [];    % log of the drawn consumption patterns
fprintf(datestr(now,'Inicio - dd-mm-yyyy HH:MM:SS \n'))
fprintf('Seed: %d, Multip: %d, Tempo aq: %d, Diam: %.2f\n',seed,mult,ta,diam);

% Load EPANET model 
d = epanet('rede-condominio-4-v2.inp');
d.solveCompleteHydraulics;



%% Pump adjustment (if needed)
d.setCurve(1,[80,80]) % padrão : [100,80]
d.runsCompleteSimulation;
d.BinUpdateClass;
d.getBinCurvesInfo;

%% Report time adjustment (automatic)

d.setBinTimeReportingStep(ta*60); 
d.setBinTimeHydraulicStep(ta*60); 
d.runsCompleteSimulation;
d.BinUpdateClass;

%% %% Pipe length/diameter adjustment

actdia = d.getLinkDiameter;
newdia = [actdia(1) ones(1,length(actdia)-1).*diam];
d.setLinkDiameter(newdia);

d.setLinkLength(3,100*mult);     %original: 100
d.setLinkLength(5,180*mult);     %original: 180
d.setLinkLength(6,50*mult);      %original: 50
d.setLinkLength(7,50*mult);      %original: 50
d.setLinkLength(8,150*mult);     %original: 150
d.setLinkLength(10,130*mult);     %original: 130
d.setLinkLength(12,83*mult);     %original: 83
d.setLinkLength(14,103*mult);    %original: 103
d.setLinkLength(16,130*mult);    %original: 130
d.setLinkLength(18,95*mult);     %original: 95
d.setLinkLength(20,120*mult);    %original: 120
d.setLinkLength(22,96*mult);     %original: 96
d.setLinkLength(24,120*mult);    %original: 120
d.setLinkLength(26,112*mult);    %original: 112
d.setLinkLength(28,85*mult);     %original: 85
d.setLinkLength(29,80*mult);     %original: 800
d.setLinkLength(30,72*mult);     %original: 72
d.setLinkLength(33,148*mult);    %original: 148
d.setLinkLength(34,100*mult);    %original: 100
d.setLinkLength(36,156*mult);    %original: 156
%d.BinUpdateClass;

%% Check number of elements
nos = d.getNodeCount; % Number of nodes -> pressure measurement
tre = d.getLinkCount; % Number of links (pipes) -> flow measurement
tmp = d.getBinComputedLinkFlow;
len = size(tmp,1);
clear tmp

%% Scene data (Config manually)
num_rnf = 1;                    % Number of fixed-level reservoirs 
nos_med = [2:1:20];             % Measurement nodes IDs (joint nodes + consumption units)
tr_med = [1 9 11 15 17 19 23 25 31 33 37 35 3 5 7 13 21 27 29]; % Pipes with flow measurement
nos_vaz = [21:1:38];                % Nodes for leak simulation
tot_nos_vaz = length(nos_vaz);      % total leakage nodes

%% Node coordinates
elev = d.getNodeElevations;   % Elevation of nodes
for i = 1:(nos-num_rnf)
coord = d.getNodeCoordinates(i); 
coordenadas{i} = [coord(1) coord(2) elev(i)];
end
clear i coord

%% Node mapping
cpm = node_map(nos_med,coordenadas,len);
cpv = node_map(nos_vaz,coordenadas,len);

%% Adjacency matrix (insert manually)
MCJ = [
    0	100	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
    100	0	180	0	0	131	0	0	0	0	0	0	0	0	0	0	0	0	0;
    0	180	0	102	86	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
    0	0	102	0	0	0	0	0	150	130	0	0	0	0	0	0	0	0	0;  
    0	0	86	0	0	0	0	0	0	0	106	130	94	0	0	0	0	0	0;
    0	131	0	0	0	0	120	0	0	0	0	0	0	98	120	0	0	0	0;
    0	0	0	0	0	120	0	165	0	0	0	0	0	0	0	0	0	158	100;
    0	0	0	0	0	0	165	0	0	0	0	0	0	0	0	74	150	0	0;
    0	0	0	150	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
    0	0	0	130	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	106	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	130	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	94	0	0	0	0	0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	98	0	0	0	0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	120	0	0	0	0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	74	0	0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	150	0	0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	158	0	0	0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	100	0	0	0	0	0	0	0	0	0	0	0	0

];

%% Timestamp creation
ttime = d.getTimeSimulationDuration;
step = d.getTimeReportingStep;
tv = double([0:step:ttime]');
tmin = double(step/60);
tot_horas = double(ttime/3600);
% To format as HH:MM:SS
hora = linspace(0,tot_horas,1+(ttime/step));
hora = hours(hora);
hora.Format = 'hh:mm';

%% Including 8 extra consumption patterns to enrich the model
% Follow the patterns available in the EPANET model
% In this case, there were 8 time points and 8 corresponding demand values

hrs = [1 5 8 12 15 18 21 24];
pts = [0.1 0.4 0.8 2 2.2 1.3 1.9 0.5];
[pstr, pv] = pat_interp(d, hrs, pts);
d.addPattern(pstr,pv); %P7

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
d.addPattern(pstr,pv); %P14

%% Additional demand patterns to simulate external consumption (P15 a P25)
% This loop generates randomized patterns within a predefined range [min,
% max] for demands


min = 1;
max = 3.5;
hrs = [1 4 7 10 13 16 19 21 24];
for i = 1:11
    pts = min + (max-min) .* rand(1,9);
    [pstr, pv] = pat_interp(d, hrs, pts);
    d.addPattern(pstr,pv);
end

%% Parameters to randomize base demands on external consumption nodes
bdmin = 1; % minimal base demand
bdmax = 3;  % maximum base demand

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
lk_arr(1:100) = {'noleak'};  % label to create the leakage array

% Leakage on node 21
day(101:200) = 1;
lk_tp(101:200) = 'c';
lk_tm(101:200) = 0;
lk_arr(101:200) = {'no21-di'}; 

% Leakage on node 22
day(201:300) = 1;
lk_tp(201:300) = 'c';
lk_tm(201:300) = 0;
lk_arr(201:300) = {'no22-di'}; 

% Leakage on node 23
day(301:400) = 1;
lk_tp(301:400) = 'c';
lk_tm(301:400) = 0;
lk_arr(301:400) = {'no23-di'}; 

% Leakage on node 24
day(401:500) = 1;
lk_tp(401:500) = 'c';
lk_tm(401:500) = 0;
lk_arr(401:500) = {'no24-di'}; 

% Leakage on node 25
day(501:600) = 1;
lk_tp(501:600) = 'c';
lk_tm(501:600) = 0;
lk_arr(501:600) = {'no25-di'}; 

% Leakage on node 26
day(601:700) = 1;
lk_tp(601:700) = 'c';
lk_tm(601:700) = 0;
lk_arr(601:700) = {'no26-di'}; 

% Leakage on node 27
day(701:800) = 1;
lk_tp(701:800) = 'c';
lk_tm(701:800) = 0;
lk_arr(701:800) = {'no27-di'}; 

% Leakage on node 28
day(801:900) = 1;
lk_tp(801:900) = 'c';
lk_tm(801:900) = 0;
lk_arr(801:900) = {'no28-di'}; 

% Leakage on node 29
day(901:1000) = 1;
lk_tp(901:1000) = 'c';
lk_tm(901:1000) = 0;
lk_arr(901:1000) = {'no29-di'}; 

% Leakage on node 30
day(1001:1100) = 1;
lk_tp(1001:1100) = 'c';
lk_tm(1001:1100) = 0;
lk_arr(1001:1100) = {'no30-di'}; 

% Leakage on node 31
day(1101:1200) = 1;
lk_tp(1101:1200) = 'c';
lk_tm(1101:1200) = 0;
lk_arr(1101:1200) = {'no31-di'}; 

% Leakage on node 32
day(1201:1300) = 1;
lk_tp(1201:1300) = 'c';
lk_tm(1201:1300) = 0;
lk_arr(1201:1300) = {'no32-di'}; 

% Leakage on node 33
day(1301:1400) = 1;
lk_tp(1301:1400) = 'c';
lk_tm(1301:1400) = 0;
lk_arr(1301:1400) = {'no33-di'}; 

% Leakage on node 34
day(1401:1500) = 1;
lk_tp(1401:1500) = 'c';
lk_tm(1401:1500) = 0;
lk_arr(1401:1500) = {'no34-di'}; 

% Leakage on node 35
day(1501:1600) = 1;
lk_tp(1501:1600) = 'c';
lk_tm(1501:1600) = 0;
lk_arr(1501:1600) = {'no35-di'}; 

% Leakage on node 36
day(1601:1700) = 1;
lk_tp(1601:1700) = 'c';
lk_tm(1601:1700) = 0;
lk_arr(1601:1700) = {'no36-di'}; 

% Leakage on node 37
day(1701:1800) = 1;
lk_tp(1701:1800) = 'c';
lk_tm(1701:1800) = 0;
lk_arr(1701:1800) = {'no37-di'}; 


% Leakage on node 38
day(1801:1900) = 1;
lk_tp(1801:1900) = 'c';
lk_tm(1801:1900) = 0;
lk_arr(1801:1900) = {'no38-di'}; 

sz = length(lk_tm);


%% Running the model
% use this clause if you want to run some specific days for 
% debugging or configuration checks
% for c = [1, 220, 280, 330, 400, 450] 

%Use this line to run the whole schedule
for c = 1:sz 
    
    if (rem(c,500) == 0)
        fprintf('Day %d \n',c)  % Print some checkpoints on the command window
    end
    
    wkday = repmat(day(c),len,1);
    lktype = lk_tp(c);
    lktime = lk_tm(c);
    lk_id = char(lk_arr(c));
    
    switch lk_id 
	    % Creates the 'leak-array' for supervised learning
        % Changes the model to create the desired leakage condition
        % Please set the labels for each case 
        case 'noleak'
            leak_array = [zeros(1,18)];
        
		case 'no21-di'
            leak_array = [1 zeros(1,17)]; 
            d.setNodeEmitterCoeff(21,0.3);%Sets the node emitter coefficient 
            % to 0.3 to produce leakage on node N21

        case 'no22-di'
            leak_array = [zeros(1,1) 1 zeros(1,16)]; 
            d.setNodeEmitterCoeff(22,0.3);

        case 'no23-di'
            leak_array = [zeros(1,2) 1 zeros(1,15)]; 
            d.setNodeEmitterCoeff(23,0.3);
 
        case 'no24-di'
            leak_array = [zeros(1,3) 1 zeros(1,14)]; 
            d.setNodeEmitterCoeff(24,0.3);

        case 'no25-di'
            leak_array = [zeros(1,4) 1 zeros(1,13)]; 
            d.setNodeEmitterCoeff(25,0.3);

        case 'no26-di'
            leak_array = [zeros(1,5) 1 zeros(1,12)]; 
            d.setNodeEmitterCoeff(26,0.3);

        case 'no27-di'
            leak_array = [zeros(1,6) 1 zeros(1,11)]; 
            d.setNodeEmitterCoeff(27,0.3);

        case 'no28-di'
            leak_array = [zeros(1,7) 1 zeros(1,10)]; 
            d.setNodeEmitterCoeff(28,0.3);

        case 'no29-di'
            leak_array = [zeros(1,8) 1 zeros(1,9)]; 
            d.setNodeEmitterCoeff(29,0.3);

        case 'no30-di'
            leak_array = [zeros(1,9) 1 zeros(1,8)]; 
            d.setNodeEmitterCoeff(30,0.3);
  
        case 'no31-di'
            leak_array = [zeros(1,10) 1 zeros(1,7)]; 
            d.setNodeEmitterCoeff(31,0.3);
          
        case 'no32-di'
            leak_array = [zeros(1,11) 1 zeros(1,6)]; 
            d.setNodeEmitterCoeff(32,0.3);
 
        case 'no33-di'
            leak_array = [zeros(1,12) 1 zeros(1,5)]; 
            d.setNodeEmitterCoeff(33,0.3);
            
        case 'no34-di'
            leak_array = [zeros(1,13) 1 zeros(1,4)]; 
            d.setNodeEmitterCoeff(34,0.3);
 
        case 'no35-di'
            leak_array = [zeros(1,14) 1 zeros(1,3)]; 
            d.setNodeEmitterCoeff(35,0.3);
 
        case 'no36-di'
            leak_array = [zeros(1,15) 1 zeros(1,2)]; 
            d.setNodeEmitterCoeff(36,0.3);
            
        case 'no37-di'
            leak_array = [zeros(1,16) 1 0]; 
            d.setNodeEmitterCoeff(37,0.3);
             
        case 'no38-di'
            leak_array = [zeros(1,17) 1 ]; 
            d.setNodeEmitterCoeff(38,0.3);
                     
        otherwise
            fprintf('ERRO')
    end
    
    % Randomize demand patterns to the consumption nodes
    % In this model, normal consumers are nodes N3-N13
    % The designed patterns for consumption nodes are P3-P6 (from the
    % EPANET model)and P7-P14
    for j = 3:13  % Nodes N3-N13
        pc = randi([3,14]);  %Patterns P3-P14
        d.setNodeDemandPatternIndex(j,pc);
        log_pc(dsct+1,j) = pc;
    end
    
    % Randomize demand patterns to the external consumer nodes
    % In this model, the outside consumers are nodes N40-N69
    % The designed patterns for outside consumers are P15-P20 (generated on
    % lines 152-163)
    for i = 40:69
        d.setNodeDemandPatternIndex(i,randi([15,20]));
        val = bdmin + (bdmax-bdmin) .* rand;
        d.setNodeBaseDemands(i,val);
    end
    
    % Execute the model
    dsct = dsct +1;             % increment the dataset counter
    Data{dsct,1} = runmodel(d,tv,tmin,cpm,cpv,leak_array,lktype,lktime,nos_med,tot_nos_vaz,tr_med,len,wkday);
    
	
    % Check the average pressure on the inlet water meter
    % pmi = d.getBinComputedNodePressure;
    % avgPress(dsct,1) = mean(pmi(:,2));
    % fprintf('Avg. inlet pressure: %.3f mca \n', mean(avgPress))
    
    % Check for negative pressure (invalid condition)
    t = Data{dsct,1}(:,first:last);
    if any(t(:)<0)
        fprintf('ALERT: Negative pressure on dataset %d \n', dsct)
    end
    
    switch lk_id  % reset leakage nodes
        case 'noleak'
            
        case 'no20-di'
            d.setNodeEmitterCoeff(20,0); % disables the leakage on node N20

        case 'no21-di'
            d.setNodeEmitterCoeff(21,0);
            
        case 'no22-di'
            d.setNodeEmitterCoeff(22,0);
             
        case 'no23-di'
            d.setNodeEmitterCoeff(23,0);
            
        case 'no24-di'
            d.setNodeEmitterCoeff(24,0);

        case 'no25-di'
            d.setNodeEmitterCoeff(25,0);

        case 'no26-di'
            d.setNodeEmitterCoeff(26,0);
            
        case 'no27-di'
            d.setNodeEmitterCoeff(27,0);
 
        case 'no28-di'
            d.setNodeEmitterCoeff(28,0);
             
        case 'no29-di'
            d.setNodeEmitterCoeff(29,0);
 
        case 'no30-di'
            d.setNodeEmitterCoeff(30,0);
            
        case 'no31-di'
            d.setNodeEmitterCoeff(31,0);
            
        case 'no32-di'
            d.setNodeEmitterCoeff(32,0);
              
        case 'no33-di'
            d.setNodeEmitterCoeff(33,0);
             
        case 'no34-di'
            d.setNodeEmitterCoeff(34,0);
            
        case 'no35-di'
            d.setNodeEmitterCoeff(35,0);

        case 'no36-di'
            d.setNodeEmitterCoeff(36,0);
            
        case 'no37-di'
            d.setNodeEmitterCoeff(37,0);
             
        case 'no38-di'
            d.setNodeEmitterCoeff(38,0);
             
        otherwise
            fprintf('ERROR')   
    end
        
end




%% CSV export
if (gen_csv == 1)
    
    cd C:\Users\Aluno\Documents\MATLAB\output   % choose the output folder

    ddata = cell2mat(Data);
    nome = sprintf('csv-cena%d-dist-x%d-seed%d-%dd-%dmin.csv',nc,mult,seed,sz,ta);
    csvwrite(nome,ddata)

	cd C:\Users\Aluno\Documents\MATLAB\  % go back to your MATLAB default folder
	
end
beep
end
end
end
end

if (gen_csv == 1)
    csvwrite('Mat-adj-com-junc.csv',MCJ) % adjacency matrix
end

fprintf('Avg pressure on inlet water meter: %.3f mca \n', mean(avgPress))
disp('End')
