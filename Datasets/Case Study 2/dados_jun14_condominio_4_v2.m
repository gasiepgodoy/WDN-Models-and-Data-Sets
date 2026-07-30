%% Geração de dados - rede condomínio 4 - v2 (Quali - Cena 2)
close all
clear
clc
seed = 11;
rng(seed)         % fixa a seed do gerador de números pseudo-aleatórios
dsct = 0;       % variável para contar quantos datasets foram gerados
log_pc = [];    % variável de log dos padrões sorteados

datestr(now,'dd-mm-yyyy HH:MM:SS')

%Carregar o modelo EPANET
d = epanet('rede-condominio-4-v2.inp');
d.solveCompleteHydraulics;
%Inclui as medições nas juntas
%Usa o novo mapeamento de vazamento, só 3 colunas
%Modelo resumido - 1 laço for para todas as simulações

%% Altura dos nós
%d.setNodeElevations(12,35);  %altera a cota do nó N12

%% Ajuste da bomba
%d.setCurve(1,[480,20])   % [vazão,carga hidráulica] - distância longa
%d.setCurve(1,[80,25]) % distância curta
%d.runsCompleteSimulation;
%d.BinUpdateClass;
%d.getBinCurvesInfo

%% Comprimento dos trechos
mult = 10;                   %multiplicador
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

%% Verificar o número de elementos
nos = d.getNodeCount; %Pressão nos nós
tre = d.getLinkCount; %Vazão nos trechos
tmp = d.getBinComputedLinkFlow;
len = length(tmp);
clear tmp

%% Dados do cenário (Configurar manualmente)
num_rnf = 1;                    % Número de reservatórios de nível fixo
nos_med = [2:1:20];             % nós de medição (juntas + pontos de consumo)
tr_med = [1 9 11 15 17 19 23 25 31 33 37 35 3 5 7 13 21 27 29]; % trechos de medição de vazão
nos_vaz = [21:1:38];                % nós de vazamento
tot_nos_vaz = length(nos_vaz);      % total de pontos de vazamento

%% Coleta das coordenadas dos nós
elev = d.getNodeElevations;
for i = 1:(nos-num_rnf)
coord = d.getNodeCoordinates(i); 
coordenadas{i} = [coord(1) coord(2) elev(i)];
end
clear i coord

%% Mapeamento dos pontos de medição e vazamento
cpm = node_map(nos_med,coordenadas,len);
cpv = node_map(nos_vaz,coordenadas,len);

%% Matriz de adjacência (manual) 
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

%% Criação da estampa de tempo
ttime = d.getTimeSimulationDuration;
step = d.getTimeHydraulicStep;
tv = double([0:step:ttime]');
tmin = double(step/60);
tot_horas = double(ttime/3600);
% Para colocar no formato HH:MM:SS
hora = linspace(0,tot_horas,1+(ttime/step));
hora = hours(hora);
hora.Format = 'hh:mm';

%% Criação de mais 8 padrões de consumo para randomizar
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

%% Padrões de consumo adicionais para o consumo externo (P15 a P25)
min = 0;
max = 3.5;
hrs = [1 4 7 10 13 16 19 21 24];
for i = 1:11
    pts = min + (max-min) .* rand(1,9);
    [pstr, pv] = pat_interp(d, hrs, pts);
    d.addPattern(pstr,pv);
end

%% Parâmetros para variar o consumo-base nos nós externos
bdmin = 1;
bdmax = 2;

%% Programa de simulação
% 1 a 10 - sem vazamento
day(1:100) = 1;              % dia da semana (1 = dia de semana; 2 = fds)
lk_tp(1:100) = 'n';          % tipo de vazamento (c=cte.; i=intermitente; n=nenhum)
lk_tm(1:100) = 0;            % hora do vazamento 
lk_arr(1:100) = {'noleak'};  % id do array de vazamentos

% 11 a 15 - vazamento no nó 21 - dia inteiro
day(101:150) = 1;
lk_tp(101:150) = 'c';
lk_tm(101:150) = 0;
lk_arr(101:150) = {'no21-di'}; 

% 16 a 20 - vazamento no nó 21 - meio dia
day(151:200) = 1;
lk_tp(151:200) = 'i';
lk_tm(151:200) = 12;
lk_arr(151:200) = {'no21-md'};

% 21 a 25 - vazamento no nó 22 - dia inteiro
day(201:250) = 1;
lk_tp(201:250) = 'c';
lk_tm(201:250) = 0;
lk_arr(201:250) = {'no22-di'}; 

% 26 a 30 - vazamento no nó 22 - meio dia
day(251:300) = 1;
lk_tp(251:300) = 'i';
lk_tm(251:300) = 12;
lk_arr(251:300) = {'no22-md'};

% 31 a 35 - vazamento no nó 23 - dia inteiro
day(301:350) = 1;
lk_tp(301:350) = 'c';
lk_tm(301:350) = 0;
lk_arr(301:350) = {'no23-di'}; 

% 36 a 40 - vazamento no nó 23 - meio dia
day(351:400) = 1;
lk_tp(351:400) = 'i';
lk_tm(351:400) = 12;
lk_arr(351:400) = {'no23-md'};

% 41 a 45 - vazamento no nó 24 - dia inteiro
day(401:450) = 1;
lk_tp(401:450) = 'c';
lk_tm(401:450) = 0;
lk_arr(401:450) = {'no24-di'}; 

% 46 a 50 - vazamento no nó 24 - meio dia
day(451:500) = 1;
lk_tp(451:500) = 'i';
lk_tm(451:500) = 12;
lk_arr(451:500) = {'no24-md'};

% 51 a 55 - vazamento no nó 25 - dia inteiro
day(501:550) = 1;
lk_tp(501:550) = 'c';
lk_tm(501:550) = 0;
lk_arr(501:550) = {'no25-di'}; 

% 56 a 60 - vazamento no nó 25 - meio dia 
day(551:600) = 1;
lk_tp(551:600) = 'i';
lk_tm(551:600) = 12;
lk_arr(551:600) = {'no25-md'};

% 61 a 65 - vazamento no nó 26 - dia inteiro
day(601:650) = 1;
lk_tp(601:650) = 'c';
lk_tm(601:650) = 0;
lk_arr(601:650) = {'no26-di'}; 

% 66 a 70 - vazamento no nó 26 - meio dia
day(651:700) = 1;
lk_tp(651:700) = 'i';
lk_tm(651:700) = 12;
lk_arr(651:700) = {'no26-md'};

% 71 a 75 - vazamento no nó 27 - dia inteiro
day(701:750) = 1;
lk_tp(701:750) = 'c';
lk_tm(701:750) = 0;
lk_arr(701:750) = {'no27-di'}; 

% 76 a 80 - vazamento no nó 27 - meio dia
day(751:800) = 1;
lk_tp(751:800) = 'i';
lk_tm(751:800) = 12;
lk_arr(751:800) = {'no27-md'};

% 81 a 85 - vazamento no nó 28 - dia inteiro
day(801:850) = 1;
lk_tp(801:850) = 'c';
lk_tm(801:850) = 0;
lk_arr(801:850) = {'no28-di'}; 

% 86 a 90 - vazamento no nó 28 - meio dia
day(851:900) = 1;
lk_tp(851:900) = 'i';
lk_tm(851:900) = 12;
lk_arr(851:900) = {'no28-md'};

% 91 a 95 - vazamento no nó 29 - dia inteiro
day(901:950) = 1;
lk_tp(901:950) = 'c';
lk_tm(901:950) = 0;
lk_arr(901:950) = {'no29-di'}; 

% 96 a 100 - vazamento no nó 29 - meio dia
day(951:1000) = 1;
lk_tp(951:1000) = 'i';
lk_tm(951:1000) = 12;
lk_arr(951:1000) = {'no29-md'};

% 101 a 105 - vazamento no nó 30 - dia inteiro
day(1001:1050) = 1;
lk_tp(1001:1050) = 'c';
lk_tm(1001:1050) = 0;
lk_arr(1001:1050) = {'no30-di'}; 

% 106 a 110 - vazamento no nó 30 - meio dia
day(1051:1100) = 1;
lk_tp(1051:1100) = 'i';
lk_tm(1051:1100) = 12;
lk_arr(1051:1100) = {'no30-md'};

% 111 a 115 - vazamento no nó 31 - dia inteiro
day(1101:1150) = 1;
lk_tp(1101:1150) = 'c';
lk_tm(1101:1150) = 0;
lk_arr(1101:1150) = {'no31-di'}; 

% 116 a 120 - vazamento no nó 31 - meio dia
day(1151:1200) = 1;
lk_tp(1151:1200) = 'i';
lk_tm(1151:1200) = 12;
lk_arr(1151:1200) = {'no31-md'};

% 121 a 125 - vazamento no nó 32 - dia inteiro
day(1201:1250) = 1;
lk_tp(1201:1250) = 'c';
lk_tm(1201:1250) = 0;
lk_arr(1201:1250) = {'no32-di'}; 

% 126 a 130 - vazamento no nó 32 - meio dia
day(1251:1300) = 1;
lk_tp(1251:1300) = 'i';
lk_tm(1251:1300) = 12;
lk_arr(1251:1300) = {'no32-md'};

% 131 a 135 - vazamento no nó 33 - dia inteiro
day(1301:1350) = 1;
lk_tp(1301:1350) = 'c';
lk_tm(1301:1350) = 0;
lk_arr(1301:1350) = {'no33-di'}; 

% 136 a 140 - vazamento no nó 33 - meio dia
day(1351:1400) = 1;
lk_tp(1351:1400) = 'i';
lk_tm(1351:1400) = 12;
lk_arr(1351:1400) = {'no33-md'};

% 141 a 145 - vazamento no nó 34 - dia inteiro
day(1401:1450) = 1;
lk_tp(1401:1450) = 'c';
lk_tm(1401:1450) = 0;
lk_arr(1401:1450) = {'no34-di'}; 

% 146 a 150 - vazamento no nó 34 - meio dia
day(1451:1500) = 1;
lk_tp(1451:1500) = 'i';
lk_tm(1451:1500) = 12;
lk_arr(1451:1500) = {'no34-md'};

% 151 a 155 - vazamento no nó 35 - dia inteiro
day(1501:1550) = 1;
lk_tp(1501:1550) = 'c';
lk_tm(1501:1550) = 0;
lk_arr(1501:1550) = {'no35-di'}; 

% 156 a 160 - vazamento no nó 35 - meio dia
day(1551:1600) = 1;
lk_tp(1551:1600) = 'i';
lk_tm(1551:1600) = 12;
lk_arr(1551:1600) = {'no35-md'};

% 161 a 165 - vazamento no nó 36 - dia inteiro
day(1601:1650) = 1;
lk_tp(1601:1650) = 'c';
lk_tm(1601:1650) = 0;
lk_arr(1601:1650) = {'no36-di'}; 

% 166 a 170 - vazamento no nó 36 - meio dia
day(1651:1700) = 1;
lk_tp(1651:1700) = 'i';
lk_tm(1651:1700) = 12;
lk_arr(1651:1700) = {'no36-md'};

% 171 a 175 - vazamento no nó 37 - dia inteiro
day(1701:1750) = 1;
lk_tp(1701:1750) = 'c';
lk_tm(1701:1750) = 0;
lk_arr(1701:1750) = {'no37-di'}; 

% 176 a 180 - vazamento no nó 37 - meio dia
day(1751:1800) = 1;
lk_tp(1751:1800) = 'i';
lk_tm(1751:1800) = 12;
lk_arr(1751:1800) = {'no37-md'};

% 181 a 185 - vazamento no nó 38 - dia inteiro
day(1801:1850) = 1;
lk_tp(1801:1850) = 'c';
lk_tm(1801:1850) = 0;
lk_arr(1801:1850) = {'no38-di'}; 

% 186 a 190 - vazamento no nó 38 - meio dia
day(1851:1900) = 1;
lk_tp(1851:1900) = 'i';
lk_tm(1851:1900) = 12;
lk_arr(1851:1900) = {'no38-md'};

sz = length(lk_tm);

%% Execução do modelo 

for c = [1, 220, 330]
%for c = 1:sz % ajustar de acordo com o número de simulações
    
    if (rem(c,50) == 0)
        fprintf('Day %d \n',c)
    end
    
    wkday = repmat(day(c),len,1);
    lktype = lk_tp(c);
    lktime = lk_tm(c);
    lk_id = char(lk_arr(c));
    
    switch lk_id % definir o leak_array e ajustar a demanda
        case 'noleak'
            leak_array = [zeros(1,18)];
        case 'no21-di'
            leak_array = [1 zeros(1,17)]; 
            d.setNodeBaseDemands(21,2);         % consumo-base
            d.setNodeDemandPatternIndex(21,2);  % dia inteiro
        case 'no21-md'
            leak_array = [1 zeros(1,17)]; 
            d.setNodeBaseDemands(21,2);         % consumo-base
            d.setNodeDemandPatternIndex(21,1);  % meio dia 
        case 'no22-di'
            leak_array = [zeros(1,1) 1 zeros(1,16)]; 
            d.setNodeBaseDemands(22,2);         % consumo-base
            d.setNodeDemandPatternIndex(22,2);  % dia inteiro
        case 'no22-md'
            leak_array = [zeros(1,1) 1 zeros(1,16)];
            d.setNodeBaseDemands(22,2);         % consumo-base
            d.setNodeDemandPatternIndex(22,1);  % meio dia  
        case 'no23-di'
            leak_array = [zeros(1,2) 1 zeros(1,15)]; 
            d.setNodeBaseDemands(23,2);         % consumo-base
            d.setNodeDemandPatternIndex(23,2);  % dia inteiro
        case 'no23-md'
            leak_array = [zeros(1,2) 1 zeros(1,15)];
            d.setNodeBaseDemands(23,2);         % consumo-base
            d.setNodeDemandPatternIndex(23,1);  % meio dia  
        case 'no24-di'
            leak_array = [zeros(1,3) 1 zeros(1,14)]; 
            d.setNodeBaseDemands(24,2);         % consumo-base
            d.setNodeDemandPatternIndex(24,2);  % dia inteiro
        case 'no24-md'
            leak_array = [zeros(1,3) 1 zeros(1,14)];
            d.setNodeBaseDemands(24,2);         % consumo-base
            d.setNodeDemandPatternIndex(24,1);  % meio dia  
        case 'no25-di'
            leak_array = [zeros(1,4) 1 zeros(1,13)]; 
            d.setNodeBaseDemands(25,2);         % consumo-base
            d.setNodeDemandPatternIndex(25,2);  % dia inteiro
        case 'no25-md'
            leak_array = [zeros(1,4) 1 zeros(1,13)];
            d.setNodeBaseDemands(25,2);         % consumo-base
            d.setNodeDemandPatternIndex(25,1);  % meio dia  
        case 'no26-di'
            leak_array = [zeros(1,5) 1 zeros(1,12)]; 
            d.setNodeBaseDemands(26,2);         % consumo-base
            d.setNodeDemandPatternIndex(26,2);  % dia inteiro
        case 'no26-md'
            leak_array = [zeros(1,5) 1 zeros(1,12)];
            d.setNodeBaseDemands(26,2);         % consumo-base
            d.setNodeDemandPatternIndex(26,1);  % meio dia  
        case 'no27-di'
            leak_array = [zeros(1,6) 1 zeros(1,11)]; 
            d.setNodeBaseDemands(27,2);         % consumo-base
            d.setNodeDemandPatternIndex(27,2);  % dia inteiro
        case 'no27-md'
            leak_array = [zeros(1,6) 1 zeros(1,11)];
            d.setNodeBaseDemands(27,2);         % consumo-base
            d.setNodeDemandPatternIndex(27,1);  % meio dia  
        case 'no28-di'
            leak_array = [zeros(1,7) 1 zeros(1,10)]; 
            d.setNodeBaseDemands(28,2);         % consumo-base
            d.setNodeDemandPatternIndex(28,2);  % dia inteiro
        case 'no28-md'
            leak_array = [zeros(1,7) 1 zeros(1,10)];
            d.setNodeBaseDemands(28,2);         % consumo-base
            d.setNodeDemandPatternIndex(28,1);  % meio dia  
        case 'no29-di'
            leak_array = [zeros(1,8) 1 zeros(1,9)]; 
            d.setNodeBaseDemands(29,2);         % consumo-base
            d.setNodeDemandPatternIndex(29,2);  % dia inteiro
        case 'no29-md'
            leak_array = [zeros(1,8) 1 zeros(1,9)];
            d.setNodeBaseDemands(29,2);         % consumo-base
            d.setNodeDemandPatternIndex(29,1);  % meio dia  
        case 'no30-di'
            leak_array = [zeros(1,9) 1 zeros(1,8)]; 
            d.setNodeBaseDemands(30,2);         % consumo-base
            d.setNodeDemandPatternIndex(30,2);  % dia inteiro
        case 'no30-md'
            leak_array = [zeros(1,9) 1 zeros(1,8)];
            d.setNodeBaseDemands(30,2);         % consumo-base
            d.setNodeDemandPatternIndex(30,1);  % meio dia  
        case 'no31-di'
            leak_array = [zeros(1,10) 1 zeros(1,7)]; 
            d.setNodeBaseDemands(31,2);         % consumo-base
            d.setNodeDemandPatternIndex(31,2);  % dia inteiro
        case 'no31-md'
            leak_array = [zeros(1,10) 1 zeros(1,7)];
            d.setNodeBaseDemands(31,2);         % consumo-base
            d.setNodeDemandPatternIndex(31,1);  % meio dia            
        case 'no32-di'
            leak_array = [zeros(1,11) 1 zeros(1,6)]; 
            d.setNodeBaseDemands(32,2);         % consumo-base
            d.setNodeDemandPatternIndex(32,2);  % dia inteiro
        case 'no32-md'
            leak_array = [zeros(1,11) 1 zeros(1,6)];
            d.setNodeBaseDemands(32,2);         % consumo-base
            d.setNodeDemandPatternIndex(32,1);  % meio dia  
        case 'no33-di'
            leak_array = [zeros(1,12) 1 zeros(1,5)]; 
            d.setNodeBaseDemands(33,2);         % consumo-base
            d.setNodeDemandPatternIndex(33,2);  % dia inteiro
        case 'no33-md'
            leak_array = [zeros(1,12) 1 zeros(1,5)];
            d.setNodeBaseDemands(33,2);         % consumo-base
            d.setNodeDemandPatternIndex(33,1);  % meio dia              
        case 'no34-di'
            leak_array = [zeros(1,13) 1 zeros(1,4)]; 
            d.setNodeBaseDemands(34,2);         % consumo-base
            d.setNodeDemandPatternIndex(34,2);  % dia inteiro
        case 'no34-md'
            leak_array = [zeros(1,13) 1 zeros(1,4)];
            d.setNodeBaseDemands(34,2);         % consumo-base
            d.setNodeDemandPatternIndex(34,1);  % meio dia  
        case 'no35-di'
            leak_array = [zeros(1,14) 1 zeros(1,3)]; 
            d.setNodeBaseDemands(35,2);         % consumo-base
            d.setNodeDemandPatternIndex(35,2);  % dia inteiro
        case 'no35-md'
            leak_array = [zeros(1,14) 1 zeros(1,3)];
            d.setNodeBaseDemands(35,2);         % consumo-base
            d.setNodeDemandPatternIndex(35,1);  % meio dia  
        case 'no36-di'
            leak_array = [zeros(1,15) 1 zeros(1,2)]; 
            d.setNodeBaseDemands(36,2);         % consumo-base
            d.setNodeDemandPatternIndex(36,2);  % dia inteiro
        case 'no36-md'
            leak_array = [zeros(1,15) 1 zeros(1,2)];
            d.setNodeBaseDemands(36,2);         % consumo-base
            d.setNodeDemandPatternIndex(36,1);  % meio dia              
        case 'no37-di'
            leak_array = [zeros(1,16) 1 0]; 
            d.setNodeBaseDemands(37,2);         % consumo-base
            d.setNodeDemandPatternIndex(37,2);  % dia inteiro
        case 'no37-md'
            leak_array = [zeros(1,16) 1 0];
            d.setNodeBaseDemands(37,2);         % consumo-base
            d.setNodeDemandPatternIndex(37,1);  % meio dia              
        case 'no38-di'
            leak_array = [zeros(1,17) 1 ]; 
            d.setNodeBaseDemands(38,2);         % consumo-base
            d.setNodeDemandPatternIndex(38,2);  % dia inteiro
        case 'no38-md'
            leak_array = [zeros(1,17) 1 ];
            d.setNodeBaseDemands(38,2);         % consumo-base
            d.setNodeDemandPatternIndex(38,1);  % meio dia              
            
        otherwise
            fprintf('ERRO')
    end
    
    %Sortear padrões para os pontos de consumo
    for j = 3:13
        pc = randi([3,14]);
        d.setNodeDemandPatternIndex(j,pc);
        log_pc(dsct+1,j) = pc;
    end
    
    %Sortear padrões de consumo dos nós externos
    for i = 40:69
        d.setNodeDemandPatternIndex(i,randi([15,20]));
        val = bdmin + (bdmax-bdmin) .* rand;
        d.setNodeBaseDemands(i,val);
    end
    
    %Rodar o modelo
    dsct = dsct +1;             % incrementar o contador de datasets
    Data{dsct,1} = runmodel(d,tv,tmin,cpm,cpv,leak_array,lktype,lktime,nos_med,tot_nos_vaz,tr_med,len,wkday);
    Datasimp{dsct,1} = runmodel_simple(d,tv,tmin,cpm,cpv,leak_array,lktype,lktime,nos_med,tot_nos_vaz,tr_med,len,wkday);
    switch lk_id % reset dos vazamentos
        case 'noleak'
            
        case 'no20-di'
            d.setNodeBaseDemands(20,0); % zera o consumo
        case 'no20-md'
            d.setNodeBaseDemands(20,0); 
        case 'no21-di'
            d.setNodeBaseDemands(21,0); % zera o consumo
        case 'no21-md'
            d.setNodeBaseDemands(21,0);             
        case 'no22-di'
            d.setNodeBaseDemands(22,0); % zera o consumo
        case 'no22-md'
            d.setNodeBaseDemands(22,0);             
        case 'no23-di'
            d.setNodeBaseDemands(23,0); % zera o consumo
        case 'no23-md'
            d.setNodeBaseDemands(23,0);             
        case 'no24-di'
            d.setNodeBaseDemands(24,0); % zera o consumo
        case 'no24-md'
            d.setNodeBaseDemands(24,0); 
        case 'no25-di'
            d.setNodeBaseDemands(25,0); % zera o consumo
        case 'no25-md'
            d.setNodeBaseDemands(25,0); 
        case 'no26-di'
            d.setNodeBaseDemands(26,0); % zera o consumo
        case 'no26-md'
            d.setNodeBaseDemands(26,0);             
        case 'no27-di'
            d.setNodeBaseDemands(27,0); % zera o consumo
        case 'no27-md'
            d.setNodeBaseDemands(27,0); 
        case 'no28-di'
            d.setNodeBaseDemands(28,0); % zera o consumo
        case 'no28-md'
            d.setNodeBaseDemands(28,0);             
        case 'no29-di'
            d.setNodeBaseDemands(29,0); % zera o consumo
        case 'no29-md'
            d.setNodeBaseDemands(29,0); 
        case 'no30-di'
            d.setNodeBaseDemands(30,0); % zera o consumo
        case 'no30-md'
            d.setNodeBaseDemands(30,0);             
        case 'no31-di'
            d.setNodeBaseDemands(31,0); % zera o consumo
        case 'no31-md'
            d.setNodeBaseDemands(31,0);             
        case 'no32-di'
            d.setNodeBaseDemands(32,0); % zera o consumo
        case 'no32-md'
            d.setNodeBaseDemands(32,0);              
        case 'no33-di'
            d.setNodeBaseDemands(33,0); % zera o consumo
        case 'no33-md'
            d.setNodeBaseDemands(33,0);              
        case 'no34-di'
            d.setNodeBaseDemands(34,0); % zera o consumo
        case 'no34-md'
            d.setNodeBaseDemands(34,0);              
        case 'no35-di'
            d.setNodeBaseDemands(35,0); % zera o consumo
        case 'no35-md'
            d.setNodeBaseDemands(35,0);  
        case 'no36-di'
            d.setNodeBaseDemands(36,0); % zera o consumo
        case 'no36-md'
            d.setNodeBaseDemands(36,0);              
        case 'no37-di'
            d.setNodeBaseDemands(37,0); % zera o consumo
        case 'no37-md'
            d.setNodeBaseDemands(37,0);              
        case 'no38-di'
            d.setNodeBaseDemands(38,0); % zera o consumo
        case 'no38-md'
            d.setNodeBaseDemands(38,0);              
        otherwise
            fprintf('ERRO')   
    end
    

    
end



%% Checagem de pressão negativa
init = length(nos_med)+2;
ending = init + length(nos_med)-1;
for i = 1:length(Data)
    t = Data{i}(:,21:39);
    if any(t<0)
        fprintf('ALERTA: Pressão negativa no dataset %d \n', i)
    end
end

%% Exportação para CSV
nc = 2; % número da cena

%mkdir output
%cd C:\Users\Aluno\Documents\MATLAB\output
%csvwrite('Mat-adj-com-junc.csv',MCJ) % matriz de adjacência
%csvwrite('log-pad-cons.csv',log_pc) % log dos padrões de consumo

ddata = cell2mat(Data);
ddatasimp = cell2mat(Datasimp);
nome = sprintf('csv-cena%d-dist-x%d-seed%d-%dd.csv',nc,mult,seed,sz);
csvwrite(nome,ddata)
%csvwrite('csv-cena2-aum-seed11-1900d.csv',nc,seed,ddata)
%csvwrite('csv-simplif-cena2-aum-seed11-1900d.csv',ddatasimp)
%datestr(now,'dd-mm-yyyy HH:MM:SS')

% for g = 1:dsct
%     nome = sprintf('%s_%d.csv','data',g);
%     csvwrite(nome,Data{g})
% end
% cd C:\Users\Aluno\Documents\MATLAB\
beep
disp('Fim')

