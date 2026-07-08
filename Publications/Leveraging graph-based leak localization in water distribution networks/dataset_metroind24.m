%% Geração de dados - rede condomínio 4 - v2 (Quali - Cena 2)
close all
clear
clc
gerar_csv = 1;

% Configurações importantes
nc = 2;                  % número da cena
seeds = [ 11 33 157 269 535 981 1235 2685 3561 4004 ];            % Seeds a executar 17h por seed
%seeds = [ 11 33 158 269 535  ];
tas = [10 30 60 180 360 720];     % tempos de aquisição (minutos)
mults = [1 5 10 25 50 100 200];    % multiplicadores - comprimento dos trechos
diams = [50.80 ];  %  3 pol / 2 pol / 1 pol [76.2 50.8 25.4 ]

for (v_seed=1:length(seeds))
seed = seeds(v_seed);
for (v_mults = 1:length(mults));
mult = mults(v_mults);
for (v_tas = 1:length(tas));
ta = tas(v_tas);
for (v_diams = 1:length(diams));
diam = diams(v_diams);

% intervalo de colunas para checar pressão negativa
first = 21;     
last = 39;      

%Inicialização de variáveis
rng(seed)       % fixa a seed do gerador de números pseudo-aleatórios
dsct = 0;       % variável para contar quantos datasets foram gerados
log_pc = [];    % variável de log dos padrões sorteados
fprintf(datestr(now,'Inicio - dd-mm-yyyy HH:MM:SS \n'))
fprintf('Seed: %d, Multip: %d, Tempo aq: %d, Diam: %.2f\n',seed,mult,ta,diam);

%Carregar o modelo EPANET
d = epanet('rede-condominio-4-v2.inp');
d.solveCompleteHydraulics;
%Inclui as medições nas juntas
%Usa o novo mapeamento de vazamento, só 3 colunas
%Modelo resumido - 1 laço for para todas as simulações
%Nomenclatura dinâmica conforme os parâmetros

%% Altura dos nós
%d.setNodeElevations(12,35);  %altera a cota do nó N12

%% Ajuste da bomba
%d.setCurve(1,[480,20])   % [vazão,carga hidráulica] - distância longa
d.setCurve(1,[80,80]) % padrão : [100,80]
d.runsCompleteSimulation;
d.BinUpdateClass;
%d.getBinCurvesInfo

%% Ajuste da amostragem

d.setBinTimeReportingStep(ta*60); 
d.setBinTimeHydraulicStep(ta*60); 
d.runsCompleteSimulation;
d.BinUpdateClass;

%% Comprimento dos trechos
% n_trechos = 10;
% for q = 1:n_trechos
% tmp = d.getLinkLength(q);
% d.setLinkLength(q,tmp*mult);
% end
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

%% Verificar o número de elementos
nos = d.getNodeCount; %Pressão nos nós
tre = d.getLinkCount; %Vazão nos trechos
tmp = d.getBinComputedLinkFlow;
len = size(tmp,1);
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
step = d.getTimeReportingStep;
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
min = 1;
max = 3.5;
hrs = [1 4 7 10 13 16 19 21 24];
for i = 1:11
    pts = min + (max-min) .* rand(1,9);
    [pstr, pv] = pat_interp(d, hrs, pts);
    d.addPattern(pstr,pv);
end

%% Parâmetros para variar o consumo-base nos nós externos
bdmin = 1;
bdmax = 3;

%% Programa de simulação
% 1 a 100 - sem vazamento
day(1:100) = 1;              % dia da semana (1 = dia de semana; 2 = fds)
lk_tp(1:100) = 'n';          % tipo de vazamento (c=cte.; i=intermitente; n=nenhum)
lk_tm(1:100) = 0;            % hora do vazamento 
lk_arr(1:100) = {'noleak'};  % id do array de vazamentos

% 101 a 200 - vazamento no nó 21 - dia inteiro
day(101:200) = 1;
lk_tp(101:200) = 'c';
lk_tm(101:200) = 0;
lk_arr(101:200) = {'no21-di'}; 

% 201 a 300 - vazamento no nó 22 - dia inteiro
day(201:300) = 1;
lk_tp(201:300) = 'c';
lk_tm(201:300) = 0;
lk_arr(201:300) = {'no22-di'}; 

% 301 a 400 - vazamento no nó 23 - dia inteiro
day(301:400) = 1;
lk_tp(301:400) = 'c';
lk_tm(301:400) = 0;
lk_arr(301:400) = {'no23-di'}; 

% 401 a 500 - vazamento no nó 24 - dia inteiro
day(401:500) = 1;
lk_tp(401:500) = 'c';
lk_tm(401:500) = 0;
lk_arr(401:500) = {'no24-di'}; 

% 501 a 600 - vazamento no nó 25 - dia inteiro
day(501:600) = 1;
lk_tp(501:600) = 'c';
lk_tm(501:600) = 0;
lk_arr(501:600) = {'no25-di'}; 

% 601 a 700 - vazamento no nó 26 - dia inteiro
day(601:700) = 1;
lk_tp(601:700) = 'c';
lk_tm(601:700) = 0;
lk_arr(601:700) = {'no26-di'}; 

% 701 a 800 - vazamento no nó 27 - dia inteiro
day(701:800) = 1;
lk_tp(701:800) = 'c';
lk_tm(701:800) = 0;
lk_arr(701:800) = {'no27-di'}; 

% 801 a 900 - vazamento no nó 28 - dia inteiro
day(801:900) = 1;
lk_tp(801:900) = 'c';
lk_tm(801:900) = 0;
lk_arr(801:900) = {'no28-di'}; 

% 901 a 1000 - vazamento no nó 29 - dia inteiro
day(901:1000) = 1;
lk_tp(901:1000) = 'c';
lk_tm(901:1000) = 0;
lk_arr(901:1000) = {'no29-di'}; 

% 1001 a 1100 - vazamento no nó 30 - dia inteiro
day(1001:1100) = 1;
lk_tp(1001:1100) = 'c';
lk_tm(1001:1100) = 0;
lk_arr(1001:1100) = {'no30-di'}; 

% 1101 a 1200 - vazamento no nó 31 - dia inteiro
day(1101:1200) = 1;
lk_tp(1101:1200) = 'c';
lk_tm(1101:1200) = 0;
lk_arr(1101:1200) = {'no31-di'}; 

% 1201 a 1300 - vazamento no nó 32 - dia inteiro
day(1201:1300) = 1;
lk_tp(1201:1300) = 'c';
lk_tm(1201:1300) = 0;
lk_arr(1201:1300) = {'no32-di'}; 

% 1301 a 1400 - vazamento no nó 33 - dia inteiro
day(1301:1400) = 1;
lk_tp(1301:1400) = 'c';
lk_tm(1301:1400) = 0;
lk_arr(1301:1400) = {'no33-di'}; 

% 1401 a 1500 - vazamento no nó 34 - dia inteiro
day(1401:1500) = 1;
lk_tp(1401:1500) = 'c';
lk_tm(1401:1500) = 0;
lk_arr(1401:1500) = {'no34-di'}; 

% 1501 a 1600 - vazamento no nó 35 - dia inteiro
day(1501:1600) = 1;
lk_tp(1501:1600) = 'c';
lk_tm(1501:1600) = 0;
lk_arr(1501:1600) = {'no35-di'}; 

% 1601 a 1700 - vazamento no nó 36 - dia inteiro
day(1601:1700) = 1;
lk_tp(1601:1700) = 'c';
lk_tm(1601:1700) = 0;
lk_arr(1601:1700) = {'no36-di'}; 

% 1701 a 1800 - vazamento no nó 37 - dia inteiro
day(1701:1800) = 1;
lk_tp(1701:1800) = 'c';
lk_tm(1701:1800) = 0;
lk_arr(1701:1800) = {'no37-di'}; 


% 1801 a 1900 - vazamento no nó 38 - dia inteiro
day(1801:1900) = 1;
lk_tp(1801:1900) = 'c';
lk_tm(1801:1900) = 0;
lk_arr(1801:1900) = {'no38-di'}; 

sz = length(lk_tm);

%% Execução do modelo 

%for c = [1, 215, 385, 561, 900]
for c = 1:sz 
    
    if (rem(c,500) == 0)
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
            d.setNodeEmitterCoeff(21,0.3);

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
    
    %Sortear padrões para os pontos de consumo
    for j = 3:13
        pc = randi([3,14]);
        d.setNodeDemandPatternIndex(j,pc);
        log_pc(dsct+1,j) = pc;
    end
    
    %Sortear padrões de consumo dos nós externos
    for i = 40:69
        %d.setNodeDemandPatternIndex(i,randi([15,20]));
        d.setNodeDemandPatternIndex(i,randi([3,6]));
        val = bdmin + (bdmax-bdmin) .* rand;
        d.setNodeBaseDemands(i,val);
    end
    
    %Rodar o modelo
    dsct = dsct +1;             % incrementar o contador de datasets
    Data{dsct,1} = runmodel(d,tv,tmin,cpm,cpv,leak_array,lktype,lktime,nos_med,tot_nos_vaz,tr_med,len,wkday);
    %Datasimp{dsct,1} = runmodel_simple(d,tv,tmin,cpm,cpv,leak_array,lktype,lktime,nos_med,tot_nos_vaz,tr_med,len,wkday);
    
    %checar a pressão média na entrada
    pmi = d.getBinComputedNodePressure;
    %avgPress(dsct,1) = mean(pmi(:,2));
    %fprintf('Pressao media na entrada: %.3f mca \n', mean(avgPress))
    
    %Checar pressão negativa
    t = Data{dsct,1}(:,first:last);
    if any(t(:)<0)
        fprintf('ALERTA: Pressão negativa no dataset %d \n', dsct)
    end
    
    switch lk_id % reset dos vazamentos
        case 'noleak'
            
        case 'no20-di'
            d.setNodeEmitterCoeff(20,0);

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
            fprintf('ERRO')   
    end
        
end




%% Exportação para CSV
if (gerar_csv == 1)
    %mkdir output
    cd C:\Users\Aluno\Documents\MATLAB\output
    %csvwrite('Mat-adj-com-junc.csv',MCJ) % matriz de adjacência
    %csvwrite('log-pad-cons.csv',log_pc) % log dos padrões de consumo

    ddata = cell2mat(Data);
    %ddatasimp = cell2mat(Datasimp);
    %nome = sprintf('csv-cena%d-dist-x%d-seed%d-%dd-%dmin-%.2f-mm.csv',nc,mult,seed,sz,ta,diam);
    nome = sprintf('csv-cena%d-dist-x%d-seed%d-%dd-%dmin.csv',nc,mult,seed,sz,ta);
    csvwrite(nome,ddata)
    %nome2 = sprintf('csv-simplif-cena%d-dist-x%d-seed%d-%dd-%dmin.csv',nc,mult,seed,sz,ta);
    %csvwrite(nome2,ddatasimp)

cd C:\Users\Aluno\Documents\MATLAB\
fprintf(datestr(now,'dd-mm-yyyy HH:MM:SS \n'))
end
beep
end
end
end
end

if (gerar_csv == 1)
    csvwrite('Mat-adj-com-junc.csv',MCJ) % matriz de adjacência
end

fprintf('Pressao media na entrada: %.3f mca \n', mean(avgPress))
disp('Fim')
