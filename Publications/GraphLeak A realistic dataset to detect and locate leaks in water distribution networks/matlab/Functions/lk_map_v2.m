function output = lk_map_v2(tot_nos_vaz,tv,leak_array,lktype,lktime,cpv,len)
% == Function to manage the coordinates of leakage nodes ==
% = MAXIMUM ONE ACTIVE LEAKAGE PER SIMULATION =
% ** Output: vector to indicate active leakage + 3 columns containing the 
% coordinates of the active leak node  **
% tot_nos_vaz (int) = total leakage nodes
% tv (array) = time vector (seconds)
% leak_array (array) = leakage vector (indicates leakage node)
% lktype (string) = indicates whether leakage is constant ('c'),
% intermitent ('i') ou none ('n')
% lktime (int) = Starting time of leakage (only matters for intermitent type)
% cpv (array) = leakage nodes coordinates
% len (int) = number of lines in the data set

switch lktype
    case 'c' % continous leakage 
        pv = find(leak_array == 1);
        cpva_ind = [cpv(1,pv) cpv(1,pv+tot_nos_vaz) cpv(1,pv+(2*tot_nos_vaz)) ];
        ivz = repmat(leak_array,len,1); % vector of leakage incidence
        cpva = repmat(cpva_ind,len,1);  % coordinates of the leakage node 
        
    case 'i' % intermitent leakage
        pv = find(leak_array == 1);
        lksec = lktime*3600;        % converts hour into seconds
        lktv = find(tv==lksec);     % finds the data set line in which the leakage starts
        lastline = find(tv==86400); % finds the last line in the data set
        cpva_ind = [cpv(1,pv) cpv(1,pv+tot_nos_vaz) cpv(1,pv+(2*tot_nos_vaz)) ];
        
        ivz = [zeros(lktv-1,length(leak_array)); repmat(leak_array,len-lktv+1,1)]; % vector of leakage incidence
        cpva = [zeros(lktv-1,3); repmat(cpva_ind,len-lktv+1,1)];  %coordinates of the leakage node
        
    case 'n' % no leakage
        ivz = zeros(len,length(leak_array));
        cpva = zeros(len,3);
        
    otherwise
        fprintf('INVALID LEAKAGE TIPE - see lk_map_v2 function for info')
end      


output = [ivz cpva];