function [pstr, pv] = pat_interp(epanet, hrs, pts)
% == Function to create a interpolated demand pattern from some previously
% defined data points ====
% ATTENTION: please provide at least 4 points to interpolate
% RECCOMENDATION: Include Hours 1 and 24 to establish the beginning and
% ending of the new pattern
% NOTICE: the network model must contain at least one demand pattern available
% NOTICE (2): The first value in the vector corresponds to 1 A.M.
% epanet (obj) = EPANET object
% hrs (array) = predefined hours of day
% pts (array) = demand value on the predefined hours

g = epanet;
pat = g.getPattern;             % Loads an existing demand pattern
psize = length(pat);            % Checks the demand pattern size
pct = g.getPatternCount;        % Checks the number of patterns in the model
pid = pct +1;                   % Defines the new pattern ID
day = linspace(1,psize,psize);  % Creates the time vector
pv = interp1(hrs,pts,day,'pchip');

if any(pv<0) == 1
    sprintf('ERROR: The interpolation produced a negative value')
    pv = max(pv,0);
    return
end

pstr = sprintf('%s%d','P',pid); % Writes the pattern name as a string

% To run the function do the following:
%[pstr, pv] = pat_interp(epanet,hrs,pts);
%d.addPattern(pstr,pv);

% A suggested usage is as follows:
% hrs = [1 4 7 10 13 16 19 21 24];   %time vector
% pts = min + (max-min) .* rand(1,9);  % randomized values based on a
% min-max range