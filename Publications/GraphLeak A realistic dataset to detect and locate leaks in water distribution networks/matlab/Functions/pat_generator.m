function [pstr, pv] = pat_generator(epanet,min,max)
% == Function to generate a new demand pattern for an existing WDN model ==
% The model must have at least one configured demand pattern
% epanet (obj) = EPANET object
% min (double) = minimal value for each instantaneous demand
% max (double) = maximum value for each instantaneous demand

g = epanet;
pat = g.getPattern;             % Loads an existing demand pattern
psize = length(pat);            % Checks the demand pattern size
pct = g.getPatternCount;        % Checks the number of patterns in the model
pid = pct +1;                   % Defines the new pattern ID
pv = min + (max-min) .*rand(1,psize);   % Generates a random array
pstr = sprintf('%s%d','P',pid);         % Writes the new pattern name

% To run the function do the following:
%[pstr, pv] = pat_generator(d);
%d.addPattern(pstr,pv);