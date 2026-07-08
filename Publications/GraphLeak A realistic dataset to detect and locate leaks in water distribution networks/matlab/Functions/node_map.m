% Function to map nodes coordinates (x,y,z)

function output = node_map(node_id,coord_cell,len)
% == Function of node mapping ==
% node_id (array) = array containing the IDs of the measurement nodes
% coord_cell (cell) = coordinates of all nodes
% len (int) = number of lines in the data set

j = 0;
for i = node_id
    j = j + 1;
    xn{j} = zeros(len,1) + coord_cell{i}(1);
    yn{j} = zeros(len,1) + coord_cell{i}(2);
    zn{j} = zeros(len,1) + coord_cell{i}(3);
end

XN = cell2mat(xn);
YN = cell2mat(yn);
ZN = cell2mat(zn);

output = [XN YN ZN];

end

    