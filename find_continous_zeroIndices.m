% Create a logical matrix
A = logical([1 1 0 0 1 1 0 0 0; 0 1 1 0 0 1 1 0 0; 1 0 1 0 1 0 1 0 1]);

% Find the indices of the zero elements in each column
zeroIndices = find(sum(A == 0, 1) > 0);

% Find the start and end indices of each continuous sequence of zero indices
diffIndices = diff(zeroIndices);
startIndices = zeroIndices([1 find(diffIndices > 1) + 1]);
endIndices = zeroIndices([find(diffIndices > 1) length(zeroIndices)]);

% Display the start and end indices of each continuous sequence of zero indices
for i = 1:length(startIndices)
    disp(['Sequence ', num2str(i), ': columns ', num2str(startIndices(i)), ' to ', num2str(endIndices(i)), '.']);
end


%%
% Create a logical matrix
clear;
% Create a logical matrix
A = logical([1 1 0 0 1 1 0 0 0; 1 1 0 0 1 1 0 0 0; 1 1 0 0 1 1 0 0 0]);

% Find the indices of the zero elements in each column
zeroIndices = find(sum(A == 0, 1) > 0);

% Find the start and end indices of each continuous sequence of zero indices
diffIndices = diff(zeroIndices);
startIndices = zeroIndices([1 find(diffIndices > 1) + 1]);
endIndices = zeroIndices([find(diffIndices > 1) length(zeroIndices)]);

% Split the matrix into submatrices based on the start and end indices
subMatrices = arrayfun(@(x,y) A(:,x:y), startIndices, endIndices, 'UniformOutput', false);

% Display the submatrices
for i = 1:length(subMatrices)
    disp(['Submatrix ', num2str(i), ':']);
    disp(subMatrices{i});
end


%%
% Create a matrix
A = [1 1 1; 1 2 3; 4 5 6; 1 1 1; 7 8 9; 1 1 1];

% Find the rows that contain all ones
onesRows = all(A == 1, 2);

% Remove the rows that contain all ones from the top and bottom of the matrix
A([find(onesRows, 1, 'first') : find(onesRows, 1, 'last')], :) = [];

% Display the updated matrix
disp('The updated matrix is:');
disp(A);


