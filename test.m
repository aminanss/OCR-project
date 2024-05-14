% Read an image
imagePath = 'D:\DANESHGAH\term 5\Signals and Systems\Project\image.jpg';
img = imread(imagePath);

% Perform OCR on the image
ocrResults = ocr(img);

% Display the recognized text
recognizedText = ocrResults.Text;
disp(['Recognized Text: ' recognizedText]);

% Display bounding boxes around recognized words
figure;
imshow(img);
hold on;

% Plot bounding boxes
for i = 1:numel(ocrResults.Words)
    position = ocrResults.WordBoundingBoxes(i, :);
    rectangle('Position', position, 'EdgeColor', 'r', 'LineWidth', 2);
end

hold off;
title('OCR Results');

%%


% Create a 1000x1000 matrix
matrix = rand(1000);

% Define the center of the matrix
center = [500, 500];

% Create a meshgrid centered at the center of the matrix
[X, Y] = meshgrid(1:1000, 1:1000);
X = X - center(1) ;
Y = center(2) - Y ;

% Calculate the angle and radius of each point
angle = atan2(Y, X);
radius = sqrt(X.^2 + Y.^2);

% Divide the angle by pi/4 to get the index of the part
partIndex = floor(angle / (pi/4)) + 5;
partIndex = changem(partIndex, [8], [9]); 


% Divide the radius by 10 and round down to get the index of the subpart
subpartIndex = floor(radius / 8) + 1;

% Sum the elements of the matrix for each subpart
subpartSums = accumarray([partIndex(:), subpartIndex(:)], matrix(:), [], @sum);

B = subpartSums(:)'; % This will create a 1 * 100 row vector B from A

%%

A = [1 1 1 1 1; 1 1 1 1 1; 1 0 1 0 1; 0 1 0 1 0; 1 0 1 0 1];
B = logical(A); % This will create a 5 * 5 logical matrix B from A
row = find(any(A == 0, 2), 1, 'first'); % This will return the index of the first row containing 0
%%
folders = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
for i = 1:length(folders)
    folder_name = ['mySamples/' folders(i)];
    eval([folders(i) '_folder = ''' folder_name ''';']);
end

%%
% Create a sample table
T = table({'John'; 'Mary'}, [25; 30], [100; 90], 'VariableNames', {'Name', 'Age', 'Score'});

% Add a new row to the top of the table
newRow = {'Alice', 20, 95};
T = vertcat(table(newRow, 'VariableNames', T.Properties.VariableNames), T);

% Number the columns
T.Properties.VariableNames = strcat({'Column '}, string(1:width(T)));

% Display the updated table
T
%%
% Create a cell matrix
A = {[1 2 3 4 5 6 7 8 9 10]; [11 12 13 14 15 16 17 18 19 20]; [21 22 23 24 25 26 27 28 29 30]};

% Convert the cell matrix to a matrix
B = cell2mat(A);

% Convert the matrix to a table with each element in a separate column
T = array2table(B', 'VariableNamingRule', 'numbered');

% Display the table
disp(T);

%%
% Create a 1xn matrix
A = [1 2 3 1 4 5 1 6 7 1];

% Get the indices of the elements that are equal to 1
indices = find(A == 1);

% Remove the elements that are equal to 1
B = setdiff(A, A(indices));

% Display the result
disp(B);

%%
% Read the image
I = imread('ARGON.jpg');

% Convert the image to grayscale
I = rgb2gray(I);

% Convert the image to a binary image
binaryImage = imbinarize(I);

% Display the original and binary images side by side
displayOriginalAndBinaryImage(I , binaryImage)

% Find zero indices sequences in coloumns of binaryImage (cells containing a characters data)
[diffIndices, startIndices, endIndices] = findZeroIndicesSequencesInRows(binaryImage);

% Split the matrix into submatrices based on the start and end indices
% (extract minimal matrix only containing character data cells)
extractedCharacters = arrayfun(@(x,y) binaryImage(:,x:y), startIndices, endIndices, 'UniformOutput', false);

% Remove the rows that only contain Ones ( They are not data cells )
extractedCharacters = cutOffOnlyOneRows(extractedCharacters);

% Display final extracted characters on console
%displayCharactersMatrices(extractedCharacters)

% Scale all extracted characters to 1000*1000 matrices to compare them with
% dateBase of ourown
data = scaleCharactersMatrices(extractedCharacters);
data = data';

% Find the characteristic matrix of each letter found
% Iterate over the rows of the cell matrix
for i = 1 : size(data, 1)
    % Get the current row
    data{i} = findMatrixCharacteristics(data{i}) ;
end

% Create a table for extracted characters to prepare them for prediction
column2_data = cellstr(repmat('unknown', size(data, 1), 1));
dataCellMatrix = horzcat(data, column2_data);
dataTable = cell2table(dataCellMatrix);
dataTable.Properties.VariableNames = {'properties', 'letter'};

% Find space indices
% Get the indices of the elements that are equal to 1
indices = find(diffIndices == 1);

% Remove the elements that are equal to 1
diffIndices(indices) = [];

% Set the threshold value
threshold = 23;

% Find the indices of the elements that are greater than the threshold value
spaceIndices = find(diffIndices > threshold);

% Predict the extracted letters
predictedLettersCellMatrix = wideNeuralNetwork.predictFcn(dataTable);

% Convert predicted letters to a text variable
recognizedTextInThisLine = strjoin(predictedLettersCellMatrix', '');
for i = 1:length(spaceIndices)
    recognizedTextInThisLine = insertAfter(recognizedTextInThisLine, spaceIndices(i) + i - 1, ' ');
end

disp("The recognized text is : " + recognizedTextInThisLine)



