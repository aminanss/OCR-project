% Read the image
I = imread('DJ ARGON REMIX.jpg');

% Convert the image to grayscale
I_gray = rgb2gray(I);

% Convert the image to a binary image
binaryImage = imbinarize(I_gray);

% Display the original and binary images side by side
displayOriginalAndBinaryImage(I , binaryImage)

% Seperate lines of image
[diffIndicesOfLines, startIndicesOfLines, endIndicesOfLines] = findZeroIndicesSequencesInColumns(binaryImage);
lines = split_matrix(binaryImage, startIndicesOfLines, endIndicesOfLines);

recognizedText = '';

for k = 1:length(lines)

    % Find zero indices sequences in coloumns of binaryImage (cells containing a characters data)
    [diffIndices, startIndices, endIndices] = findZeroIndicesSequencesInRows(lines{k});

    % Split the matrix into submatrices based on the start and end indices
    % (extract minimal matrix only containing character data cells)
    extractedCharacters = arrayfun(@(x,y) lines{k}(:,x:y), startIndices, endIndices, 'UniformOutput', false);
    
    % Remove the rows that only contain Ones ( They are not data cells )
    extractedCharacters = cutOffOnlyOneRows(extractedCharacters);
    
    % Display final extracted characters on console
    displayCharactersMatrices(extractedCharacters)
    
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
    predictedLettersCellMatrix = SVM_ALL_LETTERS.predictFcn(dataTable);
    
    % Convert predicted letters to a text variable
    recognizedTextInThisLine = strjoin(predictedLettersCellMatrix', '');
    for i = 1:length(spaceIndices)
        recognizedTextInThisLine = insertAfter(recognizedTextInThisLine, spaceIndices(i) + i - 1, ' ');
    end
    recognizedText_withNewLine = [recognizedText newline];
    recognizedText = [recognizedText_withNewLine recognizedTextInThisLine];
end
disp("The recognized text is : " + recognizedText)



%%

myDataBase = TrainMyDataBase();

myDataBase.Properties.VariableNames = {'properties' , 'letter'};

%%
myTestBase = TrainMyTestBase();

myTestBase.Properties.VariableNames = {'properties' , 'letter'};

%%
yfit = hyperNeuralNetwork_Model.predictFcn(myTestBase);

%%
%*************************************************************************
%*************************************************************************

%myFunctions
function displayOriginalAndBinaryImage(I , binaryImage)
    figure;
    subplot(1, 2, 1);
    imshow(I);
    title('Original Image');
    subplot(1, 2, 2);
    imshow(binaryImage);
    title('Binary Image');
end

function [diffIndices, startIndices, endIndices] = findZeroIndicesSequencesInRows(binaryImage)
    % Find the indices of the zero elements in each column
    zeroIndices = find(sum(binaryImage == 0, 1) > 0);
    
    % Display the indices of the columns containing zero
%     disp(['The columns containing zero are columns ', num2str(zeroIndices), '.']);
    
    % Find the start and end indices of each continuous sequence of zero indices
    diffIndices = diff(zeroIndices);
    startIndices = zeroIndices([1 find(diffIndices > 1) + 1]);
    endIndices = zeroIndices([find(diffIndices > 1) length(zeroIndices)]);
    
    % Display the start and end indices of each continuous sequence of zero indices
%     for i = 1:length(startIndices)
%         disp(['Sequence ', num2str(i), ': columns ', num2str(startIndices(i)), ' to ', num2str(endIndices(i)), '.']);
%     end

end

function lines = split_matrix(binaryImage, start_indices, end_indices)
    num_submatrices = numel(start_indices);
    lines = cell(1, num_submatrices);

    for i = 1:num_submatrices
        lines{i} = binaryImage(start_indices(i):end_indices(i), :);
    end
end

function [diffIndices, startIndices, endIndices] = findZeroIndicesSequencesInColumns(binaryImage)
    % Find the indices of the zero elements in each column
    zeroIndices = find(sum(binaryImage == 0, 2) > 0);
    
    % Display the indices of the columns containing zero
%     disp(['The columns containing zero are columns ', num2str(zeroIndices), '.']);
    
    % Find the start and end indices of each continuous sequence of zero indices
    diffIndices = diff(zeroIndices);
    startIndices = zeroIndices([1 find(diffIndices > 1) + 1]);
    endIndices = zeroIndices([find(diffIndices > 1) length(zeroIndices)]);
    
    % Display the start and end indices of each continuous sequence of zero indices
%     for i = 1:length(startIndices)
%         disp(['Sequence ', num2str(i), ': columns ', num2str(startIndices(i)), ' to ', num2str(endIndices(i)), '.']);
%     end

end

function extractedCharacters = cutOffOnlyOneRows(extractedCharacters)
    
    for i = 1:length(extractedCharacters)
        % Find the rows that contain all ones
        onesRows = all(extractedCharacters{i} == 1, 2);
        countOfOneRowsOnTop = 0;
        countOfOneRowsInBottom = 0;
        for j = 1:length(onesRows)
            if onesRows(j,1) == 0
                countOfOneRowsOnTop = j;
                break;
            end
        end
        for j = length(onesRows): -1 : 1
            if onesRows(j,1) == 0
                countOfOneRowsInBottom = size(extractedCharacters{i}, 1) - j;
                break;
            end
        end
    
        % Remove the rows that contain all ones
        extractedCharacters{i} = extractedCharacters{i}(countOfOneRowsOnTop: size(extractedCharacters{i}, 1) - countOfOneRowsInBottom , :);
    end
    
end

function displayCharactersMatrices(extractedCharacters)
    for i = 1:length(extractedCharacters)
        disp(['Submatrix ', num2str(i), ':']);
        disp(extractedCharacters{i});
    end
end

function data = scaleCharactersMatrices(extractedCharacters)
    
    for i = 1:length(extractedCharacters)

        extractedCharacters{i} = imresize(extractedCharacters{i}, [1000 1000]);
    
    end

    data = extractedCharacters;
end

function letterMatrix = cutOffOnlyOneRowsOfLetter(letterMatrix)
    
        % Find the rows that contain all ones
        onesRows = all(letterMatrix == 1, 2);
        countOfOneRowsOnTop = 0;
        countOfOneRowsInBottom = 0;
        for j = 1:length(onesRows)
            if onesRows(j,1) == 0
                countOfOneRowsOnTop = j;
                break;
            end
        end
        for j = length(onesRows): -1 : 1
            if onesRows(j,1) == 0
                countOfOneRowsInBottom = size(letterMatrix, 1) - j;
                break;
            end
        end
    
        % Remove the rows that contain all ones
        letterMatrix = letterMatrix(countOfOneRowsOnTop: size(letterMatrix, 1) - countOfOneRowsInBottom , :);
    
end

function letterMatrix = cutOffOnlyOneColumnsOfLetter(letterMatrix)
    
        % Find the rows that contain all ones
        onesColumns = all(letterMatrix == 1, 1);
        countOfOneColumnsOnTop = 0;
        countOfOneColumnsInBottom = 0;
        for j = 1:length(onesColumns)
            if onesColumns(1,j) == 0
                countOfOneColumnsOnTop = j;
                break;
            end
        end
        for j = length(onesColumns): -1 : 1
            if onesColumns(1,j) == 0
                countOfOneColumnsInBottom = size(letterMatrix, 1) - j;
                break;
            end
        end
    
        % Remove the rows that contain all ones
        letterMatrix = letterMatrix(: , countOfOneColumnsOnTop: size(letterMatrix, 1) - countOfOneColumnsInBottom);
    
end

function characteristicsMatrix = findMatrixCharacteristics(matrix)
   
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
    partIndex = changem(partIndex, 8, 9); 
    
    % Divide the radius by 10 and round down to get the index of the subpart
    subpartIndex = floor(radius / 8) + 1;
    
    % Sum the elements of the matrix for each subpart
    subpartSums = accumarray([partIndex(:), subpartIndex(:)], matrix(:), [], @sum);
    
    characteristicsMatrix = subpartSums(:)'; % This will create a 1 * 100 row vector B from A

end

%*************************************************************************
%*************************************************************************

function myDataBase = TrainMyDataBase()
    
    % Train data for our database for each character
    for letter = 'A':'Z'
        eval([letter '_Samples = TrainDATA_' letter '();']);
    end
    for letter = 'a':'z'
        eval([letter '_Samples = TrainDATA_' letter '();']);
    end
    
    column2_A = cellstr(repmat('A', 20, 1));
    column2_B = cellstr(repmat('B', 20, 1));
    column2_C = cellstr(repmat('C', 23, 1));
    column2_D = cellstr(repmat('D', 22, 1));
    column2_E = cellstr(repmat('E', 25, 1));
    column2_F = cellstr(repmat('F', 20, 1));
    column2_G = cellstr(repmat('G', 22, 1));
    column2_H = cellstr(repmat('H', 20, 1));
    column2_I = cellstr(repmat('I', 21, 1));
    column2_J = cellstr(repmat('J', 20, 1));
    column2_K = cellstr(repmat('K', 20, 1));
    column2_L = cellstr(repmat('L', 20, 1));
    column2_M = cellstr(repmat('M', 20, 1));
    column2_N = cellstr(repmat('N', 20, 1));
    column2_O = cellstr(repmat('O', 19, 1));
    column2_P = cellstr(repmat('P', 20, 1));
    column2_Q = cellstr(repmat('Q', 19, 1));
    column2_R = cellstr(repmat('R', 21, 1));
    column2_S = cellstr(repmat('S', 21, 1));
    column2_T = cellstr(repmat('T', 20, 1));
    column2_U = cellstr(repmat('U', 18, 1));
    column2_V = cellstr(repmat('V', 21, 1));
    column2_W = cellstr(repmat('W', 21, 1));
    column2_X = cellstr(repmat('X', 20, 1));
    column2_Y = cellstr(repmat('Y', 20, 1));
    column2_Z = cellstr(repmat('Z', 21, 1));
    column2_a = cellstr(repmat('a', 20, 1));
    column2_b = cellstr(repmat('b', 20, 1));
    column2_c = cellstr(repmat('c', 20, 1));
    column2_d = cellstr(repmat('d', 20, 1));
    column2_e = cellstr(repmat('e', 20, 1));
    column2_f = cellstr(repmat('f', 20, 1));
    column2_g = cellstr(repmat('g', 19, 1));
    column2_h = cellstr(repmat('h', 20, 1));
    column2_i = cellstr(repmat('i', 20, 1));
    column2_j = cellstr(repmat('j', 20, 1));
    column2_k = cellstr(repmat('k', 20, 1));
    column2_l = cellstr(repmat('l', 20, 1));
    column2_m = cellstr(repmat('m', 20, 1));
    column2_n = cellstr(repmat('n', 20, 1));
    column2_o = cellstr(repmat('o', 20, 1));
    column2_p = cellstr(repmat('p', 19, 1));
    column2_q = cellstr(repmat('q', 19, 1));
    column2_r = cellstr(repmat('r', 20, 1));
    column2_s = cellstr(repmat('s', 20, 1));
    column2_t = cellstr(repmat('t', 20, 1));
    column2_u = cellstr(repmat('u', 20, 1));
    column2_v = cellstr(repmat('v', 20, 1));
    column2_w = cellstr(repmat('w', 20, 1));
    column2_x = cellstr(repmat('x', 20, 1));
    column2_y = cellstr(repmat('y', 19, 1));
    column2_z = cellstr(repmat('z', 20, 1));
    
    
    for letter = 'A':'Z'
        eval(['myDataBase_' letter ' = horzcat(' letter '_Samples, column2_' letter ');']);
    end
    for letter = 'a':'z'
        eval(['myDataBase_' letter ' = horzcat(' letter '_Samples, column2_' letter ');']);
    end
    
    
    
    myDataBaseCell = vertcat(myDataBase_A, myDataBase_B, myDataBase_C, ...
        myDataBase_D, myDataBase_E, myDataBase_F, myDataBase_G, myDataBase_H,...
        myDataBase_I, myDataBase_J, myDataBase_K, myDataBase_L, myDataBase_M,...
        myDataBase_N, myDataBase_O, myDataBase_P, myDataBase_Q, myDataBase_R,...
        myDataBase_S, myDataBase_T, myDataBase_U, myDataBase_V, myDataBase_W,...
        myDataBase_X, myDataBase_Y, myDataBase_Z, myDataBase_a, myDataBase_b, myDataBase_c, ...
        myDataBase_d, myDataBase_e, myDataBase_f, myDataBase_g, myDataBase_h,...
        myDataBase_i, myDataBase_j, myDataBase_k, myDataBase_l, myDataBase_m,...
        myDataBase_n, myDataBase_o, myDataBase_p, myDataBase_q, myDataBase_r,...
        myDataBase_s, myDataBase_t, myDataBase_u, myDataBase_v, myDataBase_w,...
        myDataBase_x, myDataBase_y, myDataBase_z);
    
    
    myDataBase = cell2table(myDataBaseCell);
        
end

function imageSet = TrainDATA_A()
    % Define the folder path
    folder = 'mySamples/A';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_B()
    % Define the folder path
    folder = 'mySamples/B';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_C()
    % Define the folder path
    folder = 'mySamples/C';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_D()
    % Define the folder path
    folder = 'mySamples/D';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_E()
    % Define the folder path
    folder = 'mySamples/E';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_F()
    % Define the folder path
    folder = 'mySamples/F';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_G()
    % Define the folder path
    folder = 'mySamples/G';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_H()
    % Define the folder path
    folder = 'mySamples/H';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_I()
    % Define the folder path
    folder = 'mySamples/I';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_J()
    % Define the folder path
    folder = 'mySamples/J';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_K()
    % Define the folder path
    folder = 'mySamples/K';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_L()
    % Define the folder path
    folder = 'mySamples/L';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_M()
    % Define the folder path
    folder = 'mySamples/M';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_N()
    % Define the folder path
    folder = 'mySamples/N';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_O()
    % Define the folder path
    folder = 'mySamples/O';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_P()
    % Define the folder path
    folder = 'mySamples/P';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_Q()
    % Define the folder path
    folder = 'mySamples/Q';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_R()
    % Define the folder path
    folder = 'mySamples/R';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_S()
    % Define the folder path
    folder = 'mySamples/S';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_T()
    % Define the folder path
    folder = 'mySamples/T';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_U()
    % Define the folder path
    folder = 'mySamples/U';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_V()
    % Define the folder path
    folder = 'mySamples/V';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_W()
    % Define the folder path
    folder = 'mySamples/W';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_X()
    % Define the folder path
    folder = 'mySamples/X';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_Y()
    % Define the folder path
    folder = 'mySamples/Y';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_Z()
    % Define the folder path
    folder = 'mySamples/Z';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainDATA_a()
    % Define the folder path
    folder = 'mySamples_s/A';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_b()
    % Define the folder path
    folder = 'mySamples_s/B';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_c()
    % Define the folder path
    folder = 'mySamples_s/C';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_d()
    % Define the folder path
    folder = 'mySamples_s/D';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_e()
    % Define the folder path
    folder = 'mySamples_s/E';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_f()
    % Define the folder path
    folder = 'mySamples_s/F';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_g()
    % Define the folder path
    folder = 'mySamples_s/G';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_h()
    % Define the folder path
    folder = 'mySamples_s/H';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_i()
    % Define the folder path
    folder = 'mySamples_s/I';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_j()
    % Define the folder path
    folder = 'mySamples_s/J';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_k()
    % Define the folder path
    folder = 'mySamples_s/K';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_l()
    % Define the folder path
    folder = 'mySamples_s/L';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_m()
    % Define the folder path
    folder = 'mySamples_s/M';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_n()
    % Define the folder path
    folder = 'mySamples_s/N';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_o()
    % Define the folder path
    folder = 'mySamples_s/O';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_p()
    % Define the folder path
    folder = 'mySamples_s/P';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_q()
    % Define the folder path
    folder = 'mySamples_s/Q';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_r()
    % Define the folder path
    folder = 'mySamples_s/R';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_s()
    % Define the folder path
    folder = 'mySamples_s/S';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_t()
    % Define the folder path
    folder = 'mySamples_s/T';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_u()
    % Define the folder path
    folder = 'mySamples_s/U';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_v()
    % Define the folder path
    folder = 'mySamples_s/V';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_w()
    % Define the folder path
    folder = 'mySamples_s/W';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_x()
    % Define the folder path
    folder = 'mySamples_s/X';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_y()
    % Define the folder path
    folder = 'mySamples_s/Y';
    imageSet = folderImageSetCollectorPNG(folder);
end
function imageSet = TrainDATA_z()
    % Define the folder path
    folder = 'mySamples_s/Z';
    imageSet = folderImageSetCollectorPNG(folder);
end





function imageSet = folderImageSetCollectorJPG(folder)
    % Get a list of all files in the folder
    fileList = dir(fullfile(folder, '*.jpg'));

    % Define the images matrix array
    Images_Characteristics  = cell(length(fileList), 1);
    
    % Loop through the list and read each image
    for i = 1:length(fileList)
        % Get the file name
        fileName = fileList(i).name;
        
        % Read the image
        imagePath = fullfile(folder, fileName);
        image = imread(imagePath);

        % Convert the image to grayscale
        grayImage = im2gray(image);

        % Convert the image to a binary image
        binaryImage = imbinarize(grayImage);
        
        % Remove the rows that only contain Ones ( They are not data cells )
        binaryImage = cutOffOnlyOneRowsOfLetter(binaryImage);

        % Remove the columns that only contain Ones ( They are not data cells )
        binaryImage = cutOffOnlyOneColumnsOfLetter(binaryImage);

        % Scale all extracted characters to 1000*1000 matrices to compare them with
        % dateBase of ourown
        data = imresize(binaryImage, [1000 1000]);
        dataCharacteristic = findMatrixCharacteristics(data);
        Images_Characteristics{i} = dataCharacteristic;
    end

    imageSet = Images_Characteristics;

end

function imageSet = folderImageSetCollectorPNG(folder)
    % Get a list of all files in the folder
    fileList = dir(fullfile(folder, '*.png'));

    % Define the images matrix array
    Images_Characteristics  = cell(length(fileList), 1);
    
    % Loop through the list and read each image
    for i = 1:length(fileList)
        % Get the file name
        fileName = fileList(i).name;
        
        % Read the image
        imagePath = fullfile(folder, fileName);
        image = imread(imagePath);

        % Convert the image to grayscale
        grayImage = im2gray(image);

        % Convert the image to a binary image
        binaryImage = imbinarize(grayImage);
        
        % Remove the rows that only contain Ones ( They are not data cells )
        binaryImage = cutOffOnlyOneRowsOfLetter(binaryImage);

        % Remove the columns that only contain Ones ( They are not data cells )
        binaryImage = cutOffOnlyOneColumnsOfLetter(binaryImage);

        % Scale all extracted characters to 1000*1000 matrices to compare them with
        % dateBase of ourown
        data = imresize(binaryImage, [1000 1000]);
        dataCharacteristic = findMatrixCharacteristics(data);
        Images_Characteristics{i} = dataCharacteristic;
    end

    imageSet = Images_Characteristics;

end


%*************************************************************************
%*************************************************************************
function myTestBase = TrainMyTestBase()
     % Train Test for our Testbase for each character
    for letter = 'A':'Z'
        eval([letter '_Samples = TrainTEST_' letter '();']);
    end
    
    column2_A = cellstr(repmat('A', 18, 1));
    column2_B = cellstr(repmat('B', 19, 1));
    column2_C = cellstr(repmat('C', 18, 1));
    column2_D = cellstr(repmat('D', 18, 1));
    column2_E = cellstr(repmat('E', 12, 1));
    column2_F = cellstr(repmat('F', 12, 1));
    column2_G = cellstr(repmat('G', 10, 1));
    column2_H = cellstr(repmat('H', 9, 1));
    column2_I = cellstr(repmat('I', 10, 1));
    column2_J = cellstr(repmat('J', 9, 1));
    column2_K = cellstr(repmat('K', 12, 1));
    column2_L = cellstr(repmat('L', 11, 1));
    column2_M = cellstr(repmat('M', 10, 1));
    column2_N = cellstr(repmat('N', 12, 1));
    column2_O = cellstr(repmat('O', 13, 1));
    column2_P = cellstr(repmat('P', 9, 1));
    column2_Q = cellstr(repmat('Q', 10, 1));
    column2_R = cellstr(repmat('R', 8, 1));
    column2_S = cellstr(repmat('S', 12, 1));
    column2_T = cellstr(repmat('T', 8, 1));
    column2_U = cellstr(repmat('U', 13, 1));
    column2_V = cellstr(repmat('V', 15, 1));
    column2_W = cellstr(repmat('W', 9, 1));
    column2_X = cellstr(repmat('X', 12, 1));
    column2_Y = cellstr(repmat('Y', 12, 1));
    column2_Z = cellstr(repmat('Z', 9, 1));
    
    
    for letter = 'A':'Z'
        eval(['myTestBase_' letter ' = horzcat(' letter '_Samples, column2_' letter ');']);
    end
    
    
    
    myTestBaseCell = vertcat(myTestBase_A, myTestBase_B, myTestBase_C, ...
        myTestBase_D, myTestBase_E, myTestBase_F, myTestBase_G, myTestBase_H,...
        myTestBase_I, myTestBase_J, myTestBase_K, myTestBase_L, myTestBase_M,...
        myTestBase_N, myTestBase_O, myTestBase_P, myTestBase_Q, myTestBase_R,...
        myTestBase_S, myTestBase_T, myTestBase_U, myTestBase_V, myTestBase_W,...
        myTestBase_X, myTestBase_Y, myTestBase_Z);
    
    
    myTestBase = cell2table(myTestBaseCell);
   
end

function imageSet = TrainTEST_A()
    % Define the folder path
    folder = 'Samples/A';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_B()
    % Define the folder path
    folder = 'Samples/B';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_C()
    % Define the folder path
    folder = 'Samples/C';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_D()
    % Define the folder path
    folder = 'Samples/D';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_E()
    % Define the folder path
    folder = 'Samples/E';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_F()
    % Define the folder path
    folder = 'Samples/F';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_G()
    % Define the folder path
    folder = 'Samples/G';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_H()
    % Define the folder path
    folder = 'Samples/H';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_I()
    % Define the folder path
    folder = 'Samples/I';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_J()
    % Define the folder path
    folder = 'Samples/J';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_K()
    % Define the folder path
    folder = 'Samples/K';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_L()
    % Define the folder path
    folder = 'Samples/L';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_M()
    % Define the folder path
    folder = 'Samples/M';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_N()
    % Define the folder path
    folder = 'Samples/N';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_O()
    % Define the folder path
    folder = 'Samples/O';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_P()
    % Define the folder path
    folder = 'Samples/P';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_Q()
    % Define the folder path
    folder = 'Samples/Q';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_R()
    % Define the folder path
    folder = 'Samples/R';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_S()
    % Define the folder path
    folder = 'Samples/S';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_T()
    % Define the folder path
    folder = 'Samples/T';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_U()
    % Define the folder path
    folder = 'Samples/U';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_V()
    % Define the folder path
    folder = 'Samples/V';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_W()
    % Define the folder path
    folder = 'Samples/W';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_X()
    % Define the folder path
    folder = 'Samples/X';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_Y()
    % Define the folder path
    folder = 'Samples/Y';
    imageSet = folderImageSetCollectorJPG(folder);
end
function imageSet = TrainTEST_Z()
    % Define the folder path
    folder = 'Samples/Z';
    imageSet = folderImageSetCollectorJPG(folder);
end


