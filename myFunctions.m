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

function [diffIndices, startIndices, endIndices] = findZeroIndicesSequences(binaryImage)
    % Find the indices of the zero elements in each column
    zeroIndices = find(sum(binaryImage == 0, 1) > 0);
    
    % Display the indices of the columns containing zero
    disp(['The columns containing zero are columns ', num2str(zeroIndices), '.']);
    
    % Find the start and end indices of each continuous sequence of zero indices
    diffIndices = diff(zeroIndices);
    startIndices = zeroIndices([1 find(diffIndices > 1) + 1]);
    endIndices = zeroIndices([find(diffIndices > 1) length(zeroIndices)]);
    
    % Display the start and end indices of each continuous sequence of zero indices
    for i = 1:length(startIndices)
        disp(['Sequence ', num2str(i), ': columns ', num2str(startIndices(i)), ' to ', num2str(endIndices(i)), '.']);
    end

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

        disp(['Submatrix ', num2str(i), ':']);
        disp(extractedCharacters{i});

end


