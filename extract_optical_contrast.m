clear;
clc;
close all;

%% Settings
fileName = ['.\Screenshot 2026-07-07 143107.png'];

% The CSV file is saved in the current MATLAB folder.
outputFile = 'optical_contrast_data.csv';

% Plot-axis limits inferred from the image.
xLim = [-30, 30];
yLim = [0, 0.35];

% Plot regions: [left, top, width, height], in pixels.
plotROI = [
      48, 61, 520, 254
     598, 61, 520, 254
    1155, 61, 520, 254
];

%% Read the image
if ~isfile(fileName)
    error('Image file not found:\n%s', fileName);
end

I = imread(fileName);
results = cell(3, 1);

%% Extract the three curves
for k = 1:3
    J = imcrop(I, plotROI(k, :));

    R = double(J(:, :, 1));
    G = double(J(:, :, 2));
    B = double(J(:, :, 3));

    % Select the cyan-blue Matplotlib lines.
    mask = (B > 100) & ...
           (G > 70) & ...
           (B > R + 35) & ...
           (G > R + 25);

    mask = bwareaopen(mask, 2);
    [nRows, nCols] = size(mask);

    xPixel = NaN(nCols, 1);
    yPixel = NaN(nCols, 1);

    % Use the median line position in each image column.
    for col = 1:nCols
        rows = find(mask(:, col));

        if ~isempty(rows)
            xPixel(col) = col;
            yPixel(col) = median(rows);
        end
    end

    valid = ~isnan(yPixel);
    xPixel = xPixel(valid);
    yPixel = yPixel(valid);

    if numel(xPixel) < 2
        error('Curve %d could not be extracted.', k);
    end

    % Convert image pixels to plot coordinates.
    x = xLim(1) + (xPixel - 1) / (nCols - 1) * diff(xLim);
    y = yLim(2) - (yPixel - 1) / (nRows - 1) * diff(yLim);

    [x, index] = unique(x);
    y = y(index);
    y = smoothdata(y, 'movmedian', 5);

    results{k} = table(x, y, 'VariableNames', {'x', 'y'});
end

%% Interpolate onto one common X grid
xCommon = linspace(xLim(1), xLim(2), 1000)';

curve1 = interp1(results{1}.x, results{1}.y, ...
    xCommon, 'linear', NaN);
curve2 = interp1(results{2}.x, results{2}.y, ...
    xCommon, 'linear', NaN);
curve3 = interp1(results{3}.x, results{3}.y, ...
    xCommon, 'linear', NaN);

%% Save all curves in one CSV file
allData = table(xCommon, curve1, curve2, curve3, ...
    'VariableNames', {'x_um', 'curve_1', 'curve_2', 'curve_3'});

writetable(allData, outputFile);
fprintf('Data saved to:\n%s\n', fullfile(pwd, outputFile));

%% Plot the extracted data
figure('Color', 'w');
plot(xCommon, curve1, 'LineWidth', 1.5);
hold on;
plot(xCommon, curve2, 'LineWidth', 1.5);
plot(xCommon, curve3, 'LineWidth', 1.5);
grid on;
xlim(xLim);
ylim(yLim);
xlabel('x, \mum');
ylabel('Optical contrast');
title('Extracted optical contrast data');
legend('Curve 1', 'Curve 2', 'Curve 3', 'Location', 'best');
