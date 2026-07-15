% Open and plot a crystallinity profile saved by the Python scripts.
%
% Python creates the MAT file next to the PNG profile, for example:
%   output/crystallinity_profile_digitized_ttt.mat
%
% Run this file from the project folder in MATLAB.

clear; clc;

matFile = fullfile('output', 'crystallinity_profile_digitized_ttt.mat');

if ~isfile(matFile)
    error(['MAT file not found: %s\n\n' ...
           'First run, for example:\n' ...
           '  python run_article_case_digitized_ttt.py\n'], matFile);
end

S = load(matFile);

fprintf('Loaded: %s\n', matFile);
fprintf('Energy: %.4g nJ\n', S.pulse_energy_nJ);
if isfield(S, 'method')
    fprintf('Method: %s\n', string(S.method));
end
fprintf('crystallinity size: %d x %d (z x r>=0)\n', size(S.crystallinity, 1), size(S.crystallinity, 2));

figure('Color', 'w');
imagesc(S.r_symmetric_um, S.z_nm, S.crystallinity_symmetric);
set(gca, 'YDir', 'reverse');
xlim([-30 30]);
axis tight;
colormap(gray);
caxis([0 1]);
colorbar;
xlabel('r, \mum');
ylabel('z in GST, nm');
title(sprintf('Crystallinity profile, %.4g nJ', S.pulse_energy_nJ));

% Optional: plot the center-line profile Xc(z) at r ~= 0.
[~, ir0] = min(abs(S.r_um));
figure('Color', 'w');
plot(S.crystallinity(:, ir0), S.z_nm, 'LineWidth', 1.8);
set(gca, 'YDir', 'reverse');
xlim([0 1]);
grid on;
xlabel('X_c at r \approx 0');
ylabel('z in GST, nm');
title('Center-line crystallinity profile');
