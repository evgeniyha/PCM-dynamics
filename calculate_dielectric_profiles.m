% clear;
% clc;
% close all;

% CALCULATE_DIELECTRIC_PROFILES
% Builds a spatial optical-property profile at one wavelength and one t.
% Keep this script and all three input files in the same folder.

%% User settings: input and output files
contrastFile = 'optical_contrast_data.csv';
amorphousFile = 'a_gst_frantz_pr.txt';
crystallineFile = 'c_gst_frantz_pr.txt';

% Parameter values assigned to profiles 1, 2 and 3.
profileT = [0, 0.5, 1];

% Requested interpolated profile and wavelength used for plotting.
t = 0.7;
targetLambda_um = 5.4;
interpolatedOutputFile = 'interpolated_profile_t.txt';

% Select what is plotted and saved: 'nk' or 'epsilon'.
outputMode = 'epsilon';

% Set true to plot the three reference m profiles and interpolated m(x,t).
plotMProfiles = true;

% Enforce cylindrical symmetry of the 1D diameter profile about this center.
enforceCylindricalSymmetry = true;
symmetryCenter_um = 0;

%% Validate user settings and input files
if ~isscalar(t) || t < profileT(1) || t > profileT(end)
    error('t must be within [%.6g, %.6g].', profileT(1), profileT(end));
end

%% Read spatial optical-contrast profiles
contrastTable = readtable(contrastFile);

requiredColumns = {'x_um', 'curve_1', 'curve_2', 'curve_3'};

x_um = contrastTable.x_um(:).';

contrast = [contrastTable.curve_1, ...
            contrastTable.curve_2, ...
            contrastTable.curve_3].';

%% Optional cylindrical symmetry of the measured contrast
if enforceCylindricalSymmetry
    mirroredX_um = 2*symmetryCenter_um - x_um;

    for p = 1:3
        % Fill extraction gaps before pairing opposite radial positions.
        contrastFilled = fillmissing(contrast(p, :), 'linear', 2, ...
            'EndValues', 'nearest');

        contrastMirrored = interp1(x_um, contrastFilled, mirroredX_um, ...
            'linear', NaN);

        contrast(p, :) = 0.5*(contrastFilled + contrastMirrored);
    end
end

%% Normalize after symmetrization
% The maximum of the symmetric profile 3 corresponds exactly to m = 1.
referenceContrast = max(contrast(3, :), [], 'omitnan');

m = contrast / referenceContrast;

%% Read amorphous and crystalline optical constants
% Expected TXT format: wavelength_n, n, wavelength_k, k
aData = readmatrix(amorphousFile);
cData = readmatrix(crystallineFile);

lambdaA_n = aData(:, 1);
nA_raw = aData(:, 2);
lambdaA_k = aData(:, 3);
kA_raw = aData(:, 4);

lambdaC_n = cData(:, 1);
nC_raw = cData(:, 2);
lambdaC_k = cData(:, 3);
kC_raw = cData(:, 4);

% Determine the wavelength interval available in every input spectrum.
lambdaMin = max([min(lambdaA_n), min(lambdaA_k), ...
                 min(lambdaC_n), min(lambdaC_k)]);
lambdaMax = min([max(lambdaA_n), max(lambdaA_k), ...
                 max(lambdaC_n), max(lambdaC_k)]);

if ~isscalar(targetLambda_um) || ...
        targetLambda_um < lambdaMin || targetLambda_um > lambdaMax
    error(['targetLambda_um = %.6g um is outside the common ' ...
        'spectrum range [%.6g, %.6g] um.'], ...
        targetLambda_um, lambdaMin, lambdaMax);
end

% Calculate optical constants only at the requested wavelength.
n_a = interp1(lambdaA_n, nA_raw, targetLambda_um, 'linear');
k_a = interp1(lambdaA_k, kA_raw, targetLambda_um, 'linear');
n_c = interp1(lambdaC_n, nC_raw, targetLambda_um, 'linear');
k_c = interp1(lambdaC_k, kC_raw, targetLambda_um, 'linear');

%% Complex dielectric permittivities of the pure phases
epsilon_a = (n_a + 1i*k_a).^2;
epsilon_c = (n_c + 1i*k_c).^2;

% Effective factors from the Lorenz-Lorentz formula.
factor_a = (epsilon_a - 1) ./ (epsilon_a + 2);
factor_c = (epsilon_c - 1) ./ (epsilon_c + 2);

%% Output dimension
nX = numel(x_um);

%% Interpolate smoothly between the three profiles at the requested t

% Fill image-extraction gaps along x before interpolation in t.
mForInterpolation = m;
for p = 1:3
    mForInterpolation(p, :) = fillmissing(mForInterpolation(p, :), ...
        'linear', 2, 'EndValues', 'nearest');
end

% Linear interpolation between adjacent measured profiles.
m_t = interp1(profileT, mForInterpolation, t, 'linear');

if plotMProfiles
    figure('Color', 'w');
    hold on;

    plot(x_um, mForInterpolation(1, :), 'LineWidth', 2);
    plot(x_um, mForInterpolation(2, :), 'LineWidth', 2);
    plot(x_um, mForInterpolation(3, :), 'LineWidth', 2);
    plot(x_um, m_t, 'k--', 'LineWidth', 2);

    grid on;
    xlabel('x, um');
    ylabel('m');
    ylim([0, 1]);
    title(sprintf('Phase-fraction profiles and interpolation at t = %.4g', t));
    legend(sprintf('t = %.4g (profile 1)', profileT(1)), ...
        sprintf('t = %.4g (profile 2)', profileT(2)), ...
        sprintf('t = %.4g (profile 3)', profileT(3)), ...
        sprintf('interpolated t = %.4g', t), ...
        'Location', 'best');
end

factor_eff_t = factor_a .* (1 - m_t) + factor_c .* m_t;
epsilon_eff_t = (1 + 2*factor_eff_t) ./ (1 - factor_eff_t);

if strcmp(outputMode, 'nk')
    complexIndex_t = sqrt(epsilon_eff_t);
    n_eff_t = real(complexIndex_t);
    k_eff_t = imag(complexIndex_t);
end

%% Save the interpolated profile for the requested t
tColumn = repmat(t, nX, 1);
wavelengthTColumn = repmat(targetLambda_um, nX, 1);
xTColumn = x_um(:);
mTColumn = m_t(:);

if strcmp(outputMode, 'nk')
    interpolatedTable = table(tColumn, wavelengthTColumn, xTColumn, ...
        mTColumn, n_eff_t(:), k_eff_t(:), ...
        'VariableNames', {'t', 'wavelength_um', 'x_um', ...
        'm', 'n_eff', 'k_eff'});
else
    interpolatedTable = table(tColumn, wavelengthTColumn, xTColumn, ...
        mTColumn, real(epsilon_eff_t(:)), imag(epsilon_eff_t(:)), ...
        'VariableNames', {'t', 'wavelength_um', 'x_um', ...
        'm', 'epsilon_real', 'epsilon_imag'});
end

writetable(interpolatedTable, interpolatedOutputFile, ...
    'Delimiter', '\t', 'WriteMode', 'overwrite');

fprintf('Interpolated profile for t = %.6g saved to:\n%s\n', ...
    t, fullfile(pwd, interpolatedOutputFile));

%% Visualization at targetLambda_um
figure('Color', 'w');
tiledlayout(2, 1);

if strcmp(outputMode, 'nk')
    nexttile;
    plot(x_um, n_eff_t, 'LineWidth', 2);
    grid on;
    xlabel('x, um');
    ylabel('n_{eff}');
    title(sprintf('n(x) at t = %.4g, \\lambda = %.4f um', ...
        t, targetLambda_um));

    nexttile;
    plot(x_um, k_eff_t, 'LineWidth', 2);
    grid on;
    xlabel('x, um');
    ylabel('k_{eff}');
    title(sprintf('k(x) at t = %.4g, \\lambda = %.4f um', ...
        t, targetLambda_um));
else
    nexttile;
    plot(x_um, real(epsilon_eff_t), ...
        'LineWidth', 2);
    grid on;
    xlabel('x, um');
    ylabel('Re(\epsilon_{eff})');
    title(sprintf('Re(\\epsilon) at t = %.4g, \\lambda = %.4f um', ...
        t, targetLambda_um));

    nexttile;
    plot(x_um, imag(epsilon_eff_t), ...
        'LineWidth', 2);
    grid on;
    xlabel('x, um');
    ylabel('Im(\epsilon_{eff})');
    title(sprintf('Im(\\epsilon) at t = %.4g, \\lambda = %.4f um', ...
        t, targetLambda_um));
end
