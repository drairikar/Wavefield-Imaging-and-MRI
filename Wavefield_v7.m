clc; clear; clf; close all;

% load tu_logo.mat
% --- 1. SETUP PARAMETERS ---
% k_b_vec = 1:0.5:5;            % Multi-frequency
% F = numel(k_b_vec);
kb = 1 ;
num_sources = 8;              % Multi-source
num_receivers = 150;          % Receivers in a single circle

lambda0 = 2*pi/kb;             % Reference wavelength (for k_b=2)
h = lambda0 / 20;             % Grid step size

% --- 2. DEFINE DOMAIN AND OBJECT ---
x_coords_d = 0:h:lambda0;
y_coords_d = 0:h:lambda0;

% x_coords_d = -lambda0:h:lambda0;
% y_coords_d = -lambda0:h:lambda0;

[X_D, Y_D] = meshgrid(x_coords_d, y_coords_d);
Nx = length(x_coords_d); Ny = length(y_coords_d); N = Nx * Ny;
X = X_D(:); Y = Y_D(:); % Vectorized grid points

cx = lambda0/2; cy = lambda0/2; % Center of domain
% % cx = 0; cy = 0;
radius = lambda0/2;
k_obj = ones(size(X_D));
r = sqrt((X_D - cx).^2 + (Y_D - cy).^2);
k_obj(r <= radius) = 1.05;
cf = (k_obj/kb).^2 - 1;
x_true = cf(:);
% x_true = reshape(cf, N, 1);

% cf = TU_logo_2;
% x_true = cf(:);

% --- 3. DEFINE SOURCE AND RECEIVER GEOMETRY ---
% Sources
source_radius = 6 * lambda0;
source_thetas = linspace(0, 2*pi, num_sources+1); source_thetas(end) = [];
source_x = cx + source_radius * cos(source_thetas);
source_y = cy + source_radius * sin(source_thetas);
sources = [source_x', source_y'];

% Receivers:
receiver_radius = 3 * lambda0;
% r2 = 3*lambda0;
theta_rec = linspace(0, 2*pi, num_receivers+1); theta_rec(end) = [];
x_rec = cx + receiver_radius * cos(theta_rec);
y_rec = cy + receiver_radius * sin(theta_rec);
% x2 = cx + lambda0/2 + r2*cos(theta_rec);
% y2 = cy + r2*sin(theta_rec);
% x = [x_rec, x2];
% y = [y_rec, y2];
D_REC = [x_rec', y_rec'];
% D_REC = [x', y'];

% --- 4. VISUALIZE SETUP ---
figure('Name', 'Experiment Setup');
hold on;
plot(x_coords_d([1 end end 1 1]), y_coords_d([1 1 end end 1]), 'k-', 'LineWidth', 2, 'DisplayName', 'Domain D');
plot(D_REC(:,1), D_REC(:,2), '.b', 'MarkerSize', 8, 'DisplayName', 'Receivers');
plot(sources(:,1),sources(:,2), 'r*', 'MarkerSize', 10, 'LineWidth', 1.5, 'DisplayName', 'Sources');
set(gca, 'YDir','reverse');
xlabel('x');
ylabel('y');
title('Multi-Source Setup');
axis equal; grid on; legend('show');
hold off;

% --- 5. BUILD THE SYSTEM MATRIX (Correct Loop Structure) ---
A_big = zeros(num_sources * num_receivers, N, 'like', 1i);
u_big = zeros(num_sources * num_receivers, 1, 'like', 1i);
% u_big_noise = zeros(F * num_sources * num_receivers, 1, 'like', 1i);

tic;
fprintf('Building system matrix...\n');
row = 0;
for s = 1:num_sources      % Loop over each source
    rho_s = sources(s, :); % Get current source
    
    % for f = 1:F            % Loop over each frequency
    %     k = k_b_vec(f);
        
          % Recalculate incident field for the CURRENT source and frequency
    distance_inc = sqrt((X - rho_s(1)).^2 + (Y - rho_s(2)).^2);
    U_INC = (-1j./4) * besselh(0, 2, kb* distance_inc);

    % Vectorized Green's function from object points to all receivers
    dist_mn = sqrt((D_REC(:,1) - X').^2 + (D_REC(:,2) - Y').^2);
    G_matrix = (-1j./4) * besselh(0, 2, kb* dist_mn);
    
    % Build A_f for this source and frequency
    A_f = kb^2 * G_matrix .* U_INC' * h^2;

    % Stack into big matrix and generate data
    idx = row + (1:num_receivers);
    A_big(idx,:) = A_f;
    u_big(idx) = A_f * x_true;
    
    row = row + num_receivers;
end
% end
fprintf('Matrix construction complete. Time elapsed: %.2f seconds.\n', toc);

%---    NOISE INJECTION -----
fprintf('Noise injection... \n')
signal_power = norm(u_big)^2/length(u_big);
snr_db  = 20;
noise_power = signal_power/(10^(snr_db/10));
noise = sqrt(noise_power/2)*(rand(size(u_big)) + 1j*rand(size(u_big)));
u_big_noise = u_big + noise;

% --- 6. SOLVE INVERSE PROBLEM ---
fprintf('Solving with TSVD\n');
% s_svd = svd(A_big, 'econ');
[U, S, V] = svd(A_big, 'econ');
singular_vals = diag(S);
threshold = 50;
U_k = U(:, 1:threshold);
V_k = V(:, 1:threshold);
S_k = S(1:threshold, 1:threshold);

x_tsvd_noise = V_k * (S_k \ (U_k' * u_big_noise));
cf_tik_noise = reshape(real(x_tsvd_noise), Ny, Nx);

x_tsvd_noiseless = V_k * (S_k \ (U_k' * u_big));
cf_tik = reshape(real(x_tsvd_noiseless), Ny, Nx);

figure;
semilogy(singular_vals, 'o-');
xlabel('Index'); ylabel('Singular Value (log scale)');
title('Singular values of A\_big');
grid on;

% alpha_noiseless = 1e-10;
% lambda_noiseless = alpha_noiseless * s_svd(1)^2;
% 
% alpha_noisy = 1e-1;
% lambda_noisy = alpha_noisy * s_svd(1)^2;
% 
% x_tik = (A_big'*A_big + lambda_noiseless*eye(N)) \ (A_big'*u_big);
% cf_tik = reshape(real(x_tik), Ny, Nx);
% fprintf('Done.\n');
% 
% x_tik_noise = (A_big'*A_big + lambda_noisy*eye(N)) \ (A_big'*u_big_noise);
% cf_tik_noise = reshape(real(x_tik_noise), Ny, Nx);
% fprintf('Done noisy case.\n');

% --- 7. PLOT RESULTS ---
figure;
subplot(1,3,1);
imagesc(x_coords_d, y_coords_d, cf);
axis image; colorbar; title('True Contrast');

subplot(1,3,2);
imagesc(x_coords_d, y_coords_d, cf_tik);
axis image; colorbar;
title(sprintf('Reconstruction (Fixed)\n'));
% title(sprintf('Reconstruction (Fixed)\n\\lambda = %.2g', lambda_noiseless));

subplot(1,3,3)
imagesc(x_coords_d, y_coords_d, cf_tik_noise);
axis image; colorbar;
title(sprintf('Reconstruction (Noise)\n'))
% title(sprintf('Reconstruction (Noise)\n\\lambda = %.2g', lambda_noisy));

% Calculate and display error
error = norm(double(cf_tik) - double(cf)) / norm(double(cf));
fprintf('Relative reconstruction error (fixed): %.4f\n', error);

error_noise = norm(double(cf_tik_noise) - double(cf)) / norm(double(cf));

fprintf('Relative reconstruction error (noise): %.4f\n', error_noise);

% figure;
% imagesc(x_coords_d, y_coords_d, abs(u_big))
% axis image; colorbar; title('USC')