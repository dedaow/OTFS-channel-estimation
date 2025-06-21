import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import dft
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import h5py
import time
import warnings
import pickle
warnings.filterwarnings('ignore')

class OTFSChannelEstimator:
    def __init__(self, N=8, M=8):
        # OTFS frame parameters
        self.N = N  # number of Doppler bins (time slots)
        self.M = M  # number of delay bins (subcarriers)
        
        # System parameters
        self.delta_f = 15e3  # subcarrier spacing (Hz)
        self.T = 1 / self.delta_f  # block duration
        self.fc = 4e9  # carrier frequency (Hz)
        self.c = 299792458  # speed of light (m/s)
        
        # OTFS grid resolution
        self.delay_resolution = 1 / (self.M * self.delta_f)
        self.Doppler_resolution = 1 / (self.N * self.T)
        
        # Normalized DFT matrix
        self.Fn = dft(self.N) / np.sqrt(self.N)
        
        # Identity matrix for delays
        self.Im = np.eye(self.M)
        
        # DNN model
        self.dnn_model_full = None
        
        print(f"OTFS System initialized: {N}x{M} grid, Channel matrix size: {N*M}x{N*M}")
        
    def create_permutation_matrix(self):
        """Create permutation matrix P"""
        P = np.zeros((self.N * self.M, self.N * self.M), dtype=complex)
        
        for j in range(self.N):
            for i in range(self.M):
                E = np.zeros((self.M, self.N))
                E[i, j] = 1
                
                row_start = j * self.M
                row_end = (j + 1) * self.M
                col_start = i * self.N
                col_end = (i + 1) * self.N
                
                P[row_start:row_end, col_start:col_end] = E
        
        return P
    
    def validate_implementation(self):
        """Validate implementation correctness"""
        print("=== Validating Implementation ===")
        
        # Check permutation matrix properties
        P = self.create_permutation_matrix()
        is_unitary = np.allclose(P @ P.conj().T, np.eye(self.N * self.M), atol=1e-10)
        print(f"P is unitary (P*P' = I): {is_unitary}")
        
        # Check modulation-demodulation consistency (no channel)
        X_test = np.random.randn(self.M, self.N) + 1j * np.random.randn(self.M, self.N)
        s, x_orig = self.otfs_modulation(X_test)
        Y_recovered, y_recovered = self.otfs_demodulation(s)
        
        modulation_error = np.linalg.norm(X_test - Y_recovered)
        print(f"Modulation-Demodulation error (no channel): {modulation_error:.2e}")
        
        # Check vectorization consistency
        x_manual = X_test.T.flatten()
        x_diff = np.linalg.norm(x_orig - x_manual)
        print(f"Vectorization consistency: {x_diff:.2e}")
        
        # Check DFT matrix normalization
        Fn_norm = np.linalg.norm(self.Fn @ self.Fn.conj().T - np.eye(self.N))
        print(f"DFT matrix normalization error: {Fn_norm:.2e}")
        
        print("=== Validation Complete ===\n")
        
        return {
            'permutation_ok': is_unitary,
            'modulation_ok': modulation_error < 1e-10,
            'vectorization_ok': x_diff < 1e-10,
            'dft_ok': Fn_norm < 1e-10
        }
    
    def generate_3gpp_channel(self, channel_type='EVA'):
        """Generate 3GPP standard channel models"""
        if channel_type == 'EPA':
            delays = np.array([0, 30, 70, 90, 110, 190, 410]) * 1e-9
            pdp_db = np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
        elif channel_type == 'EVA':
            delays = np.array([0, 30, 150, 310, 370, 710, 1090, 1730, 2510]) * 1e-9
            pdp_db = np.array([0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9])
        elif channel_type == 'ETU':
            delays = np.array([0, 50, 120, 200, 230, 500, 1600, 2300, 5000]) * 1e-9
            pdp_db = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0])
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")
        
        # Convert dB to linear scale
        pdp_linear = 10 ** (pdp_db / 10)
        pdp_linear = pdp_linear / np.sum(pdp_linear)  # Normalize
        
        return delays, pdp_linear
    
    def generate_channel_matrix(self, max_speed_kmh=300, channel_type='EVA'):
        """Generate channel matrix"""
        # Convert speed from km/h to m/s
        max_UE_speed = max_speed_kmh * (1000 / 3600)
        
        # Maximum Doppler frequency
        nu_max = (max_UE_speed * self.fc) / self.c
        
        # Maximum Doppler spread in bins
        k_max = nu_max / self.Doppler_resolution
        
        # Get 3GPP channel parameters
        delays, pdp_linear = self.generate_3gpp_channel(channel_type)
        taps = len(pdp_linear)
        
        # Generate channel coefficients (Rayleigh fading)
        g_i = np.sqrt(pdp_linear) * (np.sqrt(0.5) * (np.random.randn(taps) + 1j * np.random.randn(taps)))
        
        # Generate delay taps
        l_i = np.round(delays / self.delay_resolution).astype(int)
        l_i = np.clip(l_i, 0, self.M-1)  # Ensure within valid range
        
        # Generate Doppler taps (Jakes model)
        k_i = k_max * np.cos(2 * np.pi * np.random.rand(taps))
        
        # Complex exponential factor
        z = np.exp(1j * 2 * np.pi / (self.N * self.M))
        
        # Delay spread
        delay_spread = np.max(l_i)
        
        # Initialize gs matrix
        gs = np.zeros((delay_spread + 1, self.N * self.M), dtype=complex)
        
        # Fill gs matrix - TDL model
        for q in range(self.N * self.M):
            for i in range(taps):
                if l_i[i] <= delay_spread:
                    gs[l_i[i], q] += g_i[i] * z**(k_i[i] * (q - l_i[i]))
        
        # Generate G matrix
        G = np.zeros((self.N * self.M, self.N * self.M), dtype=complex)
        
        for q in range(self.N * self.M):
            for ell in range(delay_spread + 1):
                if q >= ell:
                    G[q, q - ell] = gs[ell, q]
        
        # Generate delay-Doppler channel matrix H
        P = self.create_permutation_matrix()
        H = np.kron(self.Im, self.Fn) @ (P.conj().T @ G @ P) @ np.kron(self.Im, self.Fn.conj().T)
        
        return H, G, gs
    
    def generate_center_pilot_frame(self, pilot_density=0.25, pilot_spacing=2, edge_guard=1):
        """Generate pilot frame with center-based placement"""
        
        # Center position (consider Doppler shift, place the pilot at center)
        center_m = self.M // 2
        center_n = self.N // 2
        
        # Calculate target number of pilots
        total_symbols = self.N * self.M
        n_pilots = int(total_symbols * pilot_density)
        
        pilot_positions = []
        pilot_symbols = []
        
        # Start with center pilot
        center_pos = center_m * self.N + center_n
        pilot_positions.append(center_pos)
        pilot_symbols.append((1 + 1j) / np.sqrt(2))  # QPSK symbol
        
        # Expand outward in concentric rings
        radius = 1
        pilot_count = 1
        
        while pilot_count < n_pilots and radius <= max(center_m, center_n):
            for dm in range(-radius, radius + 1):
                for dn in range(-radius, radius + 1):
                    if pilot_count >= n_pilots:
                        break
                    
                    # Only consider positions on the current ring boundary
                    if abs(dm) == radius or abs(dn) == radius:
                        m_pos = center_m + dm
                        n_pos = center_n + dn
                        
                        # Check edge guard constraints
                        if (edge_guard <= m_pos < self.M - edge_guard and 
                            edge_guard <= n_pos < self.N - edge_guard):
                            
                            pos = m_pos * self.N + n_pos
                            
                            # Check pilot spacing constraints
                            valid_position = True
                            for existing_pos in pilot_positions:
                                existing_m = existing_pos // self.N
                                existing_n = existing_pos % self.N
                                
                                # Calculate Manhattan distance
                                manhattan_distance = abs(m_pos - existing_m) + abs(n_pos - existing_n)
                                
                                if manhattan_distance < pilot_spacing:
                                    valid_position = False
                                    break
                            
                            if valid_position:
                                pilot_positions.append(pos)
                                # Use QPSK symbols
                                qpsk_symbols = [(1+1j)/np.sqrt(2), (1-1j)/np.sqrt(2), 
                                              (-1+1j)/np.sqrt(2), (-1-1j)/np.sqrt(2)]
                                pilot_symbols.append(qpsk_symbols[pilot_count % 4])
                                pilot_count += 1
                
                if pilot_count >= n_pilots:
                    break
            radius += 1
        
        # Convert to numpy arrays
        pilot_positions = np.array(pilot_positions)
        pilot_symbols = np.array(pilot_symbols)
        
        # Create frame with pilots
        frame_symbols = np.zeros(self.M * self.N, dtype=complex)
        frame_symbols[pilot_positions] = pilot_symbols
        
        # Reshape to delay-Doppler grid (M×N)
        X = frame_symbols.reshape(self.M, self.N)
        
        return X, pilot_positions, pilot_symbols
    
    def otfs_modulation(self, X):
        """OTFS modulation"""
        x = X.T.flatten().reshape(-1, 1)
        P = self.create_permutation_matrix()
        s = np.kron(self.Fn.conj().T, self.Im) @ P @ x
        return s.flatten(), x.flatten()
    
    def otfs_demodulation(self, r):
        """OTFS demodulation"""
        P = self.create_permutation_matrix()
        y = np.kron(self.Im, self.Fn) @ P.conj().T @ r.reshape(-1, 1)
        Y = y.reshape(self.N, self.M).T
        return Y, y.flatten()
    
    def add_awgn(self, signal, snr_db):
        """Add AWGN to signal"""
        Es = 1.0
        SNR = 10**(snr_db / 10)
        sigma_w_2 = Es / SNR
        
        noise = np.sqrt(sigma_w_2 / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        
        return signal + noise, sigma_w_2
    
    def calculate_nmse(self, H_true, H_est):
        """Calculate NMSE using norm-based normalization
        
        NMSE = ||H_true - H_est||_F^2 / ||H_true||_F^2
        
        Args:
            H_true: True channel matrix
            H_est: Estimated channel matrix
            
        Returns:
            nmse: Normalized Mean Square Error
        """
        error = H_true - H_est
        mse = np.mean(np.abs(error)**2)
        norm_squared = np.linalg.norm(H_true, 'fro')**2
        
        if norm_squared == 0:
            return float('inf') if mse > 0 else 0
        
        nmse = mse * H_true.size / norm_squared
        return nmse
    
    def extract_dnn_features(self, Y, pilot_pos, pilot_syms, pilot_density):
        """Extract features for DNN training (without SNR and speed info)"""
        features = []
        
        # Received signal in delay-Doppler domain
        Y_flat = Y.T.flatten()
        features.extend([np.real(Y_flat), np.imag(Y_flat)])
        
        # Pilot information
        pilot_mask = np.zeros(self.N * self.M)
        pilot_mask[pilot_pos] = 1
        features.append(pilot_mask)
        
        pilot_symbols_vec = np.zeros(self.N * self.M, dtype=complex)
        pilot_symbols_vec[pilot_pos] = pilot_syms
        features.extend([np.real(pilot_symbols_vec), np.imag(pilot_symbols_vec)])
        
        # Channel-dependent features from pilots
        channel_estimates_at_pilots = np.zeros(self.N * self.M, dtype=complex)
        for i, pos in enumerate(pilot_pos):
            if i < len(pilot_syms) and np.abs(pilot_syms[i]) > 1e-6:
                channel_estimates_at_pilots[pos] = Y_flat[pos] / pilot_syms[i]
        features.extend([np.real(channel_estimates_at_pilots), np.imag(channel_estimates_at_pilots)])
        
        # System parameters (normalized, excluding SNR and speed)
        features.append([
            pilot_density,
            len(pilot_pos) / (self.N * self.M),
            self.fc / 1e10,
            self.delta_f / 1e5
        ])
        
        # Signal statistics (these can help infer SNR and channel conditions)
        signal_power = np.mean(np.abs(Y_flat)**2)
        signal_mean = np.mean(Y_flat)
        pilot_power = np.mean(np.abs(pilot_syms)**2) if len(pilot_syms) > 0 else 0
        signal_var = np.var(np.abs(Y_flat))
        signal_skewness = np.mean(((np.abs(Y_flat) - np.mean(np.abs(Y_flat))) / np.std(np.abs(Y_flat)))**3) if np.std(np.abs(Y_flat)) > 0 else 0
        signal_kurtosis = np.mean(((np.abs(Y_flat) - np.mean(np.abs(Y_flat))) / np.std(np.abs(Y_flat)))**4) if np.std(np.abs(Y_flat)) > 0 else 0
        
        features.append([
            np.real(signal_mean), np.imag(signal_mean),
            signal_power, np.std(np.abs(Y_flat)),
            pilot_power, signal_var, signal_skewness, signal_kurtosis
        ])
        
        # Frequency domain features
        Y_fft = np.fft.fft(Y_flat)
        features.extend([np.real(Y_fft), np.imag(Y_fft)])
        
        # Additional blind features
        # Correlation features
        Y_autocorr = np.correlate(Y_flat, Y_flat, mode='full')
        autocorr_peak = np.max(np.abs(Y_autocorr))
        autocorr_center_idx = len(Y_autocorr) // 2
        autocorr_lag1 = np.abs(Y_autocorr[autocorr_center_idx + 1]) if autocorr_center_idx + 1 < len(Y_autocorr) else 0
        
        # Power spectral density features
        psd = np.abs(Y_fft)**2
        psd_peak = np.max(psd)
        psd_mean = np.mean(psd)
        psd_std = np.std(psd)
        
        features.append([
            autocorr_peak, autocorr_lag1,
            psd_peak, psd_mean, psd_std
        ])
        
        return np.concatenate(features)
    
    def improved_ls_estimation(self, Y, pilot_pos, pilot_syms):
        """Improved LS estimation with interpolation"""
        Y_flat = Y.T.flatten()
        n_total = self.N * self.M
        
        # Initialize channel estimate
        H_est = np.zeros((n_total, n_total), dtype=complex)
        
        # Direct LS estimation at pilot positions
        pilot_channel_est = np.zeros(n_total, dtype=complex)
        
        for i, pos in enumerate(pilot_pos):
            if i < len(pilot_syms) and np.abs(pilot_syms[i]) > 1e-6:
                pilot_channel_est[pos] = Y_flat[pos] / pilot_syms[i]
        
        # Simple interpolation for non-pilot positions
        for pos in range(n_total):
            if pos not in pilot_pos:
                pos_delay = pos % self.M
                pos_doppler = pos // self.M
                
                weights = []
                estimates = []
                
                for i, pilot_pos_idx in enumerate(pilot_pos):
                    if i < len(pilot_syms):
                        pilot_delay = pilot_pos_idx % self.M
                        pilot_doppler = pilot_pos_idx // self.M
                        
                        delay_dist = abs(pos_delay - pilot_delay)
                        doppler_dist = abs(pos_doppler - pilot_doppler)
                        
                        distance = np.sqrt(delay_dist**2 + doppler_dist**2)
                        
                        if distance < np.sqrt(self.M**2 + self.N**2) / 2:
                            weight = np.exp(-distance / 2)
                            weights.append(weight)
                            estimates.append(pilot_channel_est[pilot_pos_idx])
                
                if len(weights) > 0:
                    weights = np.array(weights)
                    estimates = np.array(estimates)
                    pilot_channel_est[pos] = np.sum(weights * estimates) / np.sum(weights)
        
        # Create channel matrix with dominant diagonal structure
        for i in range(n_total):
            for j in range(n_total):
                if i == j:
                    H_est[i, j] = pilot_channel_est[i]
                elif abs(i - j) <= 2:
                    coupling_factor = 0.1 * np.exp(-0.5 * abs(i - j))
                    H_est[i, j] = pilot_channel_est[i] * coupling_factor
        
        return H_est
    
    def traditional_mmse_estimation(self, Y, pilot_pos, pilot_syms, sigma_w_2):
        """Traditional MMSE channel estimation"""
        H_ls = self.improved_ls_estimation(Y, pilot_pos, pilot_syms)
        
        n_total = self.N * self.M
        
        # Channel correlation matrix (simple exponential model)
        R_hh = np.zeros((n_total, n_total), dtype=complex)
        for i in range(n_total):
            for j in range(n_total):
                distance = abs(i - j)
                correlation = np.exp(-0.5 * distance)
                R_hh[i, j] = correlation
        
        # Noise covariance
        noise_cov = sigma_w_2 * np.eye(n_total)
        
        try:
            H_mmse = R_hh @ np.linalg.solve(R_hh + noise_cov, H_ls)
        except np.linalg.LinAlgError:
            H_mmse = H_ls
        
        return H_mmse
    
    def build_dnn_model_full(self, input_dim):
        """Build full channel matrix estimation DNN"""
        output_dim = (self.N * self.M) ** 2 * 2
        
        print(f"Full DNN: Input dim = {input_dim}, Output dim = {output_dim}")
        
        model = keras.Sequential([
            keras.layers.Dense(2048, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(output_dim, activation='linear')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def generate_training_dataset(self, n_samples=10000, snr_range=(0, 35), speed_range=(5, 600),
                                pilot_spacing=2, edge_guard=1):
        """Generate training dataset (SNR and speed used only for simulation, not as features)"""
        print(f"Generating training dataset with {n_samples} samples...")
        
        X_data = []
        y_data_full = []
        metadata = {
            'snr_values': [], 'speed_values': [], 'channel_types': [],
            'pilot_densities': []
        }
        
        pilot_densities = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        
        for i in range(n_samples):
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")
            
            # Random parameters (used for simulation only, not as features)
            snr_db = np.random.uniform(snr_range[0], snr_range[1])
            max_speed = np.random.uniform(speed_range[0], speed_range[1])
            channel_type = np.random.choice(['EPA', 'EVA', 'ETU'])
            pilot_density = np.random.choice(pilot_densities)
            
            # Generate channel
            H_true, G_true, gs_true = self.generate_channel_matrix(max_speed, channel_type)
            
            # Generate center pilot frame
            X_pilot, pilot_pos, pilot_syms = self.generate_center_pilot_frame(
                pilot_density, pilot_spacing, edge_guard)
            
            # OTFS modulation
            s, x = self.otfs_modulation(X_pilot)
            
            # Channel transmission
            r_clean = H_true @ x.reshape(-1, 1)
            r_noisy, sigma_w_2 = self.add_awgn(r_clean.flatten(), snr_db)
            
            # OTFS demodulation
            Y_noisy, y_vec = self.otfs_demodulation(r_noisy)
            
            # Extract features (without SNR and speed)
            features = self.extract_dnn_features(Y_noisy, pilot_pos, pilot_syms, pilot_density)
            
            # Prepare targets
            full_target = np.concatenate([np.real(H_true.flatten()), np.imag(H_true.flatten())])
            y_data_full.append(full_target)
            
            X_data.append(features)
            metadata['snr_values'].append(snr_db)
            metadata['speed_values'].append(max_speed)
            metadata['channel_types'].append(channel_type)
            metadata['pilot_densities'].append(pilot_density)
        
        result = {
            'X_data': np.array(X_data),
            'y_data_full': np.array(y_data_full),
            'metadata': metadata
        }
        
        return result
    
    def dnn_estimation(self, Y, pilot_pos, pilot_syms, pilot_density):
        """DNN-based channel estimation (blind estimation without SNR and speed)"""
        features = self.extract_dnn_features(Y, pilot_pos, pilot_syms, pilot_density)
        features = features.reshape(1, -1)
        
        if self.dnn_model_full is None:
            raise ValueError("Full DNN model not trained")
        
        prediction = self.dnn_model_full.predict(features, verbose=0)
        n_elements = self.N * self.M
        real_part = prediction[0, :n_elements**2]
        imag_part = prediction[0, n_elements**2:]
        
        return (real_part + 1j * imag_part).reshape(n_elements, n_elements)
    
    def run_comparison_test(self, n_test=1000, pilot_spacing=2, edge_guard=1):
        """Run comprehensive comparison test"""
        print(f"Running channel estimation comparison test with {n_test} samples...")
        
        results = {
            'snr_values': [], 'mse_ls': [], 'mse_mmse': [], 'mse_dnn': [],
            'nmse_ls': [], 'nmse_mmse': [], 'nmse_dnn': [],
            'time_ls': [], 'time_mmse': [], 'time_dnn': []
        }
        
        snr_test_values = [0, 5, 10, 15, 20, 25, 30]
        samples_per_snr = max(1, n_test // len(snr_test_values))
        
        for snr_db in snr_test_values:
            print(f"Testing at SNR = {snr_db} dB...")
            
            snr_results = {key: [] for key in results.keys() if key != 'snr_values'}
            
            for _ in range(samples_per_snr):
                # Generate test scenario
                max_speed = np.random.uniform(50, 400)
                channel_type = np.random.choice(['EPA', 'EVA', 'ETU'])
                pilot_density = np.random.choice([0.15, 0.2, 0.25, 0.3])
                
                # Generate true channel
                H_true, G_true, gs_true = self.generate_channel_matrix(max_speed, channel_type)
                
                # Generate center pilot frame
                X_pilot, pilot_pos, pilot_syms = self.generate_center_pilot_frame(
                    pilot_density, pilot_spacing, edge_guard)
                
                # Pilot transmission for channel estimation
                s_pilot, x_pilot = self.otfs_modulation(X_pilot)
                r_pilot_clean = H_true @ x_pilot.reshape(-1, 1)
                r_pilot_noisy, sigma_w_2 = self.add_awgn(r_pilot_clean.flatten(), snr_db)
                Y_pilot, _ = self.otfs_demodulation(r_pilot_noisy)
                
                # Channel estimation methods
                start_time = time.time()
                H_ls = self.improved_ls_estimation(Y_pilot, pilot_pos, pilot_syms)
                time_ls = time.time() - start_time
                
                start_time = time.time()
                H_mmse = self.traditional_mmse_estimation(Y_pilot, pilot_pos, pilot_syms, sigma_w_2)
                time_mmse = time.time() - start_time
                
                start_time = time.time()
                try:
                    # DNN estimation without SNR and speed info
                    H_dnn = self.dnn_estimation(Y_pilot, pilot_pos, pilot_syms, pilot_density)
                except:
                    H_dnn = H_ls  # Fallback
                time_dnn = time.time() - start_time
                
                # MSE calculation
                mse_ls = np.mean(np.abs(H_true - H_ls)**2)
                mse_mmse = np.mean(np.abs(H_true - H_mmse)**2)
                mse_dnn = np.mean(np.abs(H_true - H_dnn)**2)
                
                # NMSE calculation using norm-based normalization
                nmse_ls = self.calculate_nmse(H_true, H_ls)
                nmse_mmse = self.calculate_nmse(H_true, H_mmse)
                nmse_dnn = self.calculate_nmse(H_true, H_dnn)
                
                # Store results
                snr_results['mse_ls'].append(mse_ls)
                snr_results['mse_mmse'].append(mse_mmse)
                snr_results['mse_dnn'].append(mse_dnn)
                snr_results['nmse_ls'].append(nmse_ls)
                snr_results['nmse_mmse'].append(nmse_mmse)
                snr_results['nmse_dnn'].append(nmse_dnn)
                snr_results['time_ls'].append(time_ls)
                snr_results['time_mmse'].append(time_mmse)
                snr_results['time_dnn'].append(time_dnn)
            
            # Average results for this SNR
            results['snr_values'].append(snr_db)
            for key in snr_results:
                results[key].append(np.mean(snr_results[key]))
        
        return results
    
    def visualize_pilot_placement(self, pilot_density=0.25, pilot_spacing=2, edge_guard=1):
        """Visualize pilot placement"""
        
        X_pilot, pilot_pos, pilot_syms = self.generate_center_pilot_frame(
            pilot_density, pilot_spacing, edge_guard)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        pilot_grid = np.zeros((self.M, self.N))
        for pos in pilot_pos:
            m_idx = pos // self.N
            n_idx = pos % self.N
            pilot_grid[m_idx, n_idx] = 1
        
        im1 = ax1.imshow(pilot_grid, cmap='Blues', origin='lower', extent=[0, self.N, 0, self.M])
    
        center_m = self.M // 2
        center_n = self.N // 2
        ax1.plot(center_n + 0.5, center_m + 0.5, 'r*', markersize=15, label='Center')
        
        ax1.axhline(y=edge_guard, color='red', linestyle='--', alpha=0.7, label=f'Edge Guard ({edge_guard})')
        ax1.axhline(y=self.M - edge_guard, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(x=edge_guard, color='red', linestyle='--', alpha=0.7)
        ax1.axvline(x=self.N - edge_guard, color='red', linestyle='--', alpha=0.7)
        
        ax1.set_title(f'Pilot Placement with Edge Guard\n(Density: {pilot_density:.2f}, Edge Guard: {edge_guard})')
        ax1.set_xlabel('Doppler (bins)')
        ax1.set_ylabel('Delay (bins)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        im2 = ax2.imshow(pilot_grid, cmap='Greens', origin='lower', extent=[0, self.N, 0, self.M])
        
        ax2.plot(center_n + 0.5, center_m + 0.5, 'r*', markersize=15, label='Center')
        
        for i, pos in enumerate(pilot_pos[:5]):  
            m_idx = pos // self.N
            n_idx = pos % self.N
            
            diamond_x = []
            diamond_y = []
            for angle in np.linspace(0, 2*np.pi, 100):
                x_offset = pilot_spacing * np.cos(angle) * 0.7  
                y_offset = pilot_spacing * np.sin(angle) * 0.7
                diamond_x.append(n_idx + 0.5 + x_offset)
                diamond_y.append(m_idx + 0.5 + y_offset)
            
            ax2.plot(diamond_x, diamond_y, '--', alpha=0.5, linewidth=1, 
                    color=plt.cm.tab10(i), label=f'Spacing {pilot_spacing}' if i == 0 else '')
        
        ax2.set_title(f'Pilot Spacing Visualization\n(Pilot Spacing: {pilot_spacing}, Manhattan Distance)')
        ax2.set_xlabel('Doppler (bins)')
        ax2.set_ylabel('Delay (bins)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'pilot_placement_spacing{pilot_spacing}_edge{edge_guard}.png', dpi=300)
        plt.show()
        
        print(f"Current Pilot Information:")
        print(f"  Total pilots: {len(pilot_pos)} / {self.N*self.M}")
        print(f"  Actual pilot density: {len(pilot_pos)/(self.N*self.M):.3f}")
        print(f"  Requested density: {pilot_density:.3f}")
        print(f"  Pilot spacing (Manhattan): {pilot_spacing}")
        print(f"  Edge guard: {edge_guard}")
        
        min_distance = float('inf')
        for i, pos1 in enumerate(pilot_pos):
            for j, pos2 in enumerate(pilot_pos):
                if i != j:
                    m1, n1 = pos1 // self.N, pos1 % self.N
                    m2, n2 = pos2 // self.N, pos2 % self.N
                    manhattan_dist = abs(m1 - m2) + abs(n1 - n2)
                    min_distance = min(min_distance, manhattan_dist)
        
        print(f"  Actual minimum pilot distance: {min_distance}")
        print(f"  NMSE calculation: Using norm-based normalization")
        
        return pilot_grid
    
    def visualize_channel_estimation_results(self, n_vis=1, pilot_spacing=2, edge_guard=1):
        """Visualize channel estimation results"""
        
        print(f"Visualizing channel estimation results...")
        
        for vis_idx in range(n_vis):
            # Generate test scenario
            max_speed = 200  # km/h
            channel_type = 'EVA'
            pilot_density = 0.25
            snr_db = 20
            
            # Generate channel
            H_true, G_true, gs_true = self.generate_channel_matrix(max_speed, channel_type)
            
            # Generate pilot frame
            X_pilot, pilot_pos, pilot_syms = self.generate_center_pilot_frame(
                pilot_density, pilot_spacing, edge_guard)
            
            # OTFS modulation and demodulation
            s_pilot, x_pilot = self.otfs_modulation(X_pilot)
            r_pilot_clean = H_true @ x_pilot.reshape(-1, 1)
            r_pilot_noisy, sigma_w_2 = self.add_awgn(r_pilot_clean.flatten(), snr_db)
            Y_pilot, _ = self.otfs_demodulation(r_pilot_noisy)
            
            # Channel estimation
            H_ls = self.improved_ls_estimation(Y_pilot, pilot_pos, pilot_syms)
            H_mmse = self.traditional_mmse_estimation(Y_pilot, pilot_pos, pilot_syms, sigma_w_2)
            
            try:
                # DNN estimation without SNR and speed info
                H_dnn = self.dnn_estimation(Y_pilot, pilot_pos, pilot_syms, pilot_density)
            except:
                H_dnn = H_ls  
            
            fig = plt.figure(figsize=(16, 12))
            
            display_size = min(100, self.N * self.M)
            H_true_mag = np.abs(H_true[:display_size, :display_size])
            H_ls_mag = np.abs(H_ls[:display_size, :display_size])
            H_mmse_mag = np.abs(H_mmse[:display_size, :display_size])
            H_dnn_mag = np.abs(H_dnn[:display_size, :display_size])
            
            X, Y = np.meshgrid(np.arange(display_size), np.arange(display_size))
            
            # True channel
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.plot_surface(X, Y, H_true_mag, cmap='viridis', alpha=0.8)
            ax1.set_title(f'True Channel Magnitude ({display_size}x{display_size})')
            ax1.set_xlabel('Doppler bins')
            ax1.set_ylabel('Delay bins')
            ax1.set_zlabel('Amplitude')
            
            # LS estimation
            ax2 = fig.add_subplot(222, projection='3d')
            ax2.plot_surface(X, Y, H_ls_mag, cmap='plasma', alpha=0.8)
            ax2.set_title(f'LS Estimation ({display_size}x{display_size})')
            ax2.set_xlabel('Doppler bins')
            ax2.set_ylabel('Delay bins')
            ax2.set_zlabel('Amplitude')
            
            # MMSE estimation
            ax3 = fig.add_subplot(223, projection='3d')
            ax3.plot_surface(X, Y, H_mmse_mag, cmap='coolwarm', alpha=0.8)
            ax3.set_title(f'MMSE Estimation ({display_size}x{display_size})')
            ax3.set_xlabel('Doppler bins')
            ax3.set_ylabel('Delay bins')
            ax3.set_zlabel('Amplitude')
            
            # DNN estimation
            ax4 = fig.add_subplot(224, projection='3d')
            ax4.plot_surface(X, Y, H_dnn_mag, cmap='inferno', alpha=0.8)
            ax4.set_title(f'DNN Blind Estimation ({display_size}x{display_size})')
            ax4.set_xlabel('Doppler bins')
            ax4.set_ylabel('Delay bins')
            ax4.set_zlabel('Amplitude')
            
            plt.suptitle(f'Channel Estimation Comparison (Blind DNN)\nSNR={snr_db}dB, Speed={max_speed}km/h, {channel_type}\nPilot Spacing={pilot_spacing}, Edge Guard={edge_guard}', 
                        fontsize=14)
            plt.tight_layout()
            plt.savefig(f'channel_estimation_blind_{vis_idx}_spacing{pilot_spacing}_edge{edge_guard}.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            # Calculate performance metrics using norm-based NMSE
            mse_ls = np.mean(np.abs(H_true - H_ls)**2)
            mse_mmse = np.mean(np.abs(H_true - H_mmse)**2)
            mse_dnn = np.mean(np.abs(H_true - H_dnn)**2)
            
            nmse_ls = self.calculate_nmse(H_true, H_ls)
            nmse_mmse = self.calculate_nmse(H_true, H_mmse)
            nmse_dnn = self.calculate_nmse(H_true, H_dnn)
            
            print(f"Sample {vis_idx+1} Performance (Blind DNN, Norm-based NMSE):")
            print(f"  MSE  - LS: {mse_ls:.2e}, MMSE: {mse_mmse:.2e}, DNN(Blind): {mse_dnn:.2e}")
            print(f"  NMSE - LS: {nmse_ls:.2e}, MMSE: {nmse_mmse:.2e}, DNN(Blind): {nmse_dnn:.2e}")
            print()
    
    def comprehensive_comparison(self, n_train=20000, n_test=1000, epochs=100, batch_size=64, 
                               pilot_spacing=2, edge_guard=1):
        """Run comprehensive comparison with blind DNN estimation"""
        
        validation_results = self.validate_implementation()
        if not all(validation_results.values()):
            print("Warning: Implementation validation failed!")
            for check, passed in validation_results.items():
                status = "PASS" if passed else "FAIL"
                print(f"  {check}: {status}")
            print()
        
        # Visualize current pilot placement
        print("=== Current Pilot Placement ===")
        self.visualize_pilot_placement(0.25, pilot_spacing, edge_guard)
        
        print(f"\n=== Using Blind DNN (no SNR/speed info) ===")
        print(f"Parameters: Pilot Spacing={pilot_spacing}, Edge Guard={edge_guard}")
        print(f"NMSE: Using norm-based normalization ||H_est - H_true||²/||H_true||²")
        
        # Generate training dataset
        dataset = self.generate_training_dataset(
            n_samples=n_train,
            pilot_spacing=pilot_spacing,
            edge_guard=edge_guard
        )
        
        # Train DNN model
        print("Training Blind DNN...")
        self.dnn_model_full = self.build_dnn_model_full(dataset['X_data'].shape[1])
        X_train, X_val, y_train, y_val = train_test_split(
            dataset['X_data'], dataset['y_data_full'],
            test_size=0.2, random_state=42
        )
        
        history = self.dnn_model_full.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.8, patience=10, min_lr=1e-6)
            ],
            verbose=1
        )
        
        # Run performance tests
        test_results = self.run_comparison_test(n_test, pilot_spacing, edge_guard)
        
        result = {
            'test_results': test_results,
            'training_history': history,
            'dataset_info': {
                'input_dim': dataset['X_data'].shape[1],
                'output_dim': y_train.shape[1],
                'n_samples': len(dataset['X_data'])
            },
            'validation_results': validation_results,
            'pilot_spacing': pilot_spacing,
            'edge_guard': edge_guard
        }
        
        print(f"Visualizing blind channel estimation results...")
        self.visualize_channel_estimation_results(n_vis=2, pilot_spacing=pilot_spacing, edge_guard=edge_guard)
        
        return result
    
    def plot_results(self, result):
        """Plot comparison results"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        test_results = result['test_results']
        history = result['training_history']
        pilot_spacing = result['pilot_spacing']
        edge_guard = result['edge_guard']
        
        # Training history
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Blind DNN Training History\n(Spacing={pilot_spacing}, Edge={edge_guard})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # MSE comparison
        axes[1].semilogy(test_results['snr_values'], test_results['mse_ls'], 'o-',
                      label='LS', linewidth=2, markersize=8)
        axes[1].semilogy(test_results['snr_values'], test_results['mse_mmse'], 's-',
                      label='MMSE', linewidth=2, markersize=8)
        axes[1].semilogy(test_results['snr_values'], test_results['mse_dnn'], '^-',
                      label=f'Blind DNN (S={pilot_spacing}, E={edge_guard})', linewidth=2, markersize=8)
        axes[1].set_xlabel('SNR (dB)')
        axes[1].set_ylabel('MSE')
        axes[1].set_title(f'MSE vs SNR (Blind Estimation)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # NMSE comparison (norm-based)
        axes[2].semilogy(test_results['snr_values'], test_results['nmse_ls'], 'o-',
                      label='LS', linewidth=2, markersize=8)
        axes[2].semilogy(test_results['snr_values'], test_results['nmse_mmse'], 's-',
                      label='MMSE', linewidth=2, markersize=8)
        axes[2].semilogy(test_results['snr_values'], test_results['nmse_dnn'], '^-',
                      label=f'Blind DNN (S={pilot_spacing}, E={edge_guard})', linewidth=2, markersize=8)
        axes[2].set_xlabel('SNR (dB)')
        axes[2].set_ylabel('NMSE (Norm-based)')
        axes[2].set_title(f'NMSE vs SNR (||·||²/||H_true||²)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'otfs_blind_norm_results_spacing{pilot_spacing}_edge{edge_guard}.png', 
                   dpi=500, bbox_inches='tight')
        plt.show()


def main():
    """Main function to run the OTFS blind channel estimation comparison"""
    
    # Configuration
    N_DOPPLER = 10
    M_DELAY = 10
    N_TRAIN_SAMPLES = 50000
    N_TEST_SAMPLES = 8000
    EPOCHS = 200
    BATCH_SIZE = 128
    PILOT_SPACING = 1  
    EDGE_GUARD = 2    
    
    print("=== OTFS Blind Channel Estimation (Norm-based NMSE) ===")
    print(f"Configuration:")
    print(f"  OTFS Grid: {N_DOPPLER}x{M_DELAY}")
    print(f"  Training samples: {N_TRAIN_SAMPLES}")
    print(f"  Test samples: {N_TEST_SAMPLES}")
    print(f"  Training epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Pilot spacing (Manhattan distance): {PILOT_SPACING}")
    print(f"  Edge guard interval: {EDGE_GUARD}")
    print(f"  DNN Type: Blind (no SNR/speed information)")
    print(f"  NMSE: Norm-based normalization ||H_est - H_true||²/||H_true||²")
    print()
    
    # Initialize estimator
    estimator = OTFSChannelEstimator(N=N_DOPPLER, M=M_DELAY)
    
    # Run comprehensive comparison
    print(f"Running blind channel estimation comparison...")
    result = estimator.comprehensive_comparison(
        n_train=N_TRAIN_SAMPLES,
        n_test=N_TEST_SAMPLES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        pilot_spacing=PILOT_SPACING,
        edge_guard=EDGE_GUARD
    )
    
    # Plot and save results
    estimator.plot_results(result)
    
    # Save model and results
    with open(f'otfs_blind_norm_results_S{PILOT_SPACING}_E{EDGE_GUARD}.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    if estimator.dnn_model_full is not None:
        estimator.dnn_model_full.save(f'otfs_blind_norm_dnn_S{PILOT_SPACING}_E{EDGE_GUARD}.h5')
    
    # Performance summary
    print("\n=== Blind Estimation Performance Summary (Norm-based NMSE) ===")
    test_results = result['test_results']
    
    if 20 in test_results['snr_values']:
        idx = test_results['snr_values'].index(20)
        print(f"\nAt 20dB SNR (Blind DNN, Pilot Spacing={PILOT_SPACING}, Edge Guard={EDGE_GUARD}):")
        print(f"  MSE: LS={test_results['mse_ls'][idx]:.2e}, MMSE={test_results['mse_mmse'][idx]:.2e}, Blind DNN={test_results['mse_dnn'][idx]:.2e}")
        print(f"  NMSE: LS={test_results['nmse_ls'][idx]:.2e}, MMSE={test_results['nmse_mmse'][idx]:.2e}, Blind DNN={test_results['nmse_dnn'][idx]:.2e}")
        print(f"  Avg Time: LS={np.mean(test_results['time_ls']):.4f}s, MMSE={np.mean(test_results['time_mmse']):.4f}s, Blind DNN={np.mean(test_results['time_dnn']):.4f}s")
        
        # Calculate improvement
        ls_nmse = test_results['nmse_ls'][idx]
        dnn_nmse = test_results['nmse_dnn'][idx]
        improvement_db = 10 * np.log10(ls_nmse / dnn_nmse) if dnn_nmse > 0 else float('inf')
        print(f"  Blind DNN improvement over LS: {improvement_db:.2f} dB")
    
    print(f"\nResults saved to:")
    print(f"  - Main results: otfs_blind_norm_results_S{PILOT_SPACING}_E{EDGE_GUARD}.pkl")
    print(f"  - Model: otfs_blind_norm_dnn_S{PILOT_SPACING}_E{EDGE_GUARD}.h5")
    print(f"  - Plots: Various PNG files with 'blind_norm' prefix")


if __name__ == "__main__":
    main()
