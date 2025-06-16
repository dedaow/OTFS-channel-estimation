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
    
    def generate_center_pilot_frame(self, pilot_density=0.25, guard_interval=1):
        """生成中心集中的导频帧，带保护间隔"""
        
        # 计算中心位置
        center_m = self.M // 2
        center_n = self.N // 2
        
        # 根据导频密度计算需要的导频数量
        total_symbols = self.N * self.M
        n_pilots = int(total_symbols * pilot_density)
        
        # 创建中心区域的导频位置
        pilot_positions = []
        pilot_symbols = []
        
        # 从中心开始螺旋式放置导频
        directions = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), 
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        # 先放置中心导频
        center_pos = center_m * self.N + center_n
        pilot_positions.append(center_pos)
        pilot_symbols.append((1 + 1j) / np.sqrt(2))  # QPSK
        
        # 逐步向外扩展，保持保护间隔
        radius = 1
        pilot_count = 1
        
        while pilot_count < n_pilots and radius <= max(center_m, center_n):
            for dm in range(-radius, radius + 1):
                for dn in range(-radius, radius + 1):
                    if pilot_count >= n_pilots:
                        break
                    
                    # 只在边界上放置（螺旋式）
                    if abs(dm) == radius or abs(dn) == radius:
                        m_pos = center_m + dm
                        n_pos = center_n + dn
                        
                        # 检查是否在有效范围内
                        if (guard_interval <= m_pos < self.M - guard_interval and 
                            guard_interval <= n_pos < self.N - guard_interval):
                            
                            pos = m_pos * self.N + n_pos
                            
                            # 检查与已有导频的最小距离（保护间隔）
                            valid_position = True
                            for existing_pos in pilot_positions:
                                existing_m = existing_pos // self.N
                                existing_n = existing_pos % self.N
                                distance = max(abs(m_pos - existing_m), abs(n_pos - existing_n))
                                if distance < guard_interval:
                                    valid_position = False
                                    break
                            
                            if valid_position:
                                pilot_positions.append(pos)
                                # 循环使用QPSK符号
                                qpsk_symbols = [(1+1j)/np.sqrt(2), (1-1j)/np.sqrt(2), 
                                              (-1+1j)/np.sqrt(2), (-1-1j)/np.sqrt(2)]
                                pilot_symbols.append(qpsk_symbols[pilot_count % 4])
                                pilot_count += 1
                
                if pilot_count >= n_pilots:
                    break
            radius += 1
        
        # 转换为numpy数组
        pilot_positions = np.array(pilot_positions)
        pilot_symbols = np.array(pilot_symbols)
        
        # 创建帧
        frame_symbols = np.zeros(self.M * self.N, dtype=complex)
        frame_symbols[pilot_positions] = pilot_symbols
        
        # 重新整形为 MxN 帧
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
    
    def extract_dnn_features(self, Y, pilot_pos, pilot_syms, snr_db, max_speed, pilot_density):
        """Extract features for DNN training"""
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
        
        # System parameters (normalized)
        features.append([
            snr_db / 35.0,
            max_speed / 600.0,
            pilot_density,
            len(pilot_pos) / (self.N * self.M),
            self.fc / 1e10,
            self.delta_f / 1e5
        ])
        
        # Signal statistics
        signal_power = np.mean(np.abs(Y_flat)**2)
        signal_mean = np.mean(Y_flat)
        pilot_power = np.mean(np.abs(pilot_syms)**2) if len(pilot_syms) > 0 else 0
        
        features.append([
            np.real(signal_mean), np.imag(signal_mean),
            signal_power, np.std(np.abs(Y_flat)),
            pilot_power
        ])
        
        # Frequency domain features
        Y_fft = np.fft.fft(Y_flat)
        features.extend([np.real(Y_fft), np.imag(Y_fft)])
        
        return np.concatenate(features)
    
    def improved_ls_estimation(self, Y, pilot_pos, pilot_syms):
        """Improved LS estimation with center pilot interpolation"""
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
                                guard_interval=1):
        """Generate training dataset with center pilots"""
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
            
            # Random parameters
            snr_db = np.random.uniform(snr_range[0], snr_range[1])
            max_speed = np.random.uniform(speed_range[0], speed_range[1])
            channel_type = np.random.choice(['EPA', 'EVA', 'ETU'])
            pilot_density = np.random.choice(pilot_densities)
            
            # Generate channel
            H_true, G_true, gs_true = self.generate_channel_matrix(max_speed, channel_type)
            
            # Generate center pilot frame
            X_pilot, pilot_pos, pilot_syms = self.generate_center_pilot_frame(
                pilot_density, guard_interval)
            
            # OTFS modulation
            s, x = self.otfs_modulation(X_pilot)
            
            # Channel transmission
            r_clean = H_true @ x.reshape(-1, 1)
            r_noisy, sigma_w_2 = self.add_awgn(r_clean.flatten(), snr_db)
            
            # OTFS demodulation
            Y_noisy, y_vec = self.otfs_demodulation(r_noisy)
            
            # Extract features
            features = self.extract_dnn_features(Y_noisy, pilot_pos, pilot_syms,
                                               snr_db, max_speed, pilot_density)
            
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
    
    def dnn_estimation(self, Y, pilot_pos, pilot_syms, snr_db, max_speed, pilot_density):
        """DNN-based channel estimation"""
        features = self.extract_dnn_features(Y, pilot_pos, pilot_syms, snr_db, max_speed, pilot_density)
        features = features.reshape(1, -1)
        
        if self.dnn_model_full is None:
            raise ValueError("Full DNN model not trained")
        
        prediction = self.dnn_model_full.predict(features, verbose=0)
        n_elements = self.N * self.M
        real_part = prediction[0, :n_elements**2]
        imag_part = prediction[0, n_elements**2:]
        
        return (real_part + 1j * imag_part).reshape(n_elements, n_elements)
    
    def run_comparison_test(self, n_test=1000, guard_interval=1):
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
                    pilot_density, guard_interval)
                
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
                    H_dnn = self.dnn_estimation(Y_pilot, pilot_pos, pilot_syms, snr_db, max_speed, pilot_density)
                except:
                    H_dnn = H_ls  # Fallback
                time_dnn = time.time() - start_time
                
                # MSE and NMSE calculation
                target = H_true.flatten()
                est_ls = H_ls.flatten()
                est_mmse = H_mmse.flatten()
                est_dnn = H_dnn.flatten()
                
                mse_ls = np.mean(np.abs(target - est_ls)**2)
                mse_mmse = np.mean(np.abs(target - est_mmse)**2)
                mse_dnn = np.mean(np.abs(target - est_dnn)**2)
                
                power_true = np.mean(np.abs(target)**2)
                nmse_ls = mse_ls / power_true if power_true > 0 else float('inf')
                nmse_mmse = mse_mmse / power_true if power_true > 0 else float('inf')
                nmse_dnn = mse_dnn / power_true if power_true > 0 else float('inf')
                
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
    
    def visualize_pilot_placement(self, pilot_density=0.25, guard_interval=1):
        """可视化中心导频位置"""
        
        X_pilot, pilot_pos, pilot_syms = self.generate_center_pilot_frame(pilot_density, guard_interval)
        
        # 创建可视化
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # 创建导频位置网格
        pilot_grid = np.zeros((self.M, self.N))
        for pos in pilot_pos:
            m_idx = pos // self.N
            n_idx = pos % self.N
            pilot_grid[m_idx, n_idx] = 1
        
        # 绘制网格
        im = ax.imshow(pilot_grid, cmap='Blues', origin='lower', extent=[0, self.N, 0, self.M])
        
        # 标记中心位置
        center_m = self.M // 2
        center_n = self.N // 2
        ax.plot(center_n + 0.5, center_m + 0.5, 'r*', markersize=15, label='Center')
        
        # 标记保护间隔边界
        ax.axhline(y=guard_interval, color='red', linestyle='--', alpha=0.7, label='Guard Interval')
        ax.axhline(y=self.M - guard_interval, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=guard_interval, color='red', linestyle='--', alpha=0.7)
        ax.axvline(x=self.N - guard_interval, color='red', linestyle='--', alpha=0.7)
        
        ax.set_title(f'Center Pilot Placement\n(Density: {pilot_density:.2f}, Guard: {guard_interval})')
        ax.set_xlabel('Doppler (bins)')
        ax.set_ylabel('Delay (bins)')
        ax.legend()
        
        # 添加网格线
        ax.set_xticks(np.arange(0.5, self.N + 0.5, 1), minor=True)
        ax.set_yticks(np.arange(0.5, self.M + 0.5, 1), minor=True)
        ax.grid(which='minor', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'center_pilot_placement_guard{guard_interval}.png', dpi=300)
        plt.show()
        
        print(f"Center pilot information:")
        print(f"  Total pilots: {len(pilot_pos)} / {self.N*self.M}")
        print(f"  Pilot density: {len(pilot_pos)/(self.N*self.M):.3f}")
        print(f"  Guard interval: {guard_interval}")
    
    def visualize_channel_estimation_results(self, n_vis=1, guard_interval=1):
        """可视化信道估计结果对比"""
        
        print(f"Visualizing channel estimation results...")
        
        for vis_idx in range(n_vis):
            # 生成测试场景
            max_speed = 200  # km/h
            channel_type = 'EVA'
            pilot_density = 0.25
            snr_db = 20
            
            # 生成真实信道
            H_true, G_true, gs_true = self.generate_channel_matrix(max_speed, channel_type)
            
            # 生成中心导频帧
            X_pilot, pilot_pos, pilot_syms = self.generate_center_pilot_frame(
                pilot_density, guard_interval)
            
            # OTFS调制和传输
            s_pilot, x_pilot = self.otfs_modulation(X_pilot)
            r_pilot_clean = H_true @ x_pilot.reshape(-1, 1)
            r_pilot_noisy, sigma_w_2 = self.add_awgn(r_pilot_clean.flatten(), snr_db)
            Y_pilot, _ = self.otfs_demodulation(r_pilot_noisy)
            
            # 信道估计
            H_ls = self.improved_ls_estimation(Y_pilot, pilot_pos, pilot_syms)
            H_mmse = self.traditional_mmse_estimation(Y_pilot, pilot_pos, pilot_syms, sigma_w_2)
            
            try:
                H_dnn = self.dnn_estimation(Y_pilot, pilot_pos, pilot_syms, snr_db, max_speed, pilot_density)
            except:
                H_dnn = H_ls  # Fallback
            
            # 可视化 - 使用固定的100x100网格
            fig = plt.figure(figsize=(16, 12))
            
            # 准备数据：取信道矩阵的幅度
            H_true_mag = np.abs(H_true[:100, :100])  # 取前100x100部分
            H_ls_mag = np.abs(H_ls[:100, :100])
            H_mmse_mag = np.abs(H_mmse[:100, :100])
            H_dnn_mag = np.abs(H_dnn[:100, :100])
            
            # 创建网格
            X, Y = np.meshgrid(np.arange(100), np.arange(100))
            
            # 真实信道幅度
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.plot_surface(X, Y, H_true_mag, cmap='viridis', alpha=0.8)
            ax1.set_title('True Channel Magnitude (100x100)')
            ax1.set_xlabel('Doppler bins')
            ax1.set_ylabel('Delay bins')
            ax1.set_zlabel('Amplitude')
            
            # LS估计
            ax2 = fig.add_subplot(222, projection='3d')
            ax2.plot_surface(X, Y, H_ls_mag, cmap='plasma', alpha=0.8)
            ax2.set_title('LS Estimation (100x100)')
            ax2.set_xlabel('Doppler bins')
            ax2.set_ylabel('Delay bins')
            ax2.set_zlabel('Amplitude')
            
            # MMSE估计
            ax3 = fig.add_subplot(223, projection='3d')
            ax3.plot_surface(X, Y, H_mmse_mag, cmap='coolwarm', alpha=0.8)
            ax3.set_title('MMSE Estimation (100x100)')
            ax3.set_xlabel('Doppler bins')
            ax3.set_ylabel('Delay bins')
            ax3.set_zlabel('Amplitude')
            
            # DNN估计
            ax4 = fig.add_subplot(224, projection='3d')
            ax4.plot_surface(X, Y, H_dnn_mag, cmap='inferno', alpha=0.8)
            ax4.set_title(f'DNN Estimation (FULL, 100x100)')
            ax4.set_xlabel('Doppler bins')
            ax4.set_ylabel('Delay bins')
            ax4.set_zlabel('Amplitude')
            
            plt.suptitle(f'Channel Estimation Comparison\nSNR={snr_db}dB, Speed={max_speed}km/h, {channel_type}', 
                        fontsize=14)
            plt.tight_layout()
            plt.savefig(f'channel_estimation_comparison_full_{vis_idx}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 计算性能指标
            mse_ls = np.mean(np.abs(H_true - H_ls)**2)
            mse_mmse = np.mean(np.abs(H_true - H_mmse)**2)
            mse_dnn = np.mean(np.abs(H_true - H_dnn)**2)
            
            power_true = np.mean(np.abs(H_true)**2)
            nmse_ls = mse_ls / power_true
            nmse_mmse = mse_mmse / power_true
            nmse_dnn = mse_dnn / power_true
            
            print(f"Sample {vis_idx+1} Performance:")
            print(f"  MSE  - LS: {mse_ls:.2e}, MMSE: {mse_mmse:.2e}, DNN: {mse_dnn:.2e}")
            print(f"  NMSE - LS: {nmse_ls:.2e}, MMSE: {nmse_mmse:.2e}, DNN: {nmse_dnn:.2e}")
            print()
    
    def comprehensive_comparison(self, n_train=20000, n_test=1000, epochs=100, batch_size=64, guard_interval=1):
        """Run comprehensive comparison with center pilots - Full mode only"""
        
        validation_results = self.validate_implementation()
        if not all(validation_results.values()):
            print("Warning: Implementation validation failed!")
            for check, passed in validation_results.items():
                status = "PASS" if passed else "FAIL"
                print(f"  {status}: {check}")
            print()
        
        # Visualize pilot placement
        print("=== Center Pilot Placement ===")
        self.visualize_pilot_placement(0.25, guard_interval)
        
        print(f"\n=== FULL Mode Only ===")
        
        # Generate training dataset
        dataset = self.generate_training_dataset(
            n_samples=n_train,
            guard_interval=guard_interval
        )
        
        # Train DNN model
        print("Training DNN...")
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
        test_results = self.run_comparison_test(n_test, guard_interval)
        
        result = {
            'test_results': test_results,
            'training_history': history,
            'dataset_info': {
                'input_dim': dataset['X_data'].shape[1],
                'output_dim': y_train.shape[1],
                'n_samples': len(dataset['X_data'])
            },
            'validation_results': validation_results
        }
        
        # 可视化信道估计结果
        print(f"Visualizing full channel estimation results...")
        self.visualize_channel_estimation_results(n_vis=2, guard_interval=guard_interval)
        
        return result
    
    def plot_results(self, result):
        """Plot comparison results"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        test_results = result['test_results']
        history = result['training_history']
        
        # Training history
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'FULL: Training History (Center Pilots)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # MSE comparison
        axes[1].semilogy(test_results['snr_values'], test_results['mse_ls'], 'o-',
                      label='LS', linewidth=2, markersize=8)
        axes[1].semilogy(test_results['snr_values'], test_results['mse_mmse'], 's-',
                      label='MMSE', linewidth=2, markersize=8)
        axes[1].semilogy(test_results['snr_values'], test_results['mse_dnn'], '^-',
                      label='DNN (Center Pilots)', linewidth=2, markersize=8)
        axes[1].set_xlabel('SNR (dB)')
        axes[1].set_ylabel('MSE')
        axes[1].set_title(f'FULL: MSE vs SNR')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # NMSE comparison
        axes[2].semilogy(test_results['snr_values'], test_results['nmse_ls'], 'o-',
                      label='LS', linewidth=2, markersize=8)
        axes[2].semilogy(test_results['snr_values'], test_results['nmse_mmse'], 's-',
                      label='MMSE', linewidth=2, markersize=8)
        axes[2].semilogy(test_results['snr_values'], test_results['nmse_dnn'], '^-',
                      label='DNN (Center Pilots)', linewidth=2, markersize=8)
        axes[2].set_xlabel('SNR (dB)')
        axes[2].set_ylabel('NMSE')
        axes[2].set_title(f'FULL: NMSE vs SNR')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('otfs_center_pilot_results_full.png', dpi=500, bbox_inches='tight')
        plt.show()


def main():
    
    # Configuration
    N_DOPPLER = 10
    M_DELAY = 10
    N_TRAIN_SAMPLES = 20000
    N_TEST_SAMPLES = 4000
    EPOCHS = 200
    BATCH_SIZE = 128
    GUARD_INTERVAL = 2  
    
    print("=== OTFS Channel Estimation (Center Pilots + Guard Interval) - FULL MODE ONLY ===")
    print(f"Configuration:")
    print(f"  OTFS Grid: {N_DOPPLER}x{M_DELAY}")
    print(f"  Training samples: {N_TRAIN_SAMPLES}")
    print(f"  Test samples: {N_TEST_SAMPLES}")
    print(f"  Training epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Guard interval: {GUARD_INTERVAL}")
    print()
    
    # Initialize estimator
    estimator = OTFSChannelEstimator(N=N_DOPPLER, M=M_DELAY)
    
    # Run comprehensive comparison
    result = estimator.comprehensive_comparison(
        n_train=N_TRAIN_SAMPLES,
        n_test=N_TEST_SAMPLES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        guard_interval=GUARD_INTERVAL
    )
    
    # Plot and save results
    estimator.plot_results(result)
    
    # Save model and results
    with open('otfs_center_pilot_results_full.pkl', 'wb') as f:
        pickle.dump(result, f)
    
    if estimator.dnn_model_full is not None:
        estimator.dnn_model_full.save('otfs_dnn_full_center_pilot.h5')
    
    # Performance summary
    print("\n=== Performance Summary (Center Pilots - FULL MODE) ===")
    test_results = result['test_results']
    
    if 20 in test_results['snr_values']:
        idx = test_results['snr_values'].index(20)
        print(f"\nAt 20dB SNR:")
        print(f"  MSE: LS={test_results['mse_ls'][idx]:.2e}, MMSE={test_results['mse_mmse'][idx]:.2e}, DNN={test_results['mse_dnn'][idx]:.2e}")
        print(f"  NMSE: LS={test_results['nmse_ls'][idx]:.2e}, MMSE={test_results['nmse_mmse'][idx]:.2e}, DNN={test_results['nmse_dnn'][idx]:.2e}")
        print(f"  Avg Time: LS={np.mean(test_results['time_ls']):.4f}s, MMSE={np.mean(test_results['time_mmse']):.4f}s, DNN={np.mean(test_results['time_dnn']):.4f}s")
        
        # Calculate improvement
        ls_nmse = test_results['nmse_ls'][idx]
        dnn_nmse = test_results['nmse_dnn'][idx]
        improvement_db = 10 * np.log10(ls_nmse / dnn_nmse) if dnn_nmse > 0 else float('inf')
        print(f"  DNN improvement over LS: {improvement_db:.2f} dB")
    
    print(f"\nResults saved to:")
    print(f"  - Plots: otfs_center_pilot_results_full.png")
    print(f"  - Data: otfs_center_pilot_results_full.pkl") 
    print(f"  - Model: otfs_dnn_full_center_pilot.h5")


if __name__ == "__main__":
    main()
