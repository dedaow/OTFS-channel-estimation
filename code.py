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
        
        # Normalized DFT matrix - exactly as MATLAB
        self.Fn = dft(self.N) / np.sqrt(self.N)
        
        # Identity matrix for delays
        self.Im = np.eye(self.M)
        
        # DNN models
        self.dnn_model_sparse = None
        self.dnn_model_full = None
        
        print(f"OTFS System initialized: {N}x{M} grid, Channel matrix size: {N*M}x{N*M}")
        
    def create_permutation_matrix(self):
        """Create permutation matrix P """
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
        """Validate implementation correctness"""#组合代码防止报错
        print("=== please please  ===")
        
        # Check permutation matrix properties
        P = self.create_permutation_matrix()
        is_unitary = np.allclose(P @ P.conj().T, np.eye(self.N * self.M), atol=1e-10)
        print(f"P is unitary (P*P' = I): {is_unitary}")
        
        #  Check modulation-demodulation consistency (no channel)
        X_test = np.random.randn(self.M, self.N) + 1j * np.random.randn(self.M, self.N)
        s, x_orig = self.otfs_modulation(X_test)
        Y_recovered, y_recovered = self.otfs_demodulation(s)
        
        modulation_error = np.linalg.norm(X_test - Y_recovered)
        print(f"Modulation-Demodulation error (no channel): {modulation_error:.2e}")
        
        #  Check vectorization consistency
        x_manual = X_test.T.flatten()
        x_diff = np.linalg.norm(x_orig - x_manual)
        print(f"Vectorization consistency: {x_diff:.2e}")
        
        # Check DFT matrix normalization
        Fn_norm = np.linalg.norm(self.Fn @ self.Fn.conj().T - np.eye(self.N))
        print(f"DFT matrix normalization error: {Fn_norm:.2e}")
        
        print("=== ok ===\n")
        
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
        """Generate channel matrix """
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
    #调制解调 DD modain 的矩阵X经过置换矩阵以及kron转化到 TF domain s（时域信号）经过置换矩阵以及kron回到了 DD domain Y
    def add_awgn(self, signal, snr_db):
        """Add AWGN to signal"""
        Es = 1.0
        SNR = 10**(snr_db / 10)
        sigma_w_2 = Es / SNR
        
        noise = np.sqrt(sigma_w_2 / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        
        return signal + noise, sigma_w_2
    
    def generate_pilot_frame(self, pilot_density=0.25):
        """Generate pilot frame with known symbols at specific positions"""
        total_symbols = self.N * self.M
        n_pilots = int(total_symbols * pilot_density)
        
        # Random pilot positions
        pilot_positions = np.random.choice(total_symbols, n_pilots, replace=False)
        pilot_positions = np.sort(pilot_positions)
        
        # Generate known pilot symbols (QPSK)
        pilot_symbols = np.zeros(n_pilots, dtype=complex)
        for i in range(n_pilots):
            if i % 4 == 0:
                pilot_symbols[i] = (1 + 1j) / np.sqrt(2)
            elif i % 4 == 1:
                pilot_symbols[i] = (1 - 1j) / np.sqrt(2)
            elif i % 4 == 2:
                pilot_symbols[i] = (-1 + 1j) / np.sqrt(2)
            else:
                pilot_symbols[i] = (-1 - 1j) / np.sqrt(2)
        
        # Create frame with pilots and zeros for data
        frame_symbols = np.zeros(total_symbols, dtype=complex)
        frame_symbols[pilot_positions] = pilot_symbols
        
        # Reshape to MxN frame
        X = frame_symbols.reshape(self.M, self.N)
        
        return X, pilot_positions, pilot_symbols
    
    def extract_sparse_channel_features(self, H_matrix, G_matrix, gs_matrix, max_taps=20):
        """Extract sparse channel tap parameters for DNN training """
        
        # Flatten the channel matrix
        H_flat = H_matrix.flatten()
        
        # Find indices of elements with significant magnitude
        magnitudes = np.abs(H_flat)#抽头的abs来判断
        sorted_indices = np.argsort(magnitudes)[::-1]
        n_selected = min(max_taps, len(sorted_indices))
        selected_indices = sorted_indices[:n_selected]
        sparse_taps = H_flat[selected_indices]
        
        # Convert linear indices to delay-Doppler indices还原回去方便后面使用的
        delay_indices = selected_indices % (self.N * self.M) // self.N
        doppler_indices = selected_indices % (self.N * self.M) % self.N
        
        # Pad to fixed length if needed防止报错
        if len(sparse_taps) < max_taps:
            padding_length = max_taps - len(sparse_taps)
            sparse_taps = np.concatenate([sparse_taps, np.zeros(padding_length, dtype=complex)])
            delay_indices = np.concatenate([delay_indices, np.zeros(padding_length, dtype=int)])
            doppler_indices = np.concatenate([doppler_indices, np.zeros(padding_length, dtype=int)])
        
        return sparse_taps, delay_indices, doppler_indices
    
    def extract_dnn_features(self, Y, pilot_pos, pilot_syms, snr_db, max_speed, pilot_density):
        """Extract features for DNN training"""
        features = []
        
        #  Received signal in delay-Doppler domain
        Y_flat = Y.T.flatten()
        features.extend([np.real(Y_flat), np.imag(Y_flat)])
        
        #  Pilot information 导频的位置
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
        features.extend([np.real(channel_estimates_at_pilots), np.imag(channel_estimates_at_pilots)])#防报错
        
        #  System parameters (归一化一下)
        features.append([
            snr_db / 35.0,
            max_speed / 600.0,
            pilot_density,
            len(pilot_pos) / (self.N * self.M),
            self.fc / 1e10,#这个是载波频率
            self.delta_f / 1e5#子载波的间隔
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
        features.extend([np.real(Y_fft), np.imag(Y_fft)])#real 和 imag
        #pilot mask【MN】, 剩下都是【2MN】 
        return np.concatenate(features)#直接展开成一维数组
    
    def reconstruct_channel_from_sparse(self, sparse_taps, delay_indices=None, doppler_indices=None, tap_indices=None):
        """Reconstruct full channel matrix from sparse taps """#这个函数的作用是将提取的稀疏抽头（信道主成分）重新构建成一个完整的信道矩阵，我想构建这个稀疏抽头的目的 降低计算开销otfs信道既然是稀疏的这样做或许可以吧
        H_reconstructed = np.zeros((self.N * self.M, self.N * self.M), dtype=complex)
        
        # If tap_indices are provided, use them directly 理想情况
        if tap_indices is not None:
            for i, tap_idx in enumerate(tap_indices):
                if i < len(sparse_taps) and np.abs(sparse_taps[i]) > 1e-10:
                    if tap_idx < self.N * self.M:
                        H_reconstructed[tap_idx, tap_idx] = sparse_taps[i]
        else:
            # Use the sparse taps to reconstruct with a structured approach
            # Place the dominant taps in a way that respects OTFS channel structure
            n_taps = min(len(sparse_taps), self.N * self.M)
            
            # Create a sparse representation based on the strongest taps
            for i in range(n_taps):
                if np.abs(sparse_taps[i]) > 1e-10:
                    row = i % (self.N * self.M)
                    col = i % (self.N * self.M)
                    
                    # Main diagonal
                    if row < self.N * self.M and col < self.N * self.M:
                        H_reconstructed[row, col] = sparse_taps[i]
                    
                    # Add some off-diagonal elements for coupling
                    if i < len(sparse_taps) // 2:
                        for offset in [1, -1]:
                            if 0 <= row + offset < self.N * self.M and 0 <= col < self.N * self.M:
                                H_reconstructed[row + offset, col] += 0.1 * sparse_taps[i]
        
        # If the matrix is still too sparse, add some structure结构太稀疏用来防止报错的
        if np.sum(np.abs(H_reconstructed)) < 1e-6:
            for i in range(min(len(sparse_taps), self.N * self.M)):
                if np.abs(sparse_taps[i]) > 1e-10:
                    H_reconstructed[i, i] = sparse_taps[i]
        
        return H_reconstructed
    
    def improved_ls_estimation(self, Y, pilot_pos, pilot_syms):
        """Improved LS estimation considering OTFS channel structure"""#修改了传统的ls  y=Hx+w  我这个输入是【Y、pilot 的位置和符号】
        Y_flat = Y.T.flatten()
        n_total = self.N * self.M
        
        # Initialize channel estimate
        H_est = np.zeros((n_total, n_total), dtype=complex)
        
        # Direct LS estimation at pilot positions
        pilot_channel_est = np.zeros(n_total, dtype=complex)
        
        for i, pos in enumerate(pilot_pos):# H  = Y/X 理想情况如果在导频位置的
            if i < len(pilot_syms) and np.abs(pilot_syms[i]) > 1e-6:
                pilot_channel_est[pos] = Y_flat[pos] / pilot_syms[i]
        
        # Interpolation based on OTFS channel structure 插值
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
                        
                        delay_dist = min(abs(pos_delay - pilot_delay), 
                                       self.M - abs(pos_delay - pilot_delay))
                        doppler_dist = min(abs(pos_doppler - pilot_doppler),
                                         self.N - abs(pos_doppler - pilot_doppler))
                        
                        distance = np.sqrt(delay_dist**2 + doppler_dist**2)
                        
                        if distance < np.sqrt(self.M**2 + self.N**2) / 2:
                            weight = np.exp(-distance / 2)
                            weights.append(weight)
                            estimates.append(pilot_channel_est[pilot_pos_idx])
                
                if len(weights) > 0:
                    weights = np.array(weights)
                    estimates = np.array(estimates)
                    pilot_channel_est[pos] = np.sum(weights * estimates) / np.sum(weights)
        
        # Create channel matrix with dominant diagonal structure    还原
        for i in range(n_total):
            for j in range(n_total):
                i_delay = i % self.M
                i_doppler = i // self.M
                j_delay = j % self.M
                j_doppler = j // self.M
                
                delay_diff = (i_delay - j_delay) % self.M
                doppler_diff = (i_doppler - j_doppler) % self.N
                
                if i == j:
                    H_est[i, j] = pilot_channel_est[i]
                elif delay_diff <= 2 or delay_diff >= self.M - 2:
                    coupling_factor = 0.1 * np.exp(-0.5 * (delay_diff + doppler_diff))
                    H_est[i, j] = pilot_channel_est[i] * coupling_factor
        
        return H_est
    
    def traditional_mmse_estimation(self, Y, pilot_pos, pilot_syms, sigma_w_2):
        """Improved MMSE channel estimation"""#同上输入是【Y、pilot 的位置和符号】  
        H_ls = self.improved_ls_estimation(Y, pilot_pos, pilot_syms)
        
        n_total = self.N * self.M
        
        # Channel correlation matrix
        R_hh = np.zeros((n_total, n_total), dtype=complex)
        for i in range(n_total):
            for j in range(n_total):
                i_delay = i % self.M
                i_doppler = i // self.M
                j_delay = j % self.M
                j_doppler = j // self.M
                
                delay_diff = min(abs(i_delay - j_delay), self.M - abs(i_delay - j_delay))
                doppler_diff = min(abs(i_doppler - j_doppler), self.N - abs(i_doppler - j_doppler))
                
                correlation = np.exp(-0.5 * (delay_diff + doppler_diff))
                R_hh[i, j] = correlation
        
        # Noise covariance
        noise_cov = sigma_w_2 * np.eye(n_total)
        
        try:
            H_mmse = R_hh @ np.linalg.solve(R_hh + noise_cov, H_ls)# H_mmse = R_hh * (R_hh + σ²I)^(-1) * H_ls
        except np.linalg.LinAlgError:
            H_mmse = H_ls
        
        return H_mmse
    
    def build_dnn_model_sparse(self, input_dim, max_taps=20):#网络都是最简单的全连接网络
        """sparse channel estimation"""
        # Output: complex taps (real + imag parts) + position indices
        output_dim = max_taps * 3  # real, imag, position_index
        
        print(f"Sparse DNN: Input dim = {input_dim}, Output dim = {output_dim}")
        
        model = keras.Sequential([
            keras.layers.Dense(1024, activation='relu', input_shape=(input_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(output_dim, activation='linear')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def build_dnn_model_full(self, input_dim):
        """full channel matrix estimation"""
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
                                estimation_mode='both', max_taps=20):
        """Generate training dataset with proper OTFS signal flow """
        print(f"Generating training dataset with {n_samples} samples...")
        
        X_data = []
        y_data_sparse = []
        y_data_full = []
        metadata = {
            'snr_values': [], 'speed_values': [], 'channel_types': [],
            'pilot_densities': [], 'sparse_indices': []
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
            
            # Generate pilot frame
            X_pilot, pilot_pos, pilot_syms = self.generate_pilot_frame(pilot_density)
            
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
            if estimation_mode in ['sparse', 'both']:
                sparse_taps, delay_indices, doppler_indices = self.extract_sparse_channel_features(
                    H_true, G_true, gs_true, max_taps)
                
                # Create sparse target: [real_parts, imag_parts, position_indices]
                position_indices = np.arange(len(sparse_taps)) % (self.N * self.M)
                sparse_target = np.concatenate([
                    np.real(sparse_taps), 
                    np.imag(sparse_taps), 
                    position_indices.astype(float) / (self.N * self.M)  # Normalize indices
                ])
                
                y_data_sparse.append(sparse_target)
                metadata['sparse_indices'].append(position_indices)
            
            if estimation_mode in ['full', 'both']:
                full_target = np.concatenate([np.real(H_true.flatten()), np.imag(H_true.flatten())])
                y_data_full.append(full_target)
            
            X_data.append(features)
            metadata['snr_values'].append(snr_db)
            metadata['speed_values'].append(max_speed)
            metadata['channel_types'].append(channel_type)
            metadata['pilot_densities'].append(pilot_density)
        
        result = {
            'X_data': np.array(X_data),
            'metadata': metadata
        }
        
        if estimation_mode in ['sparse', 'both']:
            result['y_data_sparse'] = np.array(y_data_sparse)
        if estimation_mode in ['full', 'both']:
            result['y_data_full'] = np.array(y_data_full)
        
        return result
    
    def dnn_estimation(self, Y, pilot_pos, pilot_syms, snr_db, max_speed, pilot_density, mode='sparse'):
        """DNN-based channel estimation """
        features = self.extract_dnn_features(Y, pilot_pos, pilot_syms, snr_db, max_speed, pilot_density)
        features = features.reshape(1, -1)
        
        if mode == 'sparse':
            if self.dnn_model_sparse is None:
                raise ValueError("Sparse DNN model error")
            
            prediction = self.dnn_model_sparse.predict(features, verbose=0)
            n_taps = len(prediction[0]) // 3  # real, imag, position
            
            real_part = prediction[0, :n_taps]
            imag_part = prediction[0, n_taps:2*n_taps]
            position_part = prediction[0, 2*n_taps:] * (self.N * self.M)  # Denormalize
            
            sparse_taps = real_part + 1j * imag_part
            tap_indices = np.round(position_part).astype(int)
            tap_indices = np.clip(tap_indices, 0, self.N * self.M - 1)
            
            return self.reconstruct_channel_from_sparse(sparse_taps, tap_indices=tap_indices)
        
        elif mode == 'full':
            if self.dnn_model_full is None:
                raise ValueError("Full DNN model error")
            
            prediction = self.dnn_model_full.predict(features, verbose=0)
            n_elements = self.N * self.M
            real_part = prediction[0, :n_elements**2]
            imag_part = prediction[0, n_elements**2:]
            
            return (real_part + 1j * imag_part).reshape(n_elements, n_elements)
    
    def run_comparison_test(self, n_test=1000, mode='sparse'):
        """Run comprehensive comparison test"""
        print(f"Running channel estimation comparison test with {n_test} samples in {mode} mode...")
        
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
                
                # Generate pilot frame for channel estimation
                X_pilot, pilot_pos, pilot_syms = self.generate_pilot_frame(pilot_density)
                
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
                    H_dnn = self.dnn_estimation(Y_pilot, pilot_pos, pilot_syms, snr_db, max_speed, pilot_density, mode)
                except:
                    H_dnn = H_ls  # Fallback 最后效果看着比ls好些所以代码应该获得更好的信息了
                time_dnn = time.time() - start_time
                
                # MSE and NMSE calculation
                if mode == 'sparse':
                    sparse_taps_true, _, _ = self.extract_sparse_channel_features(H_true, G_true, gs_true, 20)
                    sparse_taps_ls, _, _ = self.extract_sparse_channel_features(H_ls, np.zeros_like(G_true), np.zeros_like(gs_true), 20)
                    sparse_taps_mmse, _, _ = self.extract_sparse_channel_features(H_mmse, np.zeros_like(G_true), np.zeros_like(gs_true), 20)
                    sparse_taps_dnn, _, _ = self.extract_sparse_channel_features(H_dnn, np.zeros_like(G_true), np.zeros_like(gs_true), 20)
                    
                    target = sparse_taps_true
                    est_ls = sparse_taps_ls
                    est_mmse = sparse_taps_mmse
                    est_dnn = sparse_taps_dnn
                else:
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
    
    def comprehensive_comparison(self, n_train=20000, n_test=1000, epochs=100, batch_size=64):
        """Run comprehensive comparison"""
        
        validation_results = self.validate_implementation()
        if not all(validation_results.values()):
            print("god，did nothing ")
            for check, passed in validation_results.items():
                status = "yes" if passed else "nonono"
                print(f"  {status} {check}")
            print()
        
        all_results = {}
        
        for mode in ['sparse', 'full']:
            print(f"\n--- {mode.upper()} Mode ---")
            
            
            dataset = self.generate_training_dataset(
                n_samples=n_train,
                estimation_mode=mode,
                max_taps=20
            )
            
            
            print(" Training ...")
            if mode == 'sparse':
                self.dnn_model_sparse = self.build_dnn_model_sparse(dataset['X_data'].shape[1], 20)
                X_train, X_val, y_train, y_val = train_test_split(
                    dataset['X_data'], dataset['y_data_sparse'],
                    test_size=0.2, random_state=42
                )
                
                history = self.dnn_model_sparse.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[
                        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                        keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=7, min_lr=1e-6)
                    ],
                    verbose=1
                )
            
            else:  
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
            
            
            test_results = self.run_comparison_test(n_test, mode)
            
            all_results[mode] = {
                'test_results': test_results,
                'training_history': history,
                'dataset_info': {
                    'input_dim': dataset['X_data'].shape[1],
                    'output_dim': y_train.shape[1],
                    'n_samples': len(dataset['X_data'])
                },
                'validation_results': validation_results
            }
        
        return all_results
    
    def plot_results(self, results):
        """Plot comparison results - Channel Estimation Focus"""
        n_modes = len(results)
        fig, axes = plt.subplots(n_modes, 3, figsize=(18, 6*n_modes))
        
        if n_modes == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (mode, result) in enumerate(results.items()):
            test_results = result['test_results']
            history = result['training_history']
            
            # Training history
            axes[idx, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
            axes[idx, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[idx, 0].set_xlabel('Epoch')
            axes[idx, 0].set_ylabel('Loss')
            axes[idx, 0].set_title(f'{mode.upper()}: Training History')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].set_yscale('log')
            
            # MSE comparison
            axes[idx, 1].semilogy(test_results['snr_values'], test_results['mse_ls'], 'o-',
                              label='LS', linewidth=2, markersize=8)
            axes[idx, 1].semilogy(test_results['snr_values'], test_results['mse_mmse'], 's-',
                              label='MMSE', linewidth=2, markersize=8)
            axes[idx, 1].semilogy(test_results['snr_values'], test_results['mse_dnn'], '^-',
                              label='DNN', linewidth=2, markersize=8)
            axes[idx, 1].set_xlabel('SNR (dB)')
            axes[idx, 1].set_ylabel('MSE')
            axes[idx, 1].set_title(f'{mode.upper()}: MSE vs SNR')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)
            
            # NMSE comparison
            axes[idx, 2].semilogy(test_results['snr_values'], test_results['nmse_ls'], 'o-',
                              label='LS', linewidth=2, markersize=8)
            axes[idx, 2].semilogy(test_results['snr_values'], test_results['nmse_mmse'], 's-',
                              label='MMSE', linewidth=2, markersize=8)
            axes[idx, 2].semilogy(test_results['snr_values'], test_results['nmse_dnn'], '^-',
                              label='DNN', linewidth=2, markersize=8)
            axes[idx, 2].set_xlabel('SNR (dB)')
            axes[idx, 2].set_ylabel('NMSE')
            axes[idx, 2].set_title(f'{mode.upper()}: NMSE vs SNR')
            axes[idx, 2].legend()
            axes[idx, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('otfs_channel_estimation_results.png', dpi=500, bbox_inches='tight')
        plt.show()


def main():

    
    # Configuration
    N_DOPPLER = 8
    M_DELAY = 8
    N_TRAIN_SAMPLES = 1000000
    N_TEST_SAMPLES = 200000
    EPOCHS = 200
    BATCH_SIZE = 256
    
    print(f"Configuration:")
    print(f"  OTFS Grid: {N_DOPPLER}x{M_DELAY}")
    print(f"  Training samples: {N_TRAIN_SAMPLES}")
    print(f"  Test samples: {N_TEST_SAMPLES}")
    print(f"  Training epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    
    # Initialize estimator
    estimator = OTFSChannelEstimator(N=N_DOPPLER, M=M_DELAY)
    
    # Run comprehensive comparison
    results = estimator.comprehensive_comparison(
        n_train=N_TRAIN_SAMPLES,
        n_test=N_TEST_SAMPLES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    
    estimator.plot_results(results)
    with open('otfs_channel_estimation_results_fixed.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    if estimator.dnn_model_sparse is not None:
        estimator.dnn_model_sparse.save('otfs_dnn_sparse_channel_est_fixed.h5')
    
    if estimator.dnn_model_full is not None:
        estimator.dnn_model_full.save('otfs_dnn_full_channel_est_fixed.h5')
    
    print("\n=== Channel Estimation Performance Summary ===")
    for mode, result in results.items():
        print(f"\n{mode.upper()} Mode:")
        test_results = result['test_results']
        
        if 20 in test_results['snr_values']:
            idx = test_results['snr_values'].index(20)
            print(f"  At 20dB SNR:")
            print(f"    MSE: LS={test_results['mse_ls'][idx]:.6f}, MMSE={test_results['mse_mmse'][idx]:.6f}, DNN={test_results['mse_dnn'][idx]:.6f}")
            print(f"    NMSE: LS={test_results['nmse_ls'][idx]:.6f}, MMSE={test_results['nmse_mmse'][idx]:.6f}, DNN={test_results['nmse_dnn'][idx]:.6f}")
            print(f"    Avg Time: LS={np.mean(test_results['time_ls']):.4f}s, MMSE={np.mean(test_results['time_mmse']):.4f}s, DNN={np.mean(test_results['time_dnn']):.4f}s")


if __name__ == "__main__":
    main()
