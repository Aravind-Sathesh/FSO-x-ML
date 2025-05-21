import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import torch
from torch.utils.data import Dataset
import pickle
from matplotlib.colors import Normalize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f'oam_generation_{datetime.now().strftime("%Y%m%d")}.log')]
)
logger = logging.getLogger(__name__)


class OAMGenerator:
    """Generates Orbital Angular Momentum (OAM) modes."""

    def __init__(self, grid_size=256, r_max=1.0):
        self.grid_size = grid_size
        # Physical radius corresponding to grid edge (meters)
        self.r_max = r_max
        self.setup_coordinate_grid()
        logger.info(f"Initialized OAM Generator with grid size {grid_size}")
        print(f"Initialized OAM Generator")

    def setup_coordinate_grid(self):
        """Initialize coordinate grids for calculations."""
        x = np.linspace(-self.r_max, self.r_max, self.grid_size)
        y = np.linspace(-self.r_max, self.r_max, self.grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.Phi = np.arctan2(self.Y, self.X)
        logger.info("Coordinate grid setup complete")

    def generate_oam_mode(self, l, w0=0.45):
        """
        Generate standard OAM mode (Laguerre-Gaussian approximation) with topological charge l.
        """
        R_norm = self.R / self.r_max  # Normalize radius relative to grid extent
        # Using a common approximation for LG(0, l) mode amplitude
        radial_part = (np.sqrt(2) * R_norm / w0)**np.abs(l) * \
            np.exp(-R_norm**2 / w0**2)

        mask = np.ones_like(radial_part)
        mask[R_norm > 0.95] = 0
        edge_region = (R_norm > 0.9) & (R_norm <= 0.95)
        mask[edge_region] = (0.95 - R_norm[edge_region]) / 0.05  # Smooth edge
        amplitude = radial_part * mask

        max_amp = np.max(np.abs(amplitude))
        if max_amp > 1e-9:
            amplitude = amplitude / max_amp  # Normalize max amplitude to 1
        else:  # Handle l=0 case where amplitude might be near zero off-center
            amplitude = np.exp(-R_norm**2 / w0**2) * \
                mask  # Gaussian beam for l=0
            if np.max(np.abs(amplitude)) > 1e-9:
                amplitude = amplitude / np.max(np.abs(amplitude))
            else:
                amplitude.fill(0)

        phase = l * self.Phi
        return amplitude * np.exp(1j * phase)

    def generate_petalled_mode(self, l, w0=0.45, relative_amplitude=1.0):
        """
        Generate Petalled OAM pattern by superimposing modes with opposite charges.
        """
        l_mag = abs(int(l))
        if l_mag == 0:
            logger.warning(
                "Cannot generate petalled mode for l=0, returning l=0 standard mode.")
            return self.generate_oam_mode(0, w0)

        mode_plus = self.generate_oam_mode(l_mag, w0)
        mode_minus = self.generate_oam_mode(-l_mag, w0) * relative_amplitude

        mode = mode_plus + mode_minus
        return mode


class TurbulenceSimulator:
    """Simulates atmospheric turbulence effects using Kolmogorov spectrum with von Karman modifications."""

    # Default propagation distance added here for clarity
    def __init__(self, grid_size, r0, L0=200.0, l0=0.001, wavelength=1.0e-6, propagation_distance=1000, turbulence_strength=1.5, r_max=1.0):
        self.grid_size = grid_size
        # Fried parameter (meters), defines turbulence level for given wavelength/path
        self.r0 = r0
        self.L0 = L0  # Outer scale (meters)
        self.l0 = l0  # Inner scale (meters)
        self.wavelength = wavelength  # Wavelength (meters)
        # Propagation distance (meters)
        self.propagation_distance = propagation_distance
        # User-defined multiplier for phase screen std dev
        self.turbulence_strength_factor = turbulence_strength
        # Physical radius corresponding to grid edge (meters)
        self.r_max = r_max
        self.dx = 2.0 * self.r_max / self.grid_size  # Pixel size in meters
        self.k = 2 * np.pi / self.wavelength

        self.setup_frequency_grid()

        # Calculate parameters based on r0 (assuming r0 is defined for wavelength and distance)
        self.integrated_Cn2 = self.r0**(-5/3) / (0.423 * self.k**2)
        self.effective_Cn2 = self.integrated_Cn2 / \
            self.propagation_distance  # Avg Cn2 over path
        self.effective_r0 = (
            0.423 * self.k**2 * self.effective_Cn2 * self.propagation_distance)**(-3/5)

        logger.info(
            f"Initialized Turbulence Simulator: grid={grid_size}x{grid_size}, r0={self.r0:.4f}m, "
            f"L0={self.L0:.1f}m, l0={self.l0:.4f}m, lambda={self.wavelength:.2e}m, dist={self.propagation_distance:.1f}m, "
            f"strength_factor={self.turbulence_strength_factor:.2f}, r_max={self.r_max:.2f}m, dx={self.dx:.4e}m"
        )
        logger.info(f"Effective Cn^2: {self.effective_Cn2:.3e} m^(-2/3)")
        logger.info(
            f"Effective r0 at z={self.propagation_distance}m: {self.effective_r0:.4f} m (Should approx match input r0)")
        # Print key params
        print(
            f"Initialized Turbulence Simulator (r0={self.r0:.2f}m, strength={self.turbulence_strength_factor:.1f})")

    def setup_frequency_grid(self):
        """Setup spatial frequency grid."""
        fx = np.fft.fftfreq(self.grid_size, self.dx)  # cycles per meter
        fy = np.fft.fftfreq(self.grid_size, self.dx)
        self.Fx, self.Fy = np.meshgrid(fx, fy)
        self.F = np.sqrt(self.Fx**2 + self.Fy**2)

        self.Kx = 2 * np.pi * self.Fx  # radians per meter
        self.Ky = 2 * np.pi * self.Fy
        self.K = 2 * np.pi * self.F
        self.K[self.K == 0] = 1e-12  # Avoid division by zero
        # logger.info("Frequency grid setup complete") # Reduced logging

    def von_karman_spectrum(self, K):
        """Calculate the Von Karman spectrum for refractive index fluctuations."""
        k0 = 2 * np.pi / self.L0  # Outer scale wavenumber
        km = 5.92 / self.l0      # Inner scale wavenumber
        # Power spectral density Phi_n(K) (units: m^3)
        phi_n = 0.033 * self.effective_Cn2 * \
            np.exp(-K**2 / km**2) / (K**2 + k0**2)**(11/6)
        return phi_n

    def generate_phase_screen(self):
        """Generate random phase screen using von Karman spectrum."""
        phi_n = self.von_karman_spectrum(self.K)
        # Power spectral density of the phase fluctuations (units: rad^2 m^2)
        phi_phase_spectrum = 2 * np.pi * self.k**2 * self.propagation_distance * phi_n

        delta_fx = 1.0 / (self.grid_size * self.dx)
        delta_kx = 2 * np.pi * delta_fx
        delta_ky = delta_kx

        random_components = np.random.normal(size=(self.grid_size, self.grid_size)) + \
            1j * np.random.normal(size=(self.grid_size, self.grid_size))

        # Scale random numbers by sqrt(spectrum * area_element / 2)
        phase_screen_fft = random_components * \
            np.sqrt(phi_phase_spectrum * delta_kx * delta_ky / 2.0)

        # Inverse FFT to get the spatial phase screen
        phase_screen = np.real(np.fft.ifft2(
            np.fft.ifftshift(phase_screen_fft)))

        # Optional: Scale to match theoretical standard deviation before applying factor
        phase_variance_theory = np.sum(
            phi_phase_spectrum) * delta_kx * delta_ky
        phase_std_theory = np.sqrt(phase_variance_theory)
        phase_std_generated = np.std(phase_screen)
        if phase_std_generated > 1e-9:
            phase_screen = phase_screen * \
                (phase_std_theory / phase_std_generated)

        # Apply the additional user-defined strength multiplier
        final_phase_screen = phase_screen * self.turbulence_strength_factor
        return final_phase_screen

    def angular_spectrum_propagation(self, field, z, wavelength=None):
        """
        Propagate a complex field using the Angular Spectrum method.
        """
        if wavelength is None:
            wavelength = self.wavelength
        k_prop = 2 * np.pi / wavelength

        # Calculate the propagation kernel H(kx, ky)
        kz_arg = k_prop**2 - self.Kx**2 - self.Ky**2
        kz = np.zeros_like(kz_arg, dtype=complex)
        valid_prop = kz_arg >= 0
        invalid_prop = kz_arg < 0
        kz[valid_prop] = np.sqrt(kz_arg[valid_prop])
        # Evanescent waves decay
        kz[invalid_prop] = 1j * np.sqrt(-kz_arg[invalid_prop])
        H = np.exp(1j * kz * z)

        # Perform the propagation using FFT
        field_fft = np.fft.fftshift(np.fft.fft2(field))
        propagated_fft = field_fft * H
        propagated_field = np.fft.ifft2(np.fft.ifftshift(propagated_fft))

        return propagated_field


class OAMDataset(Dataset):
    """Dataset class for OAM modes with distortions and optional phase transformation."""

    # Added r0 parameter to constructor
    def __init__(self, num_samples, grid_size=256, max_l=10, save_samples=False,
                 dataset_type="train", turbulence_strength=1.5, use_petals=False, petal_l_range=(5, 12),
                 apply_filter=True, r0=0.05):
        self.generator = OAMGenerator(grid_size)
        # Pass r0 to TurbulenceSimulator
        self.turbulence = TurbulenceSimulator(
            grid_size, r0=r0, turbulence_strength=turbulence_strength, r_max=self.generator.r_max)
        self.num_samples = num_samples
        self.max_l = max_l
        self.data = []
        self.labels = []
        self.save_samples = save_samples
        self.dataset_type = dataset_type
        self.use_petals = use_petals
        self.petal_l_range = petal_l_range
        # Filter only applies to standard modes
        self.apply_filter = apply_filter if not use_petals else False

        mode_desc = "petalled" if self.use_petals else "standard OAM"
        filter_desc = "with filter+FFT" if self.apply_filter else "raw intensity"
        logger.info(
            f"Creating {dataset_type} dataset: {num_samples} samples, type='{mode_desc}', "
            f"processing='{filter_desc}', r0={r0}, turbulence_strength={turbulence_strength}"
        )
        print(
            f"Creating {dataset_type} dataset ({mode_desc}, {filter_desc}, r0={r0}, strength={turbulence_strength})")
        self.generate_dataset()

    def _normalize_image(self, image_data):
        """Normalizes image data to [0, 1] range."""
        min_val, max_val = np.min(image_data), np.max(image_data)
        if max_val > min_val + 1e-9:
            return (image_data - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(image_data)

    def generate_dataset(self):
        """Generate dataset of OAM modes with propagation and turbulence effects."""
        os.makedirs('samples', exist_ok=True)

        z1 = 1.0  # Short distance to first screen
        z2 = self.turbulence.propagation_distance  # Propagation after screen

        for i in range(self.num_samples):
            if self.use_petals:
                l = np.random.randint(
                    self.petal_l_range[0], self.petal_l_range[1] + 1)
                clean_mode = self.generator.generate_petalled_mode(l, w0=0.45)
                label = 2 * l
                mode_type_info = f'Petalled ({label} petals)'
            else:
                l = np.random.randint(-self.max_l, self.max_l + 1)
                clean_mode = self.generator.generate_oam_mode(l, w0=0.45)
                label = l
                mode_type_info = f'OAM l={l}'

            if z1 > 0:
                field_at_screen = self.turbulence.angular_spectrum_propagation(
                    clean_mode, z1)
            else:
                field_at_screen = clean_mode

            phase_screen = self.turbulence.generate_phase_screen()
            field_after_screen = field_at_screen * np.exp(1j * phase_screen)

            final_field = self.turbulence.angular_spectrum_propagation(
                field_after_screen, z2)

            input_image = None
            processing_info = ""
            if self.use_petals:
                distorted_intensity = np.abs(final_field)**2
                input_image = self._normalize_image(
                    np.sqrt(distorted_intensity))
                processing_info = "Raw Intensity"
            else:  # Standard OAM modes
                if self.apply_filter:
                    phi_coords = self.generator.Phi
                    conjugate_phase_mask = np.exp(-1j * l * phi_coords)
                    phase_corrected_field = final_field * conjugate_phase_mask
                    focal_plane_field = np.fft.fftshift(
                        np.fft.fft2(phase_corrected_field))
                    focal_plane_intensity = np.abs(focal_plane_field)**2
                    input_image = self._normalize_image(focal_plane_intensity)
                    processing_info = "Filtered + FFT Intensity"
                else:
                    distorted_intensity = np.abs(final_field)**2
                    input_image = self._normalize_image(
                        np.sqrt(distorted_intensity))
                    processing_info = "Raw Intensity"

            if input_image is None:
                logger.error(
                    f"Failed to generate input image for sample {i}, l={l}")
                continue

            self.data.append(input_image)
            self.labels.append(label)

            # --- Visualization ---
            if self.save_samples and i < 10:
                clean_amplitude_vis = np.abs(clean_mode)
                clean_phase_vis = np.angle(clean_mode)

                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(
                    f'{mode_type_info} - {processing_info} - Sample {i}', fontsize=14)

                norm_amp = Normalize(vmin=0, vmax=np.max(clean_amplitude_vis))
                norm_phase = Normalize(vmin=-np.pi, vmax=np.pi)
                norm_turb = Normalize(
                    vmin=-np.pi*2, vmax=np.pi*2)  # Adjust if needed

                im0 = axes[0, 0].imshow(
                    clean_amplitude_vis, cmap='inferno', norm=norm_amp)
                axes[0, 0].set_title(f'Clean Amp.', fontsize=12)
                axes[0, 0].axis('off')
                fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

                im1 = axes[0, 1].imshow(
                    clean_phase_vis, cmap='hsv', norm=norm_phase)
                axes[0, 1].set_title(f'Clean Phase', fontsize=12)
                axes[0, 1].axis('off')
                fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

                im_ps = axes[0, 2].imshow(
                    phase_screen, cmap='RdBu', norm=norm_turb)
                axes[0, 2].set_title('Turbulence Phase Screen', fontsize=12)
                axes[0, 2].axis('off')
                fig.colorbar(im_ps, ax=axes[0, 2], shrink=0.8)

                if self.use_petals or (not self.use_petals and not self.apply_filter):
                    distorted_amplitude_vis = np.abs(final_field)
                    distorted_phase_vis = np.angle(final_field)
                    norm_dist_amp = Normalize(vmin=np.min(
                        distorted_amplitude_vis), vmax=np.max(distorted_amplitude_vis))

                    im2 = axes[1, 0].imshow(
                        distorted_amplitude_vis, cmap='inferno', norm=norm_dist_amp)
                    axes[1, 0].set_title('Turbulent Amp.', fontsize=12)
                    axes[1, 0].axis('off')
                    fig.colorbar(im2, ax=axes[1, 0], shrink=0.8)

                    im3 = axes[1, 1].imshow(
                        distorted_phase_vis, cmap='hsv', norm=norm_phase)
                    axes[1, 1].set_title('Turbulent Phase', fontsize=12)
                    axes[1, 1].axis('off')
                    fig.colorbar(im3, ax=axes[1, 1], shrink=0.8)

                    im4 = axes[1, 2].imshow(
                        input_image, cmap='viridis', vmin=0, vmax=1)
                    axes[1, 2].set_title(
                        'Input to CNN (Norm. Amp.)', fontsize=12)
                    axes[1, 2].axis('off')
                    fig.colorbar(im4, ax=axes[1, 2], shrink=0.8)
                else:  # Standard OAM with filter applied
                    distorted_amplitude_vis = np.abs(final_field)
                    norm_dist_amp = Normalize(vmin=np.min(
                        distorted_amplitude_vis), vmax=np.max(distorted_amplitude_vis))
                    im2 = axes[1, 0].imshow(
                        distorted_amplitude_vis, cmap='inferno', norm=norm_dist_amp)
                    axes[1, 0].set_title(
                        'Turbulent Amp (Pre-Filter)', fontsize=12)
                    axes[1, 0].axis('off')
                    fig.colorbar(im2, ax=axes[1, 0], shrink=0.8)

                    im3 = axes[1, 1].imshow(
                        input_image, cmap='viridis', vmin=0, vmax=1)
                    axes[1, 1].set_title(
                        'Input to CNN (Detector Plane)', fontsize=12)
                    axes[1, 1].axis('off')
                    fig.colorbar(im3, ax=axes[1, 1], shrink=0.8)

                    axes[1, 2].axis('off')
                    axes[1, 2].text(0.5, 0.5, f'Mode l={l}\nProcessed w/ Filter+FFT',
                                    ha='center', va='center', fontsize=10)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                filename_mode = "petalled" if self.use_petals else "oam_radial"
                filename_filter = "_filtered" if self.apply_filter else "_raw"
                filename_l = f"l_{label}" if not self.use_petals else f"p_{label}"
                plt.savefig(
                    f'samples/{self.dataset_type}_{filename_mode}{filename_filter}_sample_{i}_{filename_l}.png',
                    dpi=200, bbox_inches='tight')
                plt.close(fig)

        mode_type_log = "petalled modes" if self.use_petals else "standard OAM modes"
        if self.use_petals:
            label_info_log = f"petal counts from {self.petal_l_range[0]*2} to {self.petal_l_range[1]*2}"
        elif self.apply_filter:
            label_info_log = f"l values from {-self.max_l} to {self.max_l} (using Filter+FFT)"
        else:
            label_info_log = f"l values from {-self.max_l} to {self.max_l} (using Raw Intensity)"

        logger.info(
            f"Generated {len(self.data)} samples with {mode_type_log} - {label_info_log}")
        print(f"Dataset generation complete")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx][None, :, :]), self.labels[idx]

    def save_to_file(self, filename):
        """Save the dataset to a file using pickle."""
        data_dict = {
            'data': self.data,
            'labels': self.labels,
            'grid_size': self.generator.grid_size,
            'r_max': self.generator.r_max,
            'max_l': self.max_l,
            'use_petals': self.use_petals,
            'petal_l_range': self.petal_l_range,
            'apply_filter': self.apply_filter,
            'dataset_type': self.dataset_type,
            'turbulence_params': {
                'r0': self.turbulence.r0,
                'L0': self.turbulence.L0,
                'l0': self.turbulence.l0,
                'wavelength': self.turbulence.wavelength,
                'propagation_distance': self.turbulence.propagation_distance,
                'turbulence_strength_factor': self.turbulence.turbulence_strength_factor
            }
        }
        with open(filename, 'wb') as f:
            pickle.dump(data_dict, f)
        logger.info(f"Dataset saved to {filename}")
        print(f"Dataset saved to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        """Load dataset and metadata from file."""
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
        logger.info(f"Dataset loaded from {filename}")
        return data_dict


def main():
    """Generate and save datasets for training, validation and testing."""
    np.random.seed(42)
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('samples', exist_ok=True)

    # --- Configuration ---
    GRID_SIZE = 256
    TURBULENCE_STRENGTH = 1.0  # Kept reduced strength factor
    # <<< Increased r0 value for weaker turbulence >>>
    R0_VALUE = 0.15  # Was 0.05, increased to 0.15 (15cm)
    MAX_L_RADIAL = 5  # Max |l| for standard OAM modes (-5 to +5)

    # --- Generate Standard OAM Datasets (with Filter) ---
    print(
        f"\n--- Generating Standard OAM Datasets (Filter+FFT, r0={R0_VALUE}, strength={TURBULENCE_STRENGTH}) ---")
    train_dataset_radial = OAMDataset(
        num_samples=500, save_samples=True, dataset_type="train",
        grid_size=GRID_SIZE, r0=R0_VALUE, turbulence_strength=TURBULENCE_STRENGTH,
        use_petals=False, max_l=MAX_L_RADIAL, apply_filter=True)
    # Update filename to reflect parameters if desired, e.g., include r0 value
    train_dataset_radial.save_to_file(
        f'datasets/train_dataset.pkl')

    val_dataset_radial = OAMDataset(
        num_samples=100, save_samples=True, dataset_type="val",
        grid_size=GRID_SIZE, r0=R0_VALUE, turbulence_strength=TURBULENCE_STRENGTH,
        use_petals=False, max_l=MAX_L_RADIAL, apply_filter=True)
    val_dataset_radial.save_to_file(
        f'datasets/val_dataset.pkl')

    test_dataset_radial = OAMDataset(
        num_samples=100, save_samples=True, dataset_type="test",
        grid_size=GRID_SIZE, r0=R0_VALUE, turbulence_strength=TURBULENCE_STRENGTH,
        use_petals=False, max_l=MAX_L_RADIAL, apply_filter=True)
    test_dataset_radial.save_to_file(
        f'datasets/test_dataset.pkl')

    # --- Optional: Generate Petalled OAM Datasets (Raw Intensity) ---
    # print("\n--- Generating Petalled OAM Datasets (Raw Intensity) ---")
    # PETAL_L_RANGE = (5, 12)
    # train_dataset_petalled = OAMDataset(
    #     num_samples=500, save_samples=True, dataset_type="train",
    #     grid_size=GRID_SIZE, r0=R0_VALUE, turbulence_strength=TURBULENCE_STRENGTH,
    #     use_petals=True, petal_l_range=PETAL_L_RANGE, apply_filter=False)
    # train_dataset_petalled.save_to_file('datasets/train_dataset_petalled_raw.pkl')
    # ... (add val/test for petalled if needed) ...

    logger.info("All requested datasets generated and saved successfully")
    print("All requested datasets generated and saved successfully")


if __name__ == "__main__":
    main()
