import asyncio
import struct
import io
import logging
import json
import os
from bleak import BleakScanner, BleakClient
from collections import deque
import statistics
import numpy as np
from normalization_params import normalize_feature
from azure.iot.device.aio import IoTHubDeviceClient
from azure.iot.device.iothub import Message

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID    = "12345678-1234-5678-1234-56789abcdef1"

# Azure IoT Hub connection string (load from environment variable for security)
AZURE_IOT_CONNECTION_STRING = ""

# Calibration settings
CALIBRATION_SAMPLES = 50  # Number of samples to average for calibration
NOISE_THRESHOLD = 0.1     # Values below this are considered noise (m/s^2 or dps)

# Windowing settings
WINDOW_SIZE = 128         # 2.56 seconds at 50 Hz
WINDOW_STEP = 64          # 50% overlap (step by half the window size)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SensorProcessor:
    def __init__(self):
        # Calibration offsets
        self.ax_offset = 0.0
        self.ay_offset = 0.0
        self.az_offset = 0.0
        self.gx_offset = 0.0
        self.gy_offset = 0.0
        self.gz_offset = 0.0
        
        # Calibration data collection
        self.calibration_data = []
        self.is_calibrated = False
        
        # Sliding window buffer
        self.window_buffer = deque(maxlen=WINDOW_SIZE)
        self.samples_since_last_window = 0
        
    def add_calibration_sample(self, ax, ay, az, gx, gy, gz):
        """Collect samples for calibration"""
        self.calibration_data.append((ax, ay, az, gx, gy, gz))
        
        if len(self.calibration_data) >= CALIBRATION_SAMPLES:
            self._compute_offsets()
            return True
        return False
    
    def _compute_offsets(self):
        """Compute average offsets from calibration samples"""
        ax_samples = [s[0] for s in self.calibration_data]
        ay_samples = [s[1] for s in self.calibration_data]
        az_samples = [s[2] for s in self.calibration_data]
        gx_samples = [s[3] for s in self.calibration_data]
        gy_samples = [s[4] for s in self.calibration_data]
        gz_samples = [s[5] for s in self.calibration_data]
        
        self.ax_offset = statistics.mean(ax_samples)
        self.ay_offset = statistics.mean(ay_samples)
        self.az_offset = statistics.mean(az_samples)
        self.gx_offset = statistics.mean(gx_samples)
        self.gy_offset = statistics.mean(gy_samples)
        self.gz_offset = statistics.mean(gz_samples)
        
        self.is_calibrated = True
        logger.info(f"Calibration complete: ax={self.ax_offset:.3f}, ay={self.ay_offset:.3f}, "
                   f"az={self.az_offset:.3f}, gx={self.gx_offset:.3f}, gy={self.gy_offset:.3f}, "
                   f"gz={self.gz_offset:.3f}")
    
    def apply_noise_filter(self, value):
        """Apply dead-zone filter to remove small noise"""
        return 0.0 if abs(value) < NOISE_THRESHOLD else value
    
    def process(self, ax, ay, az, gx, gy, gz):
        """Apply calibration and noise filtering"""
        # Apply offsets
        ax_cal = ax - self.ax_offset
        ay_cal = ay - self.ay_offset
        az_cal = az - self.az_offset
        gx_cal = gx - self.gx_offset
        gy_cal = gy - self.gy_offset
        gz_cal = gz - self.gz_offset
        
        # Apply noise filter
        ax_filtered = self.apply_noise_filter(ax_cal)
        ay_filtered = self.apply_noise_filter(ay_cal)
        az_filtered = self.apply_noise_filter(az_cal)
        gx_filtered = self.apply_noise_filter(gx_cal)
        gy_filtered = self.apply_noise_filter(gy_cal)
        gz_filtered = self.apply_noise_filter(gz_cal)
        
        return ax_filtered, ay_filtered, az_filtered, gx_filtered, gy_filtered, gz_filtered
    
    def add_to_window(self, ax, ay, az, gx, gy, gz):
        """Add sample to sliding window and return window when ready"""
        self.window_buffer.append((ax, ay, az, gx, gy, gz))
        self.samples_since_last_window += 1
        
        # Check if we have a full window and have stepped far enough
        if len(self.window_buffer) == WINDOW_SIZE and self.samples_since_last_window >= WINDOW_STEP:
            self.samples_since_last_window = 0
            # Return a copy of the current window
            return list(self.window_buffer)
        
        return None

# Global sensor processor
sensor_processor = SensorProcessor()

# Global Azure IoT Hub client
iot_hub_client = None

async def send_to_azure_iot_hub(data):
    """Send data to Azure IoT Hub"""
    global iot_hub_client
    
    if iot_hub_client is None:
        logger.warning("Azure IoT Hub client not initialized, skipping data send")
        return
    
    try:
        # Convert data dictionary to JSON string
        json_string = json.dumps(data)
        
        # Create message with JSON string as body
        message = Message(json_string)
        
        # Set content type and encoding so Power BI knows it's JSON
        message.content_type = "application/json"
        message.content_encoding = "utf-8"
        
        # Send message to IoT Hub
        await iot_hub_client.send_message(message)
        logger.info("Successfully sent data to Azure IoT Hub")
        
    except Exception as e:
        logger.error(f"Failed to send data to Azure IoT Hub: {e}")

def handle_notification(_, data: bytearray):
    try:
        ax, ay, az, gx, gy, gz = struct.unpack("<ffffff", data)
     
        # Calibration phase
        if not sensor_processor.is_calibrated:
            calibrated = sensor_processor.add_calibration_sample(ax, ay, az, gx, gy, gz)
            if calibrated:
                logger.info("window_id,sample_id,acceleration_x,acceleration_y,acceleration_z,gyro_x,gyro_y,gyro_z")
            return
        
        # Process data with calibration and filtering
        ax, ay, az, gx, gy, gz = sensor_processor.process(ax, ay, az, gx, gy, gz)

        # Convert acceleration from m/s² to g's
        ax = ax / 9.80665
        ay = ay / 9.80665
        az = az / 9.80665
        
        # Add to sliding window
        window = sensor_processor.add_to_window(ax, ay, az, gx, gy, gz)
        
        # If we have a complete window, log it and print window data
        if window is not None:
            window_id = (len(sensor_processor.window_buffer) - WINDOW_SIZE) // WINDOW_STEP + 1
            
            # Print window summary to terminal
            print(f"\n{'='*80}")
            print(f"Window {window_id} - {WINDOW_SIZE} samples (2.56 seconds)")
            print(f"{'='*80}")
            
            # Extract all data for printing statistics
            ax_values = [s[0] for s in window]
            ay_values = [s[1] for s in window]
            az_values = [s[2] for s in window]
            gx_values = [s[3] for s in window]
            gy_values = [s[4] for s in window]
            gz_values = [s[5] for s in window]
            
            # Calculate acceleration magnitude (Euclidean norm)
            accel_mag_values = [(ax**2 + ay**2 + az**2)**0.5 for ax, ay, az in zip(ax_values, ay_values, az_values)]
            
            # Calculate gyroscope magnitude (Euclidean norm)
            gyro_mag_values = [(gx**2 + gy**2 + gz**2)**0.5 for gx, gy, gz in zip(gx_values, gy_values, gz_values)]
            
            # Calculate jerk (derivative of acceleration and angular velocity)
            # Sample rate is ~50 Hz, so delta_t ≈ 0.02 seconds
            sample_rate = 50.0  # Hz
            delta_t = 1.0 / sample_rate
            
            # Accelerometer jerk: d(acceleration)/dt
            accel_jx_values = [(ax_values[i+1] - ax_values[i]) / delta_t for i in range(len(ax_values) - 1)]
            accel_jy_values = [(ay_values[i+1] - ay_values[i]) / delta_t for i in range(len(ay_values) - 1)]
            accel_jz_values = [(az_values[i+1] - az_values[i]) / delta_t for i in range(len(az_values) - 1)]
            
            # Calculate acceleration jerk magnitude (Euclidean norm)
            accel_jerk_mag_values = [(jx**2 + jy**2 + jz**2)**0.5 for jx, jy, jz in zip(accel_jx_values, accel_jy_values, accel_jz_values)]
            
            # Gyroscope jerk: d(angular_velocity)/dt (angular acceleration)
            gyro_jx_values = [(gx_values[i+1] - gx_values[i]) / delta_t for i in range(len(gx_values) - 1)]
            gyro_jy_values = [(gy_values[i+1] - gy_values[i]) / delta_t for i in range(len(gy_values) - 1)]
            gyro_jz_values = [(gz_values[i+1] - gz_values[i]) / delta_t for i in range(len(gz_values) - 1)]
            
            # Calculate gyroscope jerk magnitude (Euclidean norm)
            gyro_jerk_mag_values = [(jx**2 + jy**2 + jz**2)**0.5 for jx, jy, jz in zip(gyro_jx_values, gyro_jy_values, gyro_jz_values)]
            
            # Calculate FFT (Fast Fourier Transform) for acceleration data
            # FFT gives us the frequency domain representation
            ax_fft = np.fft.fft(ax_values)
            ay_fft = np.fft.fft(ay_values)
            az_fft = np.fft.fft(az_values)
            
            # Get the frequency bins (only positive frequencies)
            n = len(ax_values)
            freq_bins = np.fft.fftfreq(n, d=delta_t)[:n//2]
            
            # Get magnitude spectrum (only positive frequencies)
            ax_magnitude = np.abs(ax_fft)[:n//2]
            ay_magnitude = np.abs(ay_fft)[:n//2]
            az_magnitude = np.abs(az_fft)[:n//2]
            
            # Find dominant frequencies (where magnitude is significant)
            # Skip DC component (index 0) and find indices where magnitude > threshold
            threshold = np.max(ax_magnitude) * 0.1  # 10% of max magnitude
            ax_dominant_freqs = freq_bins[ax_magnitude > threshold]
            ay_dominant_freqs = freq_bins[ay_magnitude > threshold]
            az_dominant_freqs = freq_bins[az_magnitude > threshold]
            
            # Calculate FFT (Fast Fourier Transform) for accelerometer jerk data
            # FFT gives us the frequency domain representation
            accel_jx_fft = np.fft.fft(accel_jx_values)
            accel_jy_fft = np.fft.fft(accel_jy_values)
            accel_jz_fft = np.fft.fft(accel_jz_values)
            
            # Get the frequency bins for jerk (only positive frequencies)
            n_jerk = len(accel_jx_values)
            freq_bins_jerk = np.fft.fftfreq(n_jerk, d=delta_t)[:n_jerk//2]
            
            # Get magnitude spectrum for jerk (only positive frequencies)
            accel_jx_magnitude = np.abs(accel_jx_fft)[:n_jerk//2]
            accel_jy_magnitude = np.abs(accel_jy_fft)[:n_jerk//2]
            accel_jz_magnitude = np.abs(accel_jz_fft)[:n_jerk//2]
            
            # Find dominant frequencies for jerk (where magnitude is significant)
            threshold_jerk = np.max(accel_jx_magnitude) * 0.1  # 10% of max magnitude
            accel_jx_dominant_freqs = freq_bins_jerk[accel_jx_magnitude > threshold_jerk]
            accel_jy_dominant_freqs = freq_bins_jerk[accel_jy_magnitude > threshold_jerk]
            accel_jz_dominant_freqs = freq_bins_jerk[accel_jz_magnitude > threshold_jerk]
            
            # Calculate FFT (Fast Fourier Transform) for gyroscope data
            # FFT gives us the frequency domain representation
            gx_fft = np.fft.fft(gx_values)
            gy_fft = np.fft.fft(gy_values)
            gz_fft = np.fft.fft(gz_values)
            
            # Get the frequency bins for gyroscope (only positive frequencies)
            n_gyro = len(gx_values)
            freq_bins_gyro = np.fft.fftfreq(n_gyro, d=delta_t)[:n_gyro//2]
            
            # Get magnitude spectrum for gyroscope (only positive frequencies)
            gx_magnitude = np.abs(gx_fft)[:n_gyro//2]
            gy_magnitude = np.abs(gy_fft)[:n_gyro//2]
            gz_magnitude = np.abs(gz_fft)[:n_gyro//2]
            
            # Find dominant frequencies for gyroscope (where magnitude is significant)
            threshold_gyro = np.max(gx_magnitude) * 0.1  # 10% of max magnitude
            gx_dominant_freqs = freq_bins_gyro[gx_magnitude > threshold_gyro]
            gy_dominant_freqs = freq_bins_gyro[gy_magnitude > threshold_gyro]
            gz_dominant_freqs = freq_bins_gyro[gz_magnitude > threshold_gyro]
            
            # Calculate FFT (Fast Fourier Transform) for acceleration magnitude
            # FFT gives us the frequency domain representation
            accel_mag_fft = np.fft.fft(accel_mag_values)
            
            # Get the frequency bins for acceleration magnitude (only positive frequencies)
            n_accel_mag = len(accel_mag_values)
            freq_bins_accel_mag = np.fft.fftfreq(n_accel_mag, d=delta_t)[:n_accel_mag//2]
            
            # Get magnitude spectrum for acceleration magnitude (only positive frequencies)
            accel_mag_magnitude = np.abs(accel_mag_fft)[:n_accel_mag//2]
            
            # Find dominant frequencies for acceleration magnitude (where magnitude is significant)
            threshold_accel_mag = np.max(accel_mag_magnitude) * 0.1  # 10% of max magnitude
            accel_mag_dominant_freqs = freq_bins_accel_mag[accel_mag_magnitude > threshold_accel_mag]
            
            # Calculate FFT (Fast Fourier Transform) for acceleration jerk magnitude
            # FFT gives us the frequency domain representation
            accel_jerk_mag_fft = np.fft.fft(accel_jerk_mag_values)
            
            # Get the frequency bins for acceleration jerk magnitude (only positive frequencies)
            n_accel_jerk_mag = len(accel_jerk_mag_values)
            freq_bins_accel_jerk_mag = np.fft.fftfreq(n_accel_jerk_mag, d=delta_t)[:n_accel_jerk_mag//2]
            
            # Get magnitude spectrum for acceleration jerk magnitude (only positive frequencies)
            accel_jerk_mag_magnitude = np.abs(accel_jerk_mag_fft)[:n_accel_jerk_mag//2]
            
            # Find dominant frequencies for acceleration jerk magnitude (where magnitude is significant)
            threshold_accel_jerk_mag = np.max(accel_jerk_mag_magnitude) * 0.1  # 10% of max magnitude
            accel_jerk_mag_dominant_freqs = freq_bins_accel_jerk_mag[accel_jerk_mag_magnitude > threshold_accel_jerk_mag]
            
            # Calculate FFT (Fast Fourier Transform) for gyroscope magnitude
            # FFT gives us the frequency domain representation
            gyro_mag_fft = np.fft.fft(gyro_mag_values)
            
            # Get the frequency bins for gyroscope magnitude (only positive frequencies)
            n_gyro_mag = len(gyro_mag_values)
            freq_bins_gyro_mag = np.fft.fftfreq(n_gyro_mag, d=delta_t)[:n_gyro_mag//2]
            
            # Get magnitude spectrum for gyroscope magnitude (only positive frequencies)
            gyro_mag_magnitude = np.abs(gyro_mag_fft)[:n_gyro_mag//2]
            
            # Find dominant frequencies for gyroscope magnitude (where magnitude is significant)
            threshold_gyro_mag = np.max(gyro_mag_magnitude) * 0.1  # 10% of max magnitude
            gyro_mag_dominant_freqs = freq_bins_gyro_mag[gyro_mag_magnitude > threshold_gyro_mag]
            
            # Calculate FFT (Fast Fourier Transform) for gyroscope jerk magnitude
            # FFT gives us the frequency domain representation
            gyro_jerk_mag_fft = np.fft.fft(gyro_jerk_mag_values)
            
            # Get the frequency bins for gyroscope jerk magnitude (only positive frequencies)
            n_gyro_jerk_mag = len(gyro_jerk_mag_values)
            freq_bins_gyro_jerk_mag = np.fft.fftfreq(n_gyro_jerk_mag, d=delta_t)[:n_gyro_jerk_mag//2]
            
            # Get magnitude spectrum for gyroscope jerk magnitude (only positive frequencies)
            gyro_jerk_mag_magnitude = np.abs(gyro_jerk_mag_fft)[:n_gyro_jerk_mag//2]
            
            # Find dominant frequencies for gyroscope jerk magnitude (where magnitude is significant)
            threshold_gyro_jerk_mag = np.max(gyro_jerk_mag_magnitude) * 0.1  # 10% of max magnitude
            gyro_jerk_mag_dominant_freqs = freq_bins_gyro_jerk_mag[gyro_jerk_mag_magnitude > threshold_gyro_jerk_mag]
            
            # Store all features in variables with standardized naming
            # Time domain - Accelerometer
            tBodyAcc_mean_X = statistics.mean(ax_values)
            tBodyAcc_mean_Y = statistics.mean(ay_values)
            tBodyAcc_mean_Z = statistics.mean(az_values)
            tBodyAcc_min_X = min(ax_values)
            tBodyAcc_min_Y = min(ay_values)
            tBodyAcc_min_Z = min(az_values)
            tBodyAcc_max_X = max(ax_values)
            tBodyAcc_max_Y = max(ay_values)
            tBodyAcc_max_Z = max(az_values)
            
            # Time domain - Accelerometer Magnitude
            tBodyAccMag_mean = statistics.mean(accel_mag_values)
            tBodyAccMag_min = min(accel_mag_values)
            tBodyAccMag_max = max(accel_mag_values)
            
            # Frequency domain - Accelerometer Magnitude
            fBodyAccMag_mean = np.mean(accel_mag_dominant_freqs) if len(accel_mag_dominant_freqs) > 0 else 0.0
            fBodyAccMag_min = np.min(accel_mag_dominant_freqs) if len(accel_mag_dominant_freqs) > 0 else 0.0
            fBodyAccMag_max = np.max(accel_mag_dominant_freqs) if len(accel_mag_dominant_freqs) > 0 else 0.0
            
            # Frequency domain - Accelerometer
            fBodyAcc_mean_X = np.mean(ax_dominant_freqs) if len(ax_dominant_freqs) > 0 else 0.0
            fBodyAcc_mean_Y = np.mean(ay_dominant_freqs) if len(ay_dominant_freqs) > 0 else 0.0
            fBodyAcc_mean_Z = np.mean(az_dominant_freqs) if len(az_dominant_freqs) > 0 else 0.0
            fBodyAcc_min_X = np.min(ax_dominant_freqs) if len(ax_dominant_freqs) > 0 else 0.0
            fBodyAcc_min_Y = np.min(ay_dominant_freqs) if len(ay_dominant_freqs) > 0 else 0.0
            fBodyAcc_min_Z = np.min(az_dominant_freqs) if len(az_dominant_freqs) > 0 else 0.0
            fBodyAcc_max_X = np.max(ax_dominant_freqs) if len(ax_dominant_freqs) > 0 else 0.0
            fBodyAcc_max_Y = np.max(ay_dominant_freqs) if len(ay_dominant_freqs) > 0 else 0.0
            fBodyAcc_max_Z = np.max(az_dominant_freqs) if len(az_dominant_freqs) > 0 else 0.0
            
            # Time domain - Accelerometer Jerk
            tBodyAccJerk_mean_X = statistics.mean(accel_jx_values)
            tBodyAccJerk_mean_Y = statistics.mean(accel_jy_values)
            tBodyAccJerk_mean_Z = statistics.mean(accel_jz_values)
            tBodyAccJerk_min_X = min(accel_jx_values)
            tBodyAccJerk_min_Y = min(accel_jy_values)
            tBodyAccJerk_min_Z = min(accel_jz_values)
            tBodyAccJerk_max_X = max(accel_jx_values)
            tBodyAccJerk_max_Y = max(accel_jy_values)
            tBodyAccJerk_max_Z = max(accel_jz_values)
            
            # Time domain - Accelerometer Jerk Magnitude
            tBodyAccJerkMag_mean = statistics.mean(accel_jerk_mag_values)
            tBodyAccJerkMag_min = min(accel_jerk_mag_values)
            tBodyAccJerkMag_max = max(accel_jerk_mag_values)
            
            # Frequency domain - Accelerometer Jerk Magnitude
            fBodyAccJerkMag_mean = np.mean(accel_jerk_mag_dominant_freqs) if len(accel_jerk_mag_dominant_freqs) > 0 else 0.0
            fBodyAccJerkMag_min = np.min(accel_jerk_mag_dominant_freqs) if len(accel_jerk_mag_dominant_freqs) > 0 else 0.0
            fBodyAccJerkMag_max = np.max(accel_jerk_mag_dominant_freqs) if len(accel_jerk_mag_dominant_freqs) > 0 else 0.0
            
            # Frequency domain - Accelerometer Jerk
            fBodyAccJerk_mean_X = np.mean(accel_jx_dominant_freqs) if len(accel_jx_dominant_freqs) > 0 else 0.0
            fBodyAccJerk_mean_Y = np.mean(accel_jy_dominant_freqs) if len(accel_jy_dominant_freqs) > 0 else 0.0
            fBodyAccJerk_mean_Z = np.mean(accel_jz_dominant_freqs) if len(accel_jz_dominant_freqs) > 0 else 0.0
            fBodyAccJerk_min_X = np.min(accel_jx_dominant_freqs) if len(accel_jx_dominant_freqs) > 0 else 0.0
            fBodyAccJerk_min_Y = np.min(accel_jy_dominant_freqs) if len(accel_jy_dominant_freqs) > 0 else 0.0
            fBodyAccJerk_min_Z = np.min(accel_jz_dominant_freqs) if len(accel_jz_dominant_freqs) > 0 else 0.0
            fBodyAccJerk_max_X = np.max(accel_jx_dominant_freqs) if len(accel_jx_dominant_freqs) > 0 else 0.0
            fBodyAccJerk_max_Y = np.max(accel_jy_dominant_freqs) if len(accel_jy_dominant_freqs) > 0 else 0.0
            fBodyAccJerk_max_Z = np.max(accel_jz_dominant_freqs) if len(accel_jz_dominant_freqs) > 0 else 0.0
            
            # Time domain - Gyroscope
            tBodyGyro_mean_X = statistics.mean(gx_values)
            tBodyGyro_mean_Y = statistics.mean(gy_values)
            tBodyGyro_mean_Z = statistics.mean(gz_values)
            tBodyGyro_min_X = min(gx_values)
            tBodyGyro_min_Y = min(gy_values)
            tBodyGyro_min_Z = min(gz_values)
            tBodyGyro_max_X = max(gx_values)
            tBodyGyro_max_Y = max(gy_values)
            tBodyGyro_max_Z = max(gz_values)
            
            # Time domain - Gyroscope Magnitude
            tBodyGyroMag_mean = statistics.mean(gyro_mag_values)
            tBodyGyroMag_min = min(gyro_mag_values)
            tBodyGyroMag_max = max(gyro_mag_values)
            
            # Frequency domain - Gyroscope Magnitude
            fBodyGyroMag_mean = np.mean(gyro_mag_dominant_freqs) if len(gyro_mag_dominant_freqs) > 0 else 0.0
            fBodyGyroMag_min = np.min(gyro_mag_dominant_freqs) if len(gyro_mag_dominant_freqs) > 0 else 0.0
            fBodyGyroMag_max = np.max(gyro_mag_dominant_freqs) if len(gyro_mag_dominant_freqs) > 0 else 0.0
            
            # Frequency domain - Gyroscope
            fBodyGyro_mean_X = np.mean(gx_dominant_freqs) if len(gx_dominant_freqs) > 0 else 0.0
            fBodyGyro_mean_Y = np.mean(gy_dominant_freqs) if len(gy_dominant_freqs) > 0 else 0.0
            fBodyGyro_mean_Z = np.mean(gz_dominant_freqs) if len(gz_dominant_freqs) > 0 else 0.0
            fBodyGyro_min_X = np.min(gx_dominant_freqs) if len(gx_dominant_freqs) > 0 else 0.0
            fBodyGyro_min_Y = np.min(gy_dominant_freqs) if len(gy_dominant_freqs) > 0 else 0.0
            fBodyGyro_min_Z = np.min(gz_dominant_freqs) if len(gz_dominant_freqs) > 0 else 0.0
            fBodyGyro_max_X = np.max(gx_dominant_freqs) if len(gx_dominant_freqs) > 0 else 0.0
            fBodyGyro_max_Y = np.max(gy_dominant_freqs) if len(gy_dominant_freqs) > 0 else 0.0
            fBodyGyro_max_Z = np.max(gz_dominant_freqs) if len(gz_dominant_freqs) > 0 else 0.0
            
            # Time domain - Gyroscope Jerk
            tBodyGyroJerk_mean_X = statistics.mean(gyro_jx_values)
            tBodyGyroJerk_mean_Y = statistics.mean(gyro_jy_values)
            tBodyGyroJerk_mean_Z = statistics.mean(gyro_jz_values)
            tBodyGyroJerk_min_X = min(gyro_jx_values)
            tBodyGyroJerk_min_Y = min(gyro_jy_values)
            tBodyGyroJerk_min_Z = min(gyro_jz_values)
            tBodyGyroJerk_max_X = max(gyro_jx_values)
            tBodyGyroJerk_max_Y = max(gyro_jy_values)
            tBodyGyroJerk_max_Z = max(gyro_jz_values)
            
            # Time domain - Gyroscope Jerk Magnitude
            tBodyGyroJerkMag_mean = statistics.mean(gyro_jerk_mag_values)
            tBodyGyroJerkMag_min = min(gyro_jerk_mag_values)
            tBodyGyroJerkMag_max = max(gyro_jerk_mag_values)
            
            # Frequency domain - Gyroscope Jerk Magnitude
            fBodyGyroJerkMag_mean = np.mean(gyro_jerk_mag_dominant_freqs) if len(gyro_jerk_mag_dominant_freqs) > 0 else 0.0
            fBodyGyroJerkMag_min = np.min(gyro_jerk_mag_dominant_freqs) if len(gyro_jerk_mag_dominant_freqs) > 0 else 0.0
            fBodyGyroJerkMag_max = np.max(gyro_jerk_mag_dominant_freqs) if len(gyro_jerk_mag_dominant_freqs) > 0 else 0.0

            # Normalize all features using the training data parameters
            azure_data = {
                "activity": 420,
                "tBodyAcc-mean()-X": normalize_feature(tBodyAcc_mean_X, "tBodyAcc-mean()-X"),
                "tBodyAcc-mean()-Y": normalize_feature(tBodyAcc_mean_Y, "tBodyAcc-mean()-Y"),
                "tBodyAcc-mean()-Z": normalize_feature(tBodyAcc_mean_Z, "tBodyAcc-mean()-Z"),
                "tBodyAcc-max()-X": normalize_feature(tBodyAcc_max_X, "tBodyAcc-max()-X"),
                "tBodyAcc-max()-Y": normalize_feature(tBodyAcc_max_Y, "tBodyAcc-max()-Y"),
                "tBodyAcc-max()-Z": normalize_feature(tBodyAcc_max_Z, "tBodyAcc-max()-Z"),
                "tBodyAcc-min()-X": normalize_feature(tBodyAcc_min_X, "tBodyAcc-min()-X"),
                "tBodyAcc-min()-Y": normalize_feature(tBodyAcc_min_Y, "tBodyAcc-min()-Y"),
                "tBodyAcc-min()-Z": normalize_feature(tBodyAcc_min_Z, "tBodyAcc-min()-Z"),
                "tBodyAccJerk-mean()-X": normalize_feature(tBodyAccJerk_mean_X, "tBodyAccJerk-mean()-X"),
                "tBodyAccJerk-mean()-Y": normalize_feature(tBodyAccJerk_mean_Y, "tBodyAccJerk-mean()-Y"),
                "tBodyAccJerk-mean()-Z": normalize_feature(tBodyAccJerk_mean_Z, "tBodyAccJerk-mean()-Z"),
                "tBodyAccJerk-max()-X": normalize_feature(tBodyAccJerk_max_X, "tBodyAccJerk-max()-X"),
                "tBodyAccJerk-max()-Y": normalize_feature(tBodyAccJerk_max_Y, "tBodyAccJerk-max()-Y"),
                "tBodyAccJerk-max()-Z": normalize_feature(tBodyAccJerk_max_Z, "tBodyAccJerk-max()-Z"),
                "tBodyAccJerk-min()-X": normalize_feature(tBodyAccJerk_min_X, "tBodyAccJerk-min()-X"),
                "tBodyAccJerk-min()-Y": normalize_feature(tBodyAccJerk_min_Y, "tBodyAccJerk-min()-Y"),
                "tBodyAccJerk-min()-Z": normalize_feature(tBodyAccJerk_min_Z, "tBodyAccJerk-min()-Z"),
                "tBodyGyro-mean()-X": normalize_feature(tBodyGyro_mean_X, "tBodyGyro-mean()-X"),
                "tBodyGyro-mean()-Y": normalize_feature(tBodyGyro_mean_Y, "tBodyGyro-mean()-Y"),
                "tBodyGyro-mean()-Z": normalize_feature(tBodyGyro_mean_Z, "tBodyGyro-mean()-Z"),
                "tBodyGyro-max()-X": normalize_feature(tBodyGyro_max_X, "tBodyGyro-max()-X"),
                "tBodyGyro-max()-Y": normalize_feature(tBodyGyro_max_Y, "tBodyGyro-max()-Y"),
                "tBodyGyro-max()-Z": normalize_feature(tBodyGyro_max_Z, "tBodyGyro-max()-Z"),
                "tBodyGyro-min()-X": normalize_feature(tBodyGyro_min_X, "tBodyGyro-min()-X"),
                "tBodyGyro-min()-Y": normalize_feature(tBodyGyro_min_Y, "tBodyGyro-min()-Y"),
                "tBodyGyro-min()-Z": normalize_feature(tBodyGyro_min_Z, "tBodyGyro-min()-Z"),
                "tBodyGyroJerk-mean()-X": normalize_feature(tBodyGyroJerk_mean_X, "tBodyGyroJerk-mean()-X"),
                "tBodyGyroJerk-mean()-Y": normalize_feature(tBodyGyroJerk_mean_Y, "tBodyGyroJerk-mean()-Y"),
                "tBodyGyroJerk-mean()-Z": normalize_feature(tBodyGyroJerk_mean_Z, "tBodyGyroJerk-mean()-Z"),
                "tBodyGyroJerk-max()-X": normalize_feature(tBodyGyroJerk_max_X, "tBodyGyroJerk-max()-X"),
                "tBodyGyroJerk-max()-Y": normalize_feature(tBodyGyroJerk_max_Y, "tBodyGyroJerk-max()-Y"),
                "tBodyGyroJerk-max()-Z": normalize_feature(tBodyGyroJerk_max_Z, "tBodyGyroJerk-max()-Z"),
                "tBodyGyroJerk-min()-X": normalize_feature(tBodyGyroJerk_min_X, "tBodyGyroJerk-min()-X"),
                "tBodyGyroJerk-min()-Y": normalize_feature(tBodyGyroJerk_min_Y, "tBodyGyroJerk-min()-Y"),
                "tBodyGyroJerk-min()-Z": normalize_feature(tBodyGyroJerk_min_Z, "tBodyGyroJerk-min()-Z"),
                "tBodyAccMag-mean()": normalize_feature(tBodyAccMag_mean, "tBodyAccMag-mean()"),
                "tBodyAccMag-max()": normalize_feature(tBodyAccMag_max, "tBodyAccMag-max()"),
                "tBodyAccMag-min()": normalize_feature(tBodyAccMag_min, "tBodyAccMag-min()"),
                "tBodyAccJerkMag-mean()": normalize_feature(tBodyAccJerkMag_mean, "tBodyAccJerkMag-mean()"),
                "tBodyAccJerkMag-max()": normalize_feature(tBodyAccJerkMag_max, "tBodyAccJerkMag-max()"),
                "tBodyAccJerkMag-min()": normalize_feature(tBodyAccJerkMag_min, "tBodyAccJerkMag-min()"),
                "tBodyGyroMag-mean()": normalize_feature(tBodyGyroMag_mean, "tBodyGyroMag-mean()"),
                "tBodyGyroMag-max()": normalize_feature(tBodyGyroMag_max, "tBodyGyroMag-max()"),
                "tBodyGyroMag-min()": normalize_feature(tBodyGyroMag_min, "tBodyGyroMag-min()"),
                "tBodyGyroJerkMag-mean()": normalize_feature(tBodyGyroJerkMag_mean, "tBodyGyroJerkMag-mean()"),
                "tBodyGyroJerkMag-max()": normalize_feature(tBodyGyroJerkMag_max, "tBodyGyroJerkMag-max()"),
                "tBodyGyroJerkMag-min()": normalize_feature(tBodyGyroJerkMag_min, "tBodyGyroJerkMag-min()"),
                "fBodyAcc-mean()-X": normalize_feature(fBodyAcc_mean_X, "fBodyAcc-mean()-X"),
                "fBodyAcc-mean()-Y": normalize_feature(fBodyAcc_mean_Y, "fBodyAcc-mean()-Y"),
                "fBodyAcc-mean()-Z": normalize_feature(fBodyAcc_mean_Z, "fBodyAcc-mean()-Z"),
                "fBodyAcc-max()-X": normalize_feature(fBodyAcc_max_X, "fBodyAcc-max()-X"),
                "fBodyAcc-max()-Y": normalize_feature(fBodyAcc_max_Y, "fBodyAcc-max()-Y"),
                "fBodyAcc-max()-Z": normalize_feature(fBodyAcc_max_Z, "fBodyAcc-max()-Z"),
                "fBodyAcc-min()-X": normalize_feature(fBodyAcc_min_X, "fBodyAcc-min()-X"),
                "fBodyAcc-min()-Y": normalize_feature(fBodyAcc_min_Y, "fBodyAcc-min()-Y"),
                "fBodyAcc-min()-Z": normalize_feature(fBodyAcc_min_Z, "fBodyAcc-min()-Z"),
                "fBodyAccJerk-mean()-X": normalize_feature(fBodyAccJerk_mean_X, "fBodyAccJerk-mean()-X"),
                "fBodyAccJerk-mean()-Y": normalize_feature(fBodyAccJerk_mean_Y, "fBodyAccJerk-mean()-Y"),
                "fBodyAccJerk-mean()-Z": normalize_feature(fBodyAccJerk_mean_Z, "fBodyAccJerk-mean()-Z"),
                "fBodyAccJerk-max()-X": normalize_feature(fBodyAccJerk_max_X, "fBodyAccJerk-max()-X"),
                "fBodyAccJerk-max()-Y": normalize_feature(fBodyAccJerk_max_Y, "fBodyAccJerk-max()-Y"),
                "fBodyAccJerk-max()-Z": normalize_feature(fBodyAccJerk_max_Z, "fBodyAccJerk-max()-Z"),
                "fBodyAccJerk-min()-X": normalize_feature(fBodyAccJerk_min_X, "fBodyAccJerk-min()-X"),
                "fBodyAccJerk-min()-Y": normalize_feature(fBodyAccJerk_min_Y, "fBodyAccJerk-min()-Y"),
                "fBodyAccJerk-min()-Z": normalize_feature(fBodyAccJerk_min_Z, "fBodyAccJerk-min()-Z"),
                "fBodyGyro-mean()-X": normalize_feature(fBodyGyro_mean_X, "fBodyGyro-mean()-X"),
                "fBodyGyro-mean()-Y": normalize_feature(fBodyGyro_mean_Y, "fBodyGyro-mean()-Y"),
                "fBodyGyro-mean()-Z": normalize_feature(fBodyGyro_mean_Z, "fBodyGyro-mean()-Z"),
                "fBodyGyro-max()-X": normalize_feature(fBodyGyro_max_X, "fBodyGyro-max()-X"),
                "fBodyGyro-max()-Y": normalize_feature(fBodyGyro_max_Y, "fBodyGyro-max()-Y"),
                "fBodyGyro-max()-Z": normalize_feature(fBodyGyro_max_Z, "fBodyGyro-max()-Z"),
                "fBodyGyro-min()-X": normalize_feature(fBodyGyro_min_X, "fBodyGyro-min()-X"),
                "fBodyGyro-min()-Y": normalize_feature(fBodyGyro_min_Y, "fBodyGyro-min()-Y"),
                "fBodyGyro-min()-Z": normalize_feature(fBodyGyro_min_Z, "fBodyGyro-min()-Z"),
                "fBodyAccMag-mean()": normalize_feature(fBodyAccMag_mean, "fBodyAccMag-mean()"),
                "fBodyAccMag-max()": normalize_feature(fBodyAccMag_max, "fBodyAccMag-max()"),
                "fBodyAccMag-min()": normalize_feature(fBodyAccMag_min, "fBodyAccMag-min()"),
                "fBodyBodyAccJerkMag-mean()": normalize_feature(fBodyAccJerkMag_mean, "fBodyBodyAccJerkMag-mean()"),
                "fBodyBodyAccJerkMag-max()": normalize_feature(fBodyAccJerkMag_max, "fBodyBodyAccJerkMag-max()"),
                "fBodyBodyAccJerkMag-min()": normalize_feature(fBodyAccJerkMag_min, "fBodyBodyAccJerkMag-min()"),
                "fBodyBodyGyroMag-mean()": normalize_feature(fBodyGyroMag_mean, "fBodyBodyGyroMag-mean()"),
                "fBodyBodyGyroMag-max()": normalize_feature(fBodyGyroMag_max, "fBodyBodyGyroMag-max()"),
                "fBodyBodyGyroMag-min()": normalize_feature(fBodyGyroMag_min, "fBodyBodyGyroMag-min()"),
                "fBodyBodyGyroJerkMag-mean()": normalize_feature(fBodyGyroJerkMag_mean, "fBodyBodyGyroJerkMag-mean()"),
                "fBodyBodyGyroJerkMag-max()": normalize_feature(fBodyGyroJerkMag_max, "fBodyBodyGyroJerkMag-max()"),
                "fBodyBodyGyroJerkMag-min()": normalize_feature(fBodyGyroJerkMag_min, "fBodyBodyGyroJerkMag-min()")
            }

            print(azure_data)
            
            # Send data to Azure IoT Hub asynchronously
            asyncio.create_task(send_to_azure_iot_hub(azure_data))
        
            # Log individual samples to file
            # for sample_id, (ax, ay, az, gx, gy, gz) in enumerate(window):
            #     logger.info(f"{window_id},{sample_id},{ax:.1f},{ay:.1f},{az:.1f},{gx:.1f},{gy:.1f},{gz:.1f}")
    except Exception as e:
        logger.error(f"Decode error: {e}")

async def main():
    global iot_hub_client
    
    # Initialize Azure IoT Hub client if connection string is provided
    if AZURE_IOT_CONNECTION_STRING:
        try:
            iot_hub_client = IoTHubDeviceClient.create_from_connection_string(AZURE_IOT_CONNECTION_STRING)
            await iot_hub_client.connect()
            logger.info("Connected to Azure IoT Hub")
        except Exception as e:
            logger.error(f"Failed to connect to Azure IoT Hub: {e}")
            logger.warning("Continuing without Azure IoT Hub connection")
    else:
        logger.warning("AZURE_IOT_CONNECTION_STRING not set, Azure IoT Hub integration disabled")
    
    logger.info("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=5.0)

    target = None
    for d in devices:
        if d.name and "Old Person Life Invader" in d.name:
            target = d
            break

    if not target:
        logger.error("Peripheral not found!")
        return

    async with BleakClient(target.address) as client:
        logger.info(f"Connected to {target.name}")
        logger.info(f"Calibrating sensors... (collecting {CALIBRATION_SAMPLES} samples)")
        logger.info("Please keep device still during calibration")
        
        await client.start_notify(CHAR_UUID, handle_notification)

        try:
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Disconnecting...")
        finally:
            await client.stop_notify(CHAR_UUID)            
            # Disconnect from Azure IoT Hub
            if iot_hub_client:
                try:
                    await iot_hub_client.disconnect()
                    logger.info("Disconnected from Azure IoT Hub")
                except Exception as e:
                    logger.error(f"Error disconnecting from Azure IoT Hub: {e}")

if __name__ == "__main__":
    asyncio.run(main())