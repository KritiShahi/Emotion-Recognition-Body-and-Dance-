"""
Kinematic Emotion Recognition Dataset - Data Setup & Feature Extraction
Day 1: Complete data processing pipeline for BVH motion capture files
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class BVHParser:
    """Parse BVH (Biovision Hierarchy) motion capture files"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.joints = []
        self.hierarchy = {}
        self.motion_data = None
        self.frame_time = 0.0
        self.num_frames = 0
        self.channels = []

    def parse(self):
        """Parse BVH file and extract hierarchy and motion data"""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # Find MOTION section
        motion_idx = next(i for i, line in enumerate(lines) if 'MOTION' in line)

        # Parse hierarchy
        self._parse_hierarchy(lines[:motion_idx])

        # Parse motion data
        self._parse_motion(lines[motion_idx:])

        return self

    def _parse_hierarchy(self, lines):
        """Extract joint hierarchy and channel information"""
        current_joint = None

        for line in lines:
            line = line.strip()

            if 'ROOT' in line or 'JOINT' in line:
                parts = line.split()
                current_joint = parts[1]
                self.joints.append(current_joint)
                self.hierarchy[current_joint] = {'channels': [], 'offset': []}

            elif 'CHANNELS' in line and current_joint:
                parts = line.split()
                num_channels = int(parts[1])
                channels = parts[2:2+num_channels]
                self.hierarchy[current_joint]['channels'] = channels
                self.channels.extend([(current_joint, ch) for ch in channels])

            elif 'OFFSET' in line and current_joint:
                parts = line.split()
                offset = [float(x) for x in parts[1:4]]
                self.hierarchy[current_joint]['offset'] = offset

    def _parse_motion(self, lines):
        """Extract motion frame data"""
        for i, line in enumerate(lines):
            if 'Frames:' in line:
                self.num_frames = int(line.split(':')[1].strip())
            elif 'Frame Time:' in line:
                self.frame_time = float(line.split(':')[1].strip())
            elif i > 2:  # Skip header lines
                break

        # Read motion data
        motion_lines = [line.strip() for line in lines[3:] if line.strip()]
        data = []
        for line in motion_lines:
            values = [float(x) for x in line.split()]
            data.append(values)

        self.motion_data = np.array(data)

    def get_joint_trajectory(self, joint_name):
        """Extract position trajectory for a specific joint"""
        if joint_name not in self.hierarchy:
            return None

        joint_idx = self.joints.index(joint_name)
        channels = self.hierarchy[joint_name]['channels']

        # Find column indices for this joint's channels
        col_start = sum(len(self.hierarchy[j]['channels']) for j in self.joints[:joint_idx])

        # Extract position channels (Xposition, Yposition, Zposition)
        pos_indices = [i for i, ch in enumerate(channels) if 'position' in ch.lower()]

        if pos_indices:
            trajectories = self.motion_data[:, col_start:col_start+len(channels)]
            return trajectories[:, pos_indices]

        return None

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class KinematicFeatureExtractor:
    """Extract emotion-relevant features from motion capture data"""

    # Key joints for emotion recognition
    KEY_JOINTS = [
        'Hips', 'Spine', 'Spine1', 'Neck', 'Head',
        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
        'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
        'LeftUpLeg', 'LeftLeg', 'LeftFoot',
        'RightUpLeg', 'RightLeg', 'RightFoot'
    ]

    def __init__(self, bvh_parser):
        self.parser = bvh_parser
        self.features = {}

    def extract_all_features(self):
        """Extract comprehensive feature set"""
        self.features = {}

        # Spatial features
        self._extract_spatial_features()

        # Kinematic features
        self._extract_kinematic_features()

        # Postural features
        self._extract_postural_features()

        # Statistical features
        self._extract_statistical_features()

        return self.features

    def _extract_spatial_features(self):
        """Body expansion, contraction, and spatial extent"""
        trajectories = self._get_available_trajectories()

        if not trajectories:
            return

        # Body bounding box volume
        all_positions = np.concatenate(list(trajectories.values()), axis=0)
        ranges = np.ptp(all_positions, axis=0)  # Range per dimension
        self.features['body_volume_mean'] = np.prod(ranges)

        # Body expansion/contraction over time
        volumes = []
        for frame in range(self.parser.num_frames):
            frame_positions = np.array([traj[frame] for traj in trajectories.values()])
            frame_ranges = np.ptp(frame_positions, axis=0)
            volumes.append(np.prod(frame_ranges))

        self.features['body_expansion_std'] = np.std(volumes)
        self.features['body_expansion_range'] = np.max(volumes) - np.min(volumes)

        # Limb extension
        if 'LeftHand' in trajectories and 'LeftShoulder' in trajectories:
            left_extension = np.linalg.norm(
                trajectories['LeftHand'] - trajectories['LeftShoulder'], axis=1
            )
            self.features['left_arm_extension_mean'] = np.mean(left_extension)
            self.features['left_arm_extension_std'] = np.std(left_extension)

        if 'RightHand' in trajectories and 'RightShoulder' in trajectories:
            right_extension = np.linalg.norm(
                trajectories['RightHand'] - trajectories['RightShoulder'], axis=1
            )
            self.features['right_arm_extension_mean'] = np.mean(right_extension)
            self.features['right_arm_extension_std'] = np.std(right_extension)

    def _extract_kinematic_features(self):
        """Velocity, acceleration, and jerk features"""
        trajectories = self._get_available_trajectories()

        for joint_name, trajectory in trajectories.items():
            # Velocity (first derivative)
            velocity = np.diff(trajectory, axis=0)
            speed = np.linalg.norm(velocity, axis=1)

            self.features[f'{joint_name}_speed_mean'] = np.mean(speed)
            self.features[f'{joint_name}_speed_std'] = np.std(speed)
            self.features[f'{joint_name}_speed_max'] = np.max(speed)

            # Acceleration (second derivative)
            acceleration = np.diff(velocity, axis=0)
            acc_magnitude = np.linalg.norm(acceleration, axis=1)

            self.features[f'{joint_name}_acceleration_mean'] = np.mean(acc_magnitude)

            # Jerk (third derivative) - smoothness indicator
            jerk = np.diff(acceleration, axis=0)
            jerk_magnitude = np.linalg.norm(jerk, axis=1)

            self.features[f'{joint_name}_jerk_mean'] = np.mean(jerk_magnitude)

    def _extract_postural_features(self):
        """Posture-related features"""
        trajectories = self._get_available_trajectories()

        # Head position relative to hips (posture)
        if 'Head' in trajectories and 'Hips' in trajectories:
            head_hip_diff = trajectories['Head'] - trajectories['Hips']
            self.features['head_height_mean'] = np.mean(head_hip_diff[:, 1])  # Y-axis
            self.features['head_forward_lean_mean'] = np.mean(head_hip_diff[:, 2])  # Z-axis

        # Shoulder width (openness)
        if 'LeftShoulder' in trajectories and 'RightShoulder' in trajectories:
            shoulder_width = np.linalg.norm(
                trajectories['LeftShoulder'] - trajectories['RightShoulder'], axis=1
            )
            self.features['shoulder_width_mean'] = np.mean(shoulder_width)
            self.features['shoulder_width_std'] = np.std(shoulder_width)

        # Spine curvature (if multiple spine joints available)
        spine_joints = [j for j in trajectories.keys() if 'Spine' in j]
        if len(spine_joints) >= 2:
            # Simple curvature measure
            spine_positions = [trajectories[j] for j in sorted(spine_joints)]
            curvatures = []
            for frame in range(len(spine_positions[0])):
                positions = np.array([sp[frame] for sp in spine_positions])
                # Measure deviation from straight line
                if len(positions) > 2:
                    curvature = np.std(positions[:, 0])  # X-axis deviation
                    curvatures.append(curvature)
            if curvatures:
                self.features['spine_curvature_mean'] = np.mean(curvatures)

    def _extract_statistical_features(self):
        """Overall movement statistics"""
        trajectories = self._get_available_trajectories()

        all_speeds = []
        for trajectory in trajectories.values():
            velocity = np.diff(trajectory, axis=0)
            speed = np.linalg.norm(velocity, axis=1)
            all_speeds.extend(speed)

        if all_speeds:
            self.features['overall_speed_mean'] = np.mean(all_speeds)
            self.features['overall_speed_std'] = np.std(all_speeds)
            self.features['overall_speed_median'] = np.median(all_speeds)
            self.features['movement_intensity'] = np.percentile(all_speeds, 90)

    def _get_available_trajectories(self):
        """Get trajectories for all available key joints"""
        trajectories = {}
        for joint in self.KEY_JOINTS:
            if joint in self.parser.joints:
                traj = self.parser.get_joint_trajectory(joint)
                if traj is not None and len(traj) > 0:
                    trajectories[joint] = traj
        return trajectories

# ============================================================================
# DATASET PROCESSOR
# ============================================================================

class EmotionDatasetProcessor:
    """Process entire emotion recognition dataset"""

    def __init__(self, data_dir, fileinfo_csv, subject_info_csv=None):
        self.data_dir = Path(data_dir)
        self.fileinfo = pd.read_csv(fileinfo_csv)
        self.subject_info = pd.read_csv(subject_info_csv) if subject_info_csv else None
        self.features_df = None

    def process_dataset(self, sample_size=None, save_path='features_dataset.csv'):
        """Process all BVH files and extract features"""

        print(f"Processing {len(self.fileinfo)} files...")
        if sample_size:
            print(f"Sampling {sample_size} files for quick testing")
            fileinfo_sample = self.fileinfo.sample(n=min(sample_size, len(self.fileinfo)),
                                                   random_state=42)
        else:
            fileinfo_sample = self.fileinfo

        all_features = []
        failed_files = []

        for idx, row in fileinfo_sample.iterrows():
            filename = row['filename']
            bvh_path = self.data_dir / f"{filename}.bvh"

            if not bvh_path.exists():
                failed_files.append(filename)
                continue

            try:
                # Parse BVH file
                parser = BVHParser(str(bvh_path))
                parser.parse()

                # Extract features
                extractor = KinematicFeatureExtractor(parser)
                features = extractor.extract_all_features()

                # Add metadata
                features['filename'] = filename
                features['actor_ID'] = row['actor_ID']
                features['emotion'] = row['emotion']
                features['gender'] = row['actor_gender']
                features['scenario_ID'] = row['scenario_ID']
                features['version'] = row['version']
                features['num_frames'] = parser.num_frames
                features['duration'] = parser.num_frames * parser.frame_time

                all_features.append(features)

                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(fileinfo_sample)} files...")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                failed_files.append(filename)

        # Create DataFrame
        self.features_df = pd.DataFrame(all_features)

        # Save to CSV
        self.features_df.to_csv(save_path, index=False)
        print(f"\n✓ Features saved to {save_path}")
        print(f"✓ Successfully processed: {len(all_features)} files")
        if failed_files:
            print(f"✗ Failed files: {len(failed_files)}")

        return self.features_df

    def get_dataset_summary(self):
        """Print dataset statistics"""
        if self.features_df is None:
            print("No features extracted yet. Run process_dataset() first.")
            return

        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)

        print(f"\nTotal samples: {len(self.features_df)}")
        print(f"Total features: {len([col for col in self.features_df.columns if col not in ['filename', 'actor_ID', 'emotion', 'gender', 'scenario_ID', 'version', 'num_frames', 'duration']])}")

        print("\n--- Emotion Distribution ---")
        print(self.features_df['emotion'].value_counts())

        print("\n--- Gender Distribution ---")
        print(self.features_df['gender'].value_counts())

        print("\n--- Actor Distribution ---")
        print(f"Unique actors: {self.features_df['actor_ID'].nunique()}")

        print("\n--- Duration Statistics ---")
        print(f"Mean duration: {self.features_df['duration'].mean():.2f}s")
        print(f"Min duration: {self.features_df['duration'].min():.2f}s")
        print(f"Max duration: {self.features_df['duration'].max():.2f}s")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Configuration
    DATA_DIR = "/content/drive/MyDrive/kinematic_dataset_final/BVH"  # Directory containing BVH files
    FILEINFO_CSV = "/content/drive/MyDrive/kinematic_dataset_final/file-info.csv"
    SUBJECT_INFO_CSV = "/content/drive/MyDrive/kinematic_dataset_final/subject_info.csv"

    # For quick testing, process subset (set to None for full dataset)
    SAMPLE_SIZE = 100  # Process 100 files for testing, set to None for all

    print("="*60)
    print("KINEMATIC EMOTION RECOGNITION - DATA SETUP")
    print("="*60)

    # Initialize processor
    processor = EmotionDatasetProcessor(
        data_dir=DATA_DIR,
        fileinfo_csv=FILEINFO_CSV,
        subject_info_csv=SUBJECT_INFO_CSV
    )

    # Process dataset
    features_df = processor.process_dataset(
        sample_size=SAMPLE_SIZE,
        save_path='features_dataset.cbsv'
    )

    # Print summary
    processor.get_dataset_summary()

    print("\n" + "="*60)
    print("✓ DATA SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review features_dataset.csv")
    print("2. Run exploratory data analysis")
    print("3. Proceed to model training (Day 2)")
