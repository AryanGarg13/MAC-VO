import numpy as np
pose =     np.load("/home/aadi_iiith/Desktop/IS/Tartan/MAC-VO/Results/MACVO-Fast@K00/10_31_205027/poses.npy")
ref_pose = np.load("/home/aadi_iiith/Desktop/IS/Tartan/MAC-VO/Results/MACVO-Fast@K00/10_31_205027/ref_poses.npy")

print(pose[:5])
print("REF")
print(ref_pose[:5])

import numpy as np
import matplotlib.pyplot as plt

# Example: poses are Nx4x4 SE(3) matrices
# Replace these with your actual lists/arrays of poses
# gt_poses = [...]        # list of 4x4 ground truth matrices
# est_poses = [...]       # list of 4x4 estimated matrices

def extract_xyz(poses):
    """Extract x,y,z from list/array of 4x4 poses."""
    poses = np.array(poses)
    return poses[:, 1], poses[:, 2], poses[:,3]

# Extract trajectories
# p1 = np.load("/home/aadi_iiith/Desktop/IS/Tartan/MAC-VO/Results/MACVO-Fast@K00/10_31_205027/poses.npy")
# p2 = np.load("/home/aadi_iiith/Desktop/IS/Tartan/MAC-VO/Results/MACVO-Fast@K00/10_31_205814/poses.npy")
# print(p1[0])
# print(p2[0])
# print(p1[2])

gt_x, gt_y, gt_z = extract_xyz(pose)
est_x, est_y, est_z = extract_xyz(ref_pose)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# X vs Y
axes[0].plot(gt_x, gt_y, 'b-', label="Ground Truth")
axes[0].plot(est_x, est_y, 'r--', label="Estimated")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].set_title("X vs Y")
axes[0].legend()
axes[0].axis("equal")

# Y vs Z
axes[1].plot(gt_y, gt_z, 'b-', label="Ground Truth")
axes[1].plot(est_y, est_z, 'r--', label="Estimated")
axes[1].set_xlabel("Y")
axes[1].set_ylabel("Z")
axes[1].set_title("Y vs Z")
axes[1].legend()
axes[1].axis("equal")

# X vs Z
axes[2].plot(gt_x, gt_z, 'b-', label="Ground Truth")
axes[2].plot(est_x, est_z, 'r--', label="Estimated")
axes[2].set_xlabel("X")
axes[2].set_ylabel("Z")
axes[2].set_title("X vs Z")
axes[2].legend()
axes[2].axis("equal")

plt.tight_layout()
plt.show()

# gt_x, gt_y, gt_z = extract_xyz(gt_poses)
# est_x, est_y, est_z = extract_xyz(est_poses)

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot Ground Truth and Estimated poses in 3D
ax.plot(gt_x, gt_y, gt_z, 'b-', label="Ground Truth")
ax.plot(est_x, est_y, est_z, 'r--', label="Estimated")

# Labels and Title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Trajectories: Ground Truth vs Estimated")
ax.legend()

plt.show()