import open3d as o3d
import numpy as np

pcd=o3d.geometry.PointCloud()
path = '/home/myw/haowei/ULIP/data/ULIP-Objaverse_triplets/objaverse_pc_parallel/4887da0aab51406dab3c5cb69ec82404/4887da0aab51406dab3c5cb69ec82404_8192.npz'
point_cloud = np.load(path)['arr_0']


pcd.points = o3d.utility.Vector3dVector(point_cloud)
# o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud('./vis_output/example.pcd', pcd, write_ascii=False, compressed=False, print_progress=False)

print(pcd)


