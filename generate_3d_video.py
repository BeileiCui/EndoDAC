import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read pcd files
pcd = o3d.io.read_point_cloud("./logs/endodac_fullmodel/models/weights_19/reconstruction/3_1_000001.ply")
# create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# add pcd to visualizer
vis.add_geometry(pcd)

# set the initial view
view_control = vis.get_view_control()
view_control.set_zoom(0.65)  # 设置初始缩放程度

view_control.rotate(100,1000)
# view_control.rotate(2100,0)
# vis.run()

# 创建视频编写器
out_path = "./logs/endodac_fullmodel/models/weights_19/reconstruction/3_1_000001.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可根据需要选择其他编码器
fps = 30  # 帧率
width, height = 924, 508  # 视频帧大小
video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

# 旋转点云的视角并保存为视频
for angle in range(0, 2100*2, 15):  # 从0度到360度，每次旋转5度
    view_control.rotate(15, 0)  # 以y轴旋转5度
    vis.poll_events()
    vis.update_renderer()

    # 获取当前帧的渲染图像
    img = vis.capture_screen_float_buffer()
    img = (np.array(img) * 255).astype(np.uint8)
    # print(img.shape[0],img.shape[1])
    # 调整图像大小以适应视频帧大小
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 写入视频帧
    video_writer.write(img)

# 关闭视频编写器
video_writer.release()

# 关闭可视化窗口
vis.destroy_window()
