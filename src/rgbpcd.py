import numpy as np
import rospy
import cv2
from sensor_msgs.msg import PointCloud2,PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcd2


class test_depth_to_pcd(object):
    #kitti dataset parameters

    def __init__(self, resize_scale):
        self.resize_scale = resize_scale
        self.resize_camera = np.array((721.5377 / self.resize_scale, 0, 609.5593 / self.resize_scale, 0,
                                       0, 721.5377 / self.resize_scale, 172.854 / self.resize_scale, 0,
                                       0, 0, 1, 0), dtype="float").reshape(3, 4)

        self.resize_camera = np.matrix(self.resize_camera).I
        self.pixel = np.array([0,0,1]).reshape(3,-1)
        
        self.vector_array = np.zeros((375//self.resize_scale,1242//self.resize_scale,3))
        self.get_depth_vector()

        #camera to ROS
        self.vector_array = self.vector_array.reshape(-1,1,3)
        self.vector_array[:,:,0],self.vector_array[:,:,1],self.vector_array[:,:,2]=self.vector_array[:,:,2],-self.vector_array[:,:,0],-self.vector_array[:,:,1]
        self.vector_array = self.vector_array.reshape(375//self.resize_scale,1242//self.resize_scale,3)
        
        return

    def get_pcd(self, depth):
        self.depth = cv2.resize(depth, (int(1242 / self.resize_scale), int(375 / self.resize_scale)))
        pcd_list = self.vector_array.copy()

        pcd_list[:,:,]  *= self.depth.reshape(int(375 / self.resize_scale), int(1242 / self.resize_scale),1)            
        pcd_list = pcd_list.reshape(-1,3)
        return pcd_list

    def get_depth_vector(self):
        fake_depth =np.zeros((352//self.resize_scale,1216//self.resize_scale))
        it = np.nditer(fake_depth, flags=['multi_index'])
        with it:
            while not it.finished:
                self.pixel[0] = it.multi_index[1]
                self.pixel[1] = it.multi_index[0]
                point = np.dot(self.resize_camera, self.pixel)          
                self.vector_array[it.multi_index] = point[0:3].T[0]
                it.iternext()





if __name__ == "__main__":
    #初始化点云转换函数
    depth2pcd =test_depth_to_pcd(1)


    #读取数据
    bgr = cv2.imread("./data/0000000150.png")
    depth = cv2.imread("./data/depth.png")
    #处理
    depth = depth[:,:,0]
    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)



    #取得点云 以及RGB
    pointcloud = depth2pcd.get_pcd(depth)
    rgb_list = rgb.reshape(-1,3)

    #因为是稀疏点云图 所以过滤掉没有深度值的点
    depth = depth.reshape(-1,1)
    filter = np.where(depth!=0)

    pointcloud = pointcloud[filter[0],:]
    rgb_list = rgb_list[filter[0],:]


    #ros初始化
    rospy.init_node("rgbpcd")
    pcd_publish = rospy.Publisher("test_pcd",PointCloud2,queue_size=1)
    rate =rospy.Rate(20)


    # #RGBF32
    # color = (rgb_list/255).astype(np.float32)
    # fields = [  PointField("x",0,PointField.FLOAT32,1),
    #                 PointField("y",4,PointField.FLOAT32,1),
    #                 PointField("z",8,PointField.FLOAT32,1),
    #                 PointField("r",12,PointField.FLOAT32,1),
    #                 PointField("g",16,PointField.FLOAT32,1),
    #                 PointField("b",20,PointField.FLOAT32,1)]


    # RGB8
    fields = [  PointField("x",0,PointField.FLOAT32,1),
                PointField("y",4,PointField.FLOAT32,1),
                PointField("z",8,PointField.FLOAT32,1),
                PointField("rgb",12,PointField.FLOAT32,1)]
    
    r,g,b = rgb_list[:,0].astype(np.uint32), rgb_list[:,1].astype(np.uint32), rgb_list[:,2].astype(np.uint32)
    rgb = np.array((r << 16) | (g << 8 ) | (b << 0),dtype=np.uint32)
    color = rgb.reshape(-1,1)
    color.dtype = np.float32


    data = np.hstack((pointcloud,color)).astype(np.float32)
    header = Header()
    header.frame_id = "map"

    while not rospy.is_shutdown():
        msg = pcd2.create_cloud(header=header,fields=fields,points=data )
        pcd_publish.publish(msg)
        rate.sleep()



