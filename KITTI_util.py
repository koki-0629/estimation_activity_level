import sys
from typing import List, Tuple

import numpy as np
import open3d as o3d

def label_to_velo_coordinate(label_vector3d: np.ndarray) -> np.ndarray:
    x = label_vector3d[0]
    y = label_vector3d[1]
    z = label_vector3d[2]
    velo_vector3d = np.zeros(3)
    velo_vector3d[0] = z
    velo_vector3d[1] = -x
    velo_vector3d[2] = -y

    return velo_vector3d

def kitti_label_to_lidar2(size, center, angle):
    angle = -angle - np.pi/2.0
    lwh = np.array([size[2], size[1], size[0]])
    #print(lwh)
    center = np.array([center[2], -center[0], -center[1]])
    center[2] += lwh[2]/2
    #print(center, lwh)
    return center, lwh, angle

class KITTILabel():
    @classmethod
    def create_from_text(cls, line: List[str], scale:float=1.0):
        return cls( type=line[0],
                    truncated=float(line[1]),
                    occluded = int(line[2]),
                    observation_angle = float(line[3]),
                    Bbox2D = np.array(line[4:8], dtype=np.float64),
                    Bbox3D_size = np.array(line[8:11], dtype=np.float64)/scale,
                    Bbox3D_center = np.array(line[11:14], dtype=np.float64)/scale, 
                    rotation_z_deg = np.rad2deg(-float(line[14])))    
                    #rotation_z_deg = float(line[14]))   

    #Bbox3D_sizeはLiDAR座標系で（height(z方向),width(x軸方向),length(y軸方向))
    #
    def __init__(self, type:str, occluded:int, Bbox3D_size:Tuple[np.float64], Bbox3D_center:Tuple[np.float64], rotation_z_deg:np.float64,
                truncated:np.float64=0.0, observation_angle:np.float64=0.0, Bbox2D:Tuple[np.float64]=(600.00, 150.00, 650.00, 200.00)):
        self.type = type
        self.truncated = truncated
        self.occluded = occluded
        self.observation_angle = observation_angle
        if len(Bbox2D) != 4:
            print("Bbox2D dont match")
            print(Bbox2D)
            sys.exit(1)
        self.Bbox2D = Bbox2D
        if len(Bbox3D_size) != 3:
            print("Bbox3D_size dont match")
            print(Bbox3D_size)
            sys.exit(1)
        elif Bbox3D_size[0] <= 0.0 or Bbox3D_size[1] <= 0.0 or Bbox3D_size[2] <= 0.0:
            print("Bbox3D_size has under 0.0 val")
            print(Bbox3D_size)
            sys.exit(1)
        self.Bbox3D_size = np.array(Bbox3D_size, dtype=np.float64)
        if len(Bbox3D_center) != 3:
            print("Bbox3D_center dont match")
            print(Bbox3D_center)
            sys.exit(1)
        self.Bbox3D_center = np.array(Bbox3D_center, dtype=np.float64) 
        self.rotation_y_deg = -rotation_z_deg
        #self.rotation_y_deg = rotation_z_deg

    def __str__(self) -> str:
        return "[{}, {}, {}, {}, {}, {}, {}, {}]".format(
                self.type, self.truncated, self.occluded, self.observation_angle,
                self.Bbox2D, self.Bbox3D_size, self.Bbox3D_center, np.deg2rad(self.rotation_y_deg))

    def __repr__(self) -> str:
        return "\n" + self.__str__()

    def get_corners(self):
        h, w, l = self.Bbox3D_size
        #print(l, w, h)
        x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
        y_corners = [0.0,  0.0,  0.0,  0.0,   -h,   -h,   -h,   -h]
        z_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
        #print(vertex_list)
        ry = np.deg2rad(self.rotation_y_deg)
        print(ry)
        #print("ry =", ry)
        rotation_matrix = np.array([[+np.cos(ry), 0.0, +np.sin(ry)],
                                    [        0.0, 1.0, 0.0],
                                    [-np.sin(ry), 0.0, +np.cos(ry)]], dtype=np.float64)

        corners = np.vstack([x_corners, y_corners, z_corners])
        corners = np.dot(rotation_matrix, corners).T
        corners = corners + self.Bbox3D_center
        #print("corners b", corners)
        for i in range(8):
            corners[i] = label_to_velo_coordinate(corners[i])
        #print("corners a", corners)
        return corners

    def get_box(self, color:Tuple[float]=None):
        h, w, l = self.Bbox3D_size
        # corners = np.array([
        #      l/2, 0.0,  w/2,
        #      l/2, 0.0, -w/2,
        #     -l/2, 0.0, -w/2,
        #     -l/2, 0.0,  w/2,
        #      l/2,  -h,  w/2,
        #      l/2,  -h, -w/2,
        #     -l/2,  -h, -w/2,
        #     -l/2,  -h,  w/2
        # ]).reshape(8, 3)

        # print(corners)
        # if color is None:
            
        #     """    
        #     if self.occluded == 0 :
        #         color = (0.0, 0.0, 1.0)
        #     elif self.occluded == 1:
        #         color = (0.0, 1.0, 0.0)
        #     elif self.occluded == 2:
        #         color = (1.0, 0.0, 0.0)
        #     elif self.occluded == 3:
        #         color = (0.0, 0.0, 0.0)
        #         print("occluded == 3")
        #     else:
        #         color = (0.0, 0.0, 0.0)
        #         print("unknown occluded type :", self.occluded)
        #     """
        
        #     if self.type == "MobileXNoRider" :
        #         color = (0.0, 0.0, 1.0)
        #     elif self.type == "MobileXRider":
        #         color = (0.0, 1.0, 0.0)
        #     elif self.type == "Cyclist":
        #         color = (1.0, 0.0, 0.0)
        #     elif self.type == 3:
        #         color = (0.0, 0.0, 0.0)
        #         print("occluded == 3")
        #     else:
        #         color = (0.0, 0.0, 0.0)
        #         print("unknown occluded type :", self.occluded)

        # AABBox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))
        # AABBox.color = np.array(color, dtype=np.float32)

        # ry = np.deg2rad(self.rotation_y_deg)
        # """
        # rotation_matrix = np.array([[+np.cos(ry), -np.sin(ry), 0.0],
        #                             [+np.sin(ry), +np.cos(ry), 0.0],
        #                             [        0.0,         0.0, 1.0]], dtype=np.float64)
        
        # """
        
        # rotation_matrix = np.array([[+np.cos(ry), 0.0, +np.sin(ry)],
        #                             [        0.0, 1.0, 0.0],
        #                             [-np.sin(ry), 0.0, +np.cos(ry)]], dtype=np.float64)
        # AABBox.rotate(rotation_matrix)
        # AABBox.translate(self.Bbox3D_center)
        # rotation_x = np.array([[1.0,  0.0, 0.0],
        #                        [0.0,  0.0, 1.0],
        #                        [0.0, -1.0, 0.0]], dtype=np.float32)
        # rotation_z = np.array([[ 0.0, 1.0, 0.0],
        #                        [-1.0, 0.0, 0.0],
        #                        [ 0.0, 0.0, 1.0]], dtype=np.float32)
        # O = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # AABBox.rotate(rotation_x, O)
        # AABBox.rotate(rotation_z, O)
        

        lhw = [h, w, l]
        center = self.Bbox3D_center
        angle = np.deg2rad(self.rotation_y_deg)
        print(center, lhw, angle)
        center, lwh, angle = kitti_label_to_lidar2(lhw, center, angle)
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, angle])
        box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        #print(lines)
        if self.type == "MobileXNoRider":
            line_set.paint_uniform_color([1.0,0.0,0.0])
        elif self.type == "MobileXRider":
            line_set.paint_uniform_color([0.0,0.0,1.0])
        return line_set


    #引数はLiDAR座標系で与える

    def translate(self, x:np.float64 =0.0, y:np.float64 =0.0, z:np.float64 =0.0) -> None:
        self.Bbox3D_center[0] -= y
        self.Bbox3D_center[1] -= z
        self.Bbox3D_center[2] += x
    
    #弧度法
    def rotate_z_deg(self, rotation_z_deg:np.float64):
        self.rotation_y_deg -= rotation_z_deg
        if self.rotation_y_deg > 180:
            self.rotation_y_deg -= 360
        if self.rotation_y_deg <= -180:
            self.rotation_y_deg += 360

    def resize(self, size:np.ndarray):
        self.Bbox3D_size = size
    
    def get_size(self) -> np.ndarray:
        return self.Bbox3D_size.copy()

    def set_type(self, type_name:str) -> None:
        self.type = type_name
    
    def label_str(self, no_cam:bool) -> str:
        ry = np.deg2rad(self.rotation_y_deg)
        if no_cam:
            if self.type == "DontCare":
                return "DontCare -1 -1 -10 -1 -1 -1 -1 -1 -1 -1 -1000 -1000 -1000 -10"
            return "{} 0.00 {} 0.00 600.00 150.00 650.00 200.00 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                self.type, self.occluded,
                self.Bbox3D_size[0], self.Bbox3D_size[1], self.Bbox3D_size[2], self.Bbox3D_center[0], self.Bbox3D_center[1], self.Bbox3D_center[2], ry)
            
            # return "{} 0.00 {} 0.00 600.00 150.00 650.00 200.00 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
            #     self.type, self.occluded,
            #     self.Bbox3D_size[0], self.Bbox3D_size[1], self.Bbox3D_size[2], self.Bbox3D_center[0], self.Bbox3D_center[1], self.Bbox3D_center[2], self.rotation_y_deg)
            

        if self.type == "DontCare":
            return "DontCare -1 -1 -10 {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1000 -1000 -1000 -10".format(
            self.Bbox2D[0], self.Bbox2D[1], self.Bbox2D[2], self.Bbox2D[3])
        
        #type 画像からのはみ出し 状態 α 2Drect(minx miny maxx maxy) 3Dsize(h w l)        3Dcenter(x y z)      rotation_y
        return "{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
            self.type, self.truncated, self.occluded, self.observation_angle, self.Bbox2D[0], self.Bbox2D[1], self.Bbox2D[2], self.Bbox2D[3],
            self.Bbox3D_size[0], self.Bbox3D_size[1], self.Bbox3D_size[2], self.Bbox3D_center[0], self.Bbox3D_center[1], self.Bbox3D_center[2], ry)

        # return "{} {:.2f} {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
        #       self.type, self.truncated, self.occluded, self.observation_angle, self.Bbox2D[0], self.Bbox2D[1], self.Bbox2D[2], self.Bbox2D[3],
        #       self.Bbox3D_size[0], self.Bbox3D_size[1], self.Bbox3D_size[2], self.Bbox3D_center[0], self.Bbox3D_center[1], self.Bbox3D_center[2], self.rotation_y_deg)

    def write_Label(label_list:List, filename:str, no_cam:bool):
        with open(filename, mode='w') as f:
            for label in label_list:
                f.write(label.label_str(no_cam) + '\n')


class KITTICalib():
    def __init__(self, lines: List[str]):
        P2 = np.zeros((4,4))
        P2[:3,:4] = np.array(lines[2].split(" ")[1:], dtype=np.float64).reshape(3,4)
        P2[3,3] = 1
        self.P2 = P2

        R0_rect = np.zeros((4,4))
        R0_rect[:3,:3] = np.array(lines[4].split(" ")[1:], dtype=np.float64).reshape(3,3)
        R0_rect[3,3] = 1
        self.R0_rect = R0_rect

        Tr_velo_to_cam = np.zeros((4,4))
        Tr_velo_to_cam[:3,:4] = np.array(lines[5].split(" ")[1:], dtype=np.float64).reshape(3,4)
        Tr_velo_to_cam[3,3] = 1
        self.Tr_velo_to_cam = Tr_velo_to_cam

        Tr_imu_to_velo = np.zeros((4,4))
        Tr_imu_to_velo[:3,:4] = np.array(lines[6].split(" ")[1:], dtype=np.float64).reshape(3,4)
        Tr_imu_to_velo[3,3] = 1
        self.Tr_imu_to_velo = Tr_imu_to_velo

    def to_hom(cart: np.ndarray) -> np.ndarray:
        ndarray = np.ones((cart.shape[0], cart.shape[1] + 1), dtype=cart.dtype)
        ndarray[:,:3] = cart
        return ndarray      
    
    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
  
        pts_rect_hom = KITTICalib.to_hom(pts_rect)  # (N, 4)
        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(self.R0_rect, self.Tr_velo_to_cam).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = KITTICalib.to_hom(pts_lidar)
        #pts_rect = np.dot(pts_lidar_hom, np.dot(self.Tr_velo_to_cam.T, self.R0_rect.T))
        pts_rect = np.dot(pts_lidar_hom, self.Tr_velo_to_cam.T)
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect[:, :3]

    def __str__(self) -> str:
        return "{{P2:\n{},\nR0_rect:\n{},\nTr_velo_to_cam:\n{},\nTr_imu_to_velo:\n{}}}".format(
                self.P2, self.R0_rect, self.Tr_velo_to_cam, self.Tr_imu_to_velo)

    def __repr__(self) -> str:
        return "\n" + self.__str__()


def create_box(color, center, lwh, angle, one_box):
    #print("lwh", lwh)
    #axis_angles = np.array([0, 0, angle + 1e-10])

    rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, angle])
    print(angle)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
    #print("box3d", box3d)
    #[Open3D WARNING] invalid color in PaintUniformColor, clipping to [0, 1]
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    #print("line_set", line_set)
    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)

    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    line_set.paint_uniform_color(color)
    #box3d.color = np.array(color)

    return line_set, box3d, one_box


def create_box_from_txt_3Donly(label_pass, threshold, color_map):
    label_list = []
    Car_box = None
    Car_line = None
    Car_score = 0.0
    Car_count = []
    with open(label_pass) as f:
        for line in f.readlines():
            line = line.split(" ")
            if line[0] == "SeniarCar":
            #if line[0] == "Car":
                Car_count.append(line[8])
                if Car_score < float(line[8]) or True:
                    Car_score = float(line[8])
                    if float(line[8]) > 0.25:
                        Car_line, Car_box = create_box(color_map[line[0]], np.array(line[1:4], dtype=np.float32), np.array(line[4:7], dtype=np.float32), np.float32(line[7]))
                        label_list.append(Car_line)
                        label_list.append(Car_box)
                    #elif float(line[8]) > 0.2:
                        Car_line, Car_box = create_box(           [0,1,0], np.array(line[1:4], dtype=np.float32), np.array(line[4:7], dtype=np.float32), np.float32(line[7]))
                    #else:
                    #    Car_line, Car_box = create_box(           [0,0,1], np.array(line[1:4], dtype=np.float32), np.array(line[4:7], dtype=np.float32), np.float32(line[7]))
                    # label_list.append(Car_line)
                    # label_list.append(Car_box)
                    #print(label_list)
            elif float(line[8]) > threshold:
                #print("skip", line)
                if not( float(line[1]) > 9.0 and float(line[1]) < 25.0 and float(line[2]) > -3.0 and float(line[2]) < 18.0):
                    continue
                line_set, box = create_box(color_map[line[0]], np.array(line[1:4], dtype=np.float32), np.array(line[4:7], dtype=np.float32), np.float32(line[7]))
                label_list.append(line_set)
                label_list.append(box)
    if Car_line is None:
        print("no SeniarCar")
    #else :
        #if len(Car_count) != 1:
        #print(Car_count)
        #label_list.append(Car_line)
        #label_list.append(Car_box)
    #print(label_list)
    return label_list


def kitti_label_to_lidar(size, center, angle):
    angle = -angle - np.pi/2.0
    lwh = np.array([size[2], size[1], size[0]])
    #print(lwh)
    center = np.array([center[2], -center[0], -center[1]])
    center[2] += lwh[2]/2
    #print(center, lwh)
    return center, lwh, angle

def create_box_kitti(color, center, lhw, angle):
    #print(center, lhw, angle)
    center, lwh, angle = kitti_label_to_lidar(center, lhw, angle)
    one_box = [center[0], center[1], center[2], lwh[0], lwh[1], lwh[2], angle]
    #print(lwh)
    return create_box(color, center, lwh, angle, one_box)

def create_box_from_txt(label_path, threshold, color_map):
    label_list = []
    box_list = []
    with open(label_path) as f:
        for line in f.readlines():
            one_box = []
            line = line.split(" ")
            #if True or float(line[15]) > threshold:
            #if (line[0] != 'MobileXNoRider' and line[0] != 'MobileXRider'):# and float(line[15]) > 0.5:
            #if (line[0] != 'DontCare'):# and float(line[15]):
            #if (line[0] == 'MobileXRider') and float(line[15]) > 0.5:
            #if float(line[15]) > 0.4:
            line_set, box, one_box = create_box_kitti(color_map[line[0]], np.array(line[8:11], dtype=np.float32), np.array(line[11:14], dtype=np.float32), np.float32(line[14]))
            label_list.append(line_set)
            #label_list.append(box)
            #one_box = [line[0], line[8], line[9], line[10], line[11], line[12], line[13], line[14]]
            box_list.append(one_box)

    return label_list#, box_list
