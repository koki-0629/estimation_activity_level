import math
import numpy as np
import open3d as o3d 


def add_intensity(points: np.ndarray) -> np.ndarray:
    """
    numpy型の点群にintensityを加える
    intensityの値は0
    既にintensityがある場合はValueError
    
    Args:
        points: 点群

    Returns:
        np.ndarray: intensityが加えられた点群
    """ 

    ndarray = np.zeros((points.shape[0], points.shape[1] + 1), dtype=points.dtype)
    ndarray[:,:3] = points

    return ndarray

def np_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    """
    点のnumpy配列をopen3dのpcd型に変換する

    Args:
        points: numpy配列

    Returns:
        o3d.geometry.PoinCloud: pcd型
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd


# XY平面上のグリッドを表現するLineSetを生成
def create_xy_plane_LineSet(scale:int=1, line_num:int=10, hegiht:np.float64=0.0) -> o3d.geometry.LineSet:
    num = line_num*2 + 1
    xval = np.concatenate([np.linspace(-line_num*scale, line_num*scale, num, dtype=np.float64), np.full(num, line_num*scale, dtype=np.float64)])
    yval = np.concatenate([np.full(num, line_num*scale, dtype=np.float64), np.linspace(-line_num*scale, line_num*scale, num, dtype=np.float64)])
    zval = np.full(num*2, hegiht, dtype=np.float64)
    points_topright = np.vstack([xval, yval, zval]).T

    xval[num:] = np.full(num, -line_num*scale, dtype=np.float64)
    yval[:num] = np.full(num, -line_num*scale, dtype=np.float64)
    points_bottomleft = np.vstack([xval, yval, zval]).T
    correspondences = []
    for i in range(num*2):
        correspondences.append((i, i))

    xy_plane = o3d.geometry.LineSet.create_from_point_cloud_correspondences(np_pcd(points_topright), np_pcd(points_bottomleft), correspondences)
    return xy_plane

def create_yz_plnae_LineSet(scalse:int=1, line_num:int=10, x_val:np.float64=0.0) -> o3d.geometry.LineSet:
    num = line_num*2 + 1
    xval = np.full(num*2, x_val, dtype=np.float64)
    yval = np.concatenate([np.linspace(-line_num*scalse, line_num*scalse, num, dtype=np.float64), np.full(num, line_num*scalse, dtype=np.float64)])
    zval = np.concatenate([np.full(num, line_num*scalse, dtype=np.float64), np.linspace(-line_num*scalse, line_num*scalse, num, dtype=np.float64)])
    points_topright = np.vstack([xval, yval, zval]).T

    yval[num:] = np.full(num, -line_num*scalse, dtype=np.float64)
    zval[:num] = np.full(num, -line_num*scalse, dtype=np.float64)
    points_bottomleft = np.vstack([xval, yval, zval]).T
    correspondences = []
    for i in range(num*2):
        correspondences.append((i, i))

    yz_plane = o3d.geometry.LineSet.create_from_point_cloud_correspondences(np_pcd(points_topright), np_pcd(points_bottomleft), correspondences)
    return yz_plane


def create_zx_plnae_LineSet(scalse:int=1, line_num:int=10, y_val:np.float64=0.0) -> o3d.geometry.LineSet:
    num = line_num*2 + 1
    xval = np.concatenate([np.full(num, line_num*scalse, dtype=np.float64), np.linspace(-line_num*scalse, line_num*scalse, num, dtype=np.float64)])
    yval = np.full(num*2, y_val, dtype=np.float64)
    zval = np.concatenate([np.linspace(-line_num*scalse, line_num*scalse, num, dtype=np.float64), np.full(num, line_num*scalse, dtype=np.float64)])
    points_topright = np.vstack([xval, yval, zval]).T

    xval[:num] = np.full(num, -line_num*scalse, dtype=np.float64)
    zval[num:] = np.full(num, -line_num*scalse, dtype=np.float64)
    points_bottomleft = np.vstack([xval, yval, zval]).T
    correspondences = []
    for i in range(num*2):
        correspondences.append((i, i))

    zx_plane = o3d.geometry.LineSet.create_from_point_cloud_correspondences(np_pcd(points_topright), np_pcd(points_bottomleft), correspondences)
    return zx_plane



def to_hom(cart: np.ndarray) -> np.ndarray:
    """xyz直交座標系を同次座標系に変換 (内部で使用しています)

    Args:
        cart (np.ndarray): [[x, y, z], ...]

    Returns:
        np.ndarray: [[x, y, z, 1], ...]

    """
    ndarray = np.ones((cart.shape[0], cart.shape[1] + 1), dtype=cart.dtype)
    ndarray[:,:3] = cart

    return ndarray

def create_rotate_matrix(dX: float, dY: float, dZ: float, dAzimuth: float, dVertical: float) -> np.ndarray:
    """filter で使用されている変換行列を生成 (内部で使用しています)

    Args:
        dX (float): X 平行移動
        dY (float): Y 平行移動
        dZ (float): Z 平行移動
        dAzimuth (float): 方位角
        dVertical (float): 仰俯角

    Returns:
        np.ndarray: 変換行列 (4x4)
    """
    translator = np.identity(4)
    translator[:,3] = np.array([dX, dY, dZ, 1], dtype=np.float64).T

    sinAzimuth, cosAzimuth = math.sin(math.radians(dAzimuth)), math.cos(math.radians(dAzimuth))
    sinVertical, cosVertical = math.sin(math.radians(dVertical)), math.cos(math.radians(dVertical))
    rotator = np.array([
        [cosAzimuth, -sinAzimuth, 0, 0],
        [sinAzimuth * cosVertical, cosAzimuth * cosVertical, -sinVertical, 0],
        [sinAzimuth * sinVertical, cosAzimuth * sinVertical, cosVertical, 0],
        [0, 0, 0, 1],
        ], dtype=np.float64)

    return np.dot(translator, rotator)

def create_rotate3_matrix(dX: float, dY: float, dZ: float, roll: float, pitch:float, yaw:float) -> np.ndarray:
    """filter で使用されている変換行列を生成 (内部で使用しています)

    Args:
        dX (float): X 平行移動
        dY (float): Y 平行移動
        dZ (float): Z 平行移動
        dAzimuth (float): 方位角
        dVertical (float): 仰俯角

    Returns:
        np.ndarray: 変換行列 (4x4)
    """
    translator = np.identity(4)
    translator[:,3] = np.array([dX, dY, dZ, 1], dtype=np.float64).T
    #x α
    sinx, cosx = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    #y β
    siny, cosy = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    #z γ
    sinz, cosz = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    rotator = np.array([
        [ cosy*cosz                 , -cosy*sinz                 ,  siny     , 0],
        [ sinx*siny*cosz + cosx*sinz, -sinx*siny*sinz + cosx*cosz, -sinx*cosy, 0],
        [-cosx*siny*cosz + sinx*sinz,  cosx*siny*sinz + sinx*cosz,  cosx*cosy, 0],
        [0, 0, 0, 1], 
        ], dtype=np.float64)

    return np.dot(translator, rotator)


def create_rotate3_matrix_telecognix(dX: float, dY: float, dZ: float, roll: float, pitch:float, yaw:float) -> np.ndarray:
    """filter で使用されている変換行列を生成 (内部で使用しています)

    Args:
        dX (float): X 平行移動
        dY (float): Y 平行移動
        dZ (float): Z 平行移動
        dAzimuth (float): 方位角
        dVertical (float): 仰俯角

    Returns:
        np.ndarray: 変換行列 (4x4)
    """

    s3, c3 = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    s2, c2 = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    s1, c1 = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    rotator = np.array([
        [ c1 * c3 - s1 * s2 * s3, -s1 * c2, c1 * s3 + s1 * s2 * c3, dX],
        [ s1 * c3 + c1 * s2 * s3,  c1 * c2, s1 * s3 - c1 * s2 * c3, dY],
        [-c2 * s3               ,  s2     , c2 * c3               , dZ],
        [0, 0, 0, 1], 
        ], dtype=np.float64)

    return rotator


def apply_calibration(points: np.ndarray, mat:np.ndarray) -> np.ndarray:
    """点群にキャリブレーションを適用

    Args:
        points (np.ndarray): 変換元の点群. [[x, y, z], ...]
        dX (float): X 平行移動
        dY (float): Y 平行移動
        dZ (float): Z 平行移動
        dAzimuth (float): 方位角
        dVertical (float): 仰俯角

    Returns:
        np.ndarray: [[x, y, z], ...]
    """

    hom = to_hom(points)
    # transformed = hom @ mat.T
    
    return np.dot(hom, mat.T)[:,:3]
    # return np.dot(hom, transformed)[:, :3]