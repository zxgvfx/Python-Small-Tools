import numpy as np
import hou
import torch
from pytorch3d.structures import Meshes,Pointclouds
from pytorch3d.io import load_obj, save_obj



class CoreFuntion:

    def __init__(self, hou_geo: hou.Geometry,):
        self.geo = hou_geo

    def create_hou_points(self, p: np.ndarray, ):
        """
        To create houdini point geometry.
        :param p: verts data.
        :return None
        """
        pt = self.geo.createPoint()
        pt.setPosition((float(p[0]), float(p[1]), float(p[2])))

    def create_hou_prim(self, p: np.ndarray, ):
        """
        To create houdini prim geometry.
        :param p: verts_id data.
        :return:
        """
        poly = self.geo.createPolygon()
        for pt in p[:-1][::-1]:
            pts = self.geo.point(int(pt))
            poly.addVertex(pts)

    def create_hou_normal(self,p,):
        """
        Todo: To create houdini normal.
        :param p: normal data
        :return:
        """
        pass

    def create_hou_color(self,p,):
        """
        Todo: To create houdini color
        :param p: texture data
        :return:
        """
        pass

    def create_np_verts(self, ) -> np.ndarray:
        """
        hou point(verts) position to numpy data。
        :param p: numpy data with index, index -> p[-1]
        :return: numpy data shape(3,)
        """
        verts_list = []
        for i in self.geo.points():
            pos = np.array(i.position())
            verts_list.append(pos)
        return np.array(verts_list,dtype=np.float32)

    def create_np_verts_id(self, )-> np.ndarray:
        """
        hou face point(face_verts) index to numpy data。
        :param p: numpy data with index, index -> p[-1]
        :return: numpy data shape(triangles or quads,)
        """
        verts_id_list = []
        for i in self.geo.prims():
            vert_id = np.array([n.number() for n in i.points()])[::-1]
            verts_id_list.append(vert_id)
        return np.array(verts_id_list,dtype=np.int64)

    def create_np_normal(self,p,) -> np.ndarray:
        """
        hou normal to numpy data
        :param p:
        :return:
        """
        pass

    def create_np_texture(self,p,) ->np.ndarray:
        """
        hou color to numpy data
        :param p:
        :return:
        """
        pass

class Convert:
    def __init__(self,
                 t3d_geo: Meshes|Pointclouds = None,
                 hou_geo: hou.Geometry = None,
                 device: torch.device=torch.device("cpu"),
                 force_device = False):
        """
        pytorch3d 数据与houdini Geometry 数据交换
        :param t3d_geo: pytoch3d数据
        :param hou_geo: houdini geometry 数据
        :param device:  cpu 或者 gpu 或者 apu
        :param force_device: 如果 值为 True 那么忽略pytorch3d 的数据位置，强制使用参数 device的驱动位置
        """
        self.t3d_geo = t3d_geo
        self.geo = hou_geo
        self.device = device
        self.force_device = force_device
        self.verts_index = torch.tensor([])
        self.verts_id_index = torch.tensor([])
        self._init_data_()


    def create_index_from_meshes(self,meshes: Meshes = None)->tuple[torch.Tensor,torch.Tensor]:
        if meshes:
            if isinstance(meshes, Meshes):
                v = meshes.verts_packed()
                f = meshes.faces_packed()
            elif isinstance(meshes,Pointclouds):
                v = meshes.points_packed()
                f = torch.tensor([])
        else:
            if isinstance(self.t3d_geo, Meshes):
                v = self.t3d_geo.verts_packed()
                f = self.t3d_geo.faces_packed()
            elif isinstance(self.t3d_geo, Pointclouds):
                v = self.t3d_geo.points_packed()
                f = torch.tensor([])
        verts_index, verts_id_index = self.create_index_by_data(v, f)
        return verts_index, verts_id_index

    def create_index_from_geo(self,geo:hou.Geometry = None)->tuple[torch.Tensor,torch.Tensor]:
        if geo:
            v_num = len(geo.points())
            f_num = len(geo.prims())
            cf = CoreFuntion(geo)
        else:
            v_num = len(self.geo.points())
            f_num = len(self.geo.prims())
            cf = CoreFuntion(self.geo)

        verts_np = cf.create_np_verts()
        v = torch.from_numpy(verts_np)
        v = v.to(device=self.device)
        # 目前只支持3边面
        if f_num != 0:
            verts_id_np = cf.create_np_verts_id()
            f = torch.from_numpy(verts_id_np)
            f = f.to(device=self.device)
            verts_index, verts_id_index = self.create_index_by_data(v,f)
        else:
            verts_index, verts_id_index = self.create_index_by_data(v)
        return verts_index, verts_id_index

    def _init_data_(self):
        if not self.t3d_geo and not self.geo:
            raise IOError("确保输入是否正确！！")
        if self.t3d_geo and not self.geo:
            self.verts_index, self.verts_id_index = self.create_index_from_meshes()
            self.geo = hou.Geometry()
        if self.geo and not self.t3d_geo :
            self.verts_index,self.verts_id_index = self.create_index_from_geo()


    def toHoudini(self,verts_index:torch.Tensor=torch.tensor([]),
                  verts_id_index:torch.Tensor=torch.tensor([]),
                  meshes:Meshes=None)->hou.Geometry:
        geo = hou.Geometry()
        v,f = verts_index,verts_id_index
        if verts_index.size().numel()==0:
            if self.verts_index.size().numel() == 0:
                if meshes:
                    v,f = self.create_index_from_meshes(meshes)
                else:
                    v,f = self.create_index_from_meshes(self.t3d_geo)
            else:
                v,f =self.verts_index,self.verts_id_index

        else:
            v,f = verts_index,verts_id_index
        return gen_geo_by_data(v,f)

    def updateFromGeo(self,geo: hou.Geometry):
        """
        从houdini中处理之后，转到 更新转化之前的 Meshes 物体
        :param geo: houdini geometry topo 必须一致
        :return:
        """
        verts_index, verts_id_index = self.create_index_from_geo(geo)
        verts,_ = torch.split(verts_index, verts_index.shape[-1]-1, dim=1)
        verts_id,_ = torch.split(verts_id_index, verts_index.shape[-1]-1, dim=1)
        if verts.size().numel() == 0:
            self.t3d_geo = Pointclouds(points=[verts[0]])
        else:
            self.t3d_geo = Meshes(verts=[verts[0]],faces=[verts_id[0]])

    def updateFromMeshes(self,meshes: Meshes):
        """
        从t3d中更新数据到houdini geometry
        :param meshes: 必须和 t3d中的Meshes 拓扑一致
        :return:
        """
        verts_index, verts_id_index = self.create_index_from_geo(meshes)
        geo = self.toHoudini(verts_index, verts_id_index)
        self.geo = geo


    def toMeshes(self,verts_index:torch.Tensor=torch.tensor([]),
                  verts_id_index:torch.Tensor=torch.tensor([]),
                 geo:hou.Geometry = None)->Meshes|Pointclouds:

        v,f = verts_index,verts_id_index
        if verts_index.size().numel() == 0:
            # error: 如果模型既有三角面，又有四边面，需要将其全部转化为三角面或四边面
            if self.verts_index.size().numel() == 0:
                if geo:
                    v,f = self.create_index_from_geo(geo)
                else:
                    raise IOError("none data to convert!!")
            else:
                v,f = self.verts_index,self.verts_id_index
        return gen_meshes_by_data(v,f)

    def create_index_by_data(self,verts_data: torch.Tensor,
                             face_data:  torch.Tensor = torch.tensor([]))\
                             -> tuple[torch.Tensor,torch.Tensor]:
        if self.force_device:
            device = self.device
        else:
            device = verts_data.device
        id = np.arange(verts_data.shape[0],dtype=np.float32).reshape(-1,1)
        point_id = torch.from_numpy(id)
        point_id = point_id.to(device=device)
        verts_data = verts_data.to(device=device)
        verts_index = torch.cat([verts_data,point_id],1)
        if face_data.size().numel()!=0:
            p_id = np.arange(face_data.shape[0],dtype=np.float32).reshape(-1,1)
            prim_id = torch.from_numpy(p_id)
            prim_id = prim_id.to(device=device)
            face_data = face_data.to(device=device)
            verts_id_index = torch.cat([face_data,prim_id], 1)
        else:
            verts_id_index = face_data
        return verts_index, verts_id_index

def gen_meshes_by_data(verts_index:torch.Tensor,
           verts_id_index:torch.Tensor)->Meshes|Pointclouds:
    if verts_index.size().numel() !=0 and verts_id_index.size().numel() !=0 :
        v = torch.split(verts_index,verts_index.shape[-1]-1,dim=1)
        f = torch.split(verts_id_index,verts_id_index.shape[-1]-1,dim=1)

        return Meshes(verts =[v[0]], faces=[f[0]])
    else:
        if verts_index.size().numel() !=0:
            v = torch.split(verts_index, verts_index.shape[-1]-1, dim=1)
            return Pointclouds(points=[v[0]])

def gen_geo_by_data(verts_index:torch.Tensor,
           verts_id_index:torch.Tensor)->hou.Geometry:
    geo = hou.Geometry()
    cvt = CoreFuntion(geo)
    np.apply_along_axis(cvt.create_hou_points, 1, verts_index.cpu().detach().numpy())
    if verts_id_index.size().numel() !=0 :
        np.apply_along_axis(cvt.create_hou_prim, 1, verts_id_index.cpu().detach().numpy())
    return geo

if __name__ == "__main__":
    import os
    dir_path = os.path.dirname(__file__)
    trg_obj = os.path.abspath(f'{dir_path}/../../file/obj/dolphin.obj')
    verts, faces, aux = load_obj(trg_obj)
    # ----------------
    trg_mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
    trg_mesh.cpu()
    obj_cvt = Convert(t3d_geo=trg_mesh)
    geo = obj_cvt.toHoudini()
    print(geo.boundingBox())

    # [-0.141481, 0.139833, 0.031976, 0.493277, -0.283368, 0.430819]
    #----------------
    geo_hou = hou.Geometry()
    box_verb = hou.sopNodeTypeCategory().nodeVerb("box")
    box_verb.setParms({
        "t": hou.Vector3(0.5, -0.5, 2.0),
        "scale": 0.5,
    })
    box_verb.execute(geo_hou , [])
    convert = Convert(hou_geo = geo_hou)
    verts,verts_id =  convert.toMeshes()
    print(verts,verts_id)
