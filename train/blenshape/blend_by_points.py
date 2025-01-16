import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


class Blendshape:
    def __init__(self,source_mesh: Meshes,
                      target_mesh: Meshes,
                      w_chamfer = 1.0,
                      w_edge = 1.0,
                      w_normal=0.01,
                      w_laplacian = 0.1):
        self.src_mesh = source_mesh
        self.trg_mesh = target_mesh
        self.w_chamfer = w_chamfer
        self.w_edge = w_edge
        self.w_normal = w_normal
        self.w_laplacian = w_laplacian
        self.init_data()

    def init_data(self):
        """初始化"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            print("WARNING: CPU only, this will be slow!")
        self.src_mesh = self.src_mesh.to(self.device)
        self.trg_mesh = self.trg_mesh.to(self.device)
        self.deform_verts = torch.full(self.src_mesh.verts_packed().shape,
                                       0.0,
                                       device=self.device,
                                       requires_grad=True)

    def optimizer(self,lr = 1.0,momentum=0.9):
        """优化器"""
        return torch.optim.SGD([self.deform_verts],
                               lr=lr,
                               momentum=momentum)

    def samplePoints(self,src_mesh,trg_mesh=None,num_points = 5000):
        if trg_mesh:
            trg_points = sample_points_from_meshes(trg_mesh, num_points)
        else:
            trg_points = sample_points_from_meshes(self.trg_mesh, num_points)
        src_points = sample_points_from_meshes(src_mesh, num_points)
        return trg_points.to(self.device),src_points.to(self.device)

    def update(self):
        return self.src_mesh.offset_verts(self.deform_verts)

def train(net:Blendshape,
          optimizer: torch.optim,
          num_epoch: int = 1):
    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []
    new_src_mesh = None
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        new_src_mesh = net.update()
        trg_points,src_points = net.samplePoints(new_src_mesh,num_points=5000)
        loss_chamfer, _ = chamfer_distance(trg_points,src_points)
        loss_edge = mesh_edge_loss(new_src_mesh)
        loss_normal = mesh_normal_consistency(new_src_mesh)
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        loss = loss_chamfer * net.w_chamfer + loss_edge * net.w_edge + loss_normal * net.w_normal + loss_laplacian * net.w_laplacian
        print('total_loss = %.6f' % loss)
        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))
        loss.backward()
        optimizer.step()
    return new_src_mesh




