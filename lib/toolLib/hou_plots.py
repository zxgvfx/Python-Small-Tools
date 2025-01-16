import hou
from PySide2.QtWidgets import QMainWindow, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pytorch3d.ops import sample_points_from_meshes

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111, projection='3d')
        super().__init__(fig)

    def plot_pointcloud(self, mesh, title=""):
        points = sample_points_from_meshes(mesh, 5000)
        x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
        self.axes.scatter3D(x, z, -y)
        self.axes.set_xlabel('x')
        self.axes.set_ylabel('z')
        self.axes.set_zlabel('y')
        self.axes.set_title(title)
        self.axes.view_init(190, 30)


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.setCentralWidget(self.sc)

    def plotPointClouds(self, mesh, title):
        self.sc.plot_pointcloud(mesh, title)
        self.show()