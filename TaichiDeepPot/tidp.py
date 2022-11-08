import taichi as ti
import numpy as np
from deepmd.infer import DeepPot as DP
from collections import Counter
import ase.io

atomic_mass = {'H': 1.007825, 'C': 12.01, 'O': 15.9994, 'N': 14.0067, 'S': 31.972071, 'P': 30.973762, 'I': 126.90447,
               'Cs': 132.905, 'Pb': 207.2}

ti.init()

def read_structure(file):
    atoms = ase.io.read(file)
    return atoms


def read_graphs(files):
    graphs = []
    if isinstance(files, list):
        for file in files:
            graphs.append(DP(file))
    else:
        graphs = DP(files)
    return graphs


@ti.data_oriented
class TiDP:
    def __init__(self, res=(512, 512), structure=None, graphs=None, type_map=None, dt=2e-1):
        self.structure = None
        self.atom_numbers = None
        self.atom_species = None
        self.type_map = type_map
        if structure is not None:
            self.set_system(structure)
        if graphs is not None:
            self.graphs = read_graphs(graphs)
        self.window = ti.ui.Window("Interactive DeepMD Visualizer with Taichi!", res=res)
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()

        self.atoms = TiAtom.field(shape=self.atom_numbers)
        self.cell = ti.Vector.field(3, dtype=ti.f64, shape=3)
        self.dt = dt

        # For drawing the atoms and box
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=self.atom_numbers)
        self.x_draw = ti.Vector.field(3, dtype=ti.f32, shape=self.atom_numbers)
        self.box_edge = ti.Vector.field(3, dtype=ti.f32, shape=24)

        self.init_system()

    def init_system(self):
        _type = np.array([self.type_map[i] for i in self.structure.get_chemical_symbols()])
        self.atoms.type.from_numpy(_type)
        self.atoms.position.from_numpy(np.array(self.structure.positions))
        self.atoms.mass.from_numpy(np.array([atomic_mass[i] for i in self.structure.get_chemical_symbols()]))
        self.cell.from_numpy(np.array(self.structure.cell.array))

        self.init_atom_color()

    @ti.kernel
    def init_atom_color(self):
        for i in self.color:
            if self.atoms.type[i] == 0:
                self.color[i] = [1, 0, 0]
            elif self.atoms.type[i] == 1:
                self.color[i] = [0, 1, 0]
            else:
                self.color[i] = [0, 0, 1]

    def set_type_map(self, type_map):
        '''
        :param type_map: should be a python dictionary
        :return:
        '''
        self.type_map = type_map

    def set_system(self, file):
        atoms = ase.io.read(file)
        self.atom_numbers = atoms.get_global_number_of_atoms()
        self.atom_species = len(Counter(atoms.get_atomic_numbers()))
        self.structure = atoms

    def set_box(self):
        idx = [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 0],
               [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]]
        _box_edge = np.array(idx) @ np.array(self.structure.cell.array)
        self.box_edge.from_numpy(_box_edge)

    @ti.kernel
    def set_x_draw(self):
        for i in self.atoms:
            self.x_draw[i] = self.atoms.position[i]

    def draw_scene(self):
        self.set_x_draw()
        self.set_box()
        self.camera.track_user_inputs(self.window, movement_speed=0.3, hold_key=ti.ui.LMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.2, 0.2, 0.2))
        self.scene.particles(self.x_draw, per_vertex_color=self.color, radius=0.5)
        self.scene.point_light(pos=(5, 20, 11), color=(0.6, 0.6, 0.6))
        self.scene.lines(self.box_edge, width=1., color=(1, 1, 1))
        self.canvas.scene(self.scene)

    def set_graphs(self, files):
        graphs = []
        if isinstance(files, list):
            for file in files:
                graphs.append(DP(file))
        else:
            graphs = DP(files)
        self.graphs = graphs

    def set_camera(self, pos, lookat):
        self.camera.position(pos[0], pos[1], pos[2])
        self.camera.lookat(lookat[0], lookat[1], lookat[2])

    def show(self):
        while self.window.running:
            self.draw_scene()
            self.window.show()

    # For simple DPMD run, under development TODO: 1. Adding NVT thermostat; 2. Adding PBC for non-orthogonal cell
    @ti.kernel
    def update(self):
        for i in self.atoms:
            self.atoms.position[i] += self.atoms.velocity[i] * self.dt
            self.atoms.velocity[i] += self.atoms.force[i] / self.atoms.mass[i] * self.dt
            for j in ti.static(range(3)):
                self.atoms.position[i][j] -= self.cell[j].norm() * ti.round(self.atoms.position[i][j] / self.cell[j].norm() - 0.5)

    def run(self):
        substep = 1
        while self.window.running:
            for _ in range(substep):
                self.update()
                x_np = self.atoms.position.to_numpy()
                atype = self.atoms.type.to_numpy()
                cell_np = self.cell.to_numpy()
                if isinstance(self.graphs, list):
                    e_np, f_np, v_np = self.graphs[0].eval(x_np.reshape([1, -1]), cell_np.reshape([1, -1]), atype)
                else:
                    e_np, f_np, v_np = self.graphs.eval(x_np.reshape([1, -1]), cell_np.reshape([1, -1]), atype)
                self.atoms.force.from_numpy(f_np[0])
            self.draw_scene()
            self.window.show()


@ti.dataclass
class TiAtom:
    position: ti.math.vec3
    velocity: ti.math.vec3
    force: ti.math.vec3
    type: ti.i32
    mass: ti.f64
