import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from deepmd.infer import DeepPot as DP
from deepmd.infer import calc_model_devi
from collections import Counter
import ase.io

atomic_mass = {'X': 1.0, 'H': 1.007825, 'C': 12.01, 'O': 15.9994, 'N': 14.0067, 'S': 31.972071, 'P': 30.973762,
               'I': 126.90447, 'Cs': 132.905, 'Pb': 207.2}

atomic_color = {'X': (1., 1., 1.), 'H': (1., 1., 1.), 'C': (0.56, 0.56, 0.56), 'O': (1., 0.05, 0.05),
                'N': (0.19, 0.31, 0.97), 'S': (0.7, 0.7, 0.), 'P': (1., 0.5, 0.), 'I': (0.58, 0., 0.58),
                'Cs': (0.34, 0.09, 0.56), 'Pb': (0.34, 0.35, 0.38)}

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)


@ti.data_oriented
class TiDP:
    def __init__(self, res=(512, 512), structure=None, graphs=None, type_map=None, dt=2e-1):
        self.ax = None
        self.by = None
        self.cz = None
        self.structure = None
        self.atom_numbers = None
        self.atom_species = None
        self.type_map = type_map
        if structure is not None:
            self.set_system(structure)
        if graphs is not None:
            self.set_graphs(graphs)
        self.window = ti.ui.Window("Interactive DeepMD Visualizer with Taichi!", res=res)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()

        self.atoms = TiAtom.field(shape=self.atom_numbers)
        self.cell = ti.Vector.field(3, dtype=ti.f32, shape=3)
        self.dt = dt

        # For interactive model-devi test when using more than one graphs
        self.probe_atom = None
        self.probe_dt = 100
        self.probe_step = []
        # self.probe_d_v = []
        self.probe_d_f = []

        # For drawing the atoms and box
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=self.atom_numbers)
        self.x_draw = ti.Vector.field(3, dtype=ti.f32, shape=self.atom_numbers)
        self.box_edge = ti.Vector.field(3, dtype=ti.f32, shape=24)

        self.init_system()

    def init_system(self):
        _type = np.array([self.type_map[i] for i in self.structure.get_chemical_symbols()])
        self.atoms.type.from_numpy(_type)
        self.atoms.position.from_numpy(np.array(self.structure.positions, dtype=np.float32))
        self.atoms.mass.from_numpy(
            np.array([atomic_mass[i] for i in self.structure.get_chemical_symbols()], dtype=np.float32))
        self.cell.from_numpy(np.array(self.structure.cell.array, dtype=np.float32))

        self.init_atom_color()
        self.set_camera([20, 30, 40], self.structure.get_center_of_mass())

    def init_atom_color(self):
        _color = []
        for i in self.structure.get_chemical_symbols():
            try:
                _color.append(atomic_color[i])
            except:
                print("Sorry, the color for {} is not defined currently, or maybe a typo? Set color to (1.0, 1.0, "
                      "1.0) for {}".format(i, i))
                _color.append(atomic_color['X'])
        self.color.from_numpy(
            np.array(_color, dtype=np.float32))

    def set_prob_atom(self, index):
        '''
        Setting the probe atom for interactive model deviation test.

        :param index: integer, the index of the probe atom
        :return:
        '''
        self.probe_atom = index

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
        self.ax = atoms.get_cell().array[0, 0]
        self.by = atoms.get_cell().array[1, 1]
        self.cz = atoms.get_cell().array[2, 2]

    def set_box(self):
        idx = [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 0, 0],
               [1, 0, 1], [0, 1, 0], [1, 1, 0],
               [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1],
               [1, 1, 1], [1, 0, 1], [1, 1, 1]]
        _box_edge = np.array(idx, dtype=np.float32) @ np.array(self.structure.cell.array, dtype=np.float32)
        self.box_edge.from_numpy(_box_edge)

    @ti.kernel
    def set_x_draw(self):
        for i in self.atoms:
            self.x_draw[i] = self.atoms.position[i]

    def draw_scene(self):
        self.set_x_draw()
        self.set_box()
        self.camera.track_user_inputs(self.window, movement_speed=0.3, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.2, 0.2, 0.2))
        self.scene.particles(self.x_draw, per_vertex_color=self.color, radius=0.5)
        self.scene.point_light(pos=self.camera.curr_position, color=(0.6, 0.6, 0.6))
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
        self.camera.up(0, 1, 0)

    def plot_model_devi(self, timestep, line1, ax, figure):
        x_np = np.expand_dims(self.atoms.position.to_numpy(), axis=0)
        atype = self.atoms.type.to_numpy()
        cell_np = np.expand_dims(self.cell.to_numpy(), axis=0)
        model_devi = calc_model_devi(x_np, cell_np, atype, self.graphs)
        self.probe_step.append(timestep / self.probe_dt)
        self.probe_d_f.append(model_devi[0][4])
        line1.set_xdata(np.array(self.probe_step))
        line1.set_ydata(np.array(self.probe_d_f))
        ax.relim()
        ax.autoscale_view()
        figure.canvas.draw()
        figure.canvas.flush_events()

    # For interactive checking. TODO: Probe atom and Matplotlib
    def show(self):
        plt.ion()
        figure, ax = plt.subplots(figsize=(7, 4.7))
        line1, = ax.plot(np.array(self.probe_step), np.array(self.probe_d_f))
        plt.title("Model Deviation", fontsize=20)
        plt.xlabel("Step")
        plt.ylabel("Model Deviation")
        timestep = 0
        while self.window.running:
            if self.probe_atom is not None:
                with self.gui.sub_window(name="Probe atom", x=0.05, y=0.05, width=0.2, height=0.2):
                    self.atoms.position[self.probe_atom].x = self.gui.slider_float("x", self.atoms.position[
                        self.probe_atom].x, 0, self.ax)
                    self.atoms.position[self.probe_atom].y = self.gui.slider_float("y", self.atoms.position[
                        self.probe_atom].y, 0, self.by)
                    self.atoms.position[self.probe_atom].z = self.gui.slider_float("z", self.atoms.position[
                        self.probe_atom].z, 0, self.cz)
            if isinstance(self.graphs, list) and timestep % self.probe_dt == 0:
                self.plot_model_devi(timestep, line1, ax, figure)

            self.draw_scene()
            self.window.show()
            timestep += 1

    # For simple DPMD run, under development TODO: 1. Adding NVT thermostat; 2. Adding PBC for non-orthogonal cell
    @ti.kernel
    def update(self):
        for i in self.atoms:
            self.atoms.position[i] += self.atoms.velocity[i] * self.dt
            self.atoms.velocity[i] += self.atoms.force[i] / self.atoms.mass[i] * self.dt
            for j in ti.static(range(3)):
                self.atoms.position[i][j] -= self.cell[j].norm() * ti.round(
                    self.atoms.position[i][j] / self.cell[j].norm() - 0.5)

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
    mass: ti.f32
