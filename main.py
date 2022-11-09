from TaichiDeepPot.tidp import TiDP as td
import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

mdl = td(graphs=["graphs/graph.000.pb", "graphs/graph.001.pb", "graphs/graph.002.pb", "graphs/graph.003.pb"],
         structure="poscars/pbi2.poscar",
         type_map={"I": 0, "Pb": 2},
         res=(512,512))

mdl.set_prob_atom(5)
mdl.show()
