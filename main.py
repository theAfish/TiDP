from TaichiDeepPot.tidp import TiDP as td
import taichi as ti

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch=arch)

mdl = td(graphs=["graphs/graph.000.pb", "graphs/graph.001.pb", "graphs/graph.002.pb", "graphs/graph.003.pb"],
         structure="poscars/cpi-gamma.poscar",
         type_map={"I": 0, "Cs": 1, "Pb": 2},
         res=(940,768))

mdl.set_prob_atom(1)
mdl.show()
