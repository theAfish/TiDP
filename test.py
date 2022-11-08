from TaichiDeepPot.tidp import TiDP as td

mdl = td(graphs=["graphs/graph.000.pb", "graphs/graph.001.pb"],
         structure="poscars/cpi-gamma.poscar",
         type_map={"I": 0, "Cs": 1, "Pb": 2})

mdl.show()

# what is it
