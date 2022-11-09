from TaichiDeepPot.tidp import TiDP as td

mdl = td(graphs=["graphs/graph.000.pb", "graphs/graph.001.pb", "graphs/graph.002.pb", "graphs/graph.003.pb"],
         structure="poscars/pbi2.poscar",
         type_map={"I": 0, "Pb": 2},
         res=(512,512))

# show model deviation
mdl.set_prob_atom(5)
mdl.show()

# # run a simple DPMD
# mdl.run()
