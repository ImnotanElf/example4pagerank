import snap

Graph = snap.GenRndGnm(snap.TNGraph, 100, 1000)
PRankH = Graph.GetPageRank()
for item in PRankH:
    print(item, PRankH[item])

UGraph = snap.GenRndGnm(snap.TUNGraph, 100, 1000)
PRankH = UGraph.GetPageRank()
for item in PRankH:
    print(item, PRankH[item])

Network = snap.GenRndGnm(snap.TNEANet, 100, 1000)
PRankH = Network.GetPageRank()
for item in PRankH:
    print(item, PRankH[item])