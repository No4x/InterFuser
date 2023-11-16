import os

towns=['town01','town02','town03','town04','town05','town06','town07','town10']
for town in towns:
    if not os.path.exists("%s" % town):
        os.mkdir("%s" % town)
    if not os.path.exists("%s/results" % town):
        os.mkdir("%s/results" % town)
