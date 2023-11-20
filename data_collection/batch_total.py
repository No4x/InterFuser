import os


for i in range(8):
    fw = open("run_bash_total.sh", "w")
    fw.write("docker start it0 it1 it2 it3  \n")
    files = []
    files = os.listdir("batch_run_local")
    for i,file in enumerate(files):
        if i !=3 :
            fw.write("bash data_collection/batch_run_local/%s  &\n" % ( file))
        else:
            fw.write("bash data_collection/batch_run_local/%s  \n" % ( file))

