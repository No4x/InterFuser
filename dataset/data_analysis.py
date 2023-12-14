import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
def analyse_data(weathers):

    for weather in weathers:
        os.makedirs(os.path.join(weather,"static/json"),exist_ok=True)
        path=weather+'/results'
        files = os.listdir(path)
        analysis_dict=dict()
        analysis_dict.update({"meta":{"exceptions":[]}})
        for file in files:
            analysis_dict = dict()
            analysis_dict.update({"meta": {"exceptions": [],"failed_number":0}})
            with open(os.path.join(path, file), 'r') as read_file:
                data = json.load(read_file)
                if "index" in data["_checkpoint"]["global_record"]:
                    analysis_dict["meta"]["total_number"]=data["_checkpoint"]["progress"][0]
                    analysis_dict["meta"].update({"scores":data["_checkpoint"]["global_record"]["scores"]} )
                    for exception in data["_checkpoint"]["global_record"]["meta"]["exceptions"]:
                        status=exception[-1]
                        if status!='Completed':

                            analysis_dict["meta"]["exceptions"].append(exception)
                            analysis_dict["meta"]["failed_number"]+=1
                    analysis_dict["meta"]["failure_rate"]=f'{analysis_dict["meta"]["failed_number"]/analysis_dict["meta"]["total_number"]:.2%}'
                    with open(os.path.join(weather,"static/json", file), 'w') as write_file:
                        json.dump(analysis_dict,write_file,indent=4)

def dataset_index(weathers):
    frames=0
    for weather in weathers:
        path=os.path.join(weather,"data")
        dirs=os.listdir(path)
        for dir in dirs:
            if len(os.listdir(os.path.join(path,dir,'lidar')))==0:
                shutil.rmtree(os.path.join(path,dir))
                continue
            with open("dataset_index.txt","a") as write_file:
                frames+=len(os.listdir(os.path.join(path,dir,'lidar')))
                str=f"{os.path.join(path,dir)}  {len(os.listdir(os.path.join(path,dir,'lidar'))) } \n"
                write_file.write(str)
    print(frames)
def create_plot(path,weather):
    dict={"failed_number":[],"total_number":[],"failure_rate":[],"score_composed":[]}
    files=os.listdir(path)
    for file in files:
        with open(os.path.join(path,file),'r') as read_file:
            data = json.load(read_file)
            for key,value in data["meta"].items():
                if key != "exceptions" :
                    if key == "scores":
                        dict["score_composed"].append(f"{value['score_composed']:.2f}")
                    elif key == "failure_rate":
                        dict[key].append(float(value.strip("%")))
                    else:
                        dict[key].append(value)
    labels=files
    x = np.arange(len(labels))  # the label locations
    width = 0.6  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width , dict['failed_number'], width/2, label='failed number')
    rects2 = ax.bar(x - width/2 , dict['total_number'], width/2, label='total number')
    rects3 = ax.bar(x + width/2 , dict['failure_rate'], width/2, label='failure rate')
    rects4 = ax.bar(x + width , dict['score_composed'], width/2, label='scores')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('values')
    ax.set_title('Statistic')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")  # Rotate labels for better readability
    ax.legend()
    plt.tight_layout()
    ax.tick_params(axis='x', labelsize=8)
    values_str = [str(val) for val in dict.values()]

    #plt.show()
    save_path=os.path.join(weather,"static/figures")
    os.makedirs(save_path,exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def count(file):
    with open(file,'r') as f:

        cnt=0
        lines = f.readlines()
        for line in lines:
            if "weather-2" in line:
                cnt+=1
        print(cnt)
if __name__=="__main__":
    weathers=["weather-0","weather-1","weather-2","weather-3"]
    #analyse_data(weathers)
    for weather in weathers:
        create_plot(os.path.join(weather, "static/json"), weather)
    #dataset_index(weathers)
    #count("dataset_index.txt")