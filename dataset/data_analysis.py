import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
def analyse_data(weathers):

    for weather in weathers:
        os.makedirs(os.path.join(weather,"static/jsons"),exist_ok=True)
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
                    with open(os.path.join(weather,"static/jsons", file), 'w') as write_file:
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
    labels=[]
    files=os.listdir(path)
    files.sort()
    for file in files:
        with open(os.path.join(path,file),'r') as read_file:
            data = json.load(read_file)
            for key,value in data["meta"].items():
                if key != "exceptions" :
                    if key == "scores":
                        dict["score_composed"].append(value['score_composed'])
                    elif key == "failure_rate":
                        dict[key].append(float(value.strip("%")))
                    else:
                        dict[key].append(value)
        label=file.split('.')[0].split('_')[1]+' '+file.split('.')[0].split('_')[2]
        labels.append(label)
    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars


    fig, (ax,ax1) = plt.subplots(2,1,'col',figsize=(12, 10))
    ax2 = ax1.twinx()
    rects1 = ax.bar(x - width/2 , dict['failed_number'], width, label='failed number',color='darkorange')
    rects2 = ax.bar(x + width/2 , dict['total_number'], width, label='total number',color='dodgerblue')
    # rects3 = ax.bar(x + width/2 , dict['failure_rate'], width/2, label='failure rate')
    # rects4 = ax.bar(x + width , dict['score_composed'], width/2, label='scores')
    line1 = ax2.plot(x, dict['failure_rate'], color='tomato', marker='o', label='Failure Rate',linestyle='dashed')
    rects3 = ax1.bar(x, dict['score_composed'], width, label='scores',color='lightgreen')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('values')
    ax2.set_ylabel('Failure Rate (%)', color='red')
    ax1.set_ylabel('scores')
    ax.set_title('Statistic')
    ax.set_xticks(x)
    #ax.set_xticklabels(labels, rotation=45, ha="right")  # Rotate labels for better readability
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.1, 0),loc='upper right')
    ax2.legend(bbox_to_anchor=(1.1, -0.09),loc='upper right')
    ax1.legend(bbox_to_anchor=(1.1, 0),loc='upper right')
    for rect, value in zip(rects1 + rects2, dict['failed_number'] + dict['total_number']):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value}', ha='center', va='bottom')
    for i, rect in enumerate(rects3):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2., height, f'{dict["score_composed"][i]:.2f}', ha='center',
                 va='bottom')

    for i, txt in enumerate(dict['failure_rate']):
        ax2.annotate(f'{txt:.2f}', (x[i], txt), textcoords="offset points", xytext=(-8, 0), ha='left',va='top')

    #plt.tight_layout()
    ax.tick_params(axis='x', labelsize=8)
    avg_failure_rate = np.mean(dict['failure_rate'])
    avg_score_composed = np.mean(dict['score_composed'])

    # Add subplots for average values
    ax1.text(0, 110, f'Average Failure Rate: {avg_failure_rate:.2f}%', ha='left', va='center', fontsize=12,
            color='red')
    ax1.text(0, 115, f'Average Score: {avg_score_composed:.2f}', ha='left', va='center', fontsize=12,
            color='black')

    #plt.show()
    save_path=os.path.join(weather,f"static/{weather}.png")
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
    analyse_data(weathers)
    for weather in weathers:
        create_plot(os.path.join(weather, "static/json"), weather)
    #dataset_index(weathers)
    #count("dataset_index.txt")