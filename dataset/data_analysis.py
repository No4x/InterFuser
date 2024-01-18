import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
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

def resize_image(args):
    input_path, output_path, target_size = args
    try:
        # Open the image file
        with Image.open(input_path) as img:
            # Resize the image
            resized_img = img.resize(target_size, Image.ANTIALIAS)
            # Save the resized image
            if os.path.isfile(output_path) != True:
                resized_img.save(output_path, format='PNG')
    except Exception as e:
        print(f"Error processing image {input_path}: {e}")

def resize_images_in_directory(args):
    input_dir, output_dir, target_size = args
    os.makedirs(output_dir, exist_ok=True)

    args_list = []
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if os.path.isfile(input_path):
            args_list.append((input_path, output_path, target_size))

    with Pool(24) as p:
        list(tqdm(p.imap(resize_image, args_list), desc=output_path,total=len(args_list)))
def copy_directory(args):
    source_dir, destination_dir=args
    try:
        # 使用 shutil.copytree 复制整个目录
        shutil.copytree(source_dir, destination_dir)
        print(f"sucess：{source_dir} to {destination_dir}")
    except Exception as e:
        print(f"falied：{e}")

def find_lack_files():
    with open("dataset_index1.txt","r") as rf:
        lines=rf.readlines()
        lines.sort()
        for line in lines:
            route=line.split(" ")[0]
            frames=int(line.split(" ")[2])
            if (os.path.exists(route) ==False)  or (len(os.listdir(os.path.join(route,"lidar")))!=frames):

                with open("lack_files.txt", "a") as f:
                    if not os.path.exists(route):
                        str=f"{route} \n"
                    else:
                        str = f"{route}  {len(os.listdir((os.path.join(route,'lidar'))))} {frames}\n"
                    f.write(str)

def caculate_aver_score(file):

    ds1=0
    ds2=0
    rc1=0
    rc2=0
    if1=0
    if2=0
    with open(file,'r') as f:
        jess_dict = json.loads(f.read())

    for i in range(len(jess_dict["_checkpoint"]["records"])):
        if i %2 ==0:
            ds1+=jess_dict["_checkpoint"]["records"][i]["scores"]["score_composed"]
            rc1+=jess_dict["_checkpoint"]["records"][i]["scores"]["score_route"]
            if1+=jess_dict["_checkpoint"]["records"][i]["scores"]["score_penalty"]
        else:
            ds2 += jess_dict["_checkpoint"]["records"][i]["scores"]["score_composed"]
            rc2 += jess_dict["_checkpoint"]["records"][i]["scores"]["score_route"]
            if2 += jess_dict["_checkpoint"]["records"][i]["scores"]["score_penalty"]


    avg_ds1=ds1/jess_dict["_checkpoint"]["progress"][0] *2
    avg_ds2=ds2/jess_dict["_checkpoint"]["progress"][0] *2
    avg_rc1 = rc1 / jess_dict["_checkpoint"]["progress"][0] * 2
    avg_rc2 = rc2 / jess_dict["_checkpoint"]["progress"][0] * 2
    avg_if1 = if1 / jess_dict["_checkpoint"]["progress"][0] * 2
    avg_if2 = if2 / jess_dict["_checkpoint"]["progress"][0] * 2

    print(f"avg_ds1: {avg_ds1}, avg_ds2: {avg_ds2},avg_rc1: {avg_rc1}, avg_rc2: {avg_rc2},avg_if1: {avg_if1}, avg_if2: {avg_if2}")

if __name__=="__main__":
    weathers=["weather-0","weather-1","weather-2","weather-3"]
    #analyse_data(weathers)
    # for weather in weathers:
        # create_plot(os.path.join(weather, "static/json"), weather)
        # dirs=os.listdir(os.path.join(weather,'data'))
        # args_list=[]
        # for dir in dirs:
        #     path=os.path.join(weather,'data',dir)
    #find_lack_files()
            #files=os.listdir(path)
        #     source_dirs=path
        #     destination_dirs=f"../dataset_re/{path}"
        #     args_list.append((source_dirs, destination_dirs))
        #with Pool(24) as pool:
            #list(tqdm(pool.imap(copy_directory,args_list),desc=destination_dirs, total=len(args_list)))
            # resize_images_in_directory((os.path.join(path, 'rgb_tf'), os.path.join(path, 'rgb_tf_resized'), (960, 160)))


            # files=os.listdir(os.path.join(path,'rgb_tf'))
            # for file in files:
            #     with Pool(24) as p:
            #         input=os.path.join(path,'rgb_tf',file)
            #         output=os.path.join(path,f'rgb_tf_resized/{file}')
            #         target_size=(960,160)
            #         r = list(tqdm(p.map(resize_image, ((input,output,target_size))), total=len(files)))
                #resize_image(os.path.join(path,'rgb_tf',file),os.path.join(path,f'rgb_tf_resized/{file}'),(960,160))
    #dataset_index(weathers)
    #count("dataset_index.txt")
    caculate_aver_score('results/interfuser_result1.json')