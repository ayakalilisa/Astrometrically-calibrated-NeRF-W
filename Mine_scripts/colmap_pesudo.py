import os
import json
import pandas as pd

def load_json(json_path):
    for file in os.listdir(json_path):
        if file.endswith(".json") and 'merged' in file:
            merged_path = os.path.join(json_path, file)
    with open(merged_path, 'r') as f:
        data = json.load(f)
    return data

def json2tsv(json_path, output_dir):
    data = load_json(json_path)
    records = []
    for i, frame in enumerate(data["frames"]):
        filename = os.path.basename(frame["file_path"]) + ".png"
        records.append({
            "filename": filename,
            "id": i,
            "scene": "aurora",
            "split": "train"  # or val/test if exist
        })

    df = pd.DataFrame(records)
    save_path = os.path.join(output_dir, "dataset_train.tsv")
    df.to_csv(save_path, sep="\t", index=False)
    print(f"Created pseudo Phototourism .tsv at {save_path}")

def json2camerabin(json_path, output_dir):
    data = load_json(json_path)
    for k in data.items():
        keys = k[1]
        for k in keys:
            print(k)



'''
def json2imagebin(json_path, output_dir):
    data = load_json(json_path)

def json2points3D(json_path, output_dir):
    data = load_json(json_path)
'''
if __name__ == "__main__":
    json_dir = r"/Star_calibration/s2p_files/jsons"
    output_dir = r"/Star_calibration/s2p_files/jsons"

    #json2tsv(json_dir, output_dir)
    json2camerabin(json_dir, output_dir)
