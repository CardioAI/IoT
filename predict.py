import json
import numpy as np
import pandas as pd
import torch

from model import resnet18 as ImpedanceNet

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load image
    impedance_path = r'D:\UCLA\LiverImpedance\CNN_Impedance_non_invasive\F1024.csv'
    assert os.path.exists(impedance_path), "file: '{}' dose not exist.".format(impedance_path)
    df = pd.read_csv(impedance_path)
    impedance_np = np.array(df)
    impedance_np = impedance_np.astype('float64')

    # [N, C, H, W]
    # expand batch dimension
    impedance_np = torch.from_numpy(impedance_np).unsqueeze(0)
    impedance_np = impedance_np

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = ImpedanceNet(num_classes=2).to(device).double()

    # load model weights
    weights_path = "./ImpedanceNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(impedance_np.unsqueeze(0))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    print(print_res)


if __name__ == '__main__':
    main()
