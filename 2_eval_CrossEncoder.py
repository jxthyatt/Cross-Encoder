import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import cv2
import gc
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import numpy as np
from models import CrossEncoder
from models.regressor import regressor
from PIL import Image
import matplotlib.pyplot as plt
import os
from dataloaders.data_loader_FreeGaze2 import ImagerLoader
from sklearn import metrics

growth_rate=32
z_dim_app=128
z_dim_gaze=64
decoder_input_c=32

network = CrossEncoder(
    growth_rate=growth_rate,
    z_dim_app=z_dim_app,
    z_dim_gaze=z_dim_gaze,
    decoder_input_c=decoder_input_c,
)
pretrain_dict = torch.load('/mnt/traffic/home/jxt/3ndPaper-Allworks/1-ProposedWork/Cross-Encoder/backup/exp7-resnet18/checkpoints/at_step_0372599.pth.tar')
network.load_state_dict(pretrain_dict)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def send_data_dict_to_gpu(data):
    for k in data:
        v = data[k]
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data

def eye_loader(path):
    try:
        im = Image.open(path)
        return im
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")

def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output,dim=1)
        q_output = F.softmax(q_output,dim=1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

network = network.to(device)
network.eval()

data_root = '/mnt/traffic/home/jxt/3ndPaper-Allworks/3-Datasets/test_dataset/'
batch_size=1


testset = ImagerLoader(data_root,transforms.Compose([
                    transforms.Resize((128,64)),transforms.ToTensor(),#image_normalize,
                    ]))
test = torch.utils.data.DataLoader(testset,
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

print(len(testset))
#data_iterator = iter(test)




list_gsim_g = []
list_esim_g = []
list_esim_e = []
list_gsim_e = []

list_ed1 = []
list_ed2 = []
list_ed3 = []
list_ed4 = []

list_js1 = []
list_js2 = []
list_js3 = []
list_js4 = []

list_kl1 = []
list_kl2 = []
list_kl3 = []
list_kl4 = []





#for i in range(len(test)):

    #current_test = next(data_iterator)



with open('/mnt/traffic/home/jxt/3ndPaper-Allworks/1-ProposedWork/Cross-Encoder/testlog/exp7-resnet18.txt', 'a') as f:

    for input_dict in test:

        f.write(str(input_dict['name_1_l']))
        f.write('\n')
        f.write(str(input_dict['name_2_l']))
        f.write('\n')


        input_dict = send_data_dict_to_gpu(input_dict)
        output_dict = network(input_dict)

        '''
        grep_1_l = output_dict['z_gaze_enc_1_l'].view(-1,z_dim_gaze)
        grep_1_r = output_dict['z_gaze_enc_1_r'].view(-1,z_dim_gaze)
        grep_2_l = output_dict['z_gaze_enc_2_l'].view(-1,z_dim_gaze)
        grep_2_r = output_dict['z_gaze_enc_2_r'].view(-1,z_dim_gaze)
        '''

        grep_1_l = output_dict['z_gaze_enc_1_l'].view(-1,3*z_dim_gaze)
        grep_1_r = output_dict['z_gaze_enc_1_r'].view(-1,3*z_dim_gaze)
        grep_2_l = output_dict['z_gaze_enc_2_l'].view(-1,3*z_dim_gaze)
        grep_2_r = output_dict['z_gaze_enc_2_r'].view(-1,3*z_dim_gaze)


        erep_1_l = output_dict['z_app_1_l']
        erep_1_r = output_dict['z_app_1_r']
        erep_2_l = output_dict['z_app_2_l']
        erep_2_r = output_dict['z_app_2_r']

        arep_1_l = torch.cat((grep_1_l,erep_1_l),dim=1)
        arep_1_r = torch.cat((grep_1_r,erep_1_r),dim=1)
        arep_2_l = torch.cat((grep_2_l,erep_2_l),dim=1)
        arep_2_r = torch.cat((grep_2_r,erep_2_r),dim=1)

        gsim_g = nn.CosineSimilarity()(grep_1_l,grep_1_r).sum().item()+nn.CosineSimilarity()(grep_2_l,grep_2_r).sum().item()
        esim_g = nn.CosineSimilarity()(erep_1_l,erep_1_r).sum().item()+nn.CosineSimilarity()(erep_2_l,erep_2_r).sum().item()
        esim_e = nn.CosineSimilarity()(erep_1_l,erep_2_l).sum().item()+nn.CosineSimilarity()(erep_1_r,erep_2_r).sum().item()
        gsim_e = nn.CosineSimilarity()(grep_1_l,grep_2_l).sum().item()+nn.CosineSimilarity()(grep_1_r,grep_2_r).sum().item()

        asim_g = nn.CosineSimilarity()(arep_1_l,arep_1_r).sum().item()+nn.CosineSimilarity()(arep_2_l,arep_2_r).sum().item()
        asim_e = nn.CosineSimilarity()(arep_1_l,arep_2_l).sum().item()+nn.CosineSimilarity()(arep_1_r,arep_2_r).sum().item()

        #print('Batch ' + str(i) + ':  ')
        f.write('DiffID_bgdim: {0:.4f}, DiffID_crackdim: {1:.4f}, SameID_crackdim: {2:.4f}, SameID_bgdim: {3:.4f}'.format(gsim_g/2,esim_g/2,esim_e/2,gsim_e/2))
        f.write('\n')
        f.write('DiffID_alldim: {0:.4f}, SameID_alldim: {1:.4f}'.format(asim_g/2,asim_e/2))
        f.write('\n')




        ed1 = F.pairwise_distance(grep_1_l, grep_1_r, keepdim=True).item() + F.pairwise_distance(grep_2_l, grep_2_r, keepdim=True).item()
        f.write("DiffID_bg ED: {:.4f}".format(ed1 / 2))
        f.write('\n')
        ed2 = F.pairwise_distance(erep_1_l, erep_1_r, keepdim=True).item() + F.pairwise_distance(erep_2_l, erep_2_r, keepdim=True).item()
        f.write("DiffID_cr ED: {:.4f}".format(ed2 / 2))
        f.write('\n')
        ed3 = F.pairwise_distance(erep_1_l, erep_2_l, keepdim=True).item() + F.pairwise_distance(erep_1_r, erep_2_r, keepdim=True).item()
        f.write("SameID_cr ED: {:.4f}".format(ed3 / 2))
        f.write('\n')
        ed4 = F.pairwise_distance(grep_1_l, grep_2_l, keepdim=True).item() + F.pairwise_distance(grep_1_r, grep_2_r, keepdim=True).item()
        f.write("SameID_bg ED: {:.4f}".format(ed4 / 2))
        f.write('\n')



        js1 = js_div(grep_1_l,grep_1_r).item() + js_div(grep_2_l,grep_2_r).item()
        f.write("DiffID_bg JS: {:.8f}".format(js1))
        f.write('\n')
        js2 = js_div(erep_1_l,erep_1_r).item() + js_div(erep_2_l,erep_2_r).item()
        f.write("DiffID_cr JS: {:.8f}".format(js2))
        f.write('\n')
        js3 = js_div(erep_1_l,erep_2_l).item() + js_div(erep_1_r,erep_2_r).item()
        f.write("SameID_cr JS: {:.8f}".format(js3))
        f.write('\n')
        js4 = js_div(grep_1_l,grep_2_l).item() + js_div(grep_1_r,grep_2_r).item()
        f.write("SameID_bg JS: {:.8f}".format(js4))
        f.write('\n')



        kl1 = nn.KLDivLoss(reduction='sum')(F.log_softmax(grep_1_l,dim=1), F.softmax(grep_1_r,dim=1)).item() + nn.KLDivLoss(reduction='sum')(F.log_softmax(grep_2_l,dim=1), F.softmax(grep_2_r,dim=1)).item()
        f.write("DiffID_bg KL: {:.8f}".format(kl1))
        f.write('\n')
        kl2 = nn.KLDivLoss(reduction='sum')(F.log_softmax(erep_1_l,dim=1), F.softmax(erep_1_r,dim=1)).item() + nn.KLDivLoss(reduction='sum')(F.log_softmax(erep_2_l,dim=1), F.softmax(erep_2_r,dim=1)).item()
        f.write("DiffID_cr KL: {:.8f}".format(kl2))
        f.write('\n')
        kl3 = nn.KLDivLoss(reduction='sum')(F.log_softmax(erep_1_l,dim=1), F.softmax(erep_2_l,dim=1)).item() + nn.KLDivLoss(reduction='sum')(F.log_softmax(erep_1_r,dim=1), F.softmax(erep_2_r,dim=1)).item()
        f.write("SameID_cr KL: {:.8f}".format(kl3))
        f.write('\n')
        kl4 = nn.KLDivLoss(reduction='sum')(F.log_softmax(grep_1_l,dim=1), F.softmax(grep_2_l,dim=1)).item() + nn.KLDivLoss(reduction='sum')(F.log_softmax(grep_1_r,dim=1), F.softmax(grep_2_r,dim=1)).item()
        f.write("SameID_bg KL: {:.8f}".format(kl4))
        f.write('\n')



        list_gsim_g.append(gsim_g/2)
        list_esim_g.append(esim_g/2)
        list_esim_e.append(esim_e/2)
        list_gsim_e.append(gsim_e/2)


        list_ed1.append(ed1/2)
        list_ed2.append(ed2/2)
        list_ed3.append(ed3/2)
        list_ed4.append(ed4/2)

        list_js1.append(js1)
        list_js2.append(js2)
        list_js3.append(js3)
        list_js4.append(js4)

        list_kl1.append(kl1)
        list_kl2.append(kl2)
        list_kl3.append(kl3)
        list_kl4.append(kl4)





    f.write('====================== Summary ======================\n')
    f.write('====================== Summary ======================\n')
    f.write('===================== DiffCrack =====================\n')
    f.write('======================== BG =========================\n')
    f.write('====================== Summary ======================\n')
    f.write('====================== Summary ======================\n')


    f.write('Cosine Similarity ==>\n')
    mean_gsim_g = np.mean(list_gsim_g)
    std_gsim_g = np.std(list_gsim_g,ddof=1)
    median_gsim_g = np.median(list_gsim_g)
    f.write("DiffID_bg_CosineMean: %f" % mean_gsim_g)
    f.write('\n')
    f.write("DiffID_bg_CosineStd: %f" % std_gsim_g)
    f.write('\n')
    f.write("DiffID_bg_CosineMedian: %f" % median_gsim_g)
    f.write('\n')

    f.write('Eu Distance ==>\n')
    mean_ed1 = np.mean(list_ed1)
    std_ed1 = np.std(list_ed1,ddof=1)
    median_ed1 = np.median(list_ed1)
    f.write("DiffID_bg_DistanceMean: %f" % mean_ed1)
    f.write('\n')
    f.write("DiffID_bg_DistanceStd: %f" % std_ed1)
    f.write('\n')
    f.write("DiffID_bg_DistanceMedian: %f" % median_ed1)
    f.write('\n')

    f.write('JS Divergence ==>\n')
    mean_js1 = np.mean(list_js1)
    std_js1 = np.std(list_js1,ddof=1)
    median_js1 = np.median(list_js1)
    f.write("DiffID_bg_JSMean: %f" % mean_js1)
    f.write('\n')
    f.write("DiffID_bg_JSStd: %f" % std_js1)
    f.write('\n')
    f.write("DiffID_bg_JSMedian: %f" % median_js1)
    f.write('\n')

    f.write('KL Divergence ==>\n')
    mean_kl1 = np.mean(list_kl1)
    std_kl1 = np.std(list_kl1,ddof=1)
    median_kl1 = np.median(list_kl1)
    f.write("DiffID_bg_KLMean: %f" % mean_kl1)
    f.write('\n')
    f.write("DiffID_bg_KLStd: %f" % std_kl1)
    f.write('\n')
    f.write("DiffID_bg_KLMedian: %f" % median_kl1)
    f.write('\n')




    f.write('====================== Summary ======================\n')
    f.write('====================== Summary ======================\n')
    f.write('===================== DiffCrack =====================\n')
    f.write('======================== Cr =========================\n')
    f.write('====================== Summary ======================\n')
    f.write('====================== Summary ======================\n')


    f.write('Cosine Similarity ==>\n')
    mean_esim_g = np.mean(list_esim_g)
    std_esim_g = np.std(list_esim_g,ddof=1)
    median_esim_g = np.median(list_esim_g)
    f.write("DiffID_cr_CosineMean: %f" % mean_esim_g)
    f.write('\n')
    f.write("DiffID_cr_CosineStd: %f" % std_esim_g)
    f.write('\n')
    f.write("DiffID_cr_CosineMedian: %f" % median_esim_g)
    f.write('\n')

    f.write('Eu Distance ==>\n')
    mean_ed2 = np.mean(list_ed2)
    std_ed2 = np.std(list_ed2,ddof=1)
    median_ed2 = np.median(list_ed2)
    f.write("DiffID_cr_DistanceMean: %f" % mean_ed2)
    f.write('\n')
    f.write("DiffID_cr_DistanceStd: %f" % std_ed2)
    f.write('\n')
    f.write("DiffID_cr_DistanceMedian: %f" % median_ed2)
    f.write('\n')

    f.write('JS Divergence ==>\n')
    mean_js2 = np.mean(list_js2)
    std_js2 = np.std(list_js2,ddof=1)
    median_js2 = np.median(list_js2)
    f.write("DiffID_cr_JSMean: %f" % mean_js2)
    f.write('\n')
    f.write("DiffID_cr_JSStd: %f" % std_js2)
    f.write('\n')
    f.write("DiffID_cr_JSMedian: %f" % median_js2)
    f.write('\n')

    f.write('KL Divergence ==>\n')
    mean_kl2 = np.mean(list_kl2)
    std_kl2 = np.std(list_kl2,ddof=1)
    median_kl2 = np.median(list_kl2)
    f.write("DiffID_cr_KLMean: %f" % mean_kl2)
    f.write('\n')
    f.write("DiffID_cr_KLStd: %f" % std_kl2)
    f.write('\n')
    f.write("DiffID_cr_KLMedian: %f" % median_kl2)
    f.write('\n')







    f.write('====================== Summary ======================\n')
    f.write('====================== Summary ======================\n')
    f.write('===================== SameCrack =====================\n')
    f.write('======================== Cr =========================\n')
    f.write('====================== Summary ======================\n')
    f.write('====================== Summary ======================\n')


    f.write('Cosine Similarity ==>\n')
    mean_esim_e = np.mean(list_esim_e)
    std_esim_e = np.std(list_esim_e,ddof=1)
    median_esim_e = np.median(list_esim_e)
    f.write("SameID_cr_CosineMean: %f" % mean_esim_e)
    f.write('\n')
    f.write("SameID_cr_CosineMean: %f" % std_esim_e)
    f.write('\n')
    f.write("SameID_cr_CosineMean: %f" % median_esim_e)
    f.write('\n')

    f.write('Eu Distance ==>\n')
    mean_ed3 = np.mean(list_ed3)
    std_ed3 = np.std(list_ed3,ddof=1)
    median_ed3 = np.median(list_ed3)
    f.write("SameID_cr_DistanceMean: %f" % mean_ed3)
    f.write('\n')
    f.write("SameID_cr_DistanceStd: %f" % std_ed3)
    f.write('\n')
    f.write("SameID_cr_DistanceMedian: %f" % median_ed3)
    f.write('\n')

    f.write('JS Divergence ==>\n')
    mean_js3 = np.mean(list_js3)
    std_js3 = np.std(list_js3,ddof=1)
    median_js3 = np.median(list_js3)
    f.write("SameID_cr_JSMean: %f" % mean_js3)
    f.write('\n')
    f.write("SameID_cr_JSStd: %f" % std_js3)
    f.write('\n')
    f.write("SameID_cr_JSMedian: %f" % median_js3)
    f.write('\n')

    f.write('KL Divergence ==>\n')
    mean_kl3 = np.mean(list_kl3)
    std_kl3 = np.std(list_kl3,ddof=1)
    median_kl3 = np.median(list_kl3)
    f.write("SameID_cr_KLMean: %f" % mean_kl3)
    f.write('\n')
    f.write("SameID_cr_KLStd: %f" % std_kl3)
    f.write('\n')
    f.write("SameID_cr_KLMedian: %f" % median_kl3)
    f.write('\n')





    f.write('====================== Summary ======================\n')
    f.write('====================== Summary ======================\n')
    f.write('===================== SameCrack =====================\n')
    f.write('======================== Bg =========================\n')
    f.write('====================== Summary ======================\n')
    f.write('====================== Summary ======================\n')


    f.write('Cosine Similarity ==>\n')
    mean_gsim_e = np.mean(list_gsim_e)
    std_gsim_e = np.std(list_gsim_e,ddof=1)
    median_gsim_e = np.median(list_gsim_e)
    f.write("SameID_bg_CosineMean: %f" % mean_gsim_e)
    f.write('\n')
    f.write("SameID_bg_CosineMean: %f" % std_gsim_e)
    f.write('\n')
    f.write("SameID_bg_CosineMedian: %f" % median_gsim_e)
    f.write('\n')

    f.write('Eu Distance ==>\n')
    mean_ed4 = np.mean(list_ed4)
    std_ed4 = np.std(list_ed4,ddof=1)
    median_ed4 = np.median(list_ed4)
    f.write("SameID_bg_DistanceMean: %f" % mean_ed4)
    f.write('\n')
    f.write("SameID_bg_DistanceStd: %f" % std_ed4)
    f.write('\n')
    f.write("SameID_bg_DistanceMedian: %f" % median_ed4)
    f.write('\n')

    f.write('JS Divergence ==>\n')
    mean_js4 = np.mean(list_js4)
    std_js4 = np.std(list_js4,ddof=1)
    median_js4 = np.median(list_js4)
    f.write("SameID_bg_JSMean: %f" % mean_js4)
    f.write('\n')
    f.write("SameID_bg_JSStd: %f" % std_js4)
    f.write('\n')
    f.write("SameID_bg_JSMedian: %f" % median_js4)
    f.write('\n')

    f.write('KL Divergence ==>\n')
    mean_kl4 = np.mean(list_kl4)
    std_kl4 = np.std(list_kl4,ddof=1)
    median_kl4 = np.median(list_kl4)
    f.write("SameID_bg_KLMean: %f" % mean_kl4)
    f.write('\n')
    f.write("SameID_bg_KLStd: %f" % std_kl4)
    f.write('\n')
    f.write("SameID_bg_KLMedian: %f" % median_kl4)
    f.write('\n')

