import os
import gc
import obspy
import scipy
import numpy as np
from numpy import polyfit, poly1d
import pandas as pd
import Vespa_analysis as VpaAly
import subprocess
from obspy.taup import taup_create
from obspy.taup import TauPyModel
import time
from func_timeout import func_set_timeout
import func_timeout

## 直接关闭 SettingWithCopyWarning
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#Function to draw curved line
def draw_curve(p1, p2, n=10):
    
    a = (p2[1] - p1[1])/ (np.cosh(p2[0]) - np.cosh(p1[0]))
    b = p1[1] - a * np.cosh(p1[0])
    x = np.linspace(p1[0], p2[0], n)
    y = a * np.cosh(x) + b
    
    return x, y

def interp_line(p1,p2,n):
    names = ["depth","vp","vs","density"]
    _data = []
    for i in range(1,len(names)):
        _x,_y = draw_curve([p1[i],p1[0]],
                        [p2[i],p2[0]],
                        n=n)
        if i == 1:
            _data.append(_y)
        _data.append(_x)
    return np.vstack(_data).T

def create_models(R_CMB,vp_CMB,vp_oc_ICB,R_ICB,vp_ICB_jump,
                  R_mars= 3389.5,
                  modelpath="/media/bihx/data/Mars/model/Core_models/Core_models/SKS/GD.nd",
                  mdoutdirs=None,Is_save=True,Is_nd_To_npz=True):
    """
    Inversion:
    model parameters:
    & R_CMB  : 外核大小
    & vp_CMB : cmb处的vp
    & dvp_oc : 外核P波速度的扰动
    & R_ICB  : 内核大小
    & vp_ICB_jump : icb处的vp
    """
    from obspy.taup import taup_create
    from obspy.taup import TauPyModel

    names  = ["depth","vp","vs","density"]
    MD     = pd.read_csv(modelpath,names=names,sep="\s+")
    CM_origin_index  = MD[MD["depth"] == "mantle"].index[0] + 1
    CMB_origin_index = MD[MD["depth"] == "outer-core"].index[0]
    ICB_origin_index = MD[MD["depth"] == "inner-core"].index[0]

    mantle_depth     = np.array(list(map(float,MD["depth"][CM_origin_index:CMB_origin_index].to_numpy())))
    ## ---- outer-core density ----
    x_oc_depth = np.array(list(map(float,MD["depth"][CMB_origin_index+1:ICB_origin_index].to_numpy())))
    y_oc_rho   = np.array(list(map(float,MD["density"][CMB_origin_index+1:ICB_origin_index].to_numpy())))
    oc_rho     = poly1d(polyfit(x_oc_depth,y_oc_rho,5))

    ## ------------- generate new model ---------------------------------
    new_model      = "%.2f-%.2f-%.2f-%.2f-%.3f.nd" % (R_CMB,vp_CMB,vp_oc_ICB,R_ICB,vp_ICB_jump)

    depth_CMB     = R_mars - R_CMB
    depth_ICB     = R_mars - R_ICB
    vp_CMB        = vp_CMB
    vp_oc_bottom  = vp_oc_ICB
    vp_oc_gradient = (vp_oc_ICB-vp_CMB)/(R_CMB-R_ICB)
    vp_ICB        = vp_oc_bottom * (1 + vp_ICB_jump)
    vp_ic_bottom  = vp_ICB + vp_oc_gradient * (R_ICB)

    point_CMB       = [depth_CMB,vp_CMB,0,oc_rho(depth_CMB)] 
    point_oc_bottom = [depth_ICB,vp_oc_bottom,0,oc_rho(depth_ICB)] 

    point_ICB       = [depth_ICB,vp_ICB,vp_ICB/np.sqrt(3),6.5] 
    point_ic_bottom = [R_mars,vp_ic_bottom,vp_ic_bottom/np.sqrt(3),6.5] 

    ddl_index    = np.where(mantle_depth - depth_CMB < 0)[0][-1] + CM_origin_index +1
    ## --- mantle --------
    a,b     = ddl_index-2,ddl_index
    x_depth = np.array(list(map(float,MD["depth"][a:b].to_numpy())))
    y_vp    = np.array(list(map(float,MD["vp"][a:b].to_numpy())))
    y_vs    = np.array(list(map(float,MD["vs"][a:b].to_numpy())))
    y_rho   = np.array(list(map(float,MD["density"][a:b].to_numpy())))

    c_vp    = poly1d(polyfit(x_depth,y_vp,3))
    c_vs    = poly1d(polyfit(x_depth,y_vs,3))
    c_rho   = poly1d(polyfit(x_depth,y_rho,3))

    source_file       = open(modelpath,'rb')
    source_file_lines = source_file.readlines()

    new_file    = open(os.path.join(mdoutdirs,new_model),'ab+')
    for i in range(0,ddl_index):
        line = source_file_lines[i]
        new_file.write(line)
        new_file.flush()
    new_line = b"%.4f\t  %.4f\t  %.4f\t  %.4f\n" % (depth_CMB,c_vp(depth_CMB),c_vs(depth_CMB),c_rho(depth_CMB))
    new_file.write(new_line)
    new_file.flush()

    ## ---- outer-core ------------------------
    new_line = b"outer-core\n"
    new_file.write(new_line)
    new_file.flush()

    data = interp_line(p1=point_CMB,p2=point_oc_bottom,n=15)
    for i in range(len(data)):
        new_line = b"%.4f\t  %.4f\t  %.4f\t  %.4f\n" % (data[i][0],data[i][1],data[i][2],data[i][3])
        new_file.write(new_line)
        new_file.flush()
    
    ## ------ inner-core -----------------------
    new_line = b"inner-core\n"
    new_file.write(new_line)
    new_file.flush()

    data = interp_line(p1=point_ICB,p2=point_ic_bottom,n=5)
    for i in range(len(data)):
        new_line = b"%.4f\t  %.4f\t  %.4f\t  %.4f\n" % (data[i][0],data[i][1],data[i][2],data[i][3])
        new_file.write(new_line)
        new_file.flush()
    
    source_file.close()
    new_file.close()

    if Is_nd_To_npz:
        taup_create.build_taup_model(filename=mdoutdirs + new_model,
                                     output_folder=mdoutdirs)
    model = TauPyModel(model=mdoutdirs + "%.2f-%.2f-%.2f-%.2f-%.3f.npz" % (R_CMB,vp_CMB,vp_oc_ICB,R_ICB,vp_ICB_jump))
    return model

def get_observed_data(phase):
    """
    aligned ~ P
    """
    excelpath    = "/media/bihx/data/Mars_Four/Figure/VesPa/Observed/result/Phase_Vespa_Sample_Uncertainty.xlsx"
    phase_info   = pd.read_excel(excelpath,sheet_name="v1")

    # vsp_mean = phase_info[phase_info["phase"] == phase]["vsp_mean"].values[0]
    vsp_mean = phase_info[phase_info["phase"] == phase]["vsp_mode"].values[0]
    vsp_std  = phase_info[phase_info["phase"] == phase]["vsp_std"].values[0]

    return vsp_mean,vsp_std

def prior(w):
    #x[0] = mu, x[1]=sigma (new or current)
    #returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    #returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    #It makes the new sigma infinitely unlikely.

    ## Vp > Vs 

    # if not (1780<w[0]<1810 and 4.9<w[1]<5.0 and 0.0001<w[2]<0.002 and 0.5<w[3]<750.5 and 0.01<w[4]<0.60):
    #     return 0
    # else:
    #     return 1
    if not (1700<w[0]<1900 and 4.6<w[1]<5.5 and 5.0<w[2]<6.5 and 0.5<w[3]<750.5 and 0.01<w[4]<0.60):
        return 0
    else:
        return 1

#Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept=np.random.uniform(0,1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return (accept < (x_new/x))
    

# def cal_taup(phases,model):

#     deg_average = 29.0
#     misfit = np.zeros(shape=len(phases))
#     for j,phase in enumerate(phases):
#         data,data_std   = get_observed_data(phase=phase)

#         if phase == "PKPPKPr":
#             ph = "PKPPKP"
#             phase_list = ["P",ph]
#             TaupInfo   = model.get_travel_times(source_depth_in_km=33,distance_in_degree=24,phase_list=phase_list)
#             if len(TaupInfo)<3:
#                 data_syn  = np.array([np.inf])
#             else:
#                 t0   = TaupInfo[-2].time
#                 s0   = TaupInfo[-2].ray_param_sec_degree
    
#                 TaupInfo   = model.get_travel_times(source_depth_in_km=33,distance_in_degree=deg_average,phase_list=["P"])
#                 data_syn   = VpaAly.cal_vespa_arrt(deg_average=24,s0=-s0,t0=t0,deg=deg_average) - TaupInfo[0].time

#         elif phase == "PKPPKPn":
#             ph = "PKPPKP"
#             phase_list = ["P",ph]
#             TaupInfo   = model.get_travel_times(source_depth_in_km=33,distance_in_degree=deg_average,phase_list=phase_list)
#             if len(TaupInfo)<2:
#                 data_syn   = np.array([np.inf])
#             else:
#                 data_syn   = TaupInfo[-1].time - TaupInfo[0].time

#         elif phase == "PKiKP" or phase== "PKIKKIKP":
#             phase_list = ["P",phase]
#             TaupInfo   = model.get_travel_times(source_depth_in_km=33,distance_in_degree=deg_average,phase_list=phase_list)
#             if len(TaupInfo)<2:
#                 continue
#             else:
#                 data_syn   = TaupInfo[1].time - TaupInfo[0].time

#         misfit[j] = abs(data - data_syn) / data_std
#     x_lik = np.exp(-1 * np.mean(misfit))
    
#     return x_lik

def cal_taup(phases,model):

    deg_average = 29.0
    misfit      = np.zeros(shape=len(phases))
    obs_d       = np.zeros(shape=len(phases))
    obs_d_std   = np.zeros(shape=len(phases))
    for j,phase in enumerate(phases):
        data,data_std   = get_observed_data(phase=phase)
        obs_d[j] = data
        obs_d_std[j] = data_std

    ## ----- calcucate theoretical arrivals -------------------------
    syn_d       = np.zeros(shape=len(phases))
    phase_list= ["P","PKiKP","PKIKKIKP","PKPPKP"]
    TaupInfo  = model.get_travel_times(source_depth_in_km=33,
                                       distance_in_degree=deg_average,
                                       phase_list=phase_list)
    if not (TaupInfo[1].name=="PKiKP" and TaupInfo[2].name=="PKIKKIKP" and TaupInfo[-1].name=="PKPPKP"):
        syn_d = np.array([np.inf, np.inf, np.inf, np.inf])
    else:
        ## PKiKP 
        syn_d[0]  = TaupInfo[1].time - TaupInfo[0].time
        ## PKKPdf
        syn_d[1]  = TaupInfo[2].time - TaupInfo[0].time
        ## PKPPKPn
        syn_d[3]  = TaupInfo[-1].time - TaupInfo[0].time

        ## PKPPKPr
        temp = model.get_travel_times(source_depth_in_km=33,
                                      distance_in_degree=20,
                                      phase_list=["PKPPKP"])
        if len(temp)<2:
            syn_d[2] = np.array([np.inf])
        else:
            t0   = temp[-2].time
            s0   = temp[-2].ray_param_sec_degree
            syn_d[2] = VpaAly.cal_vespa_arrt(deg_average=20,s0=-s0,t0=t0,deg=deg_average) - TaupInfo[0].time

    misfit = abs(obs_d - syn_d) / obs_d_std
    x_lik  = np.exp(-1 * np.mean(misfit))
    
    return x_lik

### -------------------------- main function ----------------------------------------------

transition_model = lambda x: np.random.normal(x,[80, 0.3, 0.3,250, 0.25],(5,))


label       = "SKS_GD_m2"

param_init  = [1600.0, 5.0, 6.0, 50.0, 0.10]
acceptance_rule     = acceptance
phases      = ["PKiKP","PKIKKIKP","PKPPKPr","PKPPKPn",]
mdoutdirs   = "/media/bihx/data/Mars_Four/Figure/mcmc/MCMC_SKS/models_{}/".format(label)
if not os.path.exists(mdoutdirs):
    os.makedirs(mdoutdirs)
resultpath  = "/media/bihx/data/Mars_Four/Figure/mcmc/MCMC_SKS/result/{}/".format(label)
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

x        = param_init
accepted = []
rejected = []   
# initial model and parameters
R_CMB,vp_CMB,vp_oc_ICB,R_ICB,vp_ICB_jump = x
model = create_models(R_CMB,vp_CMB,vp_oc_ICB,R_ICB,vp_ICB_jump,
                    mdoutdirs=mdoutdirs,Is_save=True,Is_nd_To_npz=True)
x_lik = cal_taup(phases,model)
j = 0
z = 0
while True:
    x_new  = transition_model(x)
    # if not (1500<x_new[0]<2300 and 4.5<x_new[1]<6.0 and 0.5<x_new[2]<750.5 and 6.0<x_new[3]<9.0 \
    #         and x_new[3]> (x_new[1]+0.00062 * (x_new[0] - x_new[2])) ):
    #     continue

    if not (1700<x_new[0]<1900 and 4.6<x_new[1]<5.5 and 5.0<x_new[2]<6.5 and 0.5<x_new[3]<750.5 and 0.01<x_new[4]<0.60 \
        and x_new[2]>x_new[1]):

        continue
    j=j+1

    R_CMB,vp_CMB,vp_oc_ICB,R_ICB,vp_ICB_jump = x_new
    try:
        model_new = create_models(R_CMB,vp_CMB,vp_oc_ICB,R_ICB,vp_ICB_jump,
                                mdoutdirs=mdoutdirs,Is_save=True,Is_nd_To_npz=True)
    except func_timeout.exceptions.FunctionTimedOut as e:
        print("{}:Time out!!!".format(x_new))
        continue
    except Exception:
        print("Model has wrong!")
        continue
    
    x_new_lik = cal_taup(phases,model_new)
    if (acceptance_rule(x_lik * 1,x_new_lik * 1)):   
        z     = z + 1    
        x     = x_new
        x_lik = x_new_lik
        accepted.append([j,x_new,x_new_lik])
    else:
        rejected.append([j,x_new,x_new_lik])  

    print("iterations={}".format(j))
    print("accepted={}".format(z))

    if (j!=0) and (j%10000 == 0):
        np.save(file=resultpath + "rejected_{}_{}.npy".format(j,label),arr=np.array(rejected))
        np.save(file=resultpath + "accepted_{}_{}.npy".format(j,label),arr=np.array(accepted))

        commd  =  "rm -rf {}*.npz".format(mdoutdirs)
        P      = subprocess.check_output(commd,shell=True)

    if z==1000:
        np.save(file=resultpath + "rejected_{}_{}.npy".format(z,label),arr=np.array(rejected))
        np.save(file=resultpath + "accepted_{}_{}.npy".format(z,label),arr=np.array(accepted))

        commd  =  "rm -rf {}*.npz".format(mdoutdirs)
        P      = subprocess.check_output(commd,shell=True)	
        break
