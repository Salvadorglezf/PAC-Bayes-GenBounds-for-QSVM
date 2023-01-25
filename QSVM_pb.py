#!/usr/bin/env python
# coding: utf-8

# # PAC Bayes bounds from E. Parrado and colleges (2012)
# ## Python implementation of the matlab code by E.Parrado in matlab from (2014)

# # Comandos Utiles

import numpy as np
#from sklearn.model_selection import train_test_split
#import pandas as pd
from scipy.stats import norm
from sklearn.svm import SVC
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
#from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import random 
from scipy import interpolate


### FUNCIONES ###

##################
## Calculo svm ###
##################

def svm_func(x,x_train,y_train,x_test,y_test):
    rbf_svc = SVC(kernel="precomputed")   #tomamos una funcion kernel precomputada
    gram_train = adhoc_kernel.evaluate(x_vec=x_train)
    # ## se entrena la svm con los datos de entrenamiento
    rbf_svc.fit(gram_train,y_train)    #entrenamos SVM con datos entrenamiento
     # ## Se calcula la matriz kernel para los datos de prueba
    gram_test = adhoc_kernel.evaluate(x_vec=x_test, y_vec=x_train)  #calculamos matriz kernel para datos de prueba 
    # ## Se busca el promedio del error del modelo usando crossfolding
    #print(1-cross_val_score(rbf_svc, gram_train, y_train, cv=2, scoring='recall_macro')) # valor de presicion para modelo de entrnemaiento
    mean_err=1-np.mean(cross_val_score(rbf_svc, gram_train, y_train, cv=2, scoring='recall_macro'))  
    print(mean_err)
    score=1-rbf_svc.score(gram_test, y_test)
    idx_sv=rbf_svc.support_ #indices vectores de soporte
    w=rbf_svc.dual_coef_    #vector con multiplicadores de langrange
    support = np.r_['r', idx_sv] #Hacer 'idx_sv' vector fila
    weights=w.reshape(-1,1)
    return [weights,support,mean_err,score,idx_sv,rbf_svc]


############################################
### Divergencia Kullback Leigber inversa ###
############################################

def inv_kl( qs: float, kl: float)-> float:
    qd_=np.ones(kl.shape[0])
    izq=float(0)
    dch=float(0)
    ikl=float(0)
    #print(kl.shape[0])
    for i in range(kl.shape[0]):
        izq=qs[i]
        dch = 1-1e-10
        while(((dch-izq)/dch)>=1e-5):
            p=(izq+dch)*0.5
            ikl=kl[i][0]-(qs[i]*np.log(qs[i]/p)+(1-qs[i])*np.log((1-qs[i])/(1-p)))
            if(ikl<0):
                dch=p
            else:
                izq=p
        qd_[i]=p
    return qd_
    
########################
## calculo pac_bayes ###
########################

def pac_bayes(gram_all,weights,idx_sv,y_train,x_train,delta):
    
    pred=np.dot(gram_all[:,idx_sv],weights) #denominador gamma phi(x)w
    mod=np.sqrt(np.dot(weights.T,pred[idx_sv])) #denominador gamma sqrt (w.T phi(x)w) acaba el calculo del modulo de phi(x)
    pred[:x_train.shape[0]].shape #Toma solo los datos de train de la matriz kernel por el vector de pesosshape
    Y_train = np.r_['c', y_train] #Hacer 'idx_sv' vector fila    
    num_gamma= np.multiply(Y_train,pred[:x_train.shape[0]]) #numerador de gamma
    gamma=num_gamma/mod #valor de gamma 
    ln=np.log( ( len(gamma)+1)/delta) # log(m+1)/delta
    mu=np.array([0.0001, 1, 50, 100]) # Se proponen 4 valores de mu
    mu_gamma= np.kron(mu,gamma) # producto tensorial entre cada valor de mu y cada gamma 
    F=1-norm.cdf(mu_gamma) #distribucion normal cumulativa a todas las gammas*mu
    qs=np.array([1, 2, 3, 4],float)  #valor esperado de cada Qs
    for i in range(F.shape[1]):
        qs[i]=np.mean(F[:,[i]])  
    kl=((mu**2/2)+ln)/len(gamma) # numerador lado derecho PBb
    get_indexes = np.argwhere(qs < 0.99)
    
    # ## Compute PB Bound
    qd=np.ones(kl.shape[0])
    qd=inv_kl(qs[get_indexes],kl[get_indexes])
    QD=100
    while(  ((mu[-1]-mu[0])/mu[-1]) >0.01 ):
        idx=np.argmin(qd)
        value=min(qd)
        if value<QD:
            QD=value
            mu_opt=mu[idx]
        if idx==0:
            mu=np.array([mu[0]/float(4),mu[0]/float(2),mu[0],mu[1]])
            qd=np.array([float(100),float(100),qd[0],qd[1]])
            idx_new_mu=np.array([0,1])
        elif idx==1:
            if qd[0]>qd[2]:
                mu=np.array([mu[0],(mu[0]+mu[1])/float(2),mu[1],mu[2]])
                qd=np.array([qd[0],float(100),qd[1],qd[2]])
                idx_new_mu=1
            else:
                mu=np.array([mu[0],mu[1],(mu[2]+mu[1])/float(2),mu[2]])
                qd=np.array([qd[0],qd[1],float(100),qd[2]])
                idx_new_mu=2
        elif idx==2:
            if qd[1]>qd[3]:
                mu=np.array([mu[1],(mu[1]+mu[2])/float(2),mu[2],mu[3]])
                qd=np.array([qd[1],float(100),qd[2],qd[3]])
                idx_new_mu=1
            else:
                mu=np.array([mu[1],mu[2],(mu[2]+mu[3])/float(2),mu[3]])
                qd=np.array([qd[1],qd[2],float(100),qd[3]])
                idx_new_mu=2
        elif idx==3:
            mu=np.array([mu[2],mu[3],(mu[3]*float(2)),mu[3]*float(4)])
            qd=np.array([qd[2],qd[3],float(100),float(100)])
            idx_new_mu=np.array([2,3])
        mu_gamma= np.kron(mu[idx_new_mu],gamma) # producto tensorial entre cada valor de mu y cada gamma
        #print(mu_gamma.shape)
        qs=np.ones(mu_gamma.shape[1])
        F=1-norm.cdf(mu_gamma)
        for i in range(F.shape[1]):
            qs[i]=np.mean(F[:,[i]])
        kl=(((mu[idx_new_mu]**2)/2)+ln)/len(gamma)
        kl=np.atleast_1d(kl)
        get_indexes = np.argwhere(qs < 0.99)
        #print(qs[get_indexes])
        #print(get_indexes)
        new_qd=inv_kl(qs[get_indexes],kl[get_indexes])
        qd[idx_new_mu]=new_qd
    #print("results for Subset",j)
    print("PAC Bayes Bound on the true error:",QD)
    #results[j]=QD/2
    print("mu_val:",mu_opt)
    return [QD/2,mod,gamma,kl]


##################################################################################
## Hacer nuevo experimento con capacidades optimas de las medidas de error #######
##################################################################################

def tau_pac_bayes(x_train,y_train,mod,gamma):
    
    eta=200
    tau=200
  
    r = 0.5
    order=np.arange(len(x_train))
    class0=order[0:int(len(order)/2)]
    class1=order[int(len(order)/2):len(order)]
    random.shuffle(class0)
    random.shuffle(class1)
    Nprior = round(int(len(x_train)*r)/2)
    
    idxprior = np.concatenate((class0[0:Nprior],class1[0:Nprior]))
    idxposterior = np.concatenate((class0[Nprior:len(class0)],class1[Nprior:len(class1)]))
    gamma=gamma[idxposterior]/mod
    
    gram_train_eta = adhoc_kernel.evaluate(x_vec=x_train[idxprior]) #Calculamos la matriz kernel par adatos entenamiento
    
    rbf_svc.fit(gram_train_eta,y_train[idxprior])    #entrenamos SVM con datos entrenamiento
    
    idx_sv_eta=rbf_svc.support_ #indices vectores de soporte
    w_eta=rbf_svc.dual_coef_    #vector con multiplicadores de langrange
    weights_eta=w_eta.reshape(-1,1)
    pred_eta=np.dot(gram_all[:,idx_sv_eta],weights_eta) #denominador gamma phi(x)w
     
    mod_eta=np.sqrt(np.dot(weights_eta.T,pred_eta[idx_sv_eta])) #denominador gamma sqrt (w.T phi(x)w) acaba el calculo del modulo de phi(x)
     
    wrw=np.dot(weights_eta.T,pred_eta[idx_sv_eta])/mod_eta/mod
    
    ## calcula tau pbbb
    
    tau2 = tau*tau
    mr = len(gamma) # m-r in the paper
    ln = np.log((mr+1)/delta);
    mu =np.array([.0001, 1, 50, 100]) #initial values for mu
    mu_opt=1
    
    mu_gamma= np.kron(mu,gamma) # producto tensorial entre cada valor de mu y cada gamma 
    F=1-norm.cdf(mu_gamma) #distribucion normal cumulativa a todas las gammas*mu
    qs=np.array([1, 2, 3, 4],float)  #valor esperado de cada Qs
    for i in range(F.shape[1]):
        qs[i]=np.mean(F[:,[i]])
    
    KLPQ= (np.log(tau2)+(1/tau2-1)+(mu*wrw-eta)**2/tau2 + mu**2*(1-wrw**2))
    kl= (KLPQ/2+ln)/(mr)
    kl=np.reshape(kl,(4,1))
    get_indexes = np.argwhere(qs < 0.99)
   
    # ## Compute inv kl 
    qd=np.ones(kl.shape[0])
    qd=inv_kl(qs[get_indexes],kl[get_indexes])
    QD=100
    idx_new_mu=0
    while(  ((mu[-1]-mu[0])/mu[-1]) >0.02 ):
        idx=np.argmin(qd)
        value=min(qd)
        if value<QD:
            QD=value
            mu_opt=mu[idx]
        if idx==0:
            mu=np.array([mu[0]/float(4),mu[0]/float(2),mu[0],mu[1]])
            qd=np.array([float(100),float(100),qd[0],qd[1]])
            idx_new_mu=np.array([0,1])
        elif idx==1:
            if qd[0]>qd[2]:
                mu=np.array([mu[0],(mu[0]+mu[1])/float(2),mu[1],mu[2]])
                qd=np.array([qd[0],float(100),qd[1],qd[2]])
                idx_new_mu=1
            else:
                mu=np.array([mu[0],mu[1],(mu[2]+mu[1])/float(2),mu[2]])
                qd=np.array([qd[0],qd[1],float(100),qd[2]])
                idx_new_mu=2
        elif idx==2:
            if qd[1]>qd[3]:
                mu=np.array([mu[1],(mu[1]+mu[2])/float(2),mu[2],mu[3]])
                qd=np.array([qd[1],float(100),qd[2],qd[3]])
                idx_new_mu=1
            else:
                mu=np.array([mu[1],mu[2],(mu[2]+mu[3])/float(2),mu[3]])
                qd=np.array([qd[1],qd[2],float(100),qd[3]])
                idx_new_mu=2
        elif idx==3:
            mu=np.array([mu[2],mu[3],(mu[3]*float(2)),mu[3]*float(4)])
            qd=np.array([qd[2],qd[3],float(100),float(100)])
            idx_new_mu=np.array([2,3])
        mu_gamma= np.kron(mu[idx_new_mu],gamma) # producto tensorial entre cada valor de mu y cada gamma
        #print(mu_gamma.shape)
        qs=np.ones(mu_gamma.shape[1])
        F=1-norm.cdf(mu_gamma)
        for i in range(F.shape[1]):
            qs[i]=np.mean(F[:,[i]])
            
        KLPQ=(np.log(tau2)+(1/tau2-1)+(mu[idx_new_mu]*wrw-eta)**2/tau2 + mu[idx_new_mu]**2*(1-wrw**2))
        kl= ((KLPQ/2)+ln)/mr
       # kl=np.reshape(kl,(4,1))
        kl=np.atleast_1d(kl)
        get_indexes = np.argwhere(qs < 0.99)
        #print(qs[get_indexes])
        #print(get_indexes)
        if len(kl)>1:
            new_qd=inv_kl(qs[get_indexes],kl[get_indexes])
        else:
            new_qd=inv_kl(qs[get_indexes],kl)
        qd[idx_new_mu]=new_qd
    #print("results for Subset",j)
    print("Tau PAC Bayes Bound on the true error:",QD)
    #results[j]=QD/2
    print("mu_val:",mu_opt)
    return [QD/2,kl]
     

def new_experiment(x_train_, y_train_,x_test_,y_test_,train_s,s_size,val,results,mean_err,score,tau,bound_pb,bound_tau):
    

    # x_train_, y_train_, x_test, y_test, adhoc_total = ad_hoc_data(  #crea sets de datos
    #     training_size=int(train_s/2),#int(big_t_set/2), #la funcion de qiskit tiene un error, devuelve el doble de datos de train y test
    #     test_size=int((3*s_size)), #cuatro veces mas grande que el test set anterior
    #     n=adhoc_dimension,
    #     gap=0.3,
    #     plot_data=False,
    #     one_hot=False,
    #     include_sample_total=True,
    # )
    x_test=x_test_
    y_test=y_test_

    idx_0=np.where(y_train_[:]==0) #indices clase 0
    idx_1=np.where(y_train_[:]==1) #indices clase 1
    ### Para PAC Bayes ###
    rbf_svc = SVC(kernel="precomputed")   #tomamos una funcion kernel precomputada
    ## Elijo capacidad
    modelQD_idx=np.where(results==min(results))
    opt_model_QD=np.array([val[modelQD_idx[0][0]],results[modelQD_idx[0][0]]])
    
    #segmento dataset y entreno
    
    x_train_QD=x_train_[idx_0[0][:int(((opt_model_QD[0]*train_s)/100)/2)]]
    x_train_QD=np.concatenate((x_train_QD,x_train_[idx_1[0][:int((((opt_model_QD[0]*train_s)/100))/2)]]),axis=0)
    y_train_QD=y_train_[idx_0[0][:int((((opt_model_QD[0]*train_s)/100))/2)]]
    y_train_QD=np.concatenate((y_train_QD,y_train_[idx_1[0][:int((((opt_model_QD[0]*train_s)/100))/2)]]),axis=0)
    x=np.concatenate((x_train_QD,x_test), axis=0)  
    
    gram_train_QD = adhoc_kernel.evaluate(x_vec=x_train_QD)
    gram_test_QD = adhoc_kernel.evaluate(x_vec=x_test, y_vec=x_train_QD)  #calculamos matriz kernel para datos de prueba 
     
    rbf_svc.fit(gram_train_QD,y_train_QD)    #entrenamos SVM con datos entrenamiento
    adhoc_score_callable_function_QD = rbf_svc.score(gram_test_QD, y_test)
    
    gram_train_QD=0
    gram_test_QD=0
    
    ## Elijo capacidad
    modelTau_idx=np.where(tau==min(tau))
    opt_model_Tau=np.array([val[modelTau_idx[0][0]],tau[modelTau_idx[0][0]]])
    
    #segmento dataset y entreno
    
    x_train_Tau=x_train_[idx_0[0][:int(((opt_model_Tau[0]*train_s)/100)/2)]]
    x_train_Tau=np.concatenate((x_train_Tau,x_train_[idx_1[0][:int((((opt_model_Tau[0]*train_s)/100))/2)]]),axis=0)
    y_train_Tau=y_train_[idx_0[0][:int((((opt_model_Tau[0]*train_s)/100))/2)]]
    y_train_Tau=np.concatenate((y_train_Tau,y_train_[idx_1[0][:int((((opt_model_Tau[0]*train_s)/100))/2)]]),axis=0)
    x=np.concatenate((x_train_Tau,x_test), axis=0)  
    
    gram_train_Tau = adhoc_kernel.evaluate(x_vec=x_train_Tau)
    gram_test_Tau = adhoc_kernel.evaluate(x_vec=x_test, y_vec=x_train_Tau)  #calculamos matriz kernel para datos de prueba 
     
    rbf_svc.fit(gram_train_Tau,y_train_Tau)    #entrenamos SVM con datos entrenamiento
    adhoc_score_callable_function_Tau = rbf_svc.score(gram_test_Tau, y_test)
    
    gram_train_Tau=0
    gram_test_Tau=0
    
    
    ### Para Validación cruzada
    
    # Elijo capacidad
    modelCV_idx=np.where(mean_err==min(mean_err))
    opt_model_CV=np.array([val[modelCV_idx[0][0]],mean_err[modelCV_idx[0][0]]])
    
    # Segmento dataset y entreno
    
    x_train_CV=x_train_[idx_0[0][:int((((opt_model_CV[0]*train_s)/100))/2)]]
    x_train_CV=np.concatenate((x_train_CV,x_train_[idx_1[0][:int((((opt_model_CV[0]*train_s)/100))/2)]]),axis=0)
    y_train_CV=y_train_[idx_0[0][:int((((opt_model_CV[0]*train_s)/100))/2)]]
    y_train_CV=np.concatenate((y_train_CV,y_train_[idx_1[0][:int((((opt_model_CV[0]*train_s)/100))/2)]]),axis=0)
    
    gram_train_CV = adhoc_kernel.evaluate(x_vec=x_train_CV)
    gram_test_CV = adhoc_kernel.evaluate(x_vec=x_test, y_vec=x_train_CV)  #calculamos matriz kernel para datos de prueba 
     
    rbf_svc.fit(gram_train_CV,y_train_CV)    #entrenamos SVM con datos entrenamiento
    adhoc_score_callable_function_CV = rbf_svc.score(gram_test_CV, y_test)
    
    x_train_CV=0
    y_train_CV=0
    gram_train_CV=0
    gram_test_CV=0
    
    
    # Para precisión
    modelSC_idx=np.where(score==min(score))
    opt_model_SC=np.array([val[modelSC_idx[0][0]],score[modelSC_idx[0][0]]])
    
    # Segmento dataset y entreno
    
    x_train_SC=x_train_[idx_0[0][:int((((opt_model_SC[0]*train_s)/100))/2)]]
    x_train_SC=np.concatenate((x_train_SC,x_train_[idx_1[0][:int((((opt_model_SC[0]*train_s)/100))/2)]]),axis=0)
    y_train_SC=y_train_[idx_0[0][:int((((opt_model_SC[0]*train_s)/100))/2)]]
    y_train_SC=np.concatenate((y_train_SC,y_train_[idx_1[0][:int((((opt_model_SC[0]*train_s)/100))/2)]]),axis=0)
    
    gram_train_SC = adhoc_kernel.evaluate(x_vec=x_train_SC)
    gram_test_SC = adhoc_kernel.evaluate(x_vec=x_test, y_vec=x_train_SC)  #calculamos matriz kernel para datos de prueba 
     
    rbf_svc.fit(gram_train_SC,y_train_SC)    #entrenamos SVM con datos entrenamiento
    adhoc_score_callable_function_SC =  rbf_svc.score(gram_test_SC, y_test)
    
    x_train_SC=0
    y_train_SC=0
    gram_train_SC=0
    gram_test_SC=0
    
    #Todos los datos
    
    gram_train_AG = adhoc_kernel.evaluate(x_vec=x_train_)
    gram_test_AG = adhoc_kernel.evaluate(x_vec=x_test, y_vec=x_train_)  #calculamos matriz kernel para datos de prueba 
     
    rbf_svc.fit(gram_train_AG,y_train_)    #entrenamos SVM con datos entrenamiento
    adhoc_score_callable_function_AG = rbf_svc.score(gram_test_AG, y_test)
    

    gram_train_AG=0
    gram_test_AG=0
    
    
    SCORES_=np.array([adhoc_score_callable_function_QD,adhoc_score_callable_function_CV,adhoc_score_callable_function_SC,adhoc_score_callable_function_AG,adhoc_score_callable_function_Tau])
    
    #cantidad de datos
    num_data=np.array([int((opt_model_QD[0]*train_s)/100),int((opt_model_CV[0]*train_s)/100),int((opt_model_SC[0]*train_s)/100),train_s,int((opt_model_Tau[0]*train_s)/100)])
    
    # capacidad modelo optimo
    plt.subplots(figsize=(13, 6.5))
    model=['PAC Bayes','Tau PAC Bayes', 'Validación cruzada', 'Precisión','Todos los datos']
    data = [num_data[0], num_data[4], num_data[1], num_data[2],num_data[3] ]
    plt.bar(model,data,color =['blue','m','orange','green','red'],width=0.4)
    plt.ylabel("Cantidad de datos",fontsize=23)
    plt.xlabel("Tipo de medida",fontsize=23)
    plt.show()
    #width
    
    # grafica scores en nuevos datos
    data = {'PAC Bayes':SCORES_[0], 'Tau PAC Bayes':SCORES_[4] ,'Validación cruzada':SCORES_[1], 'Precisión':SCORES_[2],
            'Todos los datos':SCORES_[3]}
    courses = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize = (13, 6.5))
    # creating the bar plot
    plt.bar(courses, values, color =['blue','m','orange','green','red'],
            width = 0.4)
    plt.ylabel("Precisión",fontsize=25)
    plt.xlabel("Tipo de medida",fontsize=25)
    plt.title("Precisión del modelo óptimo",fontsize=25)
    plt.show()
    #######
    print("\nScores nuevo experimento\n",data)
    
    # grafica scores en nuevos datos
    data = {'PAC Bayes':1-SCORES_[0], 'Tau PAC Bayes':1-SCORES_[4] ,'Validación cruzada':1-SCORES_[1], 'Precisión':1-SCORES_[2],
            'Todos los datos':1-SCORES_[3],'Cota de generalización con PAC Bayes':min(bound_pb),'Cota de generalización con tau PAC Bayes':min(bound_tau)}
    courses = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize = (25, 6.5))
    # creating the bar plot
    plt.bar(courses, values, color =['blue','m','orange','green','red'],
            width = 0.4)
    plt.ylabel("Error",fontsize=25)
    plt.xlabel("Tipo de medida",fontsize=25)
    plt.title("Cotas sobre el error de los modelos óptimos ",fontsize=25)
    plt.show()
    
    print("\nError nuevo experimento\n",data)
    

    
    
    # print("\n Plot QSVM surface\n")
    # print("   y/n?")
    # if(input()=='y'):
    #     print("\n   Ploting QSVM surface\n")
    #     print("\n   Please wait...\n")
    #PLOT QSVM
    rbf_svc = SVC(kernel=adhoc_kernel.evaluate)
    rbf_svc.fit(x_train_Tau,y_train_Tau)
    grid_step = 0.2
    margin = 0.2
    grid_x, grid_y = np.meshgrid(
        np.arange(-margin, max(x[:,1]) + margin, grid_step), np.arange(-margin,  max(x[:,1]) + margin, grid_step)
    )
    
    
    meshgrid_features = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    meshgrid_colors = rbf_svc.predict(meshgrid_features)
    
    plt.figure(figsize=(5, 5))
    meshgrid_colors = meshgrid_colors.reshape(grid_x.shape)
    plt.pcolormesh(grid_x, grid_y, meshgrid_colors, cmap="RdBu", shading="auto")
    
    # A train plot
    plt.scatter(
        x_train_Tau[:, 0][y_train_Tau == 0],
        x_train_Tau[:, 1][y_train_Tau == 0],
        marker="s",
        facecolors="w",
        edgecolors="r",
        label="Clase 0 de entrenamiento ",
    )
    
    #  B train plot
    plt.scatter(
        x_train_Tau[
            np.where(y_train_Tau[:] == 1), 0
        ],  # x coordinate of train_labels where class is 1
        x_train_Tau[
            np.where(y_train_Tau[:] == 1), 1
        ],  # y coordinate of train_labels where class is 1
        marker="o",
        facecolors="w",
        edgecolors="b",
        label="Clase 1 de entrenamiento",
    )
    
    # A test plot
    plt.scatter(
        x_test[np.where(y_test[:] == 0), 0],  # x coordinate of test_labels where class is 0
        x_test[np.where(y_test[:] == 0), 1],  # y coordinate of test_labels where class is 0
        marker="s",
        facecolors="r",
        edgecolors="w",
        label="Clase 0 de prueba ",
    )
    
    # B test plot
    plt.scatter(
        x_test[np.where(y_test[:] == 1), 0],  # x coordinate of test_labels where class is 1
        x_test[np.where(y_test[:] == 1), 1],  # y coordinate of test_labels where class is 1
        marker="o",
        facecolors="b",
        edgecolors="w",
        label="Clase 0 de prueba ",
    )
    
    
    #plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.title("Modelo óptimo de QSVM con Tau PAC Bayes",fontsize=20)
    
    plt.show()


def f(x_,y_,x_new):
    f2 = interpolate.interp1d(x_, y_, kind = 'cubic',axis=0, fill_value="extrapolate")
    return f2(x_new)


sets=1
delta=0.001 #condifence on the bound, that golds with prob
           #greater than delta.

# ## Importa datos de entrenamiento y prueba

seed = 12345
algorithm_globals.random_seed = seed

                            #cuantos sets de entrenamiento
val=np.arange(20, 110, 10, dtype=int)      #tamaño subsets de entrenamiento %
t_size=35 # Tamaño total del conjunto de datos  
train_s=int((t_size*80)/100)               #tamaño set de entrenamiento
s_size=int((t_size*20)/100)                 #tamaño set de prueba
subsets_len=np.dot(train_s,val)/int(100)    #tamaño de cada subset calculado del set de entrenamiento


tau_=np.zeros(len(subsets_len)) 
results_=np.zeros(len(subsets_len))      #variable donde se guardan las predicciones de PAC Bayes por cada subset
mean_err_=np.ones(len(subsets_len))     #variable donde se guardan las pruebas de k folding
score_= np.ones(len(subsets_len)) 
adhoc_dimension = 2
tau=np.ones(len(subsets_len)) 
results=np.ones(len(subsets_len))      #variable donde se guardan las predicciones de PAC Bayes por cada subset
mean_err=np.ones(len(subsets_len))     #variable donde se guardan las pruebas de k folding
score= np.ones(len(subsets_len))       #varaible donde se guarda la precision de cada subset en etapa de prueba
bound_tau= np.ones(len(subsets_len)) 
bound_pb= np.ones(len(subsets_len))
bound_tau_= np.ones(len(subsets_len)) 
bound_pb_= np.ones(len(subsets_len))
machines=np.ones(len(subsets_len))
# ## Define parámetro de confianza

### set SVM
adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)
rbf_svc = SVC(kernel="precomputed")   #tomamos una funcion kernel precomputada

#create dataset

x_train_, y_train_, x_test, y_test, adhoc_total = ad_hoc_data(  #crea sets de datos
    training_size=int(train_s/2),#int(big_t_set/2), #la funcion de qiskit tiene un error, devuelve el doble de datos de train y test
    test_size=int((3*s_size)),
    n=adhoc_dimension,
    gap=0.2,
    plot_data=True,
    one_hot=False,
    include_sample_total=True,
)

adhoc_total=0

x_test_=x_test
y_test_=y_test


x_test=np.concatenate((x_test_[:s_size],x_test_[int(len(x_test_)/2):int(len(x_test_)/2)+s_size]),axis=0)
y_test=np.concatenate((y_test_[:s_size],y_test_[int(len(y_test_)/2):int(len(y_test_)/2)+s_size]),axis=0)



for i in range(sets): # Se crean # sets de datos

    idx_0=np.where(y_train_[:]==0) #indices clase 0
    idx_1=np.where(y_train_[:]==1) #indices clase 1
    random.shuffle(x_train_[idx_0])
    random.shuffle(x_train_[idx_1])
    
        
    for j in range(len(val)): # se crean subconjuntos de datos de entrenamiento
        
        
        #Create subsets
        x_train=x_train_[idx_0[0][:int(subsets_len[j]/2)]]  #Agrega datos clase 0
        x_train=np.concatenate((x_train,x_train_[idx_1[0][:int(subsets_len[j]/2)]]),axis=0) #agrega datos clase 1
        y_train=y_train_[idx_0[0][:int(subsets_len[j]/2)]]
        y_train=np.concatenate((y_train,y_train_[idx_1[0][:int(subsets_len[j]/2)]]),axis=0)
        x=np.concatenate((x_train,x_test), axis=0)   
        print("Subset",j)
    
    
        out=svm_func(x,x_train,y_train,x_test,y_test)
        weights=out[0]
        support=out[1]
        mean_err[j]=out[2]
        score[j]=out[3]
        idx_sv=out[4]
        # if i==(sets-1):
        #     machines[j]=out[5]
        
        gram_all = adhoc_kernel.evaluate(x_vec=x) #Matriz kernel con todos los datos
    
        
        # # Inicia el calculo de la cota PAC Bayes
      
        out_2=pac_bayes(gram_all,weights,idx_sv,y_train,x_train,delta)
        results[j]=out_2[0]
        mod=out_2[1]
        gamma=out_2[2]
        bound_pb=out_2[3]
        
        # # Inicia calculo de la cota tau prior pac bayes
        out_3=tau_pac_bayes(x_train,y_train,mod,gamma)
        tau[j]=out_3[0]
        bound_tau[j]=out_3[1]
    #Analisis error del modelo
    tau_+=tau
    results_+=results
    mean_err_+=mean_err
    score_+=score
    bound_pb_+=bound_pb
    bound_tau_+=bound_tau
    
tau=tau_/sets
results =results_/sets 
mean_err =mean_err_/sets
score =score_/sets   
bound_pb=bound_pb_/sets
bound_tau=bound_tau_/sets
val_new=np.linspace(20,val[-1],22)

plt.subplots(figsize=(10, 5.4))
plt.plot(val_new,f(val,results,val_new), label='Error del clasificador estocastico de PAC Bayes',color='blue',fillstyle='none', marker='o', linestyle='-', markersize=7)  # Plot more data on the axes...
plt.plot(val_new,f(val,tau,val_new), label='Error del clasificador estocastico de tau PAC Bayes',color='m',fillstyle='none', marker='v', linestyle=':', markersize=7)  # Plot more data on the axes...
plt.plot(val_new,f(val,mean_err,val_new), label='Promedio del error de prueba con validación cruzada',color='orange', marker='.', linestyle='--', markersize=15)  # ... and some more.ax.plot(val,score, label='Precisión',color='green')  # ... and some more.
plt.plot(val_new,f(val,score,val_new), label='Error de prueba',color='green', marker=5, linestyle='-.', markersize=9)  # ... and some more.ax.set_ylabel('Precisión')  # Add a y-label to the axes.
plt.xlabel('Porcentaje del conjunto de entrenamiento',fontsize=14)  # Add an x-label to the axes.ax.legend();  # Add a legend.
plt.ylabel('Error estimado',fontsize=14)  # Add ay-label to the axes.
plt.title("Cálculo del error",fontsize=14)  # Add a title to the axes
plt.legend(loc='upper right')

plt.subplots(figsize=(10, 5.4))
plt.plot(val_new,f(val,bound_pb,val_new), label='Cota de generalización con PAC Bayes',color='blue',fillstyle='none', marker='o', linestyle='-', markersize=7)  # Plot more data on the axes...
plt.plot(val_new,f(val,bound_tau,val_new), label='Cota de generalización con tau PAC Bayes',color='m',fillstyle='none', marker='v', linestyle=':', markersize=7)  # Plot more data on the axes...
plt.plot(val_new,f(val,mean_err,val_new), label='Promedio del error de prueba con validación cruzada',color='orange', marker='.', linestyle='--', markersize=15)  # ... and some more.ax.plot(val,score, label='Precisión',color='green')  # ... and some more.
plt.plot(val_new,f(val,score,val_new), label='Error de prueba',color='green', marker=5, linestyle='-.', markersize=9)  # ... and some more.ax.set_ylabel('Precisión')  # Add a y-label to the axes.
plt.xlabel('Porcentaje del conjunto de entrenamiento',fontsize=14)  # Add an x-label to the axes.ax.legend();  # Add a legend.
plt.ylabel('Error estimado',fontsize=14) 
plt.title("Cota de generalización sobre datos de muestra",fontsize=14)  # Add a title to the axes
plt.show()

table={"Error del clasificador estocastico de PAC Bayes":results,'Error del clasificador estocastico de tau PAC Bayes':tau,'Promedio del error de prueba con validación cruzada':mean_err,'Error de prueba':score,'Cota de generalización con PAC Bayes':bound_pb,'Cota de generalización con tau PAC Bayes':bound_tau,'Porcentaje del conjunto de entrenamiento':val}

idx_0=np.where(y_train_[:]==0) #indices clase 0
idx_1=np.where(y_train_[:]==1) #indices clase 1
random.shuffle(x_train_[idx_0])
random.shuffle(x_train_[idx_1])

idx_0=np.where(y_test_[:]==0) #indices clase 0
idx_1=np.where(y_test_[:]==1) #indices clase 1
random.shuffle(x_test_[idx_0])
random.shuffle(x_test_[idx_1])


new_experiment(x_train_, y_train_,x_test_,y_test_,train_s,s_size,val,results,mean_err,score,tau,bound_pb,bound_tau)
print("\nExperiment finished.\n")
