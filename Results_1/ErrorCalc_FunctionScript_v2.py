"""
Functions script for calculating accuracy in image-based surrogate modeling

@author Anindya Bhaduri
GE Global Research Center (GRC)
"""

import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular') 
import numpy as np
#import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
     
def error_calculation(dataset,
                      x,
                      y_true,
                      y_pred,
                      Current_Model_directory,
                      file_str,
                      original_height,
                      original_width):
    
    if 'y_max' not in globals():
        globals()['y_max'] = -np.inf
        globals()['y_min'] = np.inf

    p_list =  [50, 90, 97, 99, 100] 
    for i in range(len(p_list)):
    
        # globals()[f'y_true_p{p_list[i]}'] = np.percentile(y_true.numpy(), p_list[i], axis = (1,2,3))
        # globals()[f'y_pred_p{p_list[i]}'] = np.percentile(y_pred.numpy(), p_list[i], axis = (1,2,3))
        globals()[f'y_true_p{p_list[i]}'] = np.percentile(y_true, p_list[i], axis = (1,2,3))
        globals()[f'y_pred_p{p_list[i]}'] = np.percentile(y_pred, p_list[i], axis = (1,2,3))
        
        globals()['y_max'] = np.amax([globals()['y_max'],
                                      np.amax(globals()[f'y_true_p{p_list[i]}']), 
                                      np.amax(globals()[f'y_pred_p{p_list[i]}'])])

        globals()['y_min'] = np.amin([globals()['y_min'],
                                      np.amin(globals()[f'y_true_p{p_list[i]}']), 
                                      np.amin(globals()[f'y_pred_p{p_list[i]}'])])
    
        if 'figure'+str(i) not in globals():
            globals()[f'figure{i}'], globals()[f'ax{i}'] = plt.subplots(figsize=(8,8))
        if 'figure100' not in globals():
            globals()['figure100'], globals()['ax100'] = plt.subplots(figsize=(8,8))
        if dataset == 'train':
            globals()[f'ax{i}'].loglog(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}'], 'ro', label = 'Train')
            globals()['ax100'].loglog(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}'], 'ro')
            print('r2_p' + str(p_list[i]) + '_train', r2_score(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}']))
            print('nRMSE_p' + str(p_list[i]) + '_train', mean_squared_error(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}'], squared=False)/np.mean(globals()[f'y_true_p{p_list[i]}']))         
        elif dataset == 'val':
            globals()[f'ax{i}'].loglog(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}'], 'bo', label = 'Validation')
            globals()['ax100'].loglog(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}'], 'bo')
            print('r2_p' + str(p_list[i]) + '_val', r2_score(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}']))
            print('nRMSE_p' + str(p_list[i]) + '_val', mean_squared_error(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}'], squared=False)/np.mean(globals()[f'y_true_p{p_list[i]}']))             
        elif dataset == 'test':
            globals()[f'ax{i}'].loglog(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}'], 'ko', label = 'Test')
            globals()['ax100'].loglog(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}'], 'ko')
            print('r2_p' + str(p_list[i]) + '_test', r2_score(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}'])) 
            print('nRMSE_p' + str(p_list[i]) + '_test', mean_squared_error(globals()[f'y_true_p{p_list[i]}'], globals()[f'y_pred_p{p_list[i]}'], squared=False)/np.mean(globals()[f'y_true_p{p_list[i]}'])) 
                        
        globals()[f'ax{i}'].loglog([globals()['y_min'], globals()['y_max']], [globals()['y_min'], globals()['y_max']], 'k--')
        globals()[f'ax{i}'].set_title(str(p_list[i]) + '-th percentile stress prediction')
        globals()[f'ax{i}'].set_xlabel('True')
        globals()[f'ax{i}'].set_ylabel('Predicted')
        globals()[f'ax{i}'].legend(loc = 'best')
        globals()[f'figure{i}'].savefig(Current_Model_directory + 'p' + str(p_list[i]) + '_stress_plot' + file_str + '.png')


    # y_true_mean = np.mean(y_true.numpy(), axis = (1,2,3))
    # y_pred_mean = np.mean(y_pred.numpy(), axis = (1,2,3)) 
    y_true_mean = np.mean(y_true, axis = (1,2,3))
    y_pred_mean = np.mean(y_pred, axis = (1,2,3)) 
    
    globals()['y_max'] = np.amax([globals()['y_max'],
                                  np.amax(y_true_mean), 
                                  np.amax(y_pred_mean)])

    globals()['y_min'] = np.amin([globals()['y_min'],
                                  np.amin(y_true_mean), 
                                  np.amin(y_pred_mean)])
                                       
    if 'figure'+str(i+1) not in globals():
        globals()[f'figure{i+1}'], globals()[f'ax{i+1}'] = plt.subplots(figsize=(8,8))
    if dataset == 'train':
        globals()[f'ax{i+1}'].loglog(y_true_mean, y_pred_mean, 'ro', label = 'Train')
        globals()['ax100'].loglog(y_true_mean, y_pred_mean, 'ro', label = 'Train')
        print('r2_mean_train', r2_score(y_true_mean, y_pred_mean)) 
        print('nRMSE_mean_train', mean_squared_error(y_true_mean, y_pred_mean, squared=False)/np.mean(y_true_mean)) 
    elif dataset == 'val':
        globals()[f'ax{i+1}'].loglog(y_true_mean, y_pred_mean, 'bo', label = 'Validation')
        globals()['ax100'].loglog(y_true_mean, y_pred_mean, 'bo', label = 'Validation')
        print('r2_mean_val', r2_score(y_true_mean, y_pred_mean)) 
        print('nRMSE_mean_val', mean_squared_error(y_true_mean, y_pred_mean, squared=False)/np.mean(y_true_mean))
    elif dataset == 'test':
        globals()[f'ax{i+1}'].loglog(y_true_mean, y_pred_mean, 'ko', label = 'Test')
        globals()['ax100'].loglog(y_true_mean, y_pred_mean, 'ko', label = 'Test')
        print('r2_mean_test', r2_score(y_true_mean, y_pred_mean)) 
        print('nRMSE_mean_test', mean_squared_error(y_true_mean, y_pred_mean, squared=False)/np.mean(y_true_mean))
    globals()[f'ax{i+1}'].loglog([globals()['y_min'], globals()['y_max']], [globals()['y_min'], globals()['y_max']], 'k--')
    globals()[f'ax{i+1}'].set_title('Mean stress prediction')
    globals()[f'ax{i+1}'].set_xlabel('True')
    globals()[f'ax{i+1}'].set_ylabel('Predicted')
    globals()[f'ax{i+1}'].legend(loc = 'best')
    globals()[f'figure{i+1}'].savefig(Current_Model_directory + 'mean_stress_plot' + file_str + '.png')
    
    
    # plt.rc('xtick', labelsize=18) 
    globals()['ax100'].loglog([globals()['y_min'], globals()['y_max']], [globals()['y_min'], globals()['y_max']], 'k--')
    globals()['ax100'].set_title('All stress prediction')
    globals()['ax100'].set_xlabel('True stress (MPa)', fontsize=18)
    globals()['ax100'].set_ylabel('Predicted stress (MPa)', fontsize=18)
    globals()['ax100'].tick_params(axis='x', which = 'both', labelsize= 18)
    globals()['ax100'].tick_params(axis='y', which = 'both', labelsize= 18, labelrotation = 90)
    globals()['ax100'].legend(loc = 'best', prop={"size":18})
    globals()['figure100'].savefig(Current_Model_directory + 'All_stress_plot' + file_str + '.png')
    

    if dataset == 'test':
        _show_predictions(x, 
                          y_pred, 
                          y_true, 
                          original_height,
                          original_width,
                          Current_Model_directory, 
                          file_str, 
                          cases_to_plot = 2)
        # plt.close('all')
        # # clear all global variables
        # for var in globals().copy():
        #     if var.startswith('ax') or var.startswith('figure'):
        #         del globals()[var]
    


def _display(display_list, 
             saving_filename):
    
    plt.figure(figsize=(20, 20))

    title = ['Input Image', 'True Stress Map', 'Predicted Stress Map', 'Stress Error Map']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')
        if i > 0:
            plt.colorbar(fraction = 0.046, pad = 0.04)
            # if i < 3:
                # plt.clim(tf.math.reduce_min(display_list[1]), tf.math.reduce_max(display_list[1]))
                
    if saving_filename is not None:
        plt.savefig(saving_filename)

def _show_predictions(input, 
                      pred_output, 
                      true_output, 
                      original_height,
                      original_width,
                      Current_Model_directory,
                      file_str,
                      cases_to_plot):
    
    # predictions_loss = tf.math.truediv(tf.reduce_sum(tf.math.multiply(tf.math.abs(true_output), tf.math.square(true_output - pred_output)), axis = [1,2,3]), 
                                       # tf.reduce_sum(true_output, axis = [1,2,3]))
    # ind = np.argsort(predictions_loss.numpy().flatten())
    
    predictions_loss = np.true_divide(np.sum(np.multiply(np.abs(true_output), np.square(true_output - pred_output)), axis = (1,2,3)), 
                                      np.sum(true_output, axis = (1,2,3)))
    ind = np.argsort(predictions_loss.flatten())
                                       
    ind_best = ind[:cases_to_plot]
    ind_worst = ind[-cases_to_plot:][::-1]
    
    # reconstruction visualization by cross-section, horizontal direction
    cross_section_Z = original_height//2
    cross_section_X = original_width//2

    for i in range(len(ind_best)):
        
        saving_filename1 = Current_Model_directory + 'images_best' + str(i) + file_str + '.png'
        saving_filename2 = Current_Model_directory + 'VerticalCrossSection_images_best' + str(i) + file_str + '.png'
        saving_filename3 = Current_Model_directory + 'HorizontalCrossSection_images_best' + str(i) + file_str + '.png'
        batch_ind = ind_best[i]
        
        x = input[batch_ind, :, :, 0] 
        y_pred = pred_output[batch_ind, :, :, 0]                 
        y_true = true_output[batch_ind, :, :, 0]
        # _display([x, 
                 # y_true,
                 # y_pred,
                 # tf.math.abs(y_true - y_pred)], saving_filename1)
        _display([x, 
                 y_true,
                 y_pred,
                 np.abs(y_true - y_pred)], saving_filename1)
                 
        plt.figure(figsize=(10, 8))
        plt.rc('font', size=20)
        plt.plot(y_true[:, cross_section_X - 1], 'k', label="true")  # vertical cross-section
        plt.plot(y_pred[:, cross_section_X - 1], '--', label="predicted")
        plt.legend(loc='best')
        plt.xlabel('z | (x = ' + str(cross_section_X - 1) + ')')
        plt.ylabel('von Mises stress (MPa)')
        
        if saving_filename2 is not None:
            plt.savefig(saving_filename2)
            
        plt.figure(figsize=(10, 8))
        plt.rc('font', size=20)
        plt.plot(y_true[cross_section_Z - 1, :], 'k', label="true")  # horizontal cross-section
        plt.plot(y_pred[cross_section_Z - 1, :], '--', label="predicted")
        plt.legend(loc='best')
        plt.xlabel('x | (z = ' + str(cross_section_Z - 1) + ')')
        plt.ylabel('von Mises stress (MPa)')
        
        if saving_filename3 is not None:
            plt.savefig(saving_filename3)
        
            
    for i in range(len(ind_worst)):

        saving_filename1 = Current_Model_directory  + 'images_worst' + str(i) + file_str + '.png'
        saving_filename2 = Current_Model_directory  + 'VerticalCrossSection_images_worst' + str(i) + file_str + '.png'
        saving_filename3 = Current_Model_directory  + 'HorizontalCrossSection_images_worst' + str(i) + file_str + '.png'
        batch_ind = ind_worst[i]
        
        x = input[batch_ind, :, :, 0] 
        y_pred = pred_output[batch_ind, :, :, 0]                 
        y_true = true_output[batch_ind, :, :, 0] 
        # _display([x, 
                  # y_true, 
                  # y_pred, 
                  # tf.math.abs(y_true - y_pred)], saving_filename1)
        _display([x, 
                  y_true, 
                  y_pred, 
                  np.abs(y_true - y_pred)], saving_filename1)
                  
        plt.figure(figsize=(10, 8))
        plt.rc('font', size=20) 
        plt.plot(y_true[:, cross_section_X - 1], 'k', label="true")  # vertical cross-section
        plt.plot(y_pred[:, cross_section_X - 1], '--', label="predicted")
        plt.legend(loc='best') 
        plt.xlabel('z | (x = ' + str(cross_section_X - 1) + ')')
        plt.ylabel('von Mises stress (MPa)')
        
        if saving_filename2 is not None:
            plt.savefig(saving_filename2)
            
        plt.figure(figsize=(10, 8))
        plt.rc('font', size=20)
        plt.plot(y_true[cross_section_Z - 1, :], 'k', label="true")  # horizontal cross-section
        plt.plot(y_pred[cross_section_Z - 1, :], '--', label="predicted")
        plt.legend(loc='best')
        plt.xlabel('x | (z = ' + str(cross_section_Z - 1) + ')')
        plt.ylabel('von Mises stress (MPa)')
        
        if saving_filename3 is not None:
            plt.savefig(saving_filename3)