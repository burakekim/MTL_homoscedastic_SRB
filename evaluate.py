import numpy as np
import torch
import tqdm
from sklearn.metrics import (accuracy_score, f1_score, jaccard_score,
                             precision_score, recall_score)


def evaluate(model, generator):
    """ Evaluation function. Calculates *y* and  *y_hat* arrays. 
        usage:
        from sklearn.metrics import classification_report (or any other evaluation metric)
        y_pred, y = evaluate(best_model, generator)
        y_pred  = np.hstack(y_pred)
        y = np.hstack(y)
        targets = ['building', 'not_building']
        print(classification_report(y, y_pred, target_names = targets))#, output_dict = True)
    """
    
    best_model = model
    best_model.cuda()
    best_model.eval()

    y_preds = list()
    ys = list()
    with torch.no_grad():
        for idx in tqdm.tqdm(range(len(generator))):
            X,y,z = generator[idx] #image, mask, edge_mask
            X = X.cuda()
            #X = X.detach().cpu().numpy() #no need to further increase the computational burden .
            y = y.detach().cpu().numpy()
            gt_max = np.argmax(y, axis=0)
            segmentation, edge, reconstruction, sigma = best_model.forward(X[None,:,:])
            y_pred = segmentation.argmax(dim=1)
            y_pred = y_pred.squeeze()
            gt_color = gt_max.flatten()
            y= gt_color
            y_pred = y_pred.flatten().detach().cpu().numpy()
            
            ys.append(y)
            y_preds.append(y_pred)
            
    best_model.cpu()
    return y_preds, ys    


def calculate_metrics(y_preds,ys):
    """
    Intented to complement the *evaluate* function. 
    Calculates evaluation metrics for a given *y* and *y_hat* arrays. 
    """
    y_preds = np.asarray(y_preds)
    ys = np.asarray(ys)

    include_label = [0,1] # omit background during metric calculation 

    F1 = f1_score(ys.flatten(), y_preds.flatten(), average=None, labels=include_label)
    Precision = precision_score(ys.flatten(), y_preds.flatten(), average=None,labels=include_label)
    Recall = recall_score(ys.flatten(), y_preds.flatten(), average=None, labels=include_label)
    Jaccard = jaccard_score(ys.flatten(), y_preds.flatten(), average=None,labels=include_label)
    acc = accuracy_score(ys.flatten(), y_preds.flatten())

    f1 = np.asarray(F1)
    prec = np.asarray(Precision)
    rec = np.asarray(Recall)
    jacc = np.asarray(Jaccard)

    return f1, prec, rec, jacc, acc
