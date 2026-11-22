"""Physical units, circular wind errors, Train-IQR-normalized MAE.

Functions reproduced directly from the audited SIRIUS protocol.
"""
import math
import numpy as np
import torch
from .data import SENSORS, WIND_INDEX, as_loader

def metric_tables(y_true,y_pred,scaler,H):
    """Inverse-transform to original units and evaluate circular wind direction."""
    y_true=np.asarray(y_true,dtype=np.float64);y_pred=np.asarray(y_pred,dtype=np.float64)
    if y_true.shape!=y_pred.shape or y_true.ndim!=3 or y_true.shape[1:]!=(H,9):
        raise ValueError(f'Invalid arrays for metrics {y_true.shape} / {y_pred.shape}')
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError('Non-finite predictions/targets')
    means=scaler['mean'][:9];std=scaler['sd'][:9];iqrs=scaler['iqr'][:9]
    yt=y_true*std+means;yp=y_pred*std+means
    # Circular predictions are physically identical modulo 360 degrees.
    # Publish a canonical angle in [0,360) for human-readable forecasts.
    # Keep observed y_true untouched, including any original 360-degree readings.
    yp[:,:,WIND_INDEX]=np.mod(yp[:,:,WIND_INDEX],360.0)
    err=yp-yt
    err[:,:,WIND_INDEX]=(err[:,:,WIND_INDEX]+180.0)%360.0-180.0
    ae=np.abs(err)
    sq=err**2
    by_sensor=[];by_step=[]
    for j,name in enumerate(SENSORS):
        true=yt[:,:,j]; e=err[:,:,j]
        # Conventional R² is not invariant to circular wrap (359° and 1° are
        # only 2° apart). Do NOT calculate or report it for wind direction.
        # The physically meaningful angular MAE/RMSE/nMAE are retained above.
        if j==WIND_INDEX:
            r2=None
            r2_status='not_applicable_angular'
        else:
            sst=np.sum((true-true.mean())**2)
            r2=float(1-np.sum(e**2)/sst) if sst>0 else None
            r2_status='linear' if sst>0 else 'undefined_constant_truth'
        by_sensor.append({'sensor':name,'error_type':'angular' if j==WIND_INDEX else 'linear',
                          'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),
                          'mse':float(np.mean(e*e)),'bias':float(np.mean(e)),
                          'p95_abs_error':float(np.percentile(np.abs(e),95)),
                          'nmae_iqr':float(np.mean(np.abs(e))/iqrs[j]),'iqr_train':float(iqrs[j]),
                          'r2_linear':r2,'r2_status':r2_status})
    for step in range(H):
        row={'step':step+1,'minutes_ahead':(step+1)*15,'n':int(len(yt)),
             'global_nmae_iqr':float(np.mean(ae[:,step,:].mean(axis=0)/iqrs))}
        for j,name in enumerate(SENSORS):
            row['nmae_'+name]=float(np.mean(ae[:,step,j])/iqrs[j])
            row['mae_'+name]=float(np.mean(ae[:,step,j]))
        by_step.append(row)
    global_metrics={'n_windows':int(len(yt)),'horizon':H,
                    'nmae_global_iqr':float(np.mean([x['nmae_iqr'] for x in by_sensor])),
                    'sensor_count':9,
                    'note':'Wind direction uses wrapped angular errors; conventional R2 is not applicable to wind direction (null). Other sensor R2 values use linear physical units.'}
    return global_metrics,by_sensor,by_step,yt.astype(np.float32),yp.astype(np.float32)


def mixed_circular_mse(prediction,target,wind_train_sd):
    """Normalized squared error over nine sensors, with circular wind residual.

    Each ordinary sensor contributes squared z-score residual. The wind channel
    contributes the shortest angular displacement (degrees) divided by its
    Train-only standard deviation, squared. This keeps 1/9 weighting and the
    existing scale convention while respecting 0 deg == 360 deg.

    torch.remainder is piecewise differentiable; at exact antipodal errors
    (+/-180 deg) the shortest-angle derivative has an unavoidable cusp.
    """
    if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[-1]!=len(SENSORS):
        raise ValueError(f'Wrong shapes for circular loss: {prediction.shape} / {target.shape}')
    if not math.isfinite(float(wind_train_sd)) or wind_train_sd<=0:
        raise ValueError('Wind Train standard deviation must be positive and finite')
    delta=prediction-target
    period=360.0/float(wind_train_sd)
    wind_delta=torch.remainder(delta[...,WIND_INDEX]+period/2.0,period)-period/2.0
    # One squared contribution per sensor, per forecast step, per observation.
    return (delta[...,:WIND_INDEX].square().sum()+
            wind_delta.square().sum()+
            delta[...,WIND_INDEX+1:].square().sum())/prediction.numel()


def eval_validation_objectives(model,values,starts,L,H,batch_size,device,scaler):
    """One Validation forward pass: mixed circular MSE and circular physical nMAE/IQR.

    Streaming sum avoids accumulating all [N,H,9] predictions each epoch.
    The angular difference is wrapped in z-score units using Train-only SD;
    other errors are inverse-scaled to original physical units. IQR divisors
    are fitted on Train only and identical to metric_tables() for model selection.
    """
    sd=np.asarray(scaler['sd'][:9],dtype=np.float64)
    iqr=np.asarray(scaler['iqr'][:9],dtype=np.float64)
    if not (np.isfinite(sd).all() and (sd>0).all() and
            np.isfinite(iqr).all() and (iqr>0).all()):
        raise ValueError('Invalid Train statistics for Validation nMAE/IQR')
    device_sd=torch.as_tensor(sd,dtype=torch.float64,device=device)
    wind_sd=float(sd[WIND_INDEX])
    period=360.0/wind_sd
    squared_sum=0.0;count=0
    absolute_sums=np.zeros(9,dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for x,y in as_loader(values,starts,L,H,batch_size):
            predicted=model(x.to(device)); actual=y.to(device)
            if predicted.shape!=actual.shape:
                raise ValueError(f'Validation shape mismatch {predicted.shape} != {actual.shape}')
            if not torch.isfinite(predicted).all():
                raise RuntimeError('Non-finite Validation model predictions')
            batch_loss=mixed_circular_mse(predicted,actual,wind_sd)
            if not torch.isfinite(batch_loss):
                raise RuntimeError('Non-finite Validation loss')
            squared_sum+=float(batch_loss.item())*actual.numel()
            count+=int(actual.shape[0]*actual.shape[1])
            residual=(predicted-actual).to(torch.float64)
            residual[...,WIND_INDEX]=torch.remainder(
                residual[...,WIND_INDEX]+period/2.0,period)-period/2.0
            absolute_sums+=(residual.abs()*device_sd).sum(dim=(0,1)).cpu().numpy()
    if count==0:raise RuntimeError('Empty Validation: cannot select checkpoint')
    nmae=float(np.mean(absolute_sums/count/iqr))
    mse=float(squared_sum/(count*len(SENSORS)))
    if not math.isfinite(nmae) or not math.isfinite(mse):
        raise RuntimeError('Non-finite Validation monitoring metrics')
    return {'val_mixed_loss_z':mse,'val_nmae_iqr':nmae}
