"""Voxel Dice and directed-surface HD95; no patient identifiers or I/O."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
SPACING = 2.75
STRUCTURE = ndi.generate_binary_structure(3, 1)

def points(mask,surface=False):
    if mask.ndim!=3 or not mask.any():raise ValueError('Nonempty 3D mask required')
    a=mask.astype(bool)
    if surface:a=a&~ndi.binary_erosion(a,structure=STRUCTURE,border_value=0)
    return np.argwhere(a).astype(np.float64)*SPACING


def distances(x,y):
    return cKDTree(y).query(x,workers=1)[0],cKDTree(x).query(y,workers=1)[0]


def surface_metrics(x,y):
    a,b=distances(x,y)
    return {'hd95_mm':float(max(np.percentile(a,95),np.percentile(b,95))),
        'pred_to_gt_p95_mm':float(np.percentile(a,95)),'gt_to_pred_p95_mm':float(np.percentile(b,95)),
        'pred_to_gt_max_mm':float(a.max()),'gt_to_pred_max_mm':float(b.max()),
        'mean_symmetric_distance_mm':float((a.mean()+b.mean())/2),
        'symmetric_mse_mm2':float((np.mean(a*a)+np.mean(b*b))/2),
        'gt_surface_farther_than_5p5_fraction':float(np.mean(b>5.5+1e-9))}


def dense_stats(pred,gt):
    npred=int(pred.sum());ngt=int(gt.sum());tp=int((pred&gt).sum())
    return {'dice':2*tp/(npred+ngt),'predicted_voxels':npred,'target_voxels':ngt,
        'tp':tp,'fp':npred-tp,'fn':ngt-tp}


def evaluate_labels(labels, target):
    """Macro-average five vertebrae; missing predictions retain Dice zero.

    HD95 is null for a missing target/prediction and averaged only where
    defined. Missing predictions are counted separately, never silently dropped.
    """
    labels, target = np.asarray(labels), np.asarray(target)
    if labels.shape != target.shape or labels.ndim != 3:
        raise ValueError("Expected paired 3D arrays")
    if not np.isin(labels, np.arange(6)).all() or not np.isin(target, np.arange(6)).all():
        raise ValueError("Expected labels zero through five")
    rows = []
    for k in range(1, 6):
        pred, gt = labels == k, target == k
        exists = bool(gt.any())
        missing = exists and not bool(pred.any())
        dice = dense_stats(pred, gt)['dice'] if exists else None
        hd = surface_metrics(points(pred, True), points(gt, True))['hd95_mm'] if exists and pred.any() else None
        rows.append({'vertebra': f'L{k}', 'dice': dice, 'hd95_mm': hd, 'miss': int(missing)})
    dices = [r['dice'] for r in rows if r['dice'] is not None]
    hds = [r['hd95_mm'] for r in rows if r['hd95_mm'] is not None]
    return {'dice': float(np.mean(dices)) if dices else None,
            'hd95_mm': float(np.mean(hds)) if hds else None,
            'miss_count': sum(r['miss'] for r in rows), 'per_vertebra': rows}
