import torch

def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    return -x + torch.einsum('bijd, bd -> bij', part_Us, VTx)     # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def neumann(v0, threshold, eps):
    vt = v0
    gt = v0
    nstep = 0
    max_unroll = 3
    bsz, total_hsize, seq_len = x0.size()

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, seq_len, threshold)     # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, threshold, total_hsize, seq_len)

    for t in range(max_unroll):
        vt = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], vt) 
        gt += vt**t # Add the power according to the formula
        nstep += 1
        if torch.norm(vt).item() <= eps:
            break;

    return {'result': gt,
            'nstep': nstep,
            'diff': torch.norm(vt).item()}