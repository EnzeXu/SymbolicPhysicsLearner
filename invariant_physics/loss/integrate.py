import torch
from torch import Tensor

def trapezoid(x: Tensor, tspan: Tensor = None, dt: float = None, dim: int = 0):
    xT = x.transpose(0, dim)
    if dt is not None:
        assert tspan is None
        term1 = xT[1:-1].sum(dim = 0, keepdim = True)
        integ_x: Tensor = dt * (xT[0] + 2 * term1 + xT[-1]) / 2
    elif tspan is not None:
        assert tspan.dim() == 1
        assert dt is None
        factor1 = (xT[:-1] + xT[1:]) / 2
        factor2 = (tspan[1:] - tspan[:-1]).reshape((-1,) + (1,) * (factor1.dim() - 1))
        integ_x: Tensor = (factor2 * factor1).sum(dim = 0, keepdim = True)
    return integ_x.transpose(0, dim).squeeze(dim)

def simpson(x: Tensor, dt: float = None, dim: int = -1):
    xT = x.transpose(0, dim)
    x_even = xT if len(xT) % 2 else xT[:-3]
    term1: Tensor = x_even[1:-1:2].sum(dim = 0, keepdim = True)
    term2: Tensor = x_even[2:-1:2].sum(dim = 0, keepdim = True)
    integ_x: Tensor = 2 * dt / 6 * (x_even[0] + 4 * term1 + 2 * term2 + x_even[-1])
    if not len(xT) % 2:
        term3 = xT[-4] + 3*xT[-3] + 3*xT[-2] + xT[-1]
        integ_x = integ_x + 3 * dt * term3 / 8
    return integ_x.transpose(0, dim).squeeze(dim)

def boole(x: Tensor, dt: float = None, dim: int = -1):
    xT = x.transpose(0, dim)
    if (len(xT) - 1) % 4 == 0: x_trunc, remain = xT, 0
    elif (len(xT) - 1) % 4 == 1:
        temp = 19*xT[-6] + 75*xT[-5] + 50*xT[-4] + 50*xT[-3] + 75*xT[-2] + 19*xT[-1]
        x_trunc, remain = xT[:-5], 5 / 288 * dt * temp
    elif (len(xT) - 1) % 4 == 2:
        x_trunc, remain = xT[:-2], dt / 3 * (xT[-3] + 4*xT[-2] + xT[-1])
    elif (len(xT) - 1) % 4 == 3:
        temp = (xT[-4] + 3*xT[-3] + 3*xT[-2] + xT[-1])
        x_trunc, remain = xT[:-3], 3 / 8 * dt * temp
    term1: Tensor = x_trunc[1:-1:4].sum(dim = 0, keepdim = True)
    term2: Tensor = x_trunc[2:-1:4].sum(dim = 0, keepdim = True)
    term3: Tensor = x_trunc[3:-1:4].sum(dim = 0, keepdim = True)
    term4: Tensor = x_trunc[4:-1:4].sum(dim = 0, keepdim = True)
    integ_x: Tensor = 4 * dt / 90 * (7*x_trunc[0] + 32*term1 + 12*term2 + 32*term3 + 14*term4 + 7*x_trunc[-1])
    integ_x = integ_x + remain
    return integ_x.transpose(0, dim).squeeze(dim)

def trapezoid_romberg(x: Tensor, tspan: Tensor, dim: int = -1):
    xT = x.transpose(0, dim)
    integral3 = torch.trapezoid(xT, tspan, dim = 0)
    integral2 = torch.trapezoid(xT[::2], tspan[::2], dim = 0)
    integral1 = torch.trapezoid(xT[::4], tspan[::4], dim = 0)
    romb_integ2 = (4 * integral3 - integral2) / 3
    romb_integ1 = (4 * integral2 - integral1) / 3
    romb_integ = (15 * romb_integ2 - romb_integ1) / 16
    return romb_integ.unsqueeze(0).transpose(0, dim).squeeze(dim)

def irregular_simpson(integrand: Tensor, sampled_tspan: Tensor, dim: int = 0):
    temp = integrand.transpose(0, dim)
    dt = sampled_tspan[1:] - sampled_tspan[:-1]
    dt = dt.unsqueeze(1).unsqueeze(-1)
    assert len(dt) % 2 == 0
    dt0, dt1 = dt[::2], dt[1::2]
    X0, X1 = temp[::2], temp[1::2]
    coeff = ((dt0 + dt1) / 6)
    coeff1 = ((2*dt0 - dt1) / dt0)
    coeff2 = ((dt0 + dt1)**2 / (dt0 * dt1))
    coeff3 = ((2*dt1 - dt0) / dt1)
    integ_raw = (coeff * (coeff1*X0[:-1] + coeff2*X1 + coeff3*X0[1:])).sum(dim = 0, keepdim = True)
    return integ_raw.transpose(0, dim).squeeze(dim)