import torch
import torchdiffeq

import torch.nn as nn
import torch.nn.functional as F
# import scipy
class AttnVectorField(torch.nn.Module):
    def __init__(self, dX_dt, func):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(AttnVectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field = self.func(z)
 
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out

class VectorField_control(torch.nn.Module):
    def __init__(self, dX_dt, func, hid_dim = 256, control_dim = 4):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField_control, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func
        self.control_proj = nn.Linear(hid_dim, control_dim)

    def __call__(self, t, z):
        # control_gradient is of shape (..., hidden_channels)
        self.control_proj = self.control_proj.to(z.device)

        control_gradient = self.dX_dt(t)
    
        control_gradient = self.control_proj(control_gradient) # of shape (..., input_channels)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field = self.func(z)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out
    
class VectorField(torch.nn.Module):
    def __init__(self, dX_dt, func, hid_dim = 256, control_dim = 4):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField, self).__init__()
        if not isinstance(func, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func = func
        self.attention_cache = []  # 用于记录注意力矩阵
        self.record_attention = False  # 是否记录注意力的标志

    def __call__(self, t, z):
        # control_gradient is of shape (..., hidden_channels)

        control_gradient = self.dX_dt(t)
    
        # vector_field is of shape (..., hidden_channels, input_channels)
        # 如果需要记录注意力，调用 func 时传入 return_attn=True
        if self.record_attention and hasattr(self.func, 'forward'):
            try:
                vector_field, temporal_attns, spatial_attns = self.func(z, return_attn=True)
                # 记录注意力
                self.attention_cache.append({
                    't': t.item() if t.numel() == 1 else t,
                    'temporal_attns': temporal_attns,
                    'spatial_attns': spatial_attns
                })
            except:
                vector_field = self.func(z)
        else:
            vector_field = self.func(z)
        
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out
    
    def clear_attention_cache(self):
        """清空注意力缓存"""
        self.attention_cache = []
    
    def get_attention_cache(self):
        """获取记录的注意力"""
        return self.attention_cache

class VectorFieldGDE(torch.nn.Module):
    def __init__(self, dX_dt, func_f, func_g):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func_f: As cdeint.
            func_g: As cdeint.
        """
        super(VectorFieldGDE, self).__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func_f = func_f
        self.func_g = func_g

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        
        
        vector_field_f = self.func_f(z)
        vector_field_g = self.func_g(z)
        
        vector_field_fg = torch.mul(vector_field_g, vector_field_f)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # out = (vector_field_g @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out

class VectorFieldGDE_dev_2(torch.nn.Module):
    def __init__(self, dX_dt, func_f, func_g, node_embed=None):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func_f: As cdeint.
            func_g: As cdeint.
        """
        super(VectorFieldGDE_dev_2, self).__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func_f = func_f
        self.func_g = func_g
        self.node_embed = node_embed

    def __call__(self, t, hz):
        """
        hz:init0 = (h0,z0)
        输入与输出的维度必须是一样的，这样微分方程能够连续求解
        """
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t) #(B,N,coeff_dim)
        # vector_field is of shape (..., hidden_channels, coeff_dim)
        h = hz[0] #(B,N,hidden_dim)
        z = hz[1] #(B,N,hidden_dim)
        vector_field_f = self.func_f(h) # (B,N,hidden_dim, hidden_dim)
        if self.node_embed is not None: #func_g = VectorField_dg
             
            vector_field_g = self.func_g(z, [self.node_embed[0][:,min(int(t.item()),self.node_embed[0].shape[1] - 1),:,:],self.node_embed[1]]) #  (B,N,hidden_dim, hidden_dim)
        else:
            vector_field_g = self.func_g(z) #  (B,N,hidden_dim, hidden_dim)

        # vector_field_fg = torch.mul(vector_field_g, vector_field_f) # vector_field_fg: torch.Size([64, 207, 32, 2])
        vector_field_fg = torch.matmul(vector_field_g, vector_field_f)# (B,N,hidden_dim, hidden_dim)
        # out is of shape (..., hidden_channels)
        
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        dh = (vector_field_f @ control_gradient.unsqueeze(-1)).squeeze(-1)#(B,N,hidden_dim, hidden_dim)@(B,N,coeff_dim=hidden_dim,1) ===> (B,N,hidden_dim, 1)===>  (B,N,hidden_dim)
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(-1)#(B,N,hidden_dim, hidden_dim)@(B,N,coeff_dim=hidden_dim,1) ===> (B,N,hidden_dim, 1)===>  (B,N,hidden_dim)
        # import pdb;pdb.set_trace()
        # dh: torch.Size([64, 207, 32])
        # out: torch.Size([64, 207, 32])
        # return out
        return tuple([dh,out])  # (B,N,hidden_dim), (B,N,hidden_dim)
class VectorFieldGDE_dev_3(torch.nn.Module):
    def __init__(self, dX_dt, func_f, func_g, func_j):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func_f: As cdeint.
            func_g: As cdeint.
        """
        super(VectorFieldGDE_dev_3, self).__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func_f = func_f
        self.func_g = func_g
        self.func_j = func_j

    def __call__(self, t, hz): #t是分段的，1/3为单位

        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)

        h = hz[0] # h: torch.Size([64, 207, 32])
        z = hz[1] # z: torch.Size([64, 207, 32])
        vector_field_f = self.func_f(h) # vector_field_f: torch.Size([64, 207, 32, 2])
        vector_field_g = self.func_g(z) # vector_field_g: torch.Size([64, 207, 32, 2])
        vector_field_j = self.func_g(h+z) # vector_field_g: torch.Size([64, 207, 32, 2])

        # vector_field_fg = torch.mul(vector_field_g, vector_field_f) # vector_field_fg: torch.Size([64, 207, 32, 2])
        vector_field_fg = torch.matmul(vector_field_g, vector_field_f)
        vector_field_fg = vector_field_fg + vector_field_j
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        dh = (vector_field_f @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # dh: torch.Size([64, 207, 32])
        # out: torch.Size([64, 207, 32])
        return tuple([dh,out])


class VectorFieldAGDE(torch.nn.Module):
    def __init__(self, dX_dt, func_f, func_g):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func_f: As cdeint.
            func_g: As cdeint.
        """
        super(VectorFieldGDE_dev, self).__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func_f = func_f
        self.func_g = func_g

    def __call__(self, t, hz): #t是分段的，1/3为单位

        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)

        h = hz[0] # h: torch.Size([64, 207, 32])
        z = hz[1] # z: torch.Size([64, 207, 32])
        vector_field_f = self.func_f(h) #BNHH
        vector_field_g = self.func_g(z) # BNHH

        # vector_field_fg = torch.mul(vector_field_g, vector_field_f) # vector_field_fg: torch.Size([64, 207, 32, 2])
        vector_field_fg = torch.matmul(vector_field_g, vector_field_f) #BNHI
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)

        # BNHI@BNI1=>BN1=>BNH
        dh = (vector_field_f @ control_gradient.unsqueeze(-1)).squeeze(-1)

        #  BNHI@BNI1=>BN1=>BNH
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        """
        输出必须是BNhidden_dim BNH
        """
        return tuple([dh,out])

class VectorFieldGDE_dev(torch.nn.Module):
    def __init__(self, dX_dt, func_f, func_g):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func_f: As cdeint.
            func_g: As cdeint.
        """
        super(VectorFieldGDE_dev, self).__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func_f = func_f
        self.func_g = func_g

    def __call__(self, t, hz): #t是分段的，1/3为单位

        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)

        h = hz[0] # h: torch.Size([64, 207, 32])
        z = hz[1] # z: torch.Size([64, 207, 32])
        vector_field_f = self.func_f(h) #BNHH
        vector_field_g = self.func_g(z) # BNHH

        # vector_field_fg = torch.mul(vector_field_g, vector_field_f) # vector_field_fg: torch.Size([64, 207, 32, 2])
        vector_field_fg = torch.matmul(vector_field_g, vector_field_f) #BNHI
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)

        # BNHI@BNI1=>BN1=>BNH
        dh = (vector_field_f @ control_gradient.unsqueeze(-1)).squeeze(-1)

        #  BNHI@BNI1=>BN1=>BNH
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        """
        输出必须是BNhidden_dim BNH
        """
        return tuple([dh,out])


def cdeint(dX_dt, z0, func, t, adjoint=False, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """

    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    vector_field = func(z0)

    if vector_field.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(vector_field.shape), tuple(vector_field.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if vector_field.size(-2) != z0.shape[-1]:
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-2), tuple(z0.shape),
                                   z0.shape.size(-1)))
    if vector_field.size(-1) != control_gradient.size(-1):
        raise ValueError("func did not return a tensor with the same number of input channels as dX_dt returned. "
                         "func returned shape {} (meaning {} channels), whilst dX_dt returned shape {} (meaning {}"
                         " channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-1), tuple(control_gradient.shape),
                                   control_gradient.size(-1)))
    if control_gradient.requires_grad and adjoint:
        raise ValueError("Gradients do not backpropagate through the control with adjoint=False. (This is a limitation "
                         "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)

    return out

def cdeint_gde(dX_dt, z0, func_f, func_g, t, adjoint=False, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    # only func_f() ???
    
    vector_field = func_f(z0)
    # vector_field_g = func_g(z0)
    if vector_field.shape[:-2] != z0.shape[:-1]:
        raise ValueError("func did not return a tensor with the same number of batch dimensions as z0. func returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch"
                         " dimensions)."
                         "".format(tuple(vector_field.shape), tuple(vector_field.shape[:-2]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))
    if vector_field.size(-2) != z0.shape[-1]:
        raise ValueError("func did not return a tensor with the same number of hidden channels as z0. func returned "
                         "shape {} (meaning {} channels), whilst z0 has shape {} (meaning {} channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-2), tuple(z0.shape),
                                   z0.shape.size(-1)))
    if vector_field.size(-1) != control_gradient.size(-1):
        raise ValueError("func did not return a tensor with the same number of input channels as dX_dt returned. "
                         "func returned shape {} (meaning {} channels), whilst dX_dt returned shape {} (meaning {}"
                         " channels)."
                         "".format(tuple(vector_field.shape), vector_field.size(-1), tuple(control_gradient.shape),
                                   control_gradient.size(-1)))

    if control_gradient.requires_grad and adjoint:
        raise ValueError("Gradients do not backpropagate through the control with adjoint=False. (This is a limitation "
                         "of the underlying torchdiffeq library.)")
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    # vector_field = VectorField(dX_dt=dX_dt, func=func_f)
    vector_field = VectorFieldGDE(dX_dt=dX_dt, func_f=func_f, func_g =func_g)
    
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)
    return out


 

def cdeint_gde_dev_2(dX_dt, h0, z0, func_f, func_g, t, node_embed, adjoint=False, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))

    # if control_gradient.requires_grad and adjoint:
    #     raise ValueError("Gradients do not backpropagate through the control with adjoint=False. (This is a limitation "
    #                      "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorFieldGDE_dev_2(dX_dt=dX_dt, func_f=func_f, func_g=func_g, node_embed= node_embed)
    init0 = (h0,z0) #相比与之前多了h0这个初值，func_f的初值为h0,func_g的为z0,在VectorFieldGDE_dev中要返回两个vector_field得到的隐藏状态 ,描述的是
    out = odeint(func=vector_field, y0=init0, t=t, **kwargs)
    return out[-1]

 

def cdeint_gde_dev(dX_dt, h0, z0, func_f, func_g, t, node_embed = None, adjoint=False, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """
    
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))

    # if control_gradient.requires_grad and adjoint:
    #     raise ValueError("Gradients do not backpropagate through the control with adjoint=False. (This is a limitation "
    #                      "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorFieldGDE_dev(dX_dt=dX_dt, func_f=func_f, func_g=func_g)
    init0 = (h0,z0)
    out = odeint(func=vector_field, y0=init0, t=t, **kwargs)
    return out[-1]




def cdeint_gde_dev(dX_dt, h0, z0, t, func_f = None, adjoint=False, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """
    t =  torch.linspace(0, 11,3).to(z0.device)
    # t = torch.tensor([6])
    # t = t.type_as(z0)
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))

    # if control_gradient.requires_grad and adjoint:
    #     raise ValueError("Gradients do not backpropagate through the control with adjoint=False. (This is a limitation "
    #                      "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    # vector_field = VectorFieldGDE_dev(dX_dt=dX_dt, func_f=func_f, func_g=func_f)
    # vector_field = VectorField(dX_dt=dX_dt, func=func_j, hid_dim=control_gradient.shape[-1], control_dim  = 4 )
    vector_field = VectorField(dX_dt=dX_dt, func=func_f, hid_dim=control_gradient.shape[-1], control_dim  = 4 )
    init0 = h0
    # init0 = (h0,z0)
    out = odeint(func=vector_field, y0=init0, t=t, **kwargs)
    return out[-1]




def cdeint_gde_dev_3(dX_dt, h0, z0, func_f, func_g, func_j, t, node_embed = None, adjoint=False, **kwargs):
    r"""Solves a system of controlled differential equations.

    Solves the controlled problem:
    ```
    z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s
    ```
    where z is a tensor of any shape, and X is some controlling signal.

    Arguments:
        dX_dt: The control. This should be a callable. It will be evaluated with a scalar tensor with values
            approximately in [t[0], t[-1]]. (In practice variable step size solvers will often go a little bit outside
            this range as well.) Then dX_dt should return a tensor of shape (..., input_channels), where input_channels
            is some number of channels and the '...' is some number of batch dimensions.
        z0: The initial state of the solution. It should have shape (..., hidden_channels), where '...' is some number
            of batch dimensions.
        func: Should be an instance of `torch.nn.Module`. Describes the vector field f(z). Will be called with a tensor
            z of shape (..., hidden_channels), and should return a tensor of shape
            (..., hidden_channels, input_channels), where hidden_channels and input_channels are integers defined by the
            `hidden_shape` and `dX_dt` arguments as above. The '...' corresponds to some number of batch dimensions.
        t: a one dimensional tensor describing the times to range of times to integrate over and output the results at.
            The initial time will be t[0] and the final time will be t[-1].
        adjoint: A boolean; whether to use the adjoint method to backpropagate.
        **kwargs: Any additional kwargs to pass to the odeint solver of torchdiffeq. Note that empirically, the solvers
            that seem to work best are dopri5, euler, midpoint, rk4. Avoid all three Adams methods.

    Returns:
        The value of each z_{t_i} of the solution to the CDE z_t = z_{t_0} + \int_0^t f(z_s)dX_s, where t_i = t[i]. This
        will be a tensor of shape (len(t), ..., hidden_channels).
    """
    
    control_gradient = dX_dt(torch.zeros(1, dtype=z0.dtype, device=z0.device))
    if control_gradient.shape[:-1] != z0.shape[:-1]:
        raise ValueError("dX_dt did not return a tensor with the same number of batch dimensions as z0. dX_dt returned "
                         "shape {} (meaning {} batch dimensions)), whilst z0 has shape {} (meaning {} batch "
                         "dimensions)."
                         "".format(tuple(control_gradient.shape), tuple(control_gradient.shape[:-1]), tuple(z0.shape),
                                   tuple(z0.shape[:-1])))

    # if control_gradient.requires_grad and adjoint:
    #     raise ValueError("Gradients do not backpropagate through the control with adjoint=False. (This is a limitation "
    #                      "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorFieldGDE_dev_3(dX_dt=dX_dt, func_f=func_f, func_g=func_g, func_j = func_j)
    init0 = (h0,z0)
    out = odeint(func=vector_field, y0=init0, t=t, **kwargs)
    return out[-1]

