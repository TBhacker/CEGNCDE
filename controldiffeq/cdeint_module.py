import torch
import torchdiffeq


class VectorField(torch.nn.Module):
    def __init__(self, dX_dt, func):
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

    def __call__(self, t, z):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        vector_field = self.func(z)
        # import pdb; pdb.set_trace()
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out

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
        # import pdb; pdb.set_trace()
        
        vector_field_f = self.func_f(z)
        vector_field_g = self.func_g(z)
        
        vector_field_fg = torch.mul(vector_field_g, vector_field_f)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # out = (vector_field_g @ control_gradient.unsqueeze(-1)).squeeze(-1)
        return out

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
    ## how to get h and z together?
    def __call__(self, t, hz):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t)#  torch.Size([64, 307, 2])
        # vector_field is of shape (..., hidden_channels, input_channels)

        h = hz[0] # h: torch.Size([64, 207, 32])
        z = hz[1] # z: torch.Size([64, 207, 32])
        vector_field_f = self.func_f(h) # vector_field_f: torch.Size([64, 307, 128, 2])
        vector_field_g = self.func_g(z) # vector_field_g: torch.Size([64, 307,128, 128])
        #这里的最后维度2和128的意义？？
        #2 是input的维度

        # vector_field_fg = torch.mul(vector_field_g, vector_field_f) # vector_field_fg: torch.Size([64, 307, 128, 2])
        vector_field_fg = torch.matmul(vector_field_g, vector_field_f)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        dh = (vector_field_f @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # dh: torch.Size([64, 307, 128])
        # out: torch.Size([64, 307, 128])
        return tuple([dh,out])

class VectorFieldGDE_dev_test(torch.nn.Module):
    def __init__(self, dX_dt, func_f, func_g,input_proj,tod_embedding, dow_embedding, adaptive_embedding):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func_f: As cdeint.
            func_g: As cdeint.
        """
        
        super(VectorFieldGDE_dev_test, self).__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.func_f = func_f
        self.func_g = func_g
        self.input_proj = input_proj
        self.tod_embedding = tod_embedding
        self.dow_embedding = dow_embedding
    ## how to get h and z together?
    def __call__(self, t, hz):
        # control_gradient is of shape (..., input_channels)
        control_gradient = self.dX_dt(t)
        # vector_field is of shape (..., hidden_channels, input_channels)
        # 
        # control_gradient = self.input_proj(control_gradient)
        tod = control_gradient[..., 1]
        tod_max = tod.max()
        tod_min = tod.min()
        dow = control_gradient[..., 2]
        x = control_gradient[...,:1]
        x = self.input_proj(x)
        features = [x]
        tod_emb = self.tod_embedding((tod * 288).long())  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
        features.append(tod_emb)
        dow_emb = self.dow_embedding((dow).long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
        features.append(dow_emb)
        # adp_emb = self.adaptive_embedding.expand(size=(batch_size, *self.adaptive_embedding.shape))
        # features.append(adp_emb)
        control_gradient = torch.cat(features, dim=-1)
        h = hz[0] # h: torch.Size([64, 207, 32])
        z = hz[1] # z: torch.Size([64, 207, 32])
        vector_field_f = self.func_f(h) # vector_field_f: torch.Size([64, 307, 32, 2])
        vector_field_g = self.func_g(z,vector_field_f) # vector_field_g: torch.Size([64, 307, 32, 32])

        #vector_field_fg = torch.mul(vector_field_g, vector_field_f) # vector_field_fg: torch.Size([64, 307, 32, 2])
        vector_field_fg = torch.matmul(vector_field_g, vector_field_f)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        dh = (vector_field_f @ control_gradient.unsqueeze(-1)).squeeze(-1)
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(-1)
        #out = (vector_field_g @ control_gradient.unsqueeze(-1)).squeeze(-1)
        # dh: torch.Size([64, 207, 32])
        # out: torch.Size([64, 207, 32])
        return tuple([dh,out])


def cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs):
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
        raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
                         "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorField(dX_dt=dX_dt, func=func)
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)

    return out

def cdeint_gde(dX_dt, z0, func_f, func_g, t, adjoint=True, **kwargs):
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
    # import pdb; pdb.set_trace()
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
        raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
                         "of the underlying torchdiffeq library.)")
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    # vector_field = VectorField(dX_dt=dX_dt, func=func_f)
    vector_field = VectorFieldGDE(dX_dt=dX_dt, func_f=func_f, func_g =func_g)
    
    out = odeint(func=vector_field, y0=z0, t=t, **kwargs)
    return out
###################
def cdeint_gde_dev_test(dX_dt, h0, z0, func_f, func_g, input_proj, tod_embedding, dow_embedding, adaptive_embedding, t, adjoint=True, **kwargs):
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
    
    ######
    

    if control_gradient.requires_grad and adjoint:
        raise ValueError("Gradients do not backpropagate through the control with adjoint=True. (This is a limitation "
                         "of the underlying torchdiffeq library.)")

    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    vector_field = VectorFieldGDE_dev_test(dX_dt=dX_dt, func_f=func_f, func_g=func_g,input_proj=input_proj,tod_embedding = tod_embedding, 
    dow_embedding = dow_embedding, adaptive_embedding = adaptive_embedding)
    init0 = (h0,z0)
    out = odeint(func=vector_field, y0=init0, t=t, **kwargs)
    return out[-1]
