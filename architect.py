""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch


class Architect:
    """ Compute gradients of alphas """

    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        logists, (_, _) = self.net(trn_X)  # L_trn(w)
        loss = self.net.criterion(logists, trn_y)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(
                self.net.weights(), self.v_net.weights(), gradients
            ):
                m = (
                    w_optim.state[w].get("momentum_buffer", 0.0)
                    * self.w_momentum
                )
                vw.copy_(w - xi * (m + g + self.w_weight_decay * w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim, f_loss_func):
        """Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step

            w* - best weights
            w' = w - e*grad L train(w,alpha)
            grad/alpha_val Lval(w*,alphaa) =  grad/alpha L(w',alpha) - e*grad^2_alpha_w Ltrain(w, alpha)*grad w Lval(w', alpha)
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss

        logists, (flops, mem) = self.v_net(trn_X)  # L_trn(w)
        loss = self.v_net.criterion(logists, trn_y) + f_loss_func(flops)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[: len(v_alphas)]
        dw = v_grads[len(v_alphas) :]

        hessian = self.compute_hessian(dw, trn_X, trn_y)
        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi * h
        with torch.no_grad():
            for alpha, da in zip(self.net.alphas(), dalpha):
                alpha.grad = da

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        logists, (_, _) = self.net(trn_X)  # L_trn(w)
        loss = self.net.criterion(logists, trn_y)

        dalpha_pos = torch.autograd.grad(
            loss, self.net.alphas()
        )  # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2.0 * eps * d

        logists, (_, _) = self.net(trn_X)  # L_trn(w)
        loss = self.net.criterion(logists, trn_y)

        dalpha_neg = torch.autograd.grad(
            loss, self.net.alphas()
        )  # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [
            (p - n) / (2.0 * eps) for p, n in zip(dalpha_pos, dalpha_neg)
        ]
        return hessian
