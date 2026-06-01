import torch
import torch.nn as nn


class BoundHardTanh(nn.Hardtanh):
    def __init__(self):
        super(BoundHardTanh, self).__init__()

    @staticmethod
    def convert(act_layer):
        r"""Convert a HardTanh layer to BoundHardTanh layer

        Args:
            act_layer (nn.HardTanh): The HardTanh layer object to be converted.

        Returns:
            l (BoundHardTanh): The converted layer object.
        """
        # Return the converted HardTanH
        l = BoundHardTanh()
        return l

    def boundpropogate(self, last_uA, last_lA, start_node=None):
        """
        Propagate upper and lower linear bounds through the HardTanh activation function
        based on pre-activation bounds.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.

        """
        # These are preactivation bounds that will be used for form the linear relaxation.
        preact_lb = self.lower_l
        preact_ub = self.upper_u
        preact_ub = torch.max(preact_ub, preact_lb + 1e-8)

        # Implement the linear lower and upper bounds for HardTanH you derived in Problem 4.2.
        """
        Linear bounds for HardTanH activation function:
        Case 1: l <= u <= -1: d = 0, b = -1
        Case 2: 1 <= l <= u: d = 0, b = 1
        Case 3: -1 <= l <= u <= 1: d= 1, b = 0
        Case 4: l <= -1 <= u <= 1: d = (u+1)/(u-l), ub = -1 - l * d, lb = -1 + d
        Case 5: -1 <= l <= 1 <= u: d = (1-l)/(u-l), ub = 1 - d, lb = 1 - u * d
        Case 6: l <= -1 <= 1 <= u: if |l| > |u|, same as case 4 with u = 1, else same as case 5 with l = -1
        """
        # masks
        case1 = preact_ub <= -1
        case2 = ~case1 & (preact_lb >= 1)
        case3 = ~case1 & ~case2 & (preact_lb >= -1) & (preact_ub <= 1)
        case4 = ~case1 & ~case2 & ~case3 & (preact_ub <= 1)
        case5 = ~case1 & ~case2 & ~case3 & (preact_lb >= -1) & (preact_ub > 1)
        case6 = ~case1 & ~case2 & ~case3 & (preact_lb < -1) & (preact_ub > 1)

        b_u = torch.zeros_like(preact_lb)
        b_l = torch.zeros_like(preact_lb)
        d = torch.zeros_like(preact_lb)

        # Case 1 & Case 2
        d = torch.where(case1 | case2, torch.zeros_like(d), d)
        b_u = torch.where(case1, -torch.ones_like(b_u), b_u)
        b_l = torch.where(case1, -torch.ones_like(b_l), b_l)
        b_u = torch.where(case2, torch.ones_like(b_u), b_u)
        b_l = torch.where(case2, torch.ones_like(b_l), b_l)

        # Case 3
        d = torch.where(case3, torch.ones_like(d), d)
        b_u = torch.where(case3, torch.zeros_like(b_u), b_u)
        b_l = torch.where(case3, torch.zeros_like(b_l), b_l)

        # Case 4
        d4 = (preact_ub + 1) / (preact_ub - preact_lb)
        d = torch.where(case4, d4, d)
        b_u = torch.where(case4, -1 - d4 * preact_lb, b_u)
        b_l = torch.where(case4, -1 + d4, b_l)

        # Case 5
        d5 = (1 - preact_lb) / (preact_ub - preact_lb)
        d = torch.where(case5, d5, d)
        b_u = torch.where(case5, 1 - d5, b_u)
        b_l = torch.where(case5, 1 - preact_ub * d5, b_l)

        # Case 6
        abs_l_greater = (-preact_lb) > preact_ub
        d6a = 2 / (1 - preact_lb)
        b_u6a = -1 - d6a * preact_lb
        b_l6a = -1 + d6a
        d6b = 2 / (preact_ub + 1)
        b_u6b = 1 - d6b
        b_l6b = 1 - preact_ub * d6b

        d = torch.where(case6 & abs_l_greater, d6a, d)
        d = torch.where(case6 & ~abs_l_greater, d6b, d)
        b_u = torch.where(case6 & abs_l_greater, b_u6a, b_u)
        b_u = torch.where(case6 & ~abs_l_greater, b_u6b, b_u)
        b_l = torch.where(case6 & abs_l_greater, b_l6a, b_l)
        b_l = torch.where(case6 & ~abs_l_greater, b_l6b, b_l)

        uA = lA = None
        ubias = lbias = 0

        if last_uA is not None:
            uA = d * last_uA
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            mult_pos_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            mult_neg_uA = neg_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_pos_uA.matmul(b_u.view(b_u.size(0), -1, 1)).squeeze(-1)
            ubias += mult_neg_uA.matmul(b_l.view(b_l.size(0), -1, 1)).squeeze(-1)
        if last_lA is not None:
            lA = d * last_lA
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            mult_pos_lA = pos_lA.view(last_lA.size(0), last_lA.size(1), -1)
            mult_neg_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_pos_lA.matmul(b_l.view(b_l.size(0), -1, 1)).squeeze(-1)
            lbias += mult_neg_lA.matmul(b_u.view(b_u.size(0), -1, 1)).squeeze(-1)

        return uA, ubias, lA, lbias
