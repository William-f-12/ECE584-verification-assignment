import torch
import torch.nn as nn


class BoundHardTanh(nn.Hardtanh):
    def __init__(self):
        super(BoundHardTanh, self).__init__()
        self.alpha = None

    @staticmethod
    def convert(act_layer):
        return BoundHardTanh()

    @staticmethod
    def compute_optimal_slope(l, u):
        r"""Compute the optimal fixed slope for each neuron using the analytic formula."""
        u_safe = torch.max(u, l + 1e-8)

        case1 = u_safe <= -1
        case2 = ~case1 & (l >= 1)
        case3 = ~case1 & ~case2 & (l >= -1) & (u_safe <= 1)
        case4 = ~case1 & ~case2 & ~case3 & (u_safe <= 1)
        case5 = ~case1 & ~case2 & ~case3 & (l >= -1) & (u_safe > 1)
        case6 = ~case1 & ~case2 & ~case3 & (l < -1) & (u_safe > 1)

        d = torch.zeros_like(l)
        d = torch.where(case1 | case2, torch.zeros_like(d), d)
        d = torch.where(case3, torch.ones_like(d), d)
        d = torch.where(case4, (u_safe + 1) / (u_safe - l), d)
        d = torch.where(case5, (1 - l) / (u_safe - l), d)

        abs_l_greater = (-l) > u_safe
        d6a = 2 / (1 - l)
        d6b = 2 / (u_safe + 1)
        d = torch.where(case6 & abs_l_greater, d6a, d)
        d = torch.where(case6 & ~abs_l_greater, d6b, d)

        return d

    def boundpropogate(self, last_uA, last_lA, start_node=None, alpha=None):
        preact_lb = self.lower_l
        preact_ub = self.upper_u
        preact_ub = torch.max(preact_ub, preact_lb + 1e-8)

        case1 = preact_ub <= -1
        case2 = ~case1 & (preact_lb >= 1)
        case3 = ~case1 & ~case2 & (preact_lb >= -1) & (preact_ub <= 1)

        ub = torch.zeros_like(preact_lb)
        lb = torch.zeros_like(preact_lb)
        d = torch.zeros_like(preact_lb)

        # Case 1 & Case 2
        d = torch.where(case1 | case2, torch.zeros_like(d), d)
        ub = torch.where(case1, -torch.ones_like(ub), ub)
        lb = torch.where(case1, -torch.ones_like(lb), lb)
        ub = torch.where(case2, torch.ones_like(ub), ub)
        lb = torch.where(case2, torch.ones_like(lb), lb)

        # Case 3
        d = torch.where(case3, torch.ones_like(d), d)
        ub = torch.where(case3, torch.zeros_like(ub), ub)
        lb = torch.where(case3, torch.zeros_like(lb), lb)

        # α-CROWN: override unstable neurons with learnable alpha
        unstable = ~(case1 | case2 | case3)
        if unstable.any():
            if alpha is not None:
                d_unstable = torch.sigmoid(alpha)
            else:
                d_unstable = self.compute_optimal_slope(preact_lb, preact_ub)
            f_l = preact_lb.clamp(-1, 1)
            f_u = preact_ub.clamp(-1, 1)

            # Only include candidate points within [preact_lb, preact_ub]
            # Candidates outside the interval are set to -inf (max) / +inf (min)
            include_m1 = (preact_lb <= -1) & (preact_ub >= -1)
            include_p1 = (preact_lb <= 1) & (preact_ub >= 1)

            candidates_max = torch.stack([
                f_l - d_unstable * preact_lb,
                f_u - d_unstable * preact_ub,
                torch.where(include_m1, -1.0 - d_unstable * (-1.0),
                            torch.full_like(preact_lb, float('-inf'))),
                torch.where(include_p1, 1.0 - d_unstable * 1.0,
                            torch.full_like(preact_lb, float('-inf'))),
            ])
            ub_unstable = candidates_max.max(dim=0).values

            candidates_min = torch.stack([
                f_l - d_unstable * preact_lb,
                f_u - d_unstable * preact_ub,
                torch.where(include_m1, -1.0 - d_unstable * (-1.0),
                            torch.full_like(preact_lb, float('inf'))),
                torch.where(include_p1, 1.0 - d_unstable * 1.0,
                            torch.full_like(preact_lb, float('inf'))),
            ])
            lb_unstable = candidates_min.min(dim=0).values

            d = torch.where(unstable, d_unstable, d)
            ub = torch.where(unstable, ub_unstable, ub)
            lb = torch.where(unstable, lb_unstable, lb)

        uA = lA = None
        ubias = lbias = 0

        if last_uA is not None:
            uA = d * last_uA
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            mult_pos_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            mult_neg_uA = neg_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_pos_uA.matmul(ub.view(ub.size(0), -1, 1)).squeeze(-1)
            ubias += mult_neg_uA.matmul(lb.view(lb.size(0), -1, 1)).squeeze(-1)
        if last_lA is not None:
            lA = d * last_lA
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            mult_pos_lA = pos_lA.view(last_lA.size(0), last_lA.size(1), -1)
            mult_neg_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_pos_lA.matmul(lb.view(lb.size(0), -1, 1)).squeeze(-1)
            lbias += mult_neg_lA.matmul(ub.view(ub.size(0), -1, 1)).squeeze(-1)

        return uA, ubias, lA, lbias
