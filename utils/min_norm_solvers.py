# Credits to Ozan Sener
# https://github.com/intel-isl/MultiObjectiveOptimization

import numpy as np
import torch


class MGDASolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs: list, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0
        for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        sol = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.dot(vecs[i][k].view(-1),
                                                 vecs[j][k].view(-1)).detach()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.dot(vecs[i][k].view(-1),
                                                 vecs[i][k].view(-1)).detach()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(vecs[j][k].view(-1),
                                                 vecs[j][k].view(-1)).detach()
                c, d = MGDASolver._min_norm_element_from2(dps[(i, i)],
                                                          dps[(i, j)],
                                                          dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MGDASolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs: list):
        """
        Given a list of vectors (vecs), this method finds the minimum norm
        element in the convex hull as min |u|_2 st. u = \sum c_i vecs[i]
        and \sum c_i = 1. It is quite geometric, and the main idea is the
        fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution
        lies in (0, d_{i,j})Hence, we find the best 2-task solution , and
        then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MGDASolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MGDASolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MGDASolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MGDASolver._min_norm_element_from2(v1v1.item(),
                                                        v1v2.item(),
                                                        v2v2.item())
            # try:
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            # except AttributeError:
            #     print(sol_vec)
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MGDASolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    @staticmethod
    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm
        element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if
        d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies
        in (0, d_{i,j})Hence, we find the best 2-task solution, and then
        run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MGDASolver._min_norm_2d(vecs, dps)
        print(f"🛑 DEBUG: _min_norm_2d() returned {init_sol}")
        print(f"🛑 DEBUG: vecs had {len(vecs)} elements.")

        # 🛠️ Fix: Ensure init_sol is valid before using it
        if not isinstance(init_sol, tuple) or len(init_sol) != 3:
            print("⚠️ WARNING: _min_norm_2d() returned an invalid result. Defaulting to equal weights.")
            return np.ones(len(vecs)) / len(vecs), 1.0  # Return equal scaling factors

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MGDASolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MGDASolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MGDASolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    @classmethod
    def get_scales(cls, grads, losses, normalization_type, tasks):
        scale = {}
        gn = gradient_normalizers(grads, losses, normalization_type)
        for t in tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / (gn[t] + 1e-5)
        sol, min_norm = cls.find_min_norm_element([grads[t] for t in tasks])
        for zi, t in enumerate(tasks):
            scale[t] = float(sol[zi])

        return scale


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = torch.sqrt(
                torch.stack([gr.pow(2).sum().data for gr in grads[t]]).sum())
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = min(losses[t].mean(), 10.0)
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = min(losses[t].mean() * torch.sqrt(
                torch.stack([gr.pow(2).sum().data for gr in grads[t]]).sum()),
                        10)

    elif normalization_type == 'none' or normalization_type == 'eq':
        for t in grads:
            gn[t] = 1.0
    else:
        raise ValueError('ERROR: Invalid Normalization Type')
    return gn
