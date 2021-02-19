import numpy as np

from .utils import upper_to_full


class ADMM:
    def __init__(self, lamb, n_blocks, block_size, rho, S, rho_update_func=None):
        self.lamb = lamb
        self.n_blocks = n_blocks
        self.block_size = block_size
        prob_size = n_blocks * block_size
        self.length = int(prob_size * (prob_size + 1) / 2)
        self.x = np.zeros(self.length)
        self.z = np.zeros(self.length)
        self.u = np.zeros(self.length)
        self.rho = float(rho)
        self.S = S
        self.status = 'initialized'
        self.rho_update_func = rho_update_func

    @staticmethod
    def ij_to_symmetric(i, j, size):
        return (size * (size + 1)) / 2 - (size - i) * (size - i + 1) / 2 + j - i

    @staticmethod
    def prox_logdet(S, A, eta):
        d, q = np.linalg.eigh(eta * A - S)
        q = np.matrix(q)
        X_var = (1 / (2 * float(eta))) * q * (np.diag(d + np.sqrt(np.square(d) + (4 * eta) * np.ones(d.shape)))) * q.T
        x_var = X_var[np.triu_indices(S.shape[1])]  # extract upper triangular part as update variable
        return np.matrix(x_var).T

    def update_x(self):
        a = self.z - self.u
        A = upper_to_full(a, eps=None)
        eta = self.rho
        x_update = self.prox_logdet(self.S, A, eta)
        self.x = np.array(x_update).T.reshape(-1)

    def update_z(self, index_penalty=1):
        a = self.x + self.u
        prob_size = self.n_blocks * self.block_size
        z_update = np.zeros(self.length)

        # TODO: can we parallelize these?
        for i in range(self.n_blocks):
            elems = self.n_blocks if i == 0 else (2 * self.n_blocks - 2 * i) / 2  # i=0 is diagonal
            for j in range(self.block_size):
                start_point = j if i == 0 else 0
                for k in range(start_point, self.block_size):
                    loc_list = [((l + i) * self.block_size + j, l * self.block_size + k) for l in range(int(elems))]
                    if i == 0:
                        lam_sum = sum(self.lamb[loc1, loc2] for (loc1, loc2) in loc_list)
                        indices = [self.ij_to_symmetric(loc1, loc2, prob_size) for (loc1, loc2) in loc_list]
                    else:
                        lam_sum = sum(self.lamb[loc2, loc1] for (loc1, loc2) in loc_list)
                        indices = [self.ij_to_symmetric(loc2, loc1, prob_size) for (loc1, loc2) in loc_list]
                    point_sum = sum(a[int(index)] for index in indices)
                    rho_point_sum = self.rho * point_sum

                    # Calculate soft threshold
                    ans = 0
                    # If answer is positive
                    if rho_point_sum > lam_sum:
                        ans = max((rho_point_sum - lam_sum) / (self.rho * elems), 0)
                    elif rho_point_sum < -1 * lam_sum:
                        ans = min((rho_point_sum + lam_sum) / (self.rho * elems), 0)

                    for index in indices:
                        z_update[int(index)] = ans
        self.z = z_update

    def update_u(self):
        u_update = self.u + self.x - self.z
        self.u = u_update

    def check_convergence(self, z_old, e_abs, e_rel, verbose):
        # Returns True if convergence criteria have been satisfied
        # eps_abs = eps_rel = 0.01
        # r = x - z
        # s = rho * (z - z_old)
        # e_pri = sqrt(length) * e_abs + e_rel * max(||x||, ||z||)
        # e_dual = sqrt(length) * e_abs + e_rel * ||rho * u||
        # Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
        # Returns (boolean shouldStop, primal residual value, primal threshold,
        #          dual residual value, dual threshold)
        norm = np.linalg.norm
        r = self.x - self.z
        s = self.rho * (self.z - z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = np.sqrt(self.length) * e_abs + e_rel * max(norm(self.x), norm(self.z)) + .0001
        e_dual = np.sqrt(self.length) * e_abs + e_rel * norm(self.rho * self.u) + .0001
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        if verbose:
            # Debugging information to print(convergence criteria values)
            print('\tr:', res_pri)
            print('\te_pri:', e_pri)
            print('\ts:', res_dual)
            print('\te_dual:', e_dual)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return stop, res_pri, e_pri, res_dual, e_dual

    # solve
    def run(self, max_iters, eps_abs, eps_rel, verbose):
        num_iterations = 0
        self.status = 'Incomplete: max iterations reached'
        for i in range(max_iters):
            z_old = np.copy(self.z)
            self.update_x()
            self.update_z()
            self.update_u()
            if i != 0:
                stop, res_pri, e_pri, res_dual, e_dual = self.check_convergence(z_old, eps_abs, eps_rel, verbose)
                if stop:
                    self.status = 'Optimal'
                    break

                if self.rho_update_func:
                    new_rho = self.rho_update_func(self.rho, res_pri, e_pri, res_dual, e_dual)
                else:
                    new_rho = self.rho

                scale = self.rho / new_rho
                self.rho = new_rho
                self.u = scale * self.u
            if verbose:
                # Debugging information prints current iteration #
                print('Iteration %d' % i)

        return self
