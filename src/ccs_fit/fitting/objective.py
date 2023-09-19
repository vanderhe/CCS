# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#


"""This module constructs and solves the spline objective."""


import logging
from tqdm import tqdm
import itertools
import json
import bisect
from collections import OrderedDict
import numpy as np
from cvxopt import matrix, solvers
from scipy.linalg import block_diag
from math import isnan

logger = logging.getLogger(__name__)


class Objective:
    """Objective function for the ccs method."""

    def __init__(
        self,
        l_twb,
        l_one,
        sto,
        energy_ref,
        gen_params,
    ):
        """Generates Objective class object.

        Args:

            l_twb (list): list of Twobody class objects
            l_one (list): list of Onebody class objects
            sto (ndarray): array containing number of atoms of each type
            energy_ref (ndarray): reference energies
            ge_params (dict) : optionself.sto, s

        """

        self.l_twb = l_twb
        self.l_one = l_one
        self.sto = sto
        self.sto_full = self.sto

        # print(self.sto, self.sto_full)

        self.energy_ref = energy_ref

        self.ref = self.energy_ref

        for kk, vv in gen_params.items():
            setattr(self, kk, vv)

        self.cols_sto = self.sto.shape[1]
        self.np = len(l_twb)
        self.no = len(l_one)
        self.cparams = [self.l_twb[i].N for i in range(self.np)]
        self.ns = len(energy_ref)

        logger.debug(
            "The reference energy : \n %s \n Number of pairs:%s",
            self.energy_ref,
            self.np,
        )

    def reduce_stoichiometry(self):
        reduce = True
        n_redundant = 0
        while reduce:
            check = 0
            for ci in range(np.shape(self.sto)[1]):
                if np.linalg.matrix_rank(self.sto[:, 0 : ci + 1]) < (ci + 1):
                    print("    There is linear dependence in stochiometry matrix!")
                    print(
                        "    Removing onebody term: "
                        + self.l_one[ci + n_redundant].name
                    )
                    self.sto = np.delete(self.sto, ci, 1)
                    self.l_one[ci + n_redundant].epsilon_supported = False
                    check = 1
                    n_redundant += 1
                    break
            if check == 0:
                reduce = False

        assert self.sto.shape[1] == np.linalg.matrix_rank(
            self.sto
        ), "Linear dependence in stochiometry matrix"
        self.cols_sto = self.sto.shape[1]

    def solution(self):
        """Function to solve the objective with constraints."""

        # Reduce stoichiometry
        self.reduce_stoichiometry()

        self.mm = self.get_m()
        logger.debug("\n Shape of M matrix is : %s", self.mm.shape)

        pp = matrix(np.transpose(self.mm).dot(self.mm))
        eigvals = np.linalg.eigvals(pp)
        qq = -1 * matrix(np.transpose(self.mm).dot(self.ref))
        nswitch_list = self.list_iterator()
        obj = []

        logger.info("positive definite:%s", np.all((eigvals > 0)))
        logger.info("Condition number:%f", np.linalg.cond(pp))

        # Evaluting the fittnes
        mm_trimmed = self.mm
        mm_trimmed = np.delete(mm_trimmed, 0, 1)
        pp_trimmed = matrix(np.transpose(mm_trimmed).dot(mm_trimmed))

        if self.do_unconstrained_fit == "True":
            self.unconstrained_fit()

        for n_switch_id in tqdm(
            nswitch_list, desc="    Finding optimum switch", colour="#800080"
        ):
            [gg, aa] = self.get_g(n_switch_id)
            hh = np.zeros(gg.shape[0])
            bb = np.zeros(aa.shape[0])
            sol = self.solver(pp, qq, matrix(gg), matrix(hh), matrix(aa), matrix(bb))
            obj.append(float(self.eval_obj(sol["x"])))

        obj = np.asarray(obj)
        mse = np.min(obj)
        opt_sol_index = int(np.ravel(np.argwhere(obj == mse)[0]))

        best_switch_r = np.around(
            [
                nswitch_list[opt_sol_index][elem] * self.l_twb[elem].res
                + self.l_twb[elem].Rmin
                for elem in range(self.np)
            ],
            decimals=2,
        )
        elem_pairs = [self.l_twb[elem].name for elem in range(self.np)]

        best_switch_dict = {}
        for (
            i,
            elem_pair,
        ) in enumerate(elem_pairs):
            best_switch_dict[elem_pair] = best_switch_r[i]

        results_dict = {"rmse": mse**0.5, "best_switches": best_switch_dict}

        with open("rmse.json", "w") as outfile:
            json.dump(results_dict, outfile)

        print(
            f"    The best switch is {nswitch_list[opt_sol_index][:]} with rmse: {mse**0.5}, corresponding to distances of {best_switch_r} Å for element pairs {elem_pairs[:]}."
        )

        [g_opt, aa] = self.get_g(nswitch_list[opt_sol_index])
        bb = np.zeros(aa.shape[0])

        opt_sol = self.solver(pp, qq, matrix(g_opt), matrix(hh), matrix(aa), matrix(bb))

        xx = np.array(opt_sol["x"])
        self.assign_parameter_values(xx)

        self.model_energies = np.ravel(self.mm[0 : self.l_twb[0].Nconfs, :].dot(xx))

        self.write_error()

        x_unfolded = []
        for ii in range(self.np):
            self.l_twb[ii].get_spline_coeffs()
            self.l_twb[ii].get_expcoeffs()
            x_unfolded = np.hstack(
                (x_unfolded, np.array(self.l_twb[ii].curvatures).flatten())
            )
        for onb in self.l_one:
            if onb.epsilon_supported:
                x_unfolded = np.hstack((x_unfolded, np.array(onb.epsilon)))
            else:
                x_unfolded = np.hstack((x_unfolded, 0.0))
        xx = x_unfolded

        self.write_CCS_params()

        return self.model_energies, mse, xx


    @staticmethod
    def solver(pp, qq, gg, hh, aa, bb, maxiter=300, tol=(1e-10, 1e-10, 1e-10)):
        """The solver for the objective.

        Args:

            pp (matrix): P matrix as per standard Quadratic Programming(QP)
                notation.
            qq (matrix): q matrix as per standard QP notation.
            gg (matrix): G matrix as per standard QP notation.
            hh (matrix): h matrix as per standard QP notation
            aa (matrix): A matrix as per standard QP notation.
            bb (matrix): b matrix as per standard QP notation
            maxiter (int, optional): maximum iteration steps (default: 300).
            tol (tuple, optional): tolerance value of the solution
                (default: (1e-10, 1e-10, 1e-10)).

        Returns:

            sol (dict): dictionary containing solution details

        """

        solvers.options["show_progress"] = False
        solvers.options["maxiters"] = maxiter
        solvers.options["feastol"] = tol[0]
        solvers.options["abstol"] = tol[1]
        solvers.options["reltol"] = tol[2]

        if aa:
            sol = solvers.qp(pp, qq, gg, hh, aa, bb)
        else:
            sol = solvers.qp(pp, qq, gg, hh)

        return sol

    def eval_obj(self, xx):
        """Mean squared error function.

        Args:

            xx (ndarray): the solution for the objective

        Returns:

            float: mean square error

        """

        return np.format_float_scientific(
            np.sum((self.ref - (np.ravel(self.mm.dot(xx)))) ** 2) / self.ns, precision=4
        )

    def assign_parameter_values(self, xx):
        # Onebodies
        counter = -1
        for k in range(self.no):
            i = self.no - k - 1
            if self.l_one[i].epsilon_supported:
                counter += 1
                self.l_one[i].epsilon = float(xx[-1 - counter])
        # Two-bodies
        ind = 0
        for ii in range(self.np):
            self.l_twb[ii].curvatures = np.asarray(xx[ind : ind + self.cparams[ii]])
            ind = ind + self.cparams[ii]

    def list_iterator(self):
        """Iterates over the self.np attribute."""

        tmp = []
        for elem in range(self.np):
            if self.l_twb[elem].Swtype == "rep":
                tmp.append([self.l_twb[elem].N])
            if self.l_twb[elem].Swtype == "att":
                tmp.append([0])
            if self.l_twb[elem].Swtype == "sw":
                if self.l_twb[elem].search_mode.lower() == "full":
                    tmp.append(self.l_twb[elem].indices)
                elif self.l_twb[elem].search_mode.lower() == "range":
                    range_center = self.l_twb[elem].range_center
                    range_width = self.l_twb[elem].range_width
                    Rmin = self.l_twb[elem].Rmin
                    Rcut = self.l_twb[elem].Rcut
                    res = self.l_twb[elem].res
                    range_min = max(
                        0,
                        bisect.bisect_left(
                            self.l_twb[elem].rn, (range_center - range_width / 2)
                        ),
                    )
                    range_max = min(
                        self.l_twb[elem].N,
                        bisect.bisect_left(
                            self.l_twb[elem].rn, (range_center + range_width / 2)
                        ),
                    )
                    tmp.append(self.l_twb[elem].indices[range_min:range_max])
                    print(
                        "    Range search turned on for element pair {}; {} possible switch indices in range of {:.2f}-{:.2f} Å.".format(
                            self.l_twb[elem].name,
                            len(self.l_twb[elem].indices[range_min:range_max]),
                            max(
                                Rmin,
                                int((range_center - range_width - Rmin) / res) * res
                                + Rmin,
                            ),
                            min(
                                Rcut,
                                int((range_center + range_width - Rmin) / res) * res
                                + Rmin,
                            ),
                        )
                    )
                elif self.l_twb[elem].search_mode.lower() == "point":
                    search_indices = [
                        bisect.bisect_left(self.l_twb[elem].rn, search_point)
                        for search_point in self.l_twb[elem].search_points
                    ]
                    search_indices = np.unique(search_indices).tolist()
                    print(
                        "    Switch points located at {} to for element pair {} based on point search.".format(
                            "["
                            + ", ".join(
                                [
                                    "{:.2f}".format(self.l_twb[elem].rn[search_index])
                                    for search_index in search_indices
                                ]
                            )
                            + "] Å",
                            self.l_twb[elem].name,
                        )
                    )
                    tmp.append(
                        [
                            self.l_twb[elem].indices[search_index]
                            for search_index in search_indices
                        ]
                    )
                else:
                    raise SyntaxError(
                        'Error: search mode not recognized! Please use one of the following recognized options; ["full", "range", "point"]'
                    )

        n_list = list(itertools.product(*tmp))

        return n_list

    def get_m(self):
        """Returns the M matrix.

        Returns:

            ndarray: The M matrix.

        """

        # Add energy data
        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].vv)
        vv = np.hstack([*tmp])
        mm = np.hstack((vv, self.sto))

        return mm

    def get_g(self, n_switch):
        """Returns constraints matrix.

        Args:

            n_switch (int): switching point to change signs of curvatures.

        Returns:

            ndarray: returns G and A matrix

        """

        aa = np.zeros(0)
        tmp = []
        for elem in range(self.np):
            tmp.append(self.l_twb[elem].switch_const(n_switch[elem]))
        gg = block_diag(*tmp)

        gg = block_diag(gg, np.zeros_like(np.eye(self.cols_sto)))

        return gg, aa

    def write_error(self, fname="CCS_error.out"):
        """Prints the errors in a file.

        Args:

            mdl_eng (ndarray): Energy prediction values from splines.
            ref_eng (ndarray): Reference energy values.
            mse (float): Mean square error.
            fname (str, optional): Output filename (default: 'error.out').

        """
        header = "{:<15}{:<15}{:<15}{:<15}".format(
            "Reference", "Predicted", "Error", "#atoms"
        )
        error = abs(self.energy_ref - self.model_energies)
        maxerror = max(abs(error))
        mse = ((error) ** 2).mean()
        Natoms = self.l_one[0].stomat
        for i in range(1, self.no):
            Natoms = Natoms + self.l_one[i].stomat
        footer = "MSE = {:2.5E}\nMaxerror = {:2.5E}".format(mse, maxerror)
        np.savetxt(
            fname,
            np.transpose([self.energy_ref, self.model_energies, error, Natoms]),
            header=header,
            footer=footer,
            fmt="%-15.5f",
        )


    def write_CCS_params(self, fname="CCS_params.json"):
        CCS_params = OrderedDict()

        eps_params = OrderedDict()
        for k in range(self.no):
            if self.l_one[k].epsilon_supported:
                eps_params[self.l_one[k].name] = self.l_one[k].epsilon
        CCS_params["One_body"] = eps_params

        two_bodies_dict = OrderedDict()
        for k in range(self.np):
            two_body_dict = OrderedDict()
            two_body_dict["r_min"] = self.l_twb[k].rn[0]
            two_body_dict["r_cut"] = self.l_twb[k].Rcut
            two_body_dict["dr"] = self.l_twb[k].res
            r_values = list(np.array(self.l_twb[k].rn))
            two_body_dict["r"] = list(r_values)
            two_body_dict["exp_a"] = self.l_twb[k].expcoeffs[0]
            if not isnan(self.l_twb[k].expcoeffs[1]):
                two_body_dict["exp_b"] = self.l_twb[k].expcoeffs[1]
            else:
                two_body_dict["exp_b"] = 0
            if not isnan(self.l_twb[k].expcoeffs[2]):
                two_body_dict["exp_c"] = self.l_twb[k].expcoeffs[2]
            else:
                two_body_dict["exp_c"] = 0

            a_values = list(self.l_twb[k].splcoeffs[:, 0])
            a_values.append(0)
            two_body_dict["spl_a"] = a_values
            b_values = list(self.l_twb[k].splcoeffs[:, 1])

            b_values.append(0)
            two_body_dict["spl_b"] = b_values
            c_values = list(self.l_twb[k].splcoeffs[:, 2])
            c_values.append(0)
            two_body_dict["spl_c"] = c_values
            d_values = list(self.l_twb[k].splcoeffs[:, 3])
            d_values.append(0)
            two_body_dict["spl_d"] = d_values
            two_bodies_dict[self.l_twb[k].name] = two_body_dict

        CCS_params["Two_body"] = two_bodies_dict
        with open(fname, "w") as f:
            json.dump(CCS_params, f, indent=8)

    def unconstrained_fit(self):
        # Solving unconstrained problem
        xx = np.linalg.lstsq(self.mm, self.ref, rcond=None)
        xx = xx[0]
        print(
            "    MSE of unconstrained problem is: ",
            ((self.mm.dot(xx) - self.ref) ** 2).mean(),
        )
        xx = xx.reshape(len(xx), 1)
        self.assign_parameter_values(xx)

        self.model_energies = np.ravel(self.mm[0 : self.l_twb[0].Nconfs, :].dot(xx))
        self.write_error(fname="UNC_error.out")

        x_unfolded = []
        for ii in range(self.np):
            self.l_twb[ii].get_spline_coeffs()
            self.l_twb[ii].get_expcoeffs()
            x_unfolded = np.hstack(
                (x_unfolded, np.array(self.l_twb[ii].curvatures).flatten())
            )
        for onb in self.l_one:
            if onb.epsilon_supported:
                x_unfolded = np.hstack((x_unfolded, np.array(onb.epsilon)))
            else:
                x_unfolded = np.hstack((x_unfolded, 0.0))
        xx = x_unfolded

        self.write_CCS_params(fname="UNC_params.json")
