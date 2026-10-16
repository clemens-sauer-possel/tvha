"""Collection of plots for tVHA.

The plots help investigate tVHA and tune it.
"""

import ast
import itertools
import json
import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit_aer.noise as noise
from computation_cache.computation_file_cache import ComputationFileCache
from matplotlib.ticker import MaxNLocator
from pyscf import gto, scf
from qiskit.circuit.library import BlueprintCircuit
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver
from qiskit_machine_learning.optimizers import SBPLX
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
from qiskit_nature.second_q.circuit.library import UCC, HartreeFock
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from ruamel.yaml import YAML
from tqdm import tqdm

from tvha.efficientsu2_hartreefock import EfficientSU2_HartreeFock
from tvha.fermionic_operator import FermionicOp
from tvha.tvha import VariationalHamiltonianAnsatz

logger = logging.getLogger(__name__)

# agg is a non-GUI backend so plt.show() does not work with it
plt.switch_backend("agg")  # quick fix for some runtime error in tkinter

_color_scheme_fhg = {
    "Weiß": (255, 255, 255),
    "Schwarz": (0, 0, 0),
    "Orange": (245, 130, 32),
    "Dunkelgrün": (23, 156, 125),  # Akzent 1
    "Dunkelblaugrün (mittel)": (0, 91, 127),  # Akzent 2
    "Eisblau": (166, 187, 200),  # Akzent 3
    "Dunkelblaugrün (hell)": (0, 133, 152),  # Akzent 4
    "Türkis": (57, 193, 205),  # Akzent 5
    "Gelbgrün": (178, 210, 53),  # Akzent 6
    "Aquamarin (dunkel)": (51, 124, 153),
    "Aquamarin (hell)": (102, 157, 178),
    "Himmelblau (dunkel)": (153, 189, 204),
    "Himmelblau (mittel)": (204, 222, 229),
    "Himmelblau (hell)": (229, 238, 242),
    "Dunkelblaugrün (dunkel)": (28, 63, 82),
    "Gelbbraun": (211, 199, 174),
    "Gold": (253, 185, 19),
    "Pflaume (hell)": (187, 0, 86),
    "Pflaume (dunkel)": (124, 21, 77),
}
# Rescale RGB values from [0, 255] to [0, 1]
color_scheme_fhg = {
    name: tuple(i / 255 for i in rgb_tuple) for name, rgb_tuple in _color_scheme_fhg.items()
}
color_circle = [
    color_scheme_fhg["Dunkelgrün"],
    color_scheme_fhg["Dunkelblaugrün (mittel)"],
    color_scheme_fhg["Eisblau"],
    color_scheme_fhg["Dunkelblaugrün (hell)"],
    color_scheme_fhg["Türkis"],
    color_scheme_fhg["Gelbgrün"],
]
colors_tvha = [
    x[0] for x in sorted(zip(color_circle, [0, 2, 1, 3, 4, 5], strict=False), key=lambda x: x[1])
]
colors_ucc = [color_circle[3], color_circle[5]]
color_hea = color_scheme_fhg["Orange"]
color_hf = color_scheme_fhg["Himmelblau (mittel)"]
color_fci = color_scheme_fhg["Gold"]


class VHAPlots(ComputationFileCache):
    """Collection of plots for closer evaluation of tVHA method."""

    def __init__(
        self,
        output_path: Path,
        molecule_name: str,
        problem: ElectronicStructureProblem,
        mapper: FermionicMapper | None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initializes all needed variables for plotting of properties of tVHA."""
        self._epsilon = 1e-11  # tolerance for floats to be considered equal
        output_path.mkdir(exist_ok=True)
        self.output_path = output_path.resolve()
        super().__init__(
            output_file=Path(self.output_path).joinpath(f"energies_simulator_{molecule_name}.csv"),
            **kwargs,
        )
        self.molecule_name = molecule_name
        self.problem = problem
        self.mapper = mapper or JordanWignerMapper()

        # Dummy VHA in order to easily access the final threshold gamma
        self._vha_dummy = VariationalHamiltonianAnsatz(problem=self.problem, mapper=self.mapper)

        self._init_alternative_ansatze()

        self.numerical_energies = self.get_numerical_energy()
        self._energy_data = self.data

    def _init_alternative_ansatze(self) -> None:
        ansatzes = {}

        uccsd = UCC(
            excitations=[1, 2],
            num_spatial_orbitals=self.problem.num_spatial_orbitals,
            num_particles=self.problem.num_particles,
            qubit_mapper=self.mapper,
            initial_state=HartreeFock(
                num_spatial_orbitals=self.problem.num_spatial_orbitals,
                num_particles=self.problem.num_particles,
                qubit_mapper=self.mapper,
            ),
        )
        initial_point_sd = HFInitialPoint()
        initial_point_sd.ansatz = uccsd
        initial_point_sd.problem = self.problem
        uccsd.get_initial_point = initial_point_sd.to_numpy_array
        ansatzes["UCCSD"] = uccsd

        uccsdt = UCC(
            excitations=[1, 2, 3],
            num_spatial_orbitals=self.problem.num_spatial_orbitals,
            num_particles=self.problem.num_particles,
            qubit_mapper=self.mapper,
            initial_state=HartreeFock(
                num_spatial_orbitals=self.problem.num_spatial_orbitals,
                num_particles=self.problem.num_particles,
                qubit_mapper=self.mapper,
            ),
        )
        initial_point_sdt = HFInitialPoint()
        initial_point_sdt.ansatz = uccsdt
        initial_point_sdt.problem = self.problem
        uccsdt.get_initial_point = initial_point_sdt.to_numpy_array
        ansatzes["UCCSDT"] = uccsdt

        hea = EfficientSU2_HartreeFock(
            num_spatial_orbitals=self.problem.num_spatial_orbitals,
            num_particles=self.problem.num_particles,
            mapper=self.mapper,
            num_qubits=len(self.mapper.map(self.problem.second_q_ops()[0])[0].paulis[0]),
            entanglement="reverse_linear",
            reps=3,
        )
        hea.get_initial_point = lambda: hea.preferred_init_points
        ansatzes["HEA"] = hea

        self._alternative_ansatze = ansatzes

    def _calculate_statevector_energy(
        self,
        ansatz_name: Literal["tVHA", "UCCSD", "UCCSDT", "HEA"] = "tVHA",
        trotter_steps: int = 1,
        threshold_gamma: float = 1.0,
        max_evals: int = 1000,
        cx_error_prob: float = 0,
    ) -> dict[str:float, list[float], BlueprintCircuit]:
        """Gets the ground state energy of statevector simulator calculation of tVHA.

        Args:
            ansatz_name: Only use this, if you explicitly need another ansatz than tVHA.
                If this arg is used, threshold_gamma and trotter_steps are silently discarded.
            trotter_steps: Number of steps for Trotter of the (adiabatic)
                time evolution.
            threshold_gamma: The truncation threshold to use for building the tVHA ansatz.
            max_evals: Maximum number of function evaluations of the optimization algorithm (SBPLX).
            cx_error_prob: The CNOT gate depolarizing error of a simple noise model.
                Choose '0' for statevector calculation.

        Returns: dict of energy, optimal_parameters, and ansatz
            energy: electronic part of groundstate energy of statevector calculation.
            optimal_parameters: ansatz parameters after optimization loop.
            ansatz: (VHA) ansatz used for the calculation
        """
        if cx_error_prob == 0:
            estimator = StatevectorEstimator()
        else:
            error_2q = noise.depolarizing_error(cx_error_prob, 2)
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])
            estimator = AerEstimator(
                backend_options={"noise_model": noise_model},
                approximation=True,  # TODO: find out which combination of options (approximation, shots) is suitable
                # run_options={"shots": 1e4},  # TODO: find out appropriate number of shots
            )
        if ansatz_name in ("tVHA", "VHA", None):
            ansatz = VariationalHamiltonianAnsatz(
                problem=self.problem,
                trotter_steps=trotter_steps,
                threshold_gamma=threshold_gamma,
                mapper=self.mapper,
            )
        else:
            ansatz = self._alternative_ansatze[ansatz_name]
        vqe = VQE(
            estimator=estimator,
            ansatz=ansatz,
            optimizer=SBPLX(max_evals=max_evals),
            initial_point=ansatz.get_initial_point(),
        )
        try:
            second_q_op = ansatz.hamilton_operator  # for VHA
        except AttributeError:
            second_q_op = self.problem.second_q_ops()[0]  # for EfficientSU2 and UCC
        result_vha_statevector = vqe.compute_minimum_eigenvalue(
            operator=self.mapper.map(second_q_ops=second_q_op)
        )
        energy = float(result_vha_statevector.eigenvalue)

        return {
            "energy": energy,
            "optimal_parameters": result_vha_statevector.optimal_point.tolist(),
            "ansatz": ansatz,
        }

    def calculate_datapoint(
        self,
        ansatz_name: Literal["tVHA", "UCCSD", "UCCSDT", "HEA"] = "tVHA",
        trotter_steps: int | None = None,
        threshold_gamma: float | None = None,
        max_evals: int = 1000,
        cx_error_prob: float = 0,
    ) -> dict[str : float | list[float]]:
        """Gets the ground state energy of statevector simulator calculation of tVHA.

        Args:
            ansatz_name: Only use this, if you explicitly need another ansatz than tVHA.
                If this arg is used, threshold_gamma and trotter_steps are silently discarded.
            trotter_steps: Number of steps for Trotter of the (adiabatic)
                time evolution.
            threshold_gamma: The truncation threshold to use for building the tVHA ansatz.
            max_evals: Maximum number of function evaluations of the optimization algorithm (SBPLX).
            cx_error_prob: The CNOT gate depolarizing error of a simple noise model.
                Choose '0' for statevector calculation.

        Returns: dict of energy and optimal_parameters
            energy: electronic part of groundstate energy of statevector calculation.
            optimal_parameters: ansatz parameters after optimization loop.
        """
        if ansatz_name == "tVHA" and (trotter_steps is None or threshold_gamma is None):
            raise ValueError("tVHA needs explicit arguments 'trotter_steps' and 'threshold_gamma'.")
        result = self._calculate_statevector_energy(
            ansatz_name=ansatz_name,
            trotter_steps=trotter_steps,
            threshold_gamma=threshold_gamma,
            max_evals=max_evals,
            cx_error_prob=cx_error_prob,
        )
        del result["ansatz"]
        return result

    def get_numerical_energy(self) -> dict[str, float]:
        """Gets the numerical solution of ground state energy calculation.

        Solution of the exact diagonalization of the Hamiltonian matrix eigenvalue problem,
        i.e. FCI (Full Configuration Interaction; supposed to be exact up to numerical precision).

        First attempts to read from file.
        If this does not exist, a new calculation is started
        (writing the result to a text file for possible consecutive runs).

        Returns: dict(energy_name, energy_value)
        """
        file = self.output_path.joinpath("energy_numerical.json")
        try:
            return json.loads(file.read_text())
        except (FileNotFoundError, TypeError):

            def filter_criterion(
                eigenstate: list | np.ndarray,  # noqa: ARG001
                eigenvalue: float,  # noqa: ARG001
                aux_values: dict[str, tuple[float | complex]],
            ) -> bool | np.bool_:
                eval_num_particles = aux_values.get("ParticleNumber")
                if eval_num_particles is None:
                    return True
                num_particles_close = np.isclose(
                    eval_num_particles[0], self.problem.num_alpha + self.problem.num_beta
                )

                eval_angular_momentum = aux_values.get("AngularMomentum")
                if eval_angular_momentum is None:
                    return num_particles_close
                spin = self.problem.num_alpha - self.problem.num_beta
                expected_angular_momentum = spin / 2 * (spin / 2 + 1)
                angular_momentum_close = np.isclose(
                    eval_angular_momentum[0], expected_angular_momentum
                )

                return num_particles_close and angular_momentum_close

            algo = NumPyMinimumEigensolver(filter_criterion=filter_criterion)
            solver = GroundStateEigensolver(self.mapper, algo)
            result_fci = solver.solve(self.problem)
            print(result_fci)
            numerical_energies = {
                "electronic_energy": result_fci.electronic_energies[0],
                "computed_energy": result_fci.computed_energies[0],
                "inactive_space_energy": result_fci.electronic_energies[0]
                - result_fci.computed_energies[0],
                "hartree_fock_energy": result_fci.hartree_fock_energy,
                "nuclear_repulsion_energy": result_fci.nuclear_repulsion_energy,
                "total_energy": result_fci.total_energies[0],
            }
            file.write_text(json.dumps(numerical_energies, sort_keys=False))
            return numerical_energies

    def get_VHA_circuit_counts(self) -> pd.DataFrame:  # noqa: N802
        """Gets number of CNOT gates and circuit depth for VHA.

        If you are interested in the circuit count for a larger number of Trotter steps,
        just mulitply the circuit counts from this method with the number of Trotter steps.
        With heavy circuit optimization, one might be able to get slightly smaller circuits
        but still above way is a good approximation.
        """
        file = self.output_path.joinpath("circuit_counts_vha.csv")
        try:
            data = pd.read_csv(file, index_col=False)
            return data.drop_duplicates(
                subset="threshold_gamma", keep="first", ignore_index=True
            ).sort_values("threshold_gamma")
        except FileNotFoundError:
            circuit_data_header = (
                "threshold_gamma",
                "num_nonlocal_gates",
                "depth",
                "num_parameters",
            )
            circuit_data = []
            for threshold_gamma in tqdm(
                self._vha_dummy._get_possible_thresholds_gamma(),  # noqa: SLF001
                desc="Circuit counts",
                position=0,
            ):
                ansatz = VariationalHamiltonianAnsatz(
                    problem=self.problem,
                    mapper=self.mapper,
                    trotter_steps=1,
                    threshold_gamma=threshold_gamma,
                )
                circuit = ansatz.decompose(reps=4)
                circuit_data.append(
                    [
                        threshold_gamma,
                        circuit.num_nonlocal_gates(),
                        circuit.depth(),
                        circuit.num_parameters,
                    ]
                )
            data = pd.DataFrame(data=circuit_data, columns=circuit_data_header).sort_values(
                "threshold_gamma"
            )
            data.to_csv(file, index=False)
            return data

    def get_reference_UCC_circuit_counts(  # noqa: N802
        self, excitations: Sequence[int] = (1, 2)
    ) -> dict[str, float]:
        """Gets number of CNOT gates and circuit depth for UCCSD ansatz as reference.

        Args:
            excitations: By default singles and doubles (i.e. UCCSD); for other excitations, pass
                a list with the excitation numbers (1=singles, 2=doubles, 3=triples, 4=duadruples).

        Returns: dict(entry_name, count), where entry_name can be 'num_nonlocal_gates' and 'depth'.
        """
        excitationstring = ""
        if 1 in excitations:
            excitationstring += "s"
        if 2 in excitations:
            excitationstring += "d"
        if 3 in excitations:
            excitationstring += "t"
        if 4 in excitations:
            excitationstring += "q"

        file = self.output_path.joinpath(f"circuit_counts_ucc{excitationstring}.json")
        try:
            return json.loads(file.read_text())
        except (FileNotFoundError, TypeError):
            ansatz = UCC(
                excitations=list(excitations),
                num_spatial_orbitals=self.problem.num_spatial_orbitals,
                num_particles=self.problem.num_particles,
                qubit_mapper=self.mapper,
                initial_state=HartreeFock(
                    num_spatial_orbitals=self.problem.num_spatial_orbitals,
                    num_particles=self.problem.num_particles,
                    qubit_mapper=self.mapper,
                ),
            )
            circuit = ansatz.decompose(reps=4)
            circuit_counts = {
                "num_nonlocal_gates": circuit.num_nonlocal_gates(),
                "depth": circuit.depth(),
                "num_parameters": circuit.num_parameters,
            }
            file.write_text(json.dumps(circuit_counts, sort_keys=False))
            return circuit_counts

    def get_reference_HEA_circuit_counts(self) -> dict[str, float]:  # noqa: N802
        """Gets number of CNOT gates and circuit depth for hardware efficient ansatz as reference.

        Returns: dict(entry_name, count), where entry_name can be 'num_nonlocal_gates' and 'depth'.
        """
        file = self.output_path.joinpath("circuit_counts_hea.json")
        try:
            return json.loads(file.read_text())
        except (FileNotFoundError, TypeError):
            ansatz = EfficientSU2_HartreeFock(
                num_spatial_orbitals=self.problem.num_spatial_orbitals,
                num_particles=self.problem.num_particles,
                mapper=self.mapper,
                num_qubits=len(self.mapper.map(self.problem.second_q_ops()[0])[0].paulis[0]),
                entanglement="reverse_linear",
                reps=3,
            )
            circuit = ansatz.decompose(reps=4)
            circuit_counts = {
                "num_nonlocal_gates": circuit.num_nonlocal_gates(),
                "depth": circuit.depth(),
                "num_parameters": circuit.num_parameters,
            }
            file.write_text(json.dumps(circuit_counts, sort_keys=False))
            return circuit_counts

    def plot_histogram(
        self,
        hamiltonian: FermionicOp,
        one_body_terms: bool = True,
        two_body_terms_coulomb: bool = True,
        two_body_terms_noncoulomb: bool = True,
        log_x: bool = False,
        log_y: bool = False,
        add_title: bool = True,
    ) -> None:
        """Histogram of prefactors of the second quantization operators.

        Args:
            hamiltonian: the Hamiltonian of the system.
            one_body_terms: whether to include the one-body terms into the plot.
            two_body_terms_coulomb: whether to include the Coulomb two-body terms into the plot.
            two_body_terms_noncoulomb: whether to include the non-Coulomb two-body terms into the
                plot.
            log_x: whether to use a logarithmic scale for the x axis.
            log_y: whether to use a logarithmic scale for the y axis.
            add_title: whether to add a title to the plot.
        """
        coeffs, labels, colors = [], [], []
        if one_body_terms:
            coeffs_one_body = [
                abs(op.real) for op in hamiltonian.get_one_body_hamiltonian().values()
            ]
            coeffs.append(coeffs_one_body)
            labels.append("One-body terms")
            colors.append(color_circle[1])
        if two_body_terms_coulomb:
            coeffs_two_body_coulomb = [
                abs(op.real) for op in hamiltonian.get_two_body_hamiltonian_coulomb_terms().values()
            ]
            coeffs.append(coeffs_two_body_coulomb)
            labels.append("Two-body terms (Coulomb)")
            colors.append(color_scheme_fhg["Orange"])
        if two_body_terms_noncoulomb:
            coeffs_two_body_noncoulomb = [
                abs(op.real)
                for op in hamiltonian.get_two_body_hamiltonian_noncoulomb_terms().values()
            ]
            coeffs.append(coeffs_two_body_noncoulomb)
            labels.append("Two-body terms (non-Coulomb)")
            colors.append(color_circle[0])
        coeffs_flattened = list(itertools.chain.from_iterable(coeffs))

        if log_x:
            _, bins = np.histogram(np.log10(coeffs_flattened), bins="auto")
        else:
            bins = np.histogram_bin_edges(coeffs_flattened, bins="sqrt")
            if len(bins) < 20:
                bins = np.histogram_bin_edges(coeffs_flattened, bins="auto")
            if len(bins) < 20:
                bins = 20
        ax = plt.figure().gca()
        plt.hist(
            coeffs,
            log=log_y,
            bins=10**bins if log_x else bins,
            label=labels,
            color=colors,
            stacked=True,
        )
        if log_x:
            plt.xscale("log")
        _, ymax = plt.ylim()
        if log_y:
            plt.ylim(0.9, ymax)
        else:
            plt.ylim(0, ymax)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if add_title:
            plt.title(f"Terms of Hamiltonian (${self.molecule_name}$)")
        plt.xlabel("Magnitude of coefficient (absolute value in Hartree)")
        plt.ylabel("Number of terms")
        plt.legend()
        filename = f"{self.molecule_name}_hist"
        if one_body_terms and two_body_terms_coulomb and two_body_terms_noncoulomb:
            filename += "_all_terms"
        else:
            if one_body_terms:
                filename += "_onebody"
            if two_body_terms_coulomb:
                filename += "_twobodycoulomb"
            if two_body_terms_noncoulomb:
                filename += "_twobodynoncoulomb"
        if log_x:
            filename += "_logx"
        if log_y:
            filename += "_logy"
        filename += ".svg"
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()

    def plot_cumulated_density_distribution_all_terms(
        self, hamiltonian: FermionicOp, add_title: bool = True
    ) -> None:
        """Plot the cumulative density distribution for second quantization operators.

        Args:
            hamiltonian: the Hamiltonian of the system.
            add_title: whether to add a title to the plot.
        """
        coeffs = np.sort([abs(op) for op in hamiltonian.get_compressed_hamiltonian().values()])[
            ::-1
        ]
        coeffs = np.insert(coeffs, 0, 0.0)

        cumsum = coeffs.cumsum()  # cumulative sum of elements
        prefactors_sum = coeffs.sum()  # overall sum of all prefactors
        logger.debug("Number of terms: %s", len(coeffs))
        plt.plot(cumsum / prefactors_sum, linestyle="--", marker="x")
        if add_title:
            plt.title(
                f"Cumulated density distribution of all one- and two-body terms (${self.molecule_name}$)"
            )
        plt.xlabel("Number of terms")
        plt.ylabel("Cumulated normalized weight")
        plt.grid(visible=True)
        plt.savefig(
            self.output_path.joinpath(
                f"{self.molecule_name}_cumulated_density_distribution_all_terms.svg"
            ),
            format="svg",
        )
        plt.close()

    def plot_cumulated_density_distribution_noncoulomb_terms(
        self, hamiltonian: FermionicOp, add_title: bool = True
    ) -> None:
        """Plot the cumulative density distribution for second quantization operators.

        Args:
            hamiltonian: the Hamiltonian of the system.
            add_title: whether to add a title to the plot.
        """
        coeffs_two_body = np.sort(
            [
                abs(op)
                for op in hamiltonian.get_two_body_hamiltonian_noncoulomb_terms()
                .get_compressed_hamiltonian()
                .values()
            ]
        )[::-1]
        coeffs_two_body = np.insert(coeffs_two_body, 0, 0.0)

        cumsum = coeffs_two_body.cumsum()  # cumulative sum of elements
        prefactors_sum = coeffs_two_body.sum()  # overall sum of all prefactors
        logger.debug("Number of two-body terms: %s", len(coeffs_two_body))
        plt.plot(cumsum / prefactors_sum, linestyle="--", marker="x")
        if add_title:
            plt.title(
                f"Cumulated density distribution of non-Coulomb two-body terms (${self.molecule_name}$)"
            )
        plt.xlabel("Number of terms")
        plt.ylabel("Cumulated normalized weight")
        plt.grid(visible=True)
        plt.savefig(
            self.output_path.joinpath(
                f"{self.molecule_name}_cumulated_density_distribution_noncoulomb_terms.svg"
            ),
            format="svg",
        )
        plt.close()

    def plot_cnot_count_over_truncation_threshold(
        self,
        log_y: bool = False,
        add_title: bool = True,
    ) -> None:
        """Plots CNOT count of tVHA depending on the truncation threshold.

        Args:
            log_y: whether to use a logarithmic scale for the y axis.
            add_title: whether to add a title to the plot.
        """
        circuit_data = self.get_VHA_circuit_counts()

        ax = plt.figure().gca()

        # tVHA
        plt.plot(
            circuit_data.threshold_gamma,
            circuit_data.num_nonlocal_gates,
            label="tVHA",
            marker="x",
            linestyle="dotted",
            color=colors_tvha[0],
        )
        xmin, xmax = plt.xlim()

        # UCCSD
        circuit_counts_uccsd = self.get_reference_UCC_circuit_counts()
        plt.hlines(
            y=circuit_counts_uccsd["num_nonlocal_gates"],
            xmin=xmin,
            xmax=xmax,
            label="UCCSD",
            linestyle="dashdot",
            color=colors_ucc[0],
        )

        # UCCSDT
        circuit_counts_uccsdt = self.get_reference_UCC_circuit_counts(excitations=[1, 2, 3])
        if (
            circuit_counts_uccsdt["num_nonlocal_gates"]
            != circuit_counts_uccsd["num_nonlocal_gates"]
        ):
            # In case of H2, there are no triple excitations
            plt.hlines(
                y=circuit_counts_uccsdt["num_nonlocal_gates"],
                xmin=xmin,
                xmax=xmax,
                label="UCCSDT",
                linestyle="dashdot",
                color=colors_ucc[1],
            )

        # HEA
        circuit_counts_hea = self.get_reference_HEA_circuit_counts()
        plt.hlines(
            y=circuit_counts_hea["num_nonlocal_gates"],
            xmin=xmin,
            xmax=xmax,
            label="HEA",
            linestyle="dashed",
            color=color_hea,
        )

        plt.xlim(xmin, xmax)

        plt.yscale("log" if log_y else "linear")
        if not log_y:
            _, ymax = plt.ylim()
            plt.ylim(0, ymax)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if add_title:
            plt.title(f"Two-qubit gate count for tVHA, UCC and HEA (${self.molecule_name}$)")
        plt.xlabel("Truncation threshold")
        plt.ylabel("Number of CNOTs")
        plt.legend()

        filename = f"{self.molecule_name}_cnot_count_over_truncation_threshold"
        if log_y:
            filename += "_logy"
        filename += ".svg"
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()

    def plot_circuit_depth_over_truncation_threshold(
        self,
        log_y: bool = False,
        add_cnot_count: bool = True,
        add_title: bool = True,
    ) -> None:
        """Plots circuit depth (and CNOT count) of tVHA depending on the truncation threshold.

        Args:
            log_y: whether to use a logarithmic scale for the y axis.
            add_cnot_count: whether to include the number of CNOTs to the plot.
            add_title: whether to add a title to the plot.
        """
        circuit_data = self.get_VHA_circuit_counts()

        ax = plt.figure().gca()
        labels = []

        label_string = "Circuit depth / # CNOTs" if add_cnot_count else "Circuit depth"

        # tVHA
        plt.plot(
            circuit_data.threshold_gamma,
            circuit_data.depth,
            label="Circuit depth (tVHA)" if add_cnot_count else "tVHA",
            marker="x",
            linestyle="dotted",
            color=colors_tvha[0],
        )
        if add_cnot_count:
            plt.plot(
                circuit_data.threshold_gamma,
                circuit_data.num_nonlocal_gates,
                label="# CNOTs (tVHA)",
                marker="2",
                linestyle="dashed",
                color=colors_tvha[0],
            )
        labels.append(label_string + " (tVHA)")
        xmin, xmax = plt.xlim()

        # UCCSD
        circuit_counts_uccsd = self.get_reference_UCC_circuit_counts()
        plt.hlines(
            y=circuit_counts_uccsd["depth"],
            xmin=xmin,
            xmax=xmax,
            label="Circuit depth (UCCSD)" if add_cnot_count else "UCCSD",
            linestyle="dashdot",
            color=colors_ucc[0],
        )
        if add_cnot_count:
            plt.hlines(
                y=circuit_counts_uccsd["num_nonlocal_gates"],
                xmin=xmin,
                xmax=xmax,
                label="# CNOTs (UCCSD)",
                linestyle="dashed",
                color=colors_ucc[0],
            )
        labels.append(label_string + " (UCCSD)")

        # UCCSDT
        circuit_counts_uccsdt = self.get_reference_UCC_circuit_counts(excitations=[1, 2, 3])
        # In case of H2, there are no triple excitations
        if (
            circuit_counts_uccsdt["num_nonlocal_gates"]
            != circuit_counts_uccsd["num_nonlocal_gates"]
        ):
            plt.hlines(
                y=circuit_counts_uccsdt["depth"],
                xmin=xmin,
                xmax=xmax,
                label="Circuit depth (UCCSDT)" if add_cnot_count else "UCCSDT",
                linestyle="dashdot",
                color=colors_ucc[1],
            )
            if add_cnot_count:
                plt.hlines(
                    y=circuit_counts_uccsdt["num_nonlocal_gates"],
                    xmin=xmin,
                    xmax=xmax,
                    label="# CNOTs (UCCSDT)",
                    linestyle="dashed",
                    color=colors_ucc[1],
                )
            labels.append(label_string + " (UCCSDT)")

        # HEA
        circuit_counts_hea = self.get_reference_HEA_circuit_counts()
        plt.hlines(
            y=circuit_counts_hea["depth"],
            xmin=xmin,
            xmax=xmax,
            label="Circuit depth (HEA)" if add_cnot_count else "HEA",
            linestyle="dashed",
            color=color_hea,
        )
        if add_cnot_count:
            plt.hlines(
                y=circuit_counts_hea["num_nonlocal_gates"],
                xmin=xmin,
                xmax=xmax,
                label="# CNOTs (HEA)",
                linestyle="dotted",
                color=color_hea,
            )
        labels.append(label_string + " (HEA)")

        plt.xlim(xmin, xmax)

        plt.yscale("log" if log_y else "linear")
        if not log_y:
            _, ymax = plt.ylim()
            plt.ylim(0, ymax)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        from matplotlib.legend_handler import HandlerTuple

        if add_title:
            plt.title(f"Circuit depth for tVHA, UCC and HEA (${self.molecule_name}$)")
        plt.xlabel("Truncation threshold")
        plt.ylabel("Circuit depth")
        handles, _ = plt.gca().get_legend_handles_labels()
        if add_cnot_count:
            plt.legend(
                handles=zip(handles[::2], handles[1::2], strict=True),
                labels=labels,
                handler_map={tuple: HandlerTuple(None)},
                handlelength=len(handles) - 1,
            )
        else:
            plt.legend(handles=handles, labels=labels)

        filename = f"{self.molecule_name}_circuit_depth_over_truncation_threshold"
        if log_y:
            filename += "_logy"
        filename += ".svg"
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()

    def plot_energy_over_truncation_threshold(
        self,
        trotter_steps: int | Sequence[int] = 1,
        list_of_threshold_gamma: Sequence[float] | None = None,
        max_evals: int | Sequence[int] = 1000,
        show_truncation_error_estimate: bool = False,
        show_hea: bool = True,
        show_uccsd: bool = True,
        show_uccsdt: bool = True,
        add_title: bool = True,
    ) -> None:
        """Plots energy of tVHA depending on the truncation threshold.

        Optionally show HEA, UCCSD, and UCCSDT for comparison.

        Optionally show an a posteriori estimate for the truncation error.

        H = H0 + H1, where H0 is the truncated Hamiltonian and
        H1 is the part that is discarded due to truncation.
        Accordingly, |psi> is the wave function / circuit with optimal parameters {theta}
        without truncation and
        |psi0> denotes the truncated circuit with parameters {theta0} optimized for the
        cost function E0 = <psi0|H|psi0>.
        We are interested in the energy without truncation errors E = <psi|H|psi> but don't
        know |psi> since it is represented by a circuit that is often too long to be feasible.
        So, three estimates for the error Delta E = abs(E-E0) are considered:

        1. E_discarded = <psi0|H1|psi0> estimates the contribution of the discarded part of
        the Hamiltonian and assumes that the effect is similar to the effect of the transition
        from |psi> to |psi0>. The error estimate is then given by
        Delta E_discarded = abs(E_discarded-E0).
        This estimate is easy to achive; in principle, no additional calculations are needed
        as the contribution of these terms is already calculated within
        <psi0|H|psi0> = <psi0|H0+h1|psi0>.
        Note: For simplicity of coding, this term is re-evaluated instead of extracting it from
        the raw result.

        2. -- not used --
        1 - sum(abs(alpha_i^truncated)) / sum(abs(alpha_i^whole)) uses the absolute magnitude
        of the terms in the Hamiltonian. It is a very rough estimate and is almost useless.
        But it is very easy and fast to calculate.

        3. -- not used --
        E_extrapolated = <psi{theta0}|H|psi{theta0}> uses the parameters from |psi0> but
        inserts them into the full circuit |psi>. The error estimate is then given by
        Delta E_extrapolated = abs(E_extrapolated-E0). This estimate is prone to hardware noise
        and only feasible for full circuits that are small enough compared to the noise level.

        None of the estimates give information about the Trotterization error.


        Args:
            trotter_steps: The number of Trotter steps to use.
                If given as single element, only a single line is plotted.
                If given as list, all list elements are used in sorted order
                creating a line for each number of Trotter steps.
            list_of_threshold_gamma: The truncation thresholds to use for this plot.
                If 'None', all possible truncation thresholds are used.
            max_evals: Maximum number of function evaluations of the optimization algorithm (SBPLX).
                If a single value, it will be used for all trotter_steps.
                If given as a list, the first value will be used for the first element of the
                trotter_steps list, the second for the second etc.
            show_truncation_error_estimate: Whether to add the truncation error estimate to the
                plot. Be aware that this option requires possibly slow calculations
                which aren't cached for subsequent runs.
            show_hea: Whether to add the HEA energy to the plot.
            show_uccsd: Whether to add the UCCSD energy to the plot.
            show_uccsdt: Whether to add the UCCSDT energy to the plot.
            add_title: Whether to add a title to the plot.
        """
        list_of_trotter_steps = (
            list(trotter_steps) if isinstance(trotter_steps, Iterable) else [trotter_steps]
        )
        if isinstance(max_evals, Iterable):
            list_of_max_evals = list(max_evals)
        else:
            list_of_max_evals = [max_evals] * len(list_of_trotter_steps)
        list_of_threshold_gamma = self._vha_dummy.get_thresholds_gamma(
            thresholds_gamma=list_of_threshold_gamma
        )

        if len(list_of_trotter_steps) > len(colors_tvha):
            raise ValueError(
                "The energy over truncation threshold plot is not intended for a "
                "large amount of Trotter steps. Please reduce them from "
                f"{len(list_of_trotter_steps)} to at most {len(colors_tvha)}.",
            )

        # Retrieve all needed energy data
        datapoints = []
        for trotter_steps, max_evals in zip(list_of_trotter_steps, list_of_max_evals, strict=True):
            for threshold_gamma in list_of_threshold_gamma:
                datapoints.append(
                    {
                        "ansatz_name": "tVHA",
                        "trotter_steps": trotter_steps,
                        "max_evals": max_evals,
                        "threshold_gamma": threshold_gamma,
                    }
                )
        if show_hea:
            datapoints.append({"ansatz_name": "HEA", "max_evals": list_of_max_evals[0]})
        if show_uccsd:
            datapoints.append({"ansatz_name": "UCCSD", "max_evals": list_of_max_evals[0]})
        if show_uccsdt:
            datapoints.append({"ansatz_name": "UCCSDT", "max_evals": list_of_max_evals[0]})

        energy_data = self.get_datapoints(datapoints=datapoints)

        # tVHA
        for idx, (trotter_steps, max_evals) in enumerate(
            zip(list_of_trotter_steps, list_of_max_evals, strict=True)
        ):
            energies = energy_data[
                (energy_data["ansatz_name"] == "tVHA")
                & (energy_data["trotter_steps"] == trotter_steps)
                & (energy_data["max_evals"] == max_evals)
            ].sort_values(by="threshold_gamma")["energy"]
            plt.plot(
                list_of_threshold_gamma,
                energies,
                label=f"tVHA ({trotter_steps} Trotter step{'' if trotter_steps == 1 else 's'})",
                marker=["d", "o", "X", "+", "x"][idx % len(list_of_trotter_steps)],
                linestyle="dotted",
                linewidth=0.8,
                color=colors_tvha[idx % len(list_of_trotter_steps)],
            )

            # error estimate
            if show_truncation_error_estimate:
                final_parameters = energy_data[
                    (energy_data["ansatz_name"] == "tVHA")
                    & (energy_data["trotter_steps"] == trotter_steps)
                    & (energy_data["max_evals"] == max_evals)
                ].sort_values(by="threshold_gamma")["optimal_parameters"]

                estimator = StatevectorEstimator()
                # Note: It might be a nice extension to support a noise estimator, too.

                energies_discarded_part = []
                for threshold_gamma, param in zip(
                    list_of_threshold_gamma, final_parameters, strict=True
                ):
                    tvha = VariationalHamiltonianAnsatz(
                        problem=self.problem,
                        mapper=self.mapper,
                        threshold_gamma=threshold_gamma,
                        trotter_steps=trotter_steps,
                    )

                    second_q_op_discarded_part = tvha._get_hamiltonian_gamma(
                        threshold_gamma=1.0
                    ) - tvha._get_hamiltonian_gamma(threshold_gamma=threshold_gamma)
                    operator_discarded = self.mapper.map(second_q_ops=second_q_op_discarded_part)

                    job = estimator.run(
                        tvha,
                        operator_discarded,
                        ast.literal_eval(param) if isinstance(param, str) else param,
                    )
                    estimator_result = job.result()
                    energy = (
                        estimator_result.values[0]
                        if len(estimator_result.values) == 1
                        else estimator_result.values
                    )
                    energies_discarded_part.append(energy)

                plt.fill_between(
                    x=list_of_threshold_gamma,
                    y1=[
                        e - abs(e_disc)
                        for e_disc, e in zip(energies_discarded_part, energies, strict=True)
                    ],
                    y2=energies,
                    color=colors_tvha[idx % len(list_of_trotter_steps)],
                    alpha=0.4,
                )

                # vha_full = VariationalHamiltonianAnsatz(
                #     problem=self.problem,
                #     mapper=self.mapper,
                #     threshold_gamma=1,
                #     trotter_steps=trotter_steps,
                # )

                # Very rouogh energy error estimate
                # # 2st error estimate
                # terms_alpha = [abs(term) for term in vha_full._hamiltonian_alpha.values()]
                # terms_beta = [abs(term) for term in vha_full._hamiltonian_beta.values()]
                # cumsum_gamma_terms = []
                # for threshold_gamma in list_of_threshold_gamma:
                #     cumsum_gamma_terms.append(
                #         sum(
                #             [
                #                 abs(term)
                #                 for term in vha_full._get_hamiltonian_gamma(
                #                     threshold_gamma=threshold_gamma
                #                 ).values()
                #             ]
                #         )
                #     )
                # # vha_full._get_sorted_noncoulomb_two_body_terms()
                # sum_non_gamma_terms = sum(terms_alpha + terms_beta)
                # sum_all_terms = sum_non_gamma_terms + cumsum_gamma_terms[-1]
                # relative_error_estimate = [
                #     1 - (c + sum_non_gamma_terms) / sum_all_terms for c in cumsum_gamma_terms
                # ]
                # plt.fill_between(
                #     x=list_of_threshold_gamma,
                #     y1=[e - err * e for err, e in zip(relative_error_estimate, energies, strict=True)],
                #     y2=[e + err * e for err, e in zip(relative_error_estimate, energies, strict=True)],
                #     label="a priori error estimate",
                #     color=colors_tvha[idx % len(list_of_trotter_steps)],
                #     alpha=0.2,
                # )

                # Error estimate which is not feasible for larger systems
                # # 3rd error estimate
                # second_q_op_full = vha_full.hamilton_operator
                # operator_full = self.mapper.map(second_q_ops=second_q_op_full)
                # energies_extrapolated = []
                # for param in final_parameters:
                #     job = estimator.run(
                #         vha_full,
                #         operator_full,
                #         ast.literal_eval(param) if isinstance(param, str) else param,
                #     )
                #     estimator_result = job.result()
                #     energy = (
                #         estimator_result.values[0]
                #         if len(estimator_result.values) == 1
                #         else estimator_result.values
                #     )
                #     energies_extrapolated.append(energy)
                # plt.errorbar(
                #     x=list_of_threshold_gamma,
                #     y=energies,
                #     yerr=[
                #         abs(e - e_extrapolated)
                #         for e, e_extrapolated in zip(energies, energies_extrapolated, strict=True)
                #     ],
                #     label="error estimate (parameter extrapolation to full circuit)",
                #     color=colors_tvha[idx % len(list_of_trotter_steps)],
                #     linestyle="none",
                #     capsize=4,
                # )

        xmin, xmax = plt.xlim()

        labels_close_to_hf = ["HF"]
        labels_close_to_fci = ["FCI"]

        energy_hf = (
            self.problem.reference_energy
            - self.numerical_energies["nuclear_repulsion_energy"]
            - self.numerical_energies["inactive_space_energy"]
        )
        energy_fci = self.numerical_energies["computed_energy"]

        # UCC
        if show_uccsd:
            energy_uccsd = energy_data[
                (energy_data["ansatz_name"] == "UCCSD")
                & (energy_data["max_evals"] == list_of_max_evals[0])
            ]["energy"].iloc[0]
        if show_uccsdt:
            energy_uccsdt = energy_data[
                (energy_data["ansatz_name"] == "UCCSDT")
                & (energy_data["max_evals"] == list_of_max_evals[0])
            ]["energy"].iloc[0]

        # UCCSD
        if show_uccsd:
            if np.isclose(energy_uccsd, energy_fci):
                labels_close_to_fci.append("UCCSD")
            elif np.isclose(energy_uccsd, energy_hf):
                labels_close_to_hf.append("UCCSD")
            else:
                plt.hlines(
                    y=energy_uccsd,
                    xmin=xmin,
                    xmax=xmax,
                    label="UCCSD / UCCSDT"
                    if show_uccsdt and np.isclose(energy_uccsdt, energy_uccsd)
                    else "UCCSD",
                    linestyles="dashdot",
                    color=colors_ucc[0],
                )

        # UCCSDT
        if show_uccsdt:
            if np.isclose(energy_uccsdt, energy_fci):
                labels_close_to_fci.append("UCCSDT")
            elif np.isclose(energy_uccsdt, energy_hf):
                labels_close_to_hf.append("UCCSDT")
            elif not np.isclose(energy_uccsdt, energy_uccsd):
                plt.hlines(
                    y=energy_uccsdt,
                    xmin=xmin,
                    xmax=xmax,
                    label="UCCSDT",
                    linestyles="dashdot",
                    color=colors_ucc[1],
                )

        # HEA
        if show_hea:
            energy_hea = energy_data[
                (energy_data["ansatz_name"] == "HEA")
                & (energy_data["max_evals"] == list_of_max_evals[0])
            ]["energy"].iloc[0]
            if np.isclose(energy_hea, energy_fci):
                labels_close_to_fci.append("HEA")
            elif np.isclose(energy_uccsd, energy_hf):
                labels_close_to_hf.append("HEA")
            else:
                plt.hlines(
                    y=energy_hea,
                    xmin=xmin,
                    xmax=xmax,
                    label="HEA",
                    linestyles="dashed",
                    color=color_hea,
                )

        # HF energy
        plt.hlines(
            energy_hf,
            xmin=xmin,
            xmax=xmax,
            label=" / ".join(labels_close_to_hf),
            color=color_hf,
            linestyles="solid",
            zorder=0,
        )

        # FCI energy
        plt.hlines(
            energy_fci,
            xmin=xmin,
            xmax=xmax,
            label=" / ".join(labels_close_to_fci),
            color=color_fci,
            linestyles="solid",
            zorder=0,
        )
        plt.fill_between(
            (xmin, xmax),
            energy_fci,
            energy_fci + 0.0015,
            label="chemical accuracy",
            color=color_fci,
            alpha=0.4,
            zorder=0,
        )

        plt.legend()
        plt.xlabel("Truncation threshold")
        plt.ylabel("Energy in Hartree")
        if add_title:
            plt.title(
                f"Energy of tVHA depending on the truncation threshold "
                f"(${self.molecule_name}$)"
                f"\n({list_of_max_evals} function evaluations)"
            )
        plt.xlim(xmin, xmax)
        filename = (
            f"{self.molecule_name}_energy_over_truncation_threshold_"
            f"{'error_estimate_' if show_truncation_error_estimate else ''}"
            f"{'_'.join(str(trotter_steps) + '-' + str(max_evals) for trotter_steps, max_evals in zip(list_of_trotter_steps, list_of_max_evals, strict=True))}.svg"
        )
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()

    def plot_energy_over_trotter_steps(
        self,
        list_of_trotter_steps: Sequence[int] = (1, 2, 3, 4, 5),
        threshold_gamma: float | Sequence[float] = 1.0,
        max_evals: int | Sequence[int] = 1000,
        add_title: bool = True,
    ) -> None:
        """Plots the energy of tVHA depending on the number of Trotter steps.

        Args:
            list_of_trotter_steps: The numbers of Trotter steps to use.
            threshold_gamma: The truncation threshold to use for this plot.
                If given as single element, only a single line is plotted.
                If given as list, all list elements are used in sorted order
                creating a line for each truncation threshold.
            max_evals: Maximum number of function evaluations of the optimization algorithm (SBPLX).
            add_title: Whether to add a title to the plot.
        """
        list_of_max_evals = (
            list(max_evals)
            if isinstance(max_evals, Iterable)
            else [max_evals] * len(list_of_trotter_steps)
        )
        list_of_threshold_gamma = self._vha_dummy.get_thresholds_gamma(
            threshold_gamma if isinstance(threshold_gamma, Iterable) else [threshold_gamma]
        )

        if len(list_of_threshold_gamma) > len(colors_tvha):
            raise ValueError(
                "The energy over Trotter steps plot is not intended for a "
                "large amount of Trotter steps. Please reduce them from "
                f"{len(list_of_trotter_steps)} to at most {len(colors_tvha)}.",
            )

        # Retrieve all needed energy data
        datapoints = []
        for trotter_steps, max_evals in zip(list_of_trotter_steps, list_of_max_evals, strict=True):
            for threshold_gamma in list_of_threshold_gamma:
                datapoints.append(
                    {
                        "ansatz_name": "tVHA",
                        "trotter_steps": trotter_steps,
                        "max_evals": max_evals,
                        "threshold_gamma": threshold_gamma,
                    }
                )
        for ansatz_name in ("HEA", "UCCSD", "UCCSDT"):
            datapoints.append({"ansatz_name": ansatz_name, "max_evals": list_of_max_evals[0]})

        energy_data = self.get_datapoints(datapoints=datapoints)

        ax = plt.figure().gca()

        # tVHA
        for idx, threshold_gamma in enumerate(list_of_threshold_gamma):
            energies = energy_data[
                (energy_data["ansatz_name"] == "tVHA")
                & (energy_data["threshold_gamma"] == threshold_gamma)
                & (energy_data["max_evals"] == max_evals)
            ].sort_values(by="trotter_steps")["energy"]
            plt.plot(
                list_of_trotter_steps,
                energies,
                label=f"tVHA (truncation threshold {threshold_gamma:.4g})",
                marker=["d", "o", "X", "+", "x"][idx % len(list_of_threshold_gamma)],
                linestyle="dotted",
                linewidth=0.8,
                color=colors_tvha[idx % len(list_of_threshold_gamma)],
            )

        xmin, xmax = plt.xlim()

        labels_close_to_hf = ["HF"]
        labels_close_to_fci = ["FCI"]

        energy_hf = (
            self.problem.reference_energy
            - self.numerical_energies["nuclear_repulsion_energy"]
            - self.numerical_energies["inactive_space_energy"]
        )
        energy_fci = self.numerical_energies["computed_energy"]

        # UCC
        energy_uccsd = energy_data[
            (energy_data["ansatz_name"] == "UCCSD")
            & (energy_data["max_evals"] == list_of_max_evals[0])
        ]["energy"].iloc[0]
        energy_uccsdt = energy_data[
            (energy_data["ansatz_name"] == "UCCSDT")
            & (energy_data["max_evals"] == list_of_max_evals[0])
        ]["energy"].iloc[0]

        # UCCSD
        if np.isclose(energy_uccsd, energy_fci):
            labels_close_to_fci.append("UCCSD")
        elif np.isclose(energy_uccsd, energy_hf):
            labels_close_to_hf.append("UCCSD")
        else:
            plt.hlines(
                y=energy_uccsd,
                xmin=xmin,
                xmax=xmax,
                label="UCCSD",
                linestyles="dashdot",
                color=colors_ucc[0],
            )

        # UCCSDT
        if np.isclose(energy_uccsdt, energy_fci):
            labels_close_to_fci.append("UCCSDT")
        elif not np.isclose(energy_uccsdt, energy_uccsd):
            plt.hlines(
                y=energy_uccsdt,
                xmin=xmin,
                xmax=xmax,
                label="UCCSDT",
                linestyles="dashdot",
                color=colors_ucc[1],
            )

        # HEA
        energy_hea = energy_data[
            (energy_data["ansatz_name"] == "HEA")
            & (energy_data["max_evals"] == list_of_max_evals[0])
        ]["energy"].iloc[0]
        if np.isclose(energy_hea, energy_fci):
            labels_close_to_fci.append("HEA")
        elif np.isclose(energy_uccsd, energy_hf):
            labels_close_to_hf.append("HEA")
        else:
            plt.hlines(
                y=energy_hea,
                xmin=xmin,
                xmax=xmax,
                label="HEA",
                linestyles="dashed",
                color=color_hea,
            )

        # HF energy
        plt.hlines(
            energy_hf,
            xmin=xmin,
            xmax=xmax,
            label=" / ".join(labels_close_to_hf),
            color=color_hf,
            linestyles="solid",
            zorder=0,
        )

        # FCI energy
        plt.hlines(
            energy_fci,
            xmin=xmin,
            xmax=xmax,
            label=" / ".join(labels_close_to_fci),
            color=color_fci,
            linestyles="solid",
            zorder=0,
        )
        plt.fill_between(
            (xmin, xmax),
            energy_fci,
            energy_fci + 0.0015,
            label="Chemical accuracy",
            color=color_fci,
            alpha=0.4,
            zorder=0,
        )

        plt.legend()
        plt.xlabel("Trotter steps")
        plt.ylabel("Energy in Hartree")
        if add_title:
            plt.title(
                f"Energy of tVHA denpending on the number of Trotter steps (${self.molecule_name}$)"
                f"\n({list_of_max_evals} function evaluations)"
            )
        plt.xlim(xmin, xmax)
        filename = f"{self.molecule_name}_energy_over_trotter_steps_"
        filename += "_".join(
            f"{threshold_gamma:.4g}" for threshold_gamma in list_of_threshold_gamma
        )
        filename += ".svg"
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()

    def plot_energy_heatmap(
        self,
        list_of_trotter_steps: Sequence[int] = (1, 2, 3, 4, 5),
        list_of_threshold_gamma: Sequence[float] | None = None,
        max_evals: int | Sequence[int] = 1000,
        add_title: bool = True,
    ) -> None:
        """Plots energy of tVHA depending on truncation threshold and Trotter steps as heatmap.

        Args:
            list_of_trotter_steps: The numbers of Trotter steps to use.
            list_of_threshold_gamma: The truncation thresholds to use for this plot.
                If 'None', all possible truncation thresholds are used.
            max_evals: Maximum number of function evaluations of the optimization algorithm (SBPLX).
            add_title: Whether to add a title to the plot.
        """
        if isinstance(max_evals, Iterable):
            list_of_max_evals = sorted(max_evals)
        else:
            list_of_max_evals = [max_evals] * len(list_of_trotter_steps)
        list_of_threshold_gamma = self._vha_dummy.get_thresholds_gamma(list_of_threshold_gamma)

        datapoints = self.get_datapoints(
            datapoints=[
                {
                    "threshold_gamma": threshold_gamma,
                    "trotter_steps": trotter_steps,
                    "max_evals": max_evals,
                }
                for threshold_gamma in list_of_threshold_gamma
                for trotter_steps, max_evals in zip(
                    list_of_trotter_steps, list_of_max_evals, strict=True
                )
            ]
        )
        A, B = np.meshgrid(list_of_threshold_gamma, list_of_trotter_steps, indexing="ij")  # noqa: N806

        energies_reshaped = (
            datapoints.sort_values(by=["threshold_gamma", "trotter_steps"])["energy"]
            .to_numpy()
            .reshape(A.shape)
        )

        ax = plt.figure().gca()
        plt.pcolormesh(A, B, energies_reshaped, cmap="hot", edgecolors="face")
        plt.colorbar()
        plt.xlabel("Truncation threshold")
        plt.ylabel("Number of Trotter steps")
        if add_title:
            plt.title(
                f"Energy in Hartree (${self.molecule_name}$)\n"
                f"({list_of_max_evals} function evaluations)"
            )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(
            self.output_path.joinpath(
                f"{self.molecule_name}_energy_heatmap_threshold_trottersteps.svg"
            ),
            format="svg",
        )
        plt.close()

    def plot_energy_landscape(
        self,
        alphas: Sequence[float] = np.linspace(0, 30, 10),
        betas: Sequence[float] = np.linspace(0, 30, 10),
        gammas: Sequence[float] = np.linspace(0, 30, 10),
    ) -> None:
        """Plots the energy landscape for parameters alpha and beta in the VHA ansatz.

        This plot makes only sense for a single Trotter step;
        else it would be a multidimensional plot due to the large amount of parameters.

        In contrast to the other plot methods, this one is quite inefficient since it does not
        store calculated energy values. So, every run inefficiently re-calculates all energy values.
        """
        alphas = list(alphas)
        betas = list(betas)
        gammas = list(gammas)

        A, B, C = np.meshgrid(alphas, betas, gammas, indexing="ij")  # noqa: N806
        A_small, B_small = np.meshgrid(alphas, betas, indexing="ij")  # noqa: N806
        parameter_values = list(zip(A.flatten(), B.flatten(), C.flatten(), strict=True))

        ansatz_vha = VariationalHamiltonianAnsatz(
            problem=self.problem, trotter_steps=1, mapper=self.mapper
        )
        estimator = StatevectorEstimator()
        job = estimator.run(
            circuits=[ansatz_vha] * len(parameter_values),
            observables=[self.mapper.map(second_q_ops=ansatz_vha.hamilton_operator)]
            * len(parameter_values),
            parameter_values=parameter_values,
        )
        result = job.result()
        energies = result.values.reshape(A.shape)

        # Minimum energy and alpha and beta values belonging to it
        alpha_min = A[np.unravel_index(energies.argmin(), energies.shape)]
        beta_min = B[np.unravel_index(energies.argmin(), energies.shape)]
        gamma_min = C[np.unravel_index(energies.argmin(), energies.shape)]
        print(
            f"Minimum energy (from energy landscape) {energies.min():.3f} Hartree with "
            f"α={alpha_min:.3g}, β={beta_min:.3g}, γ={gamma_min:.3g}"  # noqa: RUF001
        )

        period_alpha = []
        for beta_index in range(len(betas)):
            for gamma_index in range(len(gammas)):
                n = energies[:, beta_index, gamma_index].size
                dt = abs(alphas[1] - alphas[0])

                rfft = np.fft.rfft(energies[:, beta_index, gamma_index])
                coeffs_argmax = rfft[1:].argmax() + 1
                freq_max = np.fft.rfftfreq(n)[coeffs_argmax]
                period_alpha.append(dt / freq_max)
        period_beta = []
        for alpha_index in range(len(alphas)):
            for gamma_index in range(len(gammas)):
                n = energies[alpha_index, :, gamma_index].size
                dt = abs(betas[1] - betas[0])

                rfft = np.fft.rfft(energies[alpha_index, :, gamma_index])
                coeffs_argmax = rfft[1:].argmax() + 1
                freq_max = np.fft.rfftfreq(n)[coeffs_argmax]
                period_beta.append(dt / freq_max)
        period_gamma = []
        for alpha_index in range(len(alphas)):
            for beta_index in range(len(betas)):
                n = energies[alpha_index, beta_index, :].size
                dt = abs(gammas[1] - gammas[0])

                rfft = np.fft.rfft(energies[alpha_index, beta_index, :])
                coeffs_argmax = rfft[1:].argmax() + 1
                freq_max = np.fft.rfftfreq(n)[coeffs_argmax]
                period_gamma.append(dt / freq_max)
        logger.info("period_a %s", period_alpha)
        logger.info("period_a %s", period_beta)
        logger.info("period_a %s", period_gamma)
        logger.info("period_a (final value) %s", np.bincount(period_alpha).argmax())

        import typing

        import matplotlib as mpl

        mpl.use("TKAgg")

        # ---------------------------------------------------------------------
        # Plotting with animation
        import matplotlib.widgets
        import mpl_toolkits.axes_grid1
        from matplotlib.animation import FuncAnimation
        from matplotlib.figure import Figure

        class Player(FuncAnimation):
            @typing.no_type_check
            def __init__(
                self,
                fig,  # noqa: ANN001
                func,  # noqa: ANN001
                frames=None,  # noqa: ANN001, ARG002
                init_func=None,  # noqa: ANN001
                fargs=None,  # noqa: ANN001
                save_count=None,  # noqa: ANN001
                mini=0,  # noqa: ANN001
                maxi=100,  # noqa: ANN001
                repeating: bool = False,
                pos=(0.125, 0.92),  # noqa: ANN001
                **kwargs,  # noqa: ANN003
            ) -> None:
                self.i = 0
                self.min = mini
                self.max = maxi
                self.runs = True
                self.forwards = True
                self.fig = fig
                self.repeating = repeating
                self.func = func
                self.setup(pos)
                FuncAnimation.__init__(
                    self,
                    fig=self.fig,
                    func=self.func,
                    frames=self.play(),
                    # repeat=self.repeating,
                    init_func=init_func,
                    fargs=fargs,
                    save_count=save_count,
                    **kwargs,
                )

            @typing.no_type_check
            def play(self) -> Iterator[int]:
                while self.runs:
                    self.i = self.i + 1 if self.forwards else self.i - 1
                    if self.i > self.min and self.i < self.max:
                        yield self.i
                    elif self.repeating:
                        if self.i == self.max or self.min:
                            yield self.i
                        elif self.i > self.max:
                            self.i = self.min
                            yield self.i
                        else:  # if self.i < self.min
                            self.i = self.max
                            yield self.i
                    else:
                        self.pause()
                        yield self.i

            @typing.no_type_check
            def start(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.runs = True
                self.event_source.start()

            @typing.no_type_check
            def pause(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.runs = False
                self.event_source.stop()

            @typing.no_type_check
            def forward(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.forwards = True
                self.start()

            @typing.no_type_check
            def backward(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.forwards = False
                self.start()

            @typing.no_type_check
            def oneforward(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.forwards = True
                self.onestep()

            @typing.no_type_check
            def onebackward(self, event=None) -> None:  # noqa: ARG002, ANN001
                self.forwards = False
                self.onestep()

            @typing.no_type_check
            def on_scroll(self, event) -> None:  # noqa: ANN001
                tmp_forwards = self.forwards
                self.forwards = event.button == "up"
                self.onestep()
                self.forwards = tmp_forwards

            @typing.no_type_check
            def onestep(self) -> None:
                if self.i > self.min and self.i < self.max:
                    # increment (decrement) by 1
                    self.i = self.i + 1 if self.forwards else self.i - 1
                else:
                    if self.repeating:
                        # jump to minimum (maximum)
                        self.i = self.min if self.forwards else self.max
                    else:
                        # stay at maximum (minimum)
                        self.i = self.max if self.forwards else self.min

                self.func(self.i)
                self.fig.canvas.draw_idle()

            @typing.no_type_check
            def setup(self, pos) -> None:  # noqa: ANN001
                playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
                divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
                obax = divider.append_axes("right", size="80%", pad=0.05)
                sax = divider.append_axes("right", size="80%", pad=0.05)
                ofax = divider.append_axes("right", size="80%", pad=0.05)
                fax = divider.append_axes("right", size="100%", pad=0.05)
                self.button_back = matplotlib.widgets.Button(playerax, label="$\u25c0$")
                self.button_oneback = matplotlib.widgets.Button(obax, label="$\u29cf$")
                self.button_pause = matplotlib.widgets.Button(sax, label="$\u25a0$")
                self.button_oneforward = matplotlib.widgets.Button(ofax, label="$\u29d0$")
                self.button_forward = matplotlib.widgets.Button(fax, label="$\u25b6$")
                self.button_oneback.on_clicked(self.onebackward)
                self.button_back.on_clicked(self.backward)
                self.button_pause.on_clicked(self.pause)
                self.button_forward.on_clicked(self.forward)
                self.button_oneforward.on_clicked(self.oneforward)
                self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

        class AnimationWithButtons:
            def __init__(
                self,
                fig: Figure,
                ax: plt.Axes,
                A: np.ndarray,  # noqa: N803
                B: np.ndarray,  # noqa: N803
                gammas: Sequence[float],
                data: np.ndarray,
            ) -> None:
                self.index = 0
                self.gammas = gammas
                self.data = data
                self.fig = fig
                self.ax = ax
                self.vmin = data.min()
                self.vmax = data.max()

                norm = mpl.colors.PowerNorm(vmin=self.vmin, vmax=self.vmax, gamma=0.4)
                self.im = self.ax.pcolormesh(
                    A, B, self.data[:, :, self.index], cmap="hot", norm=norm
                )
                self.fig.colorbar(self.im)

                self.anim = Player(
                    fig=self.fig,
                    func=self.update,
                    mini=0,
                    maxi=len(self.gammas) - 1,
                    repeating=False,
                    interval=500,
                    blit=False,
                    pos=(0.05, 0.93),
                )

            def update(self, index: int) -> None:
                self.ax.set_title(f"Energy in Hartree\ngamma:\n{self.gammas[index]:.4g}")
                self.im.set_array(self.data[:, :, index])

        fig, ax = plt.subplots()
        ax.set_xlabel("$\\alpha$")
        ax.set_ylabel("$\\beta$")
        _anim = AnimationWithButtons(
            fig=fig, ax=ax, A=A_small, B=B_small, gammas=gammas, data=energies
        )

        # _anim.anim.save(
        #     self.output_path.joinpath("energy_landscape.mp4"), writer=animation.FFMpegWriter(fps=1)
        # )
        image_path = self.output_path.joinpath("energy_landscape")
        image_path.mkdir(exist_ok=True)
        _anim.anim.save(image_path.joinpath("energy_landscape.png"), writer="imagemagick")
        plt.show()
        plt.close()

    def plot_energy_over_noise(
        self,
        list_of_cx_error_prob: Iterable[float],
        trotter_steps: int | Iterable[int] = 1,
        threshold_gamma: float | Iterable[float] = 1.0,
        max_evals: int = 1000,
        add_title: bool = True,
        skip_uccsdt: bool = False,
    ) -> None:
        """Plots the energy of tVHA depending on the CX error probability.

        Args:
            list_of_cx_error_prob: The CX error probabilities to use for the noise model.
            trotter_steps: The numbers of Trotter steps to use.
                If given as single element, only a single line is plotted.
                If given as list, all list elements are used in sorted order
                creating a line for each number of Trotter steps.
            threshold_gamma: The truncation threshold to use for this plot.
                If given as single element, only a single line is plotted.
                If given as list, all list elements are used in sorted order
                creating a line for each truncation threshold.
            max_evals: Maximum number of function evaluations of the optimization algorithm (SBPLX).
            add_title: whether to add a title to the plot.
            skip_uccsdt: whether to skip calculation of UCCSDT (useful for larger systems
                as it might take lots of time).
        """
        list_of_cx_error_prob = list(list_of_cx_error_prob)
        list_of_trotter_steps = (
            list(trotter_steps) if isinstance(trotter_steps, Iterable) else [trotter_steps]
        )
        list_of_threshold_gamma = self._vha_dummy.get_thresholds_gamma(
            threshold_gamma if isinstance(threshold_gamma, Iterable) else [threshold_gamma]
        )

        # Retrieve all needed energy data
        datapoints = []
        for cx_error_prob in list_of_cx_error_prob:
            for trotter_steps in list_of_trotter_steps:
                for threshold_gamma in list_of_threshold_gamma:
                    datapoints.append(
                        {
                            "ansatz_name": "tVHA",
                            "trotter_steps": trotter_steps,
                            "max_evals": max_evals,
                            "threshold_gamma": threshold_gamma,
                            "cx_error_prob": cx_error_prob,
                        }
                    )
        for cx_error_prob in list_of_cx_error_prob:
            for ansatz_name in ("HEA", "UCCSD") if skip_uccsdt else ("HEA", "UCCSD", "UCCSDT"):
                datapoints.append(
                    {
                        "ansatz_name": ansatz_name,
                        "max_evals": max_evals,
                        "cx_error_prob": cx_error_prob,
                    }
                )

        energy_data = self.get_datapoints(datapoints=datapoints)

        # tVHA
        for id_trotter, trotter_steps in enumerate(list_of_trotter_steps):
            for id_gamma, threshold_gamma in enumerate(list_of_threshold_gamma):
                energies = energy_data[
                    (energy_data["ansatz_name"] == "tVHA")
                    & (energy_data["trotter_steps"] == trotter_steps)
                    & (energy_data["threshold_gamma"] == threshold_gamma)
                    & (energy_data["max_evals"] == max_evals)
                ].sort_values(by="cx_error_prob")["energy"]
                plt.plot(
                    list_of_cx_error_prob,
                    energies,
                    label=f"tVHA γ={threshold_gamma:.3g} "  # noqa: RUF001
                    f"({trotter_steps} Trotter step{'' if trotter_steps == 1 else 's'})",
                    marker=["d", "o", "X", "+", "x"][id_trotter % len(list_of_trotter_steps)],
                    linestyle="dotted",
                    color=color_circle[id_gamma % len(list_of_threshold_gamma)],
                )

        energy_hf = (
            self.problem.reference_energy
            - self.numerical_energies["nuclear_repulsion_energy"]
            - self.numerical_energies["inactive_space_energy"]
        )

        # UCC
        energies_uccsd = energy_data[
            (energy_data["ansatz_name"] == "UCCSD") & (energy_data["max_evals"] == max_evals)
        ].sort_values(by="cx_error_prob")["energy"]
        if not skip_uccsdt:
            energies_uccsdt = energy_data[
                (energy_data["ansatz_name"] == "UCCSDT") & (energy_data["max_evals"] == max_evals)
            ].sort_values(by="cx_error_prob")["energy"]

        # UCCSD
        if not skip_uccsdt:
            uccsd_equals_uccsdt = bool(np.all(np.isclose(energies_uccsd, energies_uccsdt)))
        plt.plot(
            list_of_cx_error_prob,
            energies_uccsd,
            label="UCCSD" if skip_uccsdt or not uccsd_equals_uccsdt else "UCCSD / UCCSDT",
            marker="x",
            linestyle="dashdot",
            color=colors_ucc[0],
        )

        # UCCSDT
        if not skip_uccsdt and not uccsd_equals_uccsdt:
            plt.plot(
                list_of_cx_error_prob,
                energies_uccsdt,
                label="UCCSDT",
                marker="x",
                linestyle="dashdot",
                color=colors_ucc[1],
            )

        # HEA
        energies_hea = energy_data[
            (energy_data["ansatz_name"] == "HEA") & (energy_data["max_evals"] == max_evals)
        ].sort_values(by="cx_error_prob")["energy"]
        plt.plot(
            list_of_cx_error_prob,
            energies_hea,
            label="HEA",
            marker="x",
            linestyle="dashed",
            color=color_hea,
        )

        energy_fci = self.numerical_energies["computed_energy"]

        plt.ylim(
            energy_fci - (energy_hf - energy_fci) * 0.05,
            energy_hf + (energy_hf - energy_fci) * 0.05,
        )
        linthresh = min((x for x in list_of_cx_error_prob if x > 0), default=1e-7)
        plt.xscale("symlog", linthresh=linthresh)
        xmin, xmax = plt.xlim()

        # HF energy
        plt.hlines(
            energy_hf,
            xmin=xmin,
            xmax=xmax,
            label="HF",
            color=color_hf,
            linestyles="solid",
            zorder=0,
        )

        # FCI energy
        plt.hlines(
            energy_fci,
            xmin=xmin,
            xmax=xmax,
            label="FCI",
            color=color_fci,
            linestyles="solid",
            zorder=0,
        )
        plt.fill_between(
            (xmin, xmax),
            energy_fci,
            energy_fci + 0.0015,
            label="chemical accuracy",
            color=color_fci,
            alpha=0.4,
            zorder=0,
        )

        ticks, _ = plt.xticks()
        plt.xticks(ticks=[0, *ticks] if 0 not in ticks else ticks)
        plt.xlim(0, xmax)

        plt.legend()
        plt.xlabel("CNOT depolarization error probability")
        plt.ylabel("Energy in Hartree")
        if add_title:
            plt.title(
                "Energy for different noise levels for tVHA, UCC and HEA " + self.molecule_name
            )

        filename = f"{self.molecule_name}_energy_over_noise_"
        filename += "_".join(f"{cx_error_prob:.4g}" for cx_error_prob in list_of_cx_error_prob)
        filename += ".svg"
        plt.savefig(self.output_path.joinpath(filename), format="svg")
        plt.close()


def plot_parameter_count(
    output_path: Path,
    problem: ElectronicStructureProblem,
    mapper: FermionicMapper,
    molecule_names: Sequence[str],
    log_y: bool = False,
    add_title: bool = True,
) -> None:
    """Plots the number of parameters for each ansatz and molecule.

    Compares the required number of parameters for the different ansätze.
    It is shown for all given molecules in a single plot.

    Args:
        output_path: the folder for saving the figure.
        problem: the formulation of the electronic structure ploblem.
        mapper: the mapper from Fermionic operator to spin operator.
        molecule_names: the names of the molecules for the comparison.
        log_y: whether to use a logarithmic scale for the y axis.
        add_title: whether to add a title to the plot.
    """
    molecule_names = list(molecule_names)

    num_parameters_vha = []
    num_parameters_uccsd = []
    num_parameters_uccsdt = []
    num_parameters_hea = []
    for molecule_name in molecule_names:
        vha_plots = VHAPlots(
            output_path=output_path.joinpath(f"plots_{molecule_name.replace('_', '')}"),
            molecule_name=molecule_name,
            problem=problem,
            mapper=mapper,
        )
        num_parameters_vha.append(vha_plots.get_VHA_circuit_counts()["num_parameters"][0])
        num_parameters_uccsd.append(
            vha_plots.get_reference_UCC_circuit_counts(excitations=(1, 2))["num_parameters"]
        )
        num_parameters_uccsdt.append(
            vha_plots.get_reference_UCC_circuit_counts(excitations=(1, 2, 3))["num_parameters"]
        )
        num_parameters_hea.append(vha_plots.get_reference_HEA_circuit_counts()["num_parameters"])

    ax = plt.figure().gca()

    x_offset = 0.24
    width = 0.21

    # VHA
    plt.bar(
        x=np.arange(len(molecule_names)) - x_offset,
        height=[5 * num for num in num_parameters_vha],
        width=width,
        edgecolor=colors_tvha[0],
        linestyle="dotted",
        fill=False,
        label="VHA (5 Trotter steps)",
    )
    plt.bar(
        x=np.arange(len(molecule_names)) - x_offset,
        height=[2 * num for num in num_parameters_vha],
        width=width,
        edgecolor=colors_tvha[0],
        fill=False,
        label="VHA (2 Trotter steps)",
    )
    plt.bar(
        x=np.arange(len(molecule_names)) - x_offset,
        height=num_parameters_vha,
        width=width,
        color=colors_tvha[0],
        edgecolor=colors_tvha[0],
        label="VHA (1 Trotter step)",
    )

    # UCC
    plt.bar(
        x=np.arange(len(molecule_names)),
        height=num_parameters_uccsdt,
        width=width,
        edgecolor=colors_ucc[0],
        fill=False,
        label="UCCSDT",
    )
    plt.bar(
        x=np.arange(len(molecule_names)),
        height=num_parameters_uccsd,
        width=width,
        tick_label=[
            "$H_2 / CH_2$" if name == "H_2" else "$" + name + "$" for name in molecule_names
        ],
        color=colors_ucc[0],
        edgecolor=colors_ucc[0],
        label="UCCSD",
    )

    # HEA
    plt.bar(
        x=np.arange(len(molecule_names)) + x_offset,
        height=[5 * num / 3 for num in num_parameters_hea],
        width=width,
        edgecolor=color_hea,
        fill=False,
        label="HEA (5 layers)",
    )
    plt.bar(
        x=np.arange(len(molecule_names)) + x_offset,
        height=num_parameters_hea,
        width=width,
        color=color_hea,
        label="HEA (3 layers)",
    )

    plt.yscale("log" if log_y else "linear")
    if not log_y:
        _, ymax = plt.ylim()
        plt.ylim(0, ymax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if add_title:
        plt.title(
            "Parameter count for tVHA, UCC and HEA "
            f"({' '.join(['$' + name + '$' for name in molecule_names])})"
        )
    plt.ylabel("Number of parameters")

    handles, labels = plt.gca().get_legend_handles_labels()
    new_order = [2, 1, 0, 4, 3, 6, 5]
    plt.legend([handles[idx] for idx in new_order], [labels[idx] for idx in new_order])

    filename = "parameter_count"
    if log_y:
        filename += "_logy"
    filename += ".svg"
    plt.savefig(output_path.joinpath(filename), format="svg")
    plt.close()


def find_somos(local_occ: list[list[int]]) -> list[int]:
    """Identify SOMO (singly occupied molecular orbitals) indices.

    Args:
        local_occ:           UHF Orbital occupation.

    Returns:
        idx:                 Identified SOMOs indices.

    """
    if not len(local_occ[0]) == len(local_occ[1]):
        raise ValueError("This function works with UHF or UKS formalism.")
    occ_diff = np.array(local_occ[0]) - np.array(local_occ[1])
    somos = np.where(occ_diff == 1)[0]
    return somos.tolist()


def conv_mf(
    local_mol: gto.Mole, dm0: np.ndarray | None = None, stability_loop: bool = False
) -> scf.hf.SCF:
    """Create an scf instance and converge it.

    Args:
        local_mol:          PySCF Mole object.
        dm0:                Initial guess for the density matrix.
        stability_loop:     Whether to use the stability loop.

    Returns:
        mean field object.
    """
    # Initial SCF guess.
    initial_guess = "minao"

    mf = scf.UHF(local_mol)
    mf.init_guess = initial_guess
    mf.max_cycle = 250
    mf = mf.newton()
    mf.kernel(dm0=dm0)
    if stability_loop:
        mo_new = mf.stability()[0]
        while not np.all(np.isclose(mo_new, mf.mo_coeff)) or not mf.converged:
            mf.kernel(dm0=mf.make_rdm1(mo=mo_new))
            mo_new = mf.stability()[0]
    if not mf.converged:
        raise ValueError("SCF calculations did not converge.")
    return mf


def get_minimal_active_space_size(lowest_spin: int) -> tuple[int, int]:
    """Gets the size of the minimal active space.

    Minimal active space is defined as Number of singly occupied in the high spin configurations.
    Meaning (3 number of electrons, 3 number of orbitals) for the CH and NO3, and (2, 2) for CH_2.

    Returns: tuple(active_space_size, active_space_electrons)
    """
    if lowest_spin == 1:
        return (3, 3)
    if lowest_spin == 0:
        return (2, 2)
    raise ValueError("The system spin must be either 0 or 1.")


def get_electronic_structure_problem(
    molecule_name: str,
    basis_set: str = "def2-svp",
    path_data_files: Path = Path(__file__).parent,
) -> ElectronicStructureProblem:
    """Initializes ElectronicStructureProblem from molecule name."""
    # molecules without active space
    if molecule_name.startswith("H_"):
        try:
            number_of_atoms = int(molecule_name.split("_")[-1])
        except Exception as err:
            raise ValueError(f"Unsupported molecule {molecule_name}") from err
        molecule = MoleculeInfo(
            symbols=["H"] * number_of_atoms,
            coords=[(0.0, 0.0, 0.74279 * i) for i in range(number_of_atoms)],
            multiplicity=1,
            charge=0,
        )
        driver = PySCFDriver.from_molecule(molecule)
        driver.basis = basis_set
        return driver.run()
    if molecule_name == "LiH":
        molecule = MoleculeInfo(
            symbols=["Li", "H"],
            coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.596)],
            multiplicity=1,
            charge=0,
        )
        driver = PySCFDriver.from_molecule(molecule)
        driver.basis = basis_set
        return driver.run()
    if molecule_name == "H_2O":
        molecule = MoleculeInfo(
            symbols=["O", "H", "H"],
            coords=[(0.0, 0.0, 0.0), (0.758602, 0.0, 0.504284), (0.758602, 0.0, -0.504284)],
            multiplicity=1,
            charge=0,
        )
        driver = PySCFDriver.from_molecule(molecule)
        driver.basis = basis_set
        return driver.run()

    # molecules with active space
    if molecule_name == "CH":
        input_structure = "ch_opt.xyz"
        lowest_spin = 1
        system_charge = 0
    elif molecule_name == "CH_2":
        input_structure = "ch2_opt.xyz"
        lowest_spin = 0
        system_charge = 0
    elif molecule_name == "NO_3":
        input_structure = "no3_opt.xyz"
        lowest_spin = 1
        system_charge = 0
    else:
        raise ValueError(f"Unsupported molecule {molecule_name}")
    _, active_space_electrons = get_minimal_active_space_size(lowest_spin=lowest_spin)

    mol = gto.Mole()
    mol.atom = str(path_data_files.joinpath(input_structure))
    mol.basis = basis_set
    mol.verbose = 4

    # High spin is usually easier to converge for the mean-field theory.
    mol.spin = lowest_spin + 2
    mol.charge = system_charge
    mol.build()

    # Calculate UHF for the high spin state.
    # High spin
    hs_mf = conv_mf(mol, stability_loop=False)

    # Find the indices of singly occupied molecular orbitals (from the high spin).
    somos_idx = find_somos(hs_mf.mo_occ)

    driver = PySCFDriver(
        atom=[f"{i[0]} {i[1][0]} {i[1][1]} {i[1][2]}" for i in mol._atom],  # noqa: SLF001
        charge=system_charge,
        spin=lowest_spin,
        basis=basis_set,
    )
    full_problem = driver.run()
    active_space_transformer = ActiveSpaceTransformer(
        num_electrons=(
            int((active_space_electrons + lowest_spin) / 2),
            int((active_space_electrons - lowest_spin) / 2),
        ),
        num_spatial_orbitals=len(somos_idx),
        active_orbitals=somos_idx,
    )
    return active_space_transformer.transform(full_problem)


def main() -> None:
    """Main function to create the plots."""
    # ---------------------------------------------------------------------------------------------
    # --- SETTINGS --------------------------------------------------------------------------------
    # molecule_name = "H_2"
    # molecule_name = "H_4"
    molecule_name = "LiH"
    # molecule_name = "CH_2"
    # molecule_name = "NO_3" # more complex due to non-diagonal fock operator
    # molecule_name = "CH" # HF is already within chemical accuracy

    mapper = JordanWignerMapper()

    config_path = Path(__file__).parent.joinpath("vha_plots_config.yaml")

    # ---------------------------------------------------------------------------------------------
    # Load configuration
    yaml = YAML(typ="safe")
    with config_path.open("r") as f:
        config = yaml.load(f)

    # Get global plotting settings
    plotting_config = config["plotting"]

    # Get molecule-specific settings
    molecules_config = config["molecules"]
    molecule_plots_config = molecules_config.get(molecule_name, {})

    # Get basis_set with fallback to "def2-svp"
    basis_set = molecule_plots_config.get("basis_set", "def2-svp")

    # Create output folder
    output_dir_template = plotting_config.get("output_dir_template", "plots/plots_{molecule}")

    output_folder = Path(__file__).parent.joinpath(
        output_dir_template.format(molecule=molecule_name.replace("_", ""))
    )
    output_folder.mkdir(exist_ok=True)

    problem = get_electronic_structure_problem(
        molecule_name=molecule_name, basis_set=basis_set, path_data_files=output_folder.parent
    )

    vha = VariationalHamiltonianAnsatz(problem=problem, trotter_steps=1, mapper=mapper)

    vha_plots = VHAPlots(
        output_path=output_folder,
        molecule_name=molecule_name,
        problem=problem,
        mapper=mapper,
        force_sequential=True,
    )

    # All energies are given as electronic energies without the nuclear repulsion energy
    # unless stated explicitly to be the total energy.

    if logger.getEffectiveLevel() <= logging.DEBUG:
        datapoint = vha_plots.get_datapoints(trotter_steps=1, threshold_gamma=1.0)
        energy_statevector = datapoint["energy"].iloc[0]
        optimal_parameters = datapoint["optimal_parameters"].iloc[0]

        if len(optimal_parameters) == 3:
            optimal_parameters_string = (
                f"α={optimal_parameters[0]:.5g}, "  # noqa: RUF001
                f"β={optimal_parameters[1]:.5g}, "
                f"γ={optimal_parameters[2]:.5g}"  # noqa: RUF001
            )
        else:
            optimal_parameters_string = str(optimal_parameters)

        result_string = (
            f"Total ground state energy in Hartree Fock approximation: {problem.reference_energy:.3f}\n"
            "FCI energy (numerical diagonalization; only active space for active space calculations): "
            f"{vha_plots.numerical_energies['computed_energy'] + vha_plots.numerical_energies['nuclear_repulsion_energy']:.3f}\n"
            "Total ground state energy (VHA statevector simulator): "
            f"{energy_statevector + vha_plots.numerical_energies['nuclear_repulsion_energy']:.3f}\n"
            "Improvement over HF approx: "
            f"{problem.reference_energy - (energy_statevector + vha_plots.numerical_energies['nuclear_repulsion_energy']):.3g}\n"
            "Difference to FCI energy: "
            f"{vha_plots.numerical_energies['computed_energy'] - energy_statevector:.3g}\n"
            f"Optimal parameters: {optimal_parameters_string}"
        )
        print(result_string)

    add_title = plotting_config.get("add_title", False)

    # ---------------------------------------------------------------------------------------------
    # --- GENERATE PLOTS --------------------------------------------------------------------------

    if plotting_config.get("plot_histograms", False):
        print("Plotting histograms...")
        for log_x in (True, False):
            for log_y in (True, False):
                vha_plots.plot_histogram(
                    hamiltonian=vha.hamilton_operator, log_x=log_x, log_y=log_y, add_title=add_title
                )
        print("Done.")

    if plotting_config.get("plot_density_distributions", False):
        print("Plotting cumulated density distribution...")
        vha_plots.plot_cumulated_density_distribution_all_terms(
            hamiltonian=vha.hamilton_operator, add_title=add_title
        )
        vha_plots.plot_cumulated_density_distribution_noncoulomb_terms(
            hamiltonian=vha.hamilton_operator, add_title=add_title
        )
        print("Done.")

    if plotting_config.get("plot_cnot_count", False):
        print("Plotting CNOT count over truncation threshold...")
        for log_y in (True, False):
            vha_plots.plot_cnot_count_over_truncation_threshold(log_y=log_y, add_title=add_title)
        print("Done.")

    if plotting_config.get("plot_circuit_depth", False):
        print("Plotting circuit depth over truncation threshold...")
        for log_y in (True, False):
            vha_plots.plot_circuit_depth_over_truncation_threshold(
                log_y=log_y, add_cnot_count=False, add_title=add_title
            )
        print("Done.")

    if plotting_config.get("plot_energy_landscape", False):  # be aware: highly inefficient
        print("Plotting energy landscape...")
        vha_plots.plot_energy_landscape(
            alphas=np.linspace(-1 * 3, 2 * 3, 50),
            betas=np.linspace(-1 * 9, 2 * 9, 50),
            gammas=np.linspace(-1 * 21, 2 * 21, 50),
        )
        print("Done.")

    if plotting_config.get("plot_energy_over_threshold", False):
        plot_config = molecule_plots_config.get("plot_energy_over_threshold", {})
        print("Plotting energy over truncation threshold...")
        vha_plots.plot_energy_over_truncation_threshold(add_title=add_title, **plot_config)
        print("Done.")

    if plotting_config.get("plot_energy_over_trotter", False):
        plot_config = molecule_plots_config.get("plot_energy_over_trotter", {})
        print("Plotting energy over Trotter steps...")
        vha_plots.plot_energy_over_trotter_steps(add_title=add_title, **plot_config)
        print("Done.")

    if plotting_config.get("plot_energy_heatmap", False):
        plot_config = molecule_plots_config.get("plot_energy_heatmap", {})
        print("Plotting energy over truncation threshold and Trotter steps...")
        vha_plots.plot_energy_heatmap(add_title=add_title, **plot_config)
        print("Done.")

    if plotting_config.get("plot_energy_over_noise", False):
        plot_config = molecule_plots_config.get("plot_energy_over_noise", {})
        print("Plotting energy over noise...")
        vha_plots.plot_energy_over_noise(add_title=add_title, **plot_config)
        print("Done.")

    if plotting_config.get("plot_error_estimate", False):
        plot_config = molecule_plots_config.get("plot_error_estimate", {})
        print("Plotting error estimate over truncation threshold...")
        vha_plots.plot_energy_over_truncation_threshold(
            show_truncation_error_estimate=True, add_title=add_title, **plot_config
        )
        print("Done.")

    if plotting_config.get("plot_parameter_count", False):
        print("Plotting number of parameters over ansatz...")
        plot_parameter_count(
            output_path=output_folder.parent,
            problem=problem,
            mapper=mapper,
            molecule_names=("H_2", "H_4", "LiH"),
            log_y=False,
            add_title=add_title,
        )
        plot_parameter_count(
            output_path=output_folder.parent,
            problem=problem,
            mapper=mapper,
            molecule_names=("H_2", "H_4", "LiH"),
            log_y=True,
            add_title=add_title,
        )
        print("Done.")


if __name__ == "__main__":
    main()
