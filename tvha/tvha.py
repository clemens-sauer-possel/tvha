"""Truncated Variational Hamiltonian Ansatz (tVHA)."""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression, ParameterVector
from qiskit.circuit.library import BlueprintCircuit, PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.mappers.fermionic_mapper import FermionicMapper
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import ElectronicStructureProblem

from tvha.fermionic_operator import FermionicOp

# from .fermionic_operator import FermionicOp

logger = logging.getLogger(__name__)

# TODO: switch to parity mapper for future calculations / enable 2-qubit reduction


class VariationalHamiltonianAnsatz(BlueprintCircuit):
    """Implements truncated Variational Hamiltonian Ansatz (tVHA).

    Implements the Variational Hamiltonian Ansatz (VHA) as well as its variation, the
    truncated Variational Hamiltonian Ansatz (tVHA).

    VHA consists of alternating layers of Fock operator (known ground state Hamiltonian) and
    full Hamiltonian.
    In this implementation, the full Hamiltonian is split into one-body terms, Coulombic two-body
    terms, and the remaining non-Coulomb two-body terms.
    Truncation in tVHA is performed on the non-Coulomb two-body terms since their
    circuit representation consists of multiple CNOT gates and is thus the most problematic part
    on NISQ devices.

    For further information about the theory behind (adiabatic evolution) see
    https://arxiv.org/pdf/quant-ph/0001106.pdf
    Nice overview of adiabatic theory for quantum computing:
    https://cs269q.stanford.edu/lectures/adiabatic_note.pdf
    """

    def __init__(
        self,
        problem: ElectronicStructureProblem,
        mapper: FermionicMapper | None,
        trotter_steps: int = 1,
        threshold_gamma: float = 1.0,
        trotterization_order: int = 1,
        name: str | None = "tVHA",
        insert_barriers: bool = False,
    ) -> None:
        """Variational Hamiltonian Ansatz.

        Initializes Variational Hamiltonian Ansatz (VHA) as well as its variation, the
        truncated Variational Hamiltonian Ansatz (tVHA).

        Args:
            problem: The representation of the molecule as ElectronicStructureProblem
            mapper: maps fermionic operators to qubit operators.
            trotter_steps: Number of steps to discretize the adiabatic time evolution.
                In the limit of infinite disretization steps the adiabatic theorem guarantees
                to yield the ground state of the final Hamiltonian.
            threshold_gamma: Threshold for filtering terms in the Hamiltonian H_gamma
                for construction of the ansatz circuit.
                A threshold of 0.9 means that 90% are considered.
                The terms are sorted based on the coefficient value (or rather `abs(coeff_value)`)
                before applying the threshold.
                The terms are filtered such that terms are added until
                `threshold*sum(coeff_values)` is reached; due to the calculation method
                the final threshold might be (potentially significantly) larger than
                the initial threshold argument; for larger molecules, these deviations should
                become smaller due to the large number of terms in the Hamiltonian.
                If a threshold_gamma of 1 (i.e. 100%) is chosen, this resembles the VHA algorithm;
                any other threshold_gamma means that the tVHA algorithm is implemented.
            trotterization_order: Scheme to expand exponentiation of non-commuting operators;
                typically 1st order Trotterization exp(A+B)/approx exp(A)*exp(B) is sufficient.
                Increasing the number of Trotter steps is most likely more efficient
                to yield higher accuracy.
            name: Name of the block in the overall circuit.
            insert_barriers: Whether to insert barriers between different building blocks.
                Solely for visualization purposes.
        """
        self._epsilon = 1e-11  # tolerance for floats to be considered equal
        self.problem = problem
        if isinstance(mapper, ParityMapper):
            raise ValueError("ParityMapper is not supported yet.")
        self.mapper = mapper or JordanWignerMapper()
        self.trotterization_order = trotterization_order
        self._electronic_energy = self.problem.hamiltonian

        self.trotter_steps = int(trotter_steps)

        self.hamilton_operator = self.get_hamilton_operator()
        self.fock_operator = self.get_fock_operator()
        self._interaction_hamiltonian = self._get_interaction_hamiltonian()

        self.possible_thresholds_gamma = self._get_possible_thresholds_gamma()
        _, self._threshold_gamma = self.get_threshold_gamma(initial_threshold_gamma=threshold_gamma)
        self._hamiltonian_alpha = self._get_hamiltonian_alpha()
        self._hamiltonian_beta = self._get_hamiltonian_beta()
        self._hamiltonian_gamma = self._get_hamiltonian_gamma(threshold_gamma=self._threshold_gamma)

        self.insert_barriers = insert_barriers
        self._alpha_parameter_vector = ParameterVector(
            name="α",  # noqa: RUF001
            length=self.trotter_steps,
        )
        self._beta_parameter_vector = ParameterVector(name="β", length=self.trotter_steps)
        self._gamma_parameter_vector = ParameterVector(
            name="γ",  # noqa: RUF001
            length=self.trotter_steps,
        )
        self.preferred_initial_parameters = self._get_preferred_initial_parameters()

        super().__init__(name=name)

        self._num_qubits: int | None = None
        self.num_qubits = self.problem.num_spin_orbitals  # type: ignore

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        return self._num_qubits if self._num_qubits is not None else 0

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits for the n-local circuit.

        Args:
            num_qubits: The new number of qubits.
        """
        if self._num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self._num_qubits = int(num_qubits)
            self.qregs = [QuantumRegister(num_qubits, name="q")]

    def get_hamilton_operator(self) -> FermionicOp:
        """Returns the Hamiltonian of the systems.

        Be aware, that qiskit notation sees an ElectronicEnergy object as "Hamiltonian" while
        this method returns a FermionicOp object (i.e. the "pure" Hamilton operator).

        The Hamiltonian is optimized in the sense that the amount of terms is reduced based
        on information redundancies (due to antisymmetric and hermitian properties).
        """
        hamilton_operator = self._electronic_energy.second_q_op().get_compressed_hamiltonian()
        if not hamilton_operator.is_hermitian():
            raise ValueError("Hamiltonian is not hermitian.")
        return hamilton_operator

    def get_fock_operator(self) -> FermionicOp:
        """Returns the Fock operator (typically associated with parameter 'alpha' in VHA)."""
        density = ElectronicIntegrals.from_raw_integrals(
            np.diag(self.problem.orbital_occupations),  # type: ignore
            h1_b=np.diag(self.problem.orbital_occupations_b),  # type: ignore
        )
        fock_operator = (
            ElectronicEnergy(self._electronic_energy.fock(density=density)).second_q_op().simplify()
        )
        fock_operator._data = {  # noqa: SLF001
            label: coeff.real for label, coeff in fock_operator.items()
        }
        return fock_operator

    def _get_interaction_hamiltonian(self) -> FermionicOp:
        """Returns the interaction Hamiltonian.

        Full Hamiltonian minus Fock operator. Thus, one-body and two-body terms are present.
        This Hamiltonian is not used directly to create any circuit but as an intermediate to get
        H_β and H_γ.
        """  # noqa: RUF002
        interaction_hamiltonian = self.hamilton_operator - self.fock_operator
        interaction_hamiltonian = interaction_hamiltonian.get_compressed_hamiltonian()

        # Sanity check if the Hamiltonian is initialized correctly until this point
        if not interaction_hamiltonian.is_hermitian():
            raise ValueError("Interaction Hamiltonian is not hermitian.")

        # Sanity check if the Hamiltonian contains only antisymmetric terms with i<j and k<l
        if logger.getEffectiveLevel() <= logging.DEBUG:
            for label in interaction_hamiltonian.get_two_body_hamiltonian():
                op = [lbl.split("_") for lbl in label.split()]
                if op[0][1] > op[1][1] or op[2][1] > op[3][1]:
                    # Antisymmetric terms with i>j and k>l
                    # Strictly speaking it should be '>=' instead of '>' but '.simplify()' method
                    # invoked in 'FermionicOp.compress_hamiltonian()' ensures already
                    # that terms i=j and k=l are eliminated.
                    raise ValueError(
                        "Unexpected values in Hamiltonian (antisymmetric counterparts of the "
                        "given value). Please make sure to use "
                        "the compressed Hamiltonian from '%s.get_compressed_hamiltonian'",
                        FermionicOp.__qualname__,
                    )

        return interaction_hamiltonian

    def _get_hamiltonian_alpha(self) -> FermionicOp:
        """Returns the Hamiltonian associated with parameter alpha.

        The Hamiltonian is split into 3 parts:
        1. Fock operator H_alpha:
            effective one-body operator taking into account some of the interaction effects
            in a mean-field-ish accurate.
        2. Interaction Hamiltonian:
            Full Hamiltonian minus Fock operator. Thus, one-body and two-body terms are present.
            This Hamiltonian is further split into
            2.1 H_beta: one-body terms and coulomb two-body terms.
            2.2 H_gamma: non-Coulomb two-body terms.
        """
        return self.fock_operator

    def _get_hamiltonian_beta(self) -> FermionicOp:
        """Returns the Hamiltonian associated with parameter beta.

        The Hamiltonian is split into 3 parts:
        1. Fock operator H_alpha:
            effective one-body operator taking into account some of the interaction effects
            in a mean-field-ish accurate.
        2. Interaction Hamiltonian:
            Full Hamiltonian minus Fock operator. Thus, one-body and two-body terms are present.
            This Hamiltonian is further split into
            2.1 H_beta: one-body terms and coulomb two-body terms.
            2.2 H_gamma: non-Coulomb two-body terms.
        """
        beta_terms_one_body = dict(self._interaction_hamiltonian.get_one_body_hamiltonian().items())
        beta_terms_two_body = dict(
            self._interaction_hamiltonian.get_two_body_hamiltonian_coulomb_terms().items()
        )

        beta_terms = beta_terms_one_body | beta_terms_two_body

        return FermionicOp(
            data=beta_terms,
            num_spin_orbitals=self._interaction_hamiltonian.num_spin_orbitals,
        )

    def _get_sorted_noncoulomb_two_body_terms(self) -> dict[str, float]:
        """Gets the non-Coulomb two-body terms (aka gamma terms) in descending order.

        The descending order is based on the absolute value of the terms
        but the terms' signs are not altered and can still be negative.

        For truncation, this method is not directly suitable since it doesn't take hermitian
        counterparts into account, i.e. the intuitive idea of truncating this list will possibly
        result in a non-hermitian Hamiltonian.
        Use '_get_hamiltonian_gamma' instead."""
        return dict(
            sorted(
                self._interaction_hamiltonian.get_two_body_hamiltonian_noncoulomb_terms().items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
        )

    def _get_possible_thresholds_gamma(self) -> tuple[float]:
        """Gets the available gamma thresholds, i.e. the cumulated sum of the gamma terms.

        The i-th element is the sum of all non-Coulomb two-body terms (aka gamma terms) from 0 to i.
        The zeroth element of the returned array is zero to resemble the case
        where no terms are added.
        By construction the returned tuple of possible thresholds is sorted in ascending order.
        The information, which two-body terms a^dagger_i a^dagger_j a_k a_l are associated
        with these elements, is not preserved within this method;
        if needed, use '_get_sorted_noncoulomb_two_body_terms' or '_get_hamiltonian_gamma' instead.
        """
        gamma_terms_sorted = self._get_sorted_noncoulomb_two_body_terms()
        sum_of_all_coeffs = sum(abs(coeff) for coeff in gamma_terms_sorted.values())
        cumsum = [0.0]
        labels_already_included = []
        for label, coeff in gamma_terms_sorted.items():
            # Add the hermitian counterpart to ensure the final Hamiltonian gamma stays hermitian
            label_hermitian_counterpart, switch_sign = FermionicOp.get_permuted_label(
                FermionicOp.get_label_of_hermitian_counterpart(label)
            )
            if switch_sign:
                logger.warning(
                    "Hermitian counterpart found with different sign than the original term. "
                    "Hermitian counterparts should always have the same sign. "
                    "Please check the code for bugs."
                )

            if (
                label in labels_already_included
                or label_hermitian_counterpart in labels_already_included
            ):  # i.e. already added (via hermitian counterpart)
                continue
            if label == label_hermitian_counterpart:
                labels_already_included.append(label)
                cumsum.append(cumsum[-1] + abs(coeff))
            else:
                coeff_hermitian_counterpart = gamma_terms_sorted[label_hermitian_counterpart]
                if not np.isclose(
                    coeff, coeff_hermitian_counterpart, rtol=self._epsilon, atol=self._epsilon
                ):
                    logger.warning(
                        "Coefficient %s and its hermitian counterpart %s have "
                        "different values (%s and %s) although they are expected to be the same.",
                        label,
                        label_hermitian_counterpart,
                        coeff,
                        coeff_hermitian_counterpart,
                    )
                labels_already_included.extend([label, label_hermitian_counterpart])
                cumsum.append(cumsum[-1] + abs(coeff) + abs(coeff_hermitian_counterpart))
            # Above checks the case if the hermitian counterpart is the same
            # as the original label, i.e.
            # "+_0 +_1 -_0 -_1" would be skipped,
            # "+_0 +_1 -_2 -_3" would add its hermitian counterpart "+_3 +_2 -_1 -_0" or
            # rather its antisymmetric permuted counterpart "+_2 +_3 -_0 -_1"

        if not np.isclose(cumsum[-1], sum_of_all_coeffs):
            raise ValueError(
                "The cummulative sum of all coefficients has a wrong value. "
                "Please check the code for bugs."
            )
        possible_thresholds_gamma = np.array(cumsum) / sum_of_all_coeffs
        if np.any(possible_thresholds_gamma < -self._epsilon) or np.any(
            possible_thresholds_gamma > (1.0 + self._epsilon)
        ):
            raise ValueError(
                "The truncation threshold must be between zero and one. Check the code for bugs."
            )
        return tuple(round(threshold, 12) for threshold in possible_thresholds_gamma)

    def get_threshold_gamma(self, initial_threshold_gamma: float) -> tuple[int, float]:
        """Gets the final truncation threshold for the tVHA ansatz and its index/ordinal number.

        Returns: tuple[index, threshold_gamma]"""
        index = int(
            np.searchsorted(self.possible_thresholds_gamma, initial_threshold_gamma, side="left")
        )
        # Take care of those edge cases where the initial threshold maps to one of the
        # possible thresholds but might slightly deviate due to float representation
        if index > 0 and np.isclose(
            self.possible_thresholds_gamma[index - 1],
            initial_threshold_gamma,
            rtol=self._epsilon,
            atol=self._epsilon,
        ):
            return index - 1, self.possible_thresholds_gamma[index - 1]
        # All other cases
        return index, self.possible_thresholds_gamma[index]

    def get_thresholds_gamma(self, initial_thresholds_gamma: Iterable[float] | None) -> list[float]:
        """Gets sorted list of final truncation thresholds with removed duplicates."""
        if initial_thresholds_gamma is None:
            return self.possible_thresholds_gamma
        return sorted({self.get_threshold_gamma(t)[1] for t in initial_thresholds_gamma})

    def _get_hamiltonian_gamma(self, threshold_gamma: float = 1.0) -> FermionicOp:
        """Returns the Hamiltonian associated with parameter gamma.

        The Hamiltonian is split into 3 parts:
        1. Fock operator H_alpha:
            effective one-body operator taking into account some of the interaction effects
            in a mean-field-ish accurate.
        2. Interaction Hamiltonian:
            Full Hamiltonian minus Fock operator. Thus, one-body and two-body terms are present.
            This Hamiltonian is further split into
            2.1 H_beta: one-body terms and coulomb two-body terms.
            2.2 H_gamma: non-Coulomb two-body terms.

        Args:
            threshold_gamma: Determines how many of the two-body terms are kept/discarded.
                The one-body terms are always kept.
                The two-body Coulomb terms are always kept.
                The non-Coulomb two-body terms are filtered based on the threshold.
                The truncation threshold is given as percentage, reading as follows:
                If the threshold is e.g. 0.9, then the largest 90% of the prefactors are returned
                and the smallest 10% are discarded.
                Since there is a finite number of coefficients with potentially arbitrary values,
                it cannot be guaranteed that the percentage is perfectly matched;
                instead the largest 0.9+epsilon percent are returned in above example
                (if you are interested in the final truncation threshold, you can you
                method 'get_threshold_gamma').
                Terms are added until `threshold*sum(coeff_values)` is reached;
                due to the calculation method the final threshold might be
                (potentially significantly) larger than the initial threshold argument.
                If multiple operators have exactly the same coefficient and the threshold
                would be reached adding only part of them, then (quite arbitrarily)
                just the first ones arising in the data will be added.
                This weighting based on the prefactors does not guarantee that these have
                the same effect on the final ground state energy but are a good educated guess.
        Returns: FermionicOp
        """
        if threshold_gamma < -self._epsilon or threshold_gamma > 1.0 + self._epsilon:
            raise ValueError(
                f"The threshold is a percentage and must be between 0 and 1, not {threshold_gamma}"
            )

        gamma_terms_sorted = self._get_sorted_noncoulomb_two_body_terms()

        sum_of_all_coeffs = sum(abs(coeff) for coeff in gamma_terms_sorted.values())
        threshold_absolute_value = sum_of_all_coeffs * threshold_gamma

        gamma_terms_truncated = {}
        cumsum = 0.0
        for label, coeff in gamma_terms_sorted.items():
            if cumsum >= threshold_absolute_value:
                break
            if label in gamma_terms_truncated:  # i.e. already added via hermitian counterpart
                continue

            label_hermitian_counterpart, switch_sign = FermionicOp.get_permuted_label(
                FermionicOp.get_label_of_hermitian_counterpart(label)
            )
            if switch_sign:
                logger.warning(
                    "Hermitian counterpart found with different sign than the original term. "
                    "Hermitian counterparts should always have the same sign. "
                    "Please check the code for bugs."
                )

            coeff = coeff.real
            if not np.isclose(coeff, gamma_terms_sorted[label_hermitian_counterpart]):
                raise ValueError(
                    "Not-Hermitian operator encountered. Please check the code for bugs."
                )

            if label == label_hermitian_counterpart:
                cumsum += abs(coeff)
                gamma_terms_truncated[label] = coeff
            else:
                cumsum += 2 * abs(coeff)
                gamma_terms_truncated[label] = coeff
                gamma_terms_truncated[label_hermitian_counterpart] = coeff
            # Above checks the case if the hermitian counterpart is the same
            # as the original label, i.e.
            # "+_0 +_1 -_0 -_1" would be skipped,
            # "+_0 +_1 -_2 -_3" would add its hermitian counterpart "+_3 +_2 -_1 -_0" or
            # rather its antisymmetric permuted counterpart "+_2 +_3 -_0 -_1"

        return FermionicOp(
            data=gamma_terms_truncated,
            num_spin_orbitals=self._interaction_hamiltonian.num_spin_orbitals,
        )

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of qubits specified.")

        if self.problem is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No electronic structure problem specified.")

        return valid

    def _get_preferred_initial_parameters(self) -> dict[Parameter, float]:
        """Returns the preferred initial parameters of the VHA ansatz.

        The initial parameters are chosen as described in equation (3)
        in https://arxiv.org/pdf/1811.04476.pdf /
        https://iopscience.iop.org/article/10.1088/2058-9565/ab1e85/meta.
        The factor τ mentioned in the paper is set in the _build method
        such that the prefactors here are normalized to 1.
        """
        preferred_initial_parameters = {}
        # Constant alpha values
        for alpha_s in self._alpha_parameter_vector:
            preferred_initial_parameters[alpha_s] = 1.0
        # Increasing beta values
        for s, beta_s in enumerate(self._beta_parameter_vector):
            preferred_initial_parameters[beta_s] = (s + 1) / self.trotter_steps
        # Increasing gamma values
        for s, gamma_s in enumerate(self._gamma_parameter_vector):
            preferred_initial_parameters[gamma_s] = (s + 1) / self.trotter_steps
        return preferred_initial_parameters

    def get_initial_point(self) -> list[float]:
        """Gets the initial point for the ansatz parameters.

        The initial parameters are chosen as described in equation (3)
        in https://arxiv.org/pdf/1811.04476.pdf

        Since qiskit does some non-intuitive sorting of the parameters,
        this method takes care of returning the values as 'initial_point' (i.e. as list)
        which is sorted in the same order as qiskit's parameters (i.e. object self.parameters).

        If there is the chance to pass a dictionary to the parameter binding function
        (e.g. an estimator's 'run' method),
        passing the dictionary 'self.preferred_initial_parameters' is preferred
        since it is based on far less (implicit) assumptions on implementation details.
        """
        logger.info(
            "If possible, use the variable 'preferred_initial_parameters' of 'VHA' class object "
            "to get the initial parameters "
            "since its order is far less implementation-dependent (and thus less error-prone) "
            "than this list."
        )
        if not self._is_built:
            self._build()
        return [self.preferred_initial_parameters[param] for param in self.parameters]

    def _build(self) -> None:
        """If not already built, build the circuit.

        As side effect, it sets the preferred_initial_parameters
        (not possible as stand-alone method...).
        """
        if self._is_built:
            return

        super()._build()

        if self.num_qubits == 0:
            return

        circuit = QuantumCircuit(*self.qregs, name=self.name)

        circuit.compose(
            HartreeFock(
                num_spatial_orbitals=self.problem.num_spatial_orbitals,
                num_particles=self.problem.num_particles,
                qubit_mapper=self.mapper,
            ),
            inplace=True,
        )
        if self.insert_barriers:
            circuit.barrier()

        # Set prefactor so that the parameters are expected to be roughly in the range [0, 2pi]
        tau = 1 / (max(self.problem.orbital_energies) - min(self.problem.orbital_energies))  # type: ignore
        tau_s = tau / self.trotter_steps
        for s in range(self.trotter_steps):
            # Create circuit for interaction Hamiltonian non-Coulomb two-body terms
            # (i.e. non-coulomb terms) associated with parameter gamma.
            operator_gamma = self.mapper.map(tau_s * self._hamiltonian_gamma)
            operator_gamma_grouped = operator_gamma.group_commuting()
            operator_gamma_trotterized = PauliEvolutionGate(
                operator_gamma_grouped,
                time=self._gamma_parameter_vector[s],
                synthesis=(
                    LieTrotter(insert_barriers=self.insert_barriers)
                    if self.trotterization_order == 1
                    else SuzukiTrotter(
                        order=self.trotterization_order, insert_barriers=self.insert_barriers
                    )
                ),
                label=f"VHA interaction term H_γ (step {s + 1}/{self.trotter_steps})",  # noqa: RUF001
            )
            circuit.append(operator_gamma_trotterized, circuit.qubits)
            if self.insert_barriers:
                circuit.barrier()

            # Create circuit for interaction Hamiltonian one-body and coulomb two-body terms
            # associated with parameter beta.
            operator_beta = self.mapper.map(tau_s * self._hamiltonian_beta)
            operator_beta_grouped = operator_beta.group_commuting()
            operator_beta_trotterized = PauliEvolutionGate(
                operator_beta_grouped,
                time=self._beta_parameter_vector[s],
                synthesis=(
                    LieTrotter(insert_barriers=self.insert_barriers)
                    if self.trotterization_order == 1
                    else SuzukiTrotter(
                        order=self.trotterization_order, insert_barriers=self.insert_barriers
                    )
                ),
                label=f"VHA interaction term H_β (step {s + 1}/{self.trotter_steps})",
            )
            circuit.append(operator_beta_trotterized, circuit.qubits)
            if self.insert_barriers:
                circuit.barrier()

            # Create circuit for initial Hamiltonian (Fock operator)
            # associated with parameter alpha.
            operator_alpha = self.mapper.map(tau_s * self._hamiltonian_alpha)
            operator_alpha_grouped = operator_alpha.group_commuting()
            operator_alpha_trotterized = PauliEvolutionGate(
                operator_alpha_grouped,
                time=self._alpha_parameter_vector[s],
                synthesis=(
                    LieTrotter(insert_barriers=self.insert_barriers)
                    if self.trotterization_order == 1
                    else SuzukiTrotter(
                        order=self.trotterization_order, insert_barriers=self.insert_barriers
                    )
                ),
                label=f"VHA Fock term H_α (step {s + 1}/{self.trotter_steps})",  # noqa: RUF001
            )
            circuit.append(operator_alpha_trotterized, circuit.qubits)
            if self.insert_barriers:
                circuit.barrier()

        # cast global phase to float if it has no free parameters
        if isinstance(circuit.global_phase, ParameterExpression):
            try:
                circuit.global_phase = float(circuit.global_phase)
            except TypeError:
                # expression contains free parameters
                pass

        try:
            block = circuit.to_gate()
        except QiskitError:
            block = circuit.to_instruction()

        self.append(block, self.qubits)

    def get_recommended_parameters(
        self,
        two_qubit_gate_error: float,
        two_qubit_gate_length: float | None = None,
        decoherence_time: float | None = None,
        list_of_threshold_gamma: Iterable[float] | None = None,
        trotterization_orders: Iterable[int] = (1, 2),
    ) -> list[dict[str, int | float]]:
        """Gets recommended parameters based on 2-qubit gate error.

        The parameters are estimated using simple assumptions about success probability and
        duration/decoherence.
        The best choice of parameters is dependent on multiple additional factors that aren't
        considered here:
        - connectivity / layout of compiled circuit
        - propagation of errors through the circuit (some errors might even cancel out each other)
        - errors from 1-qubit gates
        - gate errors might vary for actually used gates (median is used for our estimate)
        - error mitigation techniques
        - errors on hardware are time dependent

        2-qubit gate length and decoherence time can be supplied for a better
        parameter estimate based additionally on decoherence.

        These are the inequality equations used for the estimation:
        estimated_duration = depth * two_qubit_gate_length * trotter_steps !< decoherence_time
        estimated_success_probability = (1 - two_qubit_gate_error)**(num_gates * trotter_steps)
            !>= 1/e

        The best parameter set can't be determined a priori.
        Only the Pareto front based on above simple estimation is available.
        As rule of thumb to avoid calculation of all parameter sets on the Pareto front, one should
        choose some parameter sets with a truncation threshold ('threshold_gamma') larger than 0.5.
        Dependent on usage of error models and the other above mentioned factors, the best
        parameter set might differ from the recommended parameters.

        Args:
            two_qubit_gate_error: 2-qubit gate error (layered).
            two_qubit_gate_length: 2-qubit gate duration in seconds (median).
            decoherence_time: Device T2 in seconds (median).
            list_of_threshold_gamma: The truncation thresholds to use for this plot.
                If 'None', all possible truncation thresholds are used.
            trotterization_orders: The trotterization orders (1: Lie-Trotter, >=2: Suzuki-Trotter)

        Returns:
            list of dict of parameter names and their values.
        """
        list_of_threshold_gamma = self.get_thresholds_gamma(list_of_threshold_gamma)

        best_parameters_within_error_bounds = []
        for trotterization_order in trotterization_orders:
            operator_alpha_beta = self.mapper.map(
                self._get_hamiltonian_alpha() + self._get_hamiltonian_beta()
            )
            operator_alpha_beta_grouped = operator_alpha_beta.group_commuting()
            operator_alpha_beta_trotterized = PauliEvolutionGate(
                operator_alpha_beta_grouped,
                synthesis=(
                    LieTrotter()
                    if trotterization_order == 1
                    else SuzukiTrotter(order=trotterization_order)
                ),
            )
            circuit_alpha_beta = QuantumCircuit(operator_alpha_beta_trotterized.num_qubits)
            circuit_alpha_beta.append(
                operator_alpha_beta_trotterized, range(operator_alpha_beta_trotterized.num_qubits)
            )
            num_gates_alpha_beta = circuit_alpha_beta.decompose().num_nonlocal_gates()
            depth_alpha_beta = circuit_alpha_beta.decompose().depth(
                lambda instr: len(instr.qubits) > 1
            )

            best_params = {}
            for threshold_gamma in self.possible_thresholds_gamma:
                if np.isclose(threshold_gamma, 0.0):
                    num_gates = num_gates_alpha_beta
                    depth = depth_alpha_beta
                else:
                    operator = self.mapper.map(
                        self._get_hamiltonian_gamma(threshold_gamma=threshold_gamma)
                    )
                    operator_grouped = operator.group_commuting()
                    operator_trotterized = PauliEvolutionGate(
                        operator_grouped,
                        synthesis=(
                            LieTrotter()
                            if trotterization_order == 1
                            else SuzukiTrotter(order=trotterization_order)
                        ),
                    )
                    circuit = QuantumCircuit(operator_trotterized.num_qubits)
                    circuit.append(operator_trotterized, range(operator_trotterized.num_qubits))
                    num_gates = circuit.decompose().num_nonlocal_gates() + num_gates_alpha_beta
                    depth = (
                        circuit.decompose().depth(lambda instr: len(instr.qubits) > 1)
                        + depth_alpha_beta
                    )

                if decoherence_time and two_qubit_gate_length:
                    trotter_steps_max = math.floor(
                        min(
                            [
                                decoherence_time / (depth * two_qubit_gate_length),
                                1 / (num_gates * two_qubit_gate_error),
                            ]
                        )
                    )
                else:
                    trotter_steps_max = math.floor(1 / (num_gates * two_qubit_gate_error))

                if trotter_steps_max < 1:
                    break
                if best_params.get(trotter_steps_max) is None:
                    best_params[trotter_steps_max] = []
                params = {
                    "trotterization_order": trotterization_order,
                    "threshold_gamma": threshold_gamma,
                    "trotter_steps": trotter_steps_max,
                    "estimated_success_probability": (1 - two_qubit_gate_error)
                    ** (num_gates * trotter_steps_max),
                }
                if decoherence_time and two_qubit_gate_length:
                    params["estimated_duration"] = depth * two_qubit_gate_length * trotter_steps_max
                best_params[trotter_steps_max].append(params)

            for entries in best_params.values():
                best_entry = max(entries, key=lambda e: e["threshold_gamma"])
                best_parameters_within_error_bounds.append(best_entry)

        if not best_parameters_within_error_bounds:
            logger.warning(
                "Error rates are too high (2-qubit gate error = %s). "
                "No suitable parameters are found.",
                two_qubit_gate_error,
            )
        return sorted(best_parameters_within_error_bounds, key=lambda p: p["threshold_gamma"])


def print_hamiltonian(
    hamiltonian: ElectronicEnergy | FermionicOp, number_of_shown_items: int | None = 16
) -> None:
    """Pretty print hamiltonian.

    In interactive mode it relies on Latex for nicely displaying the Hamiltonian.
    """
    if isinstance(hamiltonian, ElectronicEnergy):
        fermionic_operator = hamiltonian.second_q_op()
    elif isinstance(hamiltonian, FermionicOp):
        fermionic_operator = hamiltonian
    else:
        raise TypeError(
            "Unsupported type for 'hamiltonian'. "
            "Please specify an object of type 'ElectronicEnergy' or 'FermionicOp'."
        )
    ops_sorted = sorted(fermionic_operator.items(), key=lambda x: len(x[0]))
    if number_of_shown_items is None:
        number_of_shown_items = int(1e3)  # savety guard for not too many shown items
    for i, op in enumerate(ops_sorted):
        if i >= number_of_shown_items:
            print(f"... (skipped {len(ops_sorted) - number_of_shown_items} operators)")
            break
        operator, coeff = op
        if isinstance(coeff, complex) and np.isclose(coeff.imag, 0.0):
            coeff = coeff.real
        if hasattr(sys, "ps1"):
            try:
                from IPython.display import Latex, display

                operator = (
                    operator.replace("+", "a^†")
                    .replace("-", "a")
                    .replace("_", "_{")
                    .replace(" ", "}")
                    + "}"
                )
                display(Latex(f"{coeff:.3f} ⋅ ${operator}$"))
            except ModuleNotFoundError:
                print(f"{coeff:>10.3f} ⋅ {operator}")
        else:
            print(f"{coeff:>10.3f} ⋅ {operator}")
