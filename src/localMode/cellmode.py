import numpy as np
from foamToPython.readOFField import OFField
from PODopenFOAM.ofmodes import PODmodes, write_mode_worker
from typing import List, Dict, Any, Optional, Tuple, Union
import multiprocessing
import copy


class cellMode:
    """
    Class to handle cell modes for PODI ROM.
    """

    def __init__(
        self,
        case_name: str,
        field_name: str,
        num_modes: int,
        data_type: str,
        coeffs: np.ndarray,
        parallel: bool = False,
    ):
        """
        Initialize the cellMode with given modes, fields, and coefficients.

        Parameters
        ----------
        case_name : str
            The name of the case directory containing modes and coefficients.
        field_name : str
            The name of the field to be processed.
        num_modes : int
            The number of modes to consider.
        data_type : str
            The type of data (e.g., "scalar", "vector").
        coeffs : np.ndarray
            The coefficients for the modes.
        parallel : bool, optional
            Whether the openfoam data are saved in parallel, by default False.
        """
        self.case_name: str = case_name
        self.field_name: str = field_name
        self.data_type: str = data_type
        self.parallel: bool = parallel
        self.num_modes: int = num_modes

        self.modes: List[OFField] = self.read_modes(
            self.case_name,
            self.field_name,
            self.data_type,
            self.num_modes,
            self.parallel,
        )

        if coeffs.shape[1] > self.num_modes:
            Warning(
                "Number of coefficient columns exceeds number of modes. Truncating coefficients."
            )
            coeffs = coeffs[:, : self.num_modes]
        self.coeffs: np.ndarray = coeffs

        @property
        def data_matrix(self) -> np.ndarray:
            """
            The data matrix where each column corresponds to a flattened field.

            Returns
            -------
            np.ndarray
                The data matrix.
            """
            return self._cal_data_matrix(self.modes)

    @staticmethod
    def read_modes(
        case_name: str,
        mode_name: str,
        data_type: str,
        num_modes: int,
        parallel: bool,
    ) -> List[OFField]:
        """
        Read the modes from the given case directory.

        Parameters
        ----------
        case_name : str
            The name of the case directory to read.
        mode_name : str
            The name of the mode to read.
        data_type : str
            The type of data (e.g., "scalar", "vector").
        rankRange : Tuple[int, int]
            The range of ranks to read modes from (e.g., (1, 10)).
        parallel : bool
            Whether the openfoam data are saved in parallel.

        Returns
        -------
        List[OFField]
            The read modes.
        """
        modes: List[OFField] = []
        for i in range(1, num_modes + 1):
            modes.append(
                OFField(
                    f"{case_name}/{i}/{mode_name}", data_type, True, parallel=parallel
                )
            )
        return modes

    @staticmethod
    def reconstruct(
        modes: List[OFField],
        coeffs: np.ndarray,
        rank: int,
    ) -> List[OFField] | OFField:
        """
        Reconstruct the fields using the modes and coefficients.

        Parameters
        ----------
        modes : List[OFField]
            The POD modes.
        coeffs : np.ndarray
            The coefficients for reconstruction.
            Each row corresponds to a set of coefficients for one field.
        rank : int
            The rank of the modes to use for reconstruction.
        parallel : bool
            Whether to use parallel processing.

        Returns
        -------
        List[OFField]: The reconstructed fields if parallel is True.
        OFField: The reconstructed field if parallel is False.

        """
        if coeffs.ndim == 1:
            Warning("Coefficients should be a 2D array. Reshaping to 1 x N.")
            coeffs = coeffs.reshape(1, -1)

        modes = modes[:rank]
        coeffs = coeffs[:, :rank]

        parallel = modes[0].parallel

        if parallel:
            rec_fields: List[OFField] = []
            for i in range(coeffs.shape[0]):
                rec_fields.append(
                    cellMode.reconstruct_field_parallel(modes, coeffs[i, :])
                )
        else:
            rec_fields: List[OFField] = []
            for i in range(coeffs.shape[0]):
                rec_fields.append(
                    PODmodes._reconstructField_serial(modes, coeffs[i, :])
                )

        return rec_fields

    def write_reconstructed(
        self,
        coeffs: np.ndarray,
        output_dir: str,
        start_time_dir: int,
        field_name: str = "recField",
    ) -> None:
        """
        Write the reconstructed fields to the specified output directory.

        Parameters
        ----------
        coeffs : np.ndarray
            The coefficients for reconstruction.
        output_dir : str
            The directory to write the reconstructed fields to.
        start_time_dir : int
            The starting time directory for writing fields.
        field_name : str, optional
            The name of the field to write, by default "recField".
        """

        rec_fields = self.reconstruct(self.modes, coeffs, coeffs.shape[1])

        if self.parallel:
            for i, rec_field in enumerate(rec_fields):
                rec_field.writeField(
                    output_dir,
                    start_time_dir + i,
                    field_name,
                )
        else:
            for i, rec_field in enumerate(rec_fields):
                rec_field.writeField(
                    output_dir,
                    start_time_dir + i,
                    field_name,
                )

    @staticmethod
    def reconstruct_field_parallel(_modes: List[OFField], coeffs: np.ndarray) -> OFField:
        """
        Reconstruct the original field from the POD modes and coefficients (parallel version).

        Parameters
        ----------
        _modes : List[OFField]
            The list of POD mode OpenFOAM field objects.
        coeffs : np.ndarray
            The coefficients for reconstructing the field. Shape should be (rank,).

        Returns
        -------
        list
            List of OpenFOAM field objects representing the reconstructed field.
        Raises
        ------
        ValueError
            If rank is greater than the number of modes.
        """
        rank: int = coeffs.shape[0]
        if rank > len(_modes):
            raise ValueError("Rank cannot be greater than the number of modes.")
        if coeffs.ndim != 1:
            raise ValueError("Coefficients should be a 1D array.")

        rec_field: OFField = OFField.from_OFField(_modes[0])
        _num_processors: int = len(_modes[0].internalField)
        internal_field_list: List[np.ndarray] = []
        boundary_field_list: List[Dict[str, Dict[str, Any]]] = []
        for proc_n in range(_num_processors):
            # Reconstruct internal field
            internal_field: np.ndarray = np.zeros(_modes[0].internalField[proc_n].shape)
            for i in range(rank):
                internal_field += coeffs[i] * _modes[i].internalField[proc_n]
            internal_field_list.append(internal_field)

            # Reconstruct boundary field
            boundary_field: Dict[str, Dict[str, Any]] = cellMode._copy_boundary(_modes[0].boundaryField[proc_n])
            for patch in boundary_field.keys():
                patch_type = boundary_field[patch]["type"]
                if (
                    patch_type == "fixedValue"
                    or patch_type == "fixedGradient"
                    or patch_type == "processor"
                    or patch_type == "calculated"
                ):
                    value_type = list(_modes[0].boundaryField[proc_n][patch].keys())[-1]
                    if isinstance(
                        _modes[0].boundaryField[proc_n][patch][value_type],
                        str,
                    ):
                        continue
                    elif isinstance(
                        _modes[0].boundaryField[proc_n][patch][value_type],
                        np.ndarray,
                    ):
                        boundary_field[patch][value_type] = np.zeros(
                            _modes[0].boundaryField[proc_n][patch][value_type].shape
                        )
                        for i in range(rank):
                            boundary_field[patch][value_type] += (
                                coeffs[i]
                                * _modes[i].boundaryField[proc_n][patch][value_type]
                            )
                    else:
                        raise ValueError(
                            "Unknown boundary field value type for fixedValue, fixedGradient, or processor."
                        )

            boundary_field_list.append(boundary_field)

        rec_field.internalField = internal_field_list
        rec_field.boundaryField = boundary_field_list

        return rec_field

    @staticmethod
    def _copy_boundary(_boundary_field) -> Dict[str, Dict[str, Any]]:
        # BoundaryField: handle Dict[str, Dict[str, Any]] for serial, List[Dict[str, Dict[str, Any]]] for parallel
        if isinstance(_boundary_field, dict):
            return {
                patch: {
                    key: (
                        value.copy()
                        if isinstance(value, np.ndarray)
                        else copy.deepcopy(value)
                    )
                    for key, value in info.items()
                }
                for patch, info in _boundary_field.items()
            }
        else:
            raise ValueError("Unsupported type for boundaryField. It should be dict.")

    @staticmethod
    def _cal_data_matrix(field_list: List[OFField]) -> np.ndarray:
        """
        Calculate the data matrix from a list of OFField objects.

        Parameters
        ----------
        field_list : List[OFField]
            The list of OFField objects.

        Returns
        -------
        np.ndarray
            The data matrix where each column corresponds to a flattened field.
        """
        if field_list[0].parallel:
            data_matrix: np.ndarray = PODmodes._field2ndarray_parallel(field_list)
        else:
            data_matrix: np.ndarray = PODmodes._field2ndarray_serial(field_list)

        return data_matrix

    @staticmethod
    def l2_norm(
        field_a: List[OFField], field_b: List[OFField], relative: bool = True
    ) -> np.ndarray:
        """
        Calculate the L2 norm between two lists of OFField objects.

        Parameters
        ----------
        field_a : List[OFField]
            The first list of OFField objects.
        field_b : List[OFField]
            The second list of OFField objects.
        relative : bool, optional
            Whether to calculate the relative L2 norm, by default True.

        Returns
        -------
        np.ndarray
            The L2 norm between corresponding fields in field_a and field_b.
        """
        if field_a[0].parallel != field_b[0].parallel:
            raise ValueError("Both field lists must have the same parallel setting.")

        if field_a[0].parallel:
            data_matrix_a: np.ndarray = PODmodes._field2ndarray_parallel(field_a)
            data_matrix_b: np.ndarray = PODmodes._field2ndarray_parallel(field_b)
        else:
            data_matrix_a: np.ndarray = PODmodes._field2ndarray_serial(field_a)
            data_matrix_b: np.ndarray = PODmodes._field2ndarray_serial(field_b)

        l2_norms: np.ndarray = np.linalg.norm(data_matrix_a - data_matrix_b, axis=1)
        if relative:
            l2_norms /= np.linalg.norm(data_matrix_b, axis=1)

        return l2_norms
