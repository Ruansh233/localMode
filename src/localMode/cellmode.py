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
        caseName: str,
        fieldName: str,
        NUM_MODES: int,
        dataType: str,
        parallel: bool = False,
    ):
        """
        Initialize the cellMode with given modes, fields, and coefficients.

        Parameters
        ----------
        caseName : str
            The name of the case directory containing modes and coefficients.
        fieldName : str
            The name of the field to be processed.
        NUM_MODES : int
            The number of modes to consider.
        dataType : str
            The type of data (e.g., "scalar", "vector").
        parallel : bool, optional
            Whether the openfoam data are saved in parallel, by default False.
        """
        self.caseName = caseName
        self.fieldName = fieldName
        self.dataType = dataType
        self.parallel = parallel
        self.NUM_MODES = NUM_MODES

        self.modes = self.readModes(
            self.caseName, self.fieldName, self.dataType, self.NUM_MODES, self.parallel
        )

    @staticmethod
    def readModes(
        caseName: str,
        modeName: str,
        dataType: str,
        NUM_MODES: int,
        parallel: bool,
    ) -> List[OFField]:
        """
        Read the modes from the given case directory.

        Parameters
        ----------
        caseName : str
            The name of the case directory to read.
        modeName : str
            The name of the mode to read.
        dataType : str
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
        modes = []
        for i in range(1, NUM_MODES + 1):
            modes.append(
                OFField(f"{caseName}/{i}/{modeName}", dataType, True, parallel=parallel)
            )
        return modes

    @staticmethod
    def reconstruct(
        modes: List[OFField], coeffs: np.ndarray, rank: int
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
            recOFFields = []
            for i in range(coeffs.shape[0]):
                recOFFields.append(
                    cellMode.reconstructField_parallel(modes, coeffs[i, :])
                )
        else:
            recOFFields = []
            for i in range(coeffs.shape[0]):
                recOFFields.append(
                    PODmodes._reconstructField_serial(modes, coeffs[i, :])
                )

        return recOFFields

    def writeReconstructed(
        self,
        coeffs: np.ndarray,
        outputDir: str,
        startTimeDir: int,
        fieldName: str = "recField",
    ) -> None:
        """
        Write the reconstructed fields to the specified output directory.

        Parameters
        ----------
        coeffs : np.ndarray
            The coefficients for reconstruction.
        outputDir : str
            The directory to write the reconstructed fields to.
        startTimeDir : int
            The starting time directory for writing fields.
        fieldName : str, optional
            The name of the field to write, by default "recField".
        """

        recOFFields = self.reconstruct(self.modes, coeffs, coeffs.shape[1])

        if self.parallel:
            for i, recOFField in enumerate(recOFFields):
                recOFField.writeField(
                    outputDir,
                    startTimeDir + i,
                    fieldName,
                )
        else:
            for i, recOFField in enumerate(recOFFields):
                recOFField.writeField(
                    outputDir,
                    startTimeDir + i,
                    fieldName,
                )

    @staticmethod
    def reconstructField_parallel(_modes: List[OFField], coeffs: np.ndarray):
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
        rank = coeffs.shape[0]
        if rank > len(_modes):
            raise ValueError("Rank cannot be greater than the number of modes.")
        if coeffs.ndim != 1:
            raise ValueError("Coefficients should be a 1D array.")

        recOFField = OFField.from_OFField(_modes[0])
        _num_processors = len(_modes[0].internalField)
        internalFieldList = []
        boundaryFieldList = []
        for procN in range(_num_processors):
            # Reconstruct internal field
            internalField = np.zeros(_modes[0].internalField[procN].shape)
            for i in range(rank):
                internalField += coeffs[i] * _modes[i].internalField[procN]
            internalFieldList.append(internalField)

            # Reconstruct boundary field
            boundaryField = cellMode._copy_boundary(_modes[0].boundaryField[procN])
            for patch in boundaryField.keys():
                patch_type = boundaryField[patch]["type"]
                if (
                    patch_type == "fixedValue"
                    or patch_type == "fixedGradient"
                    or patch_type == "processor"
                    or patch_type == "calculated"
                ):
                    value_type = list(_modes[0].boundaryField[procN][patch].keys())[-1]
                    if isinstance(
                        _modes[0].boundaryField[procN][patch][value_type],
                        str,
                    ):
                        continue
                    elif isinstance(
                        _modes[0].boundaryField[procN][patch][value_type],
                        np.ndarray,
                    ):
                        boundaryField[patch][value_type] = np.zeros(
                            _modes[0].boundaryField[procN][patch][value_type].shape
                        )
                        for i in range(rank):
                            boundaryField[patch][value_type] += (
                                coeffs[i]
                                * _modes[i].boundaryField[procN][patch][value_type]
                            )
                    else:
                        raise ValueError(
                            "Unknown boundary field value type for fixedValue, fixedGradient, or processor."
                        )

            boundaryFieldList.append(boundaryField)

        recOFField.internalField = internalFieldList
        recOFField.boundaryField = boundaryFieldList

        return recOFField

    @staticmethod
    def _copy_boundary(_boundaryField):
        # BoundaryField: handle Dict[str, Dict[str, Any]] for serial, List[Dict[str, Dict[str, Any]]] for parallel
        if isinstance(_boundaryField, dict):
            return {
                patch: {
                    key: (
                        value.copy()
                        if isinstance(value, np.ndarray)
                        else copy.deepcopy(value)
                    )
                    for key, value in info.items()
                }
                for patch, info in _boundaryField.items()
            }
        else:
            raise ValueError(
                "Unsupported type for boundaryField. It should be dict."
            )
