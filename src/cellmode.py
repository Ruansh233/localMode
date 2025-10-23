import numpy as np
from foamToPython.readOFField import OFField
from PODopenFOAM.ofmodes import PODmodes, write_mode_worker
from typing import List, Dict, Any, Optional, Tuple, Union
import multiprocessing


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
        modes: List[OFField],
        coeffs: np.ndarray,
        rank: int,
        parallel: bool,
        num_processors: int = 8,
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
        num_processors : int, optional
            The number of processors to use if parallel is True, by default 8.

        Returns
        -------
        List[OFField]: The reconstructed fields if parallel is True.
        OFField: The reconstructed field if parallel is False.

        """
        modes = modes[:rank]
        coeffs = coeffs[:, :rank]

        if parallel:
            recOFFields = []
            for i in range(coeffs.shape[0]):
                recOFFields.append(
                    PODmodes._reconstructField_parallel(
                        modes, coeffs[i, :], num_processors
                    )
                )
        else:
            recOFFields = []
            for i in range(coeffs.shape[0]):
                recOFFields.append(
                    PODmodes._reconstructField_serial(modes, coeffs[i, :])
                )

        return np.array(recOFFields)

    def writeReconstructed(
        self,
        recOFFields: List[OFField],
        outputDir: str,
        timeDir: int,
        fieldName: str = "recField",
        num_processors: int = 8,
    ) -> None:
        """
        Write the reconstructed fields to the specified output directory.

        Parameters
        ----------
        coeffs : np.ndarray
            The coefficients for reconstruction.
        outputDir : str
            The directory to write the reconstructed fields to.
        num_processors : int, optional
            The number of processors to use if parallel is True, by default 8.
        """

        if self.parallel:
            for i, recOFField in enumerate(recOFFields):
                tasks = [
                    (procN, timeDir - 1 + i, recOFField, outputDir, fieldName)
                    for procN in range(num_processors)
                ]
                with multiprocessing.Pool() as pool:
                    pool.map(write_mode_worker, tasks)
        else:
            for i, recOFField in enumerate(recOFFields):
                recOFField.writeField(
                    outputDir,
                    timeDir - 1 + i,
                    fieldName,
                )
