import numpy as np
from foamToPython.readOFField import OFField
from foamToPython import readList
from PODopenFOAM.ofmodes import PODmodes
from typing import List, Dict, Any, Optional, Tuple, Union


class faceMode:
    """
    Class to handle face modes for PODI ROM.
    """

    def __init__(
        self,
        dataType: str,
        fileName: list[str],
        cellCoffsFile: str,
        POD_algo: str = "eigen",
    ):
        """
        Initialize the faceMode with given modes, fields, and coefficients.

        Parameters
        ----------
        dataType : str
            The type of data (e.g., "scalar", "vector").
        fileName : list[str]
            The name of the file to read.
        cellCoffsFile : str
            The name of the cell coefficients file.
        POD_algo : str, optional
            The POD algorithm to use ('eigen' or 'svd'), by default "eigen".
        """
        self.dataType = dataType
        self.POD_algo = POD_algo
        self.fileName = fileName
        self.cellCoffsFile = cellCoffsFile

        if self.dataType not in ["scalar", "vector"]:
            raise ValueError("dataType must be 'scalar' or 'vector'.")

        if self.POD_algo not in ["eigen", "svd"]:
            raise ValueError("POD_algo must be 'eigen' or 'svd'.")

        self.fields = self.readFields(self.fileName, self.dataType)
        self.boundaryModes, self.sv, self.boundaryCoeffs = self.boundaryModes(
            self.fields, self.POD_algo
        )
        self.cellCoeffs = self.readCellCoeffs(self.cellCoffsFile)
        self.cellBasedModes = self.cellBasedModes(self.fields, self.cellCoeffs)

    @staticmethod
    def readFields(fileName: list[str], dataType: str) -> np.ndarray:
        """
        Read the fields using the modes and coefficients.

        Parameters
        ----------
        fileName : list[str]
            The name of the file to read.
        dataType : str
            The type of data (e.g., "scalar", "vector").

        Returns
        -------
        np.ndarray
            The read fields. Each row corresponds to one field.
        """

        fields = []

        for fname in fileName:
            field = readList(fname, dataType)
            if dataType == "vector":
                field = field.T.flatten()
            fields.append(field)

        return np.array(fields)

    @staticmethod
    def boundaryModes(field: np.ndarray, POD_algo: str) -> np.ndarray:
        """
        Compute the boundary modes.

        Parameters
        ----------
        field : np.ndarray
            The input field.
        POD_algo : str
            The POD algorithm to use ('eigen' or 'svd').

        Returns
        -------
        modes : np.ndarray
            The computed boundary modes.
        sv : np.ndarray
            The singular values.
        coeffs : np.ndarray
            The POD coefficients.
        """

        return PODmodes.reduction(field, POD_algo)

    @staticmethod
    def readCellCoeffs(fileName: str) -> np.ndarray:
        """
        Read the cell coefficients from a file.

        Parameters
        ----------
        fileName : str
            The name of the file to read.

        Returns
        -------
        np.ndarray
            The read cell coefficients. Each row corresponds to a set of coefficients for one field.
        """

        return np.loadtxt(fileName).T

    @staticmethod
    def cellBasedModes(fields: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """
        Compute the cell modes.

        Parameters
        ----------
        fields : np.ndarray
            The input fields.
        coeffs : np.ndarray
            The coefficients.

        Returns
        -------
        np.ndarray
            The computed cell modes.
        """

        if coeffs.ndim == 1:
            raise ValueError("Coefficients must be a 2D array.")
        if coeffs.shape[0] == coeffs.shape[1]:
            return np.linalg.inv(coeffs) @ fields
        else:
            return np.linalg.pinv(coeffs) @ fields

    @staticmethod
    def projection(field: np.ndarray, boundaryModes: np.ndarray) -> np.ndarray:
        """
        Project the field onto the modes.

        Parameters
        ----------
        field : np.ndarray
            The input field.

        Returns
        -------
        np.ndarray
            The projected coefficients.
        """
        if field.ndim == 1:
            return field.reshape(1, -1) @ boundaryModes.T
        elif field.ndim == 2:
            return field @ boundaryModes.T
        else:
            raise ValueError("Field must be either 1D or 2D array.")
        
    def cellToBoundary(self, cellCoeffs: np.ndarray) -> np.ndarray:
        """
        Convert cell coefficients to boundary coefficients.

        Parameters
        ----------
        cellCoeffs : np.ndarray
            The cell coefficients.

        Returns
        -------
        np.ndarray
            The boundary coefficients.
        """

        if cellCoeffs.ndim == 1:
            rank = cellCoeffs.shape[0]
            cellCoeffs = cellCoeffs.reshape(1, -1)
            field = cellCoeffs @ self.cellBasedModes[:rank, :]
            boundaryCoeffs = self.projection(field, self.boundaryModes)
            return boundaryCoeffs
        elif cellCoeffs.ndim == 2:
            rank = cellCoeffs.shape[1]
            field = cellCoeffs @ self.cellBasedModes[:rank, :]
            boundaryCoeffs = self.projection(field, self.boundaryModes)
            return boundaryCoeffs
        else:
            raise ValueError("cellCoeffs must be either 1D or 2D array.")
