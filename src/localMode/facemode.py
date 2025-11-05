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
        data_type: str,
        file_name: list[str],
        cell_coeffs: np.ndarray,
        POD_algo: str = "eigen",
    ):
        """
        Initialize the faceMode with given modes, fields, and coefficients.

        Parameters
        ----------
        data_type : str
            The type of data (e.g., "scalar", "vector").
        file_name : list[str]
            The name of the file to read.
        cell_coeffs : np.ndarray
            The cell coefficients.
        POD_algo : str, optional
            The POD algorithm to use ('eigen' or 'svd'), by default "eigen".
        """
        self.data_type: str = data_type
        self.POD_algo: str = POD_algo
        self.file_name: list[str] = file_name
        self.cell_coeffs: np.ndarray = cell_coeffs

        if self.data_type not in ["scalar", "vector"]:
            raise ValueError("dataType must be 'scalar' or 'vector'.")

        if self.POD_algo not in ["eigen", "svd"]:
            raise ValueError("POD_algo must be 'eigen' or 'svd'.")

        self.fields: np.ndarray = self.read_fields(self.file_name, self.data_type)
        self.boundary_modes: np.ndarray
        self.sv: np.ndarray
        self.boundary_coeffs: np.ndarray
        self.boundary_modes, self.sv, self.boundary_coeffs = self.boundary_reduction(
            self.fields, self.POD_algo
        )
        self.cell_based_modes: np.ndarray = self.cell_based_projection(self.fields, self.cell_coeffs)

    @staticmethod
    def read_fields(file_name: list[str], data_type: str) -> np.ndarray:
        """
        Read the fields using the modes and coefficients.

        Parameters
        ----------
        file_name : list[str]
            The name of the file to read.
        data_type : str
            The type of data (e.g., "scalar", "vector").

        Returns
        -------
        np.ndarray
            The read fields. Each row corresponds to one field.
        """

        fields: List[np.ndarray] = []

        for fname in file_name:
            field = readList(fname, data_type)
            if data_type == "vector":
                field = field.T.flatten()
            fields.append(field)

        return np.array(fields)

    @staticmethod
    def boundary_reduction(field: np.ndarray, POD_algo: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    def cell_based_projection(fields: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
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
    def projection(field: np.ndarray, boundary_modes: np.ndarray) -> np.ndarray:
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
            return field.reshape(1, -1) @ boundary_modes.T
        elif field.ndim == 2:
            return field @ boundary_modes.T
        else:
            raise ValueError("Field must be either 1D or 2D array.")
        
    def cell_to_boundary(self, cell_coeffs: np.ndarray) -> np.ndarray:
        """
        Convert cell coefficients to boundary coefficients.

        Parameters
        ----------
        cell_coeffs : np.ndarray
            The cell coefficients.

        Returns
        -------
        np.ndarray
            The boundary coefficients.
        """

        if cell_coeffs.ndim == 1:
            rank: int = cell_coeffs.shape[0]
            cell_coeffs = cell_coeffs.reshape(1, -1)
            field: np.ndarray = cell_coeffs @ self.cell_based_modes[:rank, :]
            boundary_coeffs: np.ndarray = self.projection(field, self.boundary_modes)
            return boundary_coeffs
        elif cell_coeffs.ndim == 2:
            rank: int = cell_coeffs.shape[1]
            field: np.ndarray = cell_coeffs @ self.cell_based_modes[:rank, :]
            boundary_coeffs: np.ndarray = self.projection(field, self.boundary_modes)
            return boundary_coeffs
        else:
            raise ValueError("cell_coeffs must be either 1D or 2D array.")
