from ._deformbase import Deformation
from ._deformgrid import DeformationGrid
from ._deformfield import DeformationField


class DeformationIdentity(Deformation):
    """ Abstract identity deformation. It is not a grid nor a field, nor
    is it forward or backward mapping.
    
    It is nothing more than a simple tool to initialize a deformation with.
    """
    pass


class DeformationGridForward(DeformationGrid):
    """ A deformation grid representing a forward mapping; to create the 
    deformed image, the pixels are mapped to their new locations. 
    """
    _forward_mapping = True


class DeformationGridBackward(DeformationGrid):
    """ A deformation grid representing a backward mapping; the field 
    represents where the pixels in the deformed image should be sampled to
    in the original image.
    """
    _forward_mapping = False


class DeformationFieldForward(DeformationField):
    """ A deformation field representing a forward mapping; to create the 
    deformed image, the pixels are mapped to their new locations. 
    """
    _forward_mapping = True


class DeformationFieldBackward(DeformationField):
    """ A deformation field representing a backward mapping; the field 
    represents where the pixels in the deformed image should be sampled to
    in the original image.
    """
    _forward_mapping = False
