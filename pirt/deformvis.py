""" deformVis module

Module for making textures and meshes move using a deformation field.
Visualization based on visvis.

"""

import visvis as vv
import numpy as np

## The GLSL code

SH_MV_DEFORM = vv.shaders.ShaderCodePart('deform', 'mesh', """
    >>--uniforms--
    uniform vec2 scaleBiasDeform;
    uniform sampler3D deformation;
    uniform vec3 deformOrigin;
    uniform vec3 deformSampling;
    uniform vec3 deformShape;
    //--uniforms--
    
    >>vec4 vertex = vec4(gl_Vertex);
    vec4 vertex = vec4(gl_Vertex);
    //vec3 loc = vertex.xyz / deformSampling - deformOrigin;
    vec3 loc = (vertex.xyz - deformOrigin) / (deformSampling * deformShape);
    vec3 dloc = texture3D( deformation, loc ).xyz;
    dloc = ( dloc - scaleBiasDeform[1] ) / scaleBiasDeform[0];  
    vertex.xyz += dloc;
    
""")

SH_3F_DEFORM = vv.shaders.ShaderCodePart('deform', '3D', """
    >>--uniforms--
    uniform vec2 scaleBiasDeform;
    uniform sampler3D deformation;
    //--uniforms--
    
    >>--in-loop--
    // loc should be scale-biassed too :/
    vec3 dloc = texture3D( deformation, loc ).xyz;
    dloc = ( dloc - scaleBiasDeform[1] ) / scaleBiasDeform[0]; 
    dloc /= extent;   
    loc = loc + dloc;
    //--in-loop--
    
    >>--post-loop--
    //--post-loop--
    // Color based on deformation
    //vec3 loc2 = edgeLoc + float(iter_depth) * ray;
    //vec3 dloc2 = texture3D( deformation, loc2 ).xyz;
    //dloc2 = ( dloc2 - scaleBiasDeform[1] );
    //gl_FragColor.rg *= (1.0-length(dloc2));
    
""")


## The classes


class DeformTexture(vv.textures.TextureObjectToVisualize):
    """ DeformTexture(deforms)
    
    Texture to manage a deformation that can be applied to a 
    texture or mesh.
    
    """
    def __init__(self, deform):
        ndim = deform.shape[-1]
        vv.textures.TextureObjectToVisualize.__init__(self, ndim, deform, True)
        
        # Interpolate deformation ofcourse!
        self._interpolate = True
        
        # Store
        self.SetData(deform)
    
    
    def SetData(self, data):
        # Overload to reset climRef
        
        # Set self._climRef
        minmax = vv.textures.minmax
        self._climRef.Set(*minmax(data))
        
        # Store shape, origin, sampling of the deform field
        self._deform_shape = data.shape
        if hasattr(data, 'origin'):
            self._deform_origin = data.origin
        else:
            self._deform_origin = [0.0 for s in data.shape]
        if hasattr(data, 'sampling'):
            self._deform_sampling = data.sampling
        else:
            self._deform_sampling = [1.0 for s in data.shape]
        
        # Update data 
        vv.textures.TextureObjectToVisualize.SetData(self, data)
    
    
    def _ScaleBias_get(self):
        """ Given clim, get scale and bias to apply in shader.
        In this case, we want to map [0 1] to the full range,
        expressed in world coordinates. In the shader, we use
        the extent (in world units) to convert to texture coords.
        
        Data to OpenGL: texData = data*scale + bias
        In shader: data_val = (texData-bias) / scale
        """
        
        # Code as in _ScaleBias_init
        ran = self._climRef.range
        if ran==0:
            ran = 1.0
        scale = 1.0 / ran # no need for data-type-correction
        bias = -self._climRef.min / ran
        
        # Done
        return scale, bias


class DeformableMixin(vv.MotionMixin):
    """ DeformableMixin
    
    Base class to mix with a Wobject class to make it deformable.
    
    """
    
    def __init__(self):
        vv.MotionMixin.__init__(self)
        
        # Init deform stuff
        self._deforms = []
        self._deformTexture = None
        self._motionAmplitude = 1.0
    
    
    def _GetMotionCount(self):
        return len(self._deforms)
    
    
    def _SetMotionIndex(self, index, ii, ww):
        
        # Prepare
        N = self.motionCount
        amp = self._motionAmplitude
        
        #print(index, ii, ww)
        # Interpolate multiple deforms
        if N == 0:
            deform = None
        elif N==1:
            deform = amp * self._deforms[0] 
        else:
            # t0 = time.time()
            deform = None
            for i, w in zip(ii, ww):
                if w != 0.0:
                    if deform is None:
                        deform = (w*amp) * self._deforms[i]
                    else:
                        deform += (w*amp) * self._deforms[i]
            # print 'deform composition took', time.time()-t0, 's'
        
        # Update deform texture  (fast update because shape is the same)
        if deform is not None:
            self._deformTexture.SetData(deform)
    
    
    def SetDeforms(self, *deforms):
        """ SetDeforms(*deforms)
        
        Set deformation arrays for this wobject. Each given argument 
        represents one deformation. Each deformation consists either
        of a tuple of arrays (one array for each dimension) or a 
        single array where the last dimension is ndim elements.
        
        Call without arguments to remove all deformations.
        
        If this wobject is a texture, it is assumed that the deformations
        exactly match with it, in a way that the outer pixels of the 
        texture match with the outer pixels of the deformatio. The 
        resolution does not have to be the same; it can often be lower
        because deformations are in general quite smooth. Smaller 
        deformation arrows result in higher FPS.
        
        If this wobject is not a texture, the given deformations represent
        a deformation somewhere in 3D space. One should use the vv.Aarray
        or pirt.Aarray class to store the arrays. The exact location
        is then specified by the origin and sampling properties.
        
        """
        
        # Get ndim
        ndim = self._ndim
        
        # Quick check
        if not deforms or (len(deforms)==1 and deforms[0] is None):
            self._deforms = []
        
        # Init deform props
        origin, sampling = None, None
        
        # Check deforms and make each deform a single array 
        deforms2 = []
        for deform in deforms:
            if isinstance(deform, np.ndarray):
                if deform.ndim == ndim+1 and deform.shape[-1] == ndim:
                    new_deform = deform
                    shape = deform.shape[:-1]
                    if hasattr(deform, 'sampling'):
                        sampling = deform.sampling[:-1]
                    if hasattr(deform, 'origin'):
                        origin = deform.origin[:-1]
                else:
                    raise ValueError('Number of dimensions for deform is incorrect.')
            
            #elif isinstance(deform, (tuple, list)):
            elif hasattr(deform, '__len__'):
                if len(deform) != ndim:
                    raise ValueError('Number of arrays for deform is incorrect.')
                # Build single texture
                new_shape = deform[0].shape+(ndim,)
                new_deform = np.zeros(new_shape, 'float32')
                for i in range(ndim):
                    if ndim==1: new_deform[:,i] = deform[i]
                    elif ndim==2: new_deform[:,:,i] = deform[i]
                    elif ndim==3: new_deform[:,:,:,i] = deform[i]
                    else: raise ValueError('DeformTexture only supports 1D, 2D and 3D.')
                #
                shape = deform[0].shape
                if hasattr(deform[0], 'sampling'):
                    sampling = deform[0].sampling
                if hasattr(deform[0], 'origin'):
                    origin = deform[0].origin
            else:
                raise ValueError('Deforms must be tuple or array.')
            deforms2.append(new_deform)
        
        # Store deformations
        self._deforms = deforms2
        
        # Pick first and create deform texture
        if self._deforms:
            self._deformTexture = DeformTexture(self._deforms[0])
        else:
            self._deformTexture = None
        
        # To update shader
        self._UpdateDeformShaderAfterSetDeforms(origin, sampling, shape)
    
    
    def _UpdateDeformShaderAfterSetDeforms(self, origin, sampling, shape):
        pass
    
    
    @vv.misc.Property
    def motionAmplitude():
        """ Get/set the relative amplitude of the deformation (default 1).
        Note that values higher than 1 can cause foldings in the deformation.
        Also note that because the deformation is backward mapping, changing
        the amplitude introduces errors in the deformation.
        """
        def fget(self):
            return self._motionAmplitude
        def fset(self, value):
            self._motionAmplitude = float(value)
            self.motionIndex = self.motionIndex
        return locals()


class DeformableTexture3D(vv.Texture3D, DeformableMixin):
    """ DeformableTexture3D(parent, data)
    
    This class represents a 3D texture that can be deformed using a 
    3D deformation field, i.e. a vector field that specifies the 
    displacement of the texture. This deformation field can (and probably 
    should) be of lower resolution than the texture.
    
    By supplying multiple deformation fields (via SetDeforms()),
    the texture becomes a moving texture subject to the given deformations.
    Note that the motion is interpolated.
    
    The deformations should be backward mapping. Note that this means
    that interpolation between the deformations and increasing the amplitude
    will yield not the exact expected deformations.
    
    """
    
    def __init__(self, *args, **kwargs):
        vv.Texture3D.__init__(self, *args, **kwargs)
        DeformableMixin.__init__(self)
        self._ndim = 3
    
    def _UpdateDeformShaderAfterSetDeforms(self, origin, sampling, shape):
        
        if self._deformTexture is None:
            self.shader.vertex.Remove(SH_3F_DEFORM)
            self.Draw()
            return
        
        # Ignore origin and sampling; it is assumed that the deformation
        # exactly matches with the texture
        
        self.shader.fragment.AddOrReplace(SH_3F_DEFORM, before='renderstyle')
        self.shader.SetStaticUniform('deformation', self._deformTexture)
        self.shader.SetStaticUniform('scaleBiasDeform', 
                                        self._deformTexture._ScaleBias_get)
        
        self.Draw()


class DeformableMesh(vv.Mesh, DeformableMixin):
    """ DeformableMesh(parent, vertices, faces=None, normals=None, values=None, verticesPerFace=3)
    
    This class represents a mesh that can be deformed using a 
    3D deformation field, i.e. a vector field that specifies the 
    displacement of a region of the 3D space.
    
    By supplying multiple deformation fields (via SetDeforms()),
    the mesh becomes a moving mesh subject to the given deformations.
    Note that the motion is interpolated.
    
    The deformations should be forward mapping; interpolation and changing
    the amplitude can be done safely. However, the normals are always the 
    same, so for extreme deformations the lighting might become incorrect.
    
    """
    
    def __init__(self, *args, **kwargs):
        vv.Mesh.__init__(self, *args, **kwargs)
        DeformableMixin.__init__(self)
        self._ndim = 3
    
    def _UpdateDeformShaderAfterSetDeforms(self, origin, sampling, shape):
        
        if self._deformTexture is None:
            for shader in [self.faceShader, self.shapeShader]:
                shader.vertex.Remove(SH_MV_DEFORM)
            self.Draw()
            return
        
        # Reverse three props (i.e. make x-y-z) and make floats
        if None in [origin, sampling, shape]:
            raise ValueError('Need origin and sampling to determine location of the deformation.')
        origin = [float(i) for i in reversed(origin)]
        sampling = [float(i) for i in reversed(sampling)]
        shape = [float(i) for i in reversed(shape)]
        
        for shader in [self.faceShader, self.shapeShader]:
            shader.vertex.AddOrReplace(SH_MV_DEFORM)
            shader.SetStaticUniform('deformation', self._deformTexture)
            shader.SetStaticUniform('scaleBiasDeform',
                                        self._deformTexture._ScaleBias_get)
            shader.SetStaticUniform('deformOrigin', origin)
            shader.SetStaticUniform('deformSampling', sampling)
            shader.SetStaticUniform('deformShape', shape)
        
        self.Draw()
