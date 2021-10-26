    
def rotate_coefficients(cos,sin,rotangle=0.):
    """
    helper definition to rotate coefficients (or really anything)
    
    inputs
    -----------
    cos : input cosine coefficients
    sin : input sine coefficients
    rotangle : float value for uniform rotation, or array of length cos.size
    
    returns
    -----------
    cos_rot : rotated cosine coefficients
    sin_rot : rotated sine coefficients
    
    todo
    -----------
    add some sort of clockwise/counterclockwise check?
    
    """
    cosT = np.cos(rotangle)
    sinT = np.sin(rotangle)
    
    cos_rot =  cosT*cos + sinT*sin
    sin_rot = -sinT*cos + cosT*sin
    
    return cos_rot,sin_rot


