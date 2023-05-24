import ipywidgets
import vtk


def vtk_show(renderer, width=400, height=300, filePath=None):
    """
    Takes vtkRenderer instance and returns an IPython Image with the rendering.

    Parameters
    ----------
    renderer : vtk.vtkRenderer
        The vtk renderer containing the scene to render.
    width, height : int
        The width and height of the image to render onto.
    filePath : str|None
        If a string is present, it is used as a file path without the extension to save a PS file.

    Returns
    -------
    ipywidgets.Image
        The image of the rendered scene.
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetOffScreenRendering(1)
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(width, height)
    renderWindow.Render()
    
    if filePath:
        exporter = vtk.vtkGL2PSExporter()
        exporter.SetRenderWindow(renderWindow)
        exporter.SetFileFormatToEPS()
        exporter.CompressOff()
        exporter.SetSortToSimple()
        exporter.DrawBackgroundOff()
        #exporter.TextAsPathOn()
        exporter.TextOn()
        exporter.SetFilePrefix(filePath)
        exporter.Write3DPropsAsRasterImageOn()
        exporter.Write()
     
    windowToImageFilter = vtk.vtkWindowToImageFilter()
    windowToImageFilter.SetInput(renderWindow)
    windowToImageFilter.Update()
     
    writer = vtk.vtkPNGWriter()
    writer.SetWriteToMemory(1)
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()
    data = data = memoryview(writer.GetResult()).tobytes()
    
    return ipywidgets.Image(value=data)

def rendering(data, arrayName='U_res', width=900, height=600, pos=None, foc=None, zoom=1.0, scalarRange=[-1,1], showColorBar=True, filePath=None, colorBarCoordinate=None):
    '''
    Renders a VTK data set to an image.

    Parameters
    ----------
    data : vtk.DataSet
        The data set to render
    arrayName : str
        The name of the point data to render.
    width, height : int
        The width and height of the resulting image.
    pos : array_like|None
        If a 3D array is given, the camera position is set to this value.
    foc : array_like|None
        If a 3D array is given, the focal point of the camera is set to this point.
    zoom : float default=1.0
        The zoom of the camera.
    scalarRange: array_like default=[-1,1]
        The range of the colorbar.
    showColorBar : bool default=True
        If true, the colorbar will be rendered, otherwise no colorbar is rendered.
    filePath : str|None
        If a string is present, it is used as a file path without the extension to save a PS file.
    colorBarCoordinate : array_like|None
        If a 2D array is given, the relative position of the colorbar is set. E.g. [0.7, 0.1] sets the colorbar at the right side.

    Returns
    -------
    ipywidgets.Image
        The image of the rendered scene.
    '''
    
    rainbowColors = [
        0.0, 0.0, 0.0, 0.5625,
        0.24444420000000006, 0.0, 0.0, 1.0,
        0.8031749, 0.0, 1.0, 1.0, 
        1.0825397, 0.5, 1.0, 0.5, 
        1.3619045, 1.0, 1.0, 0.0, 
        1.9206352000000002, 1.0, 0.0, 0.0, 
        2.2, 0.5, 0.0, 0.0
    ]
    redBlueColors = [
        0, 0.231373, 0.298039, 0.752941,
        0.5, 0.865003, 0.865003, 0.865003,
        1, 0.705882, 0.0156863, 0.14902
    ]
    
    # set white to be zero
    zeroPosition = (0 - scalarRange[0])/(scalarRange[1] - scalarRange[0])
    redBlueColors[4] = zeroPosition
    
    colorValues = redBlueColors
    
    ctf = vtk.vtkColorTransferFunction()
    ctf.SetColorSpaceToDiverging()
    for i in range(0,len(colorValues),4):
        ctf.AddRGBPoint(colorValues[i], colorValues[i+1], colorValues[i+2], colorValues[i+3])
    
    lut = vtk.vtkLookupTable()
    #lut.SetNumberOfTableValues(len(colorValues)//4)
    #for i in range(0, len(colorValues)//4, 4):
    #    lut.SetTableValue(i//4, colorValues[i+1], colorValues[i+2], colorValues[i+3])
    lut.SetNumberOfTableValues(1024)
    lut.Build()
    for i in range(1024):
        color = ctf.GetColor(i/1024.0)
        lut.SetTableValue(i, *color)
    
    scalarBar = vtk.vtkScalarBarActor()
    if colorBarCoordinate:
        scalarBar.SetPosition(colorBarCoordinate)
    scalarBar.UnconstrainedFontSizeOn()
    scalarBar.SetHeight(0.8)
    scalarBar.SetLookupTable(lut)
    scalarBar.SetTitle('Modulation')
    scalarBar.SetNumberOfLabels(4)
    scalarBar.SetTextPad(5)
    #scalarBar.SetLabelFormat('%-#6.3g mm')
    titleTextProperty = scalarBar.GetTitleTextProperty()
    titleTextProperty.SetFontSize(40)
    titleTextProperty.SetColor([0,0,0])
    labelTextProperty = scalarBar.GetLabelTextProperty()
    labelTextProperty.SetColor([0,0,0])
    labelTextProperty.SetFontSize(30)
    labelTextProperty.BoldOff()

    datasetMapper = vtk.vtkDataSetMapper()
    datasetMapper.SetInputData(data)
    datasetMapper.SetScalarModeToUsePointFieldData()
    datasetMapper.SelectColorArray(arrayName)
    datasetMapper.SetLookupTable(lut)
    datasetMapper.SetScalarRange(scalarRange)

    actor = vtk.vtkActor()
    actor.SetMapper(datasetMapper)
    actor.GetProperty().EdgeVisibilityOff()
    
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.AddActor(actor)
    if showColorBar:
        renderer.AddActor2D(scalarBar)
    renderer.ResetCameraClippingRange(-10000,100000,-100000,100000,100000,0.0001)
    #renderer.SetActiveCamera(camera)
    renderer.ResetCamera()
    c = renderer.GetActiveCamera()
    c.ParallelProjectionOn()
    if pos:
        # -51.75100000000003, 0.0, 4500.0
        c.SetPosition(*pos)
    if foc:
        # -51.75100000000003, 0.0, 400.0
        c.SetFocalPoint(*foc)
    c.Zoom(zoom)
    
    return vtk_show(renderer, width, height, filePath)
