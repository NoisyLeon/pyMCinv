import spline

spl = spline.isospl()
spl.numbp = 10
spl.thickness = 200.
spl.bspline()

spl2 = spline.isospl()
spl2.numbp = 10
spl2.thickness = 200.
spl2.bspline2()

# bspl = spl.bspline2()