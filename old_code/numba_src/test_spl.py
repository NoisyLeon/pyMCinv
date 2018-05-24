import spline

spl = spline.isospl()
# spl.init_arr(5)
# spl.numbp[0] = 10
# spl.thickness[0] = 200.
# spl.bspline(0)

spline.readspltxt('old_code/TEST/Q22A.mod1', inspl=spl)

# 
# spl2 = spline.isospl()
# spl2.numbp = 10
# spl2.thickness = 200.
# spl2.bspline2()

# bspl = spl.bspline2()

# import spline_bk
# 
# spl2 = spline_bk.isospl()
# spl2.numbp = 10
# spl2.thickness = 200.
# spl2.bspline()